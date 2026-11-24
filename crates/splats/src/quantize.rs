//! Scalar codebooks and vector clustering used by palette-based formats.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Builds an ascending codebook of exactly `size` finite values that
/// minimizes squared error over `values` (1-D k-means / Lloyd–Max).
/// Non-finite inputs are ignored; short codebooks are padded with their
/// last entry.
pub(crate) fn codebook_1d(values: &[f32], size: usize) -> Vec<f32> {
    let mut sorted: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return vec![0.0; size];
    }
    const MAX_SAMPLES: usize = 1 << 21;
    if sorted.len() > MAX_SAMPLES {
        let step = sorted.len() as f64 / MAX_SAMPLES as f64;
        sorted = (0..MAX_SAMPLES)
            .map(|i| sorted[(i as f64 * step) as usize])
            .collect();
    }
    sorted.sort_by(f32::total_cmp);
    let mut distinct = sorted.clone();
    distinct.dedup();
    let mut book = if distinct.len() <= size {
        distinct
    } else {
        lloyd_1d(&sorted, size)
    };
    let last = *book.last().expect("not empty");
    book.resize(size, last);
    book
}

fn lloyd_1d(sorted: &[f32], k: usize) -> Vec<f32> {
    let n = sorted.len();
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0f64);
    for &v in sorted {
        prefix.push(prefix.last().expect("seeded") + f64::from(v));
    }
    // Start from quantiles, which already concentrate entries where values are dense.
    let mut centers: Vec<f64> = (0..k)
        .map(|j| f64::from(sorted[((j as f64 + 0.5) / k as f64 * n as f64) as usize]))
        .collect();
    for _ in 0..25 {
        let mut changed = false;
        let mut start = 0;
        for j in 0..k {
            let end = if j + 1 == k {
                n
            } else {
                let mid = (centers[j] + centers[j + 1]) * 0.5;
                start + sorted[start..].partition_point(|&v| f64::from(v) <= mid)
            };
            if end > start {
                let mean = (prefix[end] - prefix[start]) / (end - start) as f64;
                if (mean - centers[j]).abs() > 1e-12 {
                    changed = true;
                }
                centers[j] = mean;
            }
            start = end;
        }
        centers.sort_by(f64::total_cmp);
        if !changed {
            break;
        }
    }
    centers.into_iter().map(|c| c as f32).collect()
}

/// Index of the entry of an ascending `codebook` nearest to `v`.
pub(crate) fn nearest(codebook: &[f32], v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let i = codebook.partition_point(|&c| c < v);
    let i = if i == 0 {
        0
    } else if i == codebook.len() {
        codebook.len() - 1
    } else if (v - codebook[i - 1]).abs() <= (codebook[i] - v).abs() {
        i - 1
    } else {
        i
    };
    i as u8
}

struct Cluster {
    members: Vec<u32>,
    centroid: Vec<f32>,
    sse: f64,
}

impl PartialEq for Cluster {
    fn eq(&self, other: &Self) -> bool {
        self.sse == other.sse
    }
}
impl Eq for Cluster {}
impl PartialOrd for Cluster {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cluster {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sse.total_cmp(&other.sse)
    }
}

fn distance2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(x - y);
            d * d
        })
        .sum()
}

fn make_cluster(data: &[f32], dims: usize, members: Vec<u32>) -> Cluster {
    let mut sum = vec![0.0f64; dims];
    for &m in &members {
        for (s, v) in sum
            .iter_mut()
            .zip(&data[m as usize * dims..(m as usize + 1) * dims])
        {
            *s += f64::from(*v);
        }
    }
    let inv = 1.0 / members.len().max(1) as f64;
    let centroid: Vec<f32> = sum.iter().map(|s| (s * inv) as f32).collect();
    let sse = members
        .iter()
        .map(|&m| distance2(&data[m as usize * dims..(m as usize + 1) * dims], &centroid))
        .sum();
    Cluster {
        members,
        centroid,
        sse,
    }
}

/// Splits a cluster in two with a few rounds of 2-means seeded from its
/// two most distant-looking members.
fn split(data: &[f32], dims: usize, cluster: Cluster, iterations: usize) -> (Cluster, Cluster) {
    let row = |m: u32| &data[m as usize * dims..(m as usize + 1) * dims];
    let farthest = |from: &[f32]| {
        *cluster
            .members
            .iter()
            .max_by(|&&a, &&b| distance2(row(a), from).total_cmp(&distance2(row(b), from)))
            .expect("cluster has members")
    };
    let a = farthest(&cluster.centroid);
    let b = farthest(row(a));
    let mut ca = row(a).to_vec();
    let mut cb = row(b).to_vec();
    let mut left = Vec::new();
    let mut right = Vec::new();
    for _ in 0..iterations.max(1) {
        left.clear();
        right.clear();
        for &m in &cluster.members {
            if distance2(row(m), &ca) <= distance2(row(m), &cb) {
                left.push(m);
            } else {
                right.push(m);
            }
        }
        if left.is_empty() || right.is_empty() {
            break;
        }
        ca = make_cluster(data, dims, left.clone()).centroid;
        cb = make_cluster(data, dims, right.clone()).centroid;
    }
    if left.is_empty() || right.is_empty() {
        let mut members = cluster.members;
        let half = members.len() / 2;
        right = members.split_off(half);
        left = members;
    }
    (
        make_cluster(data, dims, left),
        make_cluster(data, dims, right),
    )
}

/// Clusters `data` (rows of `dims` floats) into at most `k` groups by
/// repeatedly splitting the cluster with the largest squared error.
/// Returns the centroids (row-major) and each row's cluster index.
/// Splits every cluster in `batch`, spreading the work over `threads`
/// threads when there is enough of it.
fn split_all(
    data: &[f32],
    dims: usize,
    mut batch: Vec<Cluster>,
    iterations: usize,
    threads: usize,
) -> Vec<(Cluster, Cluster)> {
    let work: usize = batch.iter().map(|c| c.members.len()).sum();
    if threads < 2 || batch.len() < 2 || work * dims < 1 << 20 {
        return batch
            .into_iter()
            .map(|c| split(data, dims, c, iterations))
            .collect();
    }
    // Largest first, dealt to the least-loaded worker.
    batch.sort_by_key(|c| std::cmp::Reverse(c.members.len()));
    let workers = threads.min(batch.len());
    let mut groups: Vec<(usize, Vec<Cluster>)> = (0..workers).map(|_| (0, Vec::new())).collect();
    for cluster in batch {
        let group = groups
            .iter_mut()
            .min_by_key(|(load, _)| *load)
            .expect("at least one worker");
        group.0 += cluster.members.len();
        group.1.push(cluster);
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = groups
            .into_iter()
            .map(|(_, group)| {
                scope.spawn(move || {
                    group
                        .into_iter()
                        .map(|c| split(data, dims, c, iterations))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("clustering worker panicked"))
            .collect()
    })
}

pub(crate) fn kmeans(
    data: &[f32],
    dims: usize,
    k: usize,
    iterations: usize,
) -> (Vec<f32>, Vec<u32>) {
    let n = data.len().checked_div(dims).unwrap_or(0);
    if n == 0 || k == 0 {
        return (Vec::new(), vec![0; n]);
    }
    if k >= n {
        return (data[..n * dims].to_vec(), (0..n as u32).collect());
    }
    let threads = std::thread::available_parallelism()
        .map_or(1, |t| t.get())
        .min(32);
    let mut heap = BinaryHeap::new();
    let mut done = Vec::new();
    heap.push(make_cluster(data, dims, (0..n as u32).collect()));
    while heap.len() + done.len() < k {
        // Split the clusters with the largest error, as many per round as
        // are still needed; splits within a round are independent.
        let need = k - heap.len() - done.len();
        let mut batch = Vec::with_capacity(need.min(heap.len()));
        while batch.len() < need {
            let Some(cluster) = heap.pop() else { break };
            if cluster.members.len() < 2 || cluster.sse <= 0.0 {
                done.push(cluster);
            } else {
                batch.push(cluster);
            }
        }
        if batch.is_empty() {
            break;
        }
        for (a, b) in split_all(data, dims, batch, iterations, threads) {
            heap.push(a);
            heap.push(b);
        }
    }
    let clusters: Vec<Cluster> = heap.into_vec().into_iter().chain(done).collect();
    let mut centroids = Vec::with_capacity(clusters.len() * dims);
    let mut labels = vec![0u32; n];
    for (i, c) in clusters.iter().enumerate() {
        centroids.extend_from_slice(&c.centroid);
        for &m in &c.members {
            labels[m as usize] = i as u32;
        }
    }
    (centroids, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_inputs_are_exact() {
        let book = codebook_1d(&[3.0, 1.0, 2.0, 1.0, f32::NAN], 256);
        assert_eq!(book.len(), 256);
        assert_eq!(&book[..3], &[1.0, 2.0, 3.0]);
        assert!(book[3..].iter().all(|&v| v == 3.0));
        assert_eq!(nearest(&book, 2.1), 1);
        assert_eq!(nearest(&book, -9.0), 0);
        assert_eq!(nearest(&book, f32::INFINITY), 255);
    }

    #[test]
    fn codebook_error_is_small_for_uniform_data() {
        let values: Vec<f32> = (0..100_000)
            .map(|i| (i as f32 / 100_000.0) * 10.0)
            .collect();
        let book = codebook_1d(&values, 256);
        assert!(book.windows(2).all(|w| w[0] <= w[1]));
        let worst = values
            .iter()
            .map(|&v| (book[nearest(&book, v) as usize] - v).abs())
            .fold(0.0, f32::max);
        assert!(worst < 10.0 / 256.0, "worst error {worst}");
    }

    #[test]
    fn kmeans_separates_obvious_groups() {
        let mut data = Vec::new();
        for i in 0..300 {
            let base = [0.0, 10.0, -10.0][i % 3];
            data.extend_from_slice(&[base + (i as f32 * 0.001), base]);
        }
        let (centroids, labels) = kmeans(&data, 2, 3, 4);
        assert_eq!(centroids.len(), 6);
        for i in 0..300 {
            assert_eq!(labels[i], labels[i % 3], "row {i}");
        }
    }
}
