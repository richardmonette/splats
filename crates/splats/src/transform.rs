//! Whole-cloud edits: cleanup, filtering, cropping, sorting and simple
//! spatial transforms.
//!
//! Functions that remove splats return how many they removed.

use crate::cloud::{SplatCloud, SplatRef, normalize_quaternion};
use crate::math::{mat_vec, quat_mul, quat_to_mat};
use crate::sh::ShRotation;

/// Which axes have odd powers in each higher-order SH basis function
/// (x, y, z), in coefficient order. Mirroring an axis negates exactly the
/// coefficients with an odd power of it.
const SH_ODD_AXES: [[bool; 3]; 15] = [
    [false, true, false],
    [false, false, true],
    [true, false, false],
    [true, true, false],
    [false, true, true],
    [false, false, false],
    [true, false, true],
    [false, false, false],
    [false, true, false],
    [true, true, true],
    [false, true, false],
    [false, false, true],
    [true, false, false],
    [false, false, true],
    [true, false, false],
];

/// Negates the chosen axes (`[x, y, z]`). Flipping two axes is a 180°
/// rotation about the third; flipping one or three mirrors the scene.
/// Exact: positions, rotations and SH coefficients only change sign.
///
/// This converts between axis conventions, e.g. flipping y and z turns
/// right-down-forward data into right-up-back.
pub fn flip_axes(cloud: &mut SplatCloud, flip: [bool; 3]) {
    if flip == [false; 3] {
        return;
    }
    let f = flip.map(|b| if b { -1.0f32 } else { 1.0 });
    for p in &mut cloud.positions {
        for (v, s) in p.iter_mut().zip(f) {
            *v *= s;
        }
    }
    let [fx, fy, fz] = f;
    for q in &mut cloud.rotations {
        q[1] *= fy * fz;
        q[2] *= fx * fz;
        q[3] *= fx * fy;
    }
    let stride = cloud.sh_rest_stride();
    if stride == 0 {
        return;
    }
    let signs: Vec<f32> = SH_ODD_AXES[..stride / 3]
        .iter()
        .map(|odd| (0..3).filter(|&a| odd[a]).map(|a| f[a]).product::<f32>())
        .collect();
    for block in cloud.sh_rest.chunks_exact_mut(stride) {
        for (k, s) in signs.iter().enumerate() {
            if *s < 0.0 {
                for c in 0..3 {
                    block[k * 3 + c] = -block[k * 3 + c];
                }
            }
        }
    }
}

/// A unit quaternion `[w, x, y, z]` for a rotation of `degrees` about
/// `axis`. A zero axis gives the identity.
pub fn axis_angle(axis: [f32; 3], degrees: f32) -> [f32; 4] {
    let len = axis.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !(len > 0.0 && len.is_finite()) {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let half = degrees.to_radians() * 0.5;
    let s = half.sin() / len;
    [half.cos(), axis[0] * s, axis[1] * s, axis[2] * s]
}

/// Rotates the whole scene about the origin by the quaternion `rotation`
/// (`[w, x, y, z]`, normalized internally). Positions, splat orientations
/// and view-dependent SH color are all rotated.
pub fn rotate(cloud: &mut SplatCloud, rotation: [f32; 4]) {
    let q = normalize_quaternion(rotation);
    let r = quat_to_mat(q);
    for p in &mut cloud.positions {
        *p = mat_vec(&r, p.map(f64::from)).map(|v| v as f32);
    }
    for rot in &mut cloud.rotations {
        *rot = quat_mul(q, *rot);
    }
    if cloud.sh_degree > 0 {
        let sh = ShRotation::new(&r, cloud.sh_degree);
        let stride = cloud.sh_rest_stride();
        for block in cloud.sh_rest.chunks_exact_mut(stride) {
            sh.apply(block);
        }
    }
}

/// Reorders splats along a Morton (Z-order) curve so that nearby splats are
/// stored together. Runs of more than 256 splats sharing a cell are sorted
/// again within their own bounds. Non-finite positions leave the order
/// unchanged.
pub fn sort_morton(cloud: &mut SplatCloud) {
    let mut order: Vec<usize> = (0..cloud.len()).collect();
    morton_sort_indices(&cloud.positions, &mut order);
    *cloud = cloud.select(&order);
}

fn spread_bits(mut v: u32) -> u32 {
    v &= 0x0000_03ff;
    v = (v ^ (v << 16)) & 0xff00_00ff;
    v = (v ^ (v << 8)) & 0x0300_f00f;
    v = (v ^ (v << 4)) & 0x030c_30c3;
    v = (v ^ (v << 2)) & 0x0924_9249;
    v
}

fn morton_sort_indices(positions: &[[f32; 3]], indices: &mut [usize]) {
    if indices.len() < 2 {
        return;
    }
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for &i in indices.iter() {
        for a in 0..3 {
            lo[a] = lo[a].min(positions[i][a]);
            hi[a] = hi[a].max(positions[i][a]);
        }
    }
    let extent: [f32; 3] = std::array::from_fn(|a| hi[a] - lo[a]);
    if extent.iter().any(|e| !e.is_finite()) || extent.iter().all(|&e| e == 0.0) {
        return;
    }
    let mul = extent.map(|e| if e == 0.0 { 0.0 } else { 1024.0 / e });
    let code = |i: usize| -> u32 {
        let cell = |a: usize| (((positions[i][a] - lo[a]) * mul[a]).min(1023.0)) as u32;
        (spread_bits(cell(2)) << 2) | (spread_bits(cell(1)) << 1) | spread_bits(cell(0))
    };
    let mut keyed: Vec<(u32, usize)> = indices.iter().map(|&i| (code(i), i)).collect();
    keyed.sort_by_key(|&(k, _)| k);
    for (slot, (_, i)) in indices.iter_mut().zip(&keyed) {
        *slot = *i;
    }
    let mut start = 0;
    while start < keyed.len() {
        let mut end = start + 1;
        while end < keyed.len() && keyed[end].0 == keyed[start].0 {
            end += 1;
        }
        if end - start > 256 {
            morton_sort_indices(positions, &mut indices[start..end]);
        }
        start = end;
    }
}

/// Whether every value of `splat` is finite.
pub fn is_finite(splat: &SplatRef<'_>) -> bool {
    splat.opacity_logit.is_finite()
        && splat
            .position
            .iter()
            .chain(&splat.log_scale)
            .chain(&splat.rotation)
            .chain(&splat.sh_dc)
            .chain(splat.sh_rest)
            .all(|v| v.is_finite())
}

/// Removes splats containing NaN or infinite values.
pub fn remove_non_finite(cloud: &mut SplatCloud) -> usize {
    cloud.retain(|s| is_finite(&s))
}

/// Removes splats whose opacity (`0..=1`) is below `min_opacity`.
pub fn filter_min_opacity(cloud: &mut SplatCloud, min_opacity: f32) -> usize {
    cloud.retain(|s| s.opacity() >= min_opacity)
}

/// Keeps only splats whose centers lie inside the box `min..=max`.
pub fn crop_box(cloud: &mut SplatCloud, min: [f32; 3], max: [f32; 3]) -> usize {
    cloud.retain(|s| (0..3).all(|a| s.position[a] >= min[a] && s.position[a] <= max[a]))
}

/// A splat's visual weight: opacity times ellipsoid volume (up to a constant).
pub fn importance(splat: &SplatRef<'_>) -> f32 {
    splat.opacity() * splat.log_scale.iter().sum::<f32>().exp()
}

/// Reorders splats from most to least [`importance`]. Equal splats keep
/// their relative order.
pub fn sort_by_importance(cloud: &mut SplatCloud) {
    let keys: Vec<f32> = cloud.iter().map(|s| importance(&s)).collect();
    let mut order: Vec<usize> = (0..cloud.len()).collect();
    order.sort_by(|&a, &b| keys[b].total_cmp(&keys[a]));
    *cloud = cloud.select(&order);
}

/// Moves every splat by `offset`.
pub fn translate(cloud: &mut SplatCloud, offset: [f32; 3]) {
    for p in &mut cloud.positions {
        for (v, o) in p.iter_mut().zip(offset) {
            *v += o;
        }
    }
}

/// Scales the scene uniformly about the origin, including splat sizes.
///
/// # Panics
///
/// Panics if `factor` is not positive and finite.
pub fn scale_uniform(cloud: &mut SplatCloud, factor: f32) {
    assert!(
        factor > 0.0 && factor.is_finite(),
        "scale factor must be positive and finite, got {factor}"
    );
    let log = factor.ln();
    for p in &mut cloud.positions {
        for v in p {
            *v *= factor;
        }
    }
    for s in &mut cloud.log_scales {
        for v in s {
            *v += log;
        }
    }
}

/// Normalizes every rotation to a unit quaternion.
pub fn normalize_rotations(cloud: &mut SplatCloud) {
    for q in &mut cloud.rotations {
        *q = normalize_quaternion(*q);
    }
}
