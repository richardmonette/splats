//! Small rotation helpers shared by transforms and format code.
//! Quaternions are `[w, x, y, z]`; matrices are row-major.

use crate::cloud::normalize_quaternion;

pub(crate) type Mat3 = [[f64; 3]; 3];

/// Hamilton product `a ⊗ b`: rotating by `b`, then by `a`.
pub(crate) fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Rotation matrix of a (normalized) quaternion.
pub(crate) fn quat_to_mat(q: [f32; 4]) -> Mat3 {
    let [w, x, y, z] = normalize_quaternion(q).map(f64::from);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Unit quaternion of a proper rotation matrix.
pub(crate) fn mat_to_quat(m: &Mat3) -> [f32; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        ]
    };
    normalize_quaternion(q.map(|v| v as f32))
}

pub(crate) fn mat_vec(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

pub(crate) fn transpose(m: &Mat3) -> Mat3 {
    std::array::from_fn(|r| std::array::from_fn(|c| m[c][r]))
}

/// Solves `a · x = b` for `x` by Gaussian elimination with partial
/// pivoting. `a` is `n × n` and `b` is `n × m`, both row-major. Returns
/// `None` if `a` is singular.
pub(crate) fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize, m: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        if a[pivot * n + col].abs() < 1e-12 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(pivot * n + k, col * n + k);
            }
            for k in 0..m {
                b.swap(pivot * m + k, col * m + k);
            }
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col] / a[col * n + col];
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            for k in 0..m {
                b[row * m + k] -= factor * b[col * m + k];
            }
        }
    }
    for row in 0..n {
        let d = a[row * n + row];
        for k in 0..m {
            b[row * m + k] /= d;
        }
    }
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quaternion_matrix_roundtrip() {
        let q = normalize_quaternion([0.3, -0.5, 0.7, 0.2]);
        let back = mat_to_quat(&quat_to_mat(q));
        let dot: f32 = q.iter().zip(&back).map(|(a, b)| a * b).sum();
        assert!(dot.abs() > 0.99999, "{q:?} vs {back:?}");
    }

    #[test]
    fn solve_small_system() {
        // [2 1; 1 3] x = [3; 5] -> x = [0.8; 1.4]
        let x = solve(vec![2.0, 1.0, 1.0, 3.0], vec![3.0, 5.0], 2, 1).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12 && (x[1] - 1.4).abs() < 1e-12);
    }
}
