//! Real spherical-harmonic basis and rotation of SH coefficients.
//!
//! The basis is the one Gaussian splatting uses: real SH with the
//! Condon–Shortley phase, coefficients ordered by degree and then from the
//! lowest order `m` up.

use crate::cloud::{MAX_SH_DEGREE, SH_C0, SplatRef, sh_rest_coeffs};
use crate::math::{Mat3, mat_vec, solve, transpose};

const C1: f64 = 0.488_602_511_902_919_9;
const C2: [f64; 5] = [
    1.092_548_430_592_079_2,
    -1.092_548_430_592_079_2,
    0.315_391_565_252_520_05,
    -1.092_548_430_592_079_2,
    0.546_274_215_296_039_6,
];
const C3: [f64; 7] = [
    -0.590_043_589_926_643_5,
    2.890_611_442_640_554,
    -0.457_045_799_464_465_8,
    0.373_176_332_590_115_4,
    -0.457_045_799_464_465_8,
    1.445_305_721_320_277,
    -0.590_043_589_926_643_5,
];

/// Values of the 15 higher-order basis functions (degrees 1 to 3) at the
/// unit direction `d`.
fn basis_rest(d: [f64; 3]) -> [f64; 15] {
    let [x, y, z] = d;
    let (xx, yy, zz) = (x * x, y * y, z * z);
    [
        -C1 * y,
        C1 * z,
        -C1 * x,
        C2[0] * x * y,
        C2[1] * y * z,
        C2[2] * (2.0 * zz - xx - yy),
        C2[3] * x * z,
        C2[4] * (xx - yy),
        C3[0] * y * (3.0 * xx - yy),
        C3[1] * x * y * z,
        C3[2] * y * (4.0 * zz - xx - yy),
        C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        C3[4] * x * (4.0 * zz - xx - yy),
        C3[5] * z * (xx - yy),
        C3[6] * x * (xx - 3.0 * yy),
    ]
}

impl SplatRef<'_> {
    /// The splat's color seen along `direction` (from the camera toward the
    /// splat), including view-dependent SH terms. Not clamped.
    pub fn view_color(&self, direction: [f32; 3]) -> [f32; 3] {
        let d = direction.map(f64::from);
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if !(len > 0.0 && len.is_finite()) {
            return self.color();
        }
        let basis = basis_rest(d.map(|v| v / len));
        let k = self.sh_rest.len() / 3;
        std::array::from_fn(|c| {
            let mut v = f64::from(SH_C0) * f64::from(self.sh_dc[c]);
            for (i, b) in basis.iter().enumerate().take(k) {
                v += b * f64::from(self.sh_rest[i * 3 + c]);
            }
            (v + 0.5) as f32
        })
    }
}

/// Evenly spread unit directions (a Fibonacci sphere).
fn sample_directions(count: usize) -> Vec<[f64; 3]> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    (0..count)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
            let r = (1.0 - y * y).sqrt();
            let theta = golden * i as f64;
            [r * theta.cos(), y, r * theta.sin()]
        })
        .collect()
}

/// Matrices that rotate each SH band by a fixed rotation.
pub(crate) struct ShRotation {
    /// For band `l` (index `l - 1`): a `(2l+1) × (2l+1)` row-major matrix.
    bands: Vec<Vec<f64>>,
}

impl ShRotation {
    /// Builds band matrices for rotating a scene by `r` (new = r · old).
    ///
    /// The rotated coefficients `c'` must satisfy `f'(d) = f(rᵀ d)` for all
    /// directions. Each band is solved by least squares over many sample
    /// directions, which is exact because bands are closed under rotation.
    pub(crate) fn new(r: &Mat3, degree: u8) -> Self {
        let degree = degree.min(MAX_SH_DEGREE);
        let directions = sample_directions(64);
        let rt = transpose(r);
        let original: Vec<[f64; 15]> = directions.iter().map(|&d| basis_rest(d)).collect();
        let rotated: Vec<[f64; 15]> = directions
            .iter()
            .map(|&d| basis_rest(mat_vec(&rt, d)))
            .collect();
        let bands = (1..=degree as usize)
            .map(|l| {
                let n = 2 * l + 1;
                let offset = l * l - 1;
                // Normal equations: (AᵀA) M = AᵀB.
                let mut ata = vec![0.0; n * n];
                let mut atb = vec![0.0; n * n];
                for (a, b) in original.iter().zip(&rotated) {
                    for i in 0..n {
                        for j in 0..n {
                            ata[i * n + j] += a[offset + i] * a[offset + j];
                            atb[i * n + j] += a[offset + i] * b[offset + j];
                        }
                    }
                }
                solve(ata, atb, n, n).expect("SH basis samples are well conditioned")
            })
            .collect();
        Self { bands }
    }

    /// Rotates one splat's interleaved `sh_rest` block in place.
    pub(crate) fn apply(&self, rest: &mut [f32]) {
        let mut values = [0.0f64; 7];
        for (band, m) in self.bands.iter().enumerate() {
            let l = band + 1;
            let n = 2 * l + 1;
            let offset = l * l - 1;
            if rest.len() < (offset + n) * 3 {
                return;
            }
            for c in 0..3 {
                for (j, v) in values.iter_mut().enumerate().take(n) {
                    *v = f64::from(rest[(offset + j) * 3 + c]);
                }
                for i in 0..n {
                    let sum: f64 = (0..n).map(|j| m[i * n + j] * values[j]).sum();
                    rest[(offset + i) * 3 + c] = sum as f32;
                }
            }
        }
    }
}

/// Number of SH coefficients per channel in band `l`.
pub(crate) const fn band_size(l: usize) -> usize {
    2 * l + 1
}

/// Index of band `l`, order `n` (0-based from the lowest `m`) within the
/// higher-order coefficients.
pub(crate) const fn rest_index(l: usize, n: usize) -> usize {
    l * l - 1 + n
}

const _: () = assert!(rest_index(3, 6) + 1 == sh_rest_coeffs(3));
