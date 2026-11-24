//! The in-memory splat container and the math helpers its fields rely on.

use crate::error::{Error, Result};

/// Zeroth-order spherical-harmonic basis constant, `1 / (2 * sqrt(pi))`.
pub const SH_C0: f32 = 0.282_094_8;

/// Highest spherical-harmonic degree supported.
pub const MAX_SH_DEGREE: u8 = 3;

/// Number of higher-order SH coefficients per color channel for `degree`:
/// 0, 3, 8 or 15 for degrees 0 to 3.
pub const fn sh_rest_coeffs(degree: u8) -> usize {
    let d = degree as usize;
    (d + 1) * (d + 1) - 1
}

/// Logistic function, mapping a stored opacity logit to alpha in `0..=1`.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse of [`sigmoid`]. The input is clamped away from 0 and 1 so the
/// result is always finite.
pub fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// Converts a DC spherical-harmonic coefficient to a `0..=1` color value
/// (not clamped).
pub fn sh_dc_to_color(dc: f32) -> f32 {
    0.5 + SH_C0 * dc
}

/// Inverse of [`sh_dc_to_color`].
pub fn color_to_sh_dc(color: f32) -> f32 {
    (color - 0.5) / SH_C0
}

/// Returns `q` scaled to unit length, or the identity rotation if `q` has
/// zero or non-finite length.
pub fn normalize_quaternion(q: [f32; 4]) -> [f32; 4] {
    let len = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    if len.is_finite() && len > 0.0 {
        q.map(|v| v / len)
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// A borrowed view of one splat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplatRef<'a> {
    /// Center position.
    pub position: [f32; 3],
    /// Natural log of the per-axis scale (standard deviation).
    pub log_scale: [f32; 3],
    /// Orientation quaternion `[w, x, y, z]`, not necessarily normalized.
    pub rotation: [f32; 4],
    /// Opacity before the logistic function; see [`sigmoid`].
    pub opacity_logit: f32,
    /// Degree-0 spherical-harmonic coefficients for red, green and blue.
    pub sh_dc: [f32; 3],
    /// Higher-order SH coefficients, coefficient-major with interleaved RGB:
    /// `[c1.r, c1.g, c1.b, c2.r, ...]`. Holds `sh_rest_coeffs(degree) * 3`
    /// values.
    pub sh_rest: &'a [f32],
}

impl SplatRef<'_> {
    /// Alpha in `0..=1`.
    pub fn opacity(&self) -> f32 {
        sigmoid(self.opacity_logit)
    }

    /// Linear per-axis scale.
    pub fn scale(&self) -> [f32; 3] {
        self.log_scale.map(f32::exp)
    }

    /// Base color in `0..=1` (not clamped) from the DC coefficients.
    pub fn color(&self) -> [f32; 3] {
        self.sh_dc.map(sh_dc_to_color)
    }
}

/// A set of Gaussian splats stored as parallel arrays.
///
/// Every per-splat vector has one entry per splat, and `sh_rest` holds
/// [`SplatCloud::sh_rest_stride`] floats per splat. The fields are public for
/// fast bulk access; [`SplatCloud::validate`] checks that they agree.
///
/// Values are stored exactly as optimized during training: log scales, opacity
/// logits and SH coefficients. Formats that store something else convert on
/// read and write.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SplatCloud {
    /// Center positions.
    pub positions: Vec<[f32; 3]>,
    /// Natural log of each splat's per-axis scale.
    pub log_scales: Vec<[f32; 3]>,
    /// Orientation quaternions `[w, x, y, z]`.
    pub rotations: Vec<[f32; 4]>,
    /// Opacities before the logistic function.
    pub opacity_logits: Vec<f32>,
    /// Degree-0 SH coefficients (RGB).
    pub sh_dc: Vec<[f32; 3]>,
    /// Higher-order SH coefficients for all splats, splat-major, then
    /// coefficient, then RGB.
    pub sh_rest: Vec<f32>,
    /// Spherical-harmonic degree, 0 to [`MAX_SH_DEGREE`].
    pub sh_degree: u8,
    /// Whether the scene was trained with antialiased (mip-filtered)
    /// splats, which renderers draw slightly differently. Carried by
    /// formats that record it.
    pub antialiased: bool,
}

impl SplatCloud {
    /// Creates an empty cloud.
    ///
    /// # Panics
    ///
    /// Panics if `sh_degree` exceeds [`MAX_SH_DEGREE`].
    pub fn new(sh_degree: u8) -> Self {
        Self::with_capacity(sh_degree, 0)
    }

    /// Creates an empty cloud with room for `capacity` splats.
    ///
    /// # Panics
    ///
    /// Panics if `sh_degree` exceeds [`MAX_SH_DEGREE`].
    pub fn with_capacity(sh_degree: u8, capacity: usize) -> Self {
        assert!(
            sh_degree <= MAX_SH_DEGREE,
            "SH degree {sh_degree} exceeds the maximum of {MAX_SH_DEGREE}"
        );
        Self {
            positions: Vec::with_capacity(capacity),
            log_scales: Vec::with_capacity(capacity),
            rotations: Vec::with_capacity(capacity),
            opacity_logits: Vec::with_capacity(capacity),
            sh_dc: Vec::with_capacity(capacity),
            sh_rest: Vec::with_capacity(capacity * sh_rest_coeffs(sh_degree) * 3),
            sh_degree,
            antialiased: false,
        }
    }

    /// Number of splats.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the cloud has no splats.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Number of `sh_rest` floats per splat.
    pub fn sh_rest_stride(&self) -> usize {
        sh_rest_coeffs(self.sh_degree) * 3
    }

    /// Appends one splat.
    ///
    /// # Panics
    ///
    /// Panics if `splat.sh_rest` does not hold [`SplatCloud::sh_rest_stride`]
    /// values.
    pub fn push(&mut self, splat: SplatRef<'_>) {
        assert_eq!(
            splat.sh_rest.len(),
            self.sh_rest_stride(),
            "sh_rest length does not match SH degree {}",
            self.sh_degree
        );
        self.positions.push(splat.position);
        self.log_scales.push(splat.log_scale);
        self.rotations.push(splat.rotation);
        self.opacity_logits.push(splat.opacity_logit);
        self.sh_dc.push(splat.sh_dc);
        self.sh_rest.extend_from_slice(splat.sh_rest);
    }

    /// Returns splat `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn get(&self, index: usize) -> SplatRef<'_> {
        let stride = self.sh_rest_stride();
        SplatRef {
            position: self.positions[index],
            log_scale: self.log_scales[index],
            rotation: self.rotations[index],
            opacity_logit: self.opacity_logits[index],
            sh_dc: self.sh_dc[index],
            sh_rest: &self.sh_rest[index * stride..(index + 1) * stride],
        }
    }

    /// Iterates over all splats in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = SplatRef<'_>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// Checks that every array has a consistent length.
    pub fn validate(&self) -> Result<()> {
        if self.sh_degree > MAX_SH_DEGREE {
            return Err(Error::Invalid(format!(
                "SH degree {} exceeds the maximum of {MAX_SH_DEGREE}",
                self.sh_degree
            )));
        }
        let n = self.len();
        let lengths = [
            ("log_scales", self.log_scales.len()),
            ("rotations", self.rotations.len()),
            ("opacity_logits", self.opacity_logits.len()),
            ("sh_dc", self.sh_dc.len()),
        ];
        for (name, len) in lengths {
            if len != n {
                return Err(Error::Invalid(format!(
                    "{name} has {len} entries but positions has {n}"
                )));
            }
        }
        let expected = n * self.sh_rest_stride();
        if self.sh_rest.len() != expected {
            return Err(Error::Invalid(format!(
                "sh_rest has {} values; {n} splats of SH degree {} need {expected}",
                self.sh_rest.len(),
                self.sh_degree
            )));
        }
        Ok(())
    }

    /// Returns a new cloud holding the splats at `indices`, in that order.
    /// Indices may repeat.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds.
    pub fn select(&self, indices: &[usize]) -> SplatCloud {
        let mut out = SplatCloud::with_capacity(self.sh_degree, indices.len());
        out.antialiased = self.antialiased;
        for &i in indices {
            out.push(self.get(i));
        }
        out
    }

    /// Keeps only the splats for which `keep` returns `true`, preserving
    /// order. Returns the number of splats removed.
    pub fn retain(&mut self, mut keep: impl FnMut(SplatRef<'_>) -> bool) -> usize {
        let n = self.len();
        let stride = self.sh_rest_stride();
        let mut write = 0;
        for read in 0..n {
            if !keep(self.get(read)) {
                continue;
            }
            if write != read {
                self.positions[write] = self.positions[read];
                self.log_scales[write] = self.log_scales[read];
                self.rotations[write] = self.rotations[read];
                self.opacity_logits[write] = self.opacity_logits[read];
                self.sh_dc[write] = self.sh_dc[read];
                self.sh_rest
                    .copy_within(read * stride..(read + 1) * stride, write * stride);
            }
            write += 1;
        }
        self.truncate(write);
        n - write
    }

    /// Shortens the cloud to `len` splats. Does nothing if it is already
    /// shorter.
    pub fn truncate(&mut self, len: usize) {
        let stride = self.sh_rest_stride();
        self.positions.truncate(len);
        self.log_scales.truncate(len);
        self.rotations.truncate(len);
        self.opacity_logits.truncate(len);
        self.sh_dc.truncate(len);
        self.sh_rest.truncate(len * stride);
    }

    /// Appends every splat of `other`, which must have the same SH degree.
    pub fn append(&mut self, other: &SplatCloud) -> Result<()> {
        if other.sh_degree != self.sh_degree {
            return Err(Error::Invalid(format!(
                "cannot append SH degree {} splats to an SH degree {} cloud",
                other.sh_degree, self.sh_degree
            )));
        }
        other.validate()?;
        self.positions.extend_from_slice(&other.positions);
        self.log_scales.extend_from_slice(&other.log_scales);
        self.rotations.extend_from_slice(&other.rotations);
        self.opacity_logits.extend_from_slice(&other.opacity_logits);
        self.sh_dc.extend_from_slice(&other.sh_dc);
        self.sh_rest.extend_from_slice(&other.sh_rest);
        Ok(())
    }

    /// Changes the SH degree. Lowering it drops the higher bands; raising it
    /// adds zero coefficients.
    ///
    /// # Panics
    ///
    /// Panics if `degree` exceeds [`MAX_SH_DEGREE`].
    pub fn set_sh_degree(&mut self, degree: u8) {
        assert!(
            degree <= MAX_SH_DEGREE,
            "SH degree {degree} exceeds the maximum of {MAX_SH_DEGREE}"
        );
        if degree == self.sh_degree {
            return;
        }
        let old = self.sh_rest_stride();
        let new = sh_rest_coeffs(degree) * 3;
        let keep = old.min(new);
        let mut rest = vec![0.0; self.len() * new];
        for i in 0..self.len() {
            rest[i * new..i * new + keep].copy_from_slice(&self.sh_rest[i * old..i * old + keep]);
        }
        self.sh_rest = rest;
        self.sh_degree = degree;
    }

    /// Axis-aligned bounds of the splat centers as `(min, max)`, or `None`
    /// for an empty cloud.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut iter = self.positions.iter();
        let first = *iter.next()?;
        let (mut lo, mut hi) = (first, first);
        for p in iter {
            lo = std::array::from_fn(|a| lo[a].min(p[a]));
            hi = std::array::from_fn(|a| hi[a].max(p[a]));
        }
        Some((lo, hi))
    }
}
