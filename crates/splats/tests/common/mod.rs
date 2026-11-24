#![allow(dead_code)]

use splats::{SplatCloud, SplatRef, sh_rest_coeffs};

/// Small deterministic generator so tests need no dependencies.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// Uniform in `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// A random cloud with plausible value ranges.
pub fn sample_cloud(n: usize, sh_degree: u8, seed: u64) -> SplatCloud {
    let mut rng = Rng::new(seed);
    let mut cloud = SplatCloud::new(sh_degree);
    let mut rest = vec![0.0; sh_rest_coeffs(sh_degree) * 3];
    for _ in 0..n {
        for v in &mut rest {
            *v = rng.next_f32() * 0.5;
        }
        let position = [0; 3].map(|_| rng.next_f32() * 10.0);
        let log_scale = [0; 3].map(|_| rng.next_f32() * 2.0 - 4.0);
        let rotation = [0; 4].map(|_| rng.next_f32());
        let opacity_logit = rng.next_f32() * 4.0;
        let sh_dc = [0; 3].map(|_| rng.next_f32() * 1.5);
        cloud.push(SplatRef {
            position,
            log_scale,
            rotation,
            opacity_logit,
            sh_dc,
            sh_rest: &rest,
        });
    }
    cloud
}

/// Absolute difference check with a readable failure message.
#[track_caller]
pub fn assert_close(a: f32, b: f32, tolerance: f32, what: &str) {
    assert!(
        (a - b).abs() <= tolerance,
        "{what}: {a} vs {b} differs by more than {tolerance}"
    );
}

/// Asserts two quaternions describe the same rotation (q and -q are equal).
#[track_caller]
pub fn assert_same_rotation(a: [f32; 4], b: [f32; 4], tolerance: f32) {
    let a = splats::normalize_quaternion(a);
    let b = splats::normalize_quaternion(b);
    let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert!(
        dot.abs() >= 1.0 - tolerance,
        "rotations differ: {a:?} vs {b:?} (|dot| = {})",
        dot.abs()
    );
}
