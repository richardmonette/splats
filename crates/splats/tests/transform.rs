mod common;

use splats::{SplatCloud, SplatRef, sh_rest_coeffs, transform};

/// Splat `i` sits at x = i, and its SH values encode `i` so misalignment
/// between arrays is detectable.
fn indexed_cloud(n: usize, degree: u8) -> SplatCloud {
    let stride = sh_rest_coeffs(degree) * 3;
    let mut cloud = SplatCloud::new(degree);
    for i in 0..n {
        let rest: Vec<f32> = (0..stride).map(|j| (i * 1000 + j) as f32).collect();
        cloud.push(SplatRef {
            position: [i as f32, 0.0, 0.0],
            log_scale: [0.0; 3],
            rotation: [1.0, 0.0, 0.0, 0.0],
            opacity_logit: i as f32 - (n / 2) as f32,
            sh_dc: [i as f32; 3],
            sh_rest: &rest,
        });
    }
    cloud
}

fn assert_aligned(cloud: &SplatCloud) {
    cloud.validate().unwrap();
    for s in cloud.iter() {
        let i = s.position[0] as usize;
        assert_eq!(s.sh_dc, [i as f32; 3]);
        for (j, v) in s.sh_rest.iter().enumerate() {
            assert_eq!(*v, (i * 1000 + j) as f32);
        }
    }
}

#[test]
fn retain_keeps_arrays_aligned() {
    let mut cloud = indexed_cloud(20, 2);
    let removed = cloud.retain(|s| (s.position[0] as usize).is_multiple_of(3));
    assert_eq!(removed, 13);
    assert_eq!(cloud.len(), 7);
    assert_aligned(&cloud);
}

#[test]
fn set_sh_degree_truncates_and_pads() {
    let mut cloud = indexed_cloud(4, 3);
    cloud.set_sh_degree(1);
    assert_eq!(cloud.sh_rest_stride(), 9);
    assert_aligned(&cloud);

    cloud.set_sh_degree(2);
    assert_eq!(cloud.sh_rest_stride(), 24);
    for s in cloud.iter() {
        let i = s.position[0] as usize;
        for (j, v) in s.sh_rest[..9].iter().enumerate() {
            assert_eq!(*v, (i * 1000 + j) as f32);
        }
        assert!(s.sh_rest[9..].iter().all(|v| *v == 0.0));
    }

    let mut flat = indexed_cloud(3, 0);
    flat.set_sh_degree(3);
    assert_eq!(flat.sh_rest.len(), 3 * 45);
    flat.validate().unwrap();
}

#[test]
fn filter_min_opacity_uses_alpha() {
    let mut cloud = indexed_cloud(10, 0);
    // Logits run from -5 to 4; alpha >= 0.5 means logit >= 0.
    let removed = transform::filter_min_opacity(&mut cloud, 0.5);
    assert_eq!(removed, 5);
    assert!(cloud.iter().all(|s| s.opacity() >= 0.5));
}

#[test]
fn crop_box_is_inclusive() {
    let mut cloud = indexed_cloud(10, 1);
    let removed = transform::crop_box(&mut cloud, [2.0, -1.0, -1.0], [5.0, 1.0, 1.0]);
    assert_eq!(removed, 6);
    let xs: Vec<f32> = cloud.positions.iter().map(|p| p[0]).collect();
    assert_eq!(xs, [2.0, 3.0, 4.0, 5.0]);
    assert_aligned(&cloud);
}

#[test]
fn sort_by_importance_is_descending() {
    let mut cloud = common::sample_cloud(200, 1, 11);
    transform::sort_by_importance(&mut cloud);
    let keys: Vec<f32> = cloud.iter().map(|s| transform::importance(&s)).collect();
    assert!(keys.windows(2).all(|w| w[0] >= w[1]));
    cloud.validate().unwrap();
}

#[test]
fn scale_and_translate() {
    let mut cloud = indexed_cloud(3, 0);
    transform::scale_uniform(&mut cloud, 2.0);
    transform::translate(&mut cloud, [1.0, 2.0, 3.0]);
    assert_eq!(cloud.positions[2], [5.0, 2.0, 3.0]);
    common::assert_close(cloud.log_scales[0][0], 2f32.ln(), 1e-6, "log scale");
}

#[test]
fn remove_non_finite_checks_every_field() {
    let mut cloud = indexed_cloud(5, 1);
    cloud.sh_rest[9 + 4] = f32::NAN; // splat 1
    cloud.rotations[3][2] = f32::INFINITY;
    assert_eq!(transform::remove_non_finite(&mut cloud), 2);
    let xs: Vec<f32> = cloud.positions.iter().map(|p| p[0]).collect();
    assert_eq!(xs, [0.0, 2.0, 4.0]);
}

#[test]
fn append_requires_matching_degree() {
    let mut a = indexed_cloud(2, 1);
    let b = indexed_cloud(3, 1);
    a.append(&b).unwrap();
    assert_eq!(a.len(), 5);
    a.validate().unwrap();
    assert!(a.append(&indexed_cloud(1, 2)).is_err());
}

#[test]
fn validate_catches_inconsistent_lengths() {
    let mut cloud = indexed_cloud(3, 1);
    cloud.sh_rest.pop();
    assert!(cloud.validate().is_err());
}
