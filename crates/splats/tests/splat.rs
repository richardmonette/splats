mod common;

use common::{assert_close, assert_same_rotation};
use splats::formats::splat::{RECORD_SIZE, read_splat, write_splat};

#[test]
fn roundtrip_keeps_geometry_and_quantizes_color() {
    let cloud = common::sample_cloud(300, 2, 3);
    let mut bytes = Vec::new();
    write_splat(&cloud, &mut bytes).unwrap();
    assert_eq!(bytes.len(), cloud.len() * RECORD_SIZE);

    let back = read_splat(bytes.as_slice()).unwrap();
    assert_eq!(back.sh_degree, 0);
    assert_eq!(back.len(), cloud.len());
    for (a, b) in cloud.iter().zip(back.iter()) {
        assert_eq!(a.position, b.position);
        for i in 0..3 {
            assert_close(a.log_scale[i], b.log_scale[i], 1e-5, "log scale");
            let expected = a.color()[i].clamp(0.0, 1.0);
            assert_close(expected, b.color()[i], 0.5 / 255.0 + 1e-5, "color");
        }
        assert_close(a.opacity(), b.opacity(), 0.5 / 255.0 + 1e-4, "opacity");
        assert_same_rotation(a.rotation, b.rotation, 1e-3);
    }
}

#[test]
fn partial_record_is_an_error() {
    let cloud = common::sample_cloud(2, 0, 1);
    let mut bytes = Vec::new();
    write_splat(&cloud, &mut bytes).unwrap();
    assert!(read_splat(&bytes[..40]).is_err());
    assert_eq!(read_splat(&bytes[..0]).unwrap().len(), 0);
}
