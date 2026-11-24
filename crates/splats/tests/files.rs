mod common;

use std::path::PathBuf;

use splats::{Error, Format};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("splats-test-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn format_from_path_ignores_case() {
    assert_eq!(Format::from_path("a/b/scene.PLY"), Some(Format::Ply));
    assert_eq!(Format::from_path("scene.splat"), Some(Format::Splat));
    assert_eq!(Format::from_path("scene.txt"), None);
    assert_eq!(Format::from_path("scene"), None);
}

#[test]
fn files_roundtrip_through_every_format() {
    let dir = temp_dir("roundtrip");
    let cloud = common::sample_cloud(50, 1, 5);
    for format in Format::ALL {
        let path = dir.join(format!("scene.{}", format.extension()));
        splats::write_file(&cloud, &path).unwrap();
        assert_eq!(Format::detect(&path).unwrap(), format);
        let back = splats::read_file(&path).unwrap();
        assert_eq!(back.len(), cloud.len(), "{format}");
        if format.is_lossless() {
            assert_eq!(back, cloud, "{format}");
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn ply_is_detected_by_content() {
    let dir = temp_dir("detect");
    let path = dir.join("scene.bin");
    splats::write_file_as(&common::sample_cloud(3, 0, 1), &path, Format::Ply).unwrap();
    assert_eq!(Format::detect(&path).unwrap(), Format::Ply);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn unknown_extension_is_unsupported() {
    let cloud = common::sample_cloud(1, 0, 1);
    let err = splats::write_file(&cloud, std::env::temp_dir().join("x.unknown")).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err}");
}
