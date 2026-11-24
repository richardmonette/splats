//! Reads real-world sample files when `SPLATS_SAMPLES` names a folder of
//! them (files, plus subfolders holding a `meta.json`). Skipped otherwise.
//!
//! ```text
//! SPLATS_SAMPLES=~/splat-samples cargo test --test samples -- --nocapture
//! ```

use std::path::PathBuf;

use splats::{Format, transform};

fn sample_paths() -> Option<Vec<PathBuf>> {
    let dir = PathBuf::from(std::env::var_os("SPLATS_SAMPLES")?);
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("SPLATS_SAMPLES is readable") {
        let path = entry.expect("readable entry").path();
        if path.is_file() {
            paths.push(path);
        } else if path.join("meta.json").is_file() {
            paths.push(path.join("meta.json"));
        }
    }
    paths.sort();
    Some(paths)
}

#[test]
fn every_sample_reads_and_converts() {
    let Some(paths) = sample_paths() else {
        eprintln!("SPLATS_SAMPLES not set; skipping");
        return;
    };
    assert!(!paths.is_empty(), "no samples found");
    for path in paths {
        let format = Format::detect(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let cloud = splats::read_file_as(&path, format)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(!cloud.is_empty(), "{}: no splats", path.display());
        cloud.validate().unwrap();
        let invalid = cloud.iter().filter(|s| !transform::is_finite(s)).count();
        assert_eq!(invalid, 0, "{}: non-finite splats", path.display());
        // Every sample must survive a trip through every writer.
        for target in Format::ALL {
            let mut bytes = Vec::new();
            target
                .write(&cloud, &mut bytes)
                .unwrap_or_else(|e| panic!("{} -> {target}: {e}", path.display()));
            let back = target
                .read(bytes.as_slice())
                .unwrap_or_else(|e| panic!("{} -> {target} -> read: {e}", path.display()));
            assert_eq!(back.len(), cloud.len(), "{} -> {target}", path.display());
        }
        eprintln!("ok  {} ({format}, {} splats)", path.display(), cloud.len());
    }
}
