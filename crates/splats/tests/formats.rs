//! Round trips through every format, checked against per-format
//! quantization tolerances, plus robustness against damaged input.

mod common;

use splats::formats::{gltf, ksplat, sog, spz};
use splats::{Format, SplatCloud, transform};

/// Largest error allowed per attribute after a round trip.
#[derive(Clone, Copy, Debug)]
struct Tolerance {
    position: f32,
    log_scale: f32,
    color: f32,
    opacity: f32,
    /// Minimum |dot| between original and decoded unit quaternions.
    rotation_dot: f32,
    sh: f32,
}

const EXACT: Tolerance = Tolerance {
    position: 1e-6,
    log_scale: 1e-5,
    color: 1e-5,
    opacity: 1e-5,
    rotation_dot: 0.99999,
    sh: 1e-6,
};

fn tolerance(format: Format) -> Tolerance {
    match format {
        Format::Ply => EXACT,
        Format::Glb | Format::Gltf => EXACT,
        Format::CompressedPly => Tolerance {
            position: 0.02,
            log_scale: 0.005,
            color: 0.005,
            opacity: 0.003,
            rotation_dot: 0.999,
            sh: 0.035,
        },
        Format::Splat => Tolerance {
            position: 1e-6,
            log_scale: 1e-5,
            color: 0.5 / 255.0 + 1e-4,
            opacity: 0.5 / 255.0 + 1e-4,
            rotation_dot: 0.999,
            sh: 0.0,
        },
        Format::Spz => Tolerance {
            position: 3e-4,
            log_scale: 0.035,
            color: 0.006,
            opacity: 0.5 / 255.0 + 1e-4,
            rotation_dot: 0.9995,
            sh: 0.07,
        },
        Format::Ksplat => Tolerance {
            position: 2e-4,
            log_scale: 2e-3,
            color: 0.5 / 255.0 + 1e-4,
            opacity: 0.5 / 255.0 + 1e-4,
            rotation_dot: 0.9999,
            sh: 2e-3,
        },
        Format::Sog => Tolerance {
            position: 2e-3,
            log_scale: 0.05,
            color: 0.01,
            opacity: 0.5 / 255.0 + 1e-4,
            rotation_dot: 0.998,
            sh: 0.02,
        },
        other => panic!("no tolerance for {other}"),
    }
}

/// For each decoded splat, finds the original with the nearest position
/// (formats may reorder splats) and checks every attribute.
fn assert_matches(original: &SplatCloud, decoded: &SplatCloud, tol: Tolerance, what: &str) {
    assert_eq!(decoded.len(), original.len(), "{what}: splat count");
    let mut used = vec![false; original.len()];
    for d in decoded.iter() {
        let (index, distance) = original
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let dist = (0..3)
                    .map(|a| (p[a] - d.position[a]).abs())
                    .fold(0.0, f32::max);
                (i, dist)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("not empty");
        assert!(
            distance <= tol.position,
            "{what}: position off by {distance}"
        );
        assert!(!used[index], "{what}: two splats decoded onto one original");
        used[index] = true;
        let o = original.get(index);
        for a in 0..3 {
            common::assert_close(
                o.log_scale[a],
                d.log_scale[a],
                tol.log_scale,
                &format!("{what} log scale"),
            );
            let expected = o.color()[a].clamp(0.0, 1.0);
            common::assert_close(expected, d.color()[a], tol.color, &format!("{what} color"));
        }
        common::assert_close(
            o.opacity(),
            d.opacity(),
            tol.opacity,
            &format!("{what} opacity"),
        );
        common::assert_same_rotation(o.rotation, d.rotation, 1.0 - tol.rotation_dot);
        let shared = d.sh_rest.len().min(o.sh_rest.len());
        for k in 0..shared {
            common::assert_close(
                o.sh_rest[k],
                d.sh_rest[k],
                tol.sh,
                &format!("{what} SH[{k}]"),
            );
        }
    }
}

/// A cloud whose values stay inside every format's representable range.
fn representable_cloud(n: usize, degree: u8, seed: u64) -> SplatCloud {
    let mut cloud = common::sample_cloud(n, degree, seed);
    transform::normalize_rotations(&mut cloud);
    cloud
}

fn roundtrip(format: Format, cloud: &SplatCloud) -> SplatCloud {
    let mut bytes = Vec::new();
    format
        .write(cloud, &mut bytes)
        .unwrap_or_else(|e| panic!("{format} write: {e}"));
    format
        .read(bytes.as_slice())
        .unwrap_or_else(|e| panic!("{format} read: {e}"))
}

#[test]
fn every_format_roundtrips_within_tolerance() {
    for format in Format::ALL {
        for degree in [0, 1, 3] {
            let cloud = representable_cloud(500, degree, 100 + u64::from(degree));
            let back = roundtrip(format, &cloud);
            assert_eq!(
                back.sh_degree,
                degree.min(format.max_sh_degree()),
                "{format} SH degree"
            );
            assert_matches(
                &cloud,
                &back,
                tolerance(format),
                &format!("{format} (SH {degree})"),
            );
        }
    }
}

#[test]
fn ksplat_levels() {
    let cloud = representable_cloud(800, 2, 9);
    for level in 0..=2u8 {
        let mut bytes = Vec::new();
        let options = ksplat::KsplatOptions {
            compression_level: level,
            ..Default::default()
        };
        ksplat::write_ksplat_with(&cloud, &mut bytes, options).unwrap();
        let per_splat = [80, 42, 33][level as usize];
        let back = ksplat::read_ksplat(bytes.as_slice()).unwrap();
        let mut tol = tolerance(Format::Ksplat);
        match level {
            0 => {
                tol.position = 1e-6;
                tol.sh = 1e-6;
            }
            2 => tol.sh = 0.005,
            _ => {}
        }
        assert!(bytes.len() >= 5120 + 800 * per_splat, "level {level} size");
        // Small buckets force many partially filled buckets.
        assert_matches(&cloud, &back, tol, &format!("ksplat level {level}"));
        let mut small = Vec::new();
        ksplat::write_ksplat_with(
            &cloud,
            &mut small,
            ksplat::KsplatOptions {
                compression_level: level,
                block_size: 2.0,
                bucket_size: 7,
            },
        )
        .unwrap();
        let back = ksplat::read_ksplat(small.as_slice()).unwrap();
        tol.position = tol.position.min(1e-4);
        assert_matches(
            &cloud,
            &back,
            tol,
            &format!("ksplat level {level}, small buckets"),
        );
    }
}

#[test]
fn spz_versions() {
    let cloud = representable_cloud(400, 3, 21);
    for version in 2..=4 {
        let mut bytes = Vec::new();
        let options = spz::SpzWriteOptions {
            version,
            ..Default::default()
        };
        spz::write_spz_with(&cloud, &mut bytes, options).unwrap();
        let back = spz::read_spz(bytes.as_slice()).unwrap();
        let mut tol = tolerance(Format::Spz);
        if version == 2 {
            // Three bytes with w derived: least precise when w is small.
            tol.rotation_dot = 0.995;
        }
        assert_matches(&cloud, &back, tol, &format!("spz v{version}"));
    }
}

#[test]
fn spz_far_positions_use_fewer_fractional_bits() {
    let mut cloud = representable_cloud(50, 0, 4);
    transform::scale_uniform(&mut cloud, 500.0); // up to ±5000 units
    let back = roundtrip(Format::Spz, &cloud);
    let mut tol = tolerance(Format::Spz);
    tol.position = 0.01;
    tol.log_scale = 0.04;
    assert_matches(&cloud, &back, tol, "spz far positions");
}

#[test]
fn spz_coordinate_conversion_is_reversible() {
    let cloud = representable_cloud(100, 1, 8);
    let mut bytes = Vec::new();
    spz::write_spz(&cloud, &mut bytes).unwrap();
    let raw = spz::read_spz_with(
        bytes.as_slice(),
        spz::SpzReadOptions {
            convert_coordinates: false,
        },
    )
    .unwrap();
    // Stored y and z are flipped relative to the input.
    let mut expected = cloud.clone();
    transform::flip_axes(&mut expected, [false, true, true]);
    assert_matches(&expected, &raw, tolerance(Format::Spz), "spz raw axes");
}

#[test]
fn sog_versions_and_layouts() {
    let cloud = representable_cloud(600, 2, 31);
    for version in [2, 3] {
        let options = sog::SogOptions {
            version,
            ..Default::default()
        };
        let mut bytes = Vec::new();
        sog::write_sog_with(&cloud, &mut bytes, options).unwrap();
        assert_eq!(&bytes[..4], b"PK\x03\x04");
        let back = sog::read_sog(bytes.as_slice()).unwrap();
        let mut tol = tolerance(Format::Sog);
        if version == 3 {
            tol.opacity = 0.01;
        }
        assert_matches(&cloud, &back, tol, &format!("sog v{version}"));
    }

    // Unbundled form: meta.json plus images in a folder.
    let dir = std::env::temp_dir().join(format!("splats-sog-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let meta = dir.join("meta.json");
    splats::write_file(&cloud, &meta).unwrap();
    assert!(dir.join("means_l.webp").exists());
    assert_eq!(Format::detect(&meta).unwrap(), Format::Sog);
    let back = splats::read_file(&meta).unwrap();
    assert_matches(&cloud, &back, tolerance(Format::Sog), "sog unbundled");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn sog_palette_compresses_many_splats() {
    // More splats than palette entries: SH goes through clustering.
    let cloud = representable_cloud(3000, 1, 77);
    let options = sog::SogOptions {
        palette_size: Some(256),
        ..Default::default()
    };
    let mut bytes = Vec::new();
    sog::write_sog_with(&cloud, &mut bytes, options).unwrap();
    let back = sog::read_sog(bytes.as_slice()).unwrap();
    assert_eq!(back.len(), cloud.len());
    // Clustering error is bounded by the spread of the data (values in ±0.5).
    let mut tol = tolerance(Format::Sog);
    tol.sh = 1.0; // checked as RMS below
    assert_matches(&cloud, &back, tol, "sog palette");
    let rms = {
        let mut sorted_orig = cloud.clone();
        let mut sorted_back = back.clone();
        transform::sort_morton(&mut sorted_orig);
        transform::sort_morton(&mut sorted_back);
        let n = sorted_orig.sh_rest.len() as f32;
        (sorted_orig
            .sh_rest
            .iter()
            .zip(&sorted_back.sh_rest)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / n)
            .sqrt()
    };
    assert!(rms < 0.25, "palette RMS error {rms}");
}

#[test]
fn compressed_ply_is_detected_by_the_ply_reader() {
    let cloud = representable_cloud(700, 3, 13);
    let mut bytes = Vec::new();
    splats::formats::compressed_ply::write_compressed_ply(&cloud, &mut bytes).unwrap();
    let header_end = bytes
        .windows(11)
        .position(|w| w == b"end_header\n")
        .unwrap()
        + 11;
    let header = std::str::from_utf8(&bytes[..header_end]).unwrap();
    assert!(header.contains("element chunk 3\n"));
    assert!(header.contains("property uint packed_position\n"));
    assert!(header.contains("element sh 700\n"));
    assert_eq!(bytes.len() - header_end, 3 * 72 + 700 * 16 + 700 * 45);
    let back = splats::formats::ply::read_ply(bytes.as_slice()).unwrap();
    assert_matches(
        &cloud,
        &back,
        tolerance(Format::CompressedPly),
        "compressed ply via read_ply",
    );
}

#[test]
fn gltf_coordinates_and_structure() {
    let cloud = representable_cloud(200, 3, 17);
    let mut text = Vec::new();
    gltf::write_gltf(&cloud, &mut text).unwrap();
    let doc: serde_json::Value = serde_json::from_slice(&text).unwrap();
    let primitive = &doc["meshes"][0]["primitives"][0];
    assert_eq!(primitive["mode"], 0);
    assert_eq!(
        primitive["extensions"]["KHR_gaussian_splatting"]["kernel"],
        "ellipse"
    );
    assert_eq!(doc["extensionsUsed"][0], "KHR_gaussian_splatting");
    let attributes = primitive["attributes"].as_object().unwrap();
    assert!(attributes.contains_key("KHR_gaussian_splatting:SH_DEGREE_3_COEF_6"));
    assert!(attributes.contains_key("COLOR_0"));
    assert!(doc["accessors"][0]["min"].is_array());

    // Default conversion turns the scene 180° about Z.
    let raw = gltf::read_gltf_with(
        text.as_slice(),
        None,
        gltf::GltfOptions {
            convert_coordinates: false,
        },
    )
    .unwrap();
    let mut expected = cloud.clone();
    transform::flip_axes(&mut expected, [true, true, false]);
    assert_matches(&expected, &raw, EXACT, "gltf raw axes");
}

#[test]
fn gltf_node_transforms_are_applied() {
    let cloud = representable_cloud(100, 2, 23);
    let options = gltf::GltfOptions {
        convert_coordinates: false,
    };
    let mut text = Vec::new();
    gltf::write_gltf_with(&cloud, &mut text, options).unwrap();
    let mut doc: serde_json::Value = serde_json::from_slice(&text).unwrap();
    // Parent node: translate; child node: rotate 90° about Y and scale 2.
    let half = std::f32::consts::FRAC_1_SQRT_2;
    doc["nodes"] = serde_json::json!([
        { "translation": [1.0, 2.0, 3.0], "children": [1] },
        { "mesh": 0, "rotation": [0.0, half, 0.0, half], "scale": [2.0, 2.0, 2.0] }
    ]);
    let edited = serde_json::to_vec(&doc).unwrap();
    let back = gltf::read_gltf_with(edited.as_slice(), None, options).unwrap();

    let mut expected = cloud.clone();
    transform::scale_uniform(&mut expected, 2.0);
    transform::rotate(&mut expected, [half, 0.0, half, 0.0]);
    transform::translate(&mut expected, [1.0, 2.0, 3.0]);
    let tol = Tolerance {
        position: 1e-4,
        log_scale: 1e-5,
        color: 1e-4,
        opacity: 1e-5,
        rotation_dot: 0.9999,
        sh: 1e-4,
    };
    assert_matches(&expected, &back, tol, "gltf node transforms");

    doc["nodes"][1]["scale"] = serde_json::json!([1.0, 2.0, 1.0]);
    let edited = serde_json::to_vec(&doc).unwrap();
    assert!(gltf::read_gltf_with(edited.as_slice(), None, options).is_err());
}

#[test]
fn gltf_reads_spz_compressed_primitives() {
    // Build a GLB by hand whose primitive carries an SPZ payload.
    let cloud = representable_cloud(150, 1, 29);
    let mut payload_cloud = cloud.clone();
    transform::flip_axes(&mut payload_cloud, [true, false, true]); // glTF -> right-up-back
    let mut payload = Vec::new();
    spz::write_spz_with(
        &payload_cloud,
        &mut payload,
        spz::SpzWriteOptions {
            convert_coordinates: false,
            ..Default::default()
        },
    )
    .unwrap();
    while payload.len() % 4 != 0 {
        payload.push(0);
    }
    let n = cloud.len();
    let mut attributes = serde_json::Map::new();
    let mut accessors = Vec::new();
    for (name, ty) in [
        ("POSITION", "VEC3"),
        ("KHR_gaussian_splatting:ROTATION", "VEC4"),
        ("KHR_gaussian_splatting:SCALE", "VEC3"),
        ("KHR_gaussian_splatting:OPACITY", "SCALAR"),
        ("KHR_gaussian_splatting:SH_DEGREE_0_COEF_0", "VEC3"),
    ] {
        attributes.insert(name.into(), accessors.len().into());
        accessors.push(serde_json::json!({ "componentType": 5126, "count": n, "type": ty }));
    }
    let doc = serde_json::json!({
        "asset": { "version": "2.0" },
        "extensionsUsed": ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"],
        "meshes": [{ "primitives": [{
            "mode": 0,
            "attributes": attributes,
            "extensions": { "KHR_gaussian_splatting": {
                "kernel": "ellipse", "colorSpace": "srgb_rec709_display",
                "extensions": { "KHR_gaussian_splatting_compression_spz_2": { "bufferView": 0 } }
            } }
        }] }],
        "accessors": accessors,
        "bufferViews": [{ "buffer": 0, "byteLength": payload.len() }],
        "buffers": [{ "byteLength": payload.len() }]
    });
    let mut json = serde_json::to_vec(&doc).unwrap();
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let mut glb = Vec::new();
    glb.extend(0x4654_6C67u32.to_le_bytes());
    glb.extend(2u32.to_le_bytes());
    glb.extend(((12 + 8 + json.len() + 8 + payload.len()) as u32).to_le_bytes());
    glb.extend((json.len() as u32).to_le_bytes());
    glb.extend(0x4E4F_534Au32.to_le_bytes());
    glb.extend(&json);
    glb.extend((payload.len() as u32).to_le_bytes());
    glb.extend(0x004E_4942u32.to_le_bytes());
    glb.extend(&payload);

    let back = gltf::read_glb_with(
        glb.as_slice(),
        None,
        gltf::GltfOptions {
            convert_coordinates: false,
        },
    )
    .unwrap();
    assert_matches(&cloud, &back, tolerance(Format::Spz), "spz inside glb");
}

#[test]
fn damaged_files_never_panic() {
    let cloud = representable_cloud(300, 2, 41);
    for format in Format::ALL {
        let mut bytes = Vec::new();
        format.write(&cloud, &mut bytes).unwrap();
        let len = bytes.len();
        for cut in (0..20).map(|i| len * i / 20).chain([len - 1]) {
            let _ = format.read(&bytes[..cut]);
        }
        let mut rng = common::Rng::new(len as u64);
        for _ in 0..30 {
            let mut damaged = bytes.clone();
            for _ in 0..4 {
                let at = ((rng.next_f32() * 0.5 + 0.5) * (len - 1) as f32) as usize;
                damaged[at] ^= 1 << ((rng.next_f32().abs() * 7.0) as u32);
            }
            let _ = format.read(damaged.as_slice());
        }
    }
}

#[test]
fn view_dependent_color_survives_rotation() {
    let mut cloud = representable_cloud(20, 3, 51);
    let original = cloud.clone();
    let q = splats::normalize_quaternion([0.3, -0.4, 0.8, 0.2]);
    transform::rotate(&mut cloud, q);
    let rotate_vec = |v: [f32; 3]| {
        // v' = q v q*
        let [w, x, y, z] = q;
        let t = [
            2.0 * (y * v[2] - z * v[1]),
            2.0 * (z * v[0] - x * v[2]),
            2.0 * (x * v[1] - y * v[0]),
        ];
        [
            v[0] + w * t[0] + (y * t[2] - z * t[1]),
            v[1] + w * t[1] + (z * t[0] - x * t[2]),
            v[2] + w * t[2] + (x * t[1] - y * t[0]),
        ]
    };
    for (a, b) in original.iter().zip(cloud.iter()) {
        for d in [[1.0, 0.0, 0.0], [0.3, -0.5, 0.8], [-0.2, 0.9, 0.1]] {
            let before = a.view_color(d);
            let after = b.view_color(rotate_vec(d));
            for c in 0..3 {
                common::assert_close(before[c], after[c], 1e-4, "view color");
            }
        }
    }
    // A 180° turn about X is the same scene as flipping Y and Z. The
    // quaternions differ (A·R vs A·R·A) but the covariances agree.
    let mut turned = original.clone();
    transform::rotate(&mut turned, transform::axis_angle([1.0, 0.0, 0.0], 180.0));
    let mut flipped = original.clone();
    transform::flip_axes(&mut flipped, [false, true, true]);
    for (a, b) in turned.iter().zip(flipped.iter()) {
        for i in 0..3 {
            common::assert_close(a.position[i], b.position[i], 1e-5, "position");
        }
        let (ca, cb) = (covariance(&a), covariance(&b));
        for r in 0..3 {
            for c in 0..3 {
                common::assert_close(ca[r][c], cb[r][c], 1e-5, "covariance");
            }
        }
        for (x, y) in a.sh_rest.iter().zip(b.sh_rest) {
            common::assert_close(*x, *y, 1e-4, "SH");
        }
    }
}

/// `R · diag(s²) · Rᵀ` for a splat.
fn covariance(s: &splats::SplatRef<'_>) -> [[f32; 3]; 3] {
    let [w, x, y, z] = splats::normalize_quaternion(s.rotation);
    let r = [
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
    ];
    let s2 = s.scale().map(|v| v * v);
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| r[i][k] * s2[k] * r[j][k]).sum())
    })
}
