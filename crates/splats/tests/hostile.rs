//! Crafted inputs that once could panic, hang or exhaust memory, and
//! metadata that must survive conversion.

mod common;

use splats::formats::{gltf, sog, spz};
use splats::{Error, Format};

#[test]
fn gzip_header_running_past_the_end_is_an_error() {
    // gzip with the FNAME flag whose name never terminates before the
    // 8-byte trailer.
    let mut data = vec![0x1f, 0x8b, 8, 0x08, 0, 0, 0, 0, 0, 255];
    data.extend(b"abcdefgh");
    assert!(spz::read_spz(data.as_slice()).is_err());
}

fn gltf_with_accessor(accessor: serde_json::Value) -> Vec<u8> {
    let doc = serde_json::json!({
        "asset": { "version": "2.0" },
        "meshes": [{ "primitives": [{
            "mode": 0,
            "attributes": {
                "POSITION": 0,
                "KHR_gaussian_splatting:ROTATION": 0,
                "KHR_gaussian_splatting:SCALE": 0,
                "KHR_gaussian_splatting:OPACITY": 0,
                "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0": 0
            },
            "extensions": { "KHR_gaussian_splatting": { "kernel": "ellipse", "colorSpace": "srgb_rec709_display" } }
        }] }],
        "accessors": [accessor],
        "bufferViews": [{ "buffer": 0, "byteLength": 16 }],
        "buffers": [{ "byteLength": 16, "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAAAAAA==" }]
    });
    serde_json::to_vec(&doc).unwrap()
}

#[test]
fn gltf_accessor_offsets_cannot_overflow() {
    let doc = gltf_with_accessor(serde_json::json!({
        "bufferView": 0, "byteOffset": 18446744073709551612u64,
        "componentType": 5126, "count": 1, "type": "VEC3"
    }));
    assert!(gltf::read_gltf(doc.as_slice(), None).is_err());
}

#[test]
fn gltf_accessor_without_data_cannot_demand_huge_memory() {
    let doc = gltf_with_accessor(serde_json::json!({
        "componentType": 5126, "count": 1_000_000_000_000u64, "type": "VEC3"
    }));
    assert!(gltf::read_gltf(doc.as_slice(), None).is_err());
}

#[test]
fn gltf_node_graphs_cannot_explode() {
    // 64 nodes each listing the next one twice: 2^64 paths.
    let cloud = common::sample_cloud(4, 0, 1);
    let mut text = Vec::new();
    gltf::write_gltf(&cloud, &mut text).unwrap();
    let mut doc: serde_json::Value = serde_json::from_slice(&text).unwrap();
    let nodes: Vec<serde_json::Value> = (0..64)
        .map(|i| {
            if i == 63 {
                serde_json::json!({ "mesh": 0 })
            } else {
                serde_json::json!({ "children": [i + 1, i + 1] })
            }
        })
        .collect();
    doc["nodes"] = serde_json::Value::Array(nodes);
    let data = serde_json::to_vec(&doc).unwrap();
    assert!(gltf::read_gltf(data.as_slice(), None).is_err());
}

#[test]
fn gltf_tiny_node_scale_is_an_error_not_a_panic() {
    let cloud = common::sample_cloud(4, 0, 1);
    let mut text = Vec::new();
    gltf::write_gltf(&cloud, &mut text).unwrap();
    let mut doc: serde_json::Value = serde_json::from_slice(&text).unwrap();
    doc["nodes"][0]["scale"] = serde_json::json!([1e-50, 1e-50, 1e-50]);
    let data = serde_json::to_vec(&doc).unwrap();
    assert!(gltf::read_gltf(data.as_slice(), None).is_err());
}

#[test]
fn huge_declared_counts_do_not_preallocate() {
    // A PLY header claiming four billion degree-3 splats, with no data.
    let mut header =
        String::from("ply\nformat binary_little_endian 1.0\nelement vertex 4000000000\n");
    for name in [
        "x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
    ] {
        header += &format!("property float {name}\n");
    }
    for j in 0..45 {
        header += &format!("property float f_rest_{j}\n");
    }
    header += "end_header\n";
    assert!(matches!(
        splats::formats::ply::read_ply(header.as_bytes()),
        Err(Error::Parse(_))
    ));
}

#[test]
fn empty_spz_is_refused() {
    let cloud = splats::SplatCloud::new(0);
    assert!(matches!(
        spz::write_spz(&cloud, Vec::new()),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn antialiased_flag_survives_spz_and_sog() {
    let mut cloud = common::sample_cloud(64, 1, 3);
    cloud.antialiased = true;
    for format in [Format::Spz, Format::Sog] {
        let mut bytes = Vec::new();
        format.write(&cloud, &mut bytes).unwrap();
        assert!(
            format.read(bytes.as_slice()).unwrap().antialiased,
            "{format}"
        );
    }
    let mut plain = cloud.clone();
    plain.antialiased = false;
    let mut bytes = Vec::new();
    sog::write_sog(&plain, &mut bytes).unwrap();
    assert!(!sog::read_sog(bytes.as_slice()).unwrap().antialiased);
}
