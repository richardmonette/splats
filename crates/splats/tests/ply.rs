mod common;

use splats::SplatCloud;
use splats::formats::ply::{read_ply, write_ply};

fn roundtrip(cloud: &SplatCloud) -> SplatCloud {
    let mut bytes = Vec::new();
    write_ply(cloud, &mut bytes).unwrap();
    read_ply(bytes.as_slice()).unwrap()
}

#[test]
fn binary_roundtrip_is_lossless_for_every_sh_degree() {
    for degree in 0..=3 {
        let cloud = common::sample_cloud(500, degree, 7 + u64::from(degree));
        assert_eq!(roundtrip(&cloud), cloud, "SH degree {degree}");
    }
}

#[test]
fn empty_cloud_roundtrips() {
    let cloud = SplatCloud::new(2);
    assert_eq!(roundtrip(&cloud), cloud);
}

#[test]
fn writer_uses_the_conventional_property_order() {
    let cloud = common::sample_cloud(1, 1, 1);
    let mut bytes = Vec::new();
    write_ply(&cloud, &mut bytes).unwrap();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.starts_with("ply\nformat binary_little_endian 1.0\nelement vertex 1\n"));
    let end = text.find("end_header\n").unwrap();
    let props: Vec<&str> = text[..end]
        .lines()
        .filter_map(|l| l.strip_prefix("property float "))
        .collect();
    let mut expected = vec![
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();
    expected.extend((0..9).map(|j| format!("f_rest_{j}")));
    expected.extend(
        [
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ]
        .map(String::from),
    );
    assert_eq!(props, expected);
    let payload = bytes.len() - (end + "end_header\n".len());
    assert_eq!(payload, expected.len() * 4);
}

fn ascii_header(names: &[String], count: usize) -> String {
    let mut header = format!("ply\nformat ascii 1.0\ncomment test file\nelement vertex {count}\n");
    for name in names {
        header += &format!("property float {name}\n");
    }
    header + "end_header\n"
}

#[test]
fn ascii_reads_any_property_order_and_deinterleaves_sh() {
    let mut names: Vec<String> = [
        "rot_0", "rot_1", "rot_2", "rot_3", "opacity", "x", "y", "z", "scale_0", "scale_1",
        "scale_2", "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    .map(String::from)
    .to_vec();
    names.extend((0..9).map(|j| format!("f_rest_{j}")));
    let row = |x: f32| {
        let mut values = vec![
            1.0, 0.0, 0.0, 0.0, 0.5, x, 2.0, 3.0, -1.0, -2.0, -3.0, 0.25, 0.5, 0.75,
        ];
        values.extend((0..9).map(|j| 100.0 + j as f32));
        values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    // A blank line between rows is tolerated.
    let text = format!("{}{}\n\n{}\n", ascii_header(&names, 2), row(1.0), row(11.0));
    let cloud = read_ply(text.as_bytes()).unwrap();
    assert_eq!(cloud.len(), 2);
    assert_eq!(cloud.sh_degree, 1);
    let s = cloud.get(0);
    assert_eq!(s.position, [1.0, 2.0, 3.0]);
    assert_eq!(s.rotation, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(s.opacity_logit, 0.5);
    assert_eq!(s.log_scale, [-1.0, -2.0, -3.0]);
    assert_eq!(s.sh_dc, [0.25, 0.5, 0.75]);
    // The file is channel-major (f_rest_{c*3+k}); memory is [k][c].
    for k in 0..3 {
        for c in 0..3 {
            assert_eq!(s.sh_rest[k * 3 + c], 100.0 + (c * 3 + k) as f32);
        }
    }
    assert_eq!(cloud.get(1).position[0], 11.0);
}

#[test]
fn big_endian_doubles_with_extra_elements_and_properties() {
    let props: [(&str, &str, f64); 15] = [
        ("double", "x", 1.5),
        ("double", "y", -2.5),
        ("double", "z", 3.25),
        ("uchar", "custom_flag", 255.0),
        ("double", "scale_0", -1.0),
        ("double", "scale_1", -2.0),
        ("double", "scale_2", -3.0),
        ("double", "rot_0", 0.5),
        ("double", "rot_1", 0.5),
        ("double", "rot_2", 0.5),
        ("double", "rot_3", 0.5),
        ("double", "opacity", 1.25),
        ("double", "f_dc_0", 0.1),
        ("double", "f_dc_1", 0.2),
        ("double", "f_dc_2", 0.3),
    ];
    let mut header = String::from(
        "ply\nformat binary_big_endian 1.0\nelement camera 2\nproperty uchar id\n\
         property list uchar int indices\nelement vertex 1\n",
    );
    for (ty, name, _) in &props {
        header += &format!("property {ty} {name}\n");
    }
    header += "end_header\n";
    let mut bytes = header.into_bytes();
    // camera rows: (id 7, [1, 2]) and (id 8, [])
    bytes.extend([7u8, 2]);
    bytes.extend(1i32.to_be_bytes());
    bytes.extend(2i32.to_be_bytes());
    bytes.extend([8u8, 0]);
    for (ty, _, value) in &props {
        match *ty {
            "double" => bytes.extend(value.to_be_bytes()),
            "uchar" => bytes.push(*value as u8),
            _ => unreachable!(),
        }
    }
    let cloud = read_ply(bytes.as_slice()).unwrap();
    assert_eq!(cloud.len(), 1);
    let s = cloud.get(0);
    assert_eq!(s.position, [1.5, -2.5, 3.25]);
    assert_eq!(s.log_scale, [-1.0, -2.0, -3.0]);
    assert_eq!(s.rotation, [0.5; 4]);
    assert_eq!(s.opacity_logit, 1.25);
    assert_eq!(s.sh_dc, [0.1f32, 0.2, 0.3]);
}

#[test]
fn missing_properties_are_listed() {
    let text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n\
                property float z\nend_header\n0 0 0\n";
    let message = read_ply(text.as_bytes()).unwrap_err().to_string();
    for name in ["scale_0", "rot_3", "opacity", "f_dc_2"] {
        assert!(message.contains(name), "{message}");
    }
}

#[test]
fn truncated_data_is_an_error() {
    let cloud = common::sample_cloud(10, 0, 3);
    let mut bytes = Vec::new();
    write_ply(&cloud, &mut bytes).unwrap();
    bytes.truncate(bytes.len() - 10);
    let message = read_ply(bytes.as_slice()).unwrap_err().to_string();
    assert!(message.contains("ends before"), "{message}");
}

#[test]
fn impossible_sh_coefficient_count_is_an_error() {
    let mut names: Vec<String> = [
        "x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    .map(String::from)
    .to_vec();
    names.extend((0..4).map(|j| format!("f_rest_{j}")));
    let text = ascii_header(&names, 0);
    let message = read_ply(text.as_bytes()).unwrap_err().to_string();
    assert!(message.contains("SH degree"), "{message}");
}

#[test]
fn non_ply_input_is_an_error() {
    assert!(read_ply(&b"hello world\n"[..]).is_err());
    assert!(read_ply(&b""[..]).is_err());
    assert!(read_ply(&b"ply\nformat ascii 1.0\n"[..]).is_err());
}
