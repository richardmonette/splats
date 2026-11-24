//! Chunk-quantized compressed `.ply` (usually named `*.compressed.ply`).
//!
//! Splats are grouped into chunks of 256. Each chunk stores the range of
//! its positions, log scales and colors, and each splat packs position,
//! rotation, scale and color+opacity into four 32-bit words (16 bytes per
//! splat). Optional higher-order SH is stored as one byte per coefficient.
//!
//! The writer sorts splats along a Morton curve first so that chunks are
//! spatially tight; readers accept any order. [`read_ply`](super::ply::read_ply)
//! detects this layout automatically.

use std::io::{BufRead, BufWriter, Write};

use super::ply::{Header, MAX_PREALLOCATED_SPLATS, err, read_rows, skip_element};
use crate::cloud::{
    SH_C0, SplatCloud, SplatRef, logit, normalize_quaternion, sh_rest_coeffs, sigmoid,
};
use crate::error::Result;
use crate::transform;

/// Splats per chunk.
pub const CHUNK_SIZE: usize = 256;

const CHUNK_PROPS: [&str; 18] = [
    "min_x",
    "min_y",
    "min_z",
    "max_x",
    "max_y",
    "max_z",
    "min_scale_x",
    "min_scale_y",
    "min_scale_z",
    "max_scale_x",
    "max_scale_y",
    "max_scale_z",
    "min_r",
    "min_g",
    "min_b",
    "max_r",
    "max_g",
    "max_b",
];
const VERTEX_PROPS: [&str; 4] = [
    "packed_position",
    "packed_rotation",
    "packed_scale",
    "packed_color",
];

pub(crate) fn is_compressed(header: &Header) -> bool {
    header.elements.iter().any(|e| e.name == "chunk")
        && header
            .elements
            .iter()
            .any(|e| e.name == "vertex" && e.property("packed_position").is_some())
}

#[derive(Clone, Copy, Default)]
struct Chunk {
    min: [f32; 3],
    max: [f32; 3],
    min_scale: [f32; 3],
    max_scale: [f32; 3],
    min_color: [f32; 3],
    max_color: [f32; 3],
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn unorm(v: u32, bits: u32) -> f32 {
    let t = (1u32 << bits) - 1;
    (v & t) as f32 / t as f32
}

fn unpack_111011(v: u32) -> [f32; 3] {
    [unorm(v >> 21, 11), unorm(v >> 11, 10), unorm(v, 11)]
}

fn unpack_rotation(v: u32) -> [f32; 4] {
    let s = std::f32::consts::SQRT_2;
    let a = (unorm(v >> 20, 10) - 0.5) * s;
    let b = (unorm(v >> 10, 10) - 0.5) * s;
    let c = (unorm(v, 10) - 0.5) * s;
    let m = (1.0 - (a * a + b * b + c * c)).max(0.0).sqrt();
    match v >> 30 {
        0 => [m, a, b, c],
        1 => [a, m, b, c],
        2 => [a, b, m, c],
        _ => [a, b, c, m],
    }
}

fn sh_byte_to_value(b: f32) -> f32 {
    let n = if b == 0.0 {
        0.0
    } else if b == 255.0 {
        1.0
    } else {
        (b + 0.5) / 256.0
    };
    (n - 0.5) * 8.0
}

/// Reads the body of a compressed `.ply` whose header has been parsed.
pub(crate) fn read_body<R: BufRead>(reader: &mut R, header: &Header) -> Result<SplatCloud> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut cloud: Option<SplatCloud> = None;
    let sh_element = header.elements.iter().find(|e| e.name == "sh");
    let sh_degree = match sh_element.map_or(0, |e| e.properties.len()) {
        0 => 0,
        9 => 1,
        24 => 2,
        45 => 3,
        n => return Err(err(format!("{n} SH properties do not match any SH degree"))),
    };
    for element in &header.elements {
        match element.name.as_str() {
            "chunk" => {
                let index = |name: &str| element.property(name);
                let idx: Vec<Option<usize>> = CHUNK_PROPS.iter().map(|n| index(n)).collect();
                if idx[..12].iter().any(Option::is_none) {
                    return Err(err("chunk element lacks position or scale bounds"));
                }
                let has_color = idx[12..].iter().all(Option::is_some);
                chunks.reserve(element.count.min(MAX_PREALLOCATED_SPLATS));
                read_rows(reader, element, header.encoding, |row| {
                    let get = |i: usize| idx[i].map_or(0.0, |j| row[j] as f32);
                    chunks.push(Chunk {
                        min: [get(0), get(1), get(2)],
                        max: [get(3), get(4), get(5)],
                        min_scale: [get(6), get(7), get(8)],
                        max_scale: [get(9), get(10), get(11)],
                        min_color: if has_color {
                            [get(12), get(13), get(14)]
                        } else {
                            [0.0; 3]
                        },
                        max_color: if has_color {
                            [get(15), get(16), get(17)]
                        } else {
                            [1.0; 3]
                        },
                    });
                    Ok(())
                })?;
            }
            "vertex" => {
                let idx: Vec<usize> = VERTEX_PROPS
                    .iter()
                    .map(|n| {
                        element
                            .property(n)
                            .ok_or_else(|| err(format!("vertex lacks {n}")))
                    })
                    .collect::<Result<_>>()?;
                if chunks.len() < element.count.div_ceil(CHUNK_SIZE) {
                    return Err(err(format!(
                        "{} vertices need {} chunks, found {}",
                        element.count,
                        element.count.div_ceil(CHUNK_SIZE),
                        chunks.len()
                    )));
                }
                let mut out = SplatCloud::with_capacity(
                    sh_degree,
                    element.count.min(MAX_PREALLOCATED_SPLATS),
                );
                let zeros = vec![0.0f32; out.sh_rest_stride()];
                let mut i = 0usize;
                read_rows(reader, element, header.encoding, |row| {
                    let c = &chunks[i / CHUNK_SIZE];
                    let word = |k: usize| row[idx[k]] as u32;
                    let p = unpack_111011(word(0));
                    let s = unpack_111011(word(2));
                    let color = word(3);
                    let rgb = [color >> 24, color >> 16, color >> 8].map(|v| unorm(v, 8));
                    out.push(SplatRef {
                        position: std::array::from_fn(|a| lerp(c.min[a], c.max[a], p[a])),
                        log_scale: std::array::from_fn(|a| {
                            lerp(c.min_scale[a], c.max_scale[a], s[a])
                        }),
                        rotation: unpack_rotation(word(1)),
                        opacity_logit: logit(unorm(color, 8)),
                        sh_dc: std::array::from_fn(|a| {
                            (lerp(c.min_color[a], c.max_color[a], rgb[a]) - 0.5) / SH_C0
                        }),
                        sh_rest: &zeros,
                    });
                    i += 1;
                    Ok(())
                })?;
                cloud = Some(out);
            }
            "sh" => {
                let out = cloud
                    .as_mut()
                    .ok_or_else(|| err("sh element must follow the vertex element"))?;
                if element.count != out.len() {
                    return Err(err("sh element count differs from the vertex count"));
                }
                let k = sh_rest_coeffs(sh_degree);
                let stride = out.sh_rest_stride();
                // Column j (channel-major `c * k + i`) -> interleaved `i * 3 + c`.
                let mut columns = vec![usize::MAX; element.properties.len()];
                for (col, p) in element.properties.iter().enumerate() {
                    let j: usize = p
                        .name
                        .strip_prefix("f_rest_")
                        .and_then(|s| s.parse().ok())
                        .filter(|&j| j < 3 * k)
                        .ok_or_else(|| err(format!("unexpected SH property '{}'", p.name)))?;
                    columns[col] = (j % k) * 3 + j / k;
                }
                let mut i = 0usize;
                read_rows(reader, element, header.encoding, |row| {
                    let block = &mut out.sh_rest[i * stride..(i + 1) * stride];
                    for (&dst, &v) in columns.iter().zip(row) {
                        block[dst] = sh_byte_to_value(v as f32);
                    }
                    i += 1;
                    Ok(())
                })?;
            }
            _ => skip_element(reader, element, header.encoding)?,
        }
    }
    cloud.ok_or_else(|| err("no vertex element"))
}

fn normalize(x: f32, min: f32, max: f32) -> f32 {
    if x <= min {
        0.0
    } else if x >= max {
        1.0
    } else if max - min < 1e-5 {
        0.0
    } else {
        (x - min) / (max - min)
    }
}

fn pack_unorm(v: f32, bits: u32) -> u32 {
    let t = ((1u32 << bits) - 1) as f32;
    (v * t + 0.5).floor().clamp(0.0, t) as u32
}

fn pack_111011(v: [f32; 3]) -> u32 {
    (pack_unorm(v[0], 11) << 21) | (pack_unorm(v[1], 10) << 11) | pack_unorm(v[2], 11)
}

fn pack_rotation(q: [f32; 4]) -> u32 {
    let mut q = normalize_quaternion(q);
    let mut largest = 0;
    for i in 1..4 {
        if q[i].abs() > q[largest].abs() {
            largest = i;
        }
    }
    if q[largest] < 0.0 {
        q = q.map(|v| -v);
    }
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let mut out = largest as u32;
    for (i, &v) in q.iter().enumerate() {
        if i != largest {
            out = (out << 10) | pack_unorm(v * half_sqrt2 + 0.5, 10);
        }
    }
    out
}

/// Writes `cloud` as a compressed `.ply`. Splats are reordered along a
/// Morton curve.
pub fn write_compressed_ply<W: Write>(cloud: &SplatCloud, writer: W) -> Result<()> {
    cloud.validate()?;
    let mut sorted = cloud.clone();
    transform::sort_morton(&mut sorted);
    let cloud = &sorted;
    let n = cloud.len();
    let k = sh_rest_coeffs(cloud.sh_degree);
    let chunk_count = n.div_ceil(CHUNK_SIZE);

    let mut header = format!(
        "ply\nformat binary_little_endian 1.0\ncomment {}\nelement chunk {chunk_count}\n",
        concat!("splats ", env!("CARGO_PKG_VERSION"))
    );
    for name in CHUNK_PROPS {
        header += &format!("property float {name}\n");
    }
    header += &format!("element vertex {n}\n");
    for name in VERTEX_PROPS {
        header += &format!("property uint {name}\n");
    }
    if k > 0 {
        header += &format!("element sh {n}\n");
        for j in 0..3 * k {
            header += &format!("property uchar f_rest_{j}\n");
        }
    }
    header += "end_header\n";

    let mut chunks = Vec::with_capacity(chunk_count);
    for start in (0..n).step_by(CHUNK_SIZE) {
        let end = (start + CHUNK_SIZE).min(n);
        let mut c = Chunk {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
            min_scale: [f32::INFINITY; 3],
            max_scale: [f32::NEG_INFINITY; 3],
            min_color: [f32::INFINITY; 3],
            max_color: [f32::NEG_INFINITY; 3],
        };
        for s in (start..end).map(|i| cloud.get(i)) {
            let color = s.color();
            for a in 0..3 {
                c.min[a] = c.min[a].min(s.position[a]);
                c.max[a] = c.max[a].max(s.position[a]);
                c.min_scale[a] = c.min_scale[a].min(s.log_scale[a]);
                c.max_scale[a] = c.max_scale[a].max(s.log_scale[a]);
                c.min_color[a] = c.min_color[a].min(color[a]);
                c.max_color[a] = c.max_color[a].max(color[a]);
            }
        }
        for a in 0..3 {
            c.min_scale[a] = c.min_scale[a].clamp(-20.0, 20.0);
            c.max_scale[a] = c.max_scale[a].clamp(-20.0, 20.0);
        }
        chunks.push(c);
    }

    let mut out = BufWriter::with_capacity(1 << 20, writer);
    out.write_all(header.as_bytes())?;
    for c in &chunks {
        for v in c
            .min
            .iter()
            .chain(&c.max)
            .chain(&c.min_scale)
            .chain(&c.max_scale)
            .chain(&c.min_color)
            .chain(&c.max_color)
        {
            out.write_all(&v.to_le_bytes())?;
        }
    }
    for (i, s) in cloud.iter().enumerate() {
        let c = &chunks[i / CHUNK_SIZE];
        let color = s.color();
        let position = pack_111011(std::array::from_fn(|a| {
            normalize(s.position[a], c.min[a], c.max[a])
        }));
        let scale = pack_111011(std::array::from_fn(|a| {
            normalize(s.log_scale[a], c.min_scale[a], c.max_scale[a])
        }));
        let rgba = (pack_unorm(normalize(color[0], c.min_color[0], c.max_color[0]), 8) << 24)
            | (pack_unorm(normalize(color[1], c.min_color[1], c.max_color[1]), 8) << 16)
            | (pack_unorm(normalize(color[2], c.min_color[2], c.max_color[2]), 8) << 8)
            | pack_unorm(sigmoid(s.opacity_logit), 8);
        for word in [position, pack_rotation(s.rotation), scale, rgba] {
            out.write_all(&word.to_le_bytes())?;
        }
    }
    if k > 0 {
        let mut row = vec![0u8; 3 * k];
        for s in cloud.iter() {
            for c in 0..3 {
                for i in 0..k {
                    let v = s.sh_rest[i * 3 + c];
                    row[c * k + i] = ((v / 8.0 + 0.5) * 256.0).trunc().clamp(0.0, 255.0) as u8;
                }
            }
            out.write_all(&row)?;
        }
    }
    out.flush()?;
    Ok(())
}
