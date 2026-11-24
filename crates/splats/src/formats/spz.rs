//! `.spz` compressed splat files.
//!
//! Reads format versions 1 to 4: versions 1–3 are a single gzip stream,
//! version 4 is a plain header plus one zstd stream per attribute. Writes
//! versions 2, 3 (the default, most widely readable) and 4.
//!
//! Values are quantized: positions to 24-bit fixed point, color, opacity
//! and scale to 8 bits, rotations to 24 or 32 bits, and SH coefficients to
//! 5 bits (degree 1) or 4 bits (higher degrees).
//!
//! **Coordinates.** SPZ stores right-up-back axes. By default the reader and
//! writer convert to and from the splat training frame (right-down-forward)
//! by flipping y and z, which is exact. A coordinate-system record in the
//! file, when present, is honored.

use std::io::{Read, Write};

use crate::cloud::{SplatCloud, SplatRef, logit, normalize_quaternion, sh_rest_coeffs};
use crate::error::{Error, Result};
use crate::math::{Mat3, mat_to_quat};
use crate::transform;
use crate::util::{Cursor, f16_to_f32, gunzip, gzip, read_all, unzstd, zstd};

const MAGIC: u32 = 0x5053_474E;
const COLOR_SCALE: f32 = 0.15;
const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;
const FLAG_ANTIALIASED: u8 = 1;
const FLAG_EXTENSIONS: u8 = 2;
const EXT_COORDINATE_SYSTEM: u32 = 0xADBE_0003;
/// Right-up-back storage <-> right-down-forward training frame.
const STORAGE_FLIP: [bool; 3] = [false, true, true];

fn err(msg: impl std::fmt::Display) -> Error {
    Error::Parse(format!("spz: {msg}"))
}

/// Options for [`read_spz_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpzReadOptions {
    /// Convert from the stored axes to the splat training frame. Default
    /// `true`.
    pub convert_coordinates: bool,
}

impl Default for SpzReadOptions {
    fn default() -> Self {
        Self {
            convert_coordinates: true,
        }
    }
}

/// Options for [`write_spz_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpzWriteOptions {
    /// Format version: 2, 3 (default) or 4.
    pub version: u32,
    /// Convert from the splat training frame to SPZ's stored axes. Default
    /// `true`.
    pub convert_coordinates: bool,
    /// Mark the scene as trained with antialiasing even if
    /// [`SplatCloud::antialiased`] is not set. Default `false`.
    pub antialiased: bool,
    /// gzip effort for versions 2 and 3, 1 (fastest) to 9 (smallest).
    /// Default 6.
    pub compression_level: u8,
}

impl Default for SpzWriteOptions {
    fn default() -> Self {
        Self {
            version: 3,
            convert_coordinates: true,
            antialiased: false,
            compression_level: 6,
        }
    }
}

/// Reads an `.spz` file.
pub fn read_spz<R: Read>(reader: R) -> Result<SplatCloud> {
    read_spz_with(reader, SpzReadOptions::default())
}

/// Reads an `.spz` file with explicit options.
pub fn read_spz_with<R: Read>(reader: R, options: SpzReadOptions) -> Result<SplatCloud> {
    let data = read_all(reader)?;
    let (mut cloud, axes) = read_stored(&data)?;
    if options.convert_coordinates {
        to_training_frame(&mut cloud, axes);
    }
    Ok(cloud)
}

/// Decodes an SPZ blob without converting coordinates; also returns the
/// axes the data is stored in.
pub(crate) fn read_stored(data: &[u8]) -> Result<(SplatCloud, Axes)> {
    if data.starts_with(&[0x1f, 0x8b]) {
        read_legacy(&gunzip(data, "spz")?)
    } else if data.starts_with(&MAGIC.to_le_bytes()) {
        read_v4(data)
    } else {
        Err(err("not an SPZ file"))
    }
}

/// Where each stored axis points: x, y and z are each one of `R`/`L`,
/// `U`/`D`, `F`/`B` (right/left, up/down, forward/back).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Axes([u8; 3]);

/// The default storage convention.
const RUB: Axes = Axes(*b"RUB");

impl Axes {
    /// The axes for a coordinate-system record value (1 to 16).
    fn from_record(system: u32) -> Option<Self> {
        const SYSTEMS: [&[u8; 3]; 16] = [
            b"LDB", b"RDB", b"LUB", b"RUB", b"LDF", b"RDF", b"LUF", b"RUF", b"LFD", b"RFD", b"LFU",
            b"RFU", b"LBD", b"RBD", b"LBU", b"RBU",
        ];
        let index = (system as usize).checked_sub(1)?;
        SYSTEMS.get(index).map(|a| Axes(**a))
    }
}

/// Converts data stored in `from` axes to the training frame
/// (right-down-forward). Axis-aligned flips are exact; the rotated systems
/// also turn positions, orientations and SH color.
pub(crate) fn to_training_frame(cloud: &mut SplatCloud, from: Axes) {
    let direction = |letter: u8| -> [f64; 3] {
        match letter {
            b'R' => [1.0, 0.0, 0.0],
            b'L' => [-1.0, 0.0, 0.0],
            b'D' => [0.0, 1.0, 0.0],
            b'U' => [0.0, -1.0, 0.0],
            b'F' => [0.0, 0.0, 1.0],
            _ => [0.0, 0.0, -1.0],
        }
    };
    let columns = from.0.map(direction);
    if (0..3).all(|i| columns[i][i] != 0.0) {
        transform::flip_axes(cloud, std::array::from_fn(|i| columns[i][i] < 0.0));
        return;
    }
    // A signed permutation: p_training = M · p_stored, with M's columns
    // being the stored axes. Mirror x first if M is not a pure rotation.
    let mut m: Mat3 = std::array::from_fn(|r| std::array::from_fn(|c| columns[c][r]));
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det < 0.0 {
        transform::flip_axes(cloud, [true, false, false]);
        for row in &mut m {
            row[0] = -row[0];
        }
    }
    transform::rotate(cloud, mat_to_quat(&m));
}

struct Header {
    version: u32,
    count: usize,
    /// SH degree stored in the file (up to 4).
    sh_degree: u8,
    fractional_bits: u8,
    flags: u8,
}

fn check_header(h: &Header) -> Result<()> {
    if h.sh_degree > 4 {
        return Err(err(format!("invalid SH degree {}", h.sh_degree)));
    }
    if h.fractional_bits > 24 {
        return Err(err(format!(
            "invalid fractional bits {}",
            h.fractional_bits
        )));
    }
    Ok(())
}

/// Byte sizes of the six attribute blocks.
fn block_sizes(h: &Header) -> [usize; 6] {
    let n = h.count;
    [
        n * if h.version == 1 { 6 } else { 9 },
        n,
        n * 3,
        n * 3,
        n * if h.version >= 3 { 4 } else { 3 },
        n * sh_rest_coeffs(h.sh_degree) * 3,
    ]
}

fn read_legacy(raw: &[u8]) -> Result<(SplatCloud, Axes)> {
    let mut c = Cursor::new(raw, "spz");
    if c.u32()? != MAGIC {
        return Err(err("bad magic number"));
    }
    let h = Header {
        version: c.u32()?,
        count: c.u32()? as usize,
        sh_degree: c.u8()?,
        fractional_bits: c.u8()?,
        flags: c.u8()?,
    };
    c.u8()?;
    if !(1..=4).contains(&h.version) {
        return Err(Error::Unsupported(format!(
            "spz: version {} in a gzip container",
            h.version
        )));
    }
    check_header(&h)?;
    if h.count > raw.len() / 9 {
        return Err(err("file is shorter than its point count requires"));
    }
    let sizes = block_sizes(&h);
    if sizes.iter().sum::<usize>() > raw.len() - c.pos() {
        return Err(err("file is shorter than its point count requires"));
    }
    let blocks = sizes.map(|n| c.take(n).expect("length checked"));
    let mut axes = RUB;
    if h.flags & FLAG_EXTENSIONS != 0 {
        axes = read_extensions(&raw[c.pos()..]).unwrap_or(axes);
    }
    Ok((decode_blocks(&h, blocks)?, axes))
}

fn read_v4(data: &[u8]) -> Result<(SplatCloud, Axes)> {
    let mut c = Cursor::new(data, "spz");
    c.u32()?;
    let h = Header {
        version: c.u32()?,
        count: c.u32()? as usize,
        sh_degree: c.u8()?,
        fractional_bits: c.u8()?,
        flags: c.u8()?,
    };
    if h.version != 4 {
        return Err(Error::Unsupported(format!(
            "spz: version {} in a zstd container",
            h.version
        )));
    }
    check_header(&h)?;
    if h.count / 1024 > data.len() / 9 {
        return Err(err("point count is implausibly large for the file size"));
    }
    let stream_count = usize::from(c.u8()?);
    let toc_offset = c.u32()? as usize;
    if toc_offset < 32 || toc_offset > data.len() {
        return Err(err("invalid table of contents offset"));
    }
    let sizes = block_sizes(&h);
    let expected_streams = if h.sh_degree == 0 { 5 } else { 6 };
    if stream_count != expected_streams {
        return Err(err(format!(
            "expected {expected_streams} streams, found {stream_count}"
        )));
    }
    let mut axes = RUB;
    if h.flags & FLAG_EXTENSIONS != 0 && toc_offset > 32 {
        axes = read_extensions(&data[32..toc_offset]).unwrap_or(axes);
    }
    let mut toc = Cursor::at(data, toc_offset, "spz");
    let mut entries = Vec::with_capacity(stream_count);
    for _ in 0..stream_count {
        entries.push((toc.u64()? as usize, toc.u64()? as usize));
    }
    let mut streams: Vec<Vec<u8>> = Vec::with_capacity(6);
    let mut pos = toc.pos();
    for (i, &(compressed, uncompressed)) in entries.iter().enumerate() {
        if uncompressed != sizes[i] {
            return Err(err(format!(
                "stream {i} holds {uncompressed} bytes; {} points need {}",
                h.count, sizes[i]
            )));
        }
        let end = pos
            .checked_add(compressed)
            .filter(|&e| e <= data.len())
            .ok_or_else(|| err(format!("stream {i} runs past the end of the file")))?;
        let out = unzstd(&data[pos..end], uncompressed, "spz")?;
        if out.len() != uncompressed {
            return Err(err(format!("stream {i} decompressed to the wrong size")));
        }
        streams.push(out);
        pos = end;
    }
    streams.resize(6, Vec::new());
    let blocks: [&[u8]; 6] = std::array::from_fn(|i| streams[i].as_slice());
    Ok((decode_blocks(&h, blocks)?, axes))
}

/// Returns the stored axes from a coordinate-system record, if there is a
/// valid one. Unknown or truncated records are skipped, as readers do.
fn read_extensions(mut records: &[u8]) -> Option<Axes> {
    while records.len() >= 8 {
        let kind = u32::from_le_bytes(records[..4].try_into().expect("4 bytes"));
        let len = u32::from_le_bytes(records[4..8].try_into().expect("4 bytes")) as usize;
        let payload = records.get(8..8usize.checked_add(len)?)?;
        if kind == EXT_COORDINATE_SYSTEM && len >= 4 {
            let system = u32::from_le_bytes(payload[..4].try_into().expect("4 bytes"));
            return Axes::from_record(system);
        }
        records = &records[8 + len..];
    }
    None
}

fn decode_blocks(h: &Header, blocks: [&[u8]; 6]) -> Result<SplatCloud> {
    let [positions, alphas, colors, scales, rotations, sh] = blocks;
    let n = h.count;
    // Degree-4 files keep their first three bands.
    let degree = h.sh_degree.min(3);
    let file_stride = sh_rest_coeffs(h.sh_degree) * 3;
    let stride = sh_rest_coeffs(degree) * 3;
    let mut cloud = SplatCloud::with_capacity(degree, n);
    cloud.antialiased = h.flags & FLAG_ANTIALIASED != 0;
    let scale = 1.0 / (1u32 << h.fractional_bits) as f32;
    let mut rest = vec![0.0f32; stride];
    for i in 0..n {
        let position = if h.version == 1 {
            let p = &positions[i * 6..i * 6 + 6];
            [0, 1, 2].map(|a| f16_to_f32(u16::from_le_bytes([p[a * 2], p[a * 2 + 1]])))
        } else {
            let p = &positions[i * 9..i * 9 + 9];
            [0, 1, 2].map(|a| {
                let b = &p[a * 3..a * 3 + 3];
                let raw = i32::from(b[0]) | (i32::from(b[1]) << 8) | (i32::from(b[2]) << 16);
                ((raw << 8) >> 8) as f32 * scale
            })
        };
        let xyzw = if h.version >= 3 {
            decode_smallest_three(u32::from_le_bytes(
                rotations[i * 4..i * 4 + 4].try_into().expect("4 bytes"),
            ))
        } else {
            let r = &rotations[i * 3..i * 3 + 3];
            let xyz = [0, 1, 2].map(|a| f32::from(r[a]) / 127.5 - 1.0);
            let w = (1.0 - xyz.iter().map(|v| v * v).sum::<f32>())
                .max(0.0)
                .sqrt();
            [xyz[0], xyz[1], xyz[2], w]
        };
        for (v, &b) in rest
            .iter_mut()
            .zip(&sh[i * file_stride..i * file_stride + stride])
        {
            *v = (f32::from(b) - 128.0) / 128.0;
        }
        cloud.push(SplatRef {
            position,
            log_scale: [0, 1, 2].map(|a| f32::from(scales[i * 3 + a]) / 16.0 - 10.0),
            rotation: [xyzw[3], xyzw[0], xyzw[1], xyzw[2]],
            opacity_logit: logit(f32::from(alphas[i]) / 255.0),
            sh_dc: [0, 1, 2].map(|c| (f32::from(colors[i * 3 + c]) / 255.0 - 0.5) / COLOR_SCALE),
            sh_rest: &rest,
        });
    }
    Ok(cloud)
}

/// Decodes the 32-bit smallest-three quaternion; returns `[x, y, z, w]`.
fn decode_smallest_three(mut comp: u32) -> [f32; 4] {
    let largest = (comp >> 30) as usize;
    let mut q = [0.0f32; 4];
    let mut sum = 0.0;
    for i in (0..4).rev() {
        if i == largest {
            continue;
        }
        let mag = (comp & 0x1ff) as f32;
        let negative = (comp >> 9) & 1 == 1;
        comp >>= 10;
        let v = SQRT_HALF * mag / 511.0;
        q[i] = if negative { -v } else { v };
        sum += v * v;
    }
    q[largest] = (1.0 - sum).max(0.0).sqrt();
    q
}

/// Encodes a normalized `[x, y, z, w]` quaternion as smallest-three.
fn encode_smallest_three(q: [f32; 4]) -> u32 {
    let mut largest = 0;
    for i in 1..4 {
        if q[i].abs() > q[largest].abs() {
            largest = i;
        }
    }
    let negate = q[largest] < 0.0;
    let mut comp = largest as u32;
    for (i, &v) in q.iter().enumerate() {
        if i == largest {
            continue;
        }
        let sign = u32::from((v < 0.0) != negate);
        let mag = ((511.0 * v.abs() / SQRT_HALF + 0.5) as u32).min(511);
        comp = (comp << 10) | (sign << 9) | mag;
    }
    comp
}

fn to_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

fn quantize_sh(v: f32, bits: u32) -> u8 {
    let q = (v * 128.0).round() as i32 + 128;
    let bucket = 1i32 << (8 - bits);
    let q = (q + bucket / 2) / bucket * bucket;
    q.clamp(0, 255) as u8
}

/// Writes `cloud` as `.spz` (version 3).
pub fn write_spz<W: Write>(cloud: &SplatCloud, writer: W) -> Result<()> {
    write_spz_with(cloud, writer, SpzWriteOptions::default())
}

/// Writes `cloud` as `.spz` with explicit options.
pub fn write_spz_with<W: Write>(
    cloud: &SplatCloud,
    mut writer: W,
    options: SpzWriteOptions,
) -> Result<()> {
    cloud.validate()?;
    if !(2..=4).contains(&options.version) {
        return Err(Error::Unsupported(format!(
            "spz: writing version {} (use 2, 3 or 4)",
            options.version
        )));
    }
    if cloud.is_empty() {
        return Err(Error::Invalid(
            "SPZ cannot store an empty splat set (readers reject a zero point count)".into(),
        ));
    }
    let count =
        u32::try_from(cloud.len()).map_err(|_| Error::Invalid("too many splats for SPZ".into()))?;
    let converted;
    let cloud = if options.convert_coordinates {
        let mut c = cloud.clone();
        transform::flip_axes(&mut c, STORAGE_FLIP);
        converted = c;
        &converted
    } else {
        cloud
    };

    // Use as many fractional bits as the scene's extent allows (up to 12).
    let extent = cloud
        .positions
        .iter()
        .flatten()
        .filter(|v| v.is_finite())
        .fold(0.0f32, |m, v| m.max(v.abs()));
    let mut fractional_bits = 12u8;
    while fractional_bits > 0 && extent * (1u32 << fractional_bits) as f32 >= 8_388_607.0 {
        fractional_bits -= 1;
    }
    if extent >= 8_388_607.0 {
        return Err(Error::Invalid(
            "positions are too far from the origin for SPZ".into(),
        ));
    }
    let h = Header {
        version: options.version,
        count: cloud.len(),
        sh_degree: cloud.sh_degree,
        fractional_bits,
        flags: if options.antialiased || cloud.antialiased {
            FLAG_ANTIALIASED
        } else {
            0
        },
    };
    let blocks = encode_blocks(cloud, &h);

    if options.version < 4 {
        let total: usize = 16 + blocks.iter().map(Vec::len).sum::<usize>();
        let mut raw = Vec::with_capacity(total);
        raw.extend_from_slice(&MAGIC.to_le_bytes());
        raw.extend_from_slice(&options.version.to_le_bytes());
        raw.extend_from_slice(&count.to_le_bytes());
        raw.extend_from_slice(&[h.sh_degree, h.fractional_bits, h.flags, 0]);
        for block in &blocks {
            raw.extend_from_slice(block);
        }
        writer.write_all(&gzip(&raw, options.compression_level))?;
    } else {
        let streams: Vec<Vec<u8>> = blocks
            .iter()
            .take(if h.sh_degree == 0 { 5 } else { 6 })
            .map(|b| zstd(b))
            .collect();
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&[h.sh_degree, h.fractional_bits, h.flags, streams.len() as u8]);
        out.extend_from_slice(&32u32.to_le_bytes());
        out.extend_from_slice(&[0; 12]);
        for (stream, block) in streams.iter().zip(&blocks) {
            out.extend_from_slice(&(stream.len() as u64).to_le_bytes());
            out.extend_from_slice(&(block.len() as u64).to_le_bytes());
        }
        for stream in &streams {
            out.extend_from_slice(stream);
        }
        writer.write_all(&out)?;
    }
    writer.flush()?;
    Ok(())
}

fn encode_blocks(cloud: &SplatCloud, h: &Header) -> [Vec<u8>; 6] {
    let n = cloud.len();
    let mut blocks = block_sizes(h).map(Vec::with_capacity);
    let scale = (1u32 << h.fractional_bits) as f32;
    for s in cloud.iter() {
        for v in s.position {
            let i = (v * scale).round() as i32;
            blocks[0].extend_from_slice(&i.to_le_bytes()[..3]);
        }
        blocks[1].push(to_u8(s.opacity() * 255.0));
        for c in s.sh_dc {
            blocks[2].push(to_u8(c * (COLOR_SCALE * 255.0) + 0.5 * 255.0));
        }
        for v in s.log_scale {
            blocks[3].push(to_u8((v + 10.0) * 16.0));
        }
        let [w, x, y, z] = normalize_quaternion(s.rotation);
        if h.version >= 3 {
            blocks[4].extend_from_slice(&encode_smallest_three([x, y, z, w]).to_le_bytes());
        } else {
            let sign = if w < 0.0 { -127.5 } else { 127.5 };
            for v in [x, y, z] {
                blocks[4].push(to_u8(v * sign + 127.5));
            }
        }
        for (j, &v) in s.sh_rest.iter().enumerate() {
            blocks[5].push(quantize_sh(v, if j < 9 { 5 } else { 4 }));
        }
    }
    debug_assert_eq!(blocks[1].len(), n);
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smallest_three_roundtrip() {
        for q in [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, 0.5, -0.5],
            [0.1, -0.7, 0.2, 0.68],
        ] {
            let len = q.iter().map(|v: &f32| v * v).sum::<f32>().sqrt();
            let q = q.map(|v| v / len);
            let back = decode_smallest_three(encode_smallest_three(q));
            let dot: f32 = q.iter().zip(&back).map(|(a, b)| a * b).sum();
            assert!(dot.abs() > 0.9999, "{q:?} -> {back:?}");
        }
    }

    #[test]
    fn sh_quantization_buckets() {
        assert_eq!(quantize_sh(0.0, 4), 128);
        assert_eq!(quantize_sh(0.0, 5), 128);
        assert_eq!(quantize_sh(10.0, 4), 255);
        assert_eq!(quantize_sh(-10.0, 4), 0);
        assert_eq!(quantize_sh(0.07, 8), 137);
        assert_eq!(quantize_sh(0.07, 4) % 16, 0);
    }

    #[test]
    fn coordinate_system_records() {
        let record = |system: u32| {
            let mut r = EXT_COORDINATE_SYSTEM.to_le_bytes().to_vec();
            r.extend(4u32.to_le_bytes());
            r.extend(system.to_le_bytes());
            r
        };
        assert_eq!(read_extensions(&record(6)), Some(Axes(*b"RDF")));
        assert_eq!(read_extensions(&record(4)), Some(RUB));
        assert_eq!(read_extensions(&record(7)), Some(Axes(*b"LUF")));
        assert_eq!(read_extensions(&record(12)), Some(Axes(*b"RFU")));
        assert_eq!(read_extensions(&record(0)), None);
        assert_eq!(read_extensions(&record(99)), None);
        // A truncated record is skipped rather than rejected.
        assert_eq!(read_extensions(&record(6)[..10]), None);
    }

    fn one_splat(position: [f32; 3]) -> SplatCloud {
        let mut cloud = SplatCloud::new(1);
        cloud.push(SplatRef {
            position,
            log_scale: [0.0, -1.0, -2.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
            opacity_logit: 0.0,
            sh_dc: [0.0; 3],
            sh_rest: &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        });
        cloud
    }

    #[test]
    fn axis_aligned_frames_flip() {
        let mut cloud = one_splat([1.0, 2.0, 3.0]);
        to_training_frame(&mut cloud, Axes(*b"LUF"));
        assert_eq!(cloud.positions[0], [-1.0, -2.0, 3.0]);
    }

    #[test]
    fn rotated_frames_permute_axes() {
        // Stored x right, y forward, z up: training (x, y down, z forward)
        // = (x, -z, y).
        let mut cloud = one_splat([1.0, 2.0, 3.0]);
        to_training_frame(&mut cloud, Axes(*b"RFU"));
        let p = cloud.positions[0];
        for (a, b) in p.iter().zip([1.0, -3.0, 2.0]) {
            assert!((a - b).abs() < 1e-5, "{p:?}");
        }
        // A left-handed rotated system needs a mirror as well.
        let mut cloud = one_splat([1.0, 2.0, 3.0]);
        to_training_frame(&mut cloud, Axes(*b"LFU"));
        let p = cloud.positions[0];
        for (a, b) in p.iter().zip([-1.0, -3.0, 2.0]) {
            assert!((a - b).abs() < 1e-5, "{p:?}");
        }
    }
}
