//! `.ksplat` files: sectioned splat buffers with three compression levels.
//!
//! * Level 0: 32-bit floats (44 bytes per splat without SH).
//! * Level 1: positions quantized to 16 bits relative to spatial buckets,
//!   half-float scale, rotation and SH (24 bytes per splat without SH).
//! * Level 2: like level 1 with SH quantized to 8 bits.
//!
//! Color and opacity are always 8-bit. The format holds SH up to degree 2;
//! the writer drops degree-3 coefficients.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use crate::cloud::{
    SplatCloud, SplatRef, color_to_sh_dc, logit, normalize_quaternion, sh_rest_coeffs,
};
use crate::error::{Error, Result};
use crate::formats::splat::unit_to_u8;
use crate::util::{Cursor, f16_to_f32, f32_to_f16, read_all};

const HEADER_SIZE: usize = 4096;
const SECTION_HEADER_SIZE: usize = 1024;
const SCALE_RANGE: u32 = 32767;
const DEFAULT_SH_RANGE: f32 = 1.5;
const MIN_SCALE: f32 = 1e-12;

fn err(msg: impl std::fmt::Display) -> Error {
    Error::Parse(format!("ksplat: {msg}"))
}

/// Options for [`write_ksplat_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KsplatOptions {
    /// 0 (floats), 1 (default; 16-bit positions and half floats) or 2
    /// (like 1 with 8-bit SH).
    pub compression_level: u8,
    /// Edge length of the cubic buckets positions are quantized against
    /// (levels 1 and 2). Default 5.0.
    pub block_size: f32,
    /// Maximum splats per full bucket. Default 256.
    pub bucket_size: u32,
}

impl Default for KsplatOptions {
    fn default() -> Self {
        Self {
            compression_level: 1,
            block_size: 5.0,
            bucket_size: 256,
        }
    }
}

/// Bytes per splat record.
fn bytes_per_splat(level: u8, sh_degree: u8) -> usize {
    let sh = sh_rest_coeffs(sh_degree) * 3;
    match level {
        0 => 44 + sh * 4,
        1 => 24 + sh * 2,
        _ => 24 + sh,
    }
}

/// Maps the file's per-degree channel-major SH layout to interleaved
/// memory order: file index -> memory index.
fn sh_file_order(sh_degree: u8) -> Vec<usize> {
    let mut map = Vec::new();
    for l in 1..=sh_degree as usize {
        let first = l * l - 1;
        let size = 2 * l + 1;
        for c in 0..3 {
            for k in 0..size {
                map.push((first + k) * 3 + c);
            }
        }
    }
    map
}

/// Reads a `.ksplat` file.
pub fn read_ksplat<R: Read>(reader: R) -> Result<SplatCloud> {
    let data = read_all(reader)?;
    let mut h = Cursor::new(&data, "ksplat");
    let (major, minor) = (h.u8()?, h.u8()?);
    if major == 0 && minor < 1 {
        return Err(Error::Unsupported(format!(
            "ksplat: version {major}.{minor} (files from before version 0.1)"
        )));
    }
    h.take(2)?;
    let section_count = h.u32()? as usize;
    h.u32()?;
    let total = h.u32()? as usize;
    h.u32()?;
    let level = h.u16()?;
    if level > 2 {
        return Err(err(format!("unknown compression level {level}")));
    }
    let level = level as u8;
    h.take(2)?;
    h.take(12)?;
    let mut sh_min = h.f32()?;
    let mut sh_max = h.f32()?;
    // Zero (or garbage) means "use the default range".
    if sh_min == 0.0 || !sh_min.is_finite() {
        sh_min = -DEFAULT_SH_RANGE;
    }
    if sh_max == 0.0 || !sh_max.is_finite() {
        sh_max = DEFAULT_SH_RANGE;
    }
    let sections_start = HEADER_SIZE
        .checked_add(
            section_count
                .checked_mul(SECTION_HEADER_SIZE)
                .ok_or_else(|| err("too many sections"))?,
        )
        .filter(|&s| s <= data.len())
        .ok_or_else(|| err("file is shorter than its section headers"))?;

    struct Section {
        count: usize,
        bucket_size: usize,
        bucket_count: usize,
        block_size: f32,
        bucket_bytes: usize,
        scale_range: u32,
        full_buckets: usize,
        partial_buckets: usize,
        sh_degree: u8,
    }
    let mut sections = Vec::with_capacity(section_count);
    for i in 0..section_count {
        let mut s = Cursor::at(&data, HEADER_SIZE + i * SECTION_HEADER_SIZE, "ksplat");
        s.u32()?;
        let count = s.u32()? as usize;
        let bucket_size = s.u32()? as usize;
        let bucket_count = s.u32()? as usize;
        let block_size = s.f32()?;
        let bucket_bytes = usize::from(s.u16()?);
        s.take(2)?;
        let scale_range = s.u32()?;
        s.u32()?;
        let full_buckets = s.u32()? as usize;
        let partial_buckets = s.u32()? as usize;
        let sh_degree = s.u16()?;
        if sh_degree > 3 {
            return Err(err(format!("section {i} has SH degree {sh_degree}")));
        }
        sections.push(Section {
            count,
            bucket_size,
            bucket_count,
            block_size,
            bucket_bytes,
            scale_range: if scale_range == 0 {
                if level == 0 { 1 } else { SCALE_RANGE }
            } else {
                scale_range
            },
            full_buckets,
            partial_buckets,
            sh_degree: sh_degree as u8,
        });
    }
    let degree = sections.iter().map(|s| s.sh_degree).max().unwrap_or(0);
    let mut cloud = SplatCloud::with_capacity(degree, total.min(data.len() / 24));
    let mut base = sections_start;
    for (index, sec) in sections.iter().enumerate() {
        let bucket_storage = sec
            .bucket_bytes
            .checked_mul(sec.bucket_count)
            .and_then(|v| v.checked_add(sec.partial_buckets.checked_mul(4)?))
            .ok_or_else(|| err(format!("section {index} is too large")))?;
        let bps = bytes_per_splat(level, sec.sh_degree);
        let size = sec
            .count
            .checked_mul(bps)
            .and_then(|v| v.checked_add(bucket_storage))
            .ok_or_else(|| err("section is too large"))?;
        let section = base
            .checked_add(size)
            .and_then(|end| data.get(base..end))
            .ok_or_else(|| err(format!("file ends inside section {index}")))?;
        base += size;

        // Bucket index for each splat in order (levels 1 and 2).
        let mut centers: Vec<[f32; 3]> = Vec::new();
        let mut bucket_of: Vec<usize> = Vec::new();
        if level > 0 {
            if sec
                .full_buckets
                .checked_add(sec.partial_buckets)
                .is_none_or(|b| b > sec.bucket_count)
            {
                return Err(err(format!(
                    "section {index} has inconsistent bucket counts"
                )));
            }
            let mut c = Cursor::new(section, "ksplat");
            let mut partial = Vec::with_capacity(sec.partial_buckets);
            for _ in 0..sec.partial_buckets {
                partial.push(c.u32()? as usize);
            }
            for _ in 0..sec.bucket_count {
                centers.push([c.f32()?, c.f32()?, c.f32()?]);
            }
            if sec.full_buckets.saturating_mul(sec.bucket_size) > sec.count
                || partial.iter().fold(0usize, |a, &l| a.saturating_add(l)) > sec.count
            {
                return Err(err(format!(
                    "section {index} buckets hold more splats than declared"
                )));
            }
            bucket_of.reserve(sec.count);
            for b in 0..sec.full_buckets {
                bucket_of.extend(std::iter::repeat_n(b, sec.bucket_size));
            }
            for (k, &len) in partial.iter().enumerate() {
                bucket_of.extend(std::iter::repeat_n(sec.full_buckets + k, len));
            }
            if bucket_of.len() < sec.count {
                return Err(err(format!(
                    "section {index} buckets hold fewer splats than declared"
                )));
            }
        }

        let order = sh_file_order(sec.sh_degree);
        let mut rest = vec![0.0f32; cloud.sh_rest_stride()];
        let half = sec.block_size / 2.0;
        let range = sec.scale_range as f32;
        let records = &section[bucket_storage..];
        for i in 0..sec.count {
            let r = &records[i * bps..(i + 1) * bps];
            let mut c = Cursor::new(r, "ksplat");
            let (position, scale, rotation) = if level == 0 {
                (
                    [c.f32()?, c.f32()?, c.f32()?],
                    [c.f32()?, c.f32()?, c.f32()?],
                    [c.f32()?, c.f32()?, c.f32()?, c.f32()?],
                )
            } else {
                let center = centers[bucket_of[i]];
                let q = [c.u16()?, c.u16()?, c.u16()?];
                (
                    std::array::from_fn(|a| (f32::from(q[a]) - range) * (half / range) + center[a]),
                    [0; 3].map(|_| c.u16().map(f16_to_f32).unwrap_or(0.0)),
                    [0; 4].map(|_| c.u16().map(f16_to_f32).unwrap_or(0.0)),
                )
            };
            let rgba = c.array::<4>()?;
            rest.fill(0.0);
            for &dst in &order {
                rest[dst] = match level {
                    0 => c.f32()?,
                    1 => f16_to_f32(c.u16()?),
                    _ => f32::from(c.u8()?) / 255.0 * (sh_max - sh_min) + sh_min,
                };
            }
            cloud.push(SplatRef {
                position,
                log_scale: scale.map(|v| v.max(MIN_SCALE).ln()),
                rotation,
                opacity_logit: logit(f32::from(rgba[3]) / 255.0),
                sh_dc: [0, 1, 2].map(|k| color_to_sh_dc(f32::from(rgba[k]) / 255.0)),
                sh_rest: &rest,
            });
        }
    }
    Ok(cloud)
}

/// Writes `cloud` as `.ksplat` at compression level 1.
pub fn write_ksplat<W: Write>(cloud: &SplatCloud, writer: W) -> Result<()> {
    write_ksplat_with(cloud, writer, KsplatOptions::default())
}

/// Writes `cloud` as `.ksplat` with explicit options. Uses a single
/// section; SH above degree 2 is dropped.
pub fn write_ksplat_with<W: Write>(
    cloud: &SplatCloud,
    mut writer: W,
    options: KsplatOptions,
) -> Result<()> {
    cloud.validate()?;
    let level = options.compression_level;
    if level > 2 {
        return Err(Error::Unsupported(format!(
            "ksplat: compression level {level}"
        )));
    }
    if level > 0 && !(options.block_size > 0.0 && options.block_size.is_finite()) {
        return Err(Error::Invalid("ksplat block size must be positive".into()));
    }
    if options.bucket_size == 0 {
        return Err(Error::Invalid("ksplat bucket size must be positive".into()));
    }
    let reduced;
    let cloud = if cloud.sh_degree > 2 {
        let mut c = cloud.clone();
        c.set_sh_degree(2);
        reduced = c;
        &reduced
    } else {
        cloud
    };
    let n = cloud.len();
    let count =
        u32::try_from(n).map_err(|_| Error::Invalid("too many splats for ksplat".into()))?;
    let degree = cloud.sh_degree;
    let bps = bytes_per_splat(level, degree);

    // SH range, kept away from exactly 0 (which readers treat as "default").
    let (mut sh_min, mut sh_max) = cloud
        .sh_rest
        .iter()
        .filter(|v| v.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if !sh_min.is_finite() {
        (sh_min, sh_max) = (-DEFAULT_SH_RANGE, DEFAULT_SH_RANGE);
    }
    if sh_max - sh_min < 1e-6 {
        sh_max = sh_min + 1e-3;
    }
    if sh_min == 0.0 {
        sh_min = -1e-7;
    }
    if sh_max == 0.0 {
        sh_max = 1e-7;
    }

    // Buckets (levels 1 and 2): cubic cells keyed by integer coordinates.
    let mut full: Vec<([f32; 3], Vec<usize>)> = Vec::new();
    let mut partial: Vec<([f32; 3], Vec<usize>)> = Vec::new();
    if level > 0 {
        let bs = options.block_size;
        let (min, _) = cloud.bounds().unwrap_or(([0.0; 3], [0.0; 3]));
        let min = min.map(|v| if v.is_finite() { v } else { 0.0 });
        let mut cells: BTreeMap<[i64; 3], Vec<usize>> = BTreeMap::new();
        for (i, p) in cloud.positions.iter().enumerate() {
            let key = std::array::from_fn(|a| ((p[a] - min[a]) / bs).floor() as i64);
            cells.entry(key).or_default().push(i);
        }
        let size = options.bucket_size as usize;
        for (key, members) in cells {
            let center = std::array::from_fn(|a| key[a] as f32 * bs + min[a] + bs / 2.0);
            let mut chunks = members.chunks(size).peekable();
            while let Some(chunk) = chunks.next() {
                if chunk.len() == size {
                    full.push((center, chunk.to_vec()));
                } else {
                    debug_assert!(chunks.peek().is_none());
                    partial.push((center, chunk.to_vec()));
                }
            }
        }
    }
    let bucket_count = full.len() + partial.len();
    let bucket_storage = if level > 0 {
        12 * bucket_count + 4 * partial.len()
    } else {
        0
    };
    let storage = n * bps + bucket_storage;

    let mut out = Vec::with_capacity(HEADER_SIZE + SECTION_HEADER_SIZE + storage);
    out.resize(HEADER_SIZE + SECTION_HEADER_SIZE, 0);
    out[0] = 0;
    out[1] = 1;
    out[4..8].copy_from_slice(&1u32.to_le_bytes());
    out[8..12].copy_from_slice(&1u32.to_le_bytes());
    out[12..16].copy_from_slice(&count.to_le_bytes());
    out[16..20].copy_from_slice(&count.to_le_bytes());
    out[20..22].copy_from_slice(&u16::from(level).to_le_bytes());
    out[36..40].copy_from_slice(&sh_min.to_le_bytes());
    out[40..44].copy_from_slice(&sh_max.to_le_bytes());
    let s = HEADER_SIZE;
    out[s..s + 4].copy_from_slice(&count.to_le_bytes());
    out[s + 4..s + 8].copy_from_slice(&count.to_le_bytes());
    if level > 0 {
        out[s + 8..s + 12].copy_from_slice(&options.bucket_size.to_le_bytes());
        out[s + 12..s + 16].copy_from_slice(&(bucket_count as u32).to_le_bytes());
        out[s + 16..s + 20].copy_from_slice(&options.block_size.to_le_bytes());
        out[s + 20..s + 22].copy_from_slice(&12u16.to_le_bytes());
        out[s + 24..s + 28].copy_from_slice(&SCALE_RANGE.to_le_bytes());
        out[s + 32..s + 36].copy_from_slice(&(full.len() as u32).to_le_bytes());
        out[s + 36..s + 40].copy_from_slice(&(partial.len() as u32).to_le_bytes());
    }
    out[s + 28..s + 32].copy_from_slice(&(storage as u32).to_le_bytes());
    out[s + 40..s + 42].copy_from_slice(&u16::from(degree).to_le_bytes());

    let order = sh_file_order(degree);
    let record = |out: &mut Vec<u8>, i: usize, center: [f32; 3]| {
        let sp = cloud.get(i);
        let [w, x, y, z] = normalize_quaternion(sp.rotation);
        let scale = sp.scale();
        if level == 0 {
            for v in sp.position.iter().chain(&scale).chain(&[w, x, y, z]) {
                out.extend_from_slice(&v.to_le_bytes());
            }
        } else {
            let half = options.block_size / 2.0;
            let range = SCALE_RANGE as f32;
            for a in 0..3 {
                let q = ((sp.position[a] - center[a]) * range / half).round() + range;
                out.extend_from_slice(&(q.clamp(0.0, 65535.0) as u16).to_le_bytes());
            }
            for v in scale.iter().chain(&[w, x, y, z]) {
                out.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
            }
        }
        for c in sp.color() {
            out.push(unit_to_u8(c));
        }
        out.push(unit_to_u8(sp.opacity()));
        for &src in &order {
            let v = sp.sh_rest[src];
            match level {
                0 => out.extend_from_slice(&v.to_le_bytes()),
                1 => out.extend_from_slice(&f32_to_f16(v).to_le_bytes()),
                _ => {
                    let t = (v.clamp(sh_min, sh_max) - sh_min) / (sh_max - sh_min);
                    out.push((t * 255.0).round().clamp(0.0, 255.0) as u8);
                }
            }
        }
    };
    if level == 0 {
        for i in 0..n {
            record(&mut out, i, [0.0; 3]);
        }
    } else {
        for (_, members) in &partial {
            out.extend_from_slice(&(members.len() as u32).to_le_bytes());
        }
        for (center, _) in full.iter().chain(&partial) {
            for v in center {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for (center, members) in full.iter().chain(&partial) {
            for &i in members {
                record(&mut out, i, *center);
            }
        }
    }
    debug_assert_eq!(out.len(), HEADER_SIZE + SECTION_HEADER_SIZE + storage);
    writer.write_all(&out)?;
    writer.flush()?;
    Ok(())
}
