//! The compact `.splat` format: 32 bytes per splat, no header.
//!
//! Each record holds position (3 × `f32`), linear scale (3 × `f32`), color
//! RGBA (4 × `u8`, alpha is opacity) and rotation (4 × `u8`, `[w, x, y, z]`
//! mapped from `-1..=1` to `0..=255`). Floats are little-endian.
//!
//! The format stores base color only, so higher-order SH is dropped on write
//! and clouds read back have SH degree 0. Color and opacity are quantized to
//! 8 bits and rotations to about 1/128.

use std::io::{self, Read, Write};

use crate::cloud::{SplatCloud, SplatRef, color_to_sh_dc, logit, normalize_quaternion};
use crate::error::{Error, Result};

/// Bytes per splat.
pub const RECORD_SIZE: usize = 32;

/// Smallest scale stored, so zero scales still have a finite log.
const MIN_SCALE: f32 = 1e-12;

fn f32_at(bytes: &[u8], index: usize) -> f32 {
    f32::from_le_bytes(
        bytes[index * 4..index * 4 + 4]
            .try_into()
            .expect("in bounds"),
    )
}

pub(crate) fn read_full<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

fn decode(record: &[u8]) -> SplatRef<'static> {
    let rgba = &record[24..28];
    let rot = &record[28..32];
    SplatRef {
        position: [f32_at(record, 0), f32_at(record, 1), f32_at(record, 2)],
        log_scale: [3, 4, 5].map(|i| f32_at(record, i).max(MIN_SCALE).ln()),
        rotation: normalize_quaternion([0, 1, 2, 3].map(|i| (f32::from(rot[i]) - 128.0) / 128.0)),
        opacity_logit: logit(f32::from(rgba[3]) / 255.0),
        sh_dc: [0, 1, 2].map(|c| color_to_sh_dc(f32::from(rgba[c]) / 255.0)),
        sh_rest: &[],
    }
}

/// Reads a `.splat` file.
pub fn read_splat<R: Read>(mut reader: R) -> Result<SplatCloud> {
    let mut cloud = SplatCloud::new(0);
    let mut buf = vec![0u8; RECORD_SIZE * 32 * 1024];
    loop {
        let n = read_full(&mut reader, &mut buf)?;
        if n % RECORD_SIZE != 0 {
            return Err(Error::Parse(format!(
                "splat: file size is not a multiple of {RECORD_SIZE} bytes"
            )));
        }
        for record in buf[..n].chunks_exact(RECORD_SIZE) {
            cloud.push(decode(record));
        }
        if n < buf.len() {
            return Ok(cloud);
        }
    }
}

pub(crate) fn unit_to_u8(v: f32) -> u8 {
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Writes `cloud` as a `.splat` file. Splat order is preserved; call
/// [`sort_by_importance`](crate::transform::sort_by_importance) first if the
/// consumer expects the most significant splats first.
pub fn write_splat<W: Write>(cloud: &SplatCloud, writer: W) -> Result<()> {
    cloud.validate()?;
    let mut out = io::BufWriter::with_capacity(1 << 20, writer);
    let mut record = [0u8; RECORD_SIZE];
    for s in cloud.iter() {
        let scale = s.scale();
        for (i, v) in s.position.iter().chain(&scale).enumerate() {
            record[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (c, v) in s.color().into_iter().enumerate() {
            record[24 + c] = unit_to_u8(v);
        }
        record[27] = unit_to_u8(s.opacity());
        for (i, v) in normalize_quaternion(s.rotation).into_iter().enumerate() {
            record[28 + i] = (v * 128.0 + 128.0).round().clamp(0.0, 255.0) as u8;
        }
        out.write_all(&record)?;
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rotation_bytes_decode_to_identity() {
        let mut record = [0u8; RECORD_SIZE];
        record[28..32].fill(128);
        assert_eq!(decode(&record).rotation, [1.0, 0.0, 0.0, 0.0]);
    }
}
