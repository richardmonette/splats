//! Byte-level helpers shared by the format modules: a bounds-checked
//! little-endian cursor, half floats, CRC-32, gzip, zstd and base64.

use std::io::Read;

use crate::error::{Error, Result};

/// Little-endian reader over a byte slice that reports truncation as a
/// parse error naming the format.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    what: &'static str,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8], what: &'static str) -> Self {
        Self { data, pos: 0, what }
    }

    pub(crate) fn at(data: &'a [u8], pos: usize, what: &'static str) -> Self {
        Self { data, pos, what }
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.data.len())
            .ok_or_else(|| Error::Parse(format!("{}: unexpected end of data", self.what)))?;
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().expect("length checked"))
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    pub(crate) fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.array()?))
    }
}

/// Reads all of `reader` into memory.
pub(crate) fn read_all<R: Read>(mut reader: R) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    Ok(data)
}

/// Converts an IEEE binary16 value to `f32`.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let mant = f32::from(h & 0x3ff);
    sign * match exp {
        0 => mant * 2f32.powi(-24),
        31 if mant == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Converts `f32` to IEEE binary16, rounding to nearest even and clamping
/// finite values to the largest finite half (±65504).
pub(crate) fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7bff;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let rounded = if rem > halfway || (rem == halfway && half & 1 == 1) {
            half + 1
        } else {
            half
        };
        return sign | rounded as u16;
    }
    let mut out = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && out & 1 == 1) {
        out += 1;
    }
    sign | out.min(0x7bff) as u16
}

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

/// CRC-32 (IEEE), as used by gzip and ZIP.
pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = CRC_TABLE[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// Decompresses a single-member gzip stream and checks its CRC.
pub(crate) fn gunzip(data: &[u8], what: &'static str) -> Result<Vec<u8>> {
    let bad = |msg: &str| Error::Parse(format!("{what}: {msg}"));
    if data.len() < 18 || data[0] != 0x1f || data[1] != 0x8b {
        return Err(bad("not a gzip stream"));
    }
    if data[2] != 8 {
        return Err(bad("gzip stream does not use deflate"));
    }
    let flags = data[3];
    let mut c = Cursor::at(data, 10, what);
    if flags & 4 != 0 {
        let extra = c.u16()?;
        c.take(usize::from(extra))?;
    }
    for flag in [8, 16] {
        if flags & flag != 0 {
            while c.u8()? != 0 {}
        }
    }
    if flags & 2 != 0 {
        c.take(2)?;
    }
    if c.pos() > data.len() - 8 {
        return Err(bad("gzip header runs past the end of the data"));
    }
    let body = &data[c.pos()..data.len() - 8];
    let trailer = &data[data.len() - 8..];
    let crc = u32::from_le_bytes(trailer[..4].try_into().expect("4 bytes"));
    // The trailer records the decompressed size, which bounds memory use.
    let size = u32::from_le_bytes(trailer[4..].try_into().expect("4 bytes")) as usize;
    let out = miniz_oxide::inflate::decompress_to_vec_with_limit(body, size)
        .map_err(|e| bad(&format!("corrupt compressed data ({:?})", e.status)))?;
    if crc != crc32(&out) {
        return Err(bad("compressed data failed its CRC check"));
    }
    Ok(out)
}

/// Compresses `data` as a gzip stream.
pub(crate) fn gzip(data: &[u8], level: u8) -> Vec<u8> {
    let body = miniz_oxide::deflate::compress_to_vec(data, level.min(10));
    let mut out = Vec::with_capacity(body.len() + 18);
    out.extend_from_slice(&[0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255]);
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// Decompresses one zstd frame (any trailing skippable frames are ignored).
pub(crate) fn unzstd(data: &[u8], expected_len: usize, what: &'static str) -> Result<Vec<u8>> {
    let decoder = ruzstd::decoding::StreamingDecoder::new(data)
        .map_err(|e| Error::Parse(format!("{what}: corrupt zstd stream ({e})")))?;
    let mut out = Vec::with_capacity(expected_len.min(1 << 28));
    // Never inflate past the declared size (plus one byte to detect excess).
    decoder
        .take(expected_len as u64 + 1)
        .read_to_end(&mut out)
        .map_err(|e| Error::Parse(format!("{what}: corrupt zstd stream ({e})")))?;
    Ok(out)
}

/// Compresses `data` as one zstd frame.
pub(crate) fn zstd(data: &[u8]) -> Vec<u8> {
    ruzstd::encoding::compress_to_vec(data, ruzstd::encoding::CompressionLevel::Fastest)
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding.
pub(crate) fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(BASE64[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Decodes standard or URL-safe base64; padding is optional.
pub(crate) fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0;
    for &c in text.trim_end_matches('=').as_bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_float_roundtrip() {
        for v in [0.0f32, 1.0, -2.5, 0.1, 65504.0, 6.1e-5, 1e-7, -0.333] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 6e-8, "{v} -> {back}");
        }
        assert_eq!(f16_to_f32(f32_to_f16(1e9)), 65504.0);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert_eq!(f32_to_f16(1.0), 0x3c00);
    }

    #[test]
    fn crc32_matches_known_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn gzip_roundtrip() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(gunzip(&gzip(&data, 6), "test").unwrap(), data);
        assert_eq!(gunzip(&gzip(&data, 1), "test").unwrap(), data);
        let mut corrupt = gzip(&data, 6);
        let n = corrupt.len();
        corrupt[n - 6] ^= 1;
        assert!(gunzip(&corrupt, "test").is_err());
    }

    #[test]
    fn zstd_roundtrip() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 13) as u8).collect();
        assert_eq!(unzstd(&zstd(&data), data.len(), "test").unwrap(), data);
    }

    #[test]
    fn base64_roundtrip() {
        for len in 0..10 {
            let data: Vec<u8> = (0..len).map(|i: u32| (i * 37) as u8).collect();
            assert_eq!(base64_decode(&base64_encode(&data)).unwrap(), data);
        }
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b"M"), "TQ==");
    }
}
