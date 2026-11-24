//! Minimal ZIP archive support: reading stored or deflated entries, and
//! writing stored entries.

use crate::error::{Error, Result};
use crate::util::{Cursor, crc32};

const LOCAL: u32 = 0x0403_4b50;
const CENTRAL: u32 = 0x0201_4b50;
const END: u32 = 0x0605_4b50;

fn err(msg: impl std::fmt::Display) -> Error {
    Error::Parse(format!("zip: {msg}"))
}

struct Entry {
    name: String,
    method: u16,
    crc: u32,
    compressed: usize,
    uncompressed: usize,
    header_offset: usize,
}

pub(crate) struct ZipArchive<'a> {
    data: &'a [u8],
    entries: Vec<Entry>,
}

impl<'a> ZipArchive<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 22 {
            return Err(err("file is too short to be an archive"));
        }
        let search_start = data.len().saturating_sub(22 + 65_535);
        let eocd = (search_start..=data.len().saturating_sub(22))
            .rev()
            .find(|&i| data[i..i + 4] == END.to_le_bytes())
            .ok_or_else(|| err("end of central directory not found"))?;
        let mut c = Cursor::at(data, eocd + 10, "zip");
        let count = usize::from(c.u16()?);
        let _size = c.u32()?;
        let offset = c.u32()?;
        if offset == u32::MAX || count == usize::from(u16::MAX) {
            return Err(Error::Unsupported("zip64 archives".into()));
        }
        let mut c = Cursor::at(data, offset as usize, "zip");
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if c.u32()? != CENTRAL {
                return Err(err("corrupt central directory"));
            }
            c.take(6)?;
            let method = c.u16()?;
            c.take(4)?;
            let crc = c.u32()?;
            let compressed = c.u32()? as usize;
            let uncompressed = c.u32()? as usize;
            let name_len = usize::from(c.u16()?);
            let extra_len = usize::from(c.u16()?);
            let comment_len = usize::from(c.u16()?);
            c.take(8)?;
            let header_offset = c.u32()? as usize;
            let name = String::from_utf8_lossy(c.take(name_len)?).into_owned();
            c.take(extra_len + comment_len)?;
            entries.push(Entry {
                name,
                method,
                crc,
                compressed,
                uncompressed,
                header_offset,
            });
        }
        Ok(Self { data, entries })
    }

    /// Whether an entry with this name exists.
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|e| e.name == name)
    }

    /// Returns the contents of the entry called `name`.
    pub(crate) fn read(&self, name: &str) -> Result<Vec<u8>> {
        let e = self
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| err(format!("archive has no file '{name}'")))?;
        let mut c = Cursor::at(self.data, e.header_offset, "zip");
        if c.u32()? != LOCAL {
            return Err(err(format!("corrupt local header for '{name}'")));
        }
        c.take(22)?;
        let name_len = usize::from(c.u16()?);
        let extra_len = usize::from(c.u16()?);
        c.take(name_len + extra_len)?;
        let raw = c.take(e.compressed)?;
        let out = match e.method {
            0 => raw.to_vec(),
            8 => miniz_oxide::inflate::decompress_to_vec_with_limit(raw, e.uncompressed)
                .map_err(|_| err(format!("corrupt compressed data in '{name}'")))?,
            m => {
                return Err(Error::Unsupported(format!(
                    "zip: compression method {m} for '{name}'"
                )));
            }
        };
        if out.len() != e.uncompressed || crc32(&out) != e.crc {
            return Err(err(format!("'{name}' failed its integrity check")));
        }
        Ok(out)
    }
}

/// Builds a ZIP archive of stored (uncompressed) entries.
pub(crate) fn write_zip(files: &[(String, Vec<u8>)]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    // 1980-01-01 00:00, the earliest DOS date.
    let (time, date) = (0u16, 0x21u16);
    for (name, data) in files {
        let offset = u32::try_from(out.len())
            .map_err(|_| Error::Invalid("archive would exceed 4 GiB".into()))?;
        let size = u32::try_from(data.len())
            .map_err(|_| Error::Invalid("archive entry would exceed 4 GiB".into()))?;
        let crc = crc32(data);
        let name_len =
            u16::try_from(name.len()).map_err(|_| Error::Invalid("file name too long".into()))?;
        let mut common = Vec::with_capacity(26);
        common.extend_from_slice(&20u16.to_le_bytes()); // version needed
        common.extend_from_slice(&0x0800u16.to_le_bytes()); // UTF-8 names
        common.extend_from_slice(&0u16.to_le_bytes()); // stored
        common.extend_from_slice(&time.to_le_bytes());
        common.extend_from_slice(&date.to_le_bytes());
        common.extend_from_slice(&crc.to_le_bytes());
        common.extend_from_slice(&size.to_le_bytes());
        common.extend_from_slice(&size.to_le_bytes());
        common.extend_from_slice(&name_len.to_le_bytes());
        common.extend_from_slice(&0u16.to_le_bytes()); // extra length

        out.extend_from_slice(&LOCAL.to_le_bytes());
        out.extend_from_slice(&common);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);

        central.extend_from_slice(&CENTRAL.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes()); // version made by
        central.extend_from_slice(&common);
        central.extend_from_slice(&[0; 6]); // comment len, disk, internal attrs
        central.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let count = u16::try_from(files.len()).map_err(|_| Error::Invalid("too many files".into()))?;
    let cd_offset = u32::try_from(out.len())
        .map_err(|_| Error::Invalid("archive would exceed 4 GiB".into()))?;
    out.extend_from_slice(&central);
    out.extend_from_slice(&END.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let files = vec![
            ("a.txt".to_string(), b"hello".to_vec()),
            ("b/c.bin".to_string(), (0..=255u8).collect()),
            ("empty".to_string(), Vec::new()),
        ];
        let data = write_zip(&files).unwrap();
        let archive = ZipArchive::new(&data).unwrap();
        for (name, contents) in &files {
            assert_eq!(&archive.read(name).unwrap(), contents);
        }
        assert!(archive.read("missing").is_err());
    }

    #[test]
    fn reads_deflated_entries() {
        // Build an archive by hand with one deflated entry.
        let contents = b"splat splat splat splat splat".repeat(20);
        let body = miniz_oxide::deflate::compress_to_vec(&contents, 6);
        let mut files = write_zip(&[("x".to_string(), body.clone())]).unwrap();
        // Patch method to 8 and the sizes/CRC in both headers.
        let crc = crc32(&contents).to_le_bytes();
        let size = (contents.len() as u32).to_le_bytes();
        for header in [0usize, 30 + 1 + body.len()] {
            let base = if header == 0 { 4 } else { header + 6 };
            files[base + 4..base + 6].copy_from_slice(&8u16.to_le_bytes());
            files[base + 10..base + 14].copy_from_slice(&crc);
            files[base + 18..base + 22].copy_from_slice(&size);
        }
        let archive = ZipArchive::new(&files).unwrap();
        assert_eq!(archive.read("x").unwrap(), contents);
    }
}
