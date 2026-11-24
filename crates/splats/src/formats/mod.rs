//! Splat file formats and file-level helpers.
//!
//! Each format has its own module with `read_*` / `write_*` functions (and
//! `*_with` variants taking options). [`Format`] picks one at run time, and
//! [`read_file`] / [`write_file`] work from paths, detecting the format from
//! file contents or extension.

pub mod compressed_ply;
pub mod gltf;
pub mod ksplat;
pub mod ply;
pub mod sog;
pub mod splat;
pub mod spz;

use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::cloud::SplatCloud;
use crate::error::{Error, Result};

/// A supported splat file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// Gaussian splat `.ply`. Lossless.
    Ply,
    /// Chunk-quantized `.compressed.ply`, 16 bytes per splat plus SH.
    CompressedPly,
    /// Compact `.splat`, 32 bytes per splat, no higher-order SH.
    Splat,
    /// Compressed `.spz`.
    Spz,
    /// `.ksplat` sectioned splat buffer.
    Ksplat,
    /// `.sog`: quantized textures in lossless WebP, bundled in a ZIP
    /// archive (or a `meta.json` with image files beside it).
    Sog,
    /// Binary glTF `.glb` with `KHR_gaussian_splatting`.
    Glb,
    /// JSON glTF `.gltf` with `KHR_gaussian_splatting`.
    Gltf,
}

impl Format {
    /// Every supported format.
    pub const ALL: [Format; 8] = [
        Format::Ply,
        Format::CompressedPly,
        Format::Splat,
        Format::Spz,
        Format::Ksplat,
        Format::Sog,
        Format::Glb,
        Format::Gltf,
    ];

    /// The usual file extension, without the leading dot.
    pub fn extension(self) -> &'static str {
        match self {
            Format::Ply => "ply",
            Format::CompressedPly => "compressed.ply",
            Format::Splat => "splat",
            Format::Spz => "spz",
            Format::Ksplat => "ksplat",
            Format::Sog => "sog",
            Format::Glb => "glb",
            Format::Gltf => "gltf",
        }
    }

    /// A short description.
    pub fn description(self) -> &'static str {
        match self {
            Format::Ply => "Gaussian splat PLY, lossless",
            Format::CompressedPly => "chunk-quantized PLY, 16 bytes/splat + SH",
            Format::Splat => "compact 32 bytes/splat, base color only",
            Format::Spz => "quantized and gzip/zstd compressed",
            Format::Ksplat => "sectioned buffer, 3 compression levels, SH up to 2",
            Format::Sog => "quantized lossless-WebP textures in a ZIP",
            Format::Glb => "binary glTF, KHR_gaussian_splatting",
            Format::Gltf => "JSON glTF, KHR_gaussian_splatting",
        }
    }

    /// Whether writing and reading back reproduces the cloud exactly.
    pub fn is_lossless(self) -> bool {
        matches!(self, Format::Ply)
    }

    /// Highest SH degree the format stores.
    pub fn max_sh_degree(self) -> u8 {
        match self {
            Format::Splat => 0,
            Format::Ksplat => 2,
            _ => 3,
        }
    }

    /// The format for a file extension (case-insensitive, without the dot).
    pub fn from_extension(ext: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|f| f.extension().eq_ignore_ascii_case(ext))
    }

    /// The format implied by a path's name: `*.compressed.ply` and
    /// `meta.json` are recognized as well as single extensions.
    pub fn from_path(path: impl AsRef<Path>) -> Option<Self> {
        let path = path.as_ref();
        let name = path.file_name()?.to_str()?.to_ascii_lowercase();
        if name.ends_with(".compressed.ply") {
            return Some(Format::CompressedPly);
        }
        if name == "meta.json" || name.ends_with(".meta.json") {
            return Some(Format::Sog);
        }
        Self::from_extension(path.extension()?.to_str()?)
    }

    /// Determines a file's format from its contents where possible, falling
    /// back to its name.
    pub fn detect(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = BufReader::new(File::open(path)?);
        let head = file
            .fill_buf()?
            .get(..4)
            .map(<[u8]>::to_vec)
            .unwrap_or_default();
        let by_name = Self::from_path(path);
        if head.starts_with(b"ply") {
            let header = ply::read_header(&mut file)?;
            return Ok(if compressed_ply::is_compressed(&header) {
                Format::CompressedPly
            } else {
                Format::Ply
            });
        }
        if head.starts_with(&[0x1f, 0x8b]) || head == b"NGSP" {
            return Ok(Format::Spz);
        }
        if head == b"glTF" {
            return Ok(Format::Glb);
        }
        if head == b"PK\x03\x04" {
            return Ok(Format::Sog);
        }
        by_name.ok_or_else(|| {
            Error::Unsupported(format!(
                "cannot tell the format of '{}' (supported: {})",
                path.display(),
                Self::extension_list()
            ))
        })
    }

    /// Supported extensions as a display string.
    pub fn extension_list() -> String {
        Self::ALL
            .iter()
            .map(|f| format!(".{}", f.extension()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Reads a cloud in this format from a stream. Files that reference
    /// other files (`.gltf` with external buffers, unbundled SOG) need
    /// [`read_file_as`] instead.
    pub fn read<R: BufRead>(self, reader: R) -> Result<SplatCloud> {
        match self {
            Format::Ply | Format::CompressedPly => ply::read_ply(reader),
            Format::Splat => splat::read_splat(reader),
            Format::Spz => spz::read_spz(reader),
            Format::Ksplat => ksplat::read_ksplat(reader),
            Format::Sog => sog::read_sog(reader),
            Format::Glb => gltf::read_glb(reader),
            Format::Gltf => gltf::read_gltf(reader, None),
        }
    }

    /// Writes a cloud in this format with default options. Formats with
    /// limited SH store as many degrees as they can.
    pub fn write<W: Write>(self, cloud: &SplatCloud, writer: W) -> Result<()> {
        match self {
            Format::Ply => ply::write_ply(cloud, writer),
            Format::CompressedPly => compressed_ply::write_compressed_ply(cloud, writer),
            Format::Splat => splat::write_splat(cloud, writer),
            Format::Spz => spz::write_spz(cloud, writer),
            Format::Ksplat => ksplat::write_ksplat(cloud, writer),
            Format::Sog => sog::write_sog(cloud, writer),
            Format::Glb => gltf::write_glb(cloud, writer),
            Format::Gltf => gltf::write_gltf(cloud, writer),
        }
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.extension())
    }
}

/// Reads a splat file, detecting its format.
pub fn read_file(path: impl AsRef<Path>) -> Result<SplatCloud> {
    let path = path.as_ref();
    read_file_as(path, Format::detect(path)?)
}

/// Reads a splat file in the given format. Files referenced by `.gltf`
/// documents and unbundled SOG `meta.json` files are resolved relative to
/// the file's folder.
pub fn read_file_as(path: impl AsRef<Path>, format: Format) -> Result<SplatCloud> {
    let path = path.as_ref();
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let reader = BufReader::with_capacity(1 << 20, File::open(path)?);
    match format {
        Format::Gltf => gltf::read_gltf(reader, Some(dir)),
        Format::Glb => gltf::read_glb_with(reader, Some(dir), gltf::GltfOptions::default()),
        Format::Sog
            if !path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("sog")) =>
        {
            sog::read_sog_dir(path)
        }
        _ => format.read(reader),
    }
}

/// Writes a splat file in the format implied by its name.
pub fn write_file(cloud: &SplatCloud, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let format = Format::from_path(path).ok_or_else(|| {
        Error::Unsupported(format!(
            "cannot tell which format to write for '{}' (supported: {})",
            path.display(),
            Format::extension_list()
        ))
    })?;
    write_file_as(cloud, path, format)
}

/// Writes a splat file in the given format. For SOG, a path named
/// `meta.json` writes the unbundled form (JSON plus image files beside it).
pub fn write_file_as(cloud: &SplatCloud, path: impl AsRef<Path>, format: Format) -> Result<()> {
    let path = path.as_ref();
    cloud.validate()?;
    if format == Format::Sog
        && !path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("sog"))
    {
        return sog::write_sog_dir(cloud, path, sog::SogOptions::default());
    }
    let mut out = BufWriter::with_capacity(1 << 20, File::create(path)?);
    format.write(cloud, &mut out)?;
    out.flush()?;
    Ok(())
}
