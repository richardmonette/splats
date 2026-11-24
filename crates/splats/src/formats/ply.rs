//! Gaussian splat `.ply` files.
//!
//! Reading accepts ASCII and binary (little- or big-endian) files, properties
//! in any order and of any numeric type, and extra elements or properties,
//! which are skipped. Writing always produces binary little-endian `float`
//! properties in the conventional order:
//! `x y z nx ny nz f_dc_0..2 f_rest_* opacity scale_0..2 rot_0..3`.
//!
//! Higher-order spherical harmonics are stored channel-major in the file
//! (`f_rest_0..K` red, then green, then blue) and converted to and from
//! [`SplatCloud`]'s interleaved layout.
//!
//! [`read_ply`] also recognizes the chunk-quantized compressed layout and
//! hands it to [`compressed_ply`](super::compressed_ply).

use std::io::{self, BufRead, BufWriter, Read, Write};

use crate::cloud::{SplatCloud, SplatRef, sh_rest_coeffs};
use crate::error::{Error, Result};

/// Real headers are a few hundred bytes; this bounds memory on garbage input.
const MAX_HEADER_BYTES: u64 = 1 << 20;
/// Cap on up-front allocation so a corrupt count cannot exhaust memory before
/// any data is read.
pub(crate) const MAX_PREALLOCATED_SPLATS: usize = 1 << 16;

pub(crate) fn err(msg: impl std::fmt::Display) -> Error {
    Error::Parse(format!("ply: {msg}"))
}

fn eof(what: &str) -> Error {
    err(format!("file ends before {what} was fully read"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Encoding {
    Ascii,
    BinaryLe,
    BinaryBe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "char" | "int8" => Scalar::I8,
            "uchar" | "uint8" => Scalar::U8,
            "short" | "int16" => Scalar::I16,
            "ushort" | "uint16" => Scalar::U16,
            "int" | "int32" => Scalar::I32,
            "uint" | "uint32" => Scalar::U32,
            "float" | "float32" => Scalar::F32,
            "double" | "float64" => Scalar::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Scalar::I8 | Scalar::U8 => 1,
            Scalar::I16 | Scalar::U16 => 2,
            Scalar::I32 | Scalar::U32 | Scalar::F32 => 4,
            Scalar::F64 => 8,
        }
    }

    /// Decodes one value from the start of `b`, which must hold at least
    /// `self.size()` bytes.
    fn decode(self, b: &[u8], big_endian: bool) -> f64 {
        macro_rules! num {
            ($t:ty) => {{
                const N: usize = std::mem::size_of::<$t>();
                let bytes: [u8; N] = b[..N].try_into().expect("caller checks the length");
                if big_endian {
                    <$t>::from_be_bytes(bytes) as f64
                } else {
                    <$t>::from_le_bytes(bytes) as f64
                }
            }};
        }
        match self {
            Scalar::I8 => f64::from(b[0] as i8),
            Scalar::U8 => f64::from(b[0]),
            Scalar::I16 => num!(i16),
            Scalar::U16 => num!(u16),
            Scalar::I32 => num!(i32),
            Scalar::U32 => num!(u32),
            Scalar::F32 => num!(f32),
            Scalar::F64 => num!(f64),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PropKind {
    Scalar(Scalar),
    List { count: Scalar, item: Scalar },
}

#[derive(Clone, Debug)]
pub(crate) struct Property {
    pub name: String,
    pub kind: PropKind,
}

#[derive(Clone, Debug)]
pub(crate) struct Element {
    pub name: String,
    pub count: usize,
    pub properties: Vec<Property>,
}

impl Element {
    /// Index of the property called `name`.
    pub fn property(&self, name: &str) -> Option<usize> {
        self.properties.iter().position(|p| p.name == name)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Header {
    pub encoding: Encoding,
    pub elements: Vec<Element>,
}

fn read_header_line<R: BufRead>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    total: &mut u64,
) -> Result<Option<String>> {
    buf.clear();
    let n = (&mut *reader)
        .take(MAX_HEADER_BYTES + 1 - *total)
        .read_until(b'\n', buf)?;
    if n == 0 {
        return Ok(None);
    }
    *total += n as u64;
    if *total > MAX_HEADER_BYTES {
        return Err(err("header is larger than 1 MiB"));
    }
    let line = std::str::from_utf8(buf).map_err(|_| err("header is not valid text"))?;
    Ok(Some(line.trim_end_matches(['\n', '\r']).to_string()))
}

pub(crate) fn read_header<R: BufRead>(reader: &mut R) -> Result<Header> {
    let mut buf = Vec::new();
    let mut total = 0;
    let first =
        read_header_line(reader, &mut buf, &mut total)?.ok_or_else(|| err("file is empty"))?;
    if first.trim() != "ply" {
        return Err(err("missing 'ply' signature"));
    }
    let mut encoding = None;
    let mut elements: Vec<Element> = Vec::new();
    loop {
        let line = read_header_line(reader, &mut buf, &mut total)?
            .ok_or_else(|| err("file ends inside the header"))?;
        let mut tokens = line.split_whitespace();
        match tokens.next() {
            None | Some("comment" | "obj_info") => {}
            Some("format") => {
                encoding = Some(match tokens.next() {
                    Some("ascii") => Encoding::Ascii,
                    Some("binary_little_endian") => Encoding::BinaryLe,
                    Some("binary_big_endian") => Encoding::BinaryBe,
                    other => {
                        return Err(err(format!("unknown format '{}'", other.unwrap_or(""))));
                    }
                });
            }
            Some("element") => {
                let name = tokens.next().ok_or_else(|| err("element without a name"))?;
                let count = tokens
                    .next()
                    .and_then(|c| c.parse::<usize>().ok())
                    .ok_or_else(|| err(format!("element '{name}' has no valid count")))?;
                elements.push(Element {
                    name: name.to_string(),
                    count,
                    properties: Vec::new(),
                });
            }
            Some("property") => {
                let element = elements
                    .last_mut()
                    .ok_or_else(|| err("property declared before any element"))?;
                let ty = tokens
                    .next()
                    .ok_or_else(|| err("property without a type"))?;
                let kind = if ty == "list" {
                    let count = tokens.next().and_then(Scalar::parse);
                    let item = tokens.next().and_then(Scalar::parse);
                    match (count, item) {
                        (Some(count), Some(item)) => PropKind::List { count, item },
                        _ => return Err(err("list property with invalid types")),
                    }
                } else {
                    PropKind::Scalar(
                        Scalar::parse(ty)
                            .ok_or_else(|| err(format!("unknown property type '{ty}'")))?,
                    )
                };
                let name = tokens
                    .next()
                    .ok_or_else(|| err("property without a name"))?;
                element.properties.push(Property {
                    name: name.to_string(),
                    kind,
                });
            }
            Some("end_header") => break,
            Some(other) => return Err(err(format!("unexpected header keyword '{other}'"))),
        }
    }
    let encoding = encoding.ok_or_else(|| err("missing 'format' line"))?;
    Ok(Header { encoding, elements })
}

fn skip_bytes<R: Read>(reader: &mut R, n: u64, what: &str) -> Result<()> {
    let copied = io::copy(&mut reader.by_ref().take(n), &mut io::sink())?;
    if copied == n { Ok(()) } else { Err(eof(what)) }
}

fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8], what: &str) -> Result<()> {
    reader.read_exact(buf).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            eof(what)
        } else {
            e.into()
        }
    })
}

/// Skips over every row of `element`.
pub(crate) fn skip_element<R: BufRead>(
    reader: &mut R,
    element: &Element,
    encoding: Encoding,
) -> Result<()> {
    let what = format!("element '{}'", element.name);
    if encoding == Encoding::Ascii {
        let mut line = String::new();
        for _ in 0..element.count {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                return Err(eof(&what));
            }
        }
        return Ok(());
    }
    let big_endian = encoding == Encoding::BinaryBe;
    let fixed_row: Option<usize> = element
        .properties
        .iter()
        .map(|p| match p.kind {
            PropKind::Scalar(t) => Some(t.size()),
            PropKind::List { .. } => None,
        })
        .sum();
    if let Some(row) = fixed_row {
        let total = (row as u64)
            .checked_mul(element.count as u64)
            .ok_or_else(|| err(format!("{what} is too large")))?;
        return skip_bytes(reader, total, &what);
    }
    let mut small = [0u8; 8];
    for _ in 0..element.count {
        for p in &element.properties {
            match p.kind {
                PropKind::Scalar(t) => skip_bytes(reader, t.size() as u64, &what)?,
                PropKind::List { count, item } => {
                    let n = count.size();
                    read_exact_or_eof(reader, &mut small[..n], &what)?;
                    let len = count.decode(&small[..n], big_endian);
                    if len.is_nan() || len < 0.0 {
                        return Err(err(format!("negative list length in {what}")));
                    }
                    skip_bytes(
                        reader,
                        (len as u64).saturating_mul(item.size() as u64),
                        &what,
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Reads every row of an element made only of scalar properties, calling
/// `on_row` with the row's values in property order.
pub(crate) fn read_rows<R: BufRead>(
    reader: &mut R,
    element: &Element,
    encoding: Encoding,
    mut on_row: impl FnMut(&[f64]) -> Result<()>,
) -> Result<()> {
    let mut types = Vec::with_capacity(element.properties.len());
    for p in &element.properties {
        match p.kind {
            PropKind::Scalar(t) => types.push(t),
            PropKind::List { .. } => {
                return Err(Error::Unsupported(format!(
                    "ply: list property '{}' in element '{}'",
                    p.name, element.name
                )));
            }
        }
    }
    let what = format!("element '{}'", element.name);
    let mut values = vec![0.0f64; types.len()];
    match encoding {
        Encoding::Ascii => {
            let mut line = String::new();
            for i in 0..element.count {
                loop {
                    line.clear();
                    if reader.read_line(&mut line)? == 0 {
                        return Err(eof(&what));
                    }
                    if !line.trim().is_empty() {
                        break;
                    }
                }
                let mut tokens = line.split_whitespace();
                for value in values.iter_mut() {
                    let t = tokens
                        .next()
                        .ok_or_else(|| err(format!("row {i} of {what} has too few values")))?;
                    *value = t.parse().map_err(|_| {
                        err(format!("row {i} of {what} has an invalid number '{t}'"))
                    })?;
                }
                on_row(&values)?;
            }
        }
        Encoding::BinaryLe | Encoding::BinaryBe => {
            let big_endian = encoding == Encoding::BinaryBe;
            let mut offsets = Vec::with_capacity(types.len());
            let mut row_size = 0;
            for t in &types {
                offsets.push(row_size);
                row_size += t.size();
            }
            if row_size == 0 {
                for _ in 0..element.count {
                    on_row(&values)?;
                }
                return Ok(());
            }
            let rows_per_chunk = ((1usize << 20) / row_size).max(1);
            let mut buf = vec![0u8; rows_per_chunk * row_size];
            let mut remaining = element.count;
            while remaining > 0 {
                let rows = remaining.min(rows_per_chunk);
                let bytes = &mut buf[..rows * row_size];
                read_exact_or_eof(reader, bytes, &what)?;
                for raw in bytes.chunks_exact(row_size) {
                    for ((value, t), &off) in values.iter_mut().zip(&types).zip(&offsets) {
                        *value = t.decode(&raw[off..], big_endian);
                    }
                    on_row(&values)?;
                }
                remaining -= rows;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Slot {
    Position(usize),
    LogScale(usize),
    Rotation(usize),
    Opacity,
    Dc(usize),
    /// Index into the interleaved `sh_rest` block (after remapping).
    Rest(usize),
    Ignore,
}

const FIXED_NAMES: [&str; 14] = [
    "x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity",
    "f_dc_0", "f_dc_1", "f_dc_2",
];

fn slot_for(name: &str) -> Slot {
    fn index(name: &str, prefix: &str, limit: usize) -> Option<usize> {
        name.strip_prefix(prefix)?
            .parse()
            .ok()
            .filter(|&i| i < limit)
    }
    match name {
        "x" => Slot::Position(0),
        "y" => Slot::Position(1),
        "z" => Slot::Position(2),
        "opacity" => Slot::Opacity,
        _ => {
            if let Some(i) = index(name, "scale_", 3) {
                Slot::LogScale(i)
            } else if let Some(i) = index(name, "rot_", 4) {
                Slot::Rotation(i)
            } else if let Some(i) = index(name, "f_dc_", 3) {
                Slot::Dc(i)
            } else if let Some(i) = index(name, "f_rest_", usize::MAX) {
                Slot::Rest(i)
            } else {
                Slot::Ignore
            }
        }
    }
}

struct VertexLayout {
    slots: Vec<Slot>,
    sh_degree: u8,
}

impl VertexLayout {
    fn new(element: &Element) -> Result<Self> {
        let mut slots = Vec::with_capacity(element.properties.len());
        let mut seen = [false; FIXED_NAMES.len()];
        let mut rest = Vec::new();
        for p in &element.properties {
            let slot = slot_for(&p.name);
            let fixed = match slot {
                Slot::Position(i) => Some(i),
                Slot::LogScale(i) => Some(3 + i),
                Slot::Rotation(i) => Some(6 + i),
                Slot::Opacity => Some(10),
                Slot::Dc(i) => Some(11 + i),
                Slot::Rest(j) => {
                    rest.push(j);
                    None
                }
                Slot::Ignore => None,
            };
            if let Some(f) = fixed {
                if seen[f] {
                    return Err(err(format!("duplicate property '{}'", p.name)));
                }
                seen[f] = true;
            }
            slots.push(slot);
        }
        let missing: Vec<&str> = FIXED_NAMES
            .iter()
            .zip(seen)
            .filter(|(_, present)| !present)
            .map(|(name, _)| *name)
            .collect();
        if !missing.is_empty() {
            return Err(err(format!(
                "not a Gaussian splat file; missing properties: {}",
                missing.join(", ")
            )));
        }
        rest.sort_unstable();
        if rest.iter().enumerate().any(|(i, &j)| i != j) {
            return Err(err(
                "f_rest_* properties must be numbered 0..N with no gaps or repeats",
            ));
        }
        let sh_degree = match rest.len() {
            0 => 0,
            9 => 1,
            24 => 2,
            45 => 3,
            n => {
                return Err(err(format!(
                    "{n} f_rest_* properties do not match any SH degree (expected 0, 9, 24 or 45)"
                )));
            }
        };
        let k = sh_rest_coeffs(sh_degree);
        for slot in &mut slots {
            if let Slot::Rest(j) = *slot {
                // File: channel-major `c * k + i`. Memory: `i * 3 + c`.
                *slot = Slot::Rest((j % k) * 3 + j / k);
            }
        }
        Ok(Self { slots, sh_degree })
    }
}

/// Reads a Gaussian splat `.ply` file, in either the standard or the
/// compressed layout.
pub fn read_ply<R: BufRead>(mut reader: R) -> Result<SplatCloud> {
    let header = read_header(&mut reader)?;
    if super::compressed_ply::is_compressed(&header) {
        return super::compressed_ply::read_body(&mut reader, &header);
    }
    let index = header
        .elements
        .iter()
        .position(|e| e.name == "vertex")
        .ok_or_else(|| err("no 'vertex' element"))?;
    for element in &header.elements[..index] {
        skip_element(&mut reader, element, header.encoding)?;
    }
    let vertex = &header.elements[index];
    let layout = VertexLayout::new(vertex)?;
    let mut cloud =
        SplatCloud::with_capacity(layout.sh_degree, vertex.count.min(MAX_PREALLOCATED_SPLATS));
    let mut rest = vec![0.0f32; cloud.sh_rest_stride()];
    read_rows(&mut reader, vertex, header.encoding, |values| {
        let mut splat = SplatRef {
            position: [0.0; 3],
            log_scale: [0.0; 3],
            rotation: [0.0; 4],
            opacity_logit: 0.0,
            sh_dc: [0.0; 3],
            sh_rest: &[],
        };
        for (&slot, &v) in layout.slots.iter().zip(values) {
            let v = v as f32;
            match slot {
                Slot::Position(i) => splat.position[i] = v,
                Slot::LogScale(i) => splat.log_scale[i] = v,
                Slot::Rotation(i) => splat.rotation[i] = v,
                Slot::Opacity => splat.opacity_logit = v,
                Slot::Dc(i) => splat.sh_dc[i] = v,
                Slot::Rest(i) => rest[i] = v,
                Slot::Ignore => {}
            }
        }
        splat.sh_rest = &rest;
        cloud.push(splat);
        Ok(())
    })?;
    Ok(cloud)
}

/// Writes `cloud` as a binary little-endian Gaussian splat `.ply` file.
/// This is lossless.
pub fn write_ply<W: Write>(cloud: &SplatCloud, writer: W) -> Result<()> {
    cloud.validate()?;
    let k = sh_rest_coeffs(cloud.sh_degree);
    let mut names: Vec<String> = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    .map(String::from)
    .to_vec();
    names.extend((0..3 * k).map(|j| format!("f_rest_{j}")));
    names.extend(
        [
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ]
        .map(String::from),
    );
    let mut header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n",
        cloud.len()
    );
    for name in &names {
        header.push_str("property float ");
        header.push_str(name);
        header.push('\n');
    }
    header.push_str("end_header\n");

    let mut out = BufWriter::with_capacity(1 << 20, writer);
    out.write_all(header.as_bytes())?;
    let mut row: Vec<u8> = Vec::with_capacity(names.len() * 4);
    for s in cloud.iter() {
        row.clear();
        let mut put = |v: f32| row.extend_from_slice(&v.to_le_bytes());
        s.position.into_iter().for_each(&mut put);
        [0.0; 3].into_iter().for_each(&mut put);
        s.sh_dc.into_iter().for_each(&mut put);
        for c in 0..3 {
            for i in 0..k {
                put(s.sh_rest[i * 3 + c]);
            }
        }
        put(s.opacity_logit);
        s.log_scale.into_iter().for_each(&mut put);
        s.rotation.into_iter().for_each(&mut put);
        out.write_all(&row)?;
    }
    out.flush()?;
    Ok(())
}
