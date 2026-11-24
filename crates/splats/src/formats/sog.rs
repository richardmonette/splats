//! SOG scenes: splat attributes quantized into lossless WebP images plus a
//! `meta.json` description, either bundled in a `.sog` ZIP archive or
//! stored as loose files.
//!
//! Reads versions 1, 2 and 3. Writes version 2 (default, widest support)
//! or 3. Positions are stored as 16-bit values on a log curve, scales and
//! base colors through 256-entry codebooks, and higher-order SH through a
//! k-means palette shared by many splats.
//!
//! The writer reorders splats along a Morton curve.

use std::io::{Read, Write};
use std::path::Path;

use serde_json::{Value, json};

use crate::cloud::{SplatCloud, SplatRef, logit, normalize_quaternion, sh_rest_coeffs, sigmoid};
use crate::error::{Error, Result};
use crate::quantize::{codebook_1d, kmeans, nearest};
use crate::transform;
use crate::util::read_all;
use crate::zip::{ZipArchive, write_zip};

const META: &str = "meta.json";
const SQRT2: f32 = std::f32::consts::SQRT_2;
const HALF_PI: f32 = std::f32::consts::FRAC_PI_2;

/// Decodes a stored byte for a given channel.
type ChannelDecoder = Box<dyn Fn(usize, u8) -> f32>;
/// Decodes a stored byte.
type ByteDecoder = Box<dyn Fn(u8) -> f32>;

fn err(msg: impl std::fmt::Display) -> Error {
    Error::Parse(format!("sog: {msg}"))
}

/// Options for writing SOG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SogOptions {
    /// Format version, 2 (default) or 3.
    pub version: u32,
    /// Number of SH palette entries. `None` picks a size from the splat
    /// count (up to 65,536).
    pub palette_size: Option<usize>,
    /// Refinement rounds per palette split. Default 4.
    pub iterations: usize,
}

impl Default for SogOptions {
    fn default() -> Self {
        Self {
            version: 2,
            palette_size: None,
            iterations: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

struct Image {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

impl Image {
    fn pixel(&self, index: usize) -> [u8; 4] {
        self.rgba[index * 4..index * 4 + 4]
            .try_into()
            .expect("index checked by caller")
    }
}

fn decode_webp(name: &str, bytes: &[u8]) -> Result<Image> {
    let bad = |e: image_webp::DecodingError| err(format!("cannot decode '{name}': {e}"));
    let mut decoder = image_webp::WebPDecoder::new(std::io::Cursor::new(bytes)).map_err(bad)?;
    let (w, h) = decoder.dimensions();
    let size = decoder
        .output_buffer_size()
        .ok_or_else(|| err(format!("'{name}' is too large")))?;
    let mut buf = vec![0u8; size];
    decoder.read_image(&mut buf).map_err(bad)?;
    let pixels = w as usize * h as usize;
    let rgba = if decoder.has_alpha() {
        buf
    } else {
        let mut rgba = Vec::with_capacity(pixels * 4);
        for p in buf.chunks_exact(3) {
            rgba.extend_from_slice(&[p[0], p[1], p[2], 255]);
        }
        rgba
    };
    Ok(Image {
        width: w as usize,
        height: h as usize,
        rgba,
    })
}

fn encode_webp(image: &Image) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    image_webp::WebPEncoder::new(&mut out)
        .encode(
            &image.rgba,
            image.width as u32,
            image.height as u32,
            image_webp::ColorType::Rgba8,
        )
        .map_err(|e| Error::Invalid(format!("sog: cannot encode image: {e}")))?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Reads a bundled `.sog` archive.
pub fn read_sog<R: Read>(reader: R) -> Result<SplatCloud> {
    let data = read_all(reader)?;
    let archive = ZipArchive::new(&data)?;
    let meta = if archive.contains(META) {
        archive.read(META)?
    } else {
        return Err(err("archive has no meta.json"));
    };
    decode(&meta, |name| archive.read(name))
}

/// Reads an unbundled SOG scene from its `meta.json`; images are loaded
/// from the same folder.
pub fn read_sog_dir(meta_path: impl AsRef<Path>) -> Result<SplatCloud> {
    let meta_path = meta_path.as_ref();
    let dir = meta_path.parent().unwrap_or(Path::new("."));
    let meta = std::fs::read(meta_path)?;
    decode(&meta, |name| {
        let rel = Path::new(name);
        if !rel
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
        {
            return Err(Error::Unsupported(format!(
                "sog: file name '{name}' must be inside the scene's folder"
            )));
        }
        Ok(std::fs::read(dir.join(rel))?)
    })
}

fn floats(v: &Value, what: &str) -> Result<Vec<f32>> {
    v.as_array()
        .and_then(|a| a.iter().map(|x| x.as_f64().map(|f| f as f32)).collect())
        .ok_or_else(|| err(format!("'{what}' must be an array of numbers")))
}

fn file_names(section: &Value, what: &str, count: usize) -> Result<Vec<String>> {
    let names: Option<Vec<String>> = section
        .get("files")
        .and_then(Value::as_array)
        .and_then(|a| a.iter().map(|v| v.as_str().map(String::from)).collect());
    match names {
        Some(names) if names.len() >= count => Ok(names),
        _ => Err(err(format!("'{what}.files' must list {count} file names"))),
    }
}

/// A 256-entry codebook; a leading `null` (written by some old tools) is
/// extrapolated from its neighbours.
fn codebook(section: &Value, what: &str) -> Result<Vec<f32>> {
    let array = section
        .get("codebook")
        .and_then(Value::as_array)
        .ok_or_else(|| err(format!("'{what}.codebook' is missing")))?;
    if array.len() != 256 {
        return Err(err(format!("'{what}.codebook' must have 256 entries")));
    }
    let mut book: Vec<Option<f32>> = array.iter().map(|v| v.as_f64().map(|f| f as f32)).collect();
    if book[0].is_none() {
        if let (Some(a), Some(b)) = (book[1], book[255]) {
            book[0] = Some(a + (a - b) / 255.0);
        }
    }
    book.into_iter()
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(|| err(format!("'{what}.codebook' contains non-numbers")))
}

fn decode(meta: &[u8], mut load: impl FnMut(&str) -> Result<Vec<u8>>) -> Result<SplatCloud> {
    let meta: Value =
        serde_json::from_slice(meta).map_err(|e| err(format!("invalid meta.json: {e}")))?;
    let version = match meta.get("version") {
        None => 1,
        Some(v) => v
            .as_u64()
            .ok_or_else(|| err("'version' must be a number"))?,
    };
    if version > 3 {
        return Err(Error::Unsupported(format!("sog: version {version}")));
    }
    let section = |name: &str| {
        meta.get(name)
            .ok_or_else(|| err(format!("meta.json has no '{name}'")))
    };
    let count = if version == 1 {
        section("means")?
            .get("shape")
            .and_then(|s| s.get(0))
            .and_then(Value::as_u64)
            .ok_or_else(|| err("'means.shape' is missing"))? as usize
    } else {
        meta.get("count")
            .and_then(Value::as_u64)
            .ok_or_else(|| err("'count' is missing"))? as usize
    };
    let mut image = |section_name: &str, index: usize, count_files: usize| -> Result<Image> {
        let names = file_names(section(section_name)?, section_name, count_files)?;
        let img = decode_webp(&names[index], &load(&names[index])?)?;
        Ok(img)
    };
    let per_splat = |img: &Image, name: &str| -> Result<()> {
        if img.width * img.height < count {
            return Err(err(format!(
                "'{name}' image is too small for {count} splats"
            )));
        }
        Ok(())
    };

    let means = section("means")?;
    let mins = floats(&means["mins"], "means.mins")?;
    let maxs = floats(&means["maxs"], "means.maxs")?;
    if mins.len() < 3 || maxs.len() < 3 {
        return Err(err("'means' bounds need three values"));
    }
    let means_l = image("means", 0, 2)?;
    let means_u = image("means", 1, 2)?;
    let quats = image("quats", 0, 1)?;
    let scales = image("scales", 0, 1)?;
    let sh0 = image("sh0", 0, 1)?;
    for (img, name) in [
        (&means_l, "means_l"),
        (&means_u, "means_u"),
        (&quats, "quats"),
        (&scales, "scales"),
        (&sh0, "sh0"),
    ] {
        per_splat(img, name)?;
    }

    // Scale / color decoders differ between version 1 (linear ranges) and
    // later versions (codebooks).
    let scale_of: ChannelDecoder = if version == 1 {
        let s = section("scales")?;
        let (lo, hi) = (
            floats(&s["mins"], "scales.mins")?,
            floats(&s["maxs"], "scales.maxs")?,
        );
        if lo.len() < 3 || hi.len() < 3 {
            return Err(err("'scales' bounds need three values"));
        }
        Box::new(move |a, b| lo[a] + (hi[a] - lo[a]) * f32::from(b) / 255.0)
    } else {
        let book = codebook(section("scales")?, "scales")?;
        Box::new(move |_, b| book[usize::from(b)])
    };
    let (dc_of, opacity_of): (ChannelDecoder, ByteDecoder) = if version == 1 {
        let s = section("sh0")?;
        let (lo, hi) = (
            floats(&s["mins"], "sh0.mins")?,
            floats(&s["maxs"], "sh0.maxs")?,
        );
        if lo.len() < 4 || hi.len() < 4 {
            return Err(err("'sh0' bounds need four values"));
        }
        let (lo2, hi2) = (lo.clone(), hi.clone());
        (
            Box::new(move |c, b| lo[c] + (hi[c] - lo[c]) * f32::from(b) / 255.0),
            Box::new(move |b| lo2[3] + (hi2[3] - lo2[3]) * f32::from(b) / 255.0),
        )
    } else {
        let book = codebook(section("sh0")?, "sh0")?;
        let v3 = version == 3;
        (
            Box::new(move |_, b| book[usize::from(b)]),
            Box::new(move |b| {
                let alpha = if v3 {
                    (f32::from(b) / 204.0).powi(2)
                } else {
                    f32::from(b) / 255.0
                };
                logit(alpha)
            }),
        )
    };

    // Higher-order SH palette.
    let mut sh_degree = 0u8;
    let mut palette: Vec<f32> = Vec::new(); // entry-major, interleaved RGB
    let mut labels: Vec<usize> = Vec::new();
    let mut palette_len = 0usize;
    if let Some(sh_n) = meta.get("shN").filter(|v| !v.is_null()) {
        let centroids = image("shN", 0, 2)?;
        let label_img = image("shN", 1, 2)?;
        per_splat(&label_img, "shN labels")?;
        let coeffs_for = |bands: usize| (bands + 1) * (bands + 1) - 1;
        // Bands come from the metadata when present (an out-of-range value
        // means the file is read without SH), otherwise from the image width.
        let bands = match sh_n.get("bands").and_then(Value::as_u64) {
            Some(b) if (1..=4).contains(&b) => Some(b as usize),
            Some(_) => None,
            None => (1..=4).find(|&b| 64 * coeffs_for(b) == centroids.width),
        };
        if let Some(bands) = bands {
            let coeffs = coeffs_for(bands);
            if centroids.width != 64 * coeffs {
                return Err(err(format!(
                    "SH palette image is {} pixels wide; {bands} bands need {}",
                    centroids.width,
                    64 * coeffs
                )));
            }
            sh_degree = bands.min(3) as u8;
            let kept = sh_rest_coeffs(sh_degree);
            palette_len = sh_n
                .get("count")
                .and_then(Value::as_u64)
                .map_or(64 * centroids.height, |c| c as usize)
                .min(64 * centroids.height);
            let value_of: ByteDecoder = if version == 1 {
                let lo = sh_n
                    .get("mins")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| err("'shN.mins' is missing"))? as f32;
                let hi = sh_n
                    .get("maxs")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| err("'shN.maxs' is missing"))? as f32;
                Box::new(move |b| lo + (hi - lo) * f32::from(b) / 255.0)
            } else {
                let book = codebook(sh_n, "shN")?;
                Box::new(move |b| book[usize::from(b)])
            };
            palette.reserve(palette_len * kept * 3);
            for entry in 0..palette_len {
                for c in 0..kept {
                    let u = (entry % 64) * coeffs + c;
                    let v = entry / 64;
                    let px = centroids.pixel(v * centroids.width + u);
                    palette.extend_from_slice(&[value_of(px[0]), value_of(px[1]), value_of(px[2])]);
                }
            }
            labels = (0..count)
                .map(|i| {
                    let px = label_img.pixel(i);
                    let mut label = usize::from(px[0]) | (usize::from(px[1]) << 8);
                    if version >= 3 {
                        label |= usize::from(px[2]) << 16;
                    }
                    label
                })
                .collect();
        }
    }

    let mut cloud = SplatCloud::with_capacity(sh_degree, count.min(1 << 16));
    cloud.antialiased = meta.get("model").and_then(Value::as_str) == Some("antialiased");
    let stride = cloud.sh_rest_stride();
    let zeros = vec![0.0f32; stride];
    for i in 0..count {
        let (l, u) = (means_l.pixel(i), means_u.pixel(i));
        let position = std::array::from_fn(|a| {
            let q = f32::from(u16::from(l[a]) | (u16::from(u[a]) << 8)) / 65535.0;
            let span = if maxs[a] - mins[a] == 0.0 {
                1.0
            } else {
                maxs[a] - mins[a]
            };
            let n = mins[a] + span * q;
            n.signum() * (n.abs().exp() - 1.0)
        });
        let qp = quats.pixel(i);
        let rotation = if version >= 3 {
            decode_expmap(qp)
        } else {
            decode_smallest_three(qp)
        };
        let sp = scales.pixel(i);
        let cp = sh0.pixel(i);
        let rest = labels
            .get(i)
            .filter(|&&label| label < palette_len)
            .map_or(&zeros[..], |&label| {
                &palette[label * stride..(label + 1) * stride]
            });
        cloud.push(SplatRef {
            position,
            log_scale: [0, 1, 2].map(|a| scale_of(a, sp[a])),
            rotation,
            opacity_logit: opacity_of(cp[3]),
            sh_dc: [0, 1, 2].map(|c| dc_of(c, cp[c])),
            sh_rest: rest,
        });
    }
    Ok(cloud)
}

fn decode_smallest_three(px: [u8; 4]) -> [f32; 4] {
    if px[3] < 252 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let missing = usize::from(px[3] - 252);
    let vals = [0, 1, 2].map(|k| (f32::from(px[k]) / 255.0 - 0.5) * SQRT2);
    let mut q = [0.0f32; 4];
    let mut k = 0;
    for (i, slot) in q.iter_mut().enumerate() {
        if i != missing {
            *slot = vals[k];
            k += 1;
        }
    }
    q[missing] = (1.0 - vals.iter().map(|v| v * v).sum::<f32>())
        .max(0.0)
        .sqrt();
    q
}

fn decode_expmap(px: [u8; 4]) -> [f32; 4] {
    let p = [0, 1, 2].map(|k| (f32::from(px[k]) - 128.0) / 127.0);
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let t = HALF_PI * r;
    let k = if r > 1e-12 { t.sin() / r } else { HALF_PI };
    [t.cos(), p[0] * k, p[1] * k, p[2] * k]
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Writes `cloud` as a bundled `.sog` archive (version 2).
pub fn write_sog<W: Write>(cloud: &SplatCloud, writer: W) -> Result<()> {
    write_sog_with(cloud, writer, SogOptions::default())
}

/// Writes `cloud` as a bundled `.sog` archive with explicit options.
pub fn write_sog_with<W: Write>(
    cloud: &SplatCloud,
    mut writer: W,
    options: SogOptions,
) -> Result<()> {
    let files = encode(cloud, options)?;
    writer.write_all(&write_zip(&files)?)?;
    writer.flush()?;
    Ok(())
}

/// Writes an unbundled SOG scene: `meta_path` (normally `meta.json`) plus
/// image files in the same folder.
pub fn write_sog_dir(
    cloud: &SplatCloud,
    meta_path: impl AsRef<Path>,
    options: SogOptions,
) -> Result<()> {
    let meta_path = meta_path.as_ref();
    let dir = meta_path.parent().unwrap_or(Path::new("."));
    for (name, data) in encode(cloud, options)? {
        let path = if name == META {
            meta_path.to_path_buf()
        } else {
            dir.join(name)
        };
        std::fs::write(path, data)?;
    }
    Ok(())
}

fn encode_smallest_three(q: [f32; 4]) -> [u8; 4] {
    let mut q = normalize_quaternion(q);
    let mut largest = 0;
    for i in 1..4 {
        if q[i].abs() > q[largest].abs() {
            largest = i;
        }
    }
    let s = if q[largest] < 0.0 { -SQRT2 } else { SQRT2 };
    q = q.map(|v| v * s);
    let mut px = [0, 0, 0, 252 + largest as u8];
    let mut k = 0;
    for (i, &v) in q.iter().enumerate() {
        if i != largest {
            px[k] = (255.0 * (v * 0.5 + 0.5)).round().clamp(0.0, 255.0) as u8;
            k += 1;
        }
    }
    px
}

fn encode_expmap(q: [f32; 4]) -> [u8; 4] {
    let len = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !(len > 0.0 && len.is_finite()) {
        return [128, 128, 128, 255];
    }
    let mut q = q.map(|v| v / len);
    if q[0] < 0.0 {
        q = q.map(|v| -v);
    }
    let sv = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let k = if sv > 1e-12 {
        sv.atan2(q[0]) / sv / HALF_PI
    } else {
        1.0 / HALF_PI
    };
    let b = |v: f32| (127.0 * v * k + 128.0).round().clamp(0.0, 255.0) as u8;
    [b(q[1]), b(q[2]), b(q[3]), 255]
}

fn log_transform(x: f32) -> f32 {
    x.signum() * (x.abs() + 1.0).ln()
}

fn texture_size(n: usize) -> (usize, usize) {
    let w = (((n as f64).sqrt() / 4.0).ceil() as usize * 4).max(4);
    let h = ((n as f64 / w as f64 / 4.0).ceil() as usize * 4).max(4);
    (w, h)
}

fn default_palette_size(n: usize) -> usize {
    if n <= 1024 {
        return n;
    }
    let ratio = n / 1024;
    let pow = 1usize << (usize::BITS - 1 - ratio.leading_zeros());
    pow.min(64) * 1024
}

fn encode(cloud: &SplatCloud, options: SogOptions) -> Result<Vec<(String, Vec<u8>)>> {
    cloud.validate()?;
    if !(2..=3).contains(&options.version) {
        return Err(Error::Unsupported(format!(
            "sog: writing version {} (use 2 or 3)",
            options.version
        )));
    }
    let mut sorted = cloud.clone();
    transform::sort_morton(&mut sorted);
    let cloud = &sorted;
    let n = cloud.len();
    let (w, h) = texture_size(n);
    if w > 16383 || h > 16383 {
        return Err(Error::Invalid("too many splats for SOG images".into()));
    }
    let blank = || Image {
        width: w,
        height: h,
        rgba: vec![0u8; w * h * 4],
    };
    let set =
        |img: &mut Image, i: usize, px: [u8; 4]| img.rgba[i * 4..i * 4 + 4].copy_from_slice(&px);

    // Positions.
    let mut mins = [f32::INFINITY; 3];
    let mut maxs = [f32::NEG_INFINITY; 3];
    for p in &cloud.positions {
        for a in 0..3 {
            let t = log_transform(p[a]);
            if t.is_finite() {
                mins[a] = mins[a].min(t);
                maxs[a] = maxs[a].max(t);
            }
        }
    }
    for a in 0..3 {
        if !mins[a].is_finite() {
            (mins[a], maxs[a]) = (0.0, 0.0);
        }
    }
    let (mut means_l, mut means_u) = (blank(), blank());
    for (i, p) in cloud.positions.iter().enumerate() {
        let q: [u16; 3] = std::array::from_fn(|a| {
            let span = maxs[a] - mins[a];
            if span > 0.0 {
                (65535.0 * (log_transform(p[a]) - mins[a]) / span)
                    .round()
                    .clamp(0.0, 65535.0) as u16
            } else {
                0
            }
        });
        set(&mut means_l, i, [q[0] as u8, q[1] as u8, q[2] as u8, 255]);
        set(
            &mut means_u,
            i,
            [(q[0] >> 8) as u8, (q[1] >> 8) as u8, (q[2] >> 8) as u8, 255],
        );
    }

    // Rotations.
    let mut quats = blank();
    for (i, q) in cloud.rotations.iter().enumerate() {
        let px = if options.version >= 3 {
            encode_expmap(*q)
        } else {
            encode_smallest_three(*q)
        };
        set(&mut quats, i, px);
    }

    // Scales and base color through codebooks.
    let scale_book = codebook_1d(cloud.log_scales.as_flattened(), 256);
    let mut scales = blank();
    for (i, s) in cloud.log_scales.iter().enumerate() {
        let b = s.map(|v| nearest(&scale_book, v));
        set(&mut scales, i, [b[0], b[1], b[2], 255]);
    }
    let dc_book = codebook_1d(cloud.sh_dc.as_flattened(), 256);
    let mut sh0 = blank();
    for i in 0..n {
        let b = cloud.sh_dc[i].map(|v| nearest(&dc_book, v));
        let alpha = sigmoid(cloud.opacity_logits[i]);
        let a = if options.version >= 3 {
            if alpha > 0.0 {
                (204.0 * alpha.sqrt()).round().min(255.0) as u8
            } else {
                0
            }
        } else {
            (alpha.clamp(0.0, 1.0) * 255.0).round() as u8
        };
        set(&mut sh0, i, [b[0], b[1], b[2], a]);
    }

    let generator = concat!("splats ", env!("CARGO_PKG_VERSION"));
    let mut meta = json!({
        "version": options.version,
        "asset": { "generator": generator },
        "count": n,
        "means": { "mins": mins, "maxs": maxs, "files": ["means_l.webp", "means_u.webp"] },
        "scales": { "codebook": scale_book, "files": ["scales.webp"] },
        "quats": { "files": ["quats.webp"] },
        "sh0": { "codebook": dc_book, "files": ["sh0.webp"] },
    });
    if cloud.antialiased {
        meta["model"] = json!("antialiased");
    }
    let mut files = vec![
        ("means_l.webp".to_string(), encode_webp(&means_l)?),
        ("means_u.webp".to_string(), encode_webp(&means_u)?),
        ("quats.webp".to_string(), encode_webp(&quats)?),
        ("scales.webp".to_string(), encode_webp(&scales)?),
        ("sh0.webp".to_string(), encode_webp(&sh0)?),
    ];

    // Higher-order SH palette.
    if cloud.sh_degree > 0 && n > 0 {
        let coeffs = sh_rest_coeffs(cloud.sh_degree);
        let stride = coeffs * 3;
        // Palette images stay within 4096 rows so they load on WebGL2.
        let limit = if options.version >= 3 {
            64 * 4096
        } else {
            1 << 16
        };
        let k = options
            .palette_size
            .unwrap_or_else(|| default_palette_size(n))
            .clamp(1, limit)
            .min(n);
        let clean: Vec<f32> = cloud
            .sh_rest
            .iter()
            .map(|v| if v.is_finite() { *v } else { 0.0 })
            .collect();
        let (centroids, labels) = kmeans(&clean, stride, k, options.iterations);
        let entries = centroids.len() / stride;
        let book = codebook_1d(&centroids, 256);
        let rows = entries.div_ceil(64).max(1);
        let mut palette = Image {
            width: 64 * coeffs,
            height: rows,
            rgba: vec![0u8; 64 * coeffs * rows * 4],
        };
        for e in 0..entries {
            for c in 0..coeffs {
                let u = (e % 64) * coeffs + c;
                let v = e / 64;
                let base = e * stride + c * 3;
                let px = [0, 1, 2].map(|ch| nearest(&book, centroids[base + ch]));
                let index = v * palette.width + u;
                set(&mut palette, index, [px[0], px[1], px[2], 255]);
            }
        }
        let mut label_img = blank();
        for (i, &l) in labels.iter().enumerate() {
            let b = if options.version >= 3 {
                (l >> 16) as u8
            } else {
                0
            };
            set(&mut label_img, i, [l as u8, (l >> 8) as u8, b, 255]);
        }
        meta["shN"] = json!({
            "count": entries,
            "bands": cloud.sh_degree,
            "codebook": book,
            "files": ["shN_centroids.webp", "shN_labels.webp"],
        });
        files.push(("shN_centroids.webp".to_string(), encode_webp(&palette)?));
        files.push(("shN_labels.webp".to_string(), encode_webp(&label_img)?));
    }
    let meta = serde_json::to_vec(&meta).map_err(|e| Error::Invalid(e.to_string()))?;
    files.push((META.to_string(), meta));
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quaternion_encodings_roundtrip() {
        for q in [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, 0.5, -0.5],
            [0.1, 0.7, -0.2, 0.68],
            [-0.9, 0.1, 0.3, 0.2],
        ] {
            let q = normalize_quaternion(q);
            for back in [
                decode_smallest_three(encode_smallest_three(q)),
                decode_expmap(encode_expmap(q)),
            ] {
                let dot: f32 = q.iter().zip(&back).map(|(a, b)| a * b).sum();
                assert!(dot.abs() > 0.999, "{q:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn texture_and_palette_sizes() {
        assert_eq!(texture_size(0), (4, 4));
        assert_eq!(texture_size(100), (12, 12));
        assert_eq!(default_palette_size(500), 500);
        assert_eq!(default_palette_size(5000), 4096);
        assert_eq!(default_palette_size(10_000_000), 65536);
    }
}
