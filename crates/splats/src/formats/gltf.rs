//! glTF 2.0 scenes (`.glb` and `.gltf`) using the `KHR_gaussian_splatting`
//! extension.
//!
//! **Reading** collects every splat primitive in the default scene, applies
//! node transforms (translation, rotation and uniform scale, including
//! rotation of SH color), and merges them into one cloud. Attributes may use
//! any component type the extension allows. Buffers can be embedded, in the
//! GLB binary chunk, data URIs, or (via [`read_file`](crate::read_file))
//! files next to the `.gltf`. Primitives compressed with the SPZ
//! sub-extension are decoded too.
//!
//! **Writing** produces one node, one mesh and one primitive with `float`
//! attributes, plus a `COLOR_0` point-cloud fallback for viewers without
//! splat support.
//!
//! **Coordinates.** glTF is +Y up with +Z forward; splat training output is
//! usually +Y down with +Z forward. By default the reader and writer convert
//! between the two with an exact 180° turn about Z, so scenes appear upright
//! in glTF viewers. Set [`GltfOptions::convert_coordinates`] to `false` to
//! copy coordinates unchanged.

use std::borrow::Cow;
use std::io::{Read, Write};
use std::path::{Component, Path};

use serde_json::{Map, Value, json};

use crate::cloud::{SplatCloud, color_to_sh_dc, logit, normalize_quaternion, sigmoid};
use crate::error::{Error, Result};
use crate::math::{Mat3, mat_to_quat, quat_to_mat};
use crate::sh::{band_size, rest_index};
use crate::transform;
use crate::util::{Cursor, base64_decode, base64_encode, read_all};

const EXTENSION: &str = "KHR_gaussian_splatting";
const SPZ_EXTENSIONS: [&str; 2] = [
    "KHR_gaussian_splatting_compression_spz_2",
    "KHR_gaussian_splatting_compression_spz",
];
const ROTATION: &str = "KHR_gaussian_splatting:ROTATION";
const SCALE: &str = "KHR_gaussian_splatting:SCALE";
const OPACITY: &str = "KHR_gaussian_splatting:OPACITY";
const GLB_MAGIC: u32 = 0x4654_6C67;
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;
const MIN_SCALE: f32 = 1e-12;
/// Flip x and y: an exact 180° turn about Z between splat and glTF frames.
const FRAME_FLIP: [bool; 3] = [true, true, false];
/// Upper bound on splat primitive instances (guards against node graphs
/// that reuse children exponentially).
const MAX_INSTANCES: usize = 1 << 16;
/// Elements allowed for an accessor without data (it decodes to zeros).
const MAX_EMPTY_ELEMENTS: usize = 1 << 26;

fn sh_attribute(l: usize, n: usize) -> String {
    format!("KHR_gaussian_splatting:SH_DEGREE_{l}_COEF_{n}")
}

fn err(msg: impl std::fmt::Display) -> Error {
    Error::Parse(format!("gltf: {msg}"))
}

/// Options for reading and writing glTF splat scenes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GltfOptions {
    /// Convert between the splat frame (+Y down) and the glTF frame (+Y up)
    /// with a 180° turn about Z. Default `true`.
    pub convert_coordinates: bool,
}

impl Default for GltfOptions {
    fn default() -> Self {
        Self {
            convert_coordinates: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Reads a binary `.glb` file.
pub fn read_glb<R: Read>(reader: R) -> Result<SplatCloud> {
    read_glb_with(reader, None, GltfOptions::default())
}

/// Reads a JSON `.gltf` file. External buffer files are resolved against
/// `base_dir`; with `None`, only embedded data URIs are supported.
pub fn read_gltf<R: Read>(reader: R, base_dir: Option<&Path>) -> Result<SplatCloud> {
    read_gltf_with(reader, base_dir, GltfOptions::default())
}

/// [`read_glb`] with explicit options and an optional directory for
/// external buffers.
pub fn read_glb_with<R: Read>(
    reader: R,
    base_dir: Option<&Path>,
    options: GltfOptions,
) -> Result<SplatCloud> {
    let data = read_all(reader)?;
    let (json, bin) = split_glb(&data)?;
    decode_document(json, bin, base_dir, options)
}

/// [`read_gltf`] with explicit options.
pub fn read_gltf_with<R: Read>(
    reader: R,
    base_dir: Option<&Path>,
    options: GltfOptions,
) -> Result<SplatCloud> {
    let data = read_all(reader)?;
    decode_document(&data, None, base_dir, options)
}

fn split_glb(data: &[u8]) -> Result<(&[u8], Option<&[u8]>)> {
    let mut c = Cursor::new(data, "glb");
    if c.u32()? != GLB_MAGIC {
        return Err(err("not a GLB file"));
    }
    let version = c.u32()?;
    if version != 2 {
        return Err(Error::Unsupported(format!("glb: version {version}")));
    }
    let length = c.u32()? as usize;
    let data = &data[..length.min(data.len())];
    let mut c = Cursor::at(data, 12, "glb");
    let json_len = c.u32()? as usize;
    if c.u32()? != CHUNK_JSON {
        return Err(err("the first GLB chunk is not JSON"));
    }
    let json = c.take(json_len)?;
    let mut bin = None;
    while c.pos() + 8 <= data.len() {
        let len = c.u32()? as usize;
        let kind = c.u32()?;
        let chunk = c.take(len)?;
        if kind == CHUNK_BIN && bin.is_none() {
            bin = Some(chunk);
        }
    }
    Ok((json, bin))
}

fn uint(v: &Value, key: &str) -> Result<Option<usize>> {
    match v.get(key) {
        None => Ok(None),
        Some(x) => x
            .as_u64()
            .map(|n| Some(n as usize))
            .ok_or_else(|| err(format!("'{key}' must be a non-negative integer"))),
    }
}

fn item<'a>(doc: &'a Value, array: &str, index: usize) -> Result<&'a Value> {
    doc.get(array)
        .and_then(|a| a.get(index))
        .ok_or_else(|| err(format!("{array}[{index}] does not exist")))
}

fn numbers<const N: usize>(v: &Value, key: &str) -> Result<Option<[f64; N]>> {
    let Some(array) = v.get(key) else {
        return Ok(None);
    };
    let values: Option<Vec<f64>> = array
        .as_array()
        .map(|a| a.iter().map(Value::as_f64).collect())
        .unwrap_or(None);
    match values {
        Some(values) if values.len() == N => Ok(Some(values.try_into().expect("length checked"))),
        _ => Err(err(format!("'{key}' must be an array of {N} numbers"))),
    }
}

fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = s.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn load_buffers<'a>(
    doc: &Value,
    bin: Option<&'a [u8]>,
    base_dir: Option<&Path>,
) -> Result<Vec<Cow<'a, [u8]>>> {
    let Some(buffers) = doc.get("buffers").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(buffers.len());
    for (i, buffer) in buffers.iter().enumerate() {
        let length = uint(buffer, "byteLength")?
            .ok_or_else(|| err(format!("buffers[{i}] has no byteLength")))?;
        let data: Cow<'a, [u8]> = match buffer.get("uri").and_then(Value::as_str) {
            None => {
                Cow::Borrowed(bin.filter(|_| i == 0).ok_or_else(|| {
                    err(format!("buffers[{i}] has no uri and no GLB binary chunk"))
                })?)
            }
            Some(uri) if uri.starts_with("data:") => {
                let (_, payload) = uri
                    .split_once(";base64,")
                    .ok_or_else(|| err(format!("buffers[{i}] data URI is not base64")))?;
                Cow::Owned(
                    base64_decode(payload)
                        .ok_or_else(|| err(format!("buffers[{i}] has invalid base64")))?,
                )
            }
            Some(uri) => {
                let base = base_dir.ok_or_else(|| {
                    Error::Unsupported(format!(
                        "gltf: buffers[{i}] refers to external file '{uri}'; read the file from a path instead"
                    ))
                })?;
                let rel = percent_decode(uri)
                    .ok_or_else(|| err(format!("buffers[{i}] has an invalid uri")))?;
                let rel_path = Path::new(&rel);
                if rel.contains("://")
                    || !rel_path
                        .components()
                        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
                {
                    return Err(Error::Unsupported(format!(
                        "gltf: buffers[{i}] uri '{uri}' must be a relative path inside the scene's folder"
                    )));
                }
                Cow::Owned(std::fs::read(base.join(rel_path))?)
            }
        };
        if data.len() < length {
            return Err(err(format!(
                "buffers[{i}] holds {} bytes but declares {length}",
                data.len()
            )));
        }
        out.push(data);
    }
    Ok(out)
}

/// Column-major 4×4 matrix.
type Mat4 = [f64; 16];
const IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    std::array::from_fn(|i| {
        let (col, row) = (i / 4, i % 4);
        (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum()
    })
}

fn local_matrix(node: &Value) -> Result<Mat4> {
    if let Some(m) = numbers::<16>(node, "matrix")? {
        return Ok(m);
    }
    let t = numbers::<3>(node, "translation")?.unwrap_or([0.0; 3]);
    let [x, y, z, w] = numbers::<4>(node, "rotation")?.unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let s = numbers::<3>(node, "scale")?.unwrap_or([1.0; 3]);
    let r = quat_to_mat([w as f32, x as f32, y as f32, z as f32]);
    let mut m = IDENTITY;
    for col in 0..3 {
        for row in 0..3 {
            m[col * 4 + row] = r[row][col] * s[col];
        }
    }
    m[12..15].copy_from_slice(&t);
    Ok(m)
}

struct Instance {
    mesh: usize,
    primitive: usize,
    matrix: Mat4,
}

fn is_splat(primitive: &Value) -> bool {
    primitive
        .get("extensions")
        .and_then(|e| e.get(EXTENSION))
        .is_some()
}

fn push_mesh(doc: &Value, mesh: usize, matrix: &Mat4, out: &mut Vec<Instance>) -> Result<()> {
    let primitives = item(doc, "meshes", mesh)?
        .get("primitives")
        .and_then(Value::as_array)
        .ok_or_else(|| err(format!("meshes[{mesh}] has no primitives")))?;
    for (primitive, p) in primitives.iter().enumerate() {
        if is_splat(p) {
            if out.len() >= MAX_INSTANCES {
                return Err(err("too many splat primitive instances"));
            }
            out.push(Instance {
                mesh,
                primitive,
                matrix: *matrix,
            });
        }
    }
    Ok(())
}

fn visit(
    doc: &Value,
    node_index: usize,
    parent: &Mat4,
    depth: usize,
    out: &mut Vec<Instance>,
    visits: &mut usize,
) -> Result<()> {
    *visits += 1;
    if depth > 64 || *visits > MAX_INSTANCES * 4 {
        return Err(err(
            "node hierarchy is too deep, too large or contains a cycle",
        ));
    }
    let node = item(doc, "nodes", node_index)?;
    let matrix = mat4_mul(parent, &local_matrix(node)?);
    if let Some(mesh) = uint(node, "mesh")? {
        push_mesh(doc, mesh, &matrix, out)?;
    }
    if let Some(children) = node.get("children").and_then(Value::as_array) {
        for child in children {
            let child = child
                .as_u64()
                .ok_or_else(|| err("node children must be indices"))?;
            visit(doc, child as usize, &matrix, depth + 1, out, visits)?;
        }
    }
    Ok(())
}

fn splat_instances(doc: &Value) -> Result<Vec<Instance>> {
    let mut out = Vec::new();
    let mut visits = 0;
    match doc.get("scenes").and_then(Value::as_array) {
        Some(scenes) if !scenes.is_empty() => {
            let index = uint(doc, "scene")?.unwrap_or(0);
            let scene = scenes
                .get(index)
                .ok_or_else(|| err(format!("scenes[{index}] does not exist")))?;
            if let Some(roots) = scene.get("nodes").and_then(Value::as_array) {
                for root in roots {
                    let root = root
                        .as_u64()
                        .ok_or_else(|| err("scene nodes must be indices"))?;
                    visit(doc, root as usize, &IDENTITY, 0, &mut out, &mut visits)?;
                }
            }
        }
        _ => {
            let count = doc
                .get("meshes")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            for mesh in 0..count {
                push_mesh(doc, mesh, &IDENTITY, &mut out)?;
            }
        }
    }
    Ok(out)
}

/// A node transform reduced to what splats support exactly.
struct Similarity {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: f32,
}

fn decompose(m: &Mat4) -> Result<Option<Similarity>> {
    if m.iter().zip(&IDENTITY).all(|(a, b)| (a - b).abs() < 1e-9) {
        return Ok(None);
    }
    if (m[3], m[7], m[11], m[15]) != (0.0, 0.0, 0.0, 1.0) {
        return Err(Error::Unsupported(
            "gltf: projective node transforms".into(),
        ));
    }
    let cols: [[f64; 3]; 3] = std::array::from_fn(|c| [m[c * 4], m[c * 4 + 1], m[c * 4 + 2]]);
    let lens = cols.map(|c| (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt());
    let (lo, hi) = (
        lens.iter().copied().fold(f64::INFINITY, f64::min),
        lens.iter().copied().fold(0.0, f64::max),
    );
    if lo.is_nan() || lo <= 0.0 || hi / lo - 1.0 > 1e-4 {
        return Err(Error::Unsupported(
            "gltf: node transforms with non-uniform scale".into(),
        ));
    }
    let dot = |a: usize, b: usize| {
        (0..3).map(|i| cols[a][i] * cols[b][i]).sum::<f64>() / (lens[a] * lens[b])
    };
    if dot(0, 1).abs() > 1e-4 || dot(0, 2).abs() > 1e-4 || dot(1, 2).abs() > 1e-4 {
        return Err(Error::Unsupported(
            "gltf: node transforms with shear".into(),
        ));
    }
    let r: Mat3 = std::array::from_fn(|row| std::array::from_fn(|col| cols[col][row] / lens[col]));
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    if det < 0.0 {
        return Err(Error::Unsupported("gltf: mirroring node transforms".into()));
    }
    let scale = ((lens[0] + lens[1] + lens[2]) / 3.0) as f32;
    let translation = [m[12] as f32, m[13] as f32, m[14] as f32];
    if !(scale > 0.0 && scale.is_finite()) || translation.iter().any(|t| !t.is_finite()) {
        return Err(Error::Unsupported(
            "gltf: node transform is out of range for 32-bit floats".into(),
        ));
    }
    Ok(Some(Similarity {
        translation,
        rotation: mat_to_quat(&r),
        scale,
    }))
}

/// Reads accessor `index` as floats; returns the values and the number of
/// components per element.
fn accessor(doc: &Value, buffers: &[Cow<'_, [u8]>], index: usize) -> Result<(Vec<f32>, usize)> {
    let acc = item(doc, "accessors", index)?;
    let count =
        uint(acc, "count")?.ok_or_else(|| err(format!("accessors[{index}] has no count")))?;
    let comps = match acc.get("type").and_then(Value::as_str) {
        Some("SCALAR") => 1,
        Some("VEC2") => 2,
        Some("VEC3") => 3,
        Some("VEC4") => 4,
        other => {
            return Err(err(format!(
                "accessors[{index}] has unsupported type {other:?}"
            )));
        }
    };
    if acc.get("sparse").is_some() {
        return Err(Error::Unsupported(format!(
            "gltf: sparse accessor accessors[{index}]"
        )));
    }
    let component_type = uint(acc, "componentType")?
        .ok_or_else(|| err(format!("accessors[{index}] has no componentType")))?;
    let normalized = acc
        .get("normalized")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let size = match component_type {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        5125 | 5126 => 4,
        t => {
            return Err(err(format!(
                "accessors[{index}] has unknown componentType {t}"
            )));
        }
    };
    let total = count
        .checked_mul(comps)
        .ok_or_else(|| err("accessor is too large"))?;
    let Some(view_index) = uint(acc, "bufferView")? else {
        if total > MAX_EMPTY_ELEMENTS {
            return Err(err(format!("accessors[{index}] is too large")));
        }
        return Ok((vec![0.0; total], comps));
    };
    let view = item(doc, "bufferViews", view_index)?;
    let buffer_index = uint(view, "buffer")?.ok_or_else(|| err("bufferView without buffer"))?;
    let buffer = buffers
        .get(buffer_index)
        .ok_or_else(|| err(format!("buffers[{buffer_index}] does not exist")))?;
    let view_offset = uint(view, "byteOffset")?.unwrap_or(0);
    let view_length =
        uint(view, "byteLength")?.ok_or_else(|| err("bufferView without byteLength"))?;
    let element = size * comps;
    let stride = uint(view, "byteStride")?.unwrap_or(element);
    let start = view_offset
        .checked_add(uint(acc, "byteOffset")?.unwrap_or(0))
        .ok_or_else(|| err(format!("accessors[{index}] has an invalid byteOffset")))?;
    let view_end = view_offset
        .checked_add(view_length)
        .filter(|&e| e <= buffer.len())
        .ok_or_else(|| err(format!("bufferViews[{view_index}] exceeds its buffer")))?;
    if count > 0 {
        let last = stride
            .checked_mul(count - 1)
            .and_then(|v| v.checked_add(start)?.checked_add(element))
            .ok_or_else(|| err("accessor is too large"))?;
        if last > view_end || stride < element {
            return Err(err(format!(
                "accessors[{index}] does not fit in its bufferView"
            )));
        }
    }
    let mut out = Vec::with_capacity(total);
    for i in 0..count {
        let base = start + i * stride;
        for c in 0..comps {
            let b = &buffer[base + c * size..base + (c + 1) * size];
            let v = match component_type {
                5120 => {
                    let v = f32::from(b[0] as i8);
                    if normalized { (v / 127.0).max(-1.0) } else { v }
                }
                5121 => {
                    let v = f32::from(b[0]);
                    if normalized { v / 255.0 } else { v }
                }
                5122 => {
                    let v = f32::from(i16::from_le_bytes([b[0], b[1]]));
                    if normalized {
                        (v / 32767.0).max(-1.0)
                    } else {
                        v
                    }
                }
                5123 => {
                    let v = f32::from(u16::from_le_bytes([b[0], b[1]]));
                    if normalized { v / 65535.0 } else { v }
                }
                5125 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            };
            out.push(v);
        }
    }
    Ok((out, comps))
}

/// Decodes one splat primitive, in the glTF frame.
fn decode_primitive(
    doc: &Value,
    buffers: &[Cow<'_, [u8]>],
    primitive: &Value,
) -> Result<SplatCloud> {
    let ext = &primitive["extensions"][EXTENSION];
    if let Some((name, spz)) = SPZ_EXTENSIONS.iter().find_map(|n| {
        ext.get("extensions")
            .and_then(|e| e.get(*n))
            .map(|v| (*n, v))
    }) {
        let view_index =
            uint(spz, "bufferView")?.ok_or_else(|| err(format!("{name} has no bufferView")))?;
        let bytes = buffer_view(doc, buffers, view_index)?;
        // Bring the payload from its stored axes (normally right-up-back)
        // to the training frame, then to glTF's left-up-forward.
        let (mut cloud, axes) = super::spz::read_stored(bytes)?;
        super::spz::to_training_frame(&mut cloud, axes);
        transform::flip_axes(&mut cloud, FRAME_FLIP);
        return Ok(cloud);
    }

    let attributes = primitive
        .get("attributes")
        .and_then(Value::as_object)
        .ok_or_else(|| err("splat primitive has no attributes"))?;
    let attr = |name: &str| -> Result<Option<usize>> {
        match attributes.get(name) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .map(|n| Some(n as usize))
                .ok_or_else(|| err(format!("attribute {name} must be an accessor index"))),
        }
    };
    let required = |name: &str| -> Result<usize> {
        attr(name)?.ok_or_else(|| err(format!("splat primitive is missing {name}")))
    };
    let read = |index: usize, comps: usize, name: &str| -> Result<Vec<f32>> {
        let (values, actual) = accessor(doc, buffers, index)?;
        if actual != comps {
            return Err(err(format!(
                "{name} must have {comps} components, found {actual}"
            )));
        }
        Ok(values)
    };

    let positions = read(required("POSITION")?, 3, "POSITION")?;
    let n = positions.len() / 3;
    let check = |values: &[f32], comps: usize, name: &str| -> Result<()> {
        if values.len() != n * comps {
            return Err(err(format!("{name} has a different count than POSITION")));
        }
        Ok(())
    };
    let rotations = read(required(ROTATION)?, 4, ROTATION)?;
    check(&rotations, 4, ROTATION)?;
    let scales = read(required(SCALE)?, 3, SCALE)?;
    check(&scales, 3, SCALE)?;
    let opacity = read(required(OPACITY)?, 1, OPACITY)?;
    check(&opacity, 1, OPACITY)?;
    let dc_name = sh_attribute(0, 0);
    let sh_dc: Vec<f32> = if let Some(index) = attr(&dc_name)? {
        let values = read(index, 3, &dc_name)?;
        check(&values, 3, &dc_name)?;
        values
    } else if let Some(index) = attr("COLOR_0")? {
        let (values, comps) = accessor(doc, buffers, index)?;
        if comps < 3 || values.len() != n * comps {
            return Err(err("COLOR_0 has an unexpected layout"));
        }
        values
            .chunks_exact(comps)
            .flat_map(|c| [c[0], c[1], c[2]])
            .map(|linear| color_to_sh_dc(linear_to_srgb(linear)))
            .collect()
    } else {
        return Err(err(format!("splat primitive is missing {dc_name}")));
    };

    let mut degree = 0u8;
    for l in 1..=3usize {
        let present = (0..band_size(l))
            .filter(|&k| attributes.contains_key(&sh_attribute(l, k)))
            .count();
        if present == 0 {
            break;
        }
        if present != band_size(l) {
            return Err(err(format!("SH degree {l} is only partly defined")));
        }
        degree = l as u8;
    }
    let mut cloud = SplatCloud::with_capacity(degree, n);
    let stride = cloud.sh_rest_stride();
    let mut rest = vec![0.0f32; n * stride];
    for l in 1..=degree as usize {
        for k in 0..band_size(l) {
            let name = sh_attribute(l, k);
            let values = read(required(&name)?, 3, &name)?;
            check(&values, 3, &name)?;
            let i = rest_index(l, k);
            for s in 0..n {
                rest[s * stride + i * 3..s * stride + i * 3 + 3]
                    .copy_from_slice(&values[s * 3..s * 3 + 3]);
            }
        }
    }
    for s in 0..n {
        let q = &rotations[s * 4..s * 4 + 4];
        cloud.push(crate::SplatRef {
            position: [positions[s * 3], positions[s * 3 + 1], positions[s * 3 + 2]],
            log_scale: [0, 1, 2].map(|a| scales[s * 3 + a].max(MIN_SCALE).ln()),
            rotation: [q[3], q[0], q[1], q[2]],
            opacity_logit: logit(opacity[s]),
            sh_dc: [sh_dc[s * 3], sh_dc[s * 3 + 1], sh_dc[s * 3 + 2]],
            sh_rest: &rest[s * stride..(s + 1) * stride],
        });
    }
    Ok(cloud)
}

fn buffer_view<'a>(doc: &Value, buffers: &'a [Cow<'_, [u8]>], index: usize) -> Result<&'a [u8]> {
    let view = item(doc, "bufferViews", index)?;
    let buffer_index = uint(view, "buffer")?.ok_or_else(|| err("bufferView without buffer"))?;
    let buffer = buffers
        .get(buffer_index)
        .ok_or_else(|| err(format!("buffers[{buffer_index}] does not exist")))?;
    let offset = uint(view, "byteOffset")?.unwrap_or(0);
    let length = uint(view, "byteLength")?.ok_or_else(|| err("bufferView without byteLength"))?;
    offset
        .checked_add(length)
        .and_then(|end| buffer.get(offset..end))
        .ok_or_else(|| err(format!("bufferViews[{index}] exceeds its buffer")))
}

fn decode_document(
    json: &[u8],
    bin: Option<&[u8]>,
    base_dir: Option<&Path>,
    options: GltfOptions,
) -> Result<SplatCloud> {
    let doc: Value = serde_json::from_slice(json).map_err(|e| err(format!("invalid JSON: {e}")))?;
    let buffers = load_buffers(&doc, bin, base_dir)?;
    let instances = splat_instances(&doc)?;
    if instances.is_empty() {
        return Err(err(format!("no mesh primitive uses {EXTENSION}")));
    }
    let mut parts = Vec::with_capacity(instances.len());
    for inst in &instances {
        let primitive = &item(&doc, "meshes", inst.mesh)?["primitives"][inst.primitive];
        let mut cloud = decode_primitive(&doc, &buffers, primitive)?;
        if let Some(t) = decompose(&inst.matrix)? {
            transform::scale_uniform(&mut cloud, t.scale);
            transform::rotate(&mut cloud, t.rotation);
            transform::translate(&mut cloud, t.translation);
        }
        parts.push(cloud);
    }
    let degree = parts.iter().map(|c| c.sh_degree).max().unwrap_or(0);
    let mut iter = parts.into_iter();
    let mut cloud = iter.next().expect("at least one part");
    cloud.set_sh_degree(degree);
    for mut part in iter {
        part.set_sh_degree(degree);
        cloud.append(&part)?;
    }
    if options.convert_coordinates {
        transform::flip_axes(&mut cloud, FRAME_FLIP);
    }
    Ok(cloud)
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Writes `cloud` as a binary `.glb` file.
pub fn write_glb<W: Write>(cloud: &SplatCloud, writer: W) -> Result<()> {
    write_glb_with(cloud, writer, GltfOptions::default())
}

/// Writes `cloud` as a self-contained `.gltf` file (buffer embedded as a
/// base64 data URI).
pub fn write_gltf<W: Write>(cloud: &SplatCloud, writer: W) -> Result<()> {
    write_gltf_with(cloud, writer, GltfOptions::default())
}

/// [`write_glb`] with explicit options.
pub fn write_glb_with<W: Write>(
    cloud: &SplatCloud,
    mut writer: W,
    options: GltfOptions,
) -> Result<()> {
    let (doc, bin) = encode(cloud, options)?;
    let mut json = serde_json::to_vec(&doc).map_err(|e| Error::Invalid(e.to_string()))?;
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let mut bin = bin;
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let total = u32::try_from(total)
        .map_err(|_| Error::Invalid("scene is too large for a GLB file (4 GiB limit)".into()))?;
    writer.write_all(&GLB_MAGIC.to_le_bytes())?;
    writer.write_all(&2u32.to_le_bytes())?;
    writer.write_all(&total.to_le_bytes())?;
    writer.write_all(&(json.len() as u32).to_le_bytes())?;
    writer.write_all(&CHUNK_JSON.to_le_bytes())?;
    writer.write_all(&json)?;
    writer.write_all(&(bin.len() as u32).to_le_bytes())?;
    writer.write_all(&CHUNK_BIN.to_le_bytes())?;
    writer.write_all(&bin)?;
    writer.flush()?;
    Ok(())
}

/// [`write_gltf`] with explicit options.
pub fn write_gltf_with<W: Write>(
    cloud: &SplatCloud,
    mut writer: W,
    options: GltfOptions,
) -> Result<()> {
    let (mut doc, bin) = encode(cloud, options)?;
    doc["buffers"][0]["uri"] = Value::String(format!(
        "data:application/octet-stream;base64,{}",
        base64_encode(&bin)
    ));
    serde_json::to_writer(&mut writer, &doc).map_err(|e| Error::Io(e.into()))?;
    writer.flush()?;
    Ok(())
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

struct Builder {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
    attributes: Map<String, Value>,
}

impl Builder {
    fn add(
        &mut self,
        name: &str,
        ty: &str,
        mut accessor: Value,
        bytes: &[u8],
        stride: Option<usize>,
    ) {
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let mut view = json!({ "buffer": 0, "byteOffset": offset, "byteLength": bytes.len(), "target": 34962 });
        if let Some(stride) = stride {
            view["byteStride"] = json!(stride);
        }
        self.views.push(view);
        accessor["bufferView"] = json!(self.views.len() - 1);
        accessor["type"] = json!(ty);
        self.accessors.push(accessor);
        self.attributes
            .insert(name.to_string(), json!(self.accessors.len() - 1));
    }

    fn add_f32(&mut self, name: &str, ty: &str, count: usize, values: impl Iterator<Item = f32>) {
        let bytes: Vec<u8> = values.flat_map(f32::to_le_bytes).collect();
        self.add(
            name,
            ty,
            json!({ "componentType": 5126, "count": count }),
            &bytes,
            None,
        );
    }
}

fn encode(cloud: &SplatCloud, options: GltfOptions) -> Result<(Value, Vec<u8>)> {
    cloud.validate()?;
    if cloud.is_empty() {
        return Err(Error::Invalid(
            "glTF cannot store an empty splat set (accessors need at least one element)".into(),
        ));
    }
    if cloud.iter().any(|s| !transform::is_finite(&s)) {
        return Err(Error::Invalid(
            "glTF forbids NaN and infinite values; remove them first (transform::remove_non_finite)"
                .into(),
        ));
    }
    let converted;
    let cloud = if options.convert_coordinates {
        let mut c = cloud.clone();
        transform::flip_axes(&mut c, FRAME_FLIP);
        converted = c;
        &converted
    } else {
        cloud
    };
    let n = cloud.len();
    let mut b = Builder {
        bin: Vec::new(),
        views: Vec::new(),
        accessors: Vec::new(),
        attributes: Map::new(),
    };
    let (min, max) = cloud.bounds().expect("not empty");
    b.add(
        "POSITION",
        "VEC3",
        json!({ "componentType": 5126, "count": n, "min": min, "max": max }),
        &cloud
            .positions
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
        None,
    );
    b.add_f32(
        ROTATION,
        "VEC4",
        n,
        cloud.rotations.iter().flat_map(|q| {
            let [w, x, y, z] = normalize_quaternion(*q);
            [x, y, z, w]
        }),
    );
    b.add_f32(
        SCALE,
        "VEC3",
        n,
        cloud.log_scales.iter().flatten().map(|v| v.exp()),
    );
    b.add_f32(
        OPACITY,
        "SCALAR",
        n,
        cloud.opacity_logits.iter().map(|&v| sigmoid(v)),
    );
    b.add_f32(
        &sh_attribute(0, 0),
        "VEC3",
        n,
        cloud.sh_dc.iter().flatten().copied(),
    );
    let stride = cloud.sh_rest_stride();
    for l in 1..=cloud.sh_degree as usize {
        for k in 0..band_size(l) {
            let i = rest_index(l, k);
            b.add_f32(
                &sh_attribute(l, k),
                "VEC3",
                n,
                (0..n).flat_map(|s| {
                    let base = s * stride + i * 3;
                    [
                        cloud.sh_rest[base],
                        cloud.sh_rest[base + 1],
                        cloud.sh_rest[base + 2],
                    ]
                }),
            );
        }
    }
    // Point-cloud fallback for viewers without splat support: linear RGB
    // plus opacity, as normalized bytes.
    let colors: Vec<u8> = cloud
        .iter()
        .flat_map(|s| {
            let c = s.color().map(|v| srgb_to_linear(v.clamp(0.0, 1.0)));
            [c[0], c[1], c[2], s.opacity()].map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        })
        .collect();
    b.add(
        "COLOR_0",
        "VEC4",
        json!({ "componentType": 5121, "normalized": true, "count": n }),
        &colors,
        Some(4),
    );

    let doc = json!({
        "asset": { "version": "2.0", "generator": concat!("splats ", env!("CARGO_PKG_VERSION")) },
        "extensionsUsed": [EXTENSION],
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{
            "primitives": [{
                "mode": 0,
                "attributes": Value::Object(b.attributes),
                "extensions": { EXTENSION: { "kernel": "ellipse", "colorSpace": "srgb_rec709_display" } }
            }]
        }],
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{ "byteLength": b.bin.len() }]
    });
    Ok((doc, b.bin))
}
