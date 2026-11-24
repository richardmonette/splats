# splats

Read, write, convert and clean up 3D Gaussian splat scenes, in Rust.

- **Every common splat format**, read and write: `.ply`, `.compressed.ply`, `.splat`, `.spz`, `.ksplat`, `.sog` and glTF (`.glb` / `.gltf`) with the ratified `KHR_gaussian_splatting` extension.
- **Fast**: reads a million-splat scene in under a second in every binary format (see [Performance](#performance)).
- **Exact where it can be**: PLY round-trips bit for bit, axis conversions are exact sign flips, and every lossy format is tested against its quantization limits.
- **Scene editing**: remove invalid splats, filter by opacity, crop, rotate (including view-dependent color), mirror axes, scale, change SH degree, and sort by visual importance or spatial locality.
- **Safe and portable**: no `unsafe`, no C or C++ dependencies, so it builds anywhere Rust does, WebAssembly included. Damaged input returns an error instead of panicking.

## Command line

```sh
cargo install splats-cli
```

```sh
splats info scene.ply
splats convert scene.ply scene.spz
splats convert scene.ply scene.sog --min-opacity 0.02 --sh-degree 2
splats convert scan.spz scan.glb --crop -5,-5,-5,5,5,5 --rotate x:180
splats convert scene.ply web/meta.json        # SOG as loose files
splats formats
```

Format-specific choices: `--spz-version 2|3|4`, `--spz-level 1-9`, `--ksplat-level 0|1|2`, `--sog-version 2|3`, `--sog-palette N`. Run `splats convert --help` for every option.

## Library

```sh
cargo add splats
```

```rust
use splats::{transform, Format};

let mut cloud = splats::read_file("scene.ply")?;
transform::remove_non_finite(&mut cloud);
transform::filter_min_opacity(&mut cloud, 0.02);
transform::rotate(&mut cloud, transform::axis_angle([1.0, 0.0, 0.0], 180.0));
splats::write_file(&cloud, "scene.spz")?;

// Streams work too, and every format has an options variant.
let mut bytes = Vec::new();
Format::Glb.write(&cloud, &mut bytes)?;
```

`SplatCloud` stores splats as parallel arrays in the form training produces: positions, log scales, `[w, x, y, z]` rotations, opacity logits and spherical-harmonic coefficients up to degree 3. Each format converts on the way in and out.

## Formats

| Format | Read | Write | SH | Notes |
|---|:-:|:-:|:-:|---|
| `.ply` | ✓ | ✓ | 3 | Lossless. Reads ASCII and binary, any property order or type. |
| `.compressed.ply` | ✓ | ✓ | 3 | 16 bytes per splat plus 1 byte per SH coefficient, in chunks of 256. |
| `.splat` | ✓ | ✓ | 0 | 32 bytes per splat. Base color only. |
| `.spz` | ✓ | ✓ | 3 | Versions 1–4 read; 2, 3 (default) and 4 written. |
| `.ksplat` | ✓ | ✓ | 2 | Compression levels 0, 1 (default) and 2. |
| `.sog` | ✓ | ✓ | 3 | Versions 1–3 read; 2 (default) and 3 written. Bundled or as loose files. |
| `.glb` / `.gltf` | ✓ | ✓ | 3 | `KHR_gaussian_splatting`, including node transforms and SPZ-compressed primitives. Writes a point-cloud color fallback. |

Lossy formats keep as many SH bands as they support and drop the rest.

## Coordinates

Splat training usually produces +Y-down scenes. SPZ stores +Y up / +Z back and glTF stores +Y up / +Z forward, so by default those readers and writers convert with exact axis flips, and scenes appear upright in viewers. Pass `--raw-axes` (or the `convert_coordinates: false` options) to keep stored coordinates untouched. Other formats are copied as-is.

## Performance

One million splats with full degree-3 SH and random values (a worst case for compression), release build on Linux:

| Format | Size | Write | Read |
|---|--:|--:|--:|
| `.ply` | 248 MB | 0.2 s | 0.4 s |
| `.compressed.ply` | 61 MB | 1.1 s | 0.6 s |
| `.splat` | 32 MB | 0.2 s | 0.1 s |
| `.spz` | 36 MB | 8.0 s (1.8 s with `--spz-level 1`) | 0.6 s |
| `.ksplat` | 72 MB | 0.8 s | 0.8 s |
| `.sog` | 22 MB | 8.2 s (16 threads) | 0.4 s |
| `.glb` | 240 MB | 0.9 s | 0.7 s |

SOG writing spends its time clustering SH into a palette, and uses all cores. Reproduce with `cargo run --release --example bench`.

## Building and testing

```sh
cargo test --workspace
cargo run --release --example sphere -- sphere.ply 1000000
```

To check real-world files, point `SPLATS_SAMPLES` at a folder of them (loose SOG scenes go in subfolders with their `meta.json`). Every file is read and pushed through every writer:

```sh
SPLATS_SAMPLES=~/splat-samples cargo test --release --test samples -- --nocapture
```

## Roadmap

- WebAssembly package for JavaScript and TypeScript
- Python bindings
- Level-of-detail generation and spatial merging
- Streaming reads for scenes larger than memory

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
