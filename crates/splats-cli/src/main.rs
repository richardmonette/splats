//! `splats`: inspect, convert and clean up 3D Gaussian splat files.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use splats::formats::{gltf, ksplat, sog, spz};
use splats::{Format, SplatCloud, transform};

#[derive(Parser)]
#[command(
    name = "splats",
    version,
    about = "Inspect, convert and clean up 3D Gaussian splat files"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print a summary of a splat file
    Info {
        /// File to inspect
        input: PathBuf,
        #[command(flatten)]
        axes: AxesArgs,
    },
    /// Convert a splat file, optionally cleaning it up on the way
    Convert {
        /// File to read
        input: PathBuf,
        /// File to write; the format comes from its name (use meta.json for
        /// an unbundled SOG folder)
        output: PathBuf,
        #[command(flatten)]
        edits: Edits,
        #[command(flatten)]
        output_options: OutputOptions,
        #[command(flatten)]
        axes: AxesArgs,
    },
    /// List supported formats
    Formats,
}

#[derive(Args)]
struct AxesArgs {
    /// Keep SPZ and glTF coordinates exactly as stored instead of converting
    /// them to and from the training frame (+Y down)
    #[arg(long)]
    raw_axes: bool,
}

#[derive(Args)]
struct Edits {
    /// Drop splats with opacity below this value (0 to 1)
    #[arg(long, value_name = "ALPHA", value_parser = parse_unit)]
    min_opacity: Option<f32>,
    /// Keep only splats whose centers are inside this box
    #[arg(
        long,
        value_name = "MINX,MINY,MINZ,MAXX,MAXY,MAXZ",
        allow_hyphen_values = true,
        value_parser = parse_box
    )]
    crop: Option<CropBox>,
    /// Rotate the scene, e.g. `x:180` or `y:-90`; repeat to chain rotations
    #[arg(long, value_name = "AXIS:DEGREES", allow_hyphen_values = true, value_parser = parse_rotation)]
    rotate: Vec<[f32; 4]>,
    /// Negate axes, e.g. `y,z`
    #[arg(long, value_name = "AXES", value_parser = parse_axes)]
    flip: Option<[bool; 3]>,
    /// Scale the scene uniformly about the origin
    #[arg(long, value_name = "FACTOR", value_parser = parse_positive)]
    scale: Option<f32>,
    /// Reduce or pad spherical harmonics to this degree
    #[arg(long, value_name = "0-3", value_parser = clap::value_parser!(u8).range(0..=3))]
    sh_degree: Option<u8>,
    /// Sort splats from most to least visually significant
    #[arg(long)]
    sort: bool,
}

#[derive(Args)]
struct OutputOptions {
    /// SPZ format version to write (2, 3 or 4)
    #[arg(long, value_name = "N", default_value_t = 3, value_parser = clap::value_parser!(u32).range(2..=4))]
    spz_version: u32,
    /// SPZ gzip effort for versions 2 and 3: 1 (fastest) to 9 (smallest)
    #[arg(long, value_name = "N", default_value_t = 6, value_parser = clap::value_parser!(u8).range(1..=9))]
    spz_level: u8,
    /// KSPLAT compression level (0, 1 or 2)
    #[arg(long, value_name = "N", default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=2))]
    ksplat_level: u8,
    /// SOG format version to write (2 or 3)
    #[arg(long, value_name = "N", default_value_t = 2, value_parser = clap::value_parser!(u32).range(2..=3))]
    sog_version: u32,
    /// Number of SOG spherical-harmonic palette entries (default: from splat count)
    #[arg(long, value_name = "N")]
    sog_palette: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct CropBox {
    min: [f32; 3],
    max: [f32; 3],
}

fn parse_unit(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|_| format!("'{s}' is not a number"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("{v} is outside 0 to 1"))
    }
}

fn parse_positive(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|_| format!("'{s}' is not a number"))?;
    if v > 0.0 && v.is_finite() {
        Ok(v)
    } else {
        Err(format!("{v} must be positive"))
    }
}

fn parse_box(s: &str) -> Result<CropBox, String> {
    let values = s
        .split(',')
        .map(|t| {
            let t = t.trim();
            t.parse::<f32>()
                .map_err(|_| format!("'{t}' is not a number"))
        })
        .collect::<Result<Vec<f32>, String>>()?;
    let [x0, y0, z0, x1, y1, z1] = values[..] else {
        return Err(format!(
            "expected 6 comma-separated numbers, got {}",
            values.len()
        ));
    };
    let (min, max) = ([x0, y0, z0], [x1, y1, z1]);
    if (0..3).any(|a| min[a] > max[a]) {
        return Err("each minimum must be less than or equal to its maximum".into());
    }
    Ok(CropBox { min, max })
}

fn axis_vector(name: &str) -> Option<[f32; 3]> {
    match name.trim().to_ascii_lowercase().as_str() {
        "x" => Some([1.0, 0.0, 0.0]),
        "y" => Some([0.0, 1.0, 0.0]),
        "z" => Some([0.0, 0.0, 1.0]),
        _ => None,
    }
}

fn parse_rotation(s: &str) -> Result<[f32; 4], String> {
    let (axis, degrees) = s
        .split_once(':')
        .ok_or_else(|| format!("'{s}' should look like x:90"))?;
    let axis = axis_vector(axis).ok_or_else(|| format!("unknown axis '{axis}' (use x, y or z)"))?;
    let degrees: f32 = degrees
        .trim()
        .parse()
        .map_err(|_| format!("'{degrees}' is not a number"))?;
    Ok(transform::axis_angle(axis, degrees))
}

fn parse_axes(s: &str) -> Result<[bool; 3], String> {
    let mut flip = [false; 3];
    for name in s.split(',') {
        let axis =
            axis_vector(name).ok_or_else(|| format!("unknown axis '{name}' (use x, y or z)"))?;
        let i = axis.iter().position(|&v| v == 1.0).expect("unit axis");
        flip[i] = !flip[i];
    }
    Ok(flip)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Info { input, axes } => info(&input, &axes),
        Command::Convert {
            input,
            output,
            edits,
            output_options,
            axes,
        } => convert(&input, &output, &edits, &output_options, &axes),
        Command::Formats => {
            formats();
            Ok(())
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn read(path: &Path, format: Format, axes: &AxesArgs) -> splats::Result<SplatCloud> {
    if !axes.raw_axes {
        return splats::read_file_as(path, format);
    }
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let reader =
        || -> splats::Result<_> { Ok(BufReader::with_capacity(1 << 20, File::open(path)?)) };
    let raw_gltf = gltf::GltfOptions {
        convert_coordinates: false,
    };
    match format {
        Format::Spz => spz::read_spz_with(
            reader()?,
            spz::SpzReadOptions {
                convert_coordinates: false,
            },
        ),
        Format::Glb => gltf::read_glb_with(reader()?, dir, raw_gltf),
        Format::Gltf => gltf::read_gltf_with(reader()?, dir, raw_gltf),
        _ => splats::read_file_as(path, format),
    }
}

fn write(
    cloud: &SplatCloud,
    path: &Path,
    format: Format,
    options: &OutputOptions,
    axes: &AxesArgs,
) -> splats::Result<()> {
    let convert = !axes.raw_axes;
    let sog_options = sog::SogOptions {
        version: options.sog_version,
        palette_size: options.sog_palette,
        ..Default::default()
    };
    if format == Format::Sog
        && !path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("sog"))
    {
        return sog::write_sog_dir(cloud, path, sog_options);
    }
    let mut out = BufWriter::with_capacity(1 << 20, File::create(path)?);
    let gltf_options = gltf::GltfOptions {
        convert_coordinates: convert,
    };
    match format {
        Format::Spz => spz::write_spz_with(
            cloud,
            &mut out,
            spz::SpzWriteOptions {
                version: options.spz_version,
                convert_coordinates: convert,
                compression_level: options.spz_level,
                ..Default::default()
            },
        )?,
        Format::Ksplat => ksplat::write_ksplat_with(
            cloud,
            &mut out,
            ksplat::KsplatOptions {
                compression_level: options.ksplat_level,
                ..Default::default()
            },
        )?,
        Format::Sog => sog::write_sog_with(cloud, &mut out, sog_options)?,
        Format::Glb => gltf::write_glb_with(cloud, &mut out, gltf_options)?,
        Format::Gltf => gltf::write_gltf_with(cloud, &mut out, gltf_options)?,
        other => other.write(cloud, &mut out)?,
    }
    out.flush()?;
    Ok(())
}

fn load(path: &Path, axes: &AxesArgs) -> Result<(SplatCloud, Format, Duration), String> {
    let start = Instant::now();
    let context = |e: splats::Error| format!("{}: {e}", path.display());
    let format = Format::detect(path).map_err(context)?;
    let cloud = read(path, format, axes).map_err(context)?;
    Ok((cloud, format, start.elapsed()))
}

fn info(path: &Path, axes: &AxesArgs) -> Result<(), String> {
    let (cloud, format, elapsed) = load(path, axes)?;
    println!("file        {}", path.display());
    println!("format      {format} ({})", format.description());
    println!("splats      {}", group(cloud.len()));
    println!("sh degree   {}", cloud.sh_degree);
    if let Some((min, max)) = cloud.bounds() {
        println!("bounds      min {}  max {}", vec3(min), vec3(max));
    }
    if !cloud.is_empty() {
        let n = cloud.len() as f64;
        let mean = cloud.iter().map(|s| f64::from(s.opacity())).sum::<f64>() / n;
        let faint = cloud.iter().filter(|s| s.opacity() < 0.05).count();
        println!(
            "opacity     mean {mean:.3}, {:.1}% below 0.05",
            100.0 * faint as f64 / n
        );
        let mut sizes: Vec<f32> = cloud
            .iter()
            .map(|s| s.scale().into_iter().fold(0.0, f32::max))
            .collect();
        let mid = sizes.len() / 2;
        let (_, median, _) = sizes.select_nth_unstable_by(mid, f32::total_cmp);
        println!("size        median largest axis {median:.5}");
        let mut color_sum = [0.0f64; 3];
        let mut out_of_range = 0usize;
        for s in cloud.iter() {
            for (sum, c) in color_sum.iter_mut().zip(s.color()) {
                *sum += f64::from(c);
                if !(-0.05..=1.05).contains(&c) {
                    out_of_range += 1;
                }
            }
        }
        println!(
            "color       mean [{:.3}, {:.3}, {:.3}], {:.2}% of channels outside 0..1",
            color_sum[0] / n,
            color_sum[1] / n,
            color_sum[2] / n,
            100.0 * out_of_range as f64 / (3.0 * n)
        );
        let (lo, hi) = cloud
            .rotations
            .iter()
            .map(|q| q.iter().map(|v| v * v).sum::<f32>().sqrt())
            .fold((f32::INFINITY, 0.0f32), |(lo, hi), l| {
                (lo.min(l), hi.max(l))
            });
        println!("rotation    length {lo:.4} to {hi:.4}");
        let invalid = cloud.iter().filter(|s| !transform::is_finite(s)).count();
        println!("non-finite  {}", group(invalid));
    }
    println!("read in     {elapsed:.1?}");
    Ok(())
}

fn convert(
    input: &Path,
    output: &Path,
    edits: &Edits,
    options: &OutputOptions,
    axes: &AxesArgs,
) -> Result<(), String> {
    let out_format = Format::from_path(output).ok_or_else(|| {
        format!(
            "{}: unknown output format (supported: {})",
            output.display(),
            Format::extension_list()
        )
    })?;
    let (mut cloud, in_format, elapsed) = load(input, axes)?;
    println!(
        "read  {} splats from {} ({in_format}) in {elapsed:.1?}",
        group(cloud.len()),
        input.display()
    );
    report(
        transform::remove_non_finite(&mut cloud),
        "with non-finite values",
    );
    if let Some(alpha) = edits.min_opacity {
        report(
            transform::filter_min_opacity(&mut cloud, alpha),
            &format!("with opacity below {alpha}"),
        );
    }
    if let Some(b) = edits.crop {
        report(
            transform::crop_box(&mut cloud, b.min, b.max),
            "outside the crop box",
        );
    }
    if let Some(flip) = edits.flip {
        transform::flip_axes(&mut cloud, flip);
        println!("      flipped axes");
    }
    for q in &edits.rotate {
        transform::rotate(&mut cloud, *q);
    }
    if !edits.rotate.is_empty() {
        println!("      rotated");
    }
    if let Some(factor) = edits.scale {
        transform::scale_uniform(&mut cloud, factor);
        println!("      scaled by {factor}");
    }
    if let Some(degree) = edits.sh_degree {
        if degree != cloud.sh_degree {
            println!("      SH degree {} -> {degree}", cloud.sh_degree);
            cloud.set_sh_degree(degree);
        }
    }
    if cloud.sh_degree > out_format.max_sh_degree() {
        println!(
            "      note: .{} stores SH up to degree {}; higher bands are dropped",
            out_format.extension(),
            out_format.max_sh_degree()
        );
    }
    if edits.sort {
        transform::sort_by_importance(&mut cloud);
        println!("      sorted by importance");
    }
    let start = Instant::now();
    write(&cloud, output, out_format, options, axes)
        .map_err(|e| format!("{}: {e}", output.display()))?;
    println!(
        "wrote {} splats to {} ({out_format}) in {:.1?}",
        group(cloud.len()),
        output.display(),
        start.elapsed()
    );
    Ok(())
}

fn formats() {
    println!("{:<16} {:<9} description", "extension", "max SH");
    for format in Format::ALL {
        println!(
            "{:<16} {:<9} {}",
            format!(".{}", format.extension()),
            format.max_sh_degree(),
            format.description()
        );
    }
    println!(
        "{:<16} {:<9} unbundled SOG (images beside it)",
        "meta.json", 3
    );
}

fn report(removed: usize, reason: &str) {
    if removed > 0 {
        println!("      removed {} splats {reason}", group(removed));
    }
}

fn group(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn vec3(v: [f32; 3]) -> String {
    format!("[{:.3}, {:.3}, {:.3}]", v[0], v[1], v[2])
}
