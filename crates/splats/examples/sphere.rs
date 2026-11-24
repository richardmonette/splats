//! Writes a synthetic scene: a sphere of small splats colored by their
//! position, with a view-dependent sheen in the degree-1 SH band.
//!
//! ```text
//! cargo run --example sphere -- sphere.ply 200000
//! ```

use splats::{SplatCloud, SplatRef, color_to_sh_dc, logit};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "sphere.ply".into());
    let count: usize = args
        .next()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(100_000);

    let mut cloud = SplatCloud::with_capacity(1, count);
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    // Splat radius roughly matching the spacing between neighbours.
    let radius = (4.0 * std::f32::consts::PI / count.max(1) as f32).sqrt() * 0.6;
    for i in 0..count {
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / count as f32;
        let r = (1.0 - y * y).sqrt();
        let theta = golden * i as f32;
        let p = [r * theta.cos(), y, r * theta.sin()];
        let color = p.map(|v| 0.5 + 0.45 * v);
        // Degree-1 band: brighten when viewed along +x.
        let sheen = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3, -0.3, -0.3];
        cloud.push(SplatRef {
            position: p,
            log_scale: [radius.ln(), radius.ln(), (radius * 0.2).ln()],
            rotation: [1.0, 0.0, 0.0, 0.0],
            opacity_logit: logit(if y > 0.0 { 0.95 } else { 0.6 }),
            sh_dc: color.map(color_to_sh_dc),
            sh_rest: &sheen,
        });
    }
    splats::write_file(&cloud, &path)?;
    println!("wrote {count} splats to {path}");
    Ok(())
}
