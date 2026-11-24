//! Times writing and reading every format for a random scene with full
//! degree-3 spherical harmonics (a worst case for palette formats).
//!
//! ```text
//! cargo run --release --example bench -- 1000000
//! ```

use std::time::Instant;

use splats::{Format, SplatCloud, SplatRef, transform};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let count: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1_000_000);

    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let mut cloud = SplatCloud::with_capacity(3, count);
    let mut rest = [0.0f32; 45];
    for _ in 0..count {
        for v in &mut rest {
            *v = next() * 0.3;
        }
        let position = [next() * 20.0, next() * 5.0, next() * 20.0];
        let log_scale = [next() - 4.0, next() - 4.0, next() - 4.0];
        let rotation = [next(), next(), next(), next()];
        let opacity_logit = next() * 4.0;
        let sh_dc = [next(), next(), next()];
        cloud.push(SplatRef {
            position,
            log_scale,
            rotation,
            opacity_logit,
            sh_dc,
            sh_rest: &rest,
        });
    }
    transform::normalize_rotations(&mut cloud);

    println!("{count} splats, SH degree 3");
    println!(
        "{:<16} {:>12} {:>10} {:>10}",
        "format", "bytes", "write", "read"
    );
    for format in Format::ALL {
        let start = Instant::now();
        let mut bytes = Vec::new();
        format.write(&cloud, &mut bytes)?;
        let write = start.elapsed();
        let start = Instant::now();
        let back = format.read(bytes.as_slice())?;
        let read = start.elapsed();
        assert_eq!(back.len(), cloud.len());
        println!(
            "{:<16} {:>12} {:>10.1?} {:>10.1?}",
            format!(".{}", format.extension()),
            bytes.len(),
            write,
            read
        );
    }
    Ok(())
}
