//! Read, write and transform 3D Gaussian splat scenes.
//!
//! The central type is [`SplatCloud`], a structure-of-arrays container for
//! every splat's position, scale, rotation, opacity and spherical-harmonic
//! color. The [`formats`] modules convert it to and from files, and
//! [`transform`] filters, crops, sorts and edits it.
//!
//! ```
//! use splats::{SplatCloud, SplatRef};
//!
//! let mut cloud = SplatCloud::new(0);
//! cloud.push(SplatRef {
//!     position: [0.0, 1.0, 2.0],
//!     log_scale: [-4.0; 3],
//!     rotation: [1.0, 0.0, 0.0, 0.0],
//!     opacity_logit: 2.0,
//!     sh_dc: [0.0; 3],
//!     sh_rest: &[],
//! });
//!
//! let mut bytes = Vec::new();
//! splats::formats::ply::write_ply(&cloud, &mut bytes)?;
//! let back = splats::formats::ply::read_ply(bytes.as_slice())?;
//! assert_eq!(cloud, back);
//! # Ok::<(), splats::Error>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod cloud;
mod error;
pub mod formats;
mod math;
mod quantize;
mod sh;
pub mod transform;
mod util;
mod zip;

pub use cloud::{
    MAX_SH_DEGREE, SH_C0, SplatCloud, SplatRef, color_to_sh_dc, logit, normalize_quaternion,
    sh_dc_to_color, sh_rest_coeffs, sigmoid,
};
pub use error::{Error, Result};
pub use formats::{Format, read_file, read_file_as, write_file, write_file_as};
