//! Error type shared by every reader, writer and transform.

use std::fmt;
use std::io;

/// Errors returned by this crate.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An I/O error from the underlying reader or writer.
    Io(io::Error),
    /// The input is malformed.
    Parse(String),
    /// The input is well-formed but uses something this crate does not support.
    Unsupported(String),
    /// A [`SplatCloud`](crate::SplatCloud) is internally inconsistent or cannot
    /// be represented in the requested format.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::Parse(msg) => f.write_str(msg),
            Error::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Error::Invalid(msg) => write!(f, "invalid splat data: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

/// Result alias using this crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
