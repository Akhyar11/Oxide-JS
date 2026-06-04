#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

mod dot_product;
mod lif;
mod surrogate;
mod delta;
mod embedding;

pub use dot_product::*;
pub use lif::*;
pub use surrogate::*;
pub use delta::*;
pub use embedding::*;
