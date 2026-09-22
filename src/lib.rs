//! Pure-Rust linear algebra: dense matrices, matmul, norms, and LU solve.
//!
//! This initial slice intentionally excludes SVD, eigenvalues, and QR.

#![deny(missing_docs)]

pub mod error;
pub mod lu;
pub mod matmul;
pub mod matrix;
pub mod norms;

pub use error::{LinalgError, LinalgResult};
pub use lu::{lu_decompose, solve};
pub use matmul::{matmul, matmul_parallel};
pub use matrix::Matrix;
pub use norms::{frobenius_norm, l1_norm, l2_norm, max_norm};
