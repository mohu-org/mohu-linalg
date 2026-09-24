//! Pure-Rust dense linear algebra for the [mohu](https://github.com/mohu-org/mohu) stack.
//!
//! # Features
//!
//! - Dense row-major [`Matrix`]
//! - Matrix multiplication (serial and Rayon-parallel)
//! - Factorizations: LU, QR (Householder), Cholesky, economy SVD (Jacobi),
//!   symmetric eigendecomposition (Jacobi)
//! - Solvers: LU solve, Cholesky solve, least-squares via QR
//! - Norms: L1, L2/Frobenius, max, nuclear
//!
//! # Non-goals (this crate)
//!
//! Sparse formats, GPU/BLAS backends, and general (non-symmetric) eigenproblems
//! are out of scope for 0.1 — see `docs/PRD.md`.
//!
//! # Example
//!
//! ```
//! use mohu_linalg::{Matrix, matmul, solve, qr_decompose, svd, eigh, cholesky};
//!
//! let a = Matrix::new(2, 2, vec![4.0, 1.0, 1.0, 3.0]).unwrap();
//! let x = solve(&a, &[1.0, 2.0]).unwrap();
//! assert_eq!(x.len(), 2);
//!
//! let (q, r) = qr_decompose(&a).unwrap();
//! let _ = (q, r);
//! let (_u, _s, _vt) = svd(&a).unwrap();
//! let (_evals, _evecs) = eigh(&a).unwrap();
//! let _l = cholesky(&a).unwrap();
//! let _c = matmul(&a, &a).unwrap();
//! ```

#![deny(missing_docs)]

pub mod cholesky;
pub mod eig;
pub mod error;
pub mod lu;
pub mod matmul;
pub mod matrix;
pub mod norms;
pub mod qr;
pub mod svd;

pub use cholesky::{cholesky, solve_cholesky};
pub use eig::eigh;
pub use error::{LinalgError, LinalgResult};
pub use lu::{lu_decompose, solve};
pub use matmul::{matmul, matmul_parallel};
pub use matrix::Matrix;
pub use norms::{frobenius_norm, l1_norm, l2_norm, max_norm, nuclear_norm};
pub use qr::{lstsq, qr_decompose};
pub use svd::svd;
