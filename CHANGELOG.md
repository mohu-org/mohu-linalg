# Changelog

## 0.1.0 — 2026-09-22

Initial release of the full dense educational API:

- Dense `Matrix<T>` (row-major) with matmul (serial + Rayon)
- LU factorization with partial pivoting and square solve
- Thin QR (Householder) and least-squares (`lstsq`)
- Cholesky factorization / solve for SPD matrices
- Economy SVD via one-sided Jacobi
- Symmetric eigendecomposition (`eigh`) via classical Jacobi
- Norms: L1, L2/Frobenius, max, nuclear
- Documented error set including `NotPositiveDefinite`, `NotSymmetric`,
  `ConvergenceFailed`
