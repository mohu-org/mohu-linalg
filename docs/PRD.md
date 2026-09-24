# mohu-linalg — Product Requirements (0.1)

## Goals

Provide a **pure-Rust, dependency-light, dense linear algebra** crate that:

1. Serves as the numerical backend for [mohu](https://github.com/mohu-org/mohu).
2. Is usable standalone for teaching, prototyping, and small/medium workloads.
3. Exposes a fallible, documented API (`LinalgResult`, `#![deny(missing_docs)]`).
4. Has zero `unsafe` and no BLAS/LAPACK requirement.

## Non-goals (0.1)

- Sparse matrix formats (CSR, CSC, COO).
- GPU / SIMD / BLAS acceleration.
- Iterative Krylov solvers (CG, GMRES, BiCGSTAB).
- General (non-symmetric / complex) eigenvalue decomposition.
- Out-of-core or distributed factorizations.
- Automatic differentiation hooks.

## Module map

| Module | Responsibility |
|--------|----------------|
| `matrix` | Dense row-major `Matrix<T: Float>` |
| `matmul` | Serial + Rayon-parallel GEMM |
| `lu` | LU with partial pivoting + square solve |
| `qr` | Thin Householder QR + least-squares |
| `cholesky` | SPD Cholesky + solve |
| `svd` | Economy one-sided Jacobi SVD |
| `eig` | Symmetric Jacobi `eigh` |
| `norms` | L1, L2/Frobenius, max, nuclear |
| `error` | `LinalgError` / `LinalgResult` |

## API surface (public)

```text
Matrix::new / zeros / identity / from_vec2d / transpose / get / set / row
matmul, matmul_parallel
lu_decompose, solve
qr_decompose, lstsq
cholesky, solve_cholesky
svd
eigh
l1_norm, l2_norm, frobenius_norm, max_norm, nuclear_norm
LinalgError { DimensionMismatch, SingularMatrix, NotSquare, EmptyMatrix,
              NotSymmetric, NotPositiveDefinite, ConvergenceFailed }
```

## Testing strategy

- **Unit tests** co-located in each module: known small matrices, reconstruction
  identities (`QR ≈ A`, `UΣVᵀ ≈ A`, `LLᵀ ≈ A`, `AV ≈ VΛ`), error paths.
- **Integration tests** under `tests/` (matmul identities, cross-routine smoke).
- **Quality gates** (CI): `cargo build/test/clippy -D warnings`, `cargo fmt --check`.
- **Benches**: Criterion microbench for matmul sizes (see `benches/`).

## Deferred items

| Item | Rationale |
|------|-----------|
| Sparse formats | Separate crate / later major version |
| BLAS/GPU backends | Feature-gated optional deps later |
| Krylov iterative solvers | Needs sparse + preconditioners |
| General `eig` | Needs complex Schur / Hessenberg QR |
| Rank-revealing QR / pivoted Cholesky | Nice-to-have for ill-conditioned lstsq |
| Condition estimators | Follow LAPACK `xGECON` style later |

## Success criteria (0.1)

- All routines in the feature table of `README.md` marked **Done** compile and
  pass unit + integration tests.
- Clippy `-D warnings` clean; rustfmt clean.
- README + this PRD accurately reflect scope and limitations.
