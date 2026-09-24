# mohu-linalg

Pure-Rust dense linear algebra — usable standalone or as the linalg backend for
[mohu](https://github.com/mohu-org/mohu).

No BLAS, no `unsafe`, no platform-specific asm. Educational but numerically
careful implementations of the classic dense factorizations.

## Features (0.1)

| Routine | Status | Notes |
|---------|--------|-------|
| Matrix type + matmul | **Done** | Serial + Rayon-parallel |
| LU + solve | **Done** | Partial pivoting |
| QR + least-squares | **Done** | Thin Householder QR; `lstsq` |
| Cholesky + solve | **Done** | SPD lower-triangular `L` |
| SVD | **Done** | Economy one-sided Jacobi |
| Symmetric eigen (`eigh`) | **Done** | Classical Jacobi |
| Norms L1 / L2 / Frobenius / max | **Done** | Entrywise |
| Nuclear norm | **Done** | Via SVD |
| General (non-symmetric) eigen | Deferred | See `docs/PRD.md` |
| Sparse / GPU / Krylov | Deferred | See `docs/PRD.md` |

## Quick example

```rust
use mohu_linalg::{Matrix, solve, qr_decompose, svd, eigh, cholesky, nuclear_norm};

fn main() -> mohu_linalg::LinalgResult<()> {
    let a = Matrix::new(2, 2, vec![4.0, 1.0, 1.0, 3.0])?;

    let x = solve(&a, &[1.0, 2.0])?;
    println!("LU solve: {x:?}");

    let (q, r) = qr_decompose(&a)?;
    let (_u, s, _vt) = svd(&a)?;
    let (evals, _evecs) = eigh(&a)?;
    let l = cholesky(&a)?;
    let nuc = nuclear_norm(&a)?;

    println!("σ = {s:?}, λ = {evals:?}, nuclear = {nuc}");
    let _ = (q, r, l);
    Ok(())
}
```

## Public API (summary)

| Item | Module | Description |
|------|--------|-------------|
| `Matrix<T>` | `matrix` | Dense row-major matrix |
| `matmul` / `matmul_parallel` | `matmul` | `C = A B` |
| `lu_decompose` / `solve` | `lu` | `PA = LU`, square solve |
| `qr_decompose` / `lstsq` | `qr` | Thin QR, least-squares |
| `cholesky` / `solve_cholesky` | `cholesky` | SPD factor / solve |
| `svd` | `svd` | Economy `U, σ, Vᵀ` |
| `eigh` | `eig` | Symmetric eigenpairs |
| `l1_norm`, `l2_norm`, `frobenius_norm`, `max_norm`, `nuclear_norm` | `norms` | Matrix norms |
| `LinalgError` / `LinalgResult` | `error` | Fallible API |

All fallible ops return `LinalgResult` — no `unwrap` in library code.

## Limitations

- **Dense only** — no sparse CSR/CSC.
- **Pure Rust** — no BLAS/LAPACK; expect O(n³) educational performance, not MKL.
- **Real floats** — `f32` / `f64` via `num_traits::Float`; no complex eigen yet.
- **`eigh` is symmetric-only** — general eigen is deferred.
- **SVD / Jacobi** — practical for small–medium dense matrices; very large or
  pathological inputs may hit `ConvergenceFailed`.
- **No `unsafe`** — intentional; enables easy auditability.

## Docs

- Product requirements: [`docs/PRD.md`](docs/PRD.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

## License

MIT
