//! Singular value decomposition via one-sided Jacobi rotations.
//!
//! Designed for dense educational use on small-to-medium matrices. Convergence
//! is typically rapid for well-conditioned inputs; pathological cases may hit
//! the sweep limit and return [`LinalgError::ConvergenceFailed`].

use crate::error::{LinalgError, LinalgResult};
use crate::matrix::Matrix;
use num_traits::Float;

/// Maximum Jacobi sweeps before declaring non-convergence.
const MAX_SWEEPS: usize = 64;

/// Economy SVD: `A ≈ U Σ Vᵀ`.
///
/// Returns `(U, singular_values, Vᵀ)` where:
/// - `U` is `m × k` with orthonormal columns,
/// - `singular_values` has length `k = min(m, n)`, sorted **descending**,
/// - `Vᵀ` is `k × n` (so `V` would be `n × k`).
///
/// Reconstruction: `U * diag(σ) * Vᵀ ≈ A`.
///
/// # Algorithm
///
/// One-sided Jacobi SVD: right rotations orthogonalize the columns of a working
/// copy of `A` (or `Aᵀ` when `m < n`); singular values are the resulting column
/// norms. Suitable for dense matrices of modest size.
///
/// # Errors
///
/// - [`LinalgError::EmptyMatrix`] if `A` is empty.
/// - [`LinalgError::ConvergenceFailed`] if the sweep limit is exceeded.
pub fn svd<T: Float>(a: &Matrix<T>) -> LinalgResult<(Matrix<T>, Vec<T>, Matrix<T>)> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Err(LinalgError::EmptyMatrix);
    }
    if m >= n { svd_tall(a) } else { svd_wide(a) }
}

/// SVD for `m >= n` (tall or square).
fn svd_tall<T: Float>(a: &Matrix<T>) -> LinalgResult<(Matrix<T>, Vec<T>, Matrix<T>)> {
    let (m, n) = a.shape();
    debug_assert!(m >= n);

    let mut b = a.clone();
    let mut v = Matrix::identity(n);

    let tol = T::epsilon() * T::from(m.max(n)).unwrap_or_else(T::one);
    // TODO: fix the convergence test in the sweep loop below. `max_off` is a dot
    // product (scales like |A|^2) but is compared with `tol * ||B||_F` (scales like
    // |A|), so any matrix with entries around 1e3 or larger never converges:
    // svd([[1,2,3],[4,5,6],[7,8,10]] * 1000) returns ConvergenceFailed. This also
    // breaks nuclear_norm. Replace it with a `rotated` flag and stop once a full
    // sweep performs no rotation.
    //
    // TODO: also skip pairs where either column norm is below `tol * ||A||_F`.
    // On rank-deficient input one column collapses to rounding noise, which can
    // never pass the relative orthogonality test, so the sweeps stall without it.
    let mut converged = false;

    for _sweep in 0..MAX_SWEEPS {
        let mut max_off = T::zero();
        for p in 0..n {
            for q in (p + 1)..n {
                let (alpha, beta, gamma) = column_gram(&b, p, q, m);
                max_off = max_off.max(gamma.abs());
                // TODO: drop `.max(T::epsilon())` once the negligible-column skip above exists;
                // the pair test should be purely relative: |gamma| <= tol * sqrt(alpha * beta).
                let scale = (alpha * beta).sqrt().max(T::epsilon());
                if gamma.abs() <= tol * scale {
                    continue;
                }
                let (c, s) = jacobi_cs(alpha, beta, gamma);
                rotate_columns(&mut b, p, q, c, s, m);
                rotate_columns(&mut v, p, q, c, s, n);
            }
        }
        if max_off <= tol * frobenius_of(&b).max(T::one()) {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(LinalgError::ConvergenceFailed {
            iterations: MAX_SWEEPS,
        });
    }

    let mut sigma = vec![T::zero(); n];
    let mut u = Matrix::zeros(m, n);
    for j in 0..n {
        let mut nrm_sq = T::zero();
        for i in 0..m {
            nrm_sq = nrm_sq + b[(i, j)] * b[(i, j)];
        }
        let nrm = nrm_sq.sqrt();
        sigma[j] = nrm;
        // TODO: two problems here:
        // 1. The absolute `T::epsilon()` threshold is scale-dependent. Compare against
        //    `sigma_max * tol` instead (sort first, then build U).
        // 2. Columns under the threshold are left as zeros, so U does not have
        //    orthonormal columns for rank-deficient input, even though the svd() docs
        //    say it does. Fill those columns with an orthonormal completion: take the
        //    standard basis vector with the largest residual after two Gram-Schmidt
        //    passes against the earlier columns, then normalize it.
        if nrm > T::epsilon() {
            for i in 0..m {
                u[(i, j)] = b[(i, j)] / nrm;
            }
        }
    }

    sort_svd_descending(&mut u, &mut sigma, &mut v);
    Ok((u, sigma, v.transpose()))
}

/// SVD for wide matrices (`m < n`) via transpose.
fn svd_wide<T: Float>(a: &Matrix<T>) -> LinalgResult<(Matrix<T>, Vec<T>, Matrix<T>)> {
    let at = a.transpose();
    let (u_at, s, vt_at) = svd_tall(&at)?;
    // Aᵀ = U Σ Vᵀ  ⇒  A = V Σ Uᵀ, so U_A = V = vt_atᵀ, V_Aᵀ = U_atᵀ.
    Ok((vt_at.transpose(), s, u_at.transpose()))
}

fn column_gram<T: Float>(b: &Matrix<T>, p: usize, q: usize, m: usize) -> (T, T, T) {
    let mut alpha = T::zero();
    let mut beta = T::zero();
    let mut gamma = T::zero();
    for i in 0..m {
        let bp = b[(i, p)];
        let bq = b[(i, q)];
        alpha = alpha + bp * bp;
        beta = beta + bq * bq;
        gamma = gamma + bp * bq;
    }
    (alpha, beta, gamma)
}

/// Jacobi cosine/sine that diagonalizes the 2×2 Gram block `[[α,γ],[γ,β]]`.
fn jacobi_cs<T: Float>(alpha: T, beta: T, gamma: T) -> (T, T) {
    let two = T::from(2.0).unwrap_or_else(|| T::one() + T::one());
    let zeta = (beta - alpha) / (two * gamma);
    let t = if zeta >= T::zero() {
        T::one() / (zeta + (T::one() + zeta * zeta).sqrt())
    } else {
        -T::one() / (-zeta + (T::one() + zeta * zeta).sqrt())
    };
    let c = T::one() / (T::one() + t * t).sqrt();
    let s = t * c;
    (c, s)
}

/// Right-multiply columns `(p, q)` by `[[c, s], [-s, c]]`.
fn rotate_columns<T: Float>(mat: &mut Matrix<T>, p: usize, q: usize, c: T, s: T, rows: usize) {
    for i in 0..rows {
        let mp = mat[(i, p)];
        let mq = mat[(i, q)];
        mat[(i, p)] = c * mp - s * mq;
        mat[(i, q)] = s * mp + c * mq;
    }
}

fn frobenius_of<T: Float>(m: &Matrix<T>) -> T {
    m.as_slice()
        .iter()
        .fold(T::zero(), |acc, &x| acc + x * x)
        .sqrt()
}

/// Sort singular values descending; permute U and V columns to match.
fn sort_svd_descending<T: Float>(u: &mut Matrix<T>, sigma: &mut [T], v: &mut Matrix<T>) {
    let k = sigma.len();
    let m = u.nrows();
    let n = v.nrows();
    for i in 0..k {
        let mut best = i;
        for j in (i + 1)..k {
            if sigma[j] > sigma[best] {
                best = j;
            }
        }
        if best != i {
            sigma.swap(i, best);
            swap_columns(u, i, best, m);
            swap_columns(v, i, best, n);
        }
    }
}

fn swap_columns<T: Float>(mat: &mut Matrix<T>, i: usize, j: usize, rows: usize) {
    for row in 0..rows {
        let tmp = mat[(row, i)];
        mat[(row, i)] = mat[(row, j)];
        mat[(row, j)] = tmp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::matmul;

    fn reconstruct(u: &Matrix<f64>, s: &[f64], vt: &Matrix<f64>) -> Matrix<f64> {
        let k = s.len();
        let mut us = u.clone();
        for j in 0..k {
            for i in 0..us.nrows() {
                us[(i, j)] *= s[j];
            }
        }
        matmul(&us, vt).unwrap()
    }

    #[test]
    fn svd_diagonal() {
        let a = Matrix::new(2, 2, vec![3.0, 0.0, 0.0, 2.0]).unwrap();
        let (u, s, vt) = svd(&a).unwrap();
        assert!((s[0] - 3.0).abs() < 1e-8);
        assert!((s[1] - 2.0).abs() < 1e-8);
        let r = reconstruct(&u, &s, &vt);
        for i in 0..2 {
            for j in 0..2 {
                assert!((r[(i, j)] - a[(i, j)]).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn svd_known_2x2() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let (u, s, vt) = svd(&a).unwrap();
        assert!(s[0] >= s[1]);
        assert!(s[1] >= 0.0);
        let r = reconstruct(&u, &s, &vt);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (r[(i, j)] - a[(i, j)]).abs() < 1e-6,
                    "recon[{i},{j}]={} vs {}",
                    r[(i, j)],
                    a[(i, j)]
                );
            }
        }
        let utu = matmul(&u.transpose(), &u).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((utu[(i, j)] - e).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn svd_tall_and_wide() {
        let tall = Matrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let (u, s, vt) = svd(&tall).unwrap();
        assert_eq!(u.shape(), (3, 2));
        assert_eq!(s.len(), 2);
        assert_eq!(vt.shape(), (2, 2));
        let r = reconstruct(&u, &s, &vt);
        for i in 0..3 {
            for j in 0..2 {
                assert!((r[(i, j)] - tall[(i, j)]).abs() < 1e-6);
            }
        }

        let wide = tall.transpose();
        let (u2, s2, vt2) = svd(&wide).unwrap();
        assert_eq!(u2.shape(), (2, 2));
        assert_eq!(s2.len(), 2);
        assert_eq!(vt2.shape(), (2, 3));
        let r2 = reconstruct(&u2, &s2, &vt2);
        for i in 0..2 {
            for j in 0..3 {
                assert!((r2[(i, j)] - wide[(i, j)]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn svd_identity() {
        let a = Matrix::<f64>::identity(3);
        let (_u, s, _vt) = svd(&a).unwrap();
        for si in &s {
            assert!((*si - 1.0).abs() < 1e-10);
        }
    }
}
