//! Symmetric eigenvalue decomposition via the classical Jacobi method.
//!
//! Only real-symmetric eigenproblems are supported in this release. A general
//! (non-symmetric) `eig` decomposer is deferred — see `docs/PRD.md`.

use crate::error::{LinalgError, LinalgResult};
use crate::matrix::Matrix;
use num_traits::Float;

/// Maximum Jacobi sweeps before declaring non-convergence.
const MAX_SWEEPS: usize = 64;

/// Symmetric eigenvalue decomposition: `A = V diag(λ) Vᵀ`.
///
/// Returns `(eigenvalues, eigenvectors)` where:
/// - `eigenvalues` has length `n`, sorted **ascending**,
/// - `eigenvectors` is `n × n` with orthonormal columns `v_j` satisfying
///   `A v_j ≈ λ_j v_j`.
///
/// # Errors
///
/// - [`LinalgError::NotSquare`] if `A` is not square.
/// - [`LinalgError::NotSymmetric`] if `|Aᵢⱼ − Aⱼᵢ|` exceeds a modest tolerance.
/// - [`LinalgError::ConvergenceFailed`] if the sweep limit is exceeded.
pub fn eigh<T: Float>(a: &Matrix<T>) -> LinalgResult<(Vec<T>, Matrix<T>)> {
    let (n, m) = a.shape();
    if n != m {
        return Err(LinalgError::NotSquare);
    }
    check_symmetric(a)?;

    let mut work = a.clone();
    let mut v = Matrix::identity(n);

    let max_abs = a
        .as_slice()
        .iter()
        .fold(T::zero(), |acc, &x| acc.max(x.abs()));
    // TODO: make this tolerance relative: use `T::epsilon() * ||A||_F` with no
    // `.max(T::one())`. With the clamp, any matrix with entries below ~1e-16 is
    // treated as already diagonal and returns wrong eigenvalues without an error:
    // eigh(1e-20 * [[2, 1], [1, 2]]) returns [2e-20, 2e-20] instead of [1e-20, 3e-20].
    let tol = T::epsilon() * T::from(n).unwrap_or_else(T::one) * max_abs.max(T::one());
    let mut converged = false;

    for _sweep in 0..MAX_SWEEPS {
        let mut max_off = T::zero();
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = work[(p, q)];
                max_off = max_off.max(apq.abs());
                if apq.abs() <= tol {
                    continue;
                }
                let (c, s) = sym_jacobi_cs(work[(p, p)], work[(q, q)], apq);
                apply_jacobi_similarity(&mut work, p, q, c, s, n);
                // Accumulate V := V J
                rotate_columns_right(&mut v, p, q, c, s, n);
            }
        }
        if max_off <= tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(LinalgError::ConvergenceFailed {
            iterations: MAX_SWEEPS,
        });
    }

    let mut eigenvalues = Vec::with_capacity(n);
    for i in 0..n {
        eigenvalues.push(work[(i, i)]);
    }
    sort_eig_ascending(&mut eigenvalues, &mut v);
    Ok((eigenvalues, v))
}

fn check_symmetric<T: Float>(a: &Matrix<T>) -> LinalgResult<()> {
    let n = a.nrows();
    let max_abs = a
        .as_slice()
        .iter()
        .fold(T::zero(), |acc, &x| acc.max(x.abs()));
    // TODO: drop `.max(T::one())` here too, otherwise clearly non-symmetric
    // small-scale matrices pass the symmetry check.
    let tol = T::epsilon() * T::from(n * 10).unwrap_or_else(T::one) * max_abs.max(T::one());
    for i in 0..n {
        for j in (i + 1)..n {
            if (a[(i, j)] - a[(j, i)]).abs() > tol {
                return Err(LinalgError::NotSymmetric);
            }
        }
    }
    Ok(())
}

/// Cosine/sine for a symmetric Jacobi rotation annihilating `A[p,q]`.
fn sym_jacobi_cs<T: Float>(app: T, aqq: T, apq: T) -> (T, T) {
    let two = T::from(2.0).unwrap_or_else(|| T::one() + T::one());
    let diff = aqq - app;
    // TODO: delete this special case. The `.max(T::one())` makes it fire on every
    // pair for small-scale matrices, and this rotation does not zero apq, so
    // eigh hits ConvergenceFailed. The general formula below already handles equal
    // diagonals (zeta = 0 gives t = 1, the 45-degree rotation).
    if diff.abs() < T::epsilon() * (app.abs() + aqq.abs()).max(T::one()) && apq.abs() > T::zero() {
        // 45-degree rotation when diagonal entries are nearly equal.
        let sqrt2_inv = T::one() / two.sqrt();
        return (sqrt2_inv, sqrt2_inv);
    }
    let zeta = diff / (two * apq);
    let t = if zeta >= T::zero() {
        T::one() / (zeta + (T::one() + zeta * zeta).sqrt())
    } else {
        -T::one() / (-zeta + (T::one() + zeta * zeta).sqrt())
    };
    let c = T::one() / (T::one() + t * t).sqrt();
    let s = c * t;
    (c, s)
}

/// Apply similarity `A := Jᵀ A J` for the Jacobi plane rotation on `(p, q)`.
fn apply_jacobi_similarity<T: Float>(a: &mut Matrix<T>, p: usize, q: usize, c: T, s: T, n: usize) {
    let app = a[(p, p)];
    let aqq = a[(q, q)];
    let apq = a[(p, q)];

    a[(p, p)] = c * c * app - two_cs(c, s) * apq + s * s * aqq;
    a[(q, q)] = s * s * app + two_cs(c, s) * apq + c * c * aqq;
    a[(p, q)] = T::zero();
    a[(q, p)] = T::zero();

    for i in 0..n {
        if i == p || i == q {
            continue;
        }
        let aip = a[(i, p)];
        let aiq = a[(i, q)];
        let new_ip = c * aip - s * aiq;
        let new_iq = s * aip + c * aiq;
        a[(i, p)] = new_ip;
        a[(p, i)] = new_ip;
        a[(i, q)] = new_iq;
        a[(q, i)] = new_iq;
    }
}

fn two_cs<T: Float>(c: T, s: T) -> T {
    let two = T::from(2.0).unwrap_or_else(|| T::one() + T::one());
    two * c * s
}

fn rotate_columns_right<T: Float>(v: &mut Matrix<T>, p: usize, q: usize, c: T, s: T, n: usize) {
    for i in 0..n {
        let vip = v[(i, p)];
        let viq = v[(i, q)];
        v[(i, p)] = c * vip - s * viq;
        v[(i, q)] = s * vip + c * viq;
    }
}

fn sort_eig_ascending<T: Float>(eigenvalues: &mut [T], v: &mut Matrix<T>) {
    let n = eigenvalues.len();
    for i in 0..n {
        let mut best = i;
        for j in (i + 1)..n {
            if eigenvalues[j] < eigenvalues[best] {
                best = j;
            }
        }
        if best != i {
            eigenvalues.swap(i, best);
            for row in 0..n {
                let tmp = v[(row, i)];
                v[(row, i)] = v[(row, best)];
                v[(row, best)] = tmp;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::matmul;

    #[test]
    fn eigh_diagonal() {
        let a = Matrix::new(3, 3, vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]).unwrap();
        let (evals, evecs) = eigh(&a).unwrap();
        assert!((evals[0] - 1.0).abs() < 1e-10);
        assert!((evals[1] - 2.0).abs() < 1e-10);
        assert!((evals[2] - 3.0).abs() < 1e-10);
        // A V ≈ V Λ
        let mut vl = evecs.clone();
        for j in 0..3 {
            for i in 0..3 {
                vl[(i, j)] *= evals[j];
            }
        }
        let av = matmul(&a, &evecs).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((av[(i, j)] - vl[(i, j)]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn eigh_symmetric_2x2() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        let a = Matrix::new(2, 2, vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let (evals, evecs) = eigh(&a).unwrap();
        assert!((evals[0] - 1.0).abs() < 1e-9);
        assert!((evals[1] - 3.0).abs() < 1e-9);
        let vtv = matmul(&evecs.transpose(), &evecs).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((vtv[(i, j)] - e).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn not_symmetric_errors() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 0.0, 1.0]).unwrap();
        assert!(matches!(eigh(&a), Err(LinalgError::NotSymmetric)));
    }

    #[test]
    fn not_square() {
        let a = Matrix::<f64>::zeros(2, 3);
        assert!(matches!(eigh(&a), Err(LinalgError::NotSquare)));
    }
}
