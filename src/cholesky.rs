//! Cholesky factorization for symmetric positive-definite matrices.

use crate::error::{LinalgError, LinalgResult};
use crate::matrix::Matrix;
use num_traits::Float;

/// Compute the Cholesky factor `L` of a symmetric positive-definite matrix `A`,
/// so that `A = L Lᵀ` with `L` lower-triangular and positive diagonal.
///
/// Only the lower triangle of `A` is read; the upper triangle is ignored
/// (callers should pass a numerically symmetric matrix).
///
/// # Errors
///
/// - [`LinalgError::NotSquare`] if `A` is not square.
/// - [`LinalgError::NotPositiveDefinite`] if a diagonal entry becomes
///   non-positive during factorization (matrix not SPD, or too ill-conditioned).
pub fn cholesky<T: Float>(a: &Matrix<T>) -> LinalgResult<Matrix<T>> {
    let (n, m) = a.shape();
    if n != m {
        return Err(LinalgError::NotSquare);
    }

    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[(i, j)];
            for k in 0..j {
                sum = sum - l[(i, k)] * l[(j, k)];
            }
            if i == j {
                if sum <= T::zero() {
                    return Err(LinalgError::NotPositiveDefinite);
                }
                l[(i, j)] = sum.sqrt();
            } else {
                let diag = l[(j, j)];
                if diag.abs() <= T::epsilon() {
                    return Err(LinalgError::NotPositiveDefinite);
                }
                l[(i, j)] = sum / diag;
            }
        }
    }
    Ok(l)
}

/// Solve `A x = b` for SPD `A` via Cholesky: `L Lᵀ x = b`.
///
/// # Errors
///
/// - Propagates errors from [`cholesky`].
/// - [`LinalgError::DimensionMismatch`] if `b.len() != A.nrows()`.
pub fn solve_cholesky<T: Float>(a: &Matrix<T>, b: &[T]) -> LinalgResult<Vec<T>> {
    let n = a.nrows();
    if b.len() != n {
        return Err(LinalgError::DimensionMismatch {
            expected: format!("b.len() == {n}"),
            got: format!("b.len() == {}", b.len()),
        });
    }
    let l = cholesky(a)?;

    // Forward: L y = b.
    let mut y = vec![T::zero(); n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s = s - l[(i, j)] * y[j];
        }
        y[i] = s / l[(i, i)];
    }

    // Back: Lᵀ x = y.
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s = s - l[(j, i)] * x[j];
        }
        x[i] = s / l[(i, i)];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::matmul;

    #[test]
    fn cholesky_spd_reconstructs() {
        // A = [[4, 2], [2, 3]] = L L^T with L = [[2, 0], [1, sqrt(2)]]
        let a = Matrix::new(2, 2, vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let l = cholesky(&a).unwrap();
        assert!((l[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((l[(1, 0)] - 1.0).abs() < 1e-12);
        assert!((l[(1, 1)] - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!(l[(0, 1)].abs() < 1e-15);

        let lt = l.transpose();
        let reconstr = matmul(&l, &lt).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstr[(i, j)] - a[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn cholesky_solve() {
        let a = Matrix::new(3, 3, vec![4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0]).unwrap();
        let b = [1.0, 2.0, 3.0];
        let x = solve_cholesky(&a, &b).unwrap();
        // Verify A x ≈ b
        let mut ax = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                ax[i] += a[(i, j)] * x[j];
            }
        }
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-9, "ax[{i}] = {}", ax[i]);
        }
    }

    #[test]
    fn not_spd() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 1.0]).unwrap(); // indefinite
        assert!(matches!(
            cholesky(&a),
            Err(LinalgError::NotPositiveDefinite)
        ));
    }

    #[test]
    fn not_square() {
        let a = Matrix::<f64>::zeros(2, 3);
        assert!(matches!(cholesky(&a), Err(LinalgError::NotSquare)));
    }
}
