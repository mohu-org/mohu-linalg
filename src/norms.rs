//! Matrix and vector norms.

use crate::error::LinalgResult;
use crate::matrix::Matrix;
use crate::svd::svd;
use num_traits::Float;

/// Entrywise L1 norm: sum of absolute values of all entries.
pub fn l1_norm<T: Float>(m: &Matrix<T>) -> T {
    m.as_slice().iter().fold(T::zero(), |acc, &x| acc + x.abs())
}

/// Frobenius norm for general matrices; Euclidean 2-norm for row or column vectors
/// (1×N or N×1). Numerically identical to the Frobenius norm in all cases.
pub fn l2_norm<T: Float>(m: &Matrix<T>) -> T {
    frobenius_norm(m)
}

/// Frobenius norm: sqrt of sum of squared entries.
pub fn frobenius_norm<T: Float>(m: &Matrix<T>) -> T {
    let sum_sq = m.as_slice().iter().fold(T::zero(), |acc, &x| acc + x * x);
    sum_sq.sqrt()
}

/// Max (L-∞ / entrywise) norm: maximum absolute entry.
pub fn max_norm<T: Float>(m: &Matrix<T>) -> T {
    m.as_slice()
        .iter()
        .fold(T::zero(), |acc, &x| acc.max(x.abs()))
}

/// Nuclear (trace) norm: sum of singular values.
///
/// Computed via [`crate::svd::svd`]. Cost is comparable to a dense SVD.
///
/// # Errors
///
/// Propagates errors from the underlying SVD (e.g. convergence failure).
pub fn nuclear_norm<T: Float>(m: &Matrix<T>) -> LinalgResult<T> {
    let (_u, s, _vt) = svd(m)?;
    Ok(s.into_iter().fold(T::zero(), |acc, x| acc + x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    #[test]
    fn norms_hand_computed_small() {
        // Matrix [[3, -4], [0, 12]]
        let m = Matrix::new(2, 2, vec![3.0, -4.0, 0.0, 12.0]).unwrap();
        assert!((l1_norm(&m) - 19.0).abs() < 1e-12);
        assert!((frobenius_norm(&m) - 13.0).abs() < 1e-12);
        assert!((l2_norm(&m) - 13.0).abs() < 1e-12);
        assert!((max_norm(&m) - 12.0).abs() < 1e-12);

        let v = Matrix::new(2, 1, vec![3.0, 4.0]).unwrap();
        assert!((l2_norm(&v) - 5.0).abs() < 1e-12);
        assert!((l1_norm(&v) - 7.0).abs() < 1e-12);
        assert!((max_norm(&v) - 4.0).abs() < 1e-12);

        let r = Matrix::new(1, 2, vec![3.0, 4.0]).unwrap();
        assert!((l2_norm(&r) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn nuclear_norm_diagonal() {
        let m = Matrix::new(2, 2, vec![3.0, 0.0, 0.0, 2.0]).unwrap();
        let n = nuclear_norm(&m).unwrap();
        assert!((n - 5.0).abs() < 1e-8);
    }

    #[test]
    fn nuclear_norm_identity() {
        let m = Matrix::<f64>::identity(3);
        let n = nuclear_norm(&m).unwrap();
        assert!((n - 3.0).abs() < 1e-10);
    }
}
