//! Matrix–matrix multiplication (serial and parallel).

use crate::error::{LinalgError, LinalgResult};
use crate::matrix::Matrix;
use num_traits::Float;
use rayon::prelude::*;
use std::ops::Mul;

/// Naive matrix multiplication with `i`-`k`-`j` loop order.
///
/// Computes `C = A * B` where `A` is `m×k` and `B` is `k×n`.
///
/// # Errors
///
/// Returns [`LinalgError::DimensionMismatch`] if `A.ncols() != B.nrows()`.
pub fn matmul<T: Float>(a: &Matrix<T>, b: &Matrix<T>) -> LinalgResult<Matrix<T>> {
    let (m, k_a) = a.shape();
    let (k_b, n) = b.shape();
    if k_a != k_b {
        return Err(LinalgError::DimensionMismatch {
            expected: format!("inner dims equal (A.cols={k_a})"),
            got: format!("B.rows={k_b}"),
        });
    }
    let k = k_a;
    let mut c = Matrix::zeros(m, n);
    // i-k-j order: good for row-major A and C; accumulates along k.
    for i in 0..m {
        for kk in 0..k {
            let aik = a[(i, kk)];
            for j in 0..n {
                c[(i, j)] = c[(i, j)] + aik * b[(kk, j)];
            }
        }
    }
    Ok(c)
}

/// Parallel matrix multiplication using Rayon over output rows.
///
/// Same contract as [`matmul`], but each output row is computed independently
/// on a Rayon worker.
///
/// # Errors
///
/// Returns [`LinalgError::DimensionMismatch`] if `A.ncols() != B.nrows()`.
pub fn matmul_parallel<T: Float + Send + Sync>(
    a: &Matrix<T>,
    b: &Matrix<T>,
) -> LinalgResult<Matrix<T>> {
    let (m, k_a) = a.shape();
    let (k_b, n) = b.shape();
    if k_a != k_b {
        return Err(LinalgError::DimensionMismatch {
            expected: format!("inner dims equal (A.cols={k_a})"),
            got: format!("B.rows={k_b}"),
        });
    }
    let k = k_a;
    let a_data = a.as_slice();
    let b_data = b.as_slice();

    let rows: Vec<Vec<T>> = (0..m)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![T::zero(); n];
            for kk in 0..k {
                let aik = a_data[i * k + kk];
                for j in 0..n {
                    row[j] = row[j] + aik * b_data[kk * n + j];
                }
            }
            row
        })
        .collect();

    let data: Vec<T> = rows.into_iter().flatten().collect();
    Matrix::new(m, n, data)
}

/// Matrix multiplication via the `*` operator on references.
///
/// # Panics
///
/// Panics if dimensions are incompatible. Prefer [`matmul`] when you need
/// fallible error handling: the `Mul` trait's `Output` type cannot be a
/// `Result`, so a panic (with a clear message) is the only way to surface
/// dimension errors through the operator.
impl<T: Float> Mul for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        matmul(self, rhs).unwrap_or_else(|e| {
            panic!("matrix multiplication failed: {e}");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2x2() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c, Matrix::new(2, 2, vec![19.0, 22.0, 43.0, 50.0]).unwrap());
    }

    #[test]
    fn matmul_2x3_times_3x2() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = matmul(&a, &b).unwrap();
        // [[58, 64], [139, 154]]
        assert_eq!(c, Matrix::new(2, 2, vec![58.0, 64.0, 139.0, 154.0]).unwrap());
    }

    #[test]
    fn matmul_identity() {
        let a = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let i = Matrix::<f64>::identity(3);
        let c = matmul(&a, &i).unwrap();
        assert_eq!(c, a);
        let c2 = matmul(&i, &a).unwrap();
        assert_eq!(c2, a);
    }

    #[test]
    fn matmul_dim_mismatch() {
        let a = Matrix::<f64>::zeros(2, 3);
        let b = Matrix::<f64>::zeros(2, 2);
        assert!(matches!(
            matmul(&a, &b),
            Err(LinalgError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn matmul_matches_parallel_on_hardcoded() {
        // Deterministic 4x5 * 5x3 with hardcoded entries (no RNG).
        let a = Matrix::new(
            4,
            5,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, //
                6.0, 7.0, 8.0, 9.0, 10.0, //
                11.0, 12.0, 13.0, 14.0, 15.0, //
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
        )
        .unwrap();
        let b = Matrix::new(
            5,
            3,
            vec![
                0.5, 1.5, 2.5, //
                3.5, 4.5, 5.5, //
                6.5, 7.5, 8.5, //
                9.5, 10.5, 11.5, //
                12.5, 13.5, 14.5,
            ],
        )
        .unwrap();
        let serial = matmul(&a, &b).unwrap();
        let parallel = matmul_parallel(&a, &b).unwrap();
        assert_eq!(serial, parallel);

        // Spot-check first entry: 1*0.5+2*3.5+3*6.5+4*9.5+5*12.5 = 0.5+7+19.5+38+62.5 = 127.5
        assert!((serial[(0, 0)] - 127.5).abs() < 1e-12);
    }

    #[test]
    fn mul_operator() {
        let a = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let b = Matrix::new(2, 2, vec![2.0, 3.0, 4.0, 5.0]).unwrap();
        let c = &a * &b;
        assert_eq!(c, b);
    }
}
