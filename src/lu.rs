//! LU factorization with partial pivoting and linear solve.

use crate::error::{LinalgError, LinalgResult};
use crate::matrix::Matrix;
use num_traits::Float;

/// Factor `A` into `P A = L U` with partial pivoting.
///
/// Returns `(L, U, pivots)` where:
/// - `L` is unit lower-triangular,
/// - `U` is upper-triangular,
/// - `pivots[k]` is the row swapped with row `k` at elimination step `k`
///   (so the permutation can be applied to a right-hand side by performing
///   the same sequence of swaps).
///
/// # Errors
///
/// - [`LinalgError::EmptyMatrix`] if `A` has zero size (should not occur for
///   well-formed matrices).
/// - [`LinalgError::NotSquare`] if `A` is not square.
/// - [`LinalgError::SingularMatrix`] if a pivot is numerically near zero.
pub fn lu_decompose<T: Float>(a: &Matrix<T>) -> LinalgResult<(Matrix<T>, Matrix<T>, Vec<usize>)> {
    let (n, m) = a.shape();
    if n == 0 || m == 0 {
        return Err(LinalgError::EmptyMatrix);
    }
    if n != m {
        return Err(LinalgError::NotSquare);
    }

    // Working copy; will hold U on/above diagonal and L multipliers below.
    let mut work = a.clone();
    let mut pivots = Vec::with_capacity(n);

    // Scale threshold: treat pivots smaller than eps * n * max|A| as zero.
    let max_abs = a
        .as_slice()
        .iter()
        .fold(T::zero(), |acc, &x| acc.max(x.abs()));
    let n_t = T::from(n).unwrap_or_else(T::one);
    // TODO: drop `.max(T::one())` so the tolerance is relative to the matrix scale.
    // With the clamp, a well-conditioned matrix with small entries (e.g. 1e-20 * I)
    // is reported as SingularMatrix. Use `T::epsilon() * n_t * max_abs`; a zero
    // matrix then gives tol = 0 and still fails the pivot check as intended.
    let pivot_tol = T::epsilon() * n_t * max_abs.max(T::one());

    for k in 0..n {
        // Find pivot row in column k (partial pivoting).
        let mut pivot_row = k;
        let mut pivot_val = work[(k, k)].abs();
        for i in (k + 1)..n {
            let v = work[(i, k)].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = i;
            }
        }

        if pivot_val <= pivot_tol {
            return Err(LinalgError::SingularMatrix);
        }

        if pivot_row != k {
            // Swap rows k and pivot_row in the working matrix.
            for j in 0..n {
                let tmp = work[(k, j)];
                work[(k, j)] = work[(pivot_row, j)];
                work[(pivot_row, j)] = tmp;
            }
        }
        pivots.push(pivot_row);

        let diag = work[(k, k)];
        for i in (k + 1)..n {
            let mult = work[(i, k)] / diag;
            work[(i, k)] = mult; // store L multiplier
            for j in (k + 1)..n {
                work[(i, j)] = work[(i, j)] - mult * work[(k, j)];
            }
        }
    }

    // Extract L and U.
    let mut l = Matrix::identity(n);
    let mut u = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i > j {
                l[(i, j)] = work[(i, j)];
            } else {
                u[(i, j)] = work[(i, j)];
            }
        }
    }

    Ok((l, u, pivots))
}

/// Solve `A x = b` using LU factorization with partial pivoting.
///
/// # Errors
///
/// - [`LinalgError::NotSquare`] / [`LinalgError::SingularMatrix`] from the
///   factorization.
/// - [`LinalgError::DimensionMismatch`] if `b.len() != A.nrows()`.
pub fn solve<T: Float>(a: &Matrix<T>, b: &[T]) -> LinalgResult<Vec<T>> {
    let n = a.nrows();
    if b.len() != n {
        return Err(LinalgError::DimensionMismatch {
            expected: format!("b.len() == {n}"),
            got: format!("b.len() == {}", b.len()),
        });
    }

    let (l, u, pivots) = lu_decompose(a)?;

    // Apply pivot swaps to b → pb.
    let mut pb = b.to_vec();
    for (k, &pivot_row) in pivots.iter().enumerate() {
        if pivot_row != k {
            pb.swap(k, pivot_row);
        }
    }

    // Forward substitution: L y = pb (unit diagonal).
    let mut y = vec![T::zero(); n];
    for i in 0..n {
        let mut s = pb[i];
        for j in 0..i {
            s = s - l[(i, j)] * y[j];
        }
        y[i] = s; // L_ii = 1
    }

    // Back substitution: U x = y.
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s = s - u[(i, j)] * x[j];
        }
        let diag = u[(i, i)];
        // TODO: remove this check once the pivot tolerance above is relative.
        // `T::epsilon()` is an absolute threshold, so it would reject valid small-scale
        // pivots that lu_decompose already accepted. Just divide by `u[(i, i)]`.
        // Should already be guarded by lu_decompose, but keep safe.
        if diag.abs() <= T::epsilon() {
            return Err(LinalgError::SingularMatrix);
        }
        x[i] = s / diag;
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_known_3x3() {
        // A x = b with known solution x = [2, 3, -1]
        // A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], b = [8, -11, -3]
        let a = Matrix::new(3, 3, vec![2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0]).unwrap();
        let b = [8.0, -11.0, -3.0];
        let x = solve(&a, &b).unwrap();
        assert_eq!(x.len(), 3);
        assert!((x[0] - 2.0).abs() < 1e-9, "x0 = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-9, "x1 = {}", x[1]);
        assert!((x[2] - (-1.0)).abs() < 1e-9, "x2 = {}", x[2]);
    }

    #[test]
    fn singular_returns_error() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let err = lu_decompose(&a).unwrap_err();
        assert!(matches!(err, LinalgError::SingularMatrix));

        let solve_err = solve(&a, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(solve_err, LinalgError::SingularMatrix));
    }

    #[test]
    fn not_square() {
        let a = Matrix::<f64>::zeros(2, 3);
        assert!(matches!(lu_decompose(&a), Err(LinalgError::NotSquare)));
    }

    #[test]
    fn lu_reconstructs_pa() {
        let a = Matrix::new(3, 3, vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 1.0, 1.0]).unwrap();
        let (l, u, pivots) = lu_decompose(&a).unwrap();

        // Build P by applying the same swaps to identity.
        let mut p = Matrix::<f64>::identity(3);
        for (k, &pr) in pivots.iter().enumerate() {
            if pr != k {
                for j in 0..3 {
                    let tmp = p[(k, j)];
                    p[(k, j)] = p[(pr, j)];
                    p[(pr, j)] = tmp;
                }
            }
        }

        let lu = crate::matmul::matmul(&l, &u).unwrap();
        let pa = crate::matmul::matmul(&p, &a).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (lu[(i, j)] - pa[(i, j)]).abs() < 1e-10,
                    "LU != PA at ({i},{j}): {} vs {}",
                    lu[(i, j)],
                    pa[(i, j)]
                );
            }
        }
    }
}
