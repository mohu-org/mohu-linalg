//! Thin QR factorization via Householder reflections, and least-squares solve.

use crate::error::{LinalgError, LinalgResult};
use crate::matrix::Matrix;
use num_traits::Float;

/// Thin QR factorization `A = Q R` using Householder reflections.
///
/// For an `m × n` matrix with `m >= n`:
/// - `Q` is `m × n` with orthonormal columns,
/// - `R` is `n × n` upper-triangular.
///
/// For `m < n`:
/// - `Q` is `m × m` orthogonal,
/// - `R` is `m × n` upper-trapezoidal.
///
/// In both cases `matmul(Q, R) ≈ A` within a modest multiple of machine epsilon.
///
/// # Errors
///
/// - [`LinalgError::EmptyMatrix`] if `A` is empty (should not occur for valid matrices).
pub fn qr_decompose<T: Float>(a: &Matrix<T>) -> LinalgResult<(Matrix<T>, Matrix<T>)> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Err(LinalgError::EmptyMatrix);
    }

    let nreflect = m.min(n);
    let mut work = a.clone();
    let mut betas = vec![T::zero(); nreflect];

    for j in 0..nreflect {
        let (v, beta) = householder_column(&work, j, m);
        betas[j] = beta;
        apply_householder_left(&mut work, j, &v, beta, j);
        // Store essential part of v below the diagonal for later Q formation.
        for i in (j + 1)..m {
            work[(i, j)] = v[i - j];
        }
    }

    // Extract R from the upper triangle / trapezoid of work.
    let (r_rows, r_cols) = if m >= n { (n, n) } else { (m, n) };
    let mut r = Matrix::zeros(r_rows, r_cols);
    for i in 0..r_rows {
        for j in i..r_cols {
            r[(i, j)] = work[(i, j)];
        }
    }

    // Form thin/full-square Q by applying reflectors to a (partial) identity.
    let q_cols = if m >= n { n } else { m };
    let mut q = Matrix::zeros(m, q_cols);
    for i in 0..q_cols {
        q[(i, i)] = T::one();
    }
    for j in (0..nreflect).rev() {
        let mut v = vec![T::zero(); m - j];
        v[0] = T::one();
        for i in (j + 1)..m {
            v[i - j] = work[(i, j)];
        }
        apply_householder_left(&mut q, j, &v, betas[j], 0);
    }

    Ok((q, r))
}

/// Compute a Householder vector for column `j` of `work`, rows `j..m`.
///
/// Returns `(v, beta)` for the reflector `I - beta v v^T` in compact form
/// with `v[0] = 1` (or `beta = 0` if the column is already zero below).
fn householder_column<T: Float>(work: &Matrix<T>, j: usize, m: usize) -> (Vec<T>, T) {
    let len = m - j;
    let mut v = vec![T::zero(); len];
    for i in 0..len {
        v[i] = work[(j + i, j)];
    }
    let sigma = v[1..].iter().fold(T::zero(), |acc, &t| acc + t * t);
    let two = T::from(2.0).unwrap_or_else(|| T::one() + T::one());

    if sigma == T::zero() {
        // Already a multiple of e_1; no reflection needed (or flip sign only).
        return (v, T::zero());
    }

    let x0 = v[0];
    let norm = (x0 * x0 + sigma).sqrt();
    // mu = x0 + sign(x0) * ||x||, chosen to avoid cancellation.
    let mu = if x0 <= T::zero() {
        x0 - norm
    } else {
        -sigma / (x0 + norm)
    };

    for vi in v.iter_mut().skip(1) {
        *vi = *vi / mu;
    }
    v[0] = T::one();
    let beta = two * mu * mu / (sigma + mu * mu);
    (v, beta)
}

/// Apply `H = I - beta v v^T` from the left to columns `col0..` of `mat`,
/// acting on rows `j..`.
fn apply_householder_left<T: Float>(mat: &mut Matrix<T>, j: usize, v: &[T], beta: T, col0: usize) {
    if beta == T::zero() {
        return;
    }
    let n = mat.ncols();
    let len = v.len();
    for col in col0..n {
        let mut dot = T::zero();
        for i in 0..len {
            dot = dot + v[i] * mat[(j + i, col)];
        }
        let w = beta * dot;
        for i in 0..len {
            mat[(j + i, col)] = mat[(j + i, col)] - w * v[i];
        }
    }
}

/// Solve the linear least-squares problem `min ||A x - b||_2` via thin QR.
///
/// Requires `A` to be `m × n` with `m >= n` and full column rank. Returns the
/// unique minimizer when `A` has full column rank.
///
/// # Errors
///
/// - [`LinalgError::DimensionMismatch`] if `b.len() != A.nrows()` or `m < n`.
/// - [`LinalgError::SingularMatrix`] if `R` has a near-zero diagonal entry.
pub fn lstsq<T: Float>(a: &Matrix<T>, b: &[T]) -> LinalgResult<Vec<T>> {
    let (m, n) = a.shape();
    if b.len() != m {
        return Err(LinalgError::DimensionMismatch {
            expected: format!("b.len() == {m}"),
            got: format!("b.len() == {}", b.len()),
        });
    }
    if m < n {
        return Err(LinalgError::DimensionMismatch {
            expected: "m >= n for unique least-squares via thin QR".into(),
            got: format!("{m}x{n}"),
        });
    }

    let (q, r) = qr_decompose(a)?;

    // c = Q^T b  (Q is m×n).
    let mut c = vec![T::zero(); n];
    for j in 0..n {
        let mut s = T::zero();
        for i in 0..m {
            s = s + q[(i, j)] * b[i];
        }
        c[j] = s;
    }

    // Back-solve R x = c.
    let max_abs = r
        .as_slice()
        .iter()
        .fold(T::zero(), |acc, &x| acc.max(x.abs()));
    let n_t = T::from(n).unwrap_or_else(T::one);
    let pivot_tol = T::epsilon() * n_t * max_abs.max(T::one());

    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let diag = r[(i, i)];
        if diag.abs() <= pivot_tol {
            return Err(LinalgError::SingularMatrix);
        }
        let mut s = c[i];
        for j in (i + 1)..n {
            s = s - r[(i, j)] * x[j];
        }
        x[i] = s / diag;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::matmul;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn qr_square_reconstructs() {
        let a = Matrix::new(
            3,
            3,
            vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
        )
        .unwrap();
        let (q, r) = qr_decompose(&a).unwrap();
        assert_eq!(q.shape(), (3, 3));
        assert_eq!(r.shape(), (3, 3));
        assert!(r[(1, 0)].abs() < 1e-10);
        assert!(r[(2, 0)].abs() < 1e-10);
        assert!(r[(2, 1)].abs() < 1e-10);

        let qt = q.transpose();
        let qtq = matmul(&qt, &q).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(qtq[(i, j)], expected, 1e-9),
                    "Q^T Q[{i},{j}] = {}",
                    qtq[(i, j)]
                );
            }
        }

        let qr = matmul(&q, &r).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(qr[(i, j)], a[(i, j)], 1e-8),
                    "QR[{i},{j}] = {} vs {}",
                    qr[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn qr_tall_thin() {
        let a = Matrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).unwrap();
        let (q, r) = qr_decompose(&a).unwrap();
        assert_eq!(q.shape(), (4, 2));
        assert_eq!(r.shape(), (2, 2));
        let qr = matmul(&q, &r).unwrap();
        for i in 0..4 {
            for j in 0..2 {
                assert!(approx_eq(qr[(i, j)], a[(i, j)], 1e-9));
            }
        }
    }

    #[test]
    fn qr_wide() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (q, r) = qr_decompose(&a).unwrap();
        assert_eq!(q.shape(), (2, 2));
        assert_eq!(r.shape(), (2, 3));
        let qr = matmul(&q, &r).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_eq(qr[(i, j)], a[(i, j)], 1e-9));
            }
        }
    }

    #[test]
    fn lstsq_overdetermined() {
        // Fit y = 1 + 2x through (0,1), (1,3), (2,5), (3,7).
        let a = Matrix::new(4, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
        let b = [1.0, 3.0, 5.0, 7.0];
        let x = lstsq(&a, &b).unwrap();
        assert!(approx_eq(x[0], 1.0, 1e-9));
        assert!(approx_eq(x[1], 2.0, 1e-9));
    }

    #[test]
    fn lstsq_dim_errors() {
        let a = Matrix::<f64>::zeros(2, 3);
        assert!(matches!(
            lstsq(&a, &[1.0, 2.0]),
            Err(LinalgError::DimensionMismatch { .. })
        ));
        let tall = Matrix::<f64>::zeros(3, 2);
        assert!(matches!(
            lstsq(&tall, &[1.0]),
            Err(LinalgError::DimensionMismatch { .. })
        ));
    }
}
