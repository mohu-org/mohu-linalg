//! Integration tests spanning matmul and factorizations.

use mohu_linalg::{
    Matrix, cholesky, eigh, frobenius_norm, lstsq, lu_decompose, matmul, matmul_parallel,
    qr_decompose, solve, svd,
};

#[test]
fn matmul_identity_and_parallel_agree() {
    let a = Matrix::new(3, 3, vec![1.0_f64, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]).unwrap();
    let i = Matrix::<f64>::identity(3);
    let ai = matmul(&a, &i).unwrap();
    assert_eq!(ai, a);

    let b = Matrix::new(3, 2, vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let serial = matmul(&a, &b).unwrap();
    let parallel = matmul_parallel(&a, &b).unwrap();
    assert_eq!(serial, parallel);
}

#[test]
fn lu_solve_roundtrip() {
    let a = Matrix::new(
        3,
        3,
        vec![2.0_f64, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0],
    )
    .unwrap();
    let b = [1.0_f64, 0.0, 1.0];
    let x = solve(&a, &b).unwrap();
    let mut ax = [0.0_f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            ax[i] += a[(i, j)] * x[j];
        }
    }
    for i in 0..3 {
        assert!((ax[i] - b[i]).abs() < 1e-9);
    }
    let (_l, _u, pivots) = lu_decompose(&a).unwrap();
    assert_eq!(pivots.len(), 3);
}

#[test]
fn qr_and_lstsq_smoke() {
    let a = Matrix::new(4, 2, vec![1.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
    let (q, r) = qr_decompose(&a).unwrap();
    let qr = matmul(&q, &r).unwrap();
    for i in 0..4 {
        for j in 0..2 {
            assert!((qr[(i, j)] - a[(i, j)]).abs() < 1e-9);
        }
    }
    let x = lstsq(&a, &[1.0_f64, 3.0, 5.0, 7.0]).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-9);
    assert!((x[1] - 2.0).abs() < 1e-9);
}

#[test]
fn svd_reconstructs_and_cholesky_spd() {
    let a = Matrix::new(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();
    let (u, s, vt) = svd(&a).unwrap();
    let mut us = u.clone();
    for j in 0..s.len() {
        for i in 0..us.nrows() {
            us[(i, j)] *= s[j];
        }
    }
    let recon = matmul(&us, &vt).unwrap();
    let diff = Matrix::new(
        2,
        2,
        vec![
            recon[(0, 0)] - a[(0, 0)],
            recon[(0, 1)] - a[(0, 1)],
            recon[(1, 0)] - a[(1, 0)],
            recon[(1, 1)] - a[(1, 1)],
        ],
    )
    .unwrap();
    assert!(frobenius_norm(&diff) < 1e-6);

    let l = cholesky(&a).unwrap();
    let ll = matmul(&l, &l.transpose()).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert!((ll[(i, j)] - a[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn eigh_matches_known() {
    let a = Matrix::new(2, 2, vec![2.0_f64, 1.0, 1.0, 2.0]).unwrap();
    let (evals, evecs) = eigh(&a).unwrap();
    assert!((evals[0] - 1.0).abs() < 1e-9);
    assert!((evals[1] - 3.0).abs() < 1e-9);
    let av = matmul(&a, &evecs).unwrap();
    for j in 0..2 {
        for i in 0..2 {
            assert!((av[(i, j)] - evals[j] * evecs[(i, j)]).abs() < 1e-8);
        }
    }
}
