//! Criterion microbenchmarks for dense matmul.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use mohu_linalg::{Matrix, matmul, matmul_parallel};

fn dense_matrix(n: usize) -> Matrix<f64> {
    let data: Vec<f64> = (0..n * n).map(|k| ((k % 97) as f64) * 0.01 + 1.0).collect();
    Matrix::new(n, n, data).expect("bench matrix")
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for &n in &[16usize, 64, 128] {
        let a = dense_matrix(n);
        let b = dense_matrix(n);
        group.bench_with_input(BenchmarkId::new("serial", n), &n, |bencher, _| {
            bencher.iter(|| {
                let c = matmul(black_box(&a), black_box(&b)).unwrap();
                black_box(c);
            });
        });
        group.bench_with_input(BenchmarkId::new("parallel", n), &n, |bencher, _| {
            bencher.iter(|| {
                let c = matmul_parallel(black_box(&a), black_box(&b)).unwrap();
                black_box(c);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
