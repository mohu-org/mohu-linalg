//! Dense row-major matrix type.

use crate::error::{LinalgError, LinalgResult};
use num_traits::Float;
use std::ops::{Index, IndexMut};

/// Dense row-major matrix over a floating-point type.
#[derive(Debug, Clone)]
pub struct Matrix<T: Float> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Float> Matrix<T> {
    /// Create a matrix from raw row-major data.
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::EmptyMatrix`] if `rows` or `cols` is zero.
    /// Returns [`LinalgError::DimensionMismatch`] if `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> LinalgResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(LinalgError::EmptyMatrix);
        }
        let expected = rows * cols;
        if data.len() != expected {
            return Err(LinalgError::DimensionMismatch {
                expected: format!("{rows}*{cols}={expected}"),
                got: data.len().to_string(),
            });
        }
        Ok(Self { data, rows, cols })
    }

    /// Create an `rows` × `cols` matrix filled with zeros.
    ///
    /// # Panics
    ///
    /// Panics if `rows` or `cols` is zero (use [`Self::new`] for fallible construction).
    pub fn zeros(rows: usize, cols: usize) -> Self {
        assert!(
            rows > 0 && cols > 0,
            "zeros: rows and cols must be non-zero"
        );
        Self {
            data: vec![T::zero(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create an `n` × `n` identity matrix.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero.
    pub fn identity(n: usize) -> Self {
        assert!(n > 0, "identity: n must be non-zero");
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = T::one();
        }
        m
    }

    /// Build a matrix from a vector of rows.
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::EmptyMatrix`] if there are no rows or a row is empty.
    /// Returns [`LinalgError::DimensionMismatch`] if rows have inconsistent lengths.
    pub fn from_vec2d(rows_data: Vec<Vec<T>>) -> LinalgResult<Self> {
        if rows_data.is_empty() {
            return Err(LinalgError::EmptyMatrix);
        }
        let rows = rows_data.len();
        let cols = rows_data[0].len();
        if cols == 0 {
            return Err(LinalgError::EmptyMatrix);
        }
        for (i, row) in rows_data.iter().enumerate() {
            if row.len() != cols {
                return Err(LinalgError::DimensionMismatch {
                    expected: format!("row 0 length {cols}"),
                    got: format!("row {i} length {}", row.len()),
                });
            }
        }
        let data = rows_data.into_iter().flatten().collect();
        Self::new(rows, cols, data)
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// `(rows, cols)` shape.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the underlying row-major storage.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get a reference to the element at `(row, col)`, or `None` if out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(&self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Set the element at `(row, col)`.
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::DimensionMismatch`] if the index is out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: T) -> LinalgResult<()> {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col] = value;
            Ok(())
        } else {
            Err(LinalgError::DimensionMismatch {
                expected: format!("index within {}x{}", self.rows, self.cols),
                got: format!("({row}, {col})"),
            })
        }
    }

    /// Borrow row `i` as a contiguous slice.
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows`.
    pub fn row(&self, i: usize) -> &[T] {
        assert!(
            i < self.rows,
            "row index {i} out of bounds (rows={})",
            self.rows
        );
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Return the transpose as a new matrix.
    pub fn transpose(&self) -> Self {
        let mut data = vec![T::zero(); self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Self {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }
}

impl<T: Float> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(
            row < self.rows && col < self.cols,
            "index ({row}, {col}) out of bounds for {}x{} matrix",
            self.rows,
            self.cols
        );
        &self.data[row * self.cols + col]
    }
}

impl<T: Float> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(
            row < self.rows && col < self.cols,
            "index ({row}, {col}) out of bounds for {}x{} matrix",
            self.rows,
            self.cols
        );
        &mut self.data[row * self.cols + col]
    }
}

impl<T: Float> PartialEq for Matrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_construction() {
        let m = Matrix::<f64>::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 2)], 6.0);
        assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);

        let z = Matrix::<f64>::zeros(2, 2);
        assert_eq!(z[(0, 1)], 0.0);

        let id = Matrix::<f64>::identity(3);
        assert_eq!(id[(0, 0)], 1.0);
        assert_eq!(id[(0, 1)], 0.0);
        assert_eq!(id[(2, 2)], 1.0);

        let from = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        assert_eq!(from, Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    }

    #[test]
    fn invalid_construction() {
        assert!(matches!(
            Matrix::<f64>::new(2, 2, vec![1.0, 2.0]),
            Err(LinalgError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            Matrix::<f64>::new(0, 2, vec![]),
            Err(LinalgError::EmptyMatrix)
        ));
        assert!(matches!(
            Matrix::<f64>::from_vec2d(vec![]),
            Err(LinalgError::EmptyMatrix)
        ));
        assert!(matches!(
            Matrix::<f64>::from_vec2d(vec![vec![1.0], vec![2.0, 3.0]]),
            Err(LinalgError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn get_set_transpose() {
        let mut m = Matrix::<f64>::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(m.get(0, 1), Some(&2.0));
        assert_eq!(m.get(9, 0), None);
        m.set(1, 0, 9.0).unwrap();
        assert_eq!(m[(1, 0)], 9.0);
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t[(0, 1)], 9.0);
        assert_eq!(t[(2, 0)], 3.0);
    }
}
