//! Error types for linear algebra operations.

use thiserror::Error;

/// Errors produced by fallible linear algebra operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LinalgError {
    /// Matrix or vector dimensions did not match what an operation required.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected size or shape description.
        expected: String,
        /// Actual size or shape description.
        got: String,
    },

    /// Matrix is singular (or numerically singular) and cannot be factored/solved.
    #[error("singular matrix")]
    SingularMatrix,

    /// Operation requires a square matrix.
    #[error("matrix is not square")]
    NotSquare,

    /// Operation was given an empty matrix (zero rows or zero columns).
    #[error("empty matrix")]
    EmptyMatrix,
}

/// Convenient alias for results that may fail with [`LinalgError`].
pub type LinalgResult<T> = Result<T, LinalgError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let e = LinalgError::DimensionMismatch {
            expected: "2x3".into(),
            got: "3x2".into(),
        };
        assert!(e.to_string().contains("2x3"));
        assert!(e.to_string().contains("3x2"));
        assert_eq!(LinalgError::SingularMatrix.to_string(), "singular matrix");
        assert_eq!(LinalgError::NotSquare.to_string(), "matrix is not square");
        assert_eq!(LinalgError::EmptyMatrix.to_string(), "empty matrix");
    }
}
