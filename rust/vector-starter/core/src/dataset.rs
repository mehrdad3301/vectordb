use std::sync::Arc;

use crate::{Metric, Result};
use crate::VectorError;

#[derive(Debug, Clone)]
pub struct Dataset {
    vectors: Arc<[Vec<f32>]>,
    dimension: usize,
}

impl Dataset {

    pub fn len(&self) -> usize { self.vectors.len() }

    pub fn is_empty(&self) -> bool { self.vectors.is_empty() }

    pub fn dimension(&self) -> usize { self.dimension }

    pub fn vector(&self, row: usize) -> &[f32] { &self.vectors[row] }

    pub fn try_new(vectors: Vec<Vec<f32>>) -> Result<Self> {
        if vectors.is_empty() {
            return Err(VectorError::EmptyDataset);
        }

        let dimension = vectors[0].len();
        for vector in &vectors {
            if vector.len() != dimension {
                return Err(VectorError::DimensionMismatch {
                    expected: dimension,
                    actual: vector.len(),
                });
            }
        }

        Ok(Self {
            vectors: Arc::from(vectors),
            dimension: dimension.to_owned(),
        })
    }

    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    pub(crate) fn validate_for_metric(&self, metric: Metric) -> Result<()> {
        if metric == Metric::Cosine {
            for vector in self.vectors.iter() {
                if vector.iter().all(|x| *x == 0.0) {
                    return Err(VectorError::ZeroNorm { vector: 0 });
                }
            }
        }

        Ok(())
    }

    pub fn validate_query(&self, query: &[f32], metric: Metric) -> Result<()> {
        if query.len() != self.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        for (i, x) in query.iter().enumerate() {
            if !x.is_finite() {
                return Err(VectorError::NonFiniteValue { vector: i, dimension: i });
            }
        }

        if metric == Metric::Cosine {
            let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm == 0.0 {
                return Err(VectorError::ZeroNorm { vector: 0 });
            }
        }

        Ok(())
    }
}
