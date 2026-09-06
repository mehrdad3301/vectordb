use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::mem::size_of;

use crate::search::DeterministicRng;
use crate::{
    Dataset, IvfFlatConfig, IvfFlatIndex, Metric, Neighbor, Result, VectorError, VectorIndex,
};

#[derive(Debug, Clone, Copy)]
pub struct IvfPqConfig {
    pub partitions: usize,
    pub probes: usize,
    pub iterations: usize,
    pub subquantizers: usize,
    pub codebook_size: usize,
    pub rerank: usize,
    pub seed: u64,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            partitions: 16,
            probes: 4,
            iterations: 12,
            subquantizers: 4,
            codebook_size: 16,
            rerank: 32,
            seed: 0x5eed,
        }
    }
}

#[derive(Debug, Clone)]
struct QuantizedRow {
    row: usize,
    codes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct IvfPqIndex {
    dataset: Dataset,
    config: IvfPqConfig,
    centroids: Vec<Vec<f32>>,
    codebooks: Vec<Vec<Vec<f32>>>,
    lists: Vec<Vec<QuantizedRow>>,
}

impl IvfPqIndex {
    pub fn try_new(dataset: Dataset, metric: Metric, config: IvfPqConfig) -> Result<Self> {
        validate_config(&dataset, metric, &config)?;

        let coarse = IvfFlatIndex::try_new(
            dataset.clone(),
            Metric::Euclidean,
            IvfFlatConfig {
                partitions: config.partitions,
                probes: config.probes,
                iterations: config.iterations,
                seed: config.seed,
            },
        )?;
        let centroids = coarse.centroids().to_vec();

        let mut assignments = Vec::with_capacity(dataset.len());
        let mut residuals = Vec::with_capacity(dataset.len());
        for row in 0..dataset.len() {
            let vector = dataset.vector(row);
            let cluster = nearest_centroid(vector, &centroids);
            assignments.push(cluster);
            residuals.push(finite_residual(vector, &centroids[cluster])?);
        }

        let slice_dim = dataset.dimension() / config.subquantizers;
        let mut codebooks = Vec::with_capacity(config.subquantizers);
        for subquantizer in 0..config.subquantizers {
            codebooks.push(train_codebook(
                &residuals,
                subquantizer * slice_dim,
                slice_dim,
                config.codebook_size,
                config.iterations,
                codebook_seed(config.seed, subquantizer),
            ));
        }

        let mut lists = vec![Vec::new(); config.partitions];
        for (row, (cluster, residual)) in assignments.into_iter().zip(residuals).enumerate() {
            lists[cluster].push(QuantizedRow {
                row,
                codes: encode_residual(residual, &codebooks, slice_dim),
            });
        }

        Ok(Self {
            dataset,
            config,
            centroids,
            codebooks,
            lists,
        })
    }

    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    pub fn codebooks(&self) -> &[Vec<Vec<f32>>] {
        &self.codebooks
    }

    pub fn list_sizes(&self) -> Vec<usize> {
        self.lists.iter().map(Vec::len).collect()
    }

    pub fn encoded_bytes(&self) -> usize {
        self.lists.iter().flatten().map(|row| row.codes.len()).sum()
    }

    pub fn codebook_bytes(&self) -> usize {
        self.codebooks
            .iter()
            .flatten()
            .map(|centroid| centroid.len() * size_of::<f32>())
            .sum()
    }

    pub fn full_precision_bytes(&self) -> usize {
        self.dataset.len() * self.dataset.dimension() * size_of::<f32>()
    }

    pub fn search_with_probes(
        &self,
        query: &[f32],
        k: usize,
        probes: usize,
        rerank: usize,
    ) -> Result<Vec<Neighbor>> {
        self.dataset.validate_query(query, Metric::Euclidean)?;
        if probes == 0 || probes > self.centroids.len() {
            return Err(VectorError::InvalidConfig(
                "probes must satisfy 1 <= probes <= partitions",
            ));
        }
        if rerank == 0 {
            return Err(VectorError::InvalidConfig(
                "rerank must be greater than zero",
            ));
        }

        let needed = k.min(self.dataset.len());
        let shortlist_budget = rerank.max(k).min(self.dataset.len());
        let mut shortlist = ScoredTopK::new(shortlist_budget);

        let mut ranked = self
            .centroids
            .iter()
            .enumerate()
            .map(|(row, centroid)| (squared_l2_f64(query, centroid), row))
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            left.0.total_cmp(&right.0).then_with(|| left.1.cmp(&right.1))
        });

        let slice_dim = self.dataset.dimension() / self.config.subquantizers;
        for &(_, list) in ranked.iter().take(probes) {
            let query_residual = query_residual(query, &self.centroids[list]);
            let tables = lookup_tables(&query_residual, &self.codebooks, slice_dim);
            for encoded in &self.lists[list] {
                let score = encoded
                    .codes
                    .iter()
                    .enumerate()
                    .map(|(subquantizer, &code)| tables[subquantizer][code as usize])
                    .sum();
                shortlist.push(ScoredRow {
                    row: encoded.row,
                    score,
                });
            }
        }

        let candidates = shortlist.into_rows();
        let mut neighbors = Vec::new();
        let mut discarded_unrepresentable = false;
        for row in candidates {
            let distance = euclidean_f64(self.dataset.vector(row), query) as f32;
            if distance.is_finite() {
                neighbors.push(Neighbor { row, distance });
            } else {
                discarded_unrepresentable = true;
            }
        }

        if discarded_unrepresentable && neighbors.len() < needed {
            return Err(VectorError::InvalidConfig(
                "IVF-PQ result distances must remain representable as finite f32",
            ));
        }

        neighbors.sort_unstable();
        neighbors.truncate(needed);
        Ok(neighbors)
    }
}

impl VectorIndex for IvfPqIndex {
    fn kind(&self) -> &'static str {
        "ivf_pq"
    }

    fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    fn metric(&self) -> Metric {
        Metric::Euclidean
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        self.search_with_probes(query, k, self.config.probes, self.config.rerank)
    }
}

fn validate_config(dataset: &Dataset, metric: Metric, config: &IvfPqConfig) -> Result<()> {
    if metric != Metric::Euclidean {
        return Err(VectorError::InvalidConfig(
            "IVF-PQ accepts only Euclidean distance",
        ));
    }
    if config.iterations == 0 {
        return Err(VectorError::InvalidConfig(
            "iterations must be greater than zero",
        ));
    }
    if config.partitions == 0
        || config.probes == 0
        || config.probes > config.partitions
        || config.partitions > dataset.len()
    {
        return Err(VectorError::InvalidConfig(
            "probes and partitions must satisfy 1 <= probes <= partitions <= rows",
        ));
    }
    if config.subquantizers == 0 || dataset.dimension() % config.subquantizers != 0 {
        return Err(VectorError::InvalidConfig(
            "dimension must divide evenly into a positive number of subquantizers",
        ));
    }
    if config.codebook_size < 2 || config.codebook_size > 256.min(dataset.len()) {
        return Err(VectorError::InvalidConfig(
            "codebook_size must satisfy 2 <= codebook_size <= min(256, rows)",
        ));
    }
    if config.rerank == 0 {
        return Err(VectorError::InvalidConfig(
            "rerank must be greater than zero",
        ));
    }
    Ok(())
}

fn codebook_seed(seed: u64, subquantizer: usize) -> u64 {
    seed.wrapping_add(
        (subquantizer as u64)
            .wrapping_add(1)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15),
    )
}

fn nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(left_i, left), (right_i, right)| {
            let left_d = Metric::Euclidean.distance(left, vector);
            let right_d = Metric::Euclidean.distance(right, vector);
            left_d.total_cmp(&right_d).then_with(|| left_i.cmp(right_i))
        })
        .map(|(index, _)| index)
        .expect("an IVF-PQ index has at least one centroid")
}

fn finite_residual(vector: &[f32], centroid: &[f32]) -> Result<Vec<f32>> {
    let residual = vector
        .iter()
        .zip(centroid)
        .map(|(value, center)| (f64::from(*value) - f64::from(*center)) as f32)
        .collect::<Vec<_>>();
    if residual.iter().any(|value| !value.is_finite()) {
        return Err(VectorError::InvalidConfig(
            "IVF-PQ training residuals must remain finite",
        ));
    }
    Ok(residual)
}

fn train_codebook(
    residuals: &[Vec<f32>],
    offset: usize,
    slice_dim: usize,
    codebook_size: usize,
    iterations: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = DeterministicRng::new(seed);
    let mut order: Vec<usize> = (0..residuals.len()).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.index(i + 1));
    }

    let mut codebook: Vec<Vec<f32>> = order[..codebook_size]
        .iter()
        .map(|&row| residuals[row][offset..offset + slice_dim].to_vec())
        .collect();

    let mut assignments = vec![usize::MAX; residuals.len()];
    for _ in 0..iterations {
        let next = residuals
            .iter()
            .map(|residual| nearest_codeword(&residual[offset..offset + slice_dim], &codebook))
            .collect::<Vec<_>>();
        if next == assignments {
            break;
        }
        assignments = next;
        update_codebook(&mut codebook, residuals, &assignments, offset, slice_dim);
    }

    codebook
}

fn nearest_codeword(slice: &[f32], codebook: &[Vec<f32>]) -> usize {
    codebook
        .iter()
        .enumerate()
        .min_by(|(left_i, left), (right_i, right)| {
            let left_d = squared_l2_f64(slice, left);
            let right_d = squared_l2_f64(slice, right);
            left_d.total_cmp(&right_d).then_with(|| left_i.cmp(right_i))
        })
        .map(|(index, _)| index)
        .expect("a PQ codebook has at least one codeword")
}

fn update_codebook(
    codebook: &mut [Vec<f32>],
    residuals: &[Vec<f32>],
    assignments: &[usize],
    offset: usize,
    slice_dim: usize,
) {
    let mut sums = vec![vec![0.0f64; slice_dim]; codebook.len()];
    let mut counts = vec![0usize; codebook.len()];
    for (row, &code) in assignments.iter().enumerate() {
        counts[code] += 1;
        for (dim, value) in residuals[row][offset..offset + slice_dim]
            .iter()
            .enumerate()
        {
            sums[code][dim] += f64::from(*value);
        }
    }

    for (code, centroid) in codebook.iter_mut().enumerate() {
        if counts[code] == 0 {
            continue;
        }
        let n = counts[code] as f64;
        *centroid = sums[code]
            .iter()
            .map(|value| (*value / n) as f32)
            .collect();
    }
}

fn encode_residual(residual: Vec<f32>, codebooks: &[Vec<Vec<f32>>], slice_dim: usize) -> Vec<u8> {
    codebooks
        .iter()
        .enumerate()
        .map(|(subquantizer, codebook)| {
            let start = subquantizer * slice_dim;
            nearest_codeword(&residual[start..start + slice_dim], codebook) as u8
        })
        .collect()
}

fn query_residual(query: &[f32], centroid: &[f32]) -> Vec<f64> {
    query
        .iter()
        .zip(centroid)
        .map(|(value, center)| f64::from(*value) - f64::from(*center))
        .collect()
}

fn lookup_tables(
    query_residual: &[f64],
    codebooks: &[Vec<Vec<f32>>],
    slice_dim: usize,
) -> Vec<Vec<f64>> {
    codebooks
        .iter()
        .enumerate()
        .map(|(subquantizer, codebook)| {
            let start = subquantizer * slice_dim;
            codebook
                .iter()
                .map(|codeword| {
                    query_residual[start..start + slice_dim]
                        .iter()
                        .zip(codeword)
                        .map(|(value, center)| {
                            let delta = *value - f64::from(*center);
                            delta * delta
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn squared_l2_f64(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = f64::from(*left) - f64::from(*right);
            delta * delta
        })
        .sum()
}

fn euclidean_f64(left: &[f32], right: &[f32]) -> f64 {
    squared_l2_f64(left, right).sqrt()
}

#[derive(Debug, Clone, Copy)]
struct ScoredRow {
    row: usize,
    score: f64,
}

impl PartialEq for ScoredRow {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row && self.score.total_cmp(&other.score).is_eq()
    }
}

impl Eq for ScoredRow {}

impl PartialOrd for ScoredRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredRow {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.row.cmp(&other.row))
    }
}

#[derive(Debug)]
struct ScoredTopK {
    k: usize,
    heap: BinaryHeap<ScoredRow>,
}

impl ScoredTopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
        }
    }

    fn push(&mut self, row: ScoredRow) {
        if self.k == 0 {
            return;
        }
        self.heap.push(row);
        if self.heap.len() > self.k {
            self.heap.pop();
        }
    }

    fn into_rows(self) -> Vec<usize> {
        self.heap.into_iter().map(|row| row.row).collect()
    }
}
