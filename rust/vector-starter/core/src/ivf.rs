use crate::search::{DeterministicRng, TopK};
use crate::{Dataset, Metric, Neighbor, Result, VectorError, VectorIndex};

#[derive(Debug, Clone, Copy)]
pub struct IvfFlatConfig {
    pub partitions: usize,
    pub probes: usize,
    pub iterations: usize,
    pub seed: u64,
}

impl Default for IvfFlatConfig {
    fn default() -> Self {
        Self {
            partitions: 16,
            probes: 4,
            iterations: 12,
            seed: 0x5eed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IvfFlatIndex {
    dataset: Dataset,
    metric: Metric,
    config: IvfFlatConfig,
    centroids: Vec<Vec<f32>>,
    lists: Vec<Vec<usize>>,
}

impl IvfFlatIndex {
    pub fn try_new(dataset: Dataset, metric: Metric, config: IvfFlatConfig) -> Result<Self> {
        dataset.validate_for_metric(metric)?;
        validate_config(&config, dataset.len())?;

        let mut rng = DeterministicRng::new(config.seed);
        let mut order: Vec<usize> = (0..dataset.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.index(i + 1));
        }
        let mut centroids: Vec<Vec<f32>> = order[..config.partitions]
            .iter()
            .map(|&row| dataset.vector(row).to_vec())
            .collect();

        let mut assignments = vec![usize::MAX; dataset.len()];
        for _ in 0..config.iterations {
            let next = assign_rows(&dataset, metric, &centroids);
            if next == assignments {
                break;
            }
            assignments = next;
            let lists = lists_from_assignments(&assignments, config.partitions);
            update_centroids(&dataset, metric, &mut centroids, &lists);
        }

        let assignments = assign_rows(&dataset, metric, &centroids);
        let lists = lists_from_assignments(&assignments, config.partitions);

        Ok(Self {
            dataset,
            metric,
            config,
            centroids,
            lists,
        })
    }

    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    pub fn list_sizes(&self) -> Vec<usize> {
        self.lists.iter().map(Vec::len).collect()
    }

    pub fn search_with_probes(
        &self,
        query: &[f32],
        k: usize,
        probes: usize,
    ) -> Result<Vec<Neighbor>> {
        self.dataset.validate_query(query, self.metric)?;
        if probes == 0 || probes > self.centroids.len() {
            return Err(VectorError::InvalidConfig(
                "probes must satisfy 1 <= probes <= partitions",
            ));
        }

        let mut ranked = self
            .centroids
            .iter()
            .enumerate()
            .map(|(row, centroid)| Neighbor {
                row,
                distance: self.metric.distance(centroid, query),
            })
            .collect::<Vec<_>>();
        ranked.sort_unstable();

        let mut result = TopK::new(k.min(self.dataset.len()));
        for neighbor in ranked.into_iter().take(probes) {
            for &row in &self.lists[neighbor.row] {
                result.push(Neighbor {
                    row,
                    distance: self.metric.distance(self.dataset.vector(row), query),
                });
            }
        }
        Ok(result.into_sorted())
    }
}

impl VectorIndex for IvfFlatIndex {
    fn kind(&self) -> &'static str {
        "ivf_flat"
    }

    fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        self.search_with_probes(query, k, self.config.probes)
    }
}

fn validate_config(config: &IvfFlatConfig, rows: usize) -> Result<()> {
    if config.iterations == 0 {
        return Err(VectorError::InvalidConfig(
            "iterations must be greater than zero",
        ));
    }
    if config.partitions == 0
        || config.probes == 0
        || config.probes > config.partitions
        || config.partitions > rows
    {
        return Err(VectorError::InvalidConfig(
            "probes and partitions must satisfy 1 <= probes <= partitions <= rows",
        ));
    }
    Ok(())
}

fn nearest_centroid(vector: &[f32], centroids: &[Vec<f32>], metric: Metric) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(left_i, left), (right_i, right)| {
            let left_d = metric.distance(left, vector);
            let right_d = metric.distance(right, vector);
            left_d.total_cmp(&right_d).then_with(|| left_i.cmp(right_i))
        })
        .map(|(index, _)| index)
        .expect("an IVFFlat index has at least one centroid")
}

fn assign_rows(dataset: &Dataset, metric: Metric, centroids: &[Vec<f32>]) -> Vec<usize> {
    dataset
        .vectors()
        .iter()
        .map(|vector| nearest_centroid(vector, centroids, metric))
        .collect()
}

fn lists_from_assignments(assignments: &[usize], partitions: usize) -> Vec<Vec<usize>> {
    let mut lists = vec![Vec::new(); partitions];
    for (row, &cluster) in assignments.iter().enumerate() {
        lists[cluster].push(row);
    }
    lists
}

fn update_centroids(
    dataset: &Dataset,
    metric: Metric,
    centroids: &mut [Vec<f32>],
    lists: &[Vec<usize>],
) {
    let previous = centroids.to_vec();
    let mut reseeds = Vec::new();
    for (cluster, list) in lists.iter().enumerate() {
        if list.is_empty() {
            let row = farthest_row(dataset, metric, &previous, &reseeds);
            reseeds.push(row);
            centroids[cluster] = dataset.vector(row).to_vec();
            if metric == Metric::Cosine {
                normalize_or_replace(&mut centroids[cluster], dataset.vector(row));
            }
            continue;
        }

        let n = list.len() as f64;
        let mut mean = vec![0.0f64; dataset.dimension()];
        for &row in list {
            for (i, &value) in dataset.vector(row).iter().enumerate() {
                mean[i] += f64::from(value);
            }
        }
        let mut centroid: Vec<f32> = mean.into_iter().map(|value| (value / n) as f32).collect();
        if metric == Metric::Cosine {
            normalize_or_replace(&mut centroid, dataset.vector(list[0]));
        }
        centroids[cluster] = centroid;
    }
}

fn farthest_row(
    dataset: &Dataset,
    metric: Metric,
    centroids: &[Vec<f32>],
    skip: &[usize],
) -> usize {
    dataset
        .vectors()
        .iter()
        .enumerate()
        .filter(|(row, _)| !skip.contains(row))
        .max_by(|(left_row, left), (right_row, right)| {
            let left_d =
                metric.distance(left, &centroids[nearest_centroid(left, centroids, metric)]);
            let right_d = metric.distance(
                right,
                &centroids[nearest_centroid(right, centroids, metric)],
            );
            left_d
                .total_cmp(&right_d)
                .then_with(|| right_row.cmp(left_row))
        })
        .map(|(row, _)| row)
        .expect("an empty cluster can always steal a dataset row")
}

fn normalize_or_replace(centroid: &mut Vec<f32>, fallback: &[f32]) {
    if l2_normalize(centroid) {
        return;
    }
    centroid.copy_from_slice(fallback);
    l2_normalize(centroid);
}

fn l2_normalize(vector: &mut [f32]) -> bool {
    let norm = Metric::squared_norm(vector).sqrt();
    if norm == 0.0 {
        return false;
    }
    for value in vector {
        *value = (*value as f64 / norm) as f32;
    }
    true
}
