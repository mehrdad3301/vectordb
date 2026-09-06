use std::hint::select_unpredictable;

use crate::graph::{greedy_search, search_layer, prune_neighbors};
use crate::{Dataset, Metric, Neighbor, Result, VectorIndex, search::DeterministicRng};
use crate::VectorError; 

#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    pub max_connections: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_level: usize,
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_connections: 12,
            ef_construction: 64,
            ef_search: 40,
            max_level: 16,
            seed: 0x5eed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HnswIndex {
    dataset: Dataset,
    metric: Metric,
    config: HnswConfig,
    levels: Vec<usize>,
    layers: Vec<Vec<Vec<usize>>>,
    entry_point: usize,
    top_level: usize,
}

impl HnswIndex {
    pub fn try_new(dataset: Dataset, metric: Metric, config: HnswConfig) -> Result<Self> {

        dataset.validate_for_metric(metric)?; 
        validate_config(&config)?; 

        let mut rng = DeterministicRng::new(config.seed) ; 
        let mut levels = vec![]; 
        let mut layers = vec![vec![vec![]; dataset.len()]] ; 
        let mut top_level = 0 ; 
        let mut entry_point = 0 ; 

        for row in 0..dataset.len() {

            let mut target_level = 0 ; 
            while rng.coin_flip()  && target_level < config.max_level { 
                target_level += 1 ; 
            } ; 

            while layers.len() <= target_level {
                layers.push(vec![vec![]; dataset.len()]);
            }

            levels.push(target_level) ; 

            let vector = dataset.vector(row) ; 
            let mut entry = entry_point ; 
            for level in (target_level + 1..=top_level).rev() {

                entry = greedy_search(
                    &dataset, 
                    metric, 
                    vector,
                    &layers[level], 
                    entry, row) ; 

            }

            // add row to target layers and all the levels in between top level and target level, 
            // if target level > top level 

            for level in (0..=target_level.min(top_level)).rev() { 

                let found = search_layer(
                    &dataset, 
                    metric, 
                    vector, 
                    &layers[level], 
                    &[entry], 
                    config.ef_construction,
                    row) ; 

                if let Some(nearest) = found.first() { 
                    entry = nearest.row; 
                }

                let mut selected: Vec<usize> = found
                    .into_iter()
                    .map(|neighbor| neighbor.row)
                    .take(config.max_connections)
                    .collect();
    
                for &neighbor in &selected {
                    layers[level][row].push(neighbor);
                    layers[level][neighbor].push(row);
                }

                selected.push(row);
                for &owner in &selected {
                    let previous = layers[level][owner].clone();
                    prune_neighbors(
                        &dataset,
                        metric,
                        owner,
                        &mut layers[level][owner],
                        config.max_connections,
                    );
                    for neighbor in previous {
                        if !layers[level][owner].contains(&neighbor) {
                            layers[level][neighbor].retain(|&other| other != owner);
                        }
                    }
                }
                

            }


            if target_level > top_level {
                entry_point = row;
            }

            top_level = top_level.max(target_level) ; 

        }


        Ok (Self { 
            dataset, 
            metric,
            config,
            levels,
            layers, 
            entry_point,
            top_level
        })
    }

    pub fn levels(&self) -> &[usize] {
        &self.levels
    }

    pub fn top_level(&self) -> usize {
        self.top_level
    }

    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Neighbor>> {
        self.dataset.validate_query(query, self.metric)? ; 

        let mut entry = self.entry_point ; 
        for level in (1..=self.top_level).rev() { 
            entry = greedy_search(
                &self.dataset, 
                self.metric, 
                query,
                &self.layers[level], 
                entry, self.dataset.len()) ; 
        } ;

        let found = search_layer(
            &self.dataset,
            self.metric,
            query,
            &self.layers[0],
            &[entry],
            ef_search.max(k),
            self.dataset.len(),
        );

        Ok(found.into_iter().take(k).collect())
    }

    pub fn layer(&self, level: usize) -> Option<&[Vec<usize>]> {
        self.layers.get(level).map(Vec::as_slice)
    }
}

impl VectorIndex for HnswIndex {
    fn kind(&self) -> &'static str {
        "hnsw"
    }

    fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        self.search_with_ef(query, k, self.config.ef_search)
    }
}

fn validate_config(config: &HnswConfig) -> Result<()> {
    if config.max_level <= 0 { 
        return Err(VectorError::InvalidConfig(
            "max_level must be greater than zero",
        ));
    }

    if config.max_connections <= 0 {
        return Err(VectorError::InvalidConfig(
            "max_connections must be greater than zero",
        ));
    }
    if config.ef_construction < config.max_connections {
        return Err(VectorError::InvalidConfig(
            "ef_construction must be at least max_connections",
        ));
    }
    if config.ef_search == 0 {
        return Err(VectorError::InvalidConfig("ef_search must be greater than zero"));
    }
    Ok(())
}