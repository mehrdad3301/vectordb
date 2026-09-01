use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::search::TopK;
use crate::{Dataset, Metric, Neighbor};

pub(crate) fn search_layer(
    dataset: &Dataset,
    metric: Metric,
    query: &[f32],
    adjacency: &[Vec<usize>],
    entry_points: &[usize],
    ef: usize,
    allowed_rows: usize,
) -> Vec<Neighbor> {
    let width = ef.max(1);
    let limit = allowed_rows.min(dataset.len()).min(adjacency.len());

    let mut visited = HashSet::new();
    let mut candidates = BinaryHeap::new();
    let mut results = TopK::new(width);

    for &row in entry_points {
        if row >= limit || !visited.insert(row) {
            continue;
        }
        let neighbor = Neighbor {
            row,
            distance: metric.distance(dataset.vector(row), query),
        };
        candidates.push(Reverse(neighbor));
        results.push(neighbor);
    }

    while let Some(Reverse(candidate)) = candidates.pop() {
        if results.len() >= width && results.worst().is_some_and(|worst| candidate > worst) {
            break;
        }

        for &row in adjacency.get(candidate.row).into_iter().flatten() {
            if row >= limit || !visited.insert(row) {
                continue;
            }
            let neighbor = Neighbor {
                row,
                distance: metric.distance(dataset.vector(row), query),
            };
            if results.len() < width || results.worst().is_some_and(|worst| neighbor < worst) {
                candidates.push(Reverse(neighbor));
                results.push(neighbor);
            }
        }
    }

    results.into_sorted()
}

pub(crate) fn greedy_search(
    _dataset: &Dataset,
    _metric: Metric,
    _query: &[f32],
    _adjacency: &[Vec<usize>],
    _entry: usize,
    _allowed_rows: usize,
) -> usize {
    todo!("Chapter 4: greedily descend one HNSW layer")
}

pub(crate) fn prune_neighbors(
    dataset: &Dataset,
    metric: Metric,
    owner: usize,
    neighbors: &mut Vec<usize>,
    max_connections: usize,
) {
    let owner_vec = dataset.vector(owner);  

    neighbors
        .sort_by(
            |a, b| {
            let da = metric.distance(owner_vec, dataset.vector(*a));
            let db = metric.distance(owner_vec, dataset.vector(*b));
            da.total_cmp(&db).then_with(|| a.cmp(b))
        });

    // remove duplicates 
    neighbors.dedup() ;

    // remove self-edge 
    if neighbors.get(0).is_some() 
    && neighbors.get(0).unwrap().eq(&owner) { 
        neighbors.remove(0) ; 
    }
    
    neighbors.truncate(max_connections);
}
