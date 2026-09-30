use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::hybrid::simd::{DistanceMetric, QuantizedVector};
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLocation {
    pub block_offset: u64,
    pub slot: u16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchCandidate {
    pub id: u64,
    pub location: BlockLocation,
    pub distance: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    pub max_connections: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_connections: 24,
            ef_construction: 200,
            ef_search: 128,
        }
    }
}

struct Node {
    id: u64,
    vector: QuantizedVector,
    location: BlockLocation,
    links: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, Debug)]
struct DistanceIndex {
    distance: f32,
    index: usize,
}

impl PartialEq for DistanceIndex {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for DistanceIndex {}

impl PartialOrd for DistanceIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.index.cmp(&other.index))
    }
}

pub(crate) struct HnswIndex {
    metric: DistanceMetric,
    config: HnswConfig,
    nodes: Vec<Node>,
    by_key: HashMap<u64, usize>,
    entry: Option<usize>,
    maximum_level: usize,
}

impl HnswIndex {
    pub fn new(metric: DistanceMetric, config: HnswConfig) -> Result<Self> {
        if config.max_connections < 2
            || config.ef_construction < config.max_connections
            || config.ef_search == 0
        {
            return Err(Error::Invariant("invalid HNSW configuration".to_owned()));
        }
        Ok(Self {
            metric,
            config,
            nodes: Vec::new(),
            by_key: HashMap::new(),
            entry: None,
            maximum_level: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn ef_search(&self) -> usize {
        self.config.ef_search
    }

    pub fn insert(
        &mut self,
        index_key: u64,
        id: u64,
        vector: QuantizedVector,
        location: BlockLocation,
    ) -> Result<()> {
        if self.by_key.contains_key(&index_key) {
            return Err(Error::Invariant(format!(
                "HNSW already contains key {index_key}"
            )));
        }
        let level = deterministic_level(index_key, self.config.max_connections);
        let new_index = self.nodes.len();
        let Some(mut entry) = self.entry else {
            self.nodes.push(Node {
                id,
                vector,
                location,
                links: vec![Vec::new(); level + 1],
            });
            self.by_key.insert(index_key, new_index);
            self.entry = Some(new_index);
            self.maximum_level = level;
            return Ok(());
        };

        for layer in ((level + 1)..=self.maximum_level).rev() {
            entry = self.greedy_closest(&vector, entry, layer);
        }

        let mut links = vec![Vec::new(); level + 1];
        for layer in (0..=level.min(self.maximum_level)).rev() {
            let mut candidates =
                self.search_layer(&vector, entry, self.config.ef_construction, layer);
            candidates.truncate(self.connection_limit(layer));
            links[layer] = candidates.iter().map(|candidate| candidate.index).collect();
            if let Some(best) = candidates.first() {
                entry = best.index;
            }
        }

        self.nodes.push(Node {
            id,
            vector,
            location,
            links,
        });
        self.by_key.insert(index_key, new_index);

        for layer in 0..=level.min(self.maximum_level) {
            let neighbors = self.nodes[new_index].links[layer].clone();
            for neighbor in neighbors {
                self.nodes[neighbor].links[layer].push(new_index);
                self.prune_links(neighbor, layer);
            }
        }
        if level > self.maximum_level {
            self.entry = Some(new_index);
            self.maximum_level = level;
        }
        Ok(())
    }

    pub fn search(
        &self,
        query: &QuantizedVector,
        count: usize,
        ef_search: Option<usize>,
    ) -> Vec<SearchCandidate> {
        let Some(mut entry) = self.entry else {
            return Vec::new();
        };
        for layer in (1..=self.maximum_level).rev() {
            entry = self.greedy_closest(query, entry, layer);
        }
        let ef = ef_search
            .unwrap_or(self.config.ef_search)
            .max(count)
            .min(self.nodes.len());
        let mut candidates = self.search_layer(query, entry, ef, 0);
        candidates.truncate(count);
        candidates
            .into_iter()
            .map(|candidate| {
                let node = &self.nodes[candidate.index];
                SearchCandidate {
                    id: node.id,
                    location: node.location,
                    distance: candidate.distance,
                }
            })
            .collect()
    }

    fn connection_limit(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.max_connections * 2
        } else {
            self.config.max_connections
        }
    }

    fn distance(&self, query: &QuantizedVector, index: usize) -> f32 {
        query.distance(&self.nodes[index].vector, self.metric)
    }

    fn greedy_closest(&self, query: &QuantizedVector, mut current: usize, layer: usize) -> usize {
        let mut current_distance = self.distance(query, current);
        loop {
            let mut changed = false;
            if let Some(neighbors) = self.nodes[current].links.get(layer) {
                for neighbor in neighbors {
                    let distance = self.distance(query, *neighbor);
                    if distance < current_distance {
                        current = *neighbor;
                        current_distance = distance;
                        changed = true;
                    }
                }
            }
            if !changed {
                return current;
            }
        }
    }

    fn search_layer(
        &self,
        query: &QuantizedVector,
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<DistanceIndex> {
        let initial = DistanceIndex {
            distance: self.distance(query, entry),
            index: entry,
        };
        let mut visited = HashSet::from([entry]);
        let mut pending = BinaryHeap::from([Reverse(initial)]);
        let mut nearest = BinaryHeap::from([initial]);

        while let Some(Reverse(candidate)) = pending.pop() {
            let worst = nearest
                .peek()
                .map(|value| value.distance)
                .unwrap_or(f32::MAX);
            if nearest.len() >= ef && candidate.distance > worst {
                break;
            }
            if let Some(neighbors) = self.nodes[candidate.index].links.get(layer) {
                for neighbor in neighbors {
                    if !visited.insert(*neighbor) {
                        continue;
                    }
                    let candidate = DistanceIndex {
                        distance: self.distance(query, *neighbor),
                        index: *neighbor,
                    };
                    let worst = nearest
                        .peek()
                        .map(|value| value.distance)
                        .unwrap_or(f32::MAX);
                    if nearest.len() < ef || candidate.distance < worst {
                        pending.push(Reverse(candidate));
                        nearest.push(candidate);
                        if nearest.len() > ef {
                            nearest.pop();
                        }
                    }
                }
            }
        }
        let mut output = nearest.into_vec();
        output.sort_by(|left, right| left.distance.total_cmp(&right.distance));
        output
    }

    fn prune_links(&mut self, node: usize, layer: usize) {
        let limit = self.connection_limit(layer);
        if self.nodes[node].links[layer].len() <= limit {
            return;
        }
        let vector = self.nodes[node].vector.clone();
        let mut links = self.nodes[node].links[layer].clone();
        links.sort_by(|left, right| {
            vector
                .distance(&self.nodes[*left].vector, self.metric)
                .total_cmp(&vector.distance(&self.nodes[*right].vector, self.metric))
        });
        links.dedup();
        links.truncate(limit);
        self.nodes[node].links[layer] = links;
    }
}

fn deterministic_level(id: u64, max_connections: usize) -> usize {
    let hash = blake3::hash(&id.to_le_bytes());
    let random = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
    let mut divisor = max_connections.max(2) as u64;
    let mut level = 0;
    while level < 16 && random % divisor == 0 {
        level += 1;
        divisor = divisor.saturating_mul(max_connections as u64);
        if divisor == u64::MAX {
            break;
        }
    }
    level
}
