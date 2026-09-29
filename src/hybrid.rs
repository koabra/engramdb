//! Persistent fused-block catalog, HNSW search, and graph/temporal traversal.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;

use crate::fused::{FusedBlockBuilder, FusedBlockView, FusedNode, FusedNodeView, QuantizedVector};
use crate::io::{AlignedPage, DirectIo, FUSED_BLOCK_LAYOUT, FUSED_BLOCK_SIZE};
use crate::tree::Hash;
use crate::{Error, Result};

const DEFAULT_HNSW_M: usize = 16;
const DEFAULT_EF_SEARCH: usize = 256;
const MAX_HNSW_LEVEL: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockRef {
    pub hash: Hash,
    pub offset: u64,
    pub slot: u16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub id: Hash,
    pub cosine_similarity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraversalMatch {
    pub id: Hash,
    pub cosine_similarity: f32,
    pub depth: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TriModalQuery<'a> {
    pub vector: &'a [f32],
    pub minimum_cosine: f32,
    pub assertion_before: u64,
    pub valid_at: i64,
    pub max_hops: usize,
    pub edge_type: Option<u16>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HybridIndexStats {
    pub nodes: usize,
    pub blocks: usize,
    pub physical_bytes: u64,
    pub logical_bytes: u64,
}

struct StoredBlock {
    reference_hash: Hash,
    offset: u64,
    bytes: AlignedPage,
}

#[derive(Clone, Copy)]
struct NodeLocation {
    block: usize,
    slot: usize,
    reference: BlockRef,
}

/// Prototype Phase 2 index. Fused blocks are durable; the lightweight HNSW
/// graph is rebuilt deterministically on open from quantized vectors in blocks.
pub struct HybridIndex {
    io: Arc<DirectIo>,
    blocks: Vec<StoredBlock>,
    locations: Vec<NodeLocation>,
    by_id: HashMap<Hash, usize>,
    hnsw: Hnsw,
    logical_bytes: u64,
    vector_dimension: Option<usize>,
}

impl HybridIndex {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let io = Arc::new(DirectIo::open_with_layout(path, 256, FUSED_BLOCK_LAYOUT)?);
        let mut index = Self {
            io,
            blocks: Vec::new(),
            locations: Vec::new(),
            by_id: HashMap::new(),
            hnsw: Hnsw::new(DEFAULT_HNSW_M, DEFAULT_EF_SEARCH),
            logical_bytes: 0,
            vector_dimension: None,
        };
        index.load_blocks()?;
        index.rebuild_hnsw()?;
        Ok(index)
    }

    pub fn insert(&mut self, epoch: u64, nodes: Vec<FusedNode>) -> Result<Vec<BlockRef>> {
        if nodes.is_empty() {
            return Ok(Vec::new());
        }
        let mut seen = HashSet::with_capacity(nodes.len());
        for node in &nodes {
            if self
                .vector_dimension
                .is_some_and(|dimension| dimension != node.vector.len())
                || nodes
                    .first()
                    .is_some_and(|first| first.vector.len() != node.vector.len())
            {
                return Err(Error::DimensionMismatch {
                    expected: self
                        .vector_dimension
                        .unwrap_or_else(|| nodes[0].vector.len()),
                    actual: node.vector.len(),
                });
            }
            if self.by_id.contains_key(&node.id) || !seen.insert(node.id) {
                return Err(Error::Invariant(format!(
                    "duplicate fused node id {}",
                    node.id
                )));
            }
        }

        let mut builder = FusedBlockBuilder::new(epoch);
        let mut encoded = Vec::new();
        for node in nodes {
            if !builder.is_empty() && !builder.can_fit(&node)? {
                encoded.push(builder.finish()?);
                builder = FusedBlockBuilder::new(epoch);
            }
            builder.push(node)?;
        }
        if !builder.is_empty() {
            encoded.push(builder.finish()?);
        }

        let first_block = self.blocks.len();
        for block in encoded {
            let offset = self.io.append(&block)?;
            let reference_hash = Hash(*blake3::hash(block.as_slice()).as_bytes());
            self.blocks.push(StoredBlock {
                reference_hash,
                offset,
                bytes: block,
            });
        }
        self.io.sync()?;
        for block in first_block..self.blocks.len() {
            self.catalog_block(block)?;
        }
        self.rebuild_hnsw()?;
        Ok(self.locations[self.locations.len() - seen.len()..]
            .iter()
            .map(|location| location.reference)
            .collect())
    }

    pub fn get(&self, id: Hash) -> Result<Option<FusedNodeView<'_>>> {
        self.by_id
            .get(&id)
            .map(|index| self.node(*index))
            .transpose()
    }

    pub fn nearest(&self, vector: &[f32], count: usize) -> Result<Vec<SearchResult>> {
        if count == 0 || self.locations.is_empty() {
            return Ok(Vec::new());
        }
        let query = QuantizedVector::from_f32(vector)?;
        let candidates = self.hnsw.search(count, |index| {
            self.node(index)
                .and_then(|node| node.cosine_similarity(&query))
        })?;
        Ok(candidates
            .into_iter()
            .map(|scored| SearchResult {
                id: self.node(scored.index).expect("validated HNSW node").id(),
                cosine_similarity: scored.score,
            })
            .collect())
    }

    pub fn exact_nearest(&self, vector: &[f32], count: usize) -> Result<Vec<SearchResult>> {
        let query = QuantizedVector::from_f32(vector)?;
        let mut scored = Vec::with_capacity(self.locations.len());
        for index in 0..self.locations.len() {
            scored.push(Scored {
                index,
                score: self.node(index)?.cosine_similarity(&query)?,
            });
        }
        scored.sort_by(|left, right| right.cmp(left));
        scored.truncate(count);
        Ok(scored
            .into_iter()
            .map(|result| SearchResult {
                id: self.node(result.index).expect("validated exact node").id(),
                cosine_similarity: result.score,
            })
            .collect())
    }

    pub fn traverse_filtered<F>(
        &self,
        start: Hash,
        vector: &[f32],
        max_hops: usize,
        edge_type: Option<u16>,
        mut predicate: F,
    ) -> Result<Vec<TraversalMatch>>
    where
        F: for<'a> FnMut(FusedNodeView<'a>, usize, f32) -> bool,
    {
        let query = QuantizedVector::from_f32(vector)?;
        let start_index = *self
            .by_id
            .get(&start)
            .ok_or_else(|| Error::Invariant(format!("unknown fused node {start}")))?;
        let mut queue = VecDeque::from([(start_index, 0_usize)]);
        let mut visited = HashSet::from([start_index]);
        let mut matches = Vec::new();
        while let Some((index, depth)) = queue.pop_front() {
            let node = self.node(index)?;
            let similarity = node.cosine_similarity(&query)?;
            if predicate(node, depth, similarity) {
                matches.push(TraversalMatch {
                    id: node.id(),
                    cosine_similarity: similarity,
                    depth,
                });
            }
            if depth == max_hops {
                continue;
            }
            for edge in node.edges() {
                if edge_type.is_some_and(|expected| edge.edge_type != expected) {
                    continue;
                }
                if let Some(target) = self.by_id.get(&edge.target).copied() {
                    if visited.insert(target) {
                        queue.push_back((target, depth + 1));
                    }
                }
            }
        }
        matches.sort_by(|left, right| {
            right
                .cosine_similarity
                .total_cmp(&left.cosine_similarity)
                .then_with(|| left.depth.cmp(&right.depth))
        });
        Ok(matches)
    }

    pub fn tri_modal_query(
        &self,
        start: Hash,
        query: TriModalQuery<'_>,
    ) -> Result<Vec<TraversalMatch>> {
        self.traverse_filtered(
            start,
            query.vector,
            query.max_hops,
            query.edge_type,
            |node, _, similarity| {
                similarity >= query.minimum_cosine
                    && node.temporal().any(|point| {
                        point.assertion_time < query.assertion_before
                            && point.valid_time <= query.valid_at
                    })
            },
        )
    }

    pub fn stats(&self) -> HybridIndexStats {
        HybridIndexStats {
            nodes: self.locations.len(),
            blocks: self.blocks.len(),
            physical_bytes: self.blocks.len() as u64 * FUSED_BLOCK_SIZE as u64,
            logical_bytes: self.logical_bytes,
        }
    }

    pub fn sync(&self) -> Result<()> {
        self.io.sync()
    }

    fn load_blocks(&mut self) -> Result<()> {
        let mut offset = 0;
        while offset < self.io.len() {
            let bytes = self.io.read(offset)?;
            FusedBlockView::parse(bytes.as_slice(), offset)?;
            let reference_hash = Hash(*blake3::hash(bytes.as_slice()).as_bytes());
            let block = self.blocks.len();
            self.blocks.push(StoredBlock {
                reference_hash,
                offset,
                bytes,
            });
            self.catalog_block(block)?;
            offset += FUSED_BLOCK_SIZE as u64;
        }
        Ok(())
    }

    fn catalog_block(&mut self, block: usize) -> Result<()> {
        let stored = &self.blocks[block];
        let view = FusedBlockView::parse(stored.bytes.as_slice(), stored.offset)?;
        for slot in 0..view.len() {
            let node = view.node(slot, stored.offset)?;
            match self.vector_dimension {
                Some(dimension) if dimension != node.vector().len() => {
                    return Err(Error::DimensionMismatch {
                        expected: dimension,
                        actual: node.vector().len(),
                    });
                }
                None => self.vector_dimension = Some(node.vector().len()),
                Some(_) => {}
            }
            if self.by_id.contains_key(&node.id()) {
                return Err(Error::Invariant(format!(
                    "duplicate fused node id {} on disk",
                    node.id()
                )));
            }
            let index = self.locations.len();
            let reference = BlockRef {
                hash: stored.reference_hash,
                offset: stored.offset,
                slot: slot as u16,
            };
            self.logical_bytes += 64
                + node.vector().len() as u64
                + node.temporal().count() as u64 * 16
                + node.edges().count() as u64 * 40;
            self.locations.push(NodeLocation {
                block,
                slot,
                reference,
            });
            self.by_id.insert(node.id(), index);
        }
        Ok(())
    }

    fn node(&self, index: usize) -> Result<FusedNodeView<'_>> {
        let location = self
            .locations
            .get(index)
            .ok_or_else(|| Error::Invariant("HNSW references an unknown node".to_owned()))?;
        let block = &self.blocks[location.block];
        FusedBlockView::trusted(block.bytes.as_slice()).node(location.slot, block.offset)
    }

    fn rebuild_hnsw(&mut self) -> Result<()> {
        let mut hnsw = Hnsw::new(DEFAULT_HNSW_M, DEFAULT_EF_SEARCH);
        for index in 0..self.locations.len() {
            let id = self.node(index)?.id();
            hnsw.insert(index, id, |left, right| {
                let left_node = self.node(left)?;
                left_node.cosine_similarity_to(self.node(right)?)
            })?;
        }
        self.hnsw = hnsw;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Scored {
    index: usize,
    score: f32,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.index.cmp(&other.index))
    }
}

struct HnswNode {
    level: usize,
    neighbors: Vec<Vec<usize>>,
}

struct Hnsw {
    nodes: Vec<HnswNode>,
    entry: Option<usize>,
    max_level: usize,
    m: usize,
    ef_search: usize,
}

impl Hnsw {
    fn new(m: usize, ef_search: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry: None,
            max_level: 0,
            m,
            ef_search,
        }
    }

    fn insert<F>(&mut self, index: usize, id: Hash, mut similarity: F) -> Result<()>
    where
        F: FnMut(usize, usize) -> Result<f32>,
    {
        debug_assert_eq!(index, self.nodes.len());
        let level = deterministic_level(id);
        self.nodes.push(HnswNode {
            level,
            neighbors: vec![Vec::new(); level + 1],
        });
        if index == 0 {
            self.entry = Some(0);
            self.max_level = level;
            return Ok(());
        }

        for layer in 0..=level {
            let mut candidates = Vec::new();
            for candidate in 0..index {
                if self.nodes[candidate].level >= layer {
                    candidates.push(Scored {
                        index: candidate,
                        score: similarity(index, candidate)?,
                    });
                }
            }
            candidates.sort_by(|left, right| right.cmp(left));
            candidates.truncate(self.m);
            self.nodes[index].neighbors[layer] =
                candidates.iter().map(|candidate| candidate.index).collect();
            for candidate in candidates {
                let neighbor = candidate.index;
                self.nodes[neighbor].neighbors[layer].push(index);
                if self.nodes[neighbor].neighbors[layer].len() > self.m {
                    let mut ranked = self.nodes[neighbor].neighbors[layer]
                        .iter()
                        .copied()
                        .map(|candidate| {
                            Ok(Scored {
                                index: candidate,
                                score: similarity(neighbor, candidate)?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    ranked.sort_by(|left, right| right.cmp(left));
                    ranked.truncate(self.m);
                    self.nodes[neighbor].neighbors[layer] = ranked
                        .into_iter()
                        .map(|candidate| candidate.index)
                        .collect();
                }
            }
        }
        if level > self.max_level {
            self.entry = Some(index);
            self.max_level = level;
        }
        Ok(())
    }

    fn search<F>(&self, count: usize, mut score: F) -> Result<Vec<Scored>>
    where
        F: FnMut(usize) -> Result<f32>,
    {
        let Some(mut entry) = self.entry else {
            return Ok(Vec::new());
        };
        let mut entry_score = score(entry)?;
        for layer in (1..=self.max_level).rev() {
            loop {
                let mut improved = false;
                if self.nodes[entry].level < layer {
                    break;
                }
                for neighbor in &self.nodes[entry].neighbors[layer] {
                    let candidate_score = score(*neighbor)?;
                    if candidate_score > entry_score {
                        entry = *neighbor;
                        entry_score = candidate_score;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
        }
        self.search_layer(entry, self.ef_search.max(count), 0, score)
            .map(|mut results| {
                results.truncate(count);
                results
            })
    }

    fn search_layer<F>(
        &self,
        entry: usize,
        ef: usize,
        layer: usize,
        mut score: F,
    ) -> Result<Vec<Scored>>
    where
        F: FnMut(usize) -> Result<f32>,
    {
        let initial = Scored {
            index: entry,
            score: score(entry)?,
        };
        let mut candidates = BinaryHeap::from([initial]);
        let mut results = BinaryHeap::from([Reverse(initial)]);
        let mut visited = HashSet::from([entry]);
        while let Some(candidate) = candidates.pop() {
            let worst = results.peek().map(|result| result.0.score).unwrap_or(-1.0);
            if results.len() >= ef && candidate.score < worst {
                break;
            }
            for neighbor in &self.nodes[candidate.index].neighbors[layer] {
                if !visited.insert(*neighbor) {
                    continue;
                }
                let scored = Scored {
                    index: *neighbor,
                    score: score(*neighbor)?,
                };
                if results.len() < ef
                    || scored.score > results.peek().expect("non-empty results").0.score
                {
                    candidates.push(scored);
                    results.push(Reverse(scored));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut output: Vec<Scored> = results.into_iter().map(|result| result.0).collect();
        output.sort_by(|left, right| right.cmp(left));
        Ok(output)
    }
}

fn deterministic_level(id: Hash) -> usize {
    let seed = u64::from_le_bytes(id.0[..8].try_into().unwrap());
    (seed.trailing_zeros() as usize / 2).min(MAX_HNSW_LEVEL)
}
