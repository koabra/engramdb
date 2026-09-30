//! Persistent fused-block catalog, HNSW search, and graph/temporal traversal.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;

use crate::fused::{
    FusedBlockBuilder, FusedBlockView, FusedNode, FusedNodeView, GraphEdge, QuantizedVector,
    TemporalPoint,
};
use crate::io::{AlignedPage, DirectIo, FUSED_BLOCK_LAYOUT, FUSED_BLOCK_SIZE};
use crate::tree::Hash;
use crate::{Error, Result};

const DEFAULT_HNSW_M: usize = 32;
const DEFAULT_EF_CONSTRUCTION: usize = 256;
const DEFAULT_EF_SEARCH: usize = 256;
const MAX_HNSW_LEVEL: usize = 12;
const CHECKPOINT_MAGIC: &[u8; 8] = b"ENGHNSW1";
const CHECKPOINT_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockRef {
    pub hash: Hash,
    pub offset: u64,
    pub slot: u16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub id: Hash,
    /// Cosine similarity for [`VectorMetric::Cosine`], or negative squared
    /// distance for [`VectorMetric::SquaredL2`]. Higher is always better.
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraversalMatch {
    pub id: Hash,
    pub cosine_similarity: f32,
    pub depth: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeProjection {
    pub id: Hash,
    pub temporal: Vec<TemporalPoint>,
    pub vector: Vec<i8>,
    pub quantization_scale: f32,
    pub edges: Vec<GraphEdge>,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VectorMetric {
    #[default]
    Cosine,
    SquaredL2,
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
    metric: VectorMetric,
}

impl HybridIndex {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_metric(path, VectorMetric::Cosine)
    }

    pub fn open_with_metric(path: impl AsRef<Path>, metric: VectorMetric) -> Result<Self> {
        let io = Arc::new(DirectIo::open_with_layout(path, 256, FUSED_BLOCK_LAYOUT)?);
        let mut index = Self::empty_with_io(io, metric);
        index.load_blocks()?;
        index.rebuild_hnsw()?;
        Ok(index)
    }

    pub(crate) fn empty_with_io(io: Arc<DirectIo>, metric: VectorMetric) -> Self {
        Self {
            io,
            blocks: Vec::new(),
            locations: Vec::new(),
            by_id: HashMap::new(),
            hnsw: Hnsw::new(DEFAULT_HNSW_M, DEFAULT_EF_SEARCH),
            logical_bytes: 0,
            vector_dimension: None,
            metric,
        }
    }

    pub fn insert(&mut self, epoch: u64, nodes: Vec<FusedNode>) -> Result<Vec<BlockRef>> {
        self.insert_inner(epoch, nodes, true)
    }

    pub(crate) fn insert_uncommitted(
        &mut self,
        epoch: u64,
        nodes: Vec<FusedNode>,
    ) -> Result<Vec<BlockRef>> {
        self.insert_inner(epoch, nodes, false)
    }

    fn insert_inner(
        &mut self,
        epoch: u64,
        nodes: Vec<FusedNode>,
        sync: bool,
    ) -> Result<Vec<BlockRef>> {
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
        if sync {
            self.io.sync()?;
        }
        for block in first_block..self.blocks.len() {
            self.catalog_block(block)?;
        }
        self.rebuild_hnsw()?;
        Ok(self.locations[self.locations.len() - seen.len()..]
            .iter()
            .map(|location| location.reference)
            .collect())
    }

    pub(crate) fn checkpoint_bytes(&self) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        output.extend_from_slice(CHECKPOINT_MAGIC);
        output.push(CHECKPOINT_VERSION);
        output.push(match self.metric {
            VectorMetric::Cosine => 1,
            VectorMetric::SquaredL2 => 2,
        });
        put_u16(&mut output, self.hnsw.m as u16);
        put_u32(&mut output, self.hnsw.ef_search as u32);
        put_u32(&mut output, self.hnsw.ef_construction as u32);
        put_u16(&mut output, self.hnsw.max_level as u16);
        put_u16(&mut output, 0);
        put_u64(
            &mut output,
            self.hnsw
                .entry
                .map(|entry| entry as u64)
                .unwrap_or(u64::MAX),
        );
        put_u64(&mut output, self.locations.len() as u64);
        for (index, location) in self.locations.iter().enumerate() {
            let node = self.node(index)?;
            output.extend_from_slice(&node.id().0);
            output.extend_from_slice(&location.reference.hash.0);
            put_u64(&mut output, location.reference.offset);
            put_u16(&mut output, location.reference.slot);
            let hnsw_node = &self.hnsw.nodes[index];
            put_u16(&mut output, hnsw_node.level as u16);
            put_u16(&mut output, hnsw_node.neighbors.len() as u16);
            put_u16(&mut output, 0);
            for layer in &hnsw_node.neighbors {
                put_u32(&mut output, layer.len() as u32);
                for neighbor in layer {
                    put_u64(&mut output, *neighbor as u64);
                }
            }
        }
        Ok(output)
    }

    pub(crate) fn from_checkpoint(
        io: Arc<DirectIo>,
        bytes: &[u8],
        fused_length: u64,
    ) -> Result<Self> {
        let mut cursor = CheckpointCursor::new(bytes);
        if cursor.fixed::<8>()? != *CHECKPOINT_MAGIC {
            return Err(Error::Invariant(
                "invalid durable HNSW checkpoint magic".to_owned(),
            ));
        }
        if cursor.u8()? != CHECKPOINT_VERSION {
            return Err(Error::Invariant(
                "unsupported durable HNSW checkpoint version".to_owned(),
            ));
        }
        let metric = match cursor.u8()? {
            1 => VectorMetric::Cosine,
            2 => VectorMetric::SquaredL2,
            _ => {
                return Err(Error::Invariant("invalid durable HNSW metric".to_owned()));
            }
        };
        let m = cursor.u16()? as usize;
        let ef_search = cursor.u32()? as usize;
        let ef_construction = cursor.u32()? as usize;
        let max_level = cursor.u16()? as usize;
        cursor.u16()?;
        let entry_value = cursor.u64()?;
        let node_count = usize::try_from(cursor.u64()?)
            .map_err(|_| Error::Invariant("HNSW node count exceeds usize".to_owned()))?;
        let mut index = Self::empty_with_io(io, metric);
        let mut blocks_by_offset = HashMap::<u64, usize>::new();
        let mut hnsw_nodes = Vec::with_capacity(node_count);
        for node_index in 0..node_count {
            let id = Hash(cursor.fixed::<32>()?);
            let block_hash = Hash(cursor.fixed::<32>()?);
            let offset = cursor.u64()?;
            let slot = cursor.u16()? as usize;
            let level = cursor.u16()? as usize;
            let layer_count = cursor.u16()? as usize;
            cursor.u16()?;
            if offset
                .checked_add(FUSED_BLOCK_SIZE as u64)
                .is_none_or(|end| end > fused_length)
            {
                return Err(Error::CorruptMetadata {
                    offset,
                    reason: "HNSW checkpoint references data beyond fused watermark".to_owned(),
                });
            }
            if layer_count != level + 1 {
                return Err(Error::Invariant(
                    "durable HNSW layer count does not match level".to_owned(),
                ));
            }
            let block = if let Some(block) = blocks_by_offset.get(&offset).copied() {
                if index.blocks[block].reference_hash != block_hash {
                    return Err(Error::Invariant(
                        "checkpoint gives conflicting hashes for one fused offset".to_owned(),
                    ));
                }
                block
            } else {
                let block_bytes = index.io.read(offset)?;
                FusedBlockView::parse(block_bytes.as_slice(), offset)?;
                let actual_hash = Hash(*blake3::hash(block_bytes.as_slice()).as_bytes());
                if actual_hash != block_hash {
                    return Err(Error::CorruptPage {
                        offset,
                        reason: "durable HNSW block hash mismatch".to_owned(),
                    });
                }
                let block = index.blocks.len();
                index.blocks.push(StoredBlock {
                    reference_hash: actual_hash,
                    offset,
                    bytes: block_bytes,
                });
                blocks_by_offset.insert(offset, block);
                block
            };
            let view =
                FusedBlockView::trusted(index.blocks[block].bytes.as_slice()).node(slot, offset)?;
            if view.id() != id {
                return Err(Error::CorruptPage {
                    offset,
                    reason: "durable HNSW node id does not match fused slot".to_owned(),
                });
            }
            match index.vector_dimension {
                Some(dimension) if dimension != view.vector().len() => {
                    return Err(Error::DimensionMismatch {
                        expected: dimension,
                        actual: view.vector().len(),
                    });
                }
                None => index.vector_dimension = Some(view.vector().len()),
                Some(_) => {}
            }
            if index.by_id.insert(id, node_index).is_some() {
                return Err(Error::Invariant(
                    "durable HNSW checkpoint contains duplicate node ids".to_owned(),
                ));
            }
            index.logical_bytes += 64
                + view.vector().len() as u64
                + view.temporal().count() as u64 * 16
                + view.edges().count() as u64 * 40;
            index.locations.push(NodeLocation {
                block,
                slot,
                reference: BlockRef {
                    hash: block_hash,
                    offset,
                    slot: slot as u16,
                },
            });
            let mut neighbors = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let neighbor_count = cursor.u32()? as usize;
                let mut layer = Vec::with_capacity(neighbor_count);
                for _ in 0..neighbor_count {
                    let neighbor = usize::try_from(cursor.u64()?).map_err(|_| {
                        Error::Invariant("HNSW neighbor index exceeds usize".to_owned())
                    })?;
                    if neighbor >= node_count {
                        return Err(Error::Invariant(
                            "durable HNSW neighbor is outside checkpoint".to_owned(),
                        ));
                    }
                    layer.push(neighbor);
                }
                neighbors.push(layer);
            }
            hnsw_nodes.push(HnswNode { level, neighbors });
        }
        if !cursor.is_finished() {
            return Err(Error::Invariant(
                "trailing bytes in durable HNSW checkpoint".to_owned(),
            ));
        }
        let entry = if entry_value == u64::MAX {
            None
        } else {
            let entry = usize::try_from(entry_value)
                .map_err(|_| Error::Invariant("HNSW entry exceeds usize".to_owned()))?;
            if entry >= node_count {
                return Err(Error::Invariant(
                    "durable HNSW entry is outside checkpoint".to_owned(),
                ));
            }
            Some(entry)
        };
        if node_count == 0 && entry.is_some() || node_count > 0 && entry.is_none() {
            return Err(Error::Invariant(
                "durable HNSW entry does not match node count".to_owned(),
            ));
        }
        index.hnsw = Hnsw {
            nodes: hnsw_nodes,
            entry,
            max_level,
            m,
            ef_search,
            ef_construction,
        };
        Ok(index)
    }

    pub fn get(&self, id: Hash) -> Result<Option<FusedNodeView<'_>>> {
        self.by_id
            .get(&id)
            .map(|index| self.node(*index))
            .transpose()
    }

    pub fn nearest(&self, vector: &[f32], count: usize) -> Result<Vec<SearchResult>> {
        self.nearest_with_ef(vector, count, DEFAULT_EF_SEARCH)
    }

    pub fn nearest_with_ef(
        &self,
        vector: &[f32],
        count: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        if count == 0 || self.locations.is_empty() {
            return Ok(Vec::new());
        }
        let query = QuantizedVector::from_f32(vector)?;
        let candidates = self.hnsw.search(count, ef_search.max(count), |index| {
            self.node(index)
                .and_then(|node| self.score_query(node, &query))
        })?;
        Ok(candidates
            .into_iter()
            .map(|scored| SearchResult {
                id: self.node(scored.index).expect("validated HNSW node").id(),
                score: scored.score,
            })
            .collect())
    }

    pub fn exact_nearest(&self, vector: &[f32], count: usize) -> Result<Vec<SearchResult>> {
        let query = QuantizedVector::from_f32(vector)?;
        let mut scored = Vec::with_capacity(self.locations.len());
        for index in 0..self.locations.len() {
            scored.push(Scored {
                index,
                score: self.score_query(self.node(index)?, &query)?,
            });
        }
        scored.sort_by(|left, right| right.cmp(left));
        scored.truncate(count);
        Ok(scored
            .into_iter()
            .map(|result| SearchResult {
                id: self.node(result.index).expect("validated exact node").id(),
                score: result.score,
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

    pub fn tri_modal_query_vector_first(
        &self,
        start: Hash,
        query: TriModalQuery<'_>,
        limit: usize,
    ) -> Result<Vec<TraversalMatch>> {
        let reachable = self.reachable_depths(start, query.max_hops, query.edge_type)?;
        let mut output = Vec::new();
        for candidate in self.exact_nearest(query.vector, self.locations.len())? {
            if candidate.score < query.minimum_cosine {
                break;
            }
            let Some(depth) = reachable.get(&candidate.id).copied() else {
                continue;
            };
            let index = self.by_id[&candidate.id];
            let node = self.node(index)?;
            if node.temporal().any(|point| {
                point.assertion_time < query.assertion_before && point.valid_time <= query.valid_at
            }) {
                output.push(TraversalMatch {
                    id: candidate.id,
                    cosine_similarity: candidate.score,
                    depth,
                });
                if output.len() == limit {
                    break;
                }
            }
        }
        Ok(output)
    }

    pub fn projections(&self) -> Result<Vec<NodeProjection>> {
        (0..self.locations.len())
            .map(|index| {
                let node = self.node(index)?;
                Ok(NodeProjection {
                    id: node.id(),
                    temporal: node.temporal().collect(),
                    vector: node.vector().to_vec(),
                    quantization_scale: node.quantization_scale(),
                    edges: node.edges().collect(),
                })
            })
            .collect()
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

    fn reachable_depths(
        &self,
        start: Hash,
        max_hops: usize,
        edge_type: Option<u16>,
    ) -> Result<HashMap<Hash, usize>> {
        let start_index = *self
            .by_id
            .get(&start)
            .ok_or_else(|| Error::Invariant(format!("unknown fused node {start}")))?;
        let mut queue = VecDeque::from([(start_index, 0_usize)]);
        let mut visited = HashMap::from([(start, 0_usize)]);
        while let Some((index, depth)) = queue.pop_front() {
            if depth == max_hops {
                continue;
            }
            for edge in self.node(index)?.edges() {
                if edge_type.is_some_and(|expected| edge.edge_type != expected) {
                    continue;
                }
                let Some(target) = self.by_id.get(&edge.target).copied() else {
                    continue;
                };
                if let std::collections::hash_map::Entry::Vacant(entry) = visited.entry(edge.target)
                {
                    entry.insert(depth + 1);
                    queue.push_back((target, depth + 1));
                }
            }
        }
        Ok(visited)
    }

    fn rebuild_hnsw(&mut self) -> Result<()> {
        let mut hnsw = Hnsw::new(DEFAULT_HNSW_M, DEFAULT_EF_SEARCH);
        for index in 0..self.locations.len() {
            let id = self.node(index)?.id();
            hnsw.insert(index, id, |left, right| {
                let left_node = self.node(left)?;
                self.score_nodes(left_node, self.node(right)?)
            })?;
        }
        self.hnsw = hnsw;
        Ok(())
    }

    fn score_query(&self, node: FusedNodeView<'_>, query: &QuantizedVector) -> Result<f32> {
        match self.metric {
            VectorMetric::Cosine => node.cosine_similarity(query),
            VectorMetric::SquaredL2 => node.squared_l2_score(query),
        }
    }

    fn score_nodes(&self, left: FusedNodeView<'_>, right: FusedNodeView<'_>) -> Result<f32> {
        match self.metric {
            VectorMetric::Cosine => left.cosine_similarity_to(right),
            VectorMetric::SquaredL2 => left.squared_l2_score_to(right),
        }
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

#[derive(Clone)]
struct HnswNode {
    level: usize,
    neighbors: Vec<Vec<usize>>,
}

#[derive(Clone)]
struct Hnsw {
    nodes: Vec<HnswNode>,
    entry: Option<usize>,
    max_level: usize,
    m: usize,
    ef_search: usize,
    ef_construction: usize,
}

impl Hnsw {
    fn new(m: usize, ef_search: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry: None,
            max_level: 0,
            m,
            ef_search,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
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

        let mut entry = self.entry.expect("non-empty HNSW has entry");
        let mut entry_score = similarity(index, entry)?;
        for layer in ((level + 1)..=self.max_level).rev() {
            loop {
                let mut improved = false;
                for neighbor in &self.nodes[entry].neighbors[layer] {
                    let score = similarity(index, *neighbor)?;
                    if score > entry_score {
                        entry = *neighbor;
                        entry_score = score;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
        }

        for layer in (0..=level.min(self.max_level)).rev() {
            let mut candidates = self.search_layer_for_node(
                entry,
                index,
                self.ef_construction,
                layer,
                &mut similarity,
            )?;
            candidates.truncate(self.m);
            self.nodes[index].neighbors[layer] =
                candidates.iter().map(|candidate| candidate.index).collect();
            if let Some(best) = candidates.first() {
                entry = best.index;
            }
            for candidate in &candidates {
                let neighbor = candidate.index;
                self.nodes[neighbor].neighbors[layer].push(index);
                if self.nodes[neighbor].neighbors[layer].len() > self.m * 2 {
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
                    ranked.truncate(self.m * 2);
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

    fn search_layer_for_node<F>(
        &self,
        entry: usize,
        query: usize,
        ef: usize,
        layer: usize,
        similarity: &mut F,
    ) -> Result<Vec<Scored>>
    where
        F: FnMut(usize, usize) -> Result<f32>,
    {
        let initial = Scored {
            index: entry,
            score: similarity(query, entry)?,
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
                    score: similarity(query, *neighbor)?,
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

    fn search<F>(&self, count: usize, ef_search: usize, mut score: F) -> Result<Vec<Scored>>
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
        self.search_layer(entry, ef_search.max(self.ef_search).max(count), 0, score)
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
    (seed.trailing_zeros() as usize / 4).min(MAX_HNSW_LEVEL)
}

fn put_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

struct CheckpointCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> CheckpointCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.fixed::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.fixed::<2>()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.fixed::<4>()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.fixed::<8>()?))
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.position + N > self.bytes.len() {
            return Err(Error::Invariant(
                "durable HNSW checkpoint is truncated".to_owned(),
            ));
        }
        let value = self.bytes[self.position..self.position + N]
            .try_into()
            .unwrap();
        self.position += N;
        Ok(value)
    }

    fn is_finished(&self) -> bool {
        self.position == self.bytes.len()
    }
}
