use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::hybrid::hnsw::{BlockLocation, HnswConfig, HnswIndex};
use crate::hybrid::layout::{FusedBlockBuilder, FusedBlockView, GraphEdge, HybridRecord};
use crate::hybrid::simd::{DistanceMetric, QuantizedVector};
use crate::{AlignedBlock, DirectIo, Error, IoStats, Result, FUSED_BLOCK_SIZE};

#[derive(Debug, Clone, Copy)]
pub struct RecordMetadata {
    pub id: u64,
    pub assertion_time: u64,
    pub valid_from: i64,
    pub valid_to: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryHit {
    pub id: u64,
    pub distance: f32,
    pub similarity: f32,
    pub depth: usize,
    pub assertion_time: u64,
    pub block_offset: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryStats {
    pub visited_records: usize,
    pub physical_block_reads: usize,
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub hits: Vec<QueryHit>,
    pub stats: QueryStats,
}

struct RecordSnapshot {
    metadata: RecordMetadata,
    vector: QuantizedVector,
    edges: Vec<GraphEdge>,
    location: BlockLocation,
}

struct QueryContext<'a> {
    io: &'a DirectIo<FUSED_BLOCK_SIZE>,
    blocks: HashMap<u64, Arc<AlignedBlock<FUSED_BLOCK_SIZE>>>,
    reads: usize,
}

impl<'a> QueryContext<'a> {
    fn new(io: &'a DirectIo<FUSED_BLOCK_SIZE>) -> Self {
        Self {
            io,
            blocks: HashMap::new(),
            reads: 0,
        }
    }

    fn block(&mut self, offset: u64) -> Result<Arc<AlignedBlock<FUSED_BLOCK_SIZE>>> {
        if let Some(block) = self.blocks.get(&offset) {
            return Ok(Arc::clone(block));
        }
        let block = Arc::new(self.io.read(offset)?);
        FusedBlockView::parse(block.as_slice())?;
        self.reads += 1;
        self.blocks.insert(offset, Arc::clone(&block));
        Ok(block)
    }
}

pub struct HybridIndex {
    directory: PathBuf,
    io: Arc<DirectIo<FUSED_BLOCK_SIZE>>,
    dimension: usize,
    metric: DistanceMetric,
    hnsw: HnswIndex,
    locations: HashMap<u64, Vec<BlockLocation>>,
}

impl HybridIndex {
    pub fn open(
        directory: impl AsRef<Path>,
        dimension: usize,
        metric: DistanceMetric,
        config: HnswConfig,
    ) -> Result<Self> {
        if dimension == 0 || dimension > u16::MAX as usize {
            return Err(Error::Invariant(
                "hybrid-index dimension must be in 1..=65535".to_owned(),
            ));
        }
        let directory = directory.as_ref().to_path_buf();
        std::fs::create_dir_all(&directory)?;
        let io = Arc::new(DirectIo::<FUSED_BLOCK_SIZE>::open(
            directory.join("fused.blocks"),
            128,
        )?);
        let mut index = Self {
            directory,
            io,
            dimension,
            metric,
            hnsw: HnswIndex::new(metric, config)?,
            locations: HashMap::new(),
        };
        index.rebuild()?;
        Ok(index)
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.hnsw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn disk_bytes(&self) -> u64 {
        self.io.len()
    }

    pub fn io_stats(&self) -> IoStats {
        self.io.stats()
    }

    pub fn insert(&mut self, epoch: u64, records: Vec<HybridRecord>) -> Result<usize> {
        if records.is_empty() {
            return Ok(0);
        }
        for record in &records {
            record.validate(self.dimension)?;
        }
        let mut encoded = Vec::new();
        let mut cursor = 0;
        while cursor < records.len() {
            let mut builder = FusedBlockBuilder::new(epoch, self.dimension)?;
            while cursor < records.len() {
                if builder.try_push(records[cursor].clone())? {
                    cursor += 1;
                } else {
                    break;
                }
            }
            encoded.push(builder.finish()?);
        }

        let mut appended = Vec::with_capacity(encoded.len());
        for block in encoded {
            let offset = self.io.append(&block)?;
            appended.push((offset, block));
        }
        self.io.sync()?;
        for (offset, block) in &appended {
            self.register_block(*offset, block.as_slice())?;
        }
        Ok(appended.len())
    }

    pub fn nearest(
        &self,
        query: &[f32],
        count: usize,
        valid_at: i64,
        asserted_before: u64,
    ) -> Result<QueryResult> {
        if query.len() != self.dimension {
            return Err(Error::Invariant(
                "query dimension does not match hybrid index".to_owned(),
            ));
        }
        let query = QuantizedVector::encode(query)?;
        let candidates = self.hnsw.search(
            &query,
            self.hnsw.len().min(count.saturating_mul(32).max(128)),
            None,
        );
        let mut context = QueryContext::new(&self.io);
        let mut seen = HashSet::new();
        let mut hits = Vec::new();
        for candidate in candidates {
            if !seen.insert(candidate.id) {
                continue;
            }
            let Some(snapshot) =
                self.snapshot_at(candidate.id, valid_at, asserted_before, &mut context)?
            else {
                continue;
            };
            let distance = query.distance(&snapshot.vector, self.metric);
            hits.push(QueryHit {
                id: snapshot.metadata.id,
                distance,
                similarity: cosine_similarity(&query, &snapshot.vector),
                depth: 0,
                assertion_time: snapshot.metadata.assertion_time,
                block_offset: snapshot.location.block_offset,
            });
        }
        hits.sort_by(|left, right| left.distance.total_cmp(&right.distance));
        hits.truncate(count);
        Ok(QueryResult {
            hits,
            stats: QueryStats {
                visited_records: seen.len(),
                physical_block_reads: context.reads,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn traverse<F>(
        &self,
        root: u64,
        query: &[f32],
        max_hops: usize,
        minimum_similarity: f32,
        valid_at: i64,
        asserted_before: u64,
        predicate: F,
    ) -> Result<QueryResult>
    where
        F: Fn(RecordMetadata) -> bool,
    {
        if query.len() != self.dimension {
            return Err(Error::Invariant(
                "query dimension does not match hybrid index".to_owned(),
            ));
        }
        let query = QuantizedVector::encode(query)?;
        let mut context = QueryContext::new(&self.io);
        let mut pending = VecDeque::from([(root, 0_usize)]);
        let mut visited = HashSet::new();
        let mut hits = Vec::new();
        while let Some((id, depth)) = pending.pop_front() {
            if depth > max_hops || !visited.insert(id) {
                continue;
            }
            let Some(snapshot) = self.snapshot_at(id, valid_at, asserted_before, &mut context)?
            else {
                continue;
            };
            let similarity = cosine_similarity(&query, &snapshot.vector);
            if similarity >= minimum_similarity && predicate(snapshot.metadata) {
                hits.push(QueryHit {
                    id,
                    distance: 1.0 - similarity,
                    similarity,
                    depth,
                    assertion_time: snapshot.metadata.assertion_time,
                    block_offset: snapshot.location.block_offset,
                });
            }
            if depth < max_hops {
                for edge in snapshot.edges {
                    pending.push_back((edge.target, depth + 1));
                }
            }
        }
        hits.sort_by(|left, right| {
            right
                .similarity
                .total_cmp(&left.similarity)
                .then_with(|| left.depth.cmp(&right.depth))
        });
        Ok(QueryResult {
            hits,
            stats: QueryStats {
                visited_records: visited.len(),
                physical_block_reads: context.reads,
            },
        })
    }

    fn rebuild(&mut self) -> Result<()> {
        let mut offset = 0;
        while offset < self.io.len() {
            let block = self.io.read(offset)?;
            match FusedBlockView::parse(block.as_slice()) {
                Ok(_) => self.register_block(offset, block.as_slice())?,
                Err(_) if offset + FUSED_BLOCK_SIZE as u64 == self.io.len() => {
                    self.io.truncate(offset)?;
                    break;
                }
                Err(error) => return Err(error),
            }
            offset += FUSED_BLOCK_SIZE as u64;
        }
        Ok(())
    }

    fn register_block(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let block = FusedBlockView::parse(bytes)?;
        if block.dimension() != self.dimension {
            return Err(Error::Invariant(format!(
                "block dimension {} does not match index dimension {}",
                block.dimension(),
                self.dimension
            )));
        }
        for slot in 0..block.len() {
            let record = block.record(slot)?;
            let location = BlockLocation {
                block_offset: offset,
                slot: slot as u16,
            };
            let vector = QuantizedVector::from_parts(record.quantized_vector(), record.scale())?;
            self.hnsw
                .insert(index_key(offset, slot), record.id(), vector, location)?;
            self.locations
                .entry(record.id())
                .or_default()
                .push(location);
        }
        Ok(())
    }

    fn snapshot_at(
        &self,
        id: u64,
        valid_at: i64,
        asserted_before: u64,
        context: &mut QueryContext<'_>,
    ) -> Result<Option<RecordSnapshot>> {
        let Some(locations) = self.locations.get(&id) else {
            return Ok(None);
        };
        let mut selected = None;
        for location in locations {
            let block = context.block(location.block_offset)?;
            let view = FusedBlockView::parse(block.as_slice())?;
            let record = view.record(location.slot as usize)?;
            if record.assertion_time() > asserted_before
                || record.valid_from() > valid_at
                || valid_at >= record.valid_to()
            {
                continue;
            }
            if selected
                .as_ref()
                .map(|current: &RecordSnapshot| {
                    current.metadata.assertion_time >= record.assertion_time()
                })
                .unwrap_or(false)
            {
                continue;
            }
            selected = Some(RecordSnapshot {
                metadata: RecordMetadata {
                    id,
                    assertion_time: record.assertion_time(),
                    valid_from: record.valid_from(),
                    valid_to: record.valid_to(),
                },
                vector: QuantizedVector::from_parts(record.quantized_vector(), record.scale())?,
                edges: record.edges().collect(),
                location: *location,
            });
        }
        Ok(selected)
    }
}

fn index_key(offset: u64, slot: usize) -> u64 {
    (offset / FUSED_BLOCK_SIZE as u64) << 16 | slot as u64
}

fn cosine_similarity(left: &QuantizedVector, right: &QuantizedVector) -> f32 {
    1.0 - left.distance(right, DistanceMetric::Cosine)
}
