use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridFaultPoint {
    None,
    AfterBlockWrites,
    AfterBlockSync,
    DuringManifestWrite,
}

struct RecordSnapshot {
    metadata: RecordMetadata,
    vector: QuantizedVector,
    edges: Vec<GraphEdge>,
    location: BlockLocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VersionKey {
    id: u64,
    assertion_time: u64,
    valid_from: i64,
    valid_to: i64,
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
    _directory_lock: File,
    io: Arc<DirectIo<FUSED_BLOCK_SIZE>>,
    dimension: usize,
    metric: DistanceMetric,
    hnsw: HnswIndex,
    locations: HashMap<u64, Vec<BlockLocation>>,
    versions: HashSet<VersionKey>,
    committed_length: u64,
    generation: u64,
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
        let directory_lock = acquire_lock(&directory)?;
        let io = Arc::new(DirectIo::<FUSED_BLOCK_SIZE>::open(
            directory.join("fused.blocks"),
            128,
        )?);
        let manifest_path = directory.join("fused.manifest");
        let manifest = if manifest_path.exists() {
            read_manifest(&manifest_path, dimension, metric)?
        } else {
            let manifest = Manifest {
                dimension,
                metric,
                committed_length: 0,
                generation: 0,
            };
            write_manifest(&directory, &manifest, false)?;
            manifest
        };
        if io.len() < manifest.committed_length {
            return Err(Error::Invariant(
                "fused-block file is shorter than committed manifest".to_owned(),
            ));
        }
        if io.len() > manifest.committed_length {
            io.truncate(manifest.committed_length)?;
        }
        let mut index = Self {
            directory,
            _directory_lock: directory_lock,
            io,
            dimension,
            metric,
            hnsw: HnswIndex::new(metric, config)?,
            locations: HashMap::new(),
            versions: HashSet::new(),
            committed_length: manifest.committed_length,
            generation: manifest.generation,
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
        self.committed_length
    }

    pub fn io_stats(&self) -> IoStats {
        self.io.stats()
    }

    pub fn insert(&mut self, epoch: u64, records: Vec<HybridRecord>) -> Result<usize> {
        self.insert_with_fault(epoch, records, HybridFaultPoint::None)
    }

    pub fn insert_with_fault(
        &mut self,
        epoch: u64,
        records: Vec<HybridRecord>,
        fault: HybridFaultPoint,
    ) -> Result<usize> {
        if records.is_empty() {
            return Ok(0);
        }
        let mut staged_versions = HashSet::with_capacity(records.len());
        for record in &records {
            record.validate(self.dimension)?;
            let version = VersionKey {
                id: record.id,
                assertion_time: record.assertion_time,
                valid_from: record.valid_from,
                valid_to: record.valid_to,
            };
            if self.versions.contains(&version) || !staged_versions.insert(version) {
                return Err(Error::Invariant(format!(
                    "duplicate fused record version for ID {}",
                    record.id
                )));
            }
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
            let offset = match self.io.append(&block) {
                Ok(offset) => offset,
                Err(error) => {
                    self.rollback_tail()?;
                    return Err(error);
                }
            };
            appended.push((offset, block));
        }
        if fault == HybridFaultPoint::AfterBlockWrites {
            self.rollback_tail()?;
            return Err(Error::InjectedFault("after fused-block writes"));
        }
        if let Err(error) = self.io.sync() {
            self.rollback_tail()?;
            return Err(error);
        }
        if fault == HybridFaultPoint::AfterBlockSync {
            self.rollback_tail()?;
            return Err(Error::InjectedFault("after fused-block sync"));
        }
        let new_length = self.io.len();
        let manifest = Manifest {
            dimension: self.dimension,
            metric: self.metric,
            committed_length: new_length,
            generation: self.generation + 1,
        };
        if let Err(error) = write_manifest(
            &self.directory,
            &manifest,
            fault == HybridFaultPoint::DuringManifestWrite,
        ) {
            self.rollback_tail()?;
            return Err(error);
        }
        if fault == HybridFaultPoint::DuringManifestWrite {
            self.rollback_tail()?;
            return Err(Error::InjectedFault("during fused manifest write"));
        }
        self.committed_length = new_length;
        self.generation = manifest.generation;
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
        let has_versions = self.locations.values().any(|locations| locations.len() > 1);
        let mut requested = if has_versions {
            self.hnsw.len()
        } else {
            self.hnsw
                .len()
                .min(self.hnsw.ef_search().max(count.saturating_mul(8)))
        };
        let mut context = QueryContext::new(&self.io);
        let (hits, visited_records) = loop {
            let candidates = self.hnsw.search(&query, requested, Some(requested));
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
            if hits.len() >= count || requested >= self.hnsw.len() {
                hits.truncate(count);
                break (hits, seen.len());
            }
            requested = self.hnsw.len().min(requested.saturating_mul(2));
        };
        Ok(QueryResult {
            hits,
            stats: QueryStats {
                visited_records,
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
            FusedBlockView::parse(block.as_slice())?;
            self.register_block(offset, block.as_slice())?;
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
            let version = VersionKey {
                id: record.id(),
                assertion_time: record.assertion_time(),
                valid_from: record.valid_from(),
                valid_to: record.valid_to(),
            };
            if !self.versions.insert(version) {
                return Err(Error::Invariant(format!(
                    "duplicate fused record version for ID {}",
                    record.id()
                )));
            }
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

    fn rollback_tail(&self) -> Result<()> {
        self.io.truncate(self.committed_length)?;
        self.io.sync()
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

const MANIFEST_MAGIC: &[u8; 8] = b"ENGHYB02";
const MANIFEST_SIZE: usize = 64;

#[derive(Clone, Copy)]
struct Manifest {
    dimension: usize,
    metric: DistanceMetric,
    committed_length: u64,
    generation: u64,
}

fn metric_tag(metric: DistanceMetric) -> u8 {
    match metric {
        DistanceMetric::Cosine => 1,
        DistanceMetric::L2 => 2,
    }
}

fn read_manifest(path: &Path, dimension: usize, metric: DistanceMetric) -> Result<Manifest> {
    let mut bytes = [0_u8; MANIFEST_SIZE];
    let mut file = File::open(path)?;
    file.read_exact(&mut bytes)?;
    if file.metadata()?.len() != MANIFEST_SIZE as u64 || &bytes[..8] != MANIFEST_MAGIC {
        return Err(Error::Invariant(
            "invalid fused manifest size or magic".to_owned(),
        ));
    }
    let expected_crc = u32::from_le_bytes(bytes[32..36].try_into().unwrap());
    bytes[32..36].fill(0);
    if crc32fast::hash(&bytes) != expected_crc {
        return Err(Error::Invariant("fused manifest CRC32 mismatch".to_owned()));
    }
    let stored_dimension = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let stored_metric = match bytes[10] {
        1 => DistanceMetric::Cosine,
        2 => DistanceMetric::L2,
        _ => return Err(Error::Invariant("invalid fused manifest metric".to_owned())),
    };
    if stored_dimension != dimension || stored_metric != metric {
        return Err(Error::Invariant(
            "fused manifest configuration does not match requested index".to_owned(),
        ));
    }
    let committed_length = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    if !committed_length.is_multiple_of(FUSED_BLOCK_SIZE as u64) {
        return Err(Error::Invariant(
            "fused manifest watermark is not block aligned".to_owned(),
        ));
    }
    Ok(Manifest {
        dimension,
        metric,
        committed_length,
        generation: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
    })
}

fn write_manifest(directory: &Path, manifest: &Manifest, partial: bool) -> Result<()> {
    let mut bytes = [0_u8; MANIFEST_SIZE];
    bytes[..8].copy_from_slice(MANIFEST_MAGIC);
    bytes[8..10].copy_from_slice(&1_u16.to_le_bytes());
    bytes[10] = metric_tag(manifest.metric);
    bytes[12..16].copy_from_slice(&(manifest.dimension as u32).to_le_bytes());
    bytes[16..24].copy_from_slice(&manifest.committed_length.to_le_bytes());
    bytes[24..32].copy_from_slice(&manifest.generation.to_le_bytes());
    let crc = crc32fast::hash(&bytes);
    bytes[32..36].copy_from_slice(&crc.to_le_bytes());

    let temporary = directory.join("fused.manifest.tmp");
    let destination = directory.join("fused.manifest");
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary)?;
    if partial {
        file.write_all(&bytes[..MANIFEST_SIZE / 2])?;
        file.sync_all()?;
        return Ok(());
    }
    file.write_all(&bytes)?;
    file.sync_all()?;
    std::fs::rename(&temporary, &destination)?;
    File::open(directory)?.sync_all()?;
    Ok(())
}

fn acquire_lock(directory: &Path) -> Result<File> {
    let path = directory.join("hybrid.lock");
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)?;
    // SAFETY: the descriptor remains owned by HybridIndex for the lock lifetime.
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result != 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::WouldBlock {
            return Err(Error::DatabaseLocked(path.display().to_string()));
        }
        return Err(Error::Io(error));
    }
    Ok(file)
}

fn index_key(offset: u64, slot: usize) -> u64 {
    (offset / FUSED_BLOCK_SIZE as u64) << 16 | slot as u64
}

fn cosine_similarity(left: &QuantizedVector, right: &QuantizedVector) -> f32 {
    1.0 - left.distance(right, DistanceMetric::Cosine)
}
