use std::collections::HashSet;
use std::mem::size_of;

use crate::hybrid::simd::QuantizedVector;
use crate::{AlignedBlock, Error, Result, FUSED_BLOCK_SIZE};

const MAGIC: &[u8; 8] = b"ENGFUS02";
const VERSION: u16 = 1;
pub const FUSED_HEADER_SIZE: usize = 64;
const TEMPORAL_SIZE: usize = 24;
const EDGE_SIZE: usize = 16;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct FusedBlockHeader {
    magic: [u8; 8],
    version: u16,
    header_size: u16,
    epoch: u64,
    crc32: u32,
    node_type: u16,
    record_count: u16,
    dimension: u16,
    flags: u16,
    record_ids_offset: u32,
    temporal_offset: u32,
    vector_offset: u32,
    graph_offsets_offset: u32,
    graph_edges_offset: u32,
    used_len: u32,
    reserved: [u8; 8],
}

const _: [(); FUSED_HEADER_SIZE] = [(); size_of::<FusedBlockHeader>()];

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalEntry {
    pub assertion_time: u64,
    pub valid_from: i64,
    pub valid_to: i64,
}

const _: [(); TEMPORAL_SIZE] = [(); size_of::<TemporalEntry>()];

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackedGraphEdge {
    pub target: u64,
    pub weight: f32,
    pub edge_type: u16,
    reserved: u16,
}

const _: [(); EDGE_SIZE] = [(); size_of::<PackedGraphEdge>()];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphEdge {
    pub target: u64,
    pub weight: f32,
    pub edge_type: u16,
}

#[derive(Debug, Clone)]
pub struct HybridRecord {
    pub id: u64,
    pub assertion_time: u64,
    pub valid_from: i64,
    pub valid_to: i64,
    pub vector: Vec<f32>,
    pub edges: Vec<GraphEdge>,
}

impl HybridRecord {
    pub fn validate(&self, dimension: usize) -> Result<()> {
        if self.valid_from >= self.valid_to {
            return Err(Error::InvalidInterval);
        }
        if self.vector.len() != dimension {
            return Err(Error::Invariant(format!(
                "record {} has dimension {}, expected {dimension}",
                self.id,
                self.vector.len()
            )));
        }
        if self.edges.iter().any(|edge| !edge.weight.is_finite()) {
            return Err(Error::Invariant(
                "graph edge weights must be finite".to_owned(),
            ));
        }
        QuantizedVector::encode(&self.vector)?;
        Ok(())
    }
}

pub struct FusedBlockBuilder {
    epoch: u64,
    dimension: usize,
    records: Vec<HybridRecord>,
}

impl FusedBlockBuilder {
    pub fn new(epoch: u64, dimension: usize) -> Result<Self> {
        if dimension == 0 || dimension > u16::MAX as usize {
            return Err(Error::Invariant(
                "fused-block dimension must be in 1..=65535".to_owned(),
            ));
        }
        Ok(Self {
            epoch,
            dimension,
            records: Vec::new(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn try_push(&mut self, record: HybridRecord) -> Result<bool> {
        record.validate(self.dimension)?;
        if self.records.iter().any(|current| current.id == record.id) {
            return Err(Error::Invariant(
                "record IDs must be unique within a fused block".to_owned(),
            ));
        }
        self.records.push(record);
        if layout_for(&self.records, self.dimension).is_err() {
            let record = self.records.pop().unwrap();
            if self.records.is_empty() {
                return Err(Error::NodeTooLarge {
                    actual: estimated_size(std::slice::from_ref(&record), self.dimension),
                    maximum: FUSED_BLOCK_SIZE,
                });
            }
            return Ok(false);
        }
        Ok(true)
    }

    pub fn finish(self) -> Result<AlignedBlock<FUSED_BLOCK_SIZE>> {
        if self.records.is_empty() {
            return Err(Error::Invariant(
                "cannot encode an empty fused block".to_owned(),
            ));
        }
        encode_block(self.epoch, self.dimension, &self.records)
    }
}

#[derive(Clone, Copy)]
struct Layout {
    ids: usize,
    temporal: usize,
    vectors: usize,
    graph_offsets: usize,
    edges: usize,
    used: usize,
}

fn estimated_size(records: &[HybridRecord], dimension: usize) -> usize {
    FUSED_HEADER_SIZE
        + records.len() * 8
        + records.len() * TEMPORAL_SIZE
        + records.len() * 4
        + records.len() * dimension
        + (records.len() + 1) * 4
        + records
            .iter()
            .map(|record| record.edges.len() * EDGE_SIZE)
            .sum::<usize>()
        + 5 * 63
}

fn align(cursor: usize, alignment: usize) -> usize {
    (cursor + alignment - 1) & !(alignment - 1)
}

fn layout_for(records: &[HybridRecord], dimension: usize) -> Result<Layout> {
    let count = records.len();
    if count > u16::MAX as usize {
        return Err(Error::NodeTooLarge {
            actual: count,
            maximum: u16::MAX as usize,
        });
    }
    let mut cursor = FUSED_HEADER_SIZE;
    let ids = align(cursor, 8);
    cursor = ids + count * 8;
    let temporal = align(cursor, 8);
    cursor = temporal + count * TEMPORAL_SIZE;
    let vectors = align(cursor, 64);
    cursor = vectors + count * 4 + count * dimension;
    let graph_offsets = align(cursor, 4);
    cursor = graph_offsets + (count + 1) * 4;
    let edges = align(cursor, 8);
    cursor = edges
        + records
            .iter()
            .map(|record| record.edges.len() * EDGE_SIZE)
            .sum::<usize>();
    if cursor > FUSED_BLOCK_SIZE {
        return Err(Error::NodeTooLarge {
            actual: cursor,
            maximum: FUSED_BLOCK_SIZE,
        });
    }
    Ok(Layout {
        ids,
        temporal,
        vectors,
        graph_offsets,
        edges,
        used: cursor,
    })
}

fn encode_block(
    epoch: u64,
    dimension: usize,
    records: &[HybridRecord],
) -> Result<AlignedBlock<FUSED_BLOCK_SIZE>> {
    let layout = layout_for(records, dimension)?;
    let quantized = records
        .iter()
        .map(|record| QuantizedVector::encode(&record.vector))
        .collect::<Result<Vec<_>>>()?;
    let mut block = AlignedBlock::<FUSED_BLOCK_SIZE>::zeroed();
    let bytes = block.as_mut_slice();

    for (slot, record) in records.iter().enumerate() {
        put_u64(bytes, layout.ids + slot * 8, record.id);
        let temporal = layout.temporal + slot * TEMPORAL_SIZE;
        put_u64(bytes, temporal, record.assertion_time);
        put_i64(bytes, temporal + 8, record.valid_from);
        put_i64(bytes, temporal + 16, record.valid_to);
        put_f32(bytes, layout.vectors + slot * 4, quantized[slot].scale);
        let vector_start = layout.vectors + records.len() * 4 + slot * dimension;
        for (target, value) in bytes[vector_start..vector_start + dimension]
            .iter_mut()
            .zip(&quantized[slot].values)
        {
            *target = *value as u8;
        }
    }

    let mut edge_index = 0;
    for (slot, record) in records.iter().enumerate() {
        put_u32(bytes, layout.graph_offsets + slot * 4, edge_index as u32);
        for edge in &record.edges {
            let cursor = layout.edges + edge_index * EDGE_SIZE;
            put_u64(bytes, cursor, edge.target);
            put_f32(bytes, cursor + 8, edge.weight);
            put_u16(bytes, cursor + 12, edge.edge_type);
            put_u16(bytes, cursor + 14, 0);
            edge_index += 1;
        }
    }
    put_u32(
        bytes,
        layout.graph_offsets + records.len() * 4,
        edge_index as u32,
    );

    bytes[..8].copy_from_slice(MAGIC);
    put_u16(bytes, 8, VERSION);
    put_u16(bytes, 10, FUSED_HEADER_SIZE as u16);
    put_u64(bytes, 12, epoch);
    put_u32(bytes, 20, 0);
    put_u16(bytes, 24, 1);
    put_u16(bytes, 26, records.len() as u16);
    put_u16(bytes, 28, dimension as u16);
    put_u16(bytes, 30, 1);
    put_u32(bytes, 32, layout.ids as u32);
    put_u32(bytes, 36, layout.temporal as u32);
    put_u32(bytes, 40, layout.vectors as u32);
    put_u32(bytes, 44, layout.graph_offsets as u32);
    put_u32(bytes, 48, layout.edges as u32);
    put_u32(bytes, 52, layout.used as u32);
    let crc = crc32fast::hash(&bytes[FUSED_HEADER_SIZE..layout.used]);
    put_u32(bytes, 20, crc);
    Ok(block)
}

#[derive(Clone, Copy)]
struct DecodedHeader {
    epoch: u64,
    count: usize,
    dimension: usize,
    ids: usize,
    temporal: usize,
    vectors: usize,
    graph_offsets: usize,
    edges: usize,
    used: usize,
}

pub struct FusedBlockView<'a> {
    bytes: &'a [u8],
    header: DecodedHeader,
}

impl<'a> FusedBlockView<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() != FUSED_BLOCK_SIZE
            || &bytes[..8] != MAGIC
            || get_u16(bytes, 8) != VERSION
            || get_u16(bytes, 10) as usize != FUSED_HEADER_SIZE
        {
            return Err(Error::Invariant(
                "invalid fused-block magic, version, or size".to_owned(),
            ));
        }
        let header = DecodedHeader {
            epoch: get_u64(bytes, 12),
            count: get_u16(bytes, 26) as usize,
            dimension: get_u16(bytes, 28) as usize,
            ids: get_u32(bytes, 32) as usize,
            temporal: get_u32(bytes, 36) as usize,
            vectors: get_u32(bytes, 40) as usize,
            graph_offsets: get_u32(bytes, 44) as usize,
            edges: get_u32(bytes, 48) as usize,
            used: get_u32(bytes, 52) as usize,
        };
        if header.count == 0
            || header.dimension == 0
            || header.used > FUSED_BLOCK_SIZE
            || header.ids < FUSED_HEADER_SIZE
            || header.ids + header.count * 8 > header.temporal
            || header.temporal + header.count * TEMPORAL_SIZE > header.vectors
            || header.vectors + header.count * 4 + header.count * header.dimension
                > header.graph_offsets
            || header.graph_offsets + (header.count + 1) * 4 > header.edges
            || header.edges > header.used
            || (header.used - header.edges) % EDGE_SIZE != 0
        {
            return Err(Error::Invariant(
                "fused-block section offsets are invalid".to_owned(),
            ));
        }
        if crc32fast::hash(&bytes[FUSED_HEADER_SIZE..header.used]) != get_u32(bytes, 20) {
            return Err(Error::Invariant("fused-block CRC32 mismatch".to_owned()));
        }
        let edge_count = (header.used - header.edges) / EDGE_SIZE;
        let mut prior = 0;
        for slot in 0..=header.count {
            let offset = get_u32(bytes, header.graph_offsets + slot * 4) as usize;
            if offset < prior || offset > edge_count {
                return Err(Error::Invariant(
                    "fused-block CSR offsets are invalid".to_owned(),
                ));
            }
            prior = offset;
        }
        if prior != edge_count {
            return Err(Error::Invariant(
                "fused-block CSR does not cover all edges".to_owned(),
            ));
        }
        let mut ids = HashSet::with_capacity(header.count);
        for slot in 0..header.count {
            if !ids.insert(get_u64(bytes, header.ids + slot * 8)) {
                return Err(Error::Invariant(
                    "fused-block contains duplicate record IDs".to_owned(),
                ));
            }
        }
        Ok(Self { bytes, header })
    }

    pub fn epoch(&self) -> u64 {
        self.header.epoch
    }

    pub fn dimension(&self) -> usize {
        self.header.dimension
    }

    pub fn len(&self) -> usize {
        self.header.count
    }

    pub fn is_empty(&self) -> bool {
        self.header.count == 0
    }

    pub fn record(&self, slot: usize) -> Result<FusedRecordView<'a>> {
        if slot >= self.header.count {
            return Err(Error::Invariant(
                "fused-block record slot is out of bounds".to_owned(),
            ));
        }
        Ok(FusedRecordView {
            bytes: self.bytes,
            header: self.header,
            slot,
        })
    }
}

pub struct FusedRecordView<'a> {
    bytes: &'a [u8],
    header: DecodedHeader,
    slot: usize,
}

impl FusedRecordView<'_> {
    pub fn id(&self) -> u64 {
        get_u64(self.bytes, self.header.ids + self.slot * 8)
    }

    pub fn assertion_time(&self) -> u64 {
        get_u64(self.bytes, self.header.temporal + self.slot * TEMPORAL_SIZE)
    }

    pub fn valid_from(&self) -> i64 {
        get_i64(
            self.bytes,
            self.header.temporal + self.slot * TEMPORAL_SIZE + 8,
        )
    }

    pub fn valid_to(&self) -> i64 {
        get_i64(
            self.bytes,
            self.header.temporal + self.slot * TEMPORAL_SIZE + 16,
        )
    }

    pub fn scale(&self) -> f32 {
        get_f32(self.bytes, self.header.vectors + self.slot * 4)
    }

    pub fn quantized_vector(&self) -> &'_ [i8] {
        let start = self.header.vectors + self.header.count * 4 + self.slot * self.header.dimension;
        let bytes = &self.bytes[start..start + self.header.dimension];
        // SAFETY: i8 and u8 have identical size/alignment and the slice remains
        // bound to the validated block.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) }
    }

    pub fn edges(&self) -> GraphEdgeIter<'_> {
        let start = get_u32(self.bytes, self.header.graph_offsets + self.slot * 4) as usize;
        let end = get_u32(self.bytes, self.header.graph_offsets + (self.slot + 1) * 4) as usize;
        GraphEdgeIter {
            bytes: self.bytes,
            cursor: self.header.edges + start * EDGE_SIZE,
            end: self.header.edges + end * EDGE_SIZE,
        }
    }
}

pub struct GraphEdgeIter<'a> {
    bytes: &'a [u8],
    cursor: usize,
    end: usize,
}

impl Iterator for GraphEdgeIter<'_> {
    type Item = GraphEdge;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.end {
            return None;
        }
        let edge = GraphEdge {
            target: get_u64(self.bytes, self.cursor),
            weight: get_f32(self.bytes, self.cursor + 8),
            edge_type: get_u16(self.bytes, self.cursor + 12),
        };
        self.cursor += EDGE_SIZE;
        Some(edge)
    }
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_i64(bytes: &mut [u8], offset: usize, value: i64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_f32(bytes: &mut [u8], offset: usize, value: f32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn get_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn get_i64(bytes: &[u8], offset: usize) -> i64 {
    i64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn get_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
