//! Packed 64 KiB tri-modal blocks and quantized vector kernels.

use std::mem::size_of;

use crate::io::{AlignedPage, FUSED_BLOCK_LAYOUT, FUSED_BLOCK_SIZE};
use crate::tree::Hash;
use crate::{Error, Result};

const MAGIC: &[u8; 8] = b"ENGFUSE1";
const VERSION: u16 = 1;
const NODE_TYPE_TRI_MODAL: u16 = 1;
const HEADER_SIZE: usize = 64;
const DIRECTORY_ENTRY_SIZE: usize = 64;
const TEMPORAL_ENTRY_SIZE: usize = 16;
const EDGE_ENTRY_SIZE: usize = 40;
const CRC_OFFSET: usize = 56;

#[repr(C, packed)]
struct HeaderLayout {
    magic: [u8; 8],
    version: u16,
    node_type: u16,
    epoch: u64,
    node_count: u32,
    directory_offset: u32,
    directory_length: u32,
    temporal_offset: u32,
    temporal_length: u32,
    vector_offset: u32,
    vector_length: u32,
    graph_offset: u32,
    graph_length: u32,
    crc32: u32,
    reserved: u32,
}

#[repr(C, packed)]
struct DirectoryEntryLayout {
    id: [u8; 32],
    temporal_offset: u32,
    temporal_count: u16,
    vector_offset: u32,
    vector_dim: u16,
    graph_offset: u32,
    edge_count: u16,
    quant_scale: f32,
    quantized_norm: f32,
    flags: u32,
    reserved: u16,
}

const _: [(); HEADER_SIZE] = [(); size_of::<HeaderLayout>()];
const _: [(); DIRECTORY_ENTRY_SIZE] = [(); size_of::<DirectoryEntryLayout>()];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalPoint {
    pub assertion_time: u64,
    pub valid_time: i64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphEdge {
    pub target: Hash,
    pub weight: f32,
    pub edge_type: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusedNode {
    pub id: Hash,
    pub temporal: Vec<TemporalPoint>,
    pub vector: Vec<f32>,
    pub edges: Vec<GraphEdge>,
}

impl FusedNode {
    pub fn new(
        key: impl AsRef<[u8]>,
        temporal: Vec<TemporalPoint>,
        vector: Vec<f32>,
        edges: Vec<GraphEdge>,
    ) -> Result<Self> {
        validate_vector(&vector)?;
        if temporal.len() > u16::MAX as usize || edges.len() > u16::MAX as usize {
            return Err(Error::Invariant(
                "fused node cardinality exceeds the on-block u16 limit".to_owned(),
            ));
        }
        Ok(Self {
            id: Hash(*blake3::hash(key.as_ref()).as_bytes()),
            temporal,
            vector,
            edges,
        })
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedVector {
    codes: Vec<i8>,
    scale: f32,
    norm: f32,
}

impl QuantizedVector {
    pub fn from_f32(vector: &[f32]) -> Result<Self> {
        validate_vector(vector)?;
        let maximum = vector
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
        let codes: Vec<i8> = vector
            .iter()
            .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let norm = (dot_i8(&codes, &codes) as f32).sqrt();
        Ok(Self { codes, scale, norm })
    }

    pub fn codes(&self) -> &[i8] {
        &self.codes
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn quantized_norm(&self) -> f32 {
        self.norm
    }
}

fn validate_vector(vector: &[f32]) -> Result<()> {
    if vector.is_empty()
        || vector.len() > u16::MAX as usize
        || vector.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidVector);
    }
    Ok(())
}

pub struct FusedBlockBuilder {
    epoch: u64,
    nodes: Vec<FusedNode>,
}

impl FusedBlockBuilder {
    pub fn new(epoch: u64) -> Self {
        Self {
            epoch,
            nodes: Vec::new(),
        }
    }

    pub fn push(&mut self, node: FusedNode) -> Result<()> {
        if !self.can_fit(&node)? {
            let mut candidate = self.nodes.clone();
            candidate.push(node);
            return Err(Error::FusedBlockFull {
                required: required_size_unchecked(&candidate)?,
                available: FUSED_BLOCK_SIZE,
            });
        }
        self.nodes.push(node);
        Ok(())
    }

    pub fn can_fit(&self, node: &FusedNode) -> Result<bool> {
        let mut candidate = self.nodes.clone();
        candidate.push(node.clone());
        Ok(required_size_unchecked(&candidate)? <= FUSED_BLOCK_SIZE)
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn finish(self) -> Result<AlignedPage> {
        encode_block(self.epoch, &self.nodes)
    }
}

fn required_size(nodes: &[FusedNode]) -> Result<usize> {
    let required = required_size_unchecked(nodes)?;
    if required > FUSED_BLOCK_SIZE {
        return Err(Error::FusedBlockFull {
            required,
            available: FUSED_BLOCK_SIZE,
        });
    }
    Ok(required)
}

fn required_size_unchecked(nodes: &[FusedNode]) -> Result<usize> {
    let mut required = HEADER_SIZE
        .checked_add(
            nodes
                .len()
                .checked_mul(DIRECTORY_ENTRY_SIZE)
                .ok_or_else(|| Error::Invariant("directory size overflow".to_owned()))?,
        )
        .ok_or_else(|| Error::Invariant("block size overflow".to_owned()))?;
    for node in nodes {
        validate_vector(&node.vector)?;
        required = required
            .checked_add(node.temporal.len() * TEMPORAL_ENTRY_SIZE)
            .and_then(|size| size.checked_add(node.vector.len()))
            .and_then(|size| size.checked_add(node.edges.len() * EDGE_ENTRY_SIZE))
            .ok_or_else(|| Error::Invariant("block size overflow".to_owned()))?;
    }
    Ok(required)
}

fn encode_block(epoch: u64, nodes: &[FusedNode]) -> Result<AlignedPage> {
    if nodes.is_empty() {
        return Err(Error::Invariant(
            "a fused block must contain at least one node".to_owned(),
        ));
    }
    required_size(nodes)?;
    let quantized: Vec<QuantizedVector> = nodes
        .iter()
        .map(|node| QuantizedVector::from_f32(&node.vector))
        .collect::<Result<_>>()?;

    let directory_offset = HEADER_SIZE;
    let directory_length = nodes.len() * DIRECTORY_ENTRY_SIZE;
    let temporal_offset = directory_offset + directory_length;
    let temporal_length = nodes
        .iter()
        .map(|node| node.temporal.len() * TEMPORAL_ENTRY_SIZE)
        .sum::<usize>();
    let vector_offset = temporal_offset + temporal_length;
    let vector_length = nodes.iter().map(|node| node.vector.len()).sum::<usize>();
    let graph_offset = vector_offset + vector_length;
    let graph_length = nodes
        .iter()
        .map(|node| node.edges.len() * EDGE_ENTRY_SIZE)
        .sum::<usize>();

    let mut block = AlignedPage::zeroed_for(FUSED_BLOCK_LAYOUT);
    let bytes = block.as_mut_slice();
    bytes[..8].copy_from_slice(MAGIC);
    put_u16(bytes, 8, VERSION);
    put_u16(bytes, 10, NODE_TYPE_TRI_MODAL);
    put_u64(bytes, 12, epoch);
    put_u32(bytes, 20, nodes.len() as u32);
    put_u32(bytes, 24, directory_offset as u32);
    put_u32(bytes, 28, directory_length as u32);
    put_u32(bytes, 32, temporal_offset as u32);
    put_u32(bytes, 36, temporal_length as u32);
    put_u32(bytes, 40, vector_offset as u32);
    put_u32(bytes, 44, vector_length as u32);
    put_u32(bytes, 48, graph_offset as u32);
    put_u32(bytes, 52, graph_length as u32);

    let mut temporal_cursor = temporal_offset;
    let mut vector_cursor = vector_offset;
    let mut graph_cursor = graph_offset;
    for (slot, (node, vector)) in nodes.iter().zip(&quantized).enumerate() {
        let entry = directory_offset + slot * DIRECTORY_ENTRY_SIZE;
        bytes[entry..entry + 32].copy_from_slice(&node.id.0);
        put_u32(bytes, entry + 32, temporal_cursor as u32);
        put_u16(bytes, entry + 36, node.temporal.len() as u16);
        put_u32(bytes, entry + 38, vector_cursor as u32);
        put_u16(bytes, entry + 42, node.vector.len() as u16);
        put_u32(bytes, entry + 44, graph_cursor as u32);
        put_u16(bytes, entry + 48, node.edges.len() as u16);
        put_f32(bytes, entry + 50, vector.scale());
        put_f32(bytes, entry + 54, vector.quantized_norm());
        put_u32(bytes, entry + 58, 0);
        put_u16(bytes, entry + 62, 0);

        for temporal in &node.temporal {
            put_u64(bytes, temporal_cursor, temporal.assertion_time);
            put_i64(bytes, temporal_cursor + 8, temporal.valid_time);
            temporal_cursor += TEMPORAL_ENTRY_SIZE;
        }
        for code in vector.codes() {
            bytes[vector_cursor] = *code as u8;
            vector_cursor += 1;
        }
        for edge in &node.edges {
            bytes[graph_cursor..graph_cursor + 32].copy_from_slice(&edge.target.0);
            put_f32(bytes, graph_cursor + 32, edge.weight);
            put_u16(bytes, graph_cursor + 36, edge.edge_type);
            put_u16(bytes, graph_cursor + 38, 0);
            graph_cursor += EDGE_ENTRY_SIZE;
        }
    }
    put_u32(bytes, CRC_OFFSET, crc32fast::hash(&bytes[HEADER_SIZE..]));
    Ok(block)
}

#[derive(Clone, Copy)]
pub struct FusedBlockView<'a> {
    bytes: &'a [u8],
    epoch: u64,
    node_count: usize,
}

impl<'a> FusedBlockView<'a> {
    pub fn parse(bytes: &'a [u8], offset: u64) -> Result<Self> {
        if bytes.len() != FUSED_BLOCK_SIZE
            || &bytes[..8] != MAGIC
            || get_u16(bytes, 8) != VERSION
            || get_u16(bytes, 10) != NODE_TYPE_TRI_MODAL
        {
            return Err(corrupt(
                offset,
                "invalid fused block magic, version, or size",
            ));
        }
        let node_count = get_u32(bytes, 20) as usize;
        let directory = region(bytes, 24, 28, offset, "directory")?;
        let temporal = region(bytes, 32, 36, offset, "temporal")?;
        let vector = region(bytes, 40, 44, offset, "vector")?;
        let graph = region(bytes, 48, 52, offset, "graph")?;
        if directory.start != HEADER_SIZE
            || directory.end - directory.start != node_count * DIRECTORY_ENTRY_SIZE
            || temporal.start != directory.end
            || vector.start != temporal.end
            || graph.start != vector.end
        {
            return Err(corrupt(
                offset,
                "fused block regions are not tightly packed",
            ));
        }
        if crc32fast::hash(&bytes[HEADER_SIZE..]) != get_u32(bytes, CRC_OFFSET) {
            return Err(corrupt(offset, "fused block CRC32 mismatch"));
        }
        let view = Self {
            bytes,
            epoch: get_u64(bytes, 12),
            node_count,
        };
        for slot in 0..node_count {
            view.node(slot, offset)?;
        }
        Ok(view)
    }

    pub(crate) fn trusted(bytes: &'a [u8]) -> Self {
        debug_assert_eq!(bytes.len(), FUSED_BLOCK_SIZE);
        Self {
            bytes,
            epoch: get_u64(bytes, 12),
            node_count: get_u32(bytes, 20) as usize,
        }
    }

    pub fn epoch(self) -> u64 {
        self.epoch
    }

    pub fn len(self) -> usize {
        self.node_count
    }

    pub fn is_empty(self) -> bool {
        self.node_count == 0
    }

    pub fn node(self, slot: usize, offset: u64) -> Result<FusedNodeView<'a>> {
        if slot >= self.node_count {
            return Err(corrupt(offset, "fused node slot is outside the directory"));
        }
        let entry = HEADER_SIZE + slot * DIRECTORY_ENTRY_SIZE;
        let id = Hash(self.bytes[entry..entry + 32].try_into().unwrap());
        let temporal_offset = get_u32(self.bytes, entry + 32) as usize;
        let temporal_count = get_u16(self.bytes, entry + 36) as usize;
        let vector_offset = get_u32(self.bytes, entry + 38) as usize;
        let vector_dim = get_u16(self.bytes, entry + 42) as usize;
        let graph_offset = get_u32(self.bytes, entry + 44) as usize;
        let edge_count = get_u16(self.bytes, entry + 48) as usize;
        let quant_scale = get_f32(self.bytes, entry + 50);
        let quantized_norm = get_f32(self.bytes, entry + 54);
        let temporal_end = checked_end(temporal_offset, temporal_count, TEMPORAL_ENTRY_SIZE)?;
        let vector_end = checked_end(vector_offset, vector_dim, 1)?;
        let graph_end = checked_end(graph_offset, edge_count, EDGE_ENTRY_SIZE)?;
        if temporal_offset < get_u32(self.bytes, 32) as usize
            || temporal_end > get_u32(self.bytes, 40) as usize
            || vector_offset < get_u32(self.bytes, 40) as usize
            || vector_end > get_u32(self.bytes, 48) as usize
            || graph_offset < get_u32(self.bytes, 48) as usize
            || graph_end > FUSED_BLOCK_SIZE
            || !quant_scale.is_finite()
            || quant_scale <= 0.0
            || !quantized_norm.is_finite()
        {
            return Err(corrupt(offset, "fused node directory entry is invalid"));
        }
        Ok(FusedNodeView {
            id,
            temporal: &self.bytes[temporal_offset..temporal_end],
            vector: as_i8(&self.bytes[vector_offset..vector_end]),
            edges: &self.bytes[graph_offset..graph_end],
            quant_scale,
            quantized_norm,
        })
    }
}

#[derive(Clone, Copy)]
pub struct FusedNodeView<'a> {
    id: Hash,
    temporal: &'a [u8],
    vector: &'a [i8],
    edges: &'a [u8],
    quant_scale: f32,
    quantized_norm: f32,
}

impl<'a> FusedNodeView<'a> {
    pub fn id(self) -> Hash {
        self.id
    }

    pub fn vector(self) -> &'a [i8] {
        self.vector
    }

    pub fn quantization_scale(self) -> f32 {
        self.quant_scale
    }

    pub fn cosine_similarity(self, query: &QuantizedVector) -> Result<f32> {
        if query.codes.len() != self.vector.len() {
            return Err(Error::DimensionMismatch {
                expected: self.vector.len(),
                actual: query.codes.len(),
            });
        }
        if query.norm == 0.0 || self.quantized_norm == 0.0 {
            return Ok(0.0);
        }
        Ok(dot_i8(self.vector, &query.codes) as f32 / (self.quantized_norm * query.norm))
    }

    pub(crate) fn cosine_similarity_to(self, other: FusedNodeView<'_>) -> Result<f32> {
        if self.vector.len() != other.vector.len() {
            return Err(Error::DimensionMismatch {
                expected: self.vector.len(),
                actual: other.vector.len(),
            });
        }
        if self.quantized_norm == 0.0 || other.quantized_norm == 0.0 {
            return Ok(0.0);
        }
        Ok(dot_i8(self.vector, other.vector) as f32 / (self.quantized_norm * other.quantized_norm))
    }

    pub fn temporal(self) -> TemporalIter<'a> {
        TemporalIter {
            bytes: self.temporal,
            position: 0,
        }
    }

    pub fn edges(self) -> GraphEdgeIter<'a> {
        GraphEdgeIter {
            bytes: self.edges,
            position: 0,
        }
    }
}

pub struct TemporalIter<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Iterator for TemporalIter<'_> {
    type Item = TemporalPoint;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.bytes.len() {
            return None;
        }
        let point = TemporalPoint {
            assertion_time: get_u64(self.bytes, self.position),
            valid_time: get_i64(self.bytes, self.position + 8),
        };
        self.position += TEMPORAL_ENTRY_SIZE;
        Some(point)
    }
}

pub struct GraphEdgeIter<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Iterator for GraphEdgeIter<'_> {
    type Item = GraphEdge;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.bytes.len() {
            return None;
        }
        let edge = GraphEdge {
            target: Hash(
                self.bytes[self.position..self.position + 32]
                    .try_into()
                    .unwrap(),
            ),
            weight: get_f32(self.bytes, self.position + 32),
            edge_type: get_u16(self.bytes, self.position + 36),
        };
        self.position += EDGE_ENTRY_SIZE;
        Some(edge)
    }
}

pub fn simd_kernel_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return "avx2-i8";
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return "neon-i8";
    }
    "scalar-i8"
}

pub fn dot_i8(left: &[i8], right: &[i8]) -> i64 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature detection proves AVX2 support; loads are
        // unaligned and bounded by the loop in the implementation.
        return unsafe { dot_i8_avx2(left, right) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: runtime feature detection proves NEON support.
        return unsafe { dot_i8_neon(left, right) };
    }
    dot_i8_scalar(left, right)
}

fn dot_i8_scalar(left: &[i8], right: &[i8]) -> i64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| *left as i64 * *right as i64)
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(left: &[i8], right: &[i8]) -> i64 {
    use std::arch::x86_64::*;

    let mut accumulator = _mm256_setzero_si256();
    let mut position = 0;
    while position + 32 <= left.len() {
        let left_bytes = _mm256_loadu_si256(left.as_ptr().add(position).cast());
        let right_bytes = _mm256_loadu_si256(right.as_ptr().add(position).cast());
        let left_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(left_bytes));
        let left_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(left_bytes));
        let right_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(right_bytes));
        let right_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(right_bytes));
        accumulator = _mm256_add_epi32(
            accumulator,
            _mm256_add_epi32(
                _mm256_madd_epi16(left_low, right_low),
                _mm256_madd_epi16(left_high, right_high),
            ),
        );
        position += 32;
    }
    let mut lanes = [0_i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr().cast(), accumulator);
    lanes.iter().map(|value| *value as i64).sum::<i64>()
        + dot_i8_scalar(&left[position..], &right[position..])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_neon(left: &[i8], right: &[i8]) -> i64 {
    use std::arch::aarch64::*;

    let mut total = 0_i64;
    let mut position = 0;
    while position + 16 <= left.len() {
        let left_bytes = vld1q_s8(left.as_ptr().add(position));
        let right_bytes = vld1q_s8(right.as_ptr().add(position));
        let low = vmull_s8(vget_low_s8(left_bytes), vget_low_s8(right_bytes));
        let high = vmull_s8(vget_high_s8(left_bytes), vget_high_s8(right_bytes));
        total += vaddvq_s32(vpaddlq_s16(low)) as i64;
        total += vaddvq_s32(vpaddlq_s16(high)) as i64;
        position += 16;
    }
    total + dot_i8_scalar(&left[position..], &right[position..])
}

fn region(
    bytes: &[u8],
    offset_field: usize,
    length_field: usize,
    file_offset: u64,
    name: &str,
) -> Result<std::ops::Range<usize>> {
    let start = get_u32(bytes, offset_field) as usize;
    let length = get_u32(bytes, length_field) as usize;
    let end = start
        .checked_add(length)
        .ok_or_else(|| corrupt(file_offset, &format!("{name} region overflows")))?;
    if start < HEADER_SIZE || end > bytes.len() {
        return Err(corrupt(
            file_offset,
            &format!("{name} region is outside the fused block"),
        ));
    }
    Ok(start..end)
}

fn checked_end(start: usize, count: usize, width: usize) -> Result<usize> {
    start
        .checked_add(
            count
                .checked_mul(width)
                .ok_or_else(|| Error::Invariant("fused region size overflow".to_owned()))?,
        )
        .ok_or_else(|| Error::Invariant("fused region end overflow".to_owned()))
}

fn corrupt(offset: u64, reason: &str) -> Error {
    Error::CorruptPage {
        offset,
        reason: reason.to_owned(),
    }
}

fn as_i8(bytes: &[u8]) -> &[i8] {
    // SAFETY: i8 and u8 have identical size/alignment and the same valid bit
    // patterns, so this changes only the interpretation of each byte.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len()) }
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
