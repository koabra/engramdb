//! EngramDB storage and fused hybrid-index foundation.
//!
//! The public surface is intentionally small. [`Engine`] owns durable pages and
//! branch metadata; callers mutate a branch through a [`Transaction`]. Phase 2
//! adds 64 KiB fused semantic/graph/temporal blocks through [`HybridIndex`].

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(not(target_os = "linux"))]
compile_error!("EngramDB Phase 1 requires Linux io_uring and O_DIRECT");

mod branch;
mod buffer_pool;
mod error;
mod hazard;
mod hybrid;
mod io;
mod tree;

pub use branch::{Branch, Engine, FaultPoint, MergeOutcome, TemporalRecord, Transaction};
pub use buffer_pool::{BufferPool, BufferPoolStats, Page};
pub use error::{Error, Result};
pub use hazard::{HazardAtomic, HazardDomain, HazardGuard};
pub use hybrid::{
    dot_i8, selected_simd_flavor, BlockLocation, DistanceMetric, FusedBlockBuilder,
    FusedBlockHeader, FusedBlockView, FusedRecordView, GraphEdge, GraphEdgeIter, HnswConfig,
    HybridFaultPoint, HybridIndex, HybridRecord, PackedGraphEdge, QueryHit, QueryResult,
    QueryStats, RecordMetadata, SearchCandidate, SimdFlavor, TemporalEntry, FUSED_HEADER_SIZE,
};
pub use io::{AlignedBlock, AlignedPage, DirectIo, IoStats, FUSED_BLOCK_SIZE, PAGE_SIZE};
pub use tree::{Hash, NodeRef};
