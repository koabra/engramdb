//! EngramDB Phase 2: content-addressed storage with fused hybrid indexing.
//!
//! The public surface is intentionally small. [`Engine`] owns durable pages and
//! branch metadata; callers mutate a branch through a [`Transaction`].
//! [`HybridIndex`] owns independent 64 KiB vector, graph, and temporal blocks.
//! General query execution remains out of scope until Phase 3.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(not(target_os = "linux"))]
compile_error!("EngramDB requires Linux io_uring and O_DIRECT");

mod branch;
mod buffer_pool;
mod error;
mod fused;
mod hazard;
mod hybrid;
mod io;
mod tree;

pub use branch::{Branch, Engine, FaultPoint, MergeOutcome, TemporalRecord, Transaction};
pub use buffer_pool::{BufferPool, BufferPoolStats, Page};
pub use error::{Error, Result};
pub use fused::{
    dot_i8, simd_kernel_name, FusedBlockBuilder, FusedBlockView, FusedNode, FusedNodeView,
    GraphEdge, GraphEdgeIter, QuantizedVector, TemporalIter, TemporalPoint,
};
pub use hazard::{HazardAtomic, HazardDomain, HazardGuard};
pub use hybrid::{
    BlockRef, HybridIndex, HybridIndexStats, SearchResult, TraversalMatch, TriModalQuery,
    VectorMetric,
};
pub use io::{
    AlignedPage, BlockLayout, DirectIo, IoStats, FUSED_BLOCK_LAYOUT, FUSED_BLOCK_SIZE, PAGE_LAYOUT,
    PAGE_SIZE,
};
pub use tree::{Hash, NodeRef};
