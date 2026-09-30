//! EngramDB Phase 4: fused hybrid storage with inference-cache streaming.
//!
//! The public surface is intentionally small. [`Engine`] owns durable pages and
//! branch metadata; callers mutate a branch through a [`Transaction`].
//! [`HybridIndex`] owns independent 64 KiB vector, graph, and temporal blocks;
//! [`SessionManager`] provides branch-scoped EnQL execution and Arrow batches.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(not(target_os = "linux"))]
compile_error!("EngramDB requires Linux io_uring and O_DIRECT");

mod arrow_support;
mod branch;
mod buffer_pool;
mod checkpoint;
mod error;
mod flight;
mod fused;
mod hardware;
mod hazard;
mod hybrid;
mod inference;
mod io;
mod kv_cache;
mod query;
mod session;
mod tree;

pub use arrow_support::{projections_to_batch, query_rows_to_batch, query_rows_to_batches};
pub use branch::{Branch, Engine, FaultPoint, MergeOutcome, TemporalRecord, Transaction};
pub use buffer_pool::{BufferPool, BufferPoolStats, Page};
pub use error::{Error, Result};
pub use flight::{serve_flight, EngramFlightService};
pub use fused::{
    dot_i8, simd_kernel_name, FusedBlockBuilder, FusedBlockView, FusedNode, FusedNodeView,
    GraphEdge, GraphEdgeIter, QuantizedVector, TemporalIter, TemporalPoint,
};
#[cfg(feature = "gds")]
pub use hardware::GdsApi;
pub use hardware::{
    CudaStream, CufileAsyncParams, CufileError, CufileHandle, HardwareCapabilities, TransferPath,
};
pub use hazard::{HazardAtomic, HazardDomain, HazardGuard};
pub use hybrid::{
    BlockRef, HybridIndex, HybridIndexStats, NodeProjection, SearchResult, TraversalMatch,
    TriModalQuery, VectorMetric,
};
pub use inference::{InferenceManager, KvRestoreTicket, KvTransferExtent};
pub use io::{
    AlignedPage, BlockLayout, DirectIo, IoStats, FUSED_BLOCK_LAYOUT, FUSED_BLOCK_SIZE, PAGE_LAYOUT,
    PAGE_SIZE,
};
pub use kv_cache::{
    KvBlockRef, KvCacheManifest, KvCacheSnapshot, KvCacheSpec, KvCacheStore, KvDType, KvLayout,
    KV_HEADER_SIZE, KV_TRANSFER_BLOCK_SIZE,
};
pub use query::{
    execute, explain, optimize, parse_enql, plan_logical, AccessPath, CatalogStats, EnqlQuery,
    LogicalPlan, LogicalSource, PhysicalPlan, QueryRow, TemporalFilter,
};
pub use session::{Session, SessionManager, SessionStatus};
pub use tree::{Hash, NodeRef};
