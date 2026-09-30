//! EngramDB Phase 1: a Linux, content-addressed, copy-on-write storage engine.
//!
//! The public surface is intentionally small. [`Engine`] owns durable pages and
//! branch metadata; callers mutate a branch through a [`Transaction`]. Phase 2
//! indexing and query execution are deliberately out of scope.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(not(target_os = "linux"))]
compile_error!("EngramDB Phase 1 requires Linux io_uring and O_DIRECT");

mod branch;
mod buffer_pool;
mod error;
mod hazard;
mod io;
mod tree;

pub use branch::{Branch, Engine, FaultPoint, MergeOutcome, TemporalRecord, Transaction};
pub use buffer_pool::{BufferPool, BufferPoolStats, Page};
pub use error::{Error, Result};
pub use hazard::{HazardAtomic, HazardDomain, HazardGuard};
pub use io::{AlignedBlock, AlignedPage, DirectIo, IoStats, FUSED_BLOCK_SIZE, PAGE_SIZE};
pub use tree::{Hash, NodeRef};
