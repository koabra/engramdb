use std::io;

use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("io_uring operation failed with errno {0}")]
    Uring(i32),
    #[error("page at offset {offset} is corrupt: {reason}")]
    CorruptPage { offset: u64, reason: String },
    #[error("metadata log is corrupt at byte {offset}: {reason}")]
    CorruptMetadata { offset: u64, reason: String },
    #[error("node payload is too large ({actual} > {maximum} bytes)")]
    NodeTooLarge { actual: usize, maximum: usize },
    #[error("branch {0} does not exist")]
    UnknownBranch(Uuid),
    #[error("branch {0} changed while the transaction was open")]
    StaleTransaction(Uuid),
    #[error("temporal interval must satisfy valid_from < valid_to")]
    InvalidInterval,
    #[error("merge has {0} conflicting temporal range(s)")]
    MergeConflict(usize),
    #[error("injected crash at {0}")]
    InjectedFault(&'static str),
    #[error("storage invariant violated: {0}")]
    Invariant(String),
}

pub type Result<T> = std::result::Result<T, Error>;
