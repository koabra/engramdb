//! User-space page cache with lock-free snapshots and CLOCK eviction.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::Mutex;

use crate::hazard::{HazardAtomic, HazardDomain, HazardGuard};
use crate::io::{AlignedPage, DirectIo};
use crate::Result;

pub struct Page {
    offset: u64,
    bytes: AlignedPage,
}

impl Page {
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }
}

struct Entry {
    page: HazardAtomic<Page>,
    referenced: AtomicBool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BufferPoolStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub resident_pages: usize,
}

/// Reads take an immutable hash-map snapshot and protect the page with a hazard
/// pointer. Miss insertion and CLOCK bookkeeping are serialized.
pub struct BufferPool {
    io: Arc<DirectIo>,
    capacity: usize,
    entries: ArcSwap<HashMap<u64, Arc<Entry>>>,
    clock: Mutex<VecDeque<u64>>,
    writer: Mutex<()>,
    domain: HazardDomain<Page>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl BufferPool {
    pub fn new(io: Arc<DirectIo>, capacity: usize) -> Self {
        Self {
            io,
            capacity: capacity.max(1),
            entries: ArcSwap::from_pointee(HashMap::new()),
            clock: Mutex::new(VecDeque::new()),
            writer: Mutex::new(()),
            domain: HazardDomain::new(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    pub fn get(&self, offset: u64) -> Result<HazardGuard<Page>> {
        let snapshot = self.entries.load();
        if let Some(entry) = snapshot.get(&offset) {
            entry.referenced.store(true, Ordering::Relaxed);
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(entry.page.load());
        }
        drop(snapshot);

        let _writer = self.writer.lock();
        let current = self.entries.load_full();
        if let Some(entry) = current.get(&offset) {
            entry.referenced.store(true, Ordering::Relaxed);
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(entry.page.load());
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let bytes = self.io.read(offset)?;
        let page = Arc::new(Page { offset, bytes });
        let entry = Arc::new(Entry {
            page: HazardAtomic::new(Arc::clone(&page), self.domain.clone()),
            referenced: AtomicBool::new(true),
        });

        let mut next = (*current).clone();
        let mut clock = self.clock.lock();
        while next.len() >= self.capacity {
            let Some(candidate) = clock.pop_front() else {
                break;
            };
            let Some(candidate_entry) = next.get(&candidate) else {
                continue;
            };
            if candidate_entry.referenced.swap(false, Ordering::Relaxed) {
                clock.push_back(candidate);
            } else {
                next.remove(&candidate);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        next.insert(offset, entry);
        clock.push_back(offset);
        self.entries.store(Arc::new(next));
        drop(clock);

        Ok(HazardGuard::from_arc(page))
    }

    pub fn invalidate(&self, offset: u64) {
        let _writer = self.writer.lock();
        let current = self.entries.load_full();
        if !current.contains_key(&offset) {
            return;
        }
        let mut next = (*current).clone();
        next.remove(&offset);
        self.entries.store(Arc::new(next));
        self.clock.lock().retain(|candidate| *candidate != offset);
    }

    pub fn stats(&self) -> BufferPoolStats {
        BufferPoolStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            resident_pages: self.entries.load().len(),
        }
    }
}
