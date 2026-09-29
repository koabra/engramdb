//! Linux direct-I/O backed by `io_uring`.

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::fs::{self, File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use io_uring::{opcode, types, IoUring};
use parking_lot::Mutex;

use crate::{Error, Result};

pub const PAGE_SIZE: usize = 4096;
const ALIGNMENT: usize = 4096;

/// A zeroed 4 KiB allocation suitable for Linux `O_DIRECT`.
pub struct AlignedPage {
    pointer: NonNull<u8>,
}

// The allocation is uniquely owned and contains no internal references.
unsafe impl Send for AlignedPage {}
unsafe impl Sync for AlignedPage {}

impl AlignedPage {
    pub fn zeroed() -> Self {
        let layout = Layout::from_size_align(PAGE_SIZE, ALIGNMENT).expect("valid page layout");
        // SAFETY: the layout is non-zero and valid.
        let pointer = unsafe { alloc_zeroed(layout) };
        let pointer = NonNull::new(pointer).unwrap_or_else(|| handle_alloc_error(layout));
        Self { pointer }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is live for PAGE_SIZE bytes.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), PAGE_SIZE) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: this type uniquely owns its allocation.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), PAGE_SIZE) }
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.pointer.as_ptr()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.pointer.as_ptr()
    }
}

impl Clone for AlignedPage {
    fn clone(&self) -> Self {
        let mut clone = Self::zeroed();
        clone.as_mut_slice().copy_from_slice(self.as_slice());
        clone
    }
}

impl Default for AlignedPage {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Drop for AlignedPage {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(PAGE_SIZE, ALIGNMENT).expect("valid page layout");
        // SAFETY: pointer was allocated with this exact layout.
        unsafe { dealloc(self.pointer.as_ptr(), layout) };
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoStats {
    pub read_operations: u64,
    pub write_operations: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub sync_operations: u64,
}

/// Serialized submission over one `io_uring`. The kernel still executes direct
/// reads and writes asynchronously; serialization keeps completion ownership
/// simple for the Phase 1 synchronous transaction API.
pub struct DirectIo {
    path: PathBuf,
    file: File,
    ring: Mutex<IoUring>,
    next_offset: AtomicU64,
    read_operations: AtomicU64,
    write_operations: AtomicU64,
    bytes_read: AtomicU64,
    bytes_written: AtomicU64,
    sync_operations: AtomicU64,
}

impl DirectIo {
    pub fn open(path: impl AsRef<Path>, queue_depth: u32) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT | libc::O_CLOEXEC)
            .mode(0o600)
            .open(path)?;
        let mut length = file.metadata()?.len();
        if length % PAGE_SIZE as u64 != 0 {
            // A power loss can leave the final direct-I/O page short. It cannot
            // be referenced by a durable metadata record because data fsync
            // precedes metadata append, so discard only this trailing fragment.
            length -= length % PAGE_SIZE as u64;
            file.set_len(length)?;
        }
        let ring = IoUring::new(queue_depth.max(2))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            ring: Mutex::new(ring),
            next_offset: AtomicU64::new(length),
            read_operations: AtomicU64::new(0),
            write_operations: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            sync_operations: AtomicU64::new(0),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> u64 {
        self.next_offset.load(Ordering::Acquire)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn truncate(&self, length: u64) -> Result<()> {
        if length % PAGE_SIZE as u64 != 0 {
            return Err(Error::Invariant(
                "direct-I/O truncation must be page aligned".to_owned(),
            ));
        }
        self.file.set_len(length)?;
        self.next_offset.store(length, Ordering::Release);
        Ok(())
    }

    pub fn append(&self, page: &AlignedPage) -> Result<u64> {
        let offset = self
            .next_offset
            .fetch_add(PAGE_SIZE as u64, Ordering::AcqRel);
        if let Err(error) = self.write_at(offset, page) {
            let _ = self.next_offset.compare_exchange(
                offset + PAGE_SIZE as u64,
                offset,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            return Err(error);
        }
        Ok(offset)
    }

    pub fn write_at(&self, offset: u64, page: &AlignedPage) -> Result<()> {
        if offset % PAGE_SIZE as u64 != 0 {
            return Err(Error::Invariant("unaligned direct write offset".to_owned()));
        }
        let entry = opcode::Write::new(
            types::Fd(self.file.as_raw_fd()),
            page.as_ptr(),
            PAGE_SIZE as _,
        )
        .offset(offset)
        .build()
        .user_data(offset);
        self.submit(entry)?;
        self.write_operations.fetch_add(1, Ordering::Relaxed);
        self.bytes_written
            .fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);
        self.next_offset
            .fetch_max(offset + PAGE_SIZE as u64, Ordering::Release);
        Ok(())
    }

    pub fn read(&self, offset: u64) -> Result<AlignedPage> {
        if offset % PAGE_SIZE as u64 != 0 || offset >= self.len() {
            return Err(Error::CorruptPage {
                offset,
                reason: "read offset is outside the page file".to_owned(),
            });
        }
        let mut page = AlignedPage::zeroed();
        let entry = opcode::Read::new(
            types::Fd(self.file.as_raw_fd()),
            page.as_mut_ptr(),
            PAGE_SIZE as _,
        )
        .offset(offset)
        .build()
        .user_data(offset);
        self.submit(entry)?;
        self.read_operations.fetch_add(1, Ordering::Relaxed);
        self.bytes_read
            .fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);
        Ok(page)
    }

    fn submit(&self, entry: io_uring::squeue::Entry) -> Result<()> {
        let mut ring = self.ring.lock();
        // SAFETY: pointers in entries refer to AlignedPage buffers that remain
        // alive until submit_and_wait and completion consumption return.
        unsafe {
            ring.submission()
                .push(&entry)
                .map_err(|_| Error::Invariant("io_uring submission queue is full".to_owned()))?;
        }
        ring.submit_and_wait(1)?;
        let completion = ring
            .completion()
            .next()
            .ok_or_else(|| Error::Invariant("io_uring returned no completion".to_owned()))?;
        let result = completion.result();
        if result < 0 {
            return Err(Error::Uring(-result));
        }
        if result as usize != PAGE_SIZE {
            return Err(Error::Invariant(format!(
                "short direct I/O: {result} of {PAGE_SIZE} bytes"
            )));
        }
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        let entry = opcode::Fsync::new(types::Fd(self.file.as_raw_fd()))
            .build()
            .user_data(u64::MAX);
        let mut ring = self.ring.lock();
        // SAFETY: the fd remains open through completion.
        unsafe {
            ring.submission()
                .push(&entry)
                .map_err(|_| Error::Invariant("io_uring submission queue is full".to_owned()))?;
        }
        ring.submit_and_wait(1)?;
        let result = ring
            .completion()
            .next()
            .ok_or_else(|| Error::Invariant("io_uring returned no fsync completion".to_owned()))?
            .result();
        if result < 0 {
            return Err(Error::Uring(-result));
        }
        self.sync_operations.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn stats(&self) -> IoStats {
        IoStats {
            read_operations: self.read_operations.load(Ordering::Relaxed),
            write_operations: self.write_operations.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            sync_operations: self.sync_operations.load(Ordering::Relaxed),
        }
    }
}
