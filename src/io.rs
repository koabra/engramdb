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
pub const FUSED_BLOCK_SIZE: usize = 64 * 1024;

/// A zeroed, size-aligned allocation suitable for Linux `O_DIRECT`.
pub struct AlignedBlock<const SIZE: usize> {
    pointer: NonNull<u8>,
}

// The allocation is uniquely owned and contains no internal references.
unsafe impl<const SIZE: usize> Send for AlignedBlock<SIZE> {}
unsafe impl<const SIZE: usize> Sync for AlignedBlock<SIZE> {}

impl<const SIZE: usize> AlignedBlock<SIZE> {
    pub fn zeroed() -> Self {
        assert!(
            SIZE >= PAGE_SIZE && SIZE.is_power_of_two(),
            "direct-I/O block size must be a power of two and at least 4 KiB"
        );
        let layout = Layout::from_size_align(SIZE, SIZE).expect("valid block layout");
        // SAFETY: the layout is non-zero and valid.
        let pointer = unsafe { alloc_zeroed(layout) };
        let pointer = NonNull::new(pointer).unwrap_or_else(|| handle_alloc_error(layout));
        Self { pointer }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is live for SIZE bytes.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), SIZE) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: this type uniquely owns its allocation.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), SIZE) }
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.pointer.as_ptr()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    pub fn is_aligned(&self) -> bool {
        (self.pointer.as_ptr() as usize).is_multiple_of(SIZE)
    }
}

impl<const SIZE: usize> Clone for AlignedBlock<SIZE> {
    fn clone(&self) -> Self {
        let mut clone = Self::zeroed();
        clone.as_mut_slice().copy_from_slice(self.as_slice());
        clone
    }
}

impl<const SIZE: usize> Default for AlignedBlock<SIZE> {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl<const SIZE: usize> Drop for AlignedBlock<SIZE> {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(SIZE, SIZE).expect("valid block layout");
        // SAFETY: pointer was allocated with this exact layout.
        unsafe { dealloc(self.pointer.as_ptr(), layout) };
    }
}

pub type AlignedPage = AlignedBlock<PAGE_SIZE>;

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
pub struct DirectIo<const BLOCK_SIZE: usize = PAGE_SIZE> {
    path: PathBuf,
    file: File,
    ring: Mutex<IoUring>,
    next_token: AtomicU64,
    next_offset: AtomicU64,
    read_operations: AtomicU64,
    write_operations: AtomicU64,
    bytes_read: AtomicU64,
    bytes_written: AtomicU64,
    sync_operations: AtomicU64,
}

impl<const BLOCK_SIZE: usize> DirectIo<BLOCK_SIZE> {
    pub fn open(path: impl AsRef<Path>, queue_depth: u32) -> Result<Self> {
        if BLOCK_SIZE < PAGE_SIZE || !BLOCK_SIZE.is_power_of_two() {
            return Err(Error::Invariant(
                "direct-I/O block size must be a power of two and at least 4 KiB".to_owned(),
            ));
        }
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
        if !length.is_multiple_of(BLOCK_SIZE as u64) {
            // A power loss can leave the final direct-I/O page short. It cannot
            // be referenced by a durable metadata record because data fsync
            // precedes metadata append, so discard only this trailing fragment.
            length -= length % BLOCK_SIZE as u64;
            file.set_len(length)?;
        }
        let ring = IoUring::new(queue_depth.max(2))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            ring: Mutex::new(ring),
            next_token: AtomicU64::new(1),
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
        if !length.is_multiple_of(BLOCK_SIZE as u64) {
            return Err(Error::Invariant(
                "direct-I/O truncation must be page aligned".to_owned(),
            ));
        }
        self.file.set_len(length)?;
        self.next_offset.store(length, Ordering::Release);
        Ok(())
    }

    pub fn append(&self, page: &AlignedBlock<BLOCK_SIZE>) -> Result<u64> {
        let offset = self
            .next_offset
            .fetch_add(BLOCK_SIZE as u64, Ordering::AcqRel);
        if let Err(error) = self.write_at(offset, page) {
            let _ = self.next_offset.compare_exchange(
                offset + BLOCK_SIZE as u64,
                offset,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            return Err(error);
        }
        Ok(offset)
    }

    pub fn write_at(&self, offset: u64, page: &AlignedBlock<BLOCK_SIZE>) -> Result<()> {
        if !offset.is_multiple_of(BLOCK_SIZE as u64) {
            return Err(Error::Invariant("unaligned direct write offset".to_owned()));
        }
        // The SQE owns no buffer reference. Clone into request-owned storage so
        // callers cannot mutate or drop memory while the kernel can access it.
        let owned_page = page.clone();
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        let entry = opcode::Write::new(
            types::Fd(self.file.as_raw_fd()),
            owned_page.as_ptr(),
            BLOCK_SIZE as _,
        )
        .offset(offset)
        .build()
        .user_data(token);
        if let Err(error) = self.submit(entry, token, BLOCK_SIZE as i32) {
            // A failed io_uring_enter can leave submission state uncertain. The
            // allocation is intentionally leaked so an eventual kernel access
            // can never become a use-after-free. The engine treats the error as
            // fatal for this operation; process restart reclaims the memory.
            std::mem::forget(owned_page);
            return Err(error);
        }
        self.write_operations.fetch_add(1, Ordering::Relaxed);
        self.bytes_written
            .fetch_add(BLOCK_SIZE as u64, Ordering::Relaxed);
        self.next_offset
            .fetch_max(offset + BLOCK_SIZE as u64, Ordering::Release);
        Ok(())
    }

    pub fn read(&self, offset: u64) -> Result<AlignedBlock<BLOCK_SIZE>> {
        if !offset.is_multiple_of(BLOCK_SIZE as u64) || offset >= self.len() {
            return Err(Error::CorruptPage {
                offset,
                reason: "read offset is outside the page file".to_owned(),
            });
        }
        let mut page = AlignedBlock::<BLOCK_SIZE>::zeroed();
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        let entry = opcode::Read::new(
            types::Fd(self.file.as_raw_fd()),
            page.as_mut_ptr(),
            BLOCK_SIZE as _,
        )
        .offset(offset)
        .build()
        .user_data(token);
        if let Err(error) = self.submit(entry, token, BLOCK_SIZE as i32) {
            // See write_at: retain uncertain request memory for process life.
            std::mem::forget(page);
            return Err(error);
        }
        self.read_operations.fetch_add(1, Ordering::Relaxed);
        self.bytes_read
            .fetch_add(BLOCK_SIZE as u64, Ordering::Relaxed);
        Ok(page)
    }

    fn submit(
        &self,
        entry: io_uring::squeue::Entry,
        token: u64,
        expected_result: i32,
    ) -> Result<()> {
        let mut ring = self.ring.lock();
        // SAFETY: pointers in entries refer to AlignedBlock buffers that remain
        // alive until submit_and_wait and completion consumption return.
        unsafe {
            ring.submission()
                .push(&entry)
                .map_err(|_| Error::Invariant("io_uring submission queue is full".to_owned()))?;
        }
        loop {
            match ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(Error::Io(error)),
            }
            loop {
                let completion = ring
                    .completion()
                    .next()
                    .map(|entry| (entry.user_data(), entry.result()));
                let Some((completed_token, result)) = completion else {
                    break;
                };
                if completed_token != token {
                    // Completion for a previously uncertain submission. Its
                    // request buffer was leaked deliberately and remains valid.
                    continue;
                }
                if result < 0 {
                    return Err(Error::Uring(-result));
                }
                if result != expected_result {
                    return Err(Error::Invariant(format!(
                        "unexpected io_uring result: {result}, expected {expected_result}"
                    )));
                }
                return Ok(());
            }
        }
    }

    pub fn sync(&self) -> Result<()> {
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        let entry = opcode::Fsync::new(types::Fd(self.file.as_raw_fd()))
            .build()
            .user_data(token);
        self.submit(entry, token, 0)?;
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
