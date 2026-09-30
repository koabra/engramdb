//! Append-only, content-addressed durable HNSW checkpoints.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use parking_lot::Mutex;

use crate::tree::Hash;
use crate::{Error, Result};

const MAGIC: &[u8; 8] = b"ENGHCP01";
const HEADER_SIZE: usize = 56;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CheckpointRoot {
    pub hash: Hash,
    pub offset: u64,
    pub length: u64,
}

pub(crate) struct CheckpointLog {
    path: PathBuf,
    writer: Mutex<File>,
}

impl CheckpointLog {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let writer = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: Mutex::new(writer),
        })
    }

    pub fn len(&self) -> Result<u64> {
        Ok(self.writer.lock().metadata()?.len())
    }

    pub fn append(&self, payload: &[u8]) -> Result<CheckpointRoot> {
        let payload_length = u64::try_from(payload.len())
            .map_err(|_| Error::Invariant("HNSW checkpoint exceeds u64".to_owned()))?;
        let hash = Hash(*blake3::hash(payload).as_bytes());
        let mut header = Vec::with_capacity(HEADER_SIZE);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&payload_length.to_le_bytes());
        header.extend_from_slice(&hash.0);
        header.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
        header.extend_from_slice(&crc32fast::hash(&header).to_le_bytes());
        debug_assert_eq!(header.len(), HEADER_SIZE);
        let mut writer = self.writer.lock();
        let offset = writer.seek(SeekFrom::End(0))?;
        writer.write_all(&header)?;
        writer.write_all(payload)?;
        Ok(CheckpointRoot {
            hash,
            offset,
            length: HEADER_SIZE as u64 + payload_length,
        })
    }

    pub fn read(&self, root: CheckpointRoot) -> Result<Vec<u8>> {
        if root.length < HEADER_SIZE as u64 {
            return Err(Error::CorruptMetadata {
                offset: root.offset,
                reason: "HNSW checkpoint root length is shorter than header".to_owned(),
            });
        }
        let end = root
            .offset
            .checked_add(root.length)
            .ok_or_else(|| Error::CorruptMetadata {
                offset: root.offset,
                reason: "HNSW checkpoint root range overflows".to_owned(),
            })?;
        if end > self.len()? {
            return Err(Error::CorruptMetadata {
                offset: root.offset,
                reason: "HNSW checkpoint root exceeds file".to_owned(),
            });
        }
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(root.offset))?;
        let mut header = [0_u8; HEADER_SIZE];
        file.read_exact(&mut header)?;
        if &header[..8] != MAGIC {
            return Err(Error::CorruptMetadata {
                offset: root.offset,
                reason: "invalid HNSW checkpoint magic".to_owned(),
            });
        }
        let expected_header_crc = u32::from_le_bytes(header[52..56].try_into().unwrap());
        if crc32fast::hash(&header[..52]) != expected_header_crc {
            return Err(Error::CorruptMetadata {
                offset: root.offset,
                reason: "HNSW checkpoint header CRC32 mismatch".to_owned(),
            });
        }
        let payload_length = u64::from_le_bytes(header[8..16].try_into().unwrap());
        if payload_length + HEADER_SIZE as u64 != root.length {
            return Err(Error::CorruptMetadata {
                offset: root.offset,
                reason: "HNSW checkpoint root length mismatch".to_owned(),
            });
        }
        let stored_hash = Hash(header[16..48].try_into().unwrap());
        if stored_hash != root.hash {
            return Err(Error::CorruptMetadata {
                offset: root.offset,
                reason: "HNSW checkpoint root hash mismatch".to_owned(),
            });
        }
        let mut payload = vec![0_u8; payload_length as usize];
        file.read_exact(&mut payload)?;
        let expected_crc = u32::from_le_bytes(header[48..52].try_into().unwrap());
        if crc32fast::hash(&payload) != expected_crc {
            return Err(Error::CorruptMetadata {
                offset: root.offset,
                reason: "HNSW checkpoint payload CRC32 mismatch".to_owned(),
            });
        }
        if Hash(*blake3::hash(&payload).as_bytes()) != root.hash {
            return Err(Error::CorruptMetadata {
                offset: root.offset,
                reason: "HNSW checkpoint payload BLAKE3 mismatch".to_owned(),
            });
        }
        Ok(payload)
    }

    pub fn sync(&self) -> Result<()> {
        self.writer.lock().sync_data()?;
        Ok(())
    }

    pub fn truncate(&self, length: u64) -> Result<()> {
        let mut writer = self.writer.lock();
        writer.flush()?;
        writer.set_len(length)?;
        writer.seek(SeekFrom::End(0))?;
        Ok(())
    }
}
