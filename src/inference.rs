//! Branch-scoped KV-cache API and native transfer tickets for inference adapters.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    Engine, Error, HardwareCapabilities, Hash, KvCacheManifest, KvCacheSnapshot, KvCacheSpec,
    KvCacheStore, Result, TransferPath, KV_HEADER_SIZE,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvTransferExtent {
    pub file_offset: u64,
    pub device_offset: u64,
    pub payload_length: u32,
    pub transfer_length: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvRestoreTicket {
    pub data_path: PathBuf,
    pub cache_hash: Hash,
    pub spec: KvCacheSpec,
    pub total_bytes: u64,
    pub extents: Vec<KvTransferExtent>,
    pub transfer_path: TransferPath,
}

pub struct InferenceManager {
    engine: Arc<Engine>,
    store: Arc<KvCacheStore>,
    capabilities: HardwareCapabilities,
}

impl InferenceManager {
    pub fn new(engine: Arc<Engine>, store: Arc<KvCacheStore>) -> Self {
        Self {
            engine,
            store,
            capabilities: HardwareCapabilities::detect(),
        }
    }

    pub fn engine(&self) -> &Arc<Engine> {
        &self.engine
    }

    pub fn store(&self) -> &Arc<KvCacheStore> {
        &self.store
    }

    pub fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    pub fn put(&self, branch: Uuid, spec: KvCacheSpec, bytes: &[u8]) -> Result<KvCacheManifest> {
        self.engine.branch(branch)?;
        self.store.put(branch, spec, bytes)
    }

    pub fn get(&self, branch: Uuid) -> Result<Option<KvCacheSnapshot>> {
        self.engine.branch(branch)?;
        self.store.get(branch)
    }

    pub fn inherit(&self, parent: Uuid, child: Uuid) -> Result<Option<KvCacheManifest>> {
        self.engine.branch(parent)?;
        self.engine.branch(child)?;
        self.store.inherit(parent, child)
    }

    pub fn restore_ticket(&self, branch: Uuid) -> Result<Option<KvRestoreTicket>> {
        self.engine.branch(branch)?;
        let Some(manifest) = self.store.manifest(branch) else {
            return Ok(None);
        };
        let mut device_offset = 0_u64;
        let extents = manifest
            .blocks
            .iter()
            .map(|block| {
                let transfer_length = (block.payload_length as usize)
                    .div_ceil(4096)
                    .checked_mul(4096)
                    .and_then(|length| u32::try_from(length).ok())
                    .ok_or_else(|| Error::KvCache("KV transfer length overflows u32".to_owned()))?;
                let extent = KvTransferExtent {
                    file_offset: block.offset + KV_HEADER_SIZE as u64,
                    device_offset,
                    payload_length: block.payload_length,
                    transfer_length,
                };
                device_offset += transfer_length as u64;
                Ok(extent)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(KvRestoreTicket {
            data_path: self.store.data_path().to_path_buf(),
            cache_hash: manifest.cache_hash,
            spec: manifest.spec,
            total_bytes: manifest.total_bytes,
            extents,
            transfer_path: self.capabilities.transfer_path,
        }))
    }
}
