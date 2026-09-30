//! Versioned, content-addressed, 4 KiB-aligned KV-cache storage.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::io::{AlignedPage, BlockLayout, DirectIo};
use crate::{Error, FaultPoint, Hash, Result};

pub const KV_TRANSFER_BLOCK_SIZE: usize = 4 * 1024 * 1024;
pub const KV_HEADER_SIZE: usize = 4096;
const KV_PAYLOAD_SIZE: usize = KV_TRANSFER_BLOCK_SIZE - KV_HEADER_SIZE;
const BLOCK_MAGIC: &[u8; 8] = b"ENGKVC01";
const BLOCK_VERSION: u16 = 1;
const LOG_MAGIC: &[u8; 8] = b"ENGKVM01";
const LOG_HEADER_SIZE: usize = 20;
const SPEC_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum KvDType {
    Fp16 = 1,
    Bf16 = 2,
    Fp32 = 3,
    Fp8 = 4,
}

impl KvDType {
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::Fp16 | Self::Bf16 => 2,
            Self::Fp32 => 4,
            Self::Fp8 => 1,
        }
    }

    fn decode(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Fp16),
            2 => Ok(Self::Bf16),
            3 => Ok(Self::Fp32),
            4 => Ok(Self::Fp8),
            _ => Err(Error::KvCache("unknown KV dtype".to_owned())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum KvLayout {
    VllmPagedV1 = 1,
    SglangLayerMajorV1 = 2,
}

impl KvLayout {
    fn decode(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::VllmPagedV1),
            2 => Ok(Self::SglangLayerMajorV1),
            _ => Err(Error::KvCache("unknown KV tensor layout".to_owned())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheSpec {
    pub model_fingerprint: Hash,
    pub dtype: KvDType,
    pub layout: KvLayout,
    pub num_layers: u32,
    pub num_blocks: u32,
    pub num_kv_heads: u32,
    pub head_size: u32,
    pub block_size: u32,
    pub sequence_length: u32,
    pub tensor_parallel_rank: u16,
    pub tensor_parallel_world: u16,
}

impl KvCacheSpec {
    pub fn expected_bytes(self) -> Result<usize> {
        if self.num_layers == 0
            || self.num_blocks == 0
            || self.num_kv_heads == 0
            || self.head_size == 0
            || self.block_size == 0
            || self.tensor_parallel_world == 0
            || self.tensor_parallel_rank >= self.tensor_parallel_world
        {
            return Err(Error::KvCache(
                "KV tensor dimensions and tensor-parallel metadata must be valid".to_owned(),
            ));
        }
        [
            self.num_layers as usize,
            2,
            self.num_blocks as usize,
            self.num_kv_heads as usize,
            self.head_size as usize,
            self.block_size as usize,
            self.dtype.element_bytes(),
        ]
        .into_iter()
        .try_fold(1_usize, |total, value| total.checked_mul(value))
        .ok_or_else(|| Error::KvCache("KV tensor byte length overflows usize".to_owned()))
    }

    pub fn vllm_key_shape(self) -> Result<[usize; 5]> {
        let vector = 16 / self.dtype.element_bytes();
        if self.head_size as usize % vector != 0 {
            return Err(Error::KvCache(
                "vLLM key head size must be divisible by 16 / element_size".to_owned(),
            ));
        }
        Ok([
            self.num_blocks as usize,
            self.num_kv_heads as usize,
            self.head_size as usize / vector,
            self.block_size as usize,
            vector,
        ])
    }

    pub fn vllm_value_shape(self) -> [usize; 4] {
        [
            self.num_blocks as usize,
            self.num_kv_heads as usize,
            self.head_size as usize,
            self.block_size as usize,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvBlockRef {
    pub hash: Hash,
    pub offset: u64,
    pub payload_length: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheManifest {
    pub cache_hash: Hash,
    pub spec: KvCacheSpec,
    pub total_bytes: u64,
    pub blocks: Vec<KvBlockRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheSnapshot {
    pub manifest: KvCacheManifest,
    pub bytes: Vec<u8>,
}

struct StoreState {
    manifests: HashMap<Uuid, KvCacheManifest>,
    committed_length: u64,
}

pub struct KvCacheStore {
    data: Arc<DirectIo>,
    log: Mutex<File>,
    healthy: AtomicBool,
    state: RwLock<StoreState>,
}

impl KvCacheStore {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        let layout = BlockLayout::new(KV_TRANSFER_BLOCK_SIZE, 4096)?;
        let data = Arc::new(DirectIo::open_with_layout(
            directory.join("kv-cache.dat"),
            256,
            layout,
        )?);
        let log_path = directory.join("kv-cache.log");
        let log = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&log_path)?;
        let (manifests, committed_length, valid_log_length) = replay_log(&log_path)?;
        if data.len() < committed_length {
            return Err(Error::KvCache(format!(
                "KV data file is shorter than committed watermark {committed_length}"
            )));
        }
        if data.len() > committed_length {
            data.truncate(committed_length)?;
        }
        if log.metadata()?.len() > valid_log_length {
            log.set_len(valid_log_length)?;
            log.sync_data()?;
        }
        let store = Self {
            data,
            log: Mutex::new(log),
            healthy: AtomicBool::new(true),
            state: RwLock::new(StoreState {
                manifests,
                committed_length,
            }),
        };
        store.validate_all()?;
        Ok(store)
    }

    pub fn put(&self, branch: Uuid, spec: KvCacheSpec, bytes: &[u8]) -> Result<KvCacheManifest> {
        self.put_with_fault(branch, spec, bytes, FaultPoint::None)
    }

    pub fn put_with_fault(
        &self,
        branch: Uuid,
        spec: KvCacheSpec,
        bytes: &[u8],
        fault: FaultPoint,
    ) -> Result<KvCacheManifest> {
        if !self.healthy.load(Ordering::Acquire) {
            return Err(Error::MetadataPoisoned);
        }
        let expected = spec.expected_bytes()?;
        if expected != bytes.len() {
            return Err(Error::KvCache(format!(
                "KV payload has {} bytes but tensor spec requires {expected}",
                bytes.len()
            )));
        }
        let spec_bytes = encode_spec(spec);
        let spec_hash = Hash(*blake3::hash(&spec_bytes).as_bytes());
        let chunk_count = bytes.len().div_ceil(KV_PAYLOAD_SIZE);
        let mut blocks = Vec::with_capacity(chunk_count);
        for (chunk_index, payload) in bytes.chunks(KV_PAYLOAD_SIZE).enumerate() {
            let block = encode_block(
                spec,
                spec_hash,
                chunk_index as u32,
                chunk_count as u32,
                payload,
            )?;
            let offset = self.data.append(&block)?;
            blocks.push(KvBlockRef {
                hash: Hash(*blake3::hash(payload).as_bytes()),
                offset,
                payload_length: payload.len() as u32,
            });
        }
        let manifest = KvCacheManifest {
            cache_hash: Hash(*blake3::hash(bytes).as_bytes()),
            spec,
            total_bytes: bytes.len() as u64,
            blocks,
        };
        if fault == FaultPoint::AfterPageWrites {
            return Err(Error::InjectedFault("after KV-cache writes"));
        }
        self.data.sync()?;
        if fault == FaultPoint::AfterDataSync {
            return Err(Error::InjectedFault("after KV-cache sync"));
        }
        let committed_length = self.data.len();
        self.append_event(
            branch,
            Some(&manifest),
            committed_length,
            fault == FaultPoint::DuringMetadataAppend,
        )?;
        if fault == FaultPoint::DuringMetadataAppend {
            return Err(Error::InjectedFault("during KV-cache metadata append"));
        }
        let mut state = self.state.write();
        state.manifests.insert(branch, manifest.clone());
        state.committed_length = committed_length;
        Ok(manifest)
    }

    pub fn inherit(&self, parent: Uuid, child: Uuid) -> Result<Option<KvCacheManifest>> {
        let manifest = self.state.read().manifests.get(&parent).cloned();
        let committed_length = self.state.read().committed_length;
        self.append_event(child, manifest.as_ref(), committed_length, false)?;
        if let Some(manifest) = &manifest {
            self.state.write().manifests.insert(child, manifest.clone());
        }
        Ok(manifest)
    }

    pub fn manifest(&self, branch: Uuid) -> Option<KvCacheManifest> {
        self.state.read().manifests.get(&branch).cloned()
    }

    pub fn get(&self, branch: Uuid) -> Result<Option<KvCacheSnapshot>> {
        let Some(manifest) = self.manifest(branch) else {
            return Ok(None);
        };
        let mut bytes = Vec::with_capacity(manifest.total_bytes as usize);
        let spec_hash = Hash(*blake3::hash(&encode_spec(manifest.spec)).as_bytes());
        for (expected_index, reference) in manifest.blocks.iter().enumerate() {
            let block = self.data.read(reference.offset)?;
            let payload = decode_block(
                block.as_slice(),
                reference.offset,
                manifest.spec,
                spec_hash,
                expected_index as u32,
                manifest.blocks.len() as u32,
            )?;
            if payload.len() != reference.payload_length as usize
                || Hash(*blake3::hash(payload).as_bytes()) != reference.hash
            {
                return Err(Error::KvCache(
                    "KV manifest block reference does not match data".to_owned(),
                ));
            }
            bytes.extend_from_slice(payload);
        }
        bytes.truncate(manifest.total_bytes as usize);
        if Hash(*blake3::hash(&bytes).as_bytes()) != manifest.cache_hash {
            return Err(Error::KvCache("KV cache BLAKE3 mismatch".to_owned()));
        }
        Ok(Some(KvCacheSnapshot { manifest, bytes }))
    }

    pub fn validate_all(&self) -> Result<()> {
        let branches: Vec<Uuid> = self.state.read().manifests.keys().copied().collect();
        for branch in branches {
            self.get(branch)?;
        }
        Ok(())
    }

    pub fn data_length(&self) -> u64 {
        self.state.read().committed_length
    }

    pub fn data_path(&self) -> &Path {
        self.data.path()
    }

    fn append_event(
        &self,
        branch: Uuid,
        manifest: Option<&KvCacheManifest>,
        data_length: u64,
        partial: bool,
    ) -> Result<()> {
        if !self.healthy.load(Ordering::Acquire) {
            return Err(Error::MetadataPoisoned);
        }
        let mut payload = Vec::new();
        payload.extend_from_slice(branch.as_bytes());
        payload.extend_from_slice(&data_length.to_le_bytes());
        payload.push(manifest.is_some() as u8);
        if let Some(manifest) = manifest {
            let encoded = encode_manifest(manifest);
            payload.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            payload.extend_from_slice(&encoded);
        }
        let mut record = Vec::with_capacity(LOG_HEADER_SIZE + payload.len());
        record.extend_from_slice(LOG_MAGIC);
        record.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        record.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        record.extend_from_slice(&crc32fast::hash(&record).to_le_bytes());
        record.extend_from_slice(&payload);
        let mut writer = self.log.lock();
        let result = (|| {
            writer.seek(SeekFrom::End(0))?;
            if partial {
                writer.write_all(&record[..record.len() / 2])?;
            } else {
                writer.write_all(&record)?;
            }
            writer.sync_data()?;
            Ok(())
        })();
        if partial || result.is_err() {
            self.healthy.store(false, Ordering::Release);
        }
        result
    }
}

fn encode_block(
    spec: KvCacheSpec,
    spec_hash: Hash,
    chunk_index: u32,
    chunk_count: u32,
    payload: &[u8],
) -> Result<AlignedPage> {
    if payload.len() > KV_PAYLOAD_SIZE {
        return Err(Error::KvCache("KV chunk exceeds transfer block".to_owned()));
    }
    let layout = BlockLayout::new(KV_TRANSFER_BLOCK_SIZE, 4096)?;
    let mut block = AlignedPage::zeroed_for(layout);
    let bytes = block.as_mut_slice();
    bytes[..8].copy_from_slice(BLOCK_MAGIC);
    bytes[8..10].copy_from_slice(&BLOCK_VERSION.to_le_bytes());
    bytes[10] = spec.dtype as u8;
    bytes[11] = spec.layout as u8;
    bytes[12..16].copy_from_slice(&chunk_index.to_le_bytes());
    bytes[16..20].copy_from_slice(&chunk_count.to_le_bytes());
    bytes[20..24].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes[24..56].copy_from_slice(&spec_hash.0);
    bytes[56..60].copy_from_slice(&crc32fast::hash(payload).to_le_bytes());
    bytes[60..64].copy_from_slice(&crc32fast::hash(&bytes[..60]).to_le_bytes());
    bytes[KV_HEADER_SIZE..KV_HEADER_SIZE + payload.len()].copy_from_slice(payload);
    Ok(block)
}

fn decode_block(
    bytes: &[u8],
    offset: u64,
    spec: KvCacheSpec,
    spec_hash: Hash,
    expected_index: u32,
    expected_count: u32,
) -> Result<&[u8]> {
    if bytes.len() != KV_TRANSFER_BLOCK_SIZE
        || &bytes[..8] != BLOCK_MAGIC
        || u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != BLOCK_VERSION
    {
        return Err(Error::CorruptPage {
            offset,
            reason: "invalid KV-cache block magic, version, or size".to_owned(),
        });
    }
    if bytes[10] != spec.dtype as u8
        || bytes[11] != spec.layout as u8
        || u32::from_le_bytes(bytes[12..16].try_into().unwrap()) != expected_index
        || u32::from_le_bytes(bytes[16..20].try_into().unwrap()) != expected_count
        || Hash(bytes[24..56].try_into().unwrap()) != spec_hash
        || crc32fast::hash(&bytes[..60]) != u32::from_le_bytes(bytes[60..64].try_into().unwrap())
    {
        return Err(Error::CorruptPage {
            offset,
            reason: "KV-cache block header mismatch".to_owned(),
        });
    }
    let payload_length = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    if payload_length > KV_PAYLOAD_SIZE {
        return Err(Error::CorruptPage {
            offset,
            reason: "KV-cache payload length exceeds block".to_owned(),
        });
    }
    let payload = &bytes[KV_HEADER_SIZE..KV_HEADER_SIZE + payload_length];
    if crc32fast::hash(payload) != u32::from_le_bytes(bytes[56..60].try_into().unwrap()) {
        return Err(Error::CorruptPage {
            offset,
            reason: "KV-cache payload CRC32 mismatch".to_owned(),
        });
    }
    Ok(payload)
}

fn encode_spec(spec: KvCacheSpec) -> [u8; SPEC_SIZE] {
    let mut bytes = [0_u8; SPEC_SIZE];
    bytes[..32].copy_from_slice(&spec.model_fingerprint.0);
    bytes[32] = spec.dtype as u8;
    bytes[33] = spec.layout as u8;
    let values = [
        spec.num_layers,
        spec.num_blocks,
        spec.num_kv_heads,
        spec.head_size,
        spec.block_size,
        spec.sequence_length,
    ];
    for (index, value) in values.into_iter().enumerate() {
        let offset = 36 + index * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[60..62].copy_from_slice(&spec.tensor_parallel_rank.to_le_bytes());
    bytes[62..64].copy_from_slice(&spec.tensor_parallel_world.to_le_bytes());
    bytes
}

fn decode_spec(bytes: [u8; SPEC_SIZE]) -> Result<KvCacheSpec> {
    Ok(KvCacheSpec {
        model_fingerprint: Hash(bytes[..32].try_into().unwrap()),
        dtype: KvDType::decode(bytes[32])?,
        layout: KvLayout::decode(bytes[33])?,
        num_layers: u32::from_le_bytes(bytes[36..40].try_into().unwrap()),
        num_blocks: u32::from_le_bytes(bytes[40..44].try_into().unwrap()),
        num_kv_heads: u32::from_le_bytes(bytes[44..48].try_into().unwrap()),
        head_size: u32::from_le_bytes(bytes[48..52].try_into().unwrap()),
        block_size: u32::from_le_bytes(bytes[52..56].try_into().unwrap()),
        sequence_length: u32::from_le_bytes(bytes[56..60].try_into().unwrap()),
        tensor_parallel_rank: u16::from_le_bytes(bytes[60..62].try_into().unwrap()),
        tensor_parallel_world: u16::from_le_bytes(bytes[62..64].try_into().unwrap()),
    })
}

fn encode_manifest(manifest: &KvCacheManifest) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&manifest.cache_hash.0);
    bytes.extend_from_slice(&encode_spec(manifest.spec));
    bytes.extend_from_slice(&manifest.total_bytes.to_le_bytes());
    bytes.extend_from_slice(&(manifest.blocks.len() as u32).to_le_bytes());
    for block in &manifest.blocks {
        bytes.extend_from_slice(&block.hash.0);
        bytes.extend_from_slice(&block.offset.to_le_bytes());
        bytes.extend_from_slice(&block.payload_length.to_le_bytes());
    }
    bytes
}

fn decode_manifest(bytes: &[u8]) -> Result<KvCacheManifest> {
    if bytes.len() < 108 {
        return Err(Error::KvCache("KV manifest is truncated".to_owned()));
    }
    let cache_hash = Hash(bytes[..32].try_into().unwrap());
    let spec = decode_spec(bytes[32..96].try_into().unwrap())?;
    let total_bytes = u64::from_le_bytes(bytes[96..104].try_into().unwrap());
    let count = u32::from_le_bytes(bytes[104..108].try_into().unwrap()) as usize;
    if bytes.len() != 108 + count * 44 {
        return Err(Error::KvCache(
            "KV manifest block table has invalid length".to_owned(),
        ));
    }
    let mut blocks = Vec::with_capacity(count);
    for index in 0..count {
        let offset = 108 + index * 44;
        blocks.push(KvBlockRef {
            hash: Hash(bytes[offset..offset + 32].try_into().unwrap()),
            offset: u64::from_le_bytes(bytes[offset + 32..offset + 40].try_into().unwrap()),
            payload_length: u32::from_le_bytes(bytes[offset + 40..offset + 44].try_into().unwrap()),
        });
    }
    Ok(KvCacheManifest {
        cache_hash,
        spec,
        total_bytes,
        blocks,
    })
}

fn replay_log(path: &Path) -> Result<(HashMap<Uuid, KvCacheManifest>, u64, u64)> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let mut manifests = HashMap::new();
    let mut position = 0;
    let mut committed_length = 0;
    while position < bytes.len() {
        if bytes.len() - position < LOG_HEADER_SIZE {
            break;
        }
        if &bytes[position..position + 8] != LOG_MAGIC {
            return Err(Error::KvCache("invalid KV metadata magic".to_owned()));
        }
        let expected_header_crc =
            u32::from_le_bytes(bytes[position + 16..position + 20].try_into().unwrap());
        if crc32fast::hash(&bytes[position..position + 16]) != expected_header_crc {
            return Err(Error::KvCache(
                "KV metadata header CRC32 mismatch".to_owned(),
            ));
        }
        let length =
            u32::from_le_bytes(bytes[position + 8..position + 12].try_into().unwrap()) as usize;
        let end = position
            .checked_add(LOG_HEADER_SIZE + length)
            .ok_or_else(|| Error::KvCache("KV metadata length overflows".to_owned()))?;
        if end > bytes.len() {
            break;
        }
        let payload = &bytes[position + LOG_HEADER_SIZE..end];
        if crc32fast::hash(payload)
            != u32::from_le_bytes(bytes[position + 12..position + 16].try_into().unwrap())
        {
            return Err(Error::KvCache(
                "KV metadata payload CRC32 mismatch".to_owned(),
            ));
        }
        if payload.len() < 25 {
            return Err(Error::KvCache(
                "KV metadata payload is truncated".to_owned(),
            ));
        }
        let branch = Uuid::from_bytes(payload[..16].try_into().unwrap());
        let data_length = u64::from_le_bytes(payload[16..24].try_into().unwrap());
        if data_length % KV_TRANSFER_BLOCK_SIZE as u64 != 0 || data_length < committed_length {
            return Err(Error::KvCache(
                "invalid or decreasing KV data watermark".to_owned(),
            ));
        }
        committed_length = data_length;
        match payload[24] {
            0 => {
                manifests.remove(&branch);
            }
            1 => {
                if payload.len() < 29 {
                    return Err(Error::KvCache("KV manifest event is truncated".to_owned()));
                }
                let manifest_length =
                    u32::from_le_bytes(payload[25..29].try_into().unwrap()) as usize;
                if payload.len() != 29 + manifest_length {
                    return Err(Error::KvCache(
                        "KV manifest event length mismatch".to_owned(),
                    ));
                }
                let manifest = decode_manifest(&payload[29..])?;
                if manifest.blocks.iter().any(|block| {
                    block
                        .offset
                        .checked_add(KV_TRANSFER_BLOCK_SIZE as u64)
                        .is_none_or(|end| end > data_length)
                }) {
                    return Err(Error::KvCache(
                        "KV manifest references data beyond watermark".to_owned(),
                    ));
                }
                manifests.insert(branch, manifest);
            }
            _ => return Err(Error::KvCache("invalid KV manifest marker".to_owned())),
        }
        position = end;
    }
    Ok((manifests, committed_length, position as u64))
}
