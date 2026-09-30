use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use engramdb::{
    Engine, FaultPoint, HardwareCapabilities, Hash, InferenceManager, KvCacheSpec, KvCacheStore,
    KvDType, KvLayout, SessionManager, TransferPath, KV_HEADER_SIZE, KV_TRANSFER_BLOCK_SIZE,
};
use tempfile::tempdir;

fn spec(blocks: u32) -> KvCacheSpec {
    KvCacheSpec {
        model_fingerprint: Hash(*blake3::hash(b"tiny-llama-test").as_bytes()),
        dtype: KvDType::Fp32,
        layout: KvLayout::VllmPagedV1,
        num_layers: 1,
        num_blocks: blocks,
        num_kv_heads: 1,
        head_size: 4,
        block_size: 2,
        sequence_length: blocks * 2,
        tensor_parallel_rank: 0,
        tensor_parallel_world: 1,
    }
}

fn payload(spec: KvCacheSpec, seed: u8) -> Vec<u8> {
    (0..spec.expected_bytes().unwrap())
        .map(|index| seed.wrapping_add((index % 251) as u8))
        .collect()
}

#[test]
fn kv_cache_round_trips_across_multiple_aligned_blocks() {
    let directory = tempdir().unwrap();
    let branch = uuid::Uuid::new_v4();
    let spec = spec(131_072);
    let bytes = payload(spec, 7);
    assert!(bytes.len() > KV_TRANSFER_BLOCK_SIZE);
    let expected_hash;
    {
        let store = KvCacheStore::open(directory.path()).unwrap();
        let manifest = store.put(branch, spec, &bytes).unwrap();
        expected_hash = manifest.cache_hash;
        assert!(manifest.blocks.len() > 1);
        assert!(manifest
            .blocks
            .iter()
            .all(|block| block.offset % KV_TRANSFER_BLOCK_SIZE as u64 == 0));
    }
    let reopened = KvCacheStore::open(directory.path()).unwrap();
    let restored = reopened.get(branch).unwrap().unwrap();
    assert_eq!(restored.manifest.cache_hash, expected_hash);
    assert_eq!(restored.bytes, bytes);
    assert_eq!(
        restored.manifest.spec.vllm_key_shape().unwrap(),
        [131_072, 1, 1, 2, 4]
    );
    assert_eq!(
        restored.manifest.spec.vllm_value_shape(),
        [131_072, 1, 4, 2]
    );
}

#[test]
fn kv_fault_boundaries_restore_committed_watermark() {
    for fault in [
        FaultPoint::AfterPageWrites,
        FaultPoint::AfterDataSync,
        FaultPoint::DuringMetadataAppend,
    ] {
        let directory = tempdir().unwrap();
        let stable_branch = uuid::Uuid::new_v4();
        let orphan_branch = uuid::Uuid::new_v4();
        let spec = spec(1);
        let stable = payload(spec, 1);
        let committed_length;
        {
            let store = KvCacheStore::open(directory.path()).unwrap();
            store.put(stable_branch, spec, &stable).unwrap();
            committed_length = store.data_length();
            assert!(store
                .put_with_fault(orphan_branch, spec, &payload(spec, 2), fault)
                .is_err());
        }
        let recovered = KvCacheStore::open(directory.path()).unwrap();
        assert_eq!(recovered.data_length(), committed_length);
        assert_eq!(recovered.get(stable_branch).unwrap().unwrap().bytes, stable);
        assert!(recovered.get(orphan_branch).unwrap().is_none());
    }
}

#[test]
fn branch_sessions_inherit_and_isolate_kv_manifests() {
    let directory = tempdir().unwrap();
    let engine = Arc::new(Engine::open(directory.path()).unwrap());
    let store = Arc::new(KvCacheStore::open(directory.path()).unwrap());
    let inference = Arc::new(InferenceManager::new(Arc::clone(&engine), store));
    let sessions = SessionManager::new_with_inference(Arc::clone(&engine), inference);
    let main = engine.main_branch().id;
    let parent = sessions.fork_session(main).unwrap();
    let spec = spec(1);
    let parent_bytes = payload(spec, 3);
    sessions
        .put_kv_cache(parent.id, spec, &parent_bytes)
        .unwrap();
    let child = sessions.fork_session(parent.id).unwrap();
    assert_eq!(
        sessions.get_kv_cache(child.id).unwrap().unwrap().bytes,
        parent_bytes
    );
    let child_bytes = payload(spec, 9);
    sessions.put_kv_cache(child.id, spec, &child_bytes).unwrap();
    assert_eq!(
        sessions.get_kv_cache(parent.id).unwrap().unwrap().bytes,
        parent_bytes
    );
    assert_eq!(
        sessions.get_kv_cache(child.id).unwrap().unwrap().bytes,
        child_bytes
    );
    let ticket = sessions.kv_restore_ticket(child.id).unwrap().unwrap();
    assert_eq!(ticket.total_bytes, child_bytes.len() as u64);
    assert!(ticket.extents.iter().all(|extent| {
        extent.file_offset % 4096 == 0
            && extent.transfer_length % 4096 == 0
            && extent.file_offset >= KV_HEADER_SIZE as u64
    }));
}

#[test]
fn committed_kv_corruption_is_detected() {
    let directory = tempdir().unwrap();
    let branch = uuid::Uuid::new_v4();
    let spec = spec(1);
    {
        let store = KvCacheStore::open(directory.path()).unwrap();
        store.put(branch, spec, &payload(spec, 4)).unwrap();
    }
    let path = directory.path().join("kv-cache.dat");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    file.seek(SeekFrom::Start(KV_HEADER_SIZE as u64)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Start(KV_HEADER_SIZE as u64)).unwrap();
    file.write_all(&[byte[0] ^ 0xff]).unwrap();
    file.sync_data().unwrap();
    drop(file);
    assert!(KvCacheStore::open(directory.path()).is_err());
}

#[test]
fn cpu_only_capability_probe_never_claims_verified_gds() {
    let capabilities = HardwareCapabilities::detect();
    if !capabilities.nvidia_driver
        || !capabilities.nvidia_fs
        || capabilities.libcufile_path.is_none()
    {
        assert_eq!(capabilities.transfer_path, TransferPath::CpuDirect);
        assert!(!capabilities.supports_verified_gds());
    }
    assert!(capabilities.to_json().unwrap().contains("transfer_path"));
}

#[test]
fn restored_cache_produces_bit_identical_attention_output() {
    let directory = tempdir().unwrap();
    let branch = uuid::Uuid::new_v4();
    let spec = spec(1);
    let keys = [[0.25_f32, -0.5, 0.75, 1.0], [1.0, 0.5, -0.25, 0.125]];
    let values = [[1.0_f32, 2.0, 3.0, 4.0], [-1.0, -2.0, 0.5, 0.25]];
    let mut bytes = Vec::new();
    for tensor in [keys, values] {
        for row in tensor {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    assert_eq!(bytes.len(), spec.expected_bytes().unwrap());
    let query = [0.5_f32, -0.25, 1.0, 0.75];
    let expected = attention(query, keys, values);
    let store = KvCacheStore::open(directory.path()).unwrap();
    store.put(branch, spec, &bytes).unwrap();
    let restored = store.get(branch).unwrap().unwrap();
    let floats: Vec<f32> = restored
        .bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect();
    let restored_keys = [
        floats[0..4].try_into().unwrap(),
        floats[4..8].try_into().unwrap(),
    ];
    let restored_values = [
        floats[8..12].try_into().unwrap(),
        floats[12..16].try_into().unwrap(),
    ];
    let actual = attention(query, restored_keys, restored_values);
    assert_eq!(
        actual.map(f32::to_bits),
        expected.map(f32::to_bits),
        "software attention logits must be bit-identical after restore"
    );
}

fn attention(query: [f32; 4], keys: [[f32; 4]; 2], values: [[f32; 4]; 2]) -> [f32; 4] {
    let mut scores = [0_f32; 2];
    for row in 0..2 {
        scores[row] = (0..4).map(|column| query[column] * keys[row][column]).sum();
    }
    let maximum = scores[0].max(scores[1]);
    let weights = [(scores[0] - maximum).exp(), (scores[1] - maximum).exp()];
    let denominator = weights[0] + weights[1];
    std::array::from_fn(|column| {
        (weights[0] * values[0][column] + weights[1] * values[1][column]) / denominator
    })
}
