use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use engramdb::{
    Engine, Error, FaultPoint, FusedNode, GraphEdge, Hash, TemporalPoint, TriModalQuery,
};
use tempfile::tempdir;

fn node(key: &str, vector: [f32; 2], edges: Vec<GraphEdge>) -> FusedNode {
    FusedNode::new(
        key,
        vec![TemporalPoint {
            assertion_time: 10,
            valid_time: 20,
        }],
        vector.to_vec(),
        edges,
    )
    .unwrap()
}

#[test]
fn hybrid_snapshot_is_durable_and_fork_isolated() {
    let directory = tempdir().unwrap();
    let main;
    let child;
    let alpha = Hash(*blake3::hash(b"alpha").as_bytes());
    {
        let engine = Engine::open(directory.path()).unwrap();
        main = engine.main_branch().id;
        let mut transaction = engine.begin(main).unwrap();
        transaction
            .put_fused(node("alpha", [1.0, 0.0], Vec::new()))
            .unwrap();
        let committed = transaction.commit().unwrap();
        assert!(committed.hybrid_root_hash.is_some());
        child = engine.fork(main).unwrap().id;

        let mut child_write = engine.begin(child).unwrap();
        child_write
            .put_fused(node(
                "beta",
                [0.9, 0.1],
                vec![GraphEdge {
                    target: alpha,
                    weight: 1.0,
                    edge_type: 7,
                }],
            ))
            .unwrap();
        child_write.commit().unwrap();

        assert_eq!(engine.hybrid_node_count(main).unwrap(), 1);
        assert_eq!(engine.hybrid_node_count(child).unwrap(), 2);
        assert_eq!(
            engine.hybrid_nearest(main, &[0.9, 0.1], 2).unwrap().len(),
            1
        );
        assert_eq!(
            engine.hybrid_nearest(child, &[0.9, 0.1], 2).unwrap().len(),
            2
        );
    }

    let recovered = Engine::open(directory.path()).unwrap();
    assert_eq!(recovered.hybrid_node_count(main).unwrap(), 1);
    assert_eq!(recovered.hybrid_node_count(child).unwrap(), 2);
    assert_ne!(
        recovered.branch(main).unwrap().hybrid_root_hash,
        recovered.branch(child).unwrap().hybrid_root_hash
    );
    let matches = recovered
        .hybrid_query(
            child,
            alpha,
            TriModalQuery {
                vector: &[1.0, 0.0],
                minimum_cosine: 0.8,
                assertion_before: 100,
                valid_at: 20,
                max_hops: 1,
                edge_type: Some(7),
            },
        )
        .unwrap();
    assert_eq!(matches.len(), 1);
    recovered.validate().unwrap();
}

#[test]
fn hybrid_fault_boundaries_restore_all_committed_watermarks() {
    for fault in [
        FaultPoint::AfterPageWrites,
        FaultPoint::AfterDataSync,
        FaultPoint::DuringMetadataAppend,
    ] {
        let directory = tempdir().unwrap();
        let main;
        let stable_fused_length;
        let stable_hnsw_length;
        {
            let engine = Engine::open(directory.path()).unwrap();
            main = engine.main_branch().id;
            let mut stable = engine.begin(main).unwrap();
            stable
                .put_fused(node("stable", [1.0, 0.0], Vec::new()))
                .unwrap();
            stable.commit().unwrap();
            stable_fused_length = std::fs::metadata(directory.path().join("fused.dat"))
                .unwrap()
                .len();
            stable_hnsw_length = std::fs::metadata(directory.path().join("hnsw.dat"))
                .unwrap()
                .len();

            let mut interrupted = engine.begin(main).unwrap();
            interrupted
                .put_fused(node("orphan", [0.0, 1.0], Vec::new()))
                .unwrap();
            assert!(interrupted.commit_with_fault(fault).is_err());
        }

        let recovered = Engine::open(directory.path()).unwrap();
        assert_eq!(recovered.hybrid_node_count(main).unwrap(), 1);
        assert_eq!(
            std::fs::metadata(directory.path().join("fused.dat"))
                .unwrap()
                .len(),
            stable_fused_length
        );
        assert_eq!(
            std::fs::metadata(directory.path().join("hnsw.dat"))
                .unwrap()
                .len(),
            stable_hnsw_length
        );
        let nearest = recovered.hybrid_nearest(main, &[0.0, 1.0], 2).unwrap();
        assert_eq!(nearest.len(), 1);
        assert_eq!(nearest[0].id, Hash(*blake3::hash(b"stable").as_bytes()));
    }
}

#[test]
fn committed_fused_corruption_is_reported() {
    let directory = tempdir().unwrap();
    {
        let engine = Engine::open(directory.path()).unwrap();
        let main = engine.main_branch().id;
        let mut transaction = engine.begin(main).unwrap();
        transaction
            .put_fused(node("durable", [1.0, 0.0], Vec::new()))
            .unwrap();
        transaction.commit().unwrap();
    }
    let path = directory.path().join("fused.dat");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    file.seek(SeekFrom::Start(4096)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Start(4096)).unwrap();
    file.write_all(&[byte[0] ^ 0xff]).unwrap();
    file.sync_data().unwrap();
    drop(file);

    assert!(matches!(
        Engine::open(directory.path()),
        Err(Error::CorruptPage { .. })
    ));
}

#[test]
fn committed_hnsw_corruption_is_reported() {
    let directory = tempdir().unwrap();
    {
        let engine = Engine::open(directory.path()).unwrap();
        let main = engine.main_branch().id;
        let mut transaction = engine.begin(main).unwrap();
        transaction
            .put_fused(node("durable", [1.0, 0.0], Vec::new()))
            .unwrap();
        transaction.commit().unwrap();
    }
    let path = directory.path().join("hnsw.dat");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let length = file.metadata().unwrap().len();
    file.seek(SeekFrom::Start(length - 1)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Start(length - 1)).unwrap();
    file.write_all(&[byte[0] ^ 0xff]).unwrap();
    file.sync_data().unwrap();
    drop(file);

    assert!(matches!(
        Engine::open(directory.path()),
        Err(Error::CorruptMetadata { .. })
    ));
}

#[test]
fn hybrid_merge_rejects_divergent_index_roots() {
    let directory = tempdir().unwrap();
    let engine = Engine::open(directory.path()).unwrap();
    let main = engine.main_branch().id;
    let child = engine.fork(main).unwrap().id;
    let mut transaction = engine.begin(child).unwrap();
    transaction
        .put_fused(node("child-only", [1.0, 0.0], Vec::new()))
        .unwrap();
    transaction.commit().unwrap();
    assert!(matches!(
        engine.merge(main, child),
        Err(Error::Invariant(reason)) if reason.contains("hybrid merge")
    ));
}
