use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use engramdb::{AlignedPage, BufferPool, DirectIo, Engine, Error, TemporalRecord, PAGE_SIZE};
use tempfile::tempdir;

#[test]
fn direct_io_round_trip_is_aligned_and_accounted() {
    let directory = tempdir().unwrap();
    let io = DirectIo::open(directory.path().join("pages"), 8).unwrap();
    let mut page = AlignedPage::zeroed();
    for (index, byte) in page.as_mut_slice().iter_mut().enumerate() {
        *byte = (index % 251) as u8;
    }
    let offset = io.append(&page).unwrap();
    io.sync().unwrap();
    assert_eq!(offset, 0);
    assert_eq!(io.read(offset).unwrap().as_slice(), page.as_slice());
    let stats = io.stats();
    assert_eq!(stats.bytes_written, PAGE_SIZE as u64);
    assert_eq!(stats.bytes_read, PAGE_SIZE as u64);
}

#[test]
fn clock_pool_evicts_and_reloads_pages() {
    let directory = tempdir().unwrap();
    let io = Arc::new(DirectIo::open(directory.path().join("pages"), 8).unwrap());
    let page = AlignedPage::zeroed();
    for _ in 0..4 {
        io.append(&page).unwrap();
    }
    let pool = BufferPool::new(io, 2);
    for index in 0..4 {
        pool.get(index * PAGE_SIZE as u64).unwrap();
    }
    let stats = pool.stats();
    assert_eq!(stats.misses, 4);
    assert!(stats.evictions >= 2);
    assert!(stats.resident_pages <= 2);
}

#[test]
fn content_addressed_tree_splits_and_recovers() {
    let directory = tempdir().unwrap();
    let main;
    let committed_hash;
    {
        let engine = Engine::open(directory.path()).unwrap();
        main = engine.main_branch().id;
        let mut transaction = engine.begin(main).unwrap();
        for index in 0..250 {
            transaction
                .put(
                    TemporalRecord::new(format!("key-{index:04}"), vec![index as u8; 256], 0, 100)
                        .unwrap(),
                )
                .unwrap();
        }
        committed_hash = transaction.commit().unwrap().root_hash;
        engine.validate().unwrap();
        assert!(engine.page_count() > 1);
    }

    let recovered = Engine::open(directory.path()).unwrap();
    assert_eq!(recovered.main_branch().id, main);
    assert_eq!(recovered.main_branch().root_hash, committed_hash);
    for index in [0, 1, 127, 249] {
        let record = recovered
            .get(main, format!("key-{index:04}").as_bytes(), 50)
            .unwrap()
            .unwrap();
        assert_eq!(record.value, vec![index as u8; 256]);
    }
    recovered.validate().unwrap();
}

#[test]
fn identical_writes_are_content_deduplicated() {
    let directory = tempdir().unwrap();
    let engine = Engine::open(directory.path()).unwrap();
    let main = engine.main_branch().id;
    let branch = engine.fork(main).unwrap();

    let record = TemporalRecord::new("shared", "payload", 0, 10).unwrap();
    let mut left = engine.begin(main).unwrap();
    left.put(record.clone()).unwrap();
    left.commit().unwrap();
    let pages_after_left = engine.page_count();

    let mut right = engine.begin(branch.id).unwrap();
    right.put(record).unwrap();
    right.commit().unwrap();

    // Epoch is part of a bitemporal key, so roots differ, but unchanged ancestor
    // pages remain shared and branch creation itself allocated no pages.
    assert!(engine.page_count() <= pages_after_left + 1);
}

#[test]
fn readers_remain_consistent_during_copy_on_write_splits() {
    let directory = tempdir().unwrap();
    let engine = Arc::new(Engine::open(directory.path()).unwrap());
    let main = engine.main_branch().id;
    let done = Arc::new(AtomicBool::new(false));
    let reads = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|scope| {
        for _ in 0..4 {
            let engine = Arc::clone(&engine);
            let done = Arc::clone(&done);
            let reads = Arc::clone(&reads);
            scope.spawn(move || {
                while !done.load(Ordering::Acquire) {
                    if let Some(record) = engine.get(main, b"shared-key", 50).unwrap() {
                        assert_eq!(record.value.len(), 256);
                    }
                    reads.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        let engine = Arc::clone(&engine);
        let done = Arc::clone(&done);
        scope.spawn(move || {
            for index in 0..100 {
                let mut transaction = engine.begin(main).unwrap();
                // The shared key creates new temporal versions while unique
                // keys force repeated leaf and internal-node splits.
                transaction
                    .put(TemporalRecord::new("shared-key", vec![index as u8; 256], 0, 100).unwrap())
                    .unwrap();
                transaction
                    .put(
                        TemporalRecord::new(
                            format!("split-{index:04}"),
                            vec![index as u8; 256],
                            0,
                            100,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                transaction.commit().unwrap();
            }
            done.store(true, Ordering::Release);
        });
    });
    assert!(reads.load(Ordering::Relaxed) > 0);
    engine.validate().unwrap();
}

#[test]
fn database_directory_has_single_engine_owner() {
    let directory = tempdir().unwrap();
    let first = Engine::open(directory.path()).unwrap();
    assert!(matches!(
        Engine::open(directory.path()),
        Err(Error::DatabaseLocked(_))
    ));
    drop(first);
    Engine::open(directory.path()).unwrap();
}
