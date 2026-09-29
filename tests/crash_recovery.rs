use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use engramdb::{Engine, Error, FaultPoint, TemporalRecord};
use tempfile::tempdir;

#[test]
fn every_transaction_fault_boundary_recovers_last_committed_root() {
    for fault in [
        FaultPoint::AfterPageWrites,
        FaultPoint::AfterDataSync,
        FaultPoint::DuringMetadataAppend,
    ] {
        let directory = tempdir().unwrap();
        let main;
        let stable_hash;
        {
            let engine = Engine::open(directory.path()).unwrap();
            main = engine.main_branch().id;
            let mut stable = engine.begin(main).unwrap();
            stable
                .put(TemporalRecord::new("stable", "present", 0, 10).unwrap())
                .unwrap();
            stable_hash = stable.commit().unwrap().root_hash;

            let mut interrupted = engine.begin(main).unwrap();
            interrupted
                .put(TemporalRecord::new("uncommitted", "absent", 0, 10).unwrap())
                .unwrap();
            assert!(interrupted.commit_with_fault(fault).is_err());
            // Dropping the engine also drops the submission queue, matching a
            // process/power-loss boundary after the injected step.
        }

        let recovered = Engine::open(directory.path()).unwrap();
        assert_eq!(recovered.main_branch().root_hash, stable_hash);
        assert!(recovered.get(main, b"uncommitted", 5).unwrap().is_none());
        assert_eq!(
            recovered.get(main, b"stable", 5).unwrap().unwrap().value,
            b"present"
        );
        recovered.validate().unwrap();
    }
}

#[test]
fn randomized_fault_cycles_never_publish_partial_state() {
    // Deterministic xorshift drives 96 independent crash points so failures are
    // reproducible while covering all transaction boundaries repeatedly.
    let mut random = 0x4d595df4d0f33173_u64;
    for cycle in 0..96 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let fault = match random % 3 {
            0 => FaultPoint::AfterPageWrites,
            1 => FaultPoint::AfterDataSync,
            _ => FaultPoint::DuringMetadataAppend,
        };

        let directory = tempdir().unwrap();
        let main;
        {
            let engine = Engine::open(directory.path()).unwrap();
            main = engine.main_branch().id;
            let mut transaction = engine.begin(main).unwrap();
            transaction
                .put(
                    TemporalRecord::new(format!("cycle-{cycle}"), vec![cycle as u8; 128], 0, 100)
                        .unwrap(),
                )
                .unwrap();
            assert!(transaction.commit_with_fault(fault).is_err());
        }
        let recovered = Engine::open(directory.path()).unwrap();
        assert!(recovered
            .get(main, format!("cycle-{cycle}").as_bytes(), 50)
            .unwrap()
            .is_none());
        recovered.validate().unwrap();
    }
}

#[test]
fn torn_trailing_data_page_is_discarded() {
    let directory = tempdir().unwrap();
    let main;
    {
        let engine = Engine::open(directory.path()).unwrap();
        main = engine.main_branch().id;
        let mut transaction = engine.begin(main).unwrap();
        transaction
            .put(TemporalRecord::new("durable", "yes", 0, 10).unwrap())
            .unwrap();
        transaction.commit().unwrap();
    }
    let mut pages = OpenOptions::new()
        .append(true)
        .open(directory.path().join("pages.dat"))
        .unwrap();
    pages.write_all(&vec![0x5a; 777]).unwrap();
    pages.sync_data().unwrap();
    drop(pages);

    let recovered = Engine::open(directory.path()).unwrap();
    assert_eq!(
        recovered.get(main, b"durable", 1).unwrap().unwrap().value,
        b"yes"
    );
    recovered.validate().unwrap();
}

#[test]
fn committed_metadata_corruption_is_reported_not_silently_rolled_back() {
    let directory = tempdir().unwrap();
    {
        let engine = Engine::open(directory.path()).unwrap();
        let main = engine.main_branch().id;
        let mut transaction = engine.begin(main).unwrap();
        transaction
            .put(TemporalRecord::new("durable", "yes", 0, 10).unwrap())
            .unwrap();
        transaction.commit().unwrap();
    }
    let path = directory.path().join("branches.log");
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
