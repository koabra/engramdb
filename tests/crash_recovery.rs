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

#[test]
fn committed_page_corruption_is_reported() {
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
    let path = directory.path().join("pages.dat");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let length = file.metadata().unwrap().len();
    let payload_offset = length - engramdb::PAGE_SIZE as u64 + 64;
    file.seek(SeekFrom::Start(payload_offset)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Start(payload_offset)).unwrap();
    file.write_all(&[byte[0] ^ 0xff]).unwrap();
    file.sync_data().unwrap();
    drop(file);

    assert!(matches!(
        Engine::open(directory.path()),
        Err(Error::CorruptPage { .. })
    ));
}

#[test]
fn partial_metadata_write_poisons_engine_until_reopen() {
    let directory = tempdir().unwrap();
    let engine = Engine::open(directory.path()).unwrap();
    let main = engine.main_branch().id;
    let mut transaction = engine.begin(main).unwrap();
    transaction
        .put(TemporalRecord::new("uncertain", "value", 0, 10).unwrap())
        .unwrap();
    assert!(transaction
        .commit_with_fault(FaultPoint::DuringMetadataAppend)
        .is_err());
    assert!(matches!(engine.fork(main), Err(Error::MetadataPoisoned)));
    drop(engine);

    let recovered = Engine::open(directory.path()).unwrap();
    assert!(recovered.get(main, b"uncertain", 1).unwrap().is_none());
}

#[test]
fn corrupted_metadata_length_is_not_mistaken_for_partial_tail() {
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
    let mut first_header = [0_u8; 20];
    file.read_exact(&mut first_header).unwrap();
    let first_length = u32::from_le_bytes(first_header[8..12].try_into().unwrap()) as u64;
    let second_length_offset = 20 + first_length + 8;
    file.seek(SeekFrom::Start(second_length_offset)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Start(second_length_offset)).unwrap();
    file.write_all(&[byte[0] ^ 0x80]).unwrap();
    file.sync_data().unwrap();
    drop(file);

    assert!(matches!(
        Engine::open(directory.path()),
        Err(Error::CorruptMetadata { .. })
    ));
}

#[test]
fn committed_watermark_discards_nonfinal_torn_orphan_pages() {
    let directory = tempdir().unwrap();
    let main;
    let committed_length;
    {
        let engine = Engine::open(directory.path()).unwrap();
        main = engine.main_branch().id;
        let mut stable = engine.begin(main).unwrap();
        stable
            .put(TemporalRecord::new("stable", "yes", 0, 10).unwrap())
            .unwrap();
        stable.commit().unwrap();
        committed_length = std::fs::metadata(directory.path().join("pages.dat"))
            .unwrap()
            .len();

        let mut interrupted = engine.begin(main).unwrap();
        for index in 0..50 {
            interrupted
                .put(
                    TemporalRecord::new(
                        format!("orphan-{index:04}"),
                        vec![index as u8; 256],
                        0,
                        10,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        assert!(interrupted
            .commit_with_fault(FaultPoint::AfterDataSync)
            .is_err());
    }
    let pages_path = directory.path().join("pages.dat");
    let uncommitted_length = std::fs::metadata(&pages_path).unwrap().len();
    assert!(uncommitted_length >= committed_length + 2 * engramdb::PAGE_SIZE as u64);

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&pages_path)
        .unwrap();
    file.seek(SeekFrom::Start(committed_length + 64)).unwrap();
    file.write_all(&[0xff]).unwrap();
    file.sync_data().unwrap();
    drop(file);

    let recovered = Engine::open(directory.path()).unwrap();
    assert_eq!(
        std::fs::metadata(pages_path).unwrap().len(),
        committed_length
    );
    assert_eq!(
        recovered.get(main, b"stable", 1).unwrap().unwrap().value,
        b"yes"
    );
    recovered.validate().unwrap();
}
