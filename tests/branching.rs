use engramdb::{Engine, Error, TemporalRecord};
use tempfile::tempdir;

#[test]
fn fork_is_constant_storage_and_isolated() {
    let directory = tempdir().unwrap();
    let engine = Engine::open(directory.path()).unwrap();
    let main = engine.main_branch().id;
    let mut transaction = engine.begin(main).unwrap();
    transaction
        .put(TemporalRecord::new("entity", "base", 0, 100).unwrap())
        .unwrap();
    transaction.commit().unwrap();

    let pages_before = engine.page_count();
    let child = engine.fork(main).unwrap();
    assert_eq!(engine.page_count(), pages_before);
    let mut transaction = engine.begin(child.id).unwrap();
    transaction
        .put(TemporalRecord::new("entity", "child", 0, 100).unwrap())
        .unwrap();
    transaction.commit().unwrap();

    assert_eq!(
        engine.get(main, b"entity", 50).unwrap().unwrap().value,
        b"base"
    );
    assert_eq!(
        engine.get(child.id, b"entity", 50).unwrap().unwrap().value,
        b"child"
    );
}

#[test]
fn bitemporal_as_of_reads_return_prior_assertion() {
    let directory = tempdir().unwrap();
    let engine = Engine::open(directory.path()).unwrap();
    let main = engine.main_branch().id;

    let mut first = engine.begin(main).unwrap();
    first
        .put(TemporalRecord::new("price", "10", 0, 100).unwrap())
        .unwrap();
    let first_epoch = first.commit().unwrap().epoch;

    let mut second = engine.begin(main).unwrap();
    second
        .put(TemporalRecord::new("price", "12", 0, 100).unwrap())
        .unwrap();
    second.commit().unwrap();

    assert_eq!(
        engine
            .get_as_of(main, b"price", 10, first_epoch)
            .unwrap()
            .unwrap()
            .value,
        b"10"
    );
    assert_eq!(
        engine.get(main, b"price", 10).unwrap().unwrap().value,
        b"12"
    );
}

#[test]
fn stale_transactions_are_rejected() {
    let directory = tempdir().unwrap();
    let engine = Engine::open(directory.path()).unwrap();
    let main = engine.main_branch().id;
    let mut first = engine.begin(main).unwrap();
    let mut stale = engine.begin(main).unwrap();
    first
        .put(TemporalRecord::new("a", "first", 0, 1).unwrap())
        .unwrap();
    stale
        .put(TemporalRecord::new("b", "stale", 0, 1).unwrap())
        .unwrap();
    first.commit().unwrap();
    assert!(matches!(
        stale.commit(),
        Err(Error::StaleTransaction(branch)) if branch == main
    ));
}

#[test]
fn three_way_merge_accepts_non_overlapping_ranges() {
    let directory = tempdir().unwrap();
    let engine = Engine::open(directory.path()).unwrap();
    let main = engine.main_branch().id;
    let source = engine.fork(main).unwrap();

    let mut target_change = engine.begin(main).unwrap();
    target_change
        .put(TemporalRecord::new("status", "past", 0, 10).unwrap())
        .unwrap();
    target_change.commit().unwrap();

    let mut source_change = engine.begin(source.id).unwrap();
    source_change
        .put(TemporalRecord::new("status", "future", 10, 20).unwrap())
        .unwrap();
    source_change.commit().unwrap();

    let outcome = engine.merge(main, source.id).unwrap();
    assert_eq!(outcome.applied_ranges, 1);
    assert_eq!(
        engine.get(main, b"status", 5).unwrap().unwrap().value,
        b"past"
    );
    assert_eq!(
        engine.get(main, b"status", 15).unwrap().unwrap().value,
        b"future"
    );
}

#[test]
fn three_way_merge_rejects_conflicting_ranges() {
    let directory = tempdir().unwrap();
    let engine = Engine::open(directory.path()).unwrap();
    let main = engine.main_branch().id;
    let source = engine.fork(main).unwrap();

    for (branch, value, from, to) in [(main, "left", 0, 20), (source.id, "right", 10, 30)] {
        let mut transaction = engine.begin(branch).unwrap();
        transaction
            .put(TemporalRecord::new("status", value, from, to).unwrap())
            .unwrap();
        transaction.commit().unwrap();
    }

    assert!(matches!(
        engine.merge(main, source.id),
        Err(Error::MergeConflict(1))
    ));
}
