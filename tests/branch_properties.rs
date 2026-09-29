use engramdb::{Engine, TemporalRecord};
use proptest::prelude::*;
use tempfile::tempdir;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        max_shrink_iters: 2_000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn random_forks_never_leak_into_parents(
        operations in prop::collection::vec((any::<u8>(), any::<u16>()), 32..=32)
    ) {
        let directory = tempdir().unwrap();
        let engine = Engine::open(directory.path()).unwrap();
        let mut branches = vec![engine.main_branch().id];

        for (step, (selector, value)) in operations.into_iter().enumerate() {
            let parent = branches[selector as usize % branches.len()];
            let child = engine.fork(parent).unwrap();
            let key = format!("fork-{step}-{value}");
            let mut transaction = engine.begin(child.id).unwrap();
            transaction.put(
                TemporalRecord::new(key.as_bytes(), value.to_be_bytes(), 0, 100).unwrap()
            ).unwrap();
            transaction.commit().unwrap();

            prop_assert!(engine.get(parent, key.as_bytes(), 50).unwrap().is_none());
            prop_assert_eq!(
                engine.get(child.id, key.as_bytes(), 50).unwrap().unwrap().value,
                value.to_be_bytes()
            );
            branches.push(child.id);
        }
        engine.validate().unwrap();
    }
}

/// Explicit long-haul validation invoked by the Phase 1 validation script.
/// It exercises 1,000 forks and 1,000 three-way merges without growing data,
/// isolating branch-manager/DAG metadata behavior.
#[test]
#[ignore = "run with scripts/validate_phase1.sh"]
fn thousand_randomized_forks_and_merges() {
    let directory = tempdir().unwrap();
    let engine = Engine::open(directory.path()).unwrap();
    let main = engine.main_branch().id;
    let mut random = 0x243f6a8885a308d3_u64;

    for _ in 0..1_000 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let child = engine.fork(main).unwrap();
        // An empty source has no changes. Merge still performs ancestry,
        // three-way diff, durability, and metadata publication.
        let outcome = engine.merge(main, child.id).unwrap();
        assert_eq!(outcome.applied_ranges, 0);
        if random & 15 == 0 {
            engine.validate().unwrap();
        }
    }
    engine.validate().unwrap();
}
