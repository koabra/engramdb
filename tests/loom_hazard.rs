use std::sync::Arc;

use engramdb::{HazardAtomic, HazardDomain};

#[test]
fn hazard_atomic_survives_concurrent_replacement() {
    let domain = HazardDomain::new();
    let atomic = Arc::new(HazardAtomic::new(Arc::new(0_u64), domain.clone()));
    let mut threads = Vec::new();
    for reader_id in 0..4 {
        let atomic = Arc::clone(&atomic);
        threads.push(std::thread::spawn(move || {
            for _ in 0..10_000 {
                let value = *atomic.load();
                assert!(value <= 10_000);
            }
            reader_id
        }));
    }
    for value in 1..=10_000 {
        atomic.store(Arc::new(value));
    }
    for thread in threads {
        thread.join().unwrap();
    }
    domain.collect();
    assert_eq!(*atomic.load(), 10_000);
}

#[test]
fn loom_exhaustively_checks_publish_validate_reclaim_interleavings() {
    loom::model(|| {
        use loom::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
        use loom::sync::Arc as LoomArc;
        use loom::thread;

        // Minimal state-machine model of HazardAtomic::load/store. IDs stand in
        // for raw Arc pointers; `reclaimed_old` catches a use-after-free.
        let owner = LoomArc::new(AtomicUsize::new(1));
        let hazard = LoomArc::new(AtomicUsize::new(0));
        let reclaimed_old = LoomArc::new(AtomicBool::new(false));

        let reader = {
            let owner = LoomArc::clone(&owner);
            let hazard = LoomArc::clone(&hazard);
            let reclaimed_old = LoomArc::clone(&reclaimed_old);
            thread::spawn(move || {
                let candidate = owner.load(Ordering::SeqCst);
                hazard.store(candidate, Ordering::SeqCst);
                fence(Ordering::SeqCst);
                if owner.load(Ordering::SeqCst) == candidate && candidate == 1 {
                    assert!(!reclaimed_old.load(Ordering::SeqCst));
                }
                hazard.store(0, Ordering::SeqCst);
            })
        };

        let writer = {
            let owner = LoomArc::clone(&owner);
            let hazard = LoomArc::clone(&hazard);
            let reclaimed_old = LoomArc::clone(&reclaimed_old);
            thread::spawn(move || {
                let old = owner.swap(2, Ordering::SeqCst);
                fence(Ordering::SeqCst);
                if hazard.load(Ordering::SeqCst) != old {
                    reclaimed_old.store(true, Ordering::SeqCst);
                }
            })
        };

        reader.join().unwrap();
        writer.join().unwrap();
    });
}
