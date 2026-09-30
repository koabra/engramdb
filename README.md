# EngramDB

EngramDB Phase 2 is a Linux-only prototype that combines the Phase 1
content-addressed storage foundation with 64 KiB fused vector, graph, and
temporal blocks. Query execution remains reserved for Phase 3.

## Storage model

- 4 KiB aligned pages use Linux `O_DIRECT` and are submitted through
  `io_uring`.
- B+ tree nodes are deterministically encoded, protected by CRC32, and
  addressed by their BLAKE3 digest.
- Inserts write only the changed root-to-leaf path. A branch fork stores the
  existing root reference in the append-only metadata log and copies no pages.
- Data pages are synced before a commit root is appended and synced. Recovery
  discards incomplete tails, rejects corruption in durable records, and walks
  every committed DAG.
- The user-space CLOCK cache exposes immutable `Arc<Page>` values through
  hazard-pointer-protected atomic loads.

## Hybrid index

- Phase 1 tree pages remain 4 KiB; layout-aware direct I/O also supports a
  separate 64 KiB fused-block file without changing the Phase 1 format.
- Each fused block tightly packs a 64-byte header, node directory, temporal
  points, INT8 vectors, and CSR-style outbound edges. CRC32 protects the full
  payload.
- `HybridIndex` supports cosine or squared-L2 HNSW search and filtered,
  multi-hop graph traversal. AVX2 and NEON kernels scan quantized vectors
  directly from aligned block memory without constructing `Vec<f32>` values.
- Fused blocks and content-addressed HNSW checkpoints are published atomically
  through branch metadata. Forks share an immutable checkpoint root in O(1),
  and recovery truncates fused/checkpoint records beyond committed watermarks.

## Minimal use

```rust
use engramdb::{Engine, TemporalRecord};

let engine = Engine::open("/var/lib/engramdb")?;
let main = engine.main_branch();
let child = engine.fork(main.id)?;

let mut transaction = engine.begin(child.id)?;
transaction.put(TemporalRecord::new("agent:7", "ready", 0, 100)?)?;
transaction.commit()?;

let record = engine.get(child.id, b"agent:7", 50)?;
# Ok::<(), engramdb::Error>(())
```

`Engine::get_as_of` supports assertion-time reads. `Engine::merge` performs a
three-way merge for non-overlapping temporal ranges and rejects overlapping,
different updates.

## Validation

Run the complete Phase 1 suite and regenerate committed metrics:

```bash
./scripts/validate_phase1.sh
```

The script runs formatting and Clippy gates, all normal tests, a release-mode
1,000-fork/1,000-merge long-haul test, the requested `1/10/100/1000` branch
latency workload, and a 5,000-logical-worker 1 KiB write workload. Results and
machine details are written to `metrics/phase1/`.

PostgreSQL and MongoDB comparisons require separately provisioned products:

```bash
python -m pip install asyncpg pymongo
python scripts/compare_products.py postgresql \
  --output metrics/phase1/postgresql-branch-latency.csv
python scripts/compare_products.py mongodb \
  --output metrics/phase1/mongodb-branch-latency.csv
```

These results are kept separate from local EngramDB measurements so unavailable
products never produce synthetic or inferred numbers.

Run Phase 2 validation (including Phase 1 regressions):

```bash
SIFT1M_DIR=/path/to/sift1m ./scripts/validate_phase2.sh
```

The directory must contain `sift_base.fvecs`, `sift_query.fvecs`, and
`sift_groundtruth.ivecs`. Without it, the script runs the same Recall@10
pipeline on a deterministic synthetic corpus. Raw measurements and plots are
under [`metrics/phase2/`](metrics/phase2/); interpretation and limitations are
in [`docs/phase2-validation.md`](docs/phase2-validation.md).
