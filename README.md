# EngramDB

EngramDB is a Linux-only storage and hybrid-index foundation. Phase 1 provides
durable copy-on-write branches and bitemporal records. Phase 2 adds durable
64 KiB fused semantic/graph/temporal blocks, INT8 quantization, SIMD distance
kernels, HNSW candidate search, and historical multi-hop traversal. A Phase 3
query execution layer is intentionally not included.

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

- `AlignedBlock<const SIZE>` and `DirectIo<const SIZE>` preserve Phase 1's
  4 KiB format while supporting independently aligned 64 KiB fused files.
- Every fused block has a fixed 64-byte header, checked section offsets, CSR
  graph edges, temporal intervals, per-vector asymmetric INT8 parameters, and
  a header+payload CRC.
- Runtime dispatch uses AVX-512 on supported x86-64 hosts, NEON on AArch64, and
  a scalar fallback.
- `HybridIndex::nearest` performs HNSW search with temporal-version selection.
  `HybridIndex::traverse` combines semantic thresholding, valid/assertion time,
  graph hops, and a caller predicate.
- An atomic manifest publishes only synced fused-block generations; recovery
  truncates unpublished tails and rejects committed corruption.

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

Run Phase 2 validation, including Phase 1 regressions, checksum-pinned SIFT1M,
10,000 tri-modal queries, storage measurement, and Cachegrind:

```bash
./scripts/validate_phase2.sh
```

Results are written to `metrics/phase2/` and interpreted in
[`docs/phase2-validation.md`](docs/phase2-validation.md). The first run
downloads approximately 525 MB of standard SIFT1M files to
`/tmp/engramdb-sift1m`.

External product comparisons require provisioned services:

```bash
python -m pip install asyncpg neo4j qdrant-client
python scripts/compare_phase2_products.py postgresql \
  --output metrics/phase2/postgresql-query-latency.csv
python scripts/compare_phase2_products.py neo4j-qdrant \
  --output metrics/phase2/neo4j-qdrant-query-latency.csv
```
