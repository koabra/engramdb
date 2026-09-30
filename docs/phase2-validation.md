# Phase 2 validation report

## Result

Phase 2's prototype correctness and recall gates pass on commit
`2e9f0df8af49f6a8f98e23382c7950e46f6c61ca`. The implementation closes the
Phase 1 page-size boundary with layout-aware 4 KiB/64 KiB direct I/O, stores
tri-modal nodes in packed 64 KiB blocks, and provides quantized HNSW search plus
filtered graph traversal.

Phase 3 readiness is **hold**, not a production-readiness claim. The fused
layout and zero-copy node views are usable foundations, but durable HNSW
metadata, branch/snapshot integration, and hardware-counter evidence remain
Phase 2 blockers. No Phase 3 code is included.

The complete run is reproducible with:

```bash
SIFT1M_DIR=/path/to/sift1m ./scripts/validate_phase2.sh
```

Raw CSVs, system details, command provenance, and dependency-free SVG plots are
committed under [`metrics/phase2/`](../metrics/phase2/).

## Validation matrix

| Area | Executed validation | Result |
| --- | --- | --- |
| Static gates | `cargo fmt --check`; Clippy for all targets with warnings denied | Pass |
| Rust tests | 30 normal tests across Phase 1 storage/recovery/branching and Phase 2 layout/search/traversal | Pass |
| Phase 1 long haul | Release-mode 1,000 randomized durable forks and merges | Pass |
| 64 KiB readiness | Independent block geometry, aligned append/read, accounting, torn-tail truncation | Pass |
| Fused layout | Packed directory/temporal/vector/CSR round trip; CRC corruption rejection; no FP32 read materialization | Pass |
| Vector kernels | Dispatched INT8 dot product versus scalar oracle; cosine and squared-L2 ranking tests | Pass |
| Recall | SIFT1M, 1,000,000 × 128d base vectors, 200 official queries, Recall@10 > 0.95 | Pass: 0.981 mean |
| Multi-modal correctness | Three-hop semantic + temporal + edge-type traversal versus a brute-force oracle | Pass |
| Tri-modal latency | 10,000 deterministic three-hop queries on 2,000 nodes | Pass; engineering measurement below |
| Storage footprint | Physically materialized 10,000,000 nodes, 768d INT8, one temporal point, 10 edges | Pass |
| Cache behavior | Wall-clock fused/pointer proxy plus requested hardware-tool detection | Inconclusive; fused proxy slower and hardware counters unavailable |
| External products | Runnable pgvector recursive-CTE and Neo4j + Qdrant comparison drivers | Not measured; services unavailable |

The final exact-code regression transcript is
[`final-validation.log`](../metrics/phase2/final-validation.log).

## Standard recall

The run used the Hugging Face SIFT1M mirror and its official FP32 squared-L2
ground truth. File SHA-256 values and the exact command are recorded in
[`recall-run.txt`](../metrics/phase2/recall-run.txt). EngramDB used per-vector
INT8 scalar quantization, squared-L2 HNSW construction, `M=32`,
`ef_construction=256`, and `ef_search=4096`.

| Queries | Mean Recall@10 | Median | Minimum | Perfect queries | Search p50 | p90 | p99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.981 | 1.0 | 0.9 | 162 | 51.75 ms | 76.14 ms | 116.98 ms |

The recall gate passes, but the search breadth required for it is expensive.
This is a correctness/quality result, not evidence of production ANN latency.

![SIFT1M recall](../metrics/phase2/recall.svg)

## Tri-modal query latency

The workload traverses three hops from a randomized root and simultaneously
requires cosine similarity greater than 0.8, `assertion_time < 75`,
`valid_time <= 400`, and edge type 1. The index contained 2,000 deterministic
128d nodes with three outbound edges each.

| Samples | p50 | p90 | p95 | p99 | Maximum |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 2.70 µs | 5.04 µs | 6.59 µs | 8.70 µs | 18.37 µs |

These in-process, cache-warm measurements exclude network, parsing, and branch
transaction costs. They validate the fused traversal path, not the Phase 3
execution engine.

![Tri-modal latency](../metrics/phase2/query-latency.svg)

## Ten-million-node storage result

This was a real 12.60 GB file materialization, not an extrapolation. Each node
contained a 768d vector, one temporal point, and ten 40-byte graph edges.

| Nodes | Blocks | FP32 source bytes | Encoded logical bytes | Physical bytes | Physical/source | Block packing |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000,000 | 192,308 | 35,200,000,000 | 12,480,000,000 | 12,603,097,088 | 0.358× | 1.0099× |

INT8 quantization accounts for the reduction relative to FP32 source data.
The final 0.99% overhead is 64 KiB block slack and per-node directory metadata.
The measured write completed in 177.03 seconds on the validation VM.

![Storage footprint](../metrics/phase2/amplification.svg)

## Cache-layout evidence

`perf` and Valgrind/Cachegrind were unavailable, so no L2/L3 miss counts are
reported. The dependency-free wall-clock proxy scanned 128 × 256d vectors for
10,000 iterations:

| Layout | Elapsed | Relative to pointer baseline |
| --- | ---: | ---: |
| Fused block views | 24.28 ms | 1.474× slower |
| Boxed pointer vectors | 16.47 ms | baseline |

The proxy does **not** validate the document's cache-miss reduction claim.
Directory-view validation overhead and the allocator's compact placement of the
small pointer fixture affect this result. Hardware-counter profiling on a
controlled host remains mandatory.

![Cache-layout proxy](../metrics/phase2/cache-layout.svg)

## Product comparison

Only EngramDB was measured. PostgreSQL with pgvector, Neo4j, Qdrant, Docker,
`psql`, and product service endpoints were unavailable; numeric competitor
results would therefore be fabricated.

[`scripts/compare_phase2.py`](../scripts/compare_phase2.py) provisions the same
2,000-node deterministic corpus and emits the same latency schema for:

- PostgreSQL: pgvector cosine operator plus a three-hop recursive CTE.
- Neo4j + Qdrant: semantic filtering in Qdrant followed by a variable-length
  Neo4j traversal and ID intersection.

`validate_phase2.sh` appends those observations only when the corresponding
connection environment variables are present. The claim of 10–100× lower
latency is **not validated** by this environment.

## Phase 1 regressions

All Phase 1 tests, Loom checks, crash/corruption cases, and the release-mode
1,000-operation long-haul property passed. The complete Phase 1 benchmark was
also rerun under [`metrics/phase2/phase1-regression/`](../metrics/phase2/phase1-regression/).

The observed micro-write rate was 98.4 commits/s with 4.091× amplification.
That performance sample followed the 12.60 GB materialization on an overlay
filesystem and is not directly comparable to the earlier clean Phase 1 run.
It is retained as raw evidence, not presented as a performance regression
conclusion.

## Limitations and Phase 3 readiness

- Fused blocks use a separate `fused.dat`-style file and layout-aware direct
  I/O, preserving Phase 1's 4 KiB on-disk format. They are not yet committed
  through `Engine` branch metadata or snapshot roots.
- The HNSW graph is deterministic but memory-resident and rebuilt on open.
  Its adjacency is not checkpointed, content-addressed, or crash-committed.
- Full fused blocks are CRC-protected and partial direct-I/O tails are removed,
  but there is no metadata watermark that distinguishes a synced committed
  batch from a full orphan block after a crash.
- Quantization is INT8 scalar quantization. FP8 and product quantization are not
  implemented.
- x86 uses AVX2 rather than an AVX-512-specialized kernel. ARM NEON covers the
  INT8 dot product; squared-L2 falls back to scalar code on ARM.
- Packed `#[repr(C, packed)]` format definitions are encoded manually into
  aligned direct-I/O buffers. The implementation does not use `mmap`.
- Recall passes only at a high search breadth; SIFT1M p50 ANN latency is
  51.75 ms. Construction/open-time rebuild cost is not yet acceptable for a
  production million-node service.
- Hardware cache-miss evidence and external product measurements are absent.
- Phase 1 limitations remain: serialized io_uring completion, no garbage
  collection, and no real-NVMe process-kill/device-fault campaign.

Phase 3 prototype readiness is therefore **conditional no / hold**. Before
starting Phase 3, Phase 2 should add a durable content-addressed HNSW root to
branch metadata, define fused-block transactional watermarks, and obtain
hardware-counter evidence (or change the layout based on it). This branch does
not begin Phase 3.
