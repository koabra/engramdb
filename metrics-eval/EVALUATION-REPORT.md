# EngramDB hardware evaluation (local RTX 2000 Ada)

**Date:** 2026-09-30  
**Commit evaluated:** `f71a121a597a2abcaeddc450aba7f9b1da5a8c03`  
**Host:** ThinkPad P16v Gen 1 — Ubuntu Linux 7.0.0-34, i9-13900H (20 threads), 30 GiB RAM, SK Hynix NVMe, **NVIDIA RTX 2000 Ada Laptop GPU (8 GiB)**, driver 595.91.07 / CUDA 13.2  
**Code changes:** none (evaluation only). Artifacts under `metrics-eval/`.

> Recovery: the original laptop commit with the full CSV/SVG corpus was never
> pushed. This tree restores the report, summaries, and Phase 4 capability
> snapshots. See [`RECOVERY.md`](RECOVERY.md).

## Verdict

Phases 1–4 **CPU correctness and metric gates pass** on this machine. The GPU is
**usable for CUDA compute** and EngramDB now **detects the NVIDIA driver**, but
the **GDS / NVMe→VRAM path is not verified** (`transfer_path=CpuDirect`). **Not
ready for real-world release** as a production database or as a GPU KV-offload
product; ready as a **CPU research prototype** with partial inference-adapter
surfaces.

## Hardware / GPU path

| Check | Result |
| --- | --- |
| `nvidia-smi` / driver | Pass (RTX 2000 Ada, 8188 MiB) |
| CUDA driver API (`cuInit`, 64 MiB `cuMemAlloc`) | Pass |
| PCIe | Gen4 ×8 |
| `nvidia-fs` / `/dev/nvidia-fs*` | **Missing** (needs root: `nvidia-fs-dkms` or equivalent) |
| `gdsio` | **Missing** |
| User-local `libcufile.so.0` + `cuFileDriverOpen` | Opens successfully |
| Engram default build capability | `nvidia_driver=true`, `cuda_visible=true`, `transfer_path=CpuDirect` |
| Engram `--features gds` capability | Loads `libcufile.so.0`, `cufile_driver_opened=true`, still **`CpuDirect`** because `nvidia_fs=false` |
| `GdsDirectVerified` / P2P counters | **Not achieved** |
| vLLM / SGLang / LMCache / Mooncake | **Not installed** |

Honest classification matches the implementation: libcufile alone never claims
verified GDS. This laptop GPU is also outside NVIDIA’s direct-GDS support matrix
(data-center / desktop Quadro); peer DMA between the GPU root port and a
different NVMe root port is not expected here.

## Validation gates vs expectations

| Gate | Expectation | This run | Status |
| --- | --- | --- | --- |
| Clippy `-D warnings` on repo pin `1.98.1` | Pass | Fails (`manual_is_multiple_of`, `chunks_exact_to_as_chunks`) | **Gap** |
| Same gates on MSRV `1.89.0` | — | Pass | Used for eval |
| Phase 1 tests + benches | Pass | Pass | Pass |
| SIFT1M Recall@10 | **> 0.95** | **0.981** (min 0.9, n=200) | Pass |
| 10M-node materialization | Measured | 12.60 GiB physical, 74.2 s | Pass |
| Phase 3 Flight / Python / sessions | Pass | Pass | Pass |
| Phase 4 T0/T1 integrity + adapters | Pass | Pass | Pass |
| Phase 4 T2 vLLM logits | Bit-identical | Not run | Gap |
| Phase 4 T3 GDS bandwidth / TTFT | ~7 GB/s NVMe→VRAM; ~60 ms @ 100k tokens | Not measured | Gap |

## Key metrics (this host vs prior committed VM)

| Metric | Prior VM (`metrics/`) | This laptop (`metrics-eval/`) |
| --- | --- | ---: |
| Branch fork p50 @1 | 180 µs | 513 µs |
| Branch batch @1000 | 127 ms | 528 ms (~1.9k forks/s) |
| 5k × 1 KiB writes | 4,730 ops/s (4 threads) | 673 ops/s (20 threads) |
| SIFT Recall@10 | 0.981 | **0.981** |
| SIFT search p50 (`ef=4096`) | 51.8 ms | **23.2 ms** |
| Tri-modal p50 (2k nodes) | ~2.7 µs | **2.15 µs** |
| 10M-node write | 177 s | **74 s** |
| KV CPU write / read | 0.43 / 0.58 GB/s | **0.65 / 0.73 GB/s** |
| KV restore 10k tok (CPU sim) | 65 ms | **59 ms** |
| Flight nearest p50 | ~336–499 µs | **401 µs** |

Branch/write regressions vs the prior VM are consistent with durable metadata
serialization + laptop storage/power behavior; they are not GDS results.

## Product comparison (players named in the docs)

External stacks were **not runnable** here (no Docker/sudo, no
Postgres/Mongo/Neo4j/Qdrant/vLLM/LMCache/Mooncake). Numbers below are
**positioning estimates**, not head-to-head measurements on this host.

| Product / pattern | Role | How EngramDB compares on this evidence |
| --- | --- | --- |
| **PostgreSQL + pgvector** (+ recursive CTE) | Vector + graph via SQL | Engram’s O(1) CoW branch fork is a real design win vs table/schema copy. ANN quality gate matches (0.981@10) but needs high `ef_search` (~23 ms p50)—not a latency win over mature HNSW engines. |
| **Neo4j + Qdrant** (middleware) | Semantic filter then graph hop | Engram fused tri-modal path is **µs in-process**; Neo4j+Qdrant will be **ms+** with network/serialization. Unmeasured here. |
| **LMCache** | KV offload (CPU/DRAM/disk) | Same problem class. Engram’s differentiator is **GDS DMA**; without `nvidia-fs`/verified P2P it is another **CPU O_DIRECT** restore (~0.73 GB/s, high CPU%). No reason to claim superiority over LMCache today. |
| **Mooncake** | Distributed KV / RDMA-oriented offload | Mooncake targets multi-node serving. Engram is single-node prototype with no RDMA path measured. |
| **vLLM / SGLang native cache** | Inference runtime KV | Engram exposes adapters + Flight tickets only. **T2 logit identity not run.** |
| **NVIDIA GDS / `gdsio`** | NVMe→GPU bandwidth oracle | Required for Phase 4 claims (~7 GB/s, low host CPU). **Not installed / not verified.** |
| **Traditional RAG prefill** | Recompute 100k-token context | Doc target: 3–10 s GPU prefill vs ~60 ms GDS restore. **Unverified.** CPU Engram restore of ~400 MB @ 0.73 GB/s ≈ **~0.55 s** plus host→device copy. |

**Bottom line vs peers:** Engram’s unique story (branchable durable store + fused
hybrid index + GDS KV) is only half-proven: storage/indexing/Flight look solid
for a prototype; the **GPU offload leap vs LMCache/Mooncake/vLLM is still
aspirational**.

## Release gaps (blockers and near-term)

1. **GDS stack incomplete on this class of machine without root:** install/load
   `nvidia-fs`, system `libcufile`, `gdsio`; confirm laptop NVMe+GPU topology
   actually supports P2P (many laptops do not; this one is not in the direct-GDS
   matrix).
2. **No `GdsDirectVerified` path in capability detection without P2P evidence** —
   correct honesty, but means marketing must not claim GDS speedups.
3. **T2 missing:** real vLLM/SGLang restore + bit-identical logits on a model that
   fits 8 GiB (tight for large models).
4. **Pinned Rust 1.98.1 Clippy fails** — CI/release gate broken on the declared
   toolchain; MSRV 1.89 still green.
5. **Production DB gaps (unchanged):** no compaction/GC, serialized durable
   branch lock, no auth, no multi-rank KV publish, Flight buffers through host
   memory, no delta KV, hybrid merge limitations, Cachegrind still shows no
   fused-cache win.
6. **Competitor benches not wired in CI** — cannot ship comparative claims
   without provisioned Postgres/Qdrant/Neo4j/LMCache/Mooncake on the same NVMe.
7. **8 GiB VRAM ceiling** — limits end-to-end demo size for modern LLMs even if
   GDS works.

## Artifacts

- Summary JSON: `metrics-eval/evaluation-summary.json`
- Host inventory: `metrics-eval/host-inventory.txt`
- Toolchain note: `metrics-eval/toolchain-note.txt`
- Phase 4 capability / bandwidth / TTFT / integrity snapshots under
  `metrics-eval/phase4/`
- Compact Phase 1–3 summaries under `metrics-eval/phase{1,2,3}/`
- Recovery note: `metrics-eval/RECOVERY.md`
