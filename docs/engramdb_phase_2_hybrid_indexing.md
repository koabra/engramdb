# Phase 2: Hybrid Indexing (Fused Tri-Modal Page Layout)

## 1. Architectural Overview
Traditional systems force queries to hit an HNSW index, join with a graph database, and filter via a relational heap. EngramDB Phase 2 fuses these representations. A single physical read (a 64KB block) loads a semantic vector, its graph edges, and temporal metadata directly into CPU cache lines.

### Key Components
*   **The 64KB Fused Block:**
    *   *Header (64 bytes):* Epoch, CRC32, Node Type.
    *   *Temporal Array:* Pairs of `(AssertionTime, ValidTime)`.
    *   *Vector Payload:* Product Quantized (PQ) or FP8 Scalar Quantized 768d/1536d embeddings.
    *   *Graph Payload:* Compressed Sparse Row (CSR) arrays for outbound edges `(TargetNodeHash, EdgeWeight, EdgeType)`.
*   **SIMD Execution:** Use AVX-512 (x86) or NEON (ARM) to compute dot products or L2 distances directly on the 64KB block memory without deserializing into intermediate Rust `Vec<f32>` arrays.

## 2. Implementation Plan

**Step 2.1: Page Layout Definitions**
*   Create memory-mapped Rust structs using `#[repr(C, packed)]` to ensure strict memory alignment matching the 64KB block spec.
*   Implement custom memory allocators within the block to pack the Vector Payload and Graph CSR tightly.

**Step 2.2: Vector Quantization & SIMD**
*   Implement an insertion pipeline that accepts FP32 vectors and compresses them to INT8/FP8.
*   Integrate `std::arch::x86_64` intrinsics to build an extremely fast block-level nearest-neighbor scan (e.g., processing 16 FP8 dimensions per cycle).
*   Build a global lightweight HNSW index where the leaf nodes map to these fused blocks.

**Step 2.3: Multi-Hop Retrieval Algorithms**
*   Implement Graph Traversal that accepts a lambda filter. As the traverser hops through the CSR graph edges, the lambda simultaneously evaluates the bitemporal interval and SIMD vector distance using data in the same block.

## 3. Testing & Validation Methodology

*   **Recall Validation:** Load the standard `SIFT1M` or `Glove-100-Angular` datasets. Assert that our quantized fused blocks achieve an acceptable Recall@10 ($> 0.95$) compared to exact K-NN.
*   **Multi-Modal Query Validation:** Generate a synthetic dataset where "Agents" move through a geographic graph, updating their location. Query: "Find agents semantically matching 'medic' near node X, accurate as of timestamp Y." Validate results against a slow, brute-force exact scan.
*   **Cache Line Profiling:** Use `valgrind --tool=cachegrind` and `perf stat -e cache-misses` to verify that our 64KB block layout genuinely reduces L2/L3 cache misses compared to pointer-chasing representations.

## 4. Comparison Metrics & How to Obtain Them

### Metric 1: Tri-Modal Query Latency (Semantic + Graph + Time)
*   **Workload:** Traverse 3 hops deep from a root node, filtering at each hop for a vector cosine similarity $> 0.8$ and an `updated_at < T`.
*   **EngramDB:** Direct internal query execution.
*   **PostgreSQL (`pgvector` + Recursive CTE):** Write a `WITH RECURSIVE` query combining joins with `<=>` vector operators. 
*   **Neo4j + Qdrant (Middleware approach):** Write a Python script using LangChain/NetworkX that queries Qdrant for semantic matches, passes IDs to Neo4j to find 3-hop neighbors, and filters in memory.
*   **How to Obtain:** Measure P50, P90, and P99 latencies for 10,000 randomized queries across all three setups using `pytest-benchmark`. EngramDB should demonstrate $10\times - 100\times$ lower latency by avoiding network hops and IPC boundaries.

### Metric 2: Memory/Disk Amplification
*   **Measurement:** Total physical disk bytes consumed to store 10 Million nodes (with 768d vectors and 10 edges each).
*   **How to Obtain:** Run `du -sh` on PostgreSQL's data directory (including the `pgvector` IVFFlat/HNSW indexes) vs. EngramDB's raw data directory.