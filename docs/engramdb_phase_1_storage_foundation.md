# Phase 1: Storage Foundation (The Branching CoW Engine)

## 1. Architectural Overview
The goal of Phase 1 is to build the bedrock of EngramDB: a bare-metal storage engine in Rust that supports zero-cost branching and bitemporal concurrency. We discard traditional linear Write-Ahead Logs (WAL) in favor of a content-addressable, Copy-on-Write (CoW) Directed Acyclic Graph (DAG) architecture.

### Key Components
*   **I/O Layer:** Built strictly on Linux `io_uring` to maximize NVMe asynchronous queue depth. 
*   **Page Cache (Buffer Pool):** A custom user-space buffer pool bypassing the OS page cache (`O_DIRECT`). Pages are referenced via atomic reference counting (`Arc<Page>`) and Hazard Pointers to allow lock-free reads.
*   **Data Structure:** A Fractal Tree / B-Link Tree where node IDs are cryptographic hashes (Blake3) of their contents.
*   **Branching Manager:** To fork a state, the engine clones the root node's pointer and assigns a new Epoch ID. Mutations on this branch allocate new leaf blocks and bubble up new intermediate nodes up to the new root, leaving the original tree completely untouched.

## 2. Implementation Plan

**Step 1.1: Asynchronous I/O & Buffer Pool Manager**
*   Implement an `io_uring` wrapper using the `io-uring` Rust crate.
*   Allocate memory pages aligned to 4KB/64KB boundaries to support `O_DIRECT`.
*   Implement an eviction policy optimized for graph/vector workloads (e.g., CLOCK-Pro or LIRS).

**Step 2.2: The Content-Addressable B-Tree**
*   Define node structures: `LeafNode` and `InternalNode`. 
*   Implement Blake3 hashing for serialization. A parent's pointer to a child is a tuple of `(Blake3Hash, DiskOffset)`.
*   Implement structural sharing (CoW). On `insert/update`, copy the target leaf, apply the mutation, hash the new leaf, and recursively update the parent.

**Step 2.3: Transaction & Branching APIs**
*   Implement the `Branch` struct: `id: UUID`, `parent_id: UUID`, `root_hash: Hash`.
*   Create an append-only metadata log that records branch creation, commits, and merges.
*   Implement basic three-way merge resolution for non-conflicting bitemporal ranges.

## 3. Testing & Validation Methodology

*   **Concurrency Fuzzing (Loom):** Use Rust's `loom` crate to exhaustively test the lock-free Hazard Pointer implementations for race conditions during concurrent node splits and reads.
*   **Crash Consistency/Fault Injection:** Write a test harness that simulates power failure by forcefully dropping the `io_uring` submission queue at randomized intervals. Upon restart, validation scripts traverse the DAG from the last committed root hash to ensure strict immutability and zero corruption.
*   **Branch Integrity:** Write property-based tests (via `proptest`) that perform thousands of random forks and merges, asserting that reads on isolated branches never leak state into their parents.

## 4. Comparison Metrics & How to Obtain Them

### Metric 1: Branch Creation Latency
*   **EngramDB:** Native `fork()` API invocation time.
*   **PostgreSQL:** Measure the time to create an isolated schema (`CREATE SCHEMA clone; CREATE TABLE clone.x AS SELECT * FROM public.x;`) or create a new logical replication slot.
*   **MongoDB:** Measure the time to run an aggregation pipeline with `$out` to copy a collection.
*   **How to Obtain:** Run Python scripts using `time.perf_counter()` wrapping the respective DB-API/asyncpg/PyMongo commands. Plot latency at $N=1, 10, 100, 1000$ concurrent branch requests. EngramDB should remain $O(1)$ ($\sim 100 \mu s$), while Postgres/Mongo will scale linearly in $O(N)$ with data size.

### Metric 2: Write Throughput under High Concurrency (WAL Amplification)
*   **Measurement:** Ops/sec for 5,000 parallel workers performing 1KB micro-writes.
*   **How to Obtain:** 
    *   PostgreSQL: Use `pgbench -j 5000 -c 5000 -T 300` on a custom schema. Monitor disk write bandwidth via `iostat -dx 1`.
    *   EngramDB: Write a Rust-based synthetic load generator. Measure actual disk bytes written versus logical data size to calculate write amplification.