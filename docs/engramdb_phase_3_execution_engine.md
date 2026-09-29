# Phase 3: Execution Engine (Arrow Flight & Cypher/Vector API)

## 1. Architectural Overview
To interact with AI Agent orchestrators (like LangChain, AutoGen, or custom Python runtimes), EngramDB needs a high-performance network interface and a declarative query language. We avoid standard Postgres wire protocols because they force JSON/Text serialization. Instead, we use Apache Arrow Flight RPC for zero-copy memory transfers.

### Key Components
*   **Query Language (EnQL - Engram Query Language):** A hybrid of Cypher (for graph matching) with SQL-like temporal windows (`AS OF SYSTEM TIME`) and Vector predicates.
*   **Arrow Flight Server:** A gRPC-based server serving responses as Arrow RecordBatches, heavily utilizing Zero-Copy sharing to Python/Polars clients.
*   **Vectorized Execution Runtime:** A physical query planner that processes data in batches (e.g., 1024 tuples at a time) rather than row-by-row, keeping CPU instruction caches hot.

## 2. Implementation Plan

**Step 3.1: Parser and Query Planner**
*   Use Rust's `sqlparser` or `nom` to parse EnQL into an Abstract Syntax Tree (AST).
*   Translate AST to a Logical Plan (Filter, Projection, Traversal, VectorScan).
*   Build a cost-based Physical Optimizer. *Rule:* If vector predicate is highly selective (e.g., threshold > 0.98), do a vector index scan first, then traverse graph. If graph topology is strict, do graph traversal first, then vector SIMD scan.

**Step 3.2: Apache Arrow Integration**
*   Implement the `arrow` and `arrow-flight` Rust crates.
*   Map internal EngramDB fused block projections directly into Arrow `StructArrays`, `ListArrays` (for edges), and `FixedSizeListArrays` (for vectors).
*   Expose agent specific RPC methods: `ForkSession(parent_id) -> session_id`, `CommitSession(session_id)`, `Query(query_string, session_id)`.

**Step 3.3: Client SDKs**
*   Build a lightweight Python SDK (`pip install engramdb`) that wraps the Arrow Flight client, returning Pandas DataFrames or Polars LazyFrames instantaneously.

## 3. Testing & Validation Methodology

*   **Zero-Copy Validation:** Run memory profilers (`tracemalloc` in Python) to verify that calling the EngramDB SDK does not trigger Python-side memory allocations proportional to the data payload size (validating Arrow's memory mapping).
*   **Query Optimization Tests:** Write unit tests that input complex EnQL strings, dump the `EXPLAIN` physical plan, and assert that the heuristic optimizer correctly chooses the most restrictive index path.

## 4. Comparison Metrics & How to Obtain Them

### Metric 1: Serialization / Ingestion Latency (Client to DB)
*   **Workload:** A Python agent reads a batch of 5,000 JSON documents (e.g., an episodic memory dump), extracts embeddings, and sends them to the DB.
*   **EngramDB:** Client loads data into a PyArrow Table, transmits via Arrow Flight. 
*   **PostgreSQL/MongoDB:** Client uses `psycopg2.execute_batch` or `pymongo.insert_many` (which serializes Python dicts to JSONB/BSON over TCP).
*   **How to Obtain:** Profile the client-side CPU time spent during the database driver call using `cProfile` in Python. We want to prove that Postgres/Mongo drivers burn CPU serializing text, while Arrow Flight is near-instant CPU-wise.

### Metric 2: Max Concurrent Agent Sessions
*   **Measurement:** How many simultaneous speculative DB branches (sessions) can be maintained actively before RAM/CPU exhaustion.
*   **How to Obtain:** Deploy an AutoGen simulation with a progressively increasing number of agents. Compare the memory usage of EngramDB (which uses $O(1)$ CoW pointers) against Letta/Mem0 backed by PostgreSQL (which requires instantiating new session rows/tables per agent). Monitor via `htop` and `docker stats`.