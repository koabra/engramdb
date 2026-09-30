#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

metrics_dir="${PHASE2_METRICS_DIR:-metrics/phase2}"
scratch_dir="${PHASE2_SCRATCH_DIR:-/tmp/engramdb-phase2-validation}"
recall_nodes="${PHASE2_RECALL_NODES:-10000}"
recall_queries="${PHASE2_RECALL_QUERIES:-200}"
query_nodes="${PHASE2_QUERY_NODES:-2000}"
query_count="${PHASE2_QUERY_COUNT:-10000}"
amplification_nodes="${PHASE2_AMPLIFICATION_NODES:-10000000}"
mkdir -p "$metrics_dir"
rm -rf "$scratch_dir"
mkdir -p "$scratch_dir"

{
  date --iso-8601=seconds
  git rev-parse HEAD
  uname -a
  rustc --version
  cargo --version
  printf 'logical_cpus=%s\n' "$(nproc)"
  lscpu
  df -h "$repo_root"
} > "$metrics_dir/system.txt"

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --release --test branch_properties thousand_randomized_forks_and_merges -- --ignored --exact
cargo test --release --test hybrid_indexing

if [[ -n "${SIFT1M_DIR:-}" ]] \
  && [[ -f "$SIFT1M_DIR/sift_base.fvecs" ]] \
  && [[ -f "$SIFT1M_DIR/sift_query.fvecs" ]] \
  && [[ -f "$SIFT1M_DIR/sift_groundtruth.ivecs" ]]; then
  cargo run --release --bin phase2_bench -- \
    recall \
    --data-file "$scratch_dir/recall.dat" \
    --output "$metrics_dir/recall.csv" \
    --base "$SIFT1M_DIR/sift_base.fvecs" \
    --queries-file "$SIFT1M_DIR/sift_query.fvecs" \
    --groundtruth "$SIFT1M_DIR/sift_groundtruth.ivecs" \
    --limit-queries "$recall_queries"
  printf 'recall_dataset=SIFT1M\n' > "$metrics_dir/dataset.txt"
else
  cargo run --release --bin phase2_bench -- \
    recall \
    --data-file "$scratch_dir/recall.dat" \
    --output "$metrics_dir/recall.csv" \
    --nodes "$recall_nodes" \
    --queries "$recall_queries" \
    --dimensions 128
  printf 'recall_dataset=synthetic-clustered; set SIFT1M_DIR for standard validation\n' \
    > "$metrics_dir/dataset.txt"
fi

cargo run --release --bin phase2_bench -- \
  query \
  --data-file "$scratch_dir/query.dat" \
  --output "$metrics_dir/query-latency.csv" \
  --nodes "$query_nodes" \
  --queries "$query_count" \
  --dimensions 128

cargo run --release --bin phase2_bench -- \
  cache-layout \
  --output "$metrics_dir/cache-layout.csv" \
  --iterations 10000

cargo run --release --bin phase2_bench -- \
  amplification \
  --data-file "$scratch_dir/amplification.dat" \
  --output "$metrics_dir/amplification.csv" \
  --nodes "$amplification_nodes" \
  --dimensions 768 \
  --edges 10

{
  if command -v perf >/dev/null 2>&1; then
    printf 'perf=available\n'
    perf stat -x, -e cache-references,cache-misses \
      cargo run --release --bin phase2_bench -- \
        cache-layout --output "$scratch_dir/perf-cache-layout.csv" --iterations 10000 \
      2> "$metrics_dir/perf-cache.txt" || printf 'perf_run=failed\n'
  else
    printf 'perf=unavailable\n'
  fi
  if command -v valgrind >/dev/null 2>&1; then
    printf 'cachegrind=available\n'
    valgrind --tool=cachegrind --branch-sim=yes \
      --cachegrind-out-file="$metrics_dir/cachegrind-fused.out" \
      target/release/phase2_bench \
        cache-layout --layout fused \
        --output "$scratch_dir/cachegrind-fused.csv" --iterations 1000 \
      2> "$metrics_dir/cachegrind-fused.txt" || printf 'cachegrind_fused_run=failed\n'
    valgrind --tool=cachegrind --branch-sim=yes \
      --cachegrind-out-file="$metrics_dir/cachegrind-pointer.out" \
      target/release/phase2_bench \
        cache-layout --layout pointer \
        --output "$scratch_dir/cachegrind-pointer.csv" --iterations 1000 \
      2> "$metrics_dir/cachegrind-pointer.txt" || printf 'cachegrind_pointer_run=failed\n'
  else
    printf 'cachegrind=unavailable\n'
  fi
} > "$metrics_dir/profiling-availability.txt"

cp "$metrics_dir/query-latency.csv" "$metrics_dir/product-query-latency.csv"
if [[ -n "${POSTGRES_DSN:-}" ]]; then
  python3 scripts/compare_phase2.py postgresql \
    --output "$scratch_dir/postgresql-query-latency.csv" \
    --nodes "$query_nodes" --queries "$query_count" --dimensions 128
  tail -n +2 "$scratch_dir/postgresql-query-latency.csv" \
    >> "$metrics_dir/product-query-latency.csv"
fi
if [[ -n "${NEO4J_URI:-}" && -n "${NEO4J_PASSWORD:-}" && -n "${QDRANT_URL:-}" ]]; then
  python3 scripts/compare_phase2.py neo4j-qdrant \
    --output "$scratch_dir/neo4j-qdrant-query-latency.csv" \
    --nodes "$query_nodes" --queries "$query_count" --dimensions 128
  tail -n +2 "$scratch_dir/neo4j-qdrant-query-latency.csv" \
    >> "$metrics_dir/product-query-latency.csv"
fi

python3 scripts/plot_phase2.py recall \
  "$metrics_dir/recall.csv" "$metrics_dir/recall.svg"
python3 scripts/plot_phase2.py latency \
  "$metrics_dir/product-query-latency.csv" "$metrics_dir/query-latency.svg"
python3 scripts/plot_phase2.py amplification \
  "$metrics_dir/amplification.csv" "$metrics_dir/amplification.svg"
python3 scripts/plot_phase2.py cache \
  "$metrics_dir/cache-layout.csv" "$metrics_dir/cache-layout.svg"

PHASE1_METRICS_DIR="$metrics_dir/phase1-regression" \
PHASE1_SCRATCH_DIR="$scratch_dir/phase1" \
  ./scripts/validate_phase1.sh

printf 'Phase 2 validation complete; artifacts: %s\n' "$metrics_dir"
