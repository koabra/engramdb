#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

metrics_dir="${PHASE2_METRICS_DIR:-metrics/phase2}"
scratch_dir="${PHASE2_SCRATCH_DIR:-/tmp/engramdb-phase2-validation}"
sift_dir="${PHASE2_SIFT_DIR:-/tmp/engramdb-sift1m}"
base_limit="${PHASE2_SIFT_BASE_LIMIT:-1000000}"
sift_queries="${PHASE2_SIFT_QUERIES:-10000}"
query_count="${PHASE2_QUERY_COUNT:-10000}"
storage_sample="${PHASE2_STORAGE_SAMPLE:-100000}"
rm -rf "$metrics_dir"
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

if [[ "${PHASE2_SKIP_SIFT:-0}" != "1" ]]; then
  scripts/fetch_sift1m.sh "$sift_dir" > "$metrics_dir/sift1m-sha256.txt"
  cargo run --release --bin phase2-bench -- \
    sift \
    --sift-dir "$sift_dir" \
    --data-dir "$scratch_dir/sift-index" \
    --output "$metrics_dir/sift-recall.csv" \
    --base-limit "$base_limit" \
    --queries "$sift_queries"
  git rev-parse HEAD > "$metrics_dir/sift-validation-commit.txt"
fi

cargo run --release --bin phase2-bench -- \
  query \
  --data-dir "$scratch_dir/query-index" \
  --output "$metrics_dir/query-latency.csv" \
  --nodes 10000 \
  --queries "$query_count" \
  --dimension 128

cargo run --release --bin phase2-bench -- \
  storage \
  --data-dir "$scratch_dir/storage-layout" \
  --output "$metrics_dir/storage-footprint.csv" \
  --sample-nodes "$storage_sample" \
  --dimension 768 \
  --edges 10

if command -v valgrind >/dev/null && command -v cg_annotate >/dev/null; then
  valgrind --tool=cachegrind --cache-sim=yes \
    --I1=32768,8,64 --D1=49152,12,64 --LL=2097152,16,64 \
    --cachegrind-out-file="$metrics_dir/cachegrind-fused.out" \
    target/release/phase2-bench profile-fused --data-dir "$scratch_dir/profile-fused"
  valgrind --tool=cachegrind --cache-sim=yes \
    --I1=32768,8,64 --D1=49152,12,64 --LL=2097152,16,64 \
    --cachegrind-out-file="$metrics_dir/cachegrind-split.out" \
    target/release/phase2-bench profile-split
  cg_annotate "$metrics_dir/cachegrind-fused.out" > "$metrics_dir/cachegrind-fused.txt"
  cg_annotate "$metrics_dir/cachegrind-split.out" > "$metrics_dir/cachegrind-split.txt"
  python3 scripts/summarize_cachegrind.py \
    --fused "$metrics_dir/cachegrind-fused.txt" \
    --split "$metrics_dir/cachegrind-split.txt" \
    --csv "$metrics_dir/cachegrind-summary.csv" \
    --svg "$metrics_dir/cachegrind-summary.svg"
  git rev-parse HEAD > "$metrics_dir/cache-validation-commit.txt"
else
  printf 'valgrind/cg_annotate unavailable; cachegrind not executed\n' \
    > "$metrics_dir/cachegrind-unavailable.txt"
fi

if command -v perf >/dev/null; then
  if ! perf stat -x, -e cache-references,cache-misses \
    -o "$metrics_dir/perf-fused.csv" \
    target/release/phase2-bench profile-fused \
    --data-dir "$scratch_dir/perf-fused"; then
    printf 'perf unavailable or denied by kernel policy\n' > "$metrics_dir/perf-unavailable.txt"
  elif ! perf stat -x, -e cache-references,cache-misses \
    -o "$metrics_dir/perf-split.csv" \
    target/release/phase2-bench profile-split; then
    printf 'perf split-layout run unavailable or denied by kernel policy\n' \
      > "$metrics_dir/perf-unavailable.txt"
  fi
else
  printf 'perf command unavailable\n' > "$metrics_dir/perf-unavailable.txt"
fi

if [[ -n "${POSTGRES_DSN:-}" ]]; then
  python3 scripts/compare_phase2_products.py postgresql \
    --output "$metrics_dir/postgresql-query-latency.csv"
fi
if [[ -n "${NEO4J_URI:-}" ]] && [[ -n "${NEO4J_PASSWORD:-}" ]] && [[ -n "${QDRANT_URL:-}" ]]; then
  python3 scripts/compare_phase2_products.py neo4j-qdrant \
    --output "$metrics_dir/neo4j-qdrant-query-latency.csv"
fi
comparison_status="$metrics_dir/product-comparison-unavailable.txt"
rm -f "$comparison_status"
if [[ -z "${POSTGRES_DSN:-}" ]]; then
  printf 'PostgreSQL/pgvector was not configured; no numeric result was produced.\n' \
    >> "$comparison_status"
fi
if [[ -z "${NEO4J_URI:-}" ]] || [[ -z "${NEO4J_PASSWORD:-}" ]] || [[ -z "${QDRANT_URL:-}" ]]; then
  printf 'Neo4j/Qdrant was not fully configured; no numeric result was produced.\n' \
    >> "$comparison_status"
fi

if [[ -f "$metrics_dir/sift-recall.csv" ]]; then
  python3 scripts/plot_phase2_metrics.py \
    recall "$metrics_dir/sift-recall.csv" "$metrics_dir/sift-recall.svg"
fi
python3 scripts/plot_phase2_metrics.py \
  latency "$metrics_dir/query-latency.csv" "$metrics_dir/query-latency.svg"
python3 scripts/plot_phase2_metrics.py \
  storage "$metrics_dir/storage-footprint.csv" "$metrics_dir/storage-footprint.svg"

printf 'Phase 2 validation complete; artifacts: %s\n' "$metrics_dir"
