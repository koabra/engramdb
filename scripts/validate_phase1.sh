#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

metrics_dir="${PHASE1_METRICS_DIR:-metrics/phase1}"
scratch_dir="${PHASE1_SCRATCH_DIR:-/tmp/engramdb-phase1-validation}"
write_workers="${PHASE1_WRITE_WORKERS:-5000}"
physical_threads="${PHASE1_PHYSICAL_THREADS:-$(nproc)}"
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

cargo run --release --bin phase1-bench -- \
  branch \
  --data-dir "$scratch_dir/branch" \
  --output "$metrics_dir/branch-latency.csv" \
  --concurrency 1,10,100,1000 \
  --seed-records 1000

cargo run --release --bin phase1-bench -- \
  write \
  --data-dir "$scratch_dir/write" \
  --output "$metrics_dir/write-throughput.csv" \
  --workers "$write_workers" \
  --physical-threads "$physical_threads"

python3 scripts/plot_metrics.py \
  branch "$metrics_dir/branch-latency.csv" "$metrics_dir/branch-latency.svg"
python3 scripts/plot_metrics.py \
  write "$metrics_dir/write-throughput.csv" "$metrics_dir/write-throughput.svg"

printf 'Phase 1 validation complete; artifacts: %s\n' "$metrics_dir"
