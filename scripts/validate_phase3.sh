#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

metrics_dir="${PHASE3_METRICS_DIR:-metrics/phase3}"
scratch_dir="${PHASE3_SCRATCH_DIR:-/tmp/engramdb-phase3-validation}"
address="${PHASE3_ADDRESS:-127.0.0.1:50061}"
mkdir -p "$metrics_dir"
rm -rf "$scratch_dir"
mkdir -p "$scratch_dir"

{
  date --iso-8601=seconds
  git rev-parse HEAD
  uname -a
  rustc --version
  cargo --version
  python3 --version
  python3 -c 'import pyarrow, polars; print(f"pyarrow={pyarrow.__version__}"); print(f"polars={polars.__version__}")'
  printf 'logical_cpus=%s\n' "$(nproc)"
  lscpu
  df -h "$repo_root"
} > "$metrics_dir/system.txt"

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --release --test branch_properties thousand_randomized_forks_and_merges -- --ignored --exact
cargo test --release --test hybrid_durability
cargo test --release --test hybrid_indexing
cargo test --release --test execution_engine
python3 -m py_compile python/engramdb/__init__.py python/engramdb/client.py

cargo build --release --bin engramdb_server --bin phase3_bench
cargo run --release --bin phase3_bench -- \
  planner --output "$metrics_dir/planner.csv" --iterations 100000
cargo run --release --bin phase3_bench -- \
  sessions --data-dir "$scratch_dir/sessions" \
  --output "$metrics_dir/sessions.csv" --levels 1,10,100,1000,5000

target/release/engramdb_server \
  --data-dir "$scratch_dir/server" --address "$address" \
  > "$metrics_dir/server.log" 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
python3 - "$address" <<'PY'
import socket, sys, time
host, port = sys.argv[1].rsplit(":", 1)
for _ in range(200):
    try:
        with socket.create_connection((host, int(port)), timeout=0.1):
            break
    except OSError:
        time.sleep(0.05)
else:
    raise SystemExit("Flight server did not become ready")
PY

python3 scripts/phase3_python_validation.py \
  --location "grpc://$address" \
  --output-dir "$metrics_dir" \
  --documents 5000 --dimensions 128 --queries 1000

cp "$metrics_dir/ingestion.csv" "$metrics_dir/product-ingestion.csv"
if [[ -n "${POSTGRES_DSN:-}" ]]; then
  python3 scripts/compare_phase3.py postgresql \
    --output "$scratch_dir/postgresql-ingestion.csv" \
    --profile "$scratch_dir/postgresql-ingestion.prof"
  tail -n +2 "$scratch_dir/postgresql-ingestion.csv" \
    >> "$metrics_dir/product-ingestion.csv"
fi
if [[ -n "${MONGODB_URI:-}" ]]; then
  python3 scripts/compare_phase3.py mongodb \
    --output "$scratch_dir/mongodb-ingestion.csv" \
    --profile "$scratch_dir/mongodb-ingestion.prof"
  tail -n +2 "$scratch_dir/mongodb-ingestion.csv" \
    >> "$metrics_dir/product-ingestion.csv"
fi

python3 scripts/plot_phase3.py ingestion \
  "$metrics_dir/product-ingestion.csv" "$metrics_dir/ingestion.svg"
python3 scripts/plot_phase3.py sessions \
  "$metrics_dir/sessions.csv" "$metrics_dir/sessions.svg"
python3 scripts/plot_phase3.py zero-copy \
  "$metrics_dir/zero-copy.csv" "$metrics_dir/zero-copy.svg"
python3 scripts/plot_phase3.py latency \
  "$metrics_dir/flight-query-latency.csv" "$metrics_dir/flight-query-latency.svg"
python3 scripts/plot_phase3.py planner \
  "$metrics_dir/planner.csv" "$metrics_dir/planner.svg"

cleanup
trap - EXIT

printf 'Phase 3 validation complete; artifacts: %s\n' "$metrics_dir"
