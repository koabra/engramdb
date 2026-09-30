#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

metrics_dir="${PHASE4_METRICS_DIR:-metrics/phase4}"
scratch_dir="${PHASE4_SCRATCH_DIR:-/tmp/engramdb-phase4-validation}"
address="${PHASE4_ADDRESS:-127.0.0.1:50071}"
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
cargo test --release --test kv_cache
cargo test --features gds --all-targets --no-run
python3 -m py_compile \
  python/engramdb/__init__.py \
  python/engramdb/client.py \
  python/engramdb/integrations/__init__.py \
  python/engramdb/integrations/vllm.py \
  python/engramdb/integrations/sglang.py

cargo build --release --bin engramdb_server --bin phase4_bench
cargo run --release --bin phase4_bench -- \
  capability --output "$metrics_dir/capability.json"
cargo run --release --bin phase4_bench -- \
  bandwidth --data-dir "$scratch_dir/bandwidth" \
  --output "$metrics_dir/bandwidth.csv" \
  --bytes 134217728 --iterations 3
cargo run --release --bin phase4_bench -- \
  ttft-sim --data-dir "$scratch_dir/ttft" \
  --output "$metrics_dir/ttft-simulation.csv" \
  --tokens 1000,10000 --bytes-per-token 4096

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
python3 scripts/phase4_python_integrity.py \
  --location "grpc://$address" --output-dir "$metrics_dir"
cleanup
trap - EXIT

python3 scripts/compare_phase4.py raw-file \
  --input "$scratch_dir/bandwidth/kv-cache.dat" \
  --output "$metrics_dir/product-comparison.csv" --iterations 5
if [[ -n "${LMCACHE_COMPARE_COMMAND:-}" ]]; then
  python3 scripts/compare_phase4.py lmcache \
    --input "$scratch_dir/bandwidth/kv-cache.dat" \
    --output "$scratch_dir/lmcache.csv" \
    --command "$LMCACHE_COMPARE_COMMAND"
  tail -n +2 "$scratch_dir/lmcache.csv" >> "$metrics_dir/product-comparison.csv"
fi
if [[ -n "${MOONCAKE_COMPARE_COMMAND:-}" ]]; then
  python3 scripts/compare_phase4.py mooncake \
    --input "$scratch_dir/bandwidth/kv-cache.dat" \
    --output "$scratch_dir/mooncake.csv" \
    --command "$MOONCAKE_COMPARE_COMMAND"
  tail -n +2 "$scratch_dir/mooncake.csv" >> "$metrics_dir/product-comparison.csv"
fi

python3 scripts/plot_phase4.py bandwidth \
  "$metrics_dir/bandwidth.csv" "$metrics_dir/bandwidth.svg"
python3 scripts/plot_phase4.py ttft \
  "$metrics_dir/ttft-simulation.csv" "$metrics_dir/ttft-simulation.svg"
python3 scripts/plot_phase4.py integrity \
  "$metrics_dir/integrity.csv" "$metrics_dir/integrity.svg"

python3 - "$metrics_dir/capability.json" "$metrics_dir/hardware-validation.txt" <<'PY'
import json, pathlib, sys
capability = json.loads(pathlib.Path(sys.argv[1]).read_text())
verified = capability["transfer_path"] == "GdsDirectVerified"
lines = [
    f"gds_direct_verified={str(verified).lower()}",
    f"transfer_path={capability['transfer_path']}",
    f"gdsio_available={str(capability['gdsio_path'] is not None).lower()}",
]
if not verified:
    lines.append("gds_bandwidth=not measured; CPU direct-I/O fallback only")
pathlib.Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY

if [[ "${RUN_FULL_PRIOR_VALIDATION:-0}" == "1" ]]; then
  ./scripts/validate_phase3.sh
  if [[ -n "${SIFT1M_DIR:-}" ]]; then
    ./scripts/validate_phase2.sh
  fi
fi

printf 'Phase 4 validation complete; artifacts: %s\n' "$metrics_dir"
