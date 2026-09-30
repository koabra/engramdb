#!/usr/bin/env python3
"""Run external KV-cache restore commands without inventing unavailable results."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import statistics
import subprocess
import time
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "product", choices=("lmcache", "mooncake", "vllm-prefill", "raw-file")
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--command")
    parser.add_argument("--hardware-verified", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input does not exist: {args.input}")
    if args.product != "raw-file" and not args.command:
        raise SystemExit("--command is required for external products")
    latencies = []
    cpu_started = time.process_time()
    for _ in range(args.iterations):
        started = time.perf_counter_ns()
        if args.product == "raw-file":
            with args.input.open("rb", buffering=0) as source:
                while source.read(4 * 1024 * 1024):
                    pass
        else:
            environment = os.environ.copy()
            environment["ENGRAMDB_KV_INPUT"] = str(args.input)
            subprocess.run(
                shlex.split(args.command),
                check=True,
                env=environment,
                stdout=subprocess.DEVNULL,
            )
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    cpu_seconds = time.process_time() - cpu_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "product",
                "hardware_verified",
                "bytes",
                "iterations",
                "p50_ms",
                "p99_ms",
                "cpu_s",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": args.product,
                "hardware_verified": args.hardware_verified,
                "bytes": args.input.stat().st_size,
                "iterations": args.iterations,
                "p50_ms": f"{statistics.median(latencies):.6f}",
                "p99_ms": f"{percentile(latencies, 0.99):.6f}",
                "cpu_s": f"{cpu_seconds:.6f}",
            }
        )


if __name__ == "__main__":
    main()
