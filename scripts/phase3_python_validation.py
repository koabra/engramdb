#!/usr/bin/env python3
"""Validate Python Flight ingestion, query allocation scaling, and latency."""

from __future__ import annotations

import argparse
import cProfile
import csv
import gc
import io
import pstats
import time
import tracemalloc
from pathlib import Path

import pyarrow as pa

from engramdb import EngramClient


def build_table(rows: int, dimensions: int) -> pa.Table:
    vectors = [
        (1.0 if dimension % 32 == row % 32 else 0.01)
        for row in range(rows)
        for dimension in range(dimensions)
    ]
    return pa.table(
        {
            "key": pa.array([f"doc-{row:08}" for row in range(rows)]),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors, type=pa.float32()), dimensions
            ),
            "assertion_time": pa.array(range(rows), type=pa.uint64()),
            "valid_time": pa.array([100] * rows, type=pa.int64()),
        }
    )


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="grpc://127.0.0.1:50051")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=5000)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--queries", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = EngramClient(args.location)
    session = client.fork_session(client.main_branch())
    table = build_table(args.documents, args.dimensions)
    profiler = cProfile.Profile()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    profiler.enable()
    client.ingest(table, session)
    profiler.disable()
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    profiler.dump_stats(args.output_dir / "engramdb-ingest.prof")
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(30)
    (args.output_dir / "engramdb-ingest-profile.txt").write_text(
        stream.getvalue(), encoding="utf-8"
    )
    write_rows(
        args.output_dir / "ingestion.csv",
        [
            "product",
            "documents",
            "dimensions",
            "arrow_bytes",
            "client_cpu_s",
            "wall_s",
        ],
        [
            {
                "product": "engramdb-arrow-flight",
                "documents": args.documents,
                "dimensions": args.dimensions,
                "arrow_bytes": table.nbytes,
                "client_cpu_s": f"{cpu_seconds:.6f}",
                "wall_s": f"{wall_seconds:.6f}",
            }
        ],
    )

    vector = ",".join("1.0" if dimension % 32 == 0 else "0.01" for dimension in range(args.dimensions))
    zero_copy_rows = []
    for limit in (10, 100, 1000, args.documents):
        gc.collect()
        tracemalloc.start()
        result = client.query(f"VECTOR NEAREST [{vector}] LIMIT {limit}", session)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        zero_copy_rows.append(
            {
                "rows": result.num_rows,
                "arrow_bytes": result.nbytes,
                "python_peak_bytes": peak,
            }
        )
    write_rows(
        args.output_dir / "zero-copy.csv",
        ["rows", "arrow_bytes", "python_peak_bytes"],
        zero_copy_rows,
    )

    latency_rows = []
    query = f"VECTOR NEAREST [{vector}] LIMIT 10"
    for sample in range(args.queries):
        started = time.perf_counter_ns()
        result = client.query(query, session)
        latency_rows.append(
            {
                "product": "engramdb-arrow-flight",
                "sample": sample,
                "latency_us": (time.perf_counter_ns() - started) / 1000,
                "rows": result.num_rows,
            }
        )
    write_rows(
        args.output_dir / "flight-query-latency.csv",
        ["product", "sample", "latency_us", "rows"],
        latency_rows,
    )
    client.commit_session(session)


if __name__ == "__main__":
    main()
