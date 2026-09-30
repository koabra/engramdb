#!/usr/bin/env python3
"""Profile 5,000-document ingestion through PostgreSQL or MongoDB clients."""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
import os
import pstats
import time
from pathlib import Path


def documents(count: int, dimensions: int) -> list[dict[str, object]]:
    return [
        {
            "key": f"doc-{row:08}",
            "payload": {"event": "memory", "sequence": row},
            "embedding": [
                1.0 if dimension % 32 == row % 32 else 0.01
                for dimension in range(dimensions)
            ],
        }
        for row in range(count)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("product", choices=("postgresql", "mongodb"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=5000)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI"))
    return parser.parse_args()


def postgresql(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    if not args.postgres_dsn:
        raise SystemExit("POSTGRES_DSN or --postgres-dsn is required")
    try:
        import psycopg2
        from psycopg2.extras import execute_batch
    except ImportError as error:
        raise SystemExit("install the current psycopg2 package") from error
    connection = psycopg2.connect(args.postgres_dsn)
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS engram_phase3_ingest")
            cursor.execute(
                "CREATE TABLE engram_phase3_ingest "
                "(key text PRIMARY KEY, payload jsonb, embedding real[])"
            )
            execute_batch(
                cursor,
                "INSERT INTO engram_phase3_ingest VALUES (%s, %s::jsonb, %s)",
                [
                    (row["key"], json.dumps(row["payload"]), row["embedding"])
                    for row in rows
                ],
                page_size=1000,
            )
    finally:
        connection.close()


def mongodb(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    if not args.mongo_uri:
        raise SystemExit("MONGODB_URI or --mongo-uri is required")
    try:
        from pymongo import MongoClient
    except ImportError as error:
        raise SystemExit("install the current pymongo package") from error
    client = MongoClient(args.mongo_uri)
    try:
        collection = client["engram_phase3"]["ingest"]
        collection.drop()
        collection.insert_many(rows, ordered=False)
    finally:
        client.close()


def main() -> None:
    args = parse_args()
    rows = documents(args.documents, args.dimensions)
    operation = postgresql if args.product == "postgresql" else mongodb
    profiler = cProfile.Profile()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    profiler.enable()
    operation(args, rows)
    profiler.disable()
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(args.profile)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(30)
    args.profile.with_suffix(".txt").write_text(stream.getvalue(), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "product",
                "documents",
                "dimensions",
                "arrow_bytes",
                "client_cpu_s",
                "wall_s",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": args.product,
                "documents": args.documents,
                "dimensions": args.dimensions,
                "arrow_bytes": "",
                "client_cpu_s": f"{cpu_seconds:.6f}",
                "wall_s": f"{wall_seconds:.6f}",
            }
        )


if __name__ == "__main__":
    main()
