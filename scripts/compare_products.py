#!/usr/bin/env python3
"""Run like-for-like branch-copy experiments against PostgreSQL or MongoDB.

Optional client libraries are intentionally imported only for the selected
product. Install current releases with:
    python -m pip install asyncpg pymongo
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


LEVELS = (1, 10, 100, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("product", choices=("postgresql", "mongodb"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--concurrency", default="1,10,100,1000")
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI"))
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "product",
        "operation",
        "concurrency",
        "sample",
        "latency_us",
        "batch_us",
        "dataset_records",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


async def postgresql(args: argparse.Namespace, levels: list[int]) -> list[dict[str, object]]:
    if not args.postgres_dsn:
        raise SystemExit("POSTGRES_DSN or --postgres-dsn is required")
    try:
        import asyncpg
    except ImportError as error:
        raise SystemExit("install the current asyncpg package") from error

    setup = await asyncpg.connect(args.postgres_dsn)
    try:
        await setup.execute("DROP TABLE IF EXISTS public.engram_phase1_source")
        await setup.execute(
            """
            CREATE TABLE public.engram_phase1_source AS
            SELECT value AS id, repeat(chr(65 + value % 26), 512) AS payload
            FROM generate_series(1, $1::integer) AS value
            """,
            args.records,
        )
    finally:
        await setup.close()

    pool = await asyncpg.create_pool(
        args.postgres_dsn, min_size=1, max_size=max(levels), command_timeout=300
    )
    rows: list[dict[str, object]] = []
    try:
        for concurrency in levels:
            async def clone(sample: int) -> tuple[int, float]:
                schema = f"engram_clone_{uuid.uuid4().hex}"
                started = time.perf_counter_ns()
                async with pool.acquire() as connection:
                    await connection.execute(f'CREATE SCHEMA "{schema}"')
                    await connection.execute(
                        f'CREATE TABLE "{schema}".data AS '
                        "TABLE public.engram_phase1_source"
                    )
                    await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
                return sample, (time.perf_counter_ns() - started) / 1000

            batch_started = time.perf_counter_ns()
            values = await asyncio.gather(*(clone(i) for i in range(concurrency)))
            batch_us = (time.perf_counter_ns() - batch_started) / 1000
            rows.extend(
                {
                    "product": "postgresql",
                    "operation": "schema_table_copy",
                    "concurrency": concurrency,
                    "sample": sample,
                    "latency_us": latency,
                    "batch_us": batch_us,
                    "dataset_records": args.records,
                }
                for sample, latency in values
            )
    finally:
        await pool.close()
    return rows


def mongodb(args: argparse.Namespace, levels: list[int]) -> list[dict[str, object]]:
    if not args.mongo_uri:
        raise SystemExit("MONGODB_URI or --mongo-uri is required")
    try:
        from pymongo import MongoClient
    except ImportError as error:
        raise SystemExit("install the current pymongo package") from error

    client = MongoClient(args.mongo_uri)
    database = client["engram_phase1_benchmark"]
    source = database["source"]
    source.drop()
    source.insert_many(
        {"_id": index, "payload": chr(65 + index % 26) * 512}
        for index in range(args.records)
    )
    rows: list[dict[str, object]] = []
    try:
        for concurrency in levels:
            def clone(sample: int) -> tuple[int, float]:
                target = f"clone_{uuid.uuid4().hex}"
                started = time.perf_counter_ns()
                list(source.aggregate([{"$out": target}], allowDiskUse=True))
                database[target].drop()
                return sample, (time.perf_counter_ns() - started) / 1000

            batch_started = time.perf_counter_ns()
            # PyMongo is blocking. The executor preserves true overlapping
            # requests and includes driver-pool queueing in observed latency.
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                values = list(executor.map(clone, range(concurrency)))
            batch_us = (time.perf_counter_ns() - batch_started) / 1000
            rows.extend(
                {
                    "product": "mongodb",
                    "operation": "aggregation_out",
                    "concurrency": concurrency,
                    "sample": sample,
                    "latency_us": latency,
                    "batch_us": batch_us,
                    "dataset_records": args.records,
                }
                for sample, latency in values
            )
    finally:
        source.drop()
        client.close()
    return rows


def main() -> None:
    args = parse_args()
    levels = [int(value) for value in args.concurrency.split(",")]
    if any(level <= 0 for level in levels):
        raise SystemExit("concurrency values must be positive")
    if args.product == "postgresql":
        rows = asyncio.run(postgresql(args, levels))
    else:
        rows = mongodb(args, levels)
    write_rows(args.output, rows)
    p50 = statistics.median(float(row["latency_us"]) for row in rows)
    print(f"wrote {len(rows)} observations to {args.output}; global p50={p50:.1f} us")


if __name__ == "__main__":
    main()
