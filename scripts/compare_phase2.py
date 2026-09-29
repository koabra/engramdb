#!/usr/bin/env python3
"""Run Phase 2 tri-modal workloads against external product topologies.

Install only the selected mode's current clients:
    python -m pip install asyncpg
    python -m pip install neo4j qdrant-client
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import time
from pathlib import Path


def vector(item: int, dimensions: int) -> list[float]:
    cluster = item % 32
    return [
        (1.0 if dimension % 32 == cluster else 0.0)
        + ((((item * 1_103_515_245 + dimension * 12_345) >> 8) & 0xFF) / 255 - 0.5)
        * 0.08
        for dimension in range(dimensions)
    ]


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.7g}" for value in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("product", choices=("postgresql", "neo4j-qdrant"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "product",
                "query",
                "latency_us",
                "results",
                "nodes",
                "dimensions",
                "hops",
                "minimum_cosine",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


async def postgresql(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.postgres_dsn:
        raise SystemExit("POSTGRES_DSN or --postgres-dsn is required")
    try:
        import asyncpg
    except ImportError as error:
        raise SystemExit("install the current asyncpg package") from error

    connection = await asyncpg.connect(args.postgres_dsn, command_timeout=600)
    try:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await connection.execute("DROP TABLE IF EXISTS engram_phase2_edges")
        await connection.execute("DROP TABLE IF EXISTS engram_phase2_nodes")
        await connection.execute(
            f"""
            CREATE TABLE engram_phase2_nodes (
                id integer PRIMARY KEY,
                embedding vector({args.dimensions}) NOT NULL,
                updated_at bigint NOT NULL
            );
            CREATE TABLE engram_phase2_edges (
                source integer NOT NULL,
                target integer NOT NULL,
                edge_type smallint NOT NULL
            );
            """
        )
        await connection.executemany(
            "INSERT INTO engram_phase2_nodes VALUES ($1, $2::vector, $3)",
            [
                (item, vector_literal(vector(item, args.dimensions)), item % 100)
                for item in range(args.nodes)
            ],
        )
        await connection.executemany(
            "INSERT INTO engram_phase2_edges VALUES ($1, $2, 1)",
            [
                (item, (item + step) % args.nodes)
                for item in range(args.nodes)
                for step in (1, 2, 3)
            ],
        )
        await connection.execute(
            "CREATE INDEX ON engram_phase2_edges(source); "
            "CREATE INDEX engram_phase2_hnsw ON engram_phase2_nodes "
            "USING hnsw (embedding vector_cosine_ops)"
        )
        statement = await connection.prepare(
            """
            WITH RECURSIVE reachable(id, depth) AS (
                SELECT $1::integer, 0
                UNION
                SELECT edge.target, reachable.depth + 1
                FROM reachable
                JOIN engram_phase2_edges edge ON edge.source = reachable.id
                WHERE reachable.depth < 3 AND edge.edge_type = 1
            )
            SELECT DISTINCT node.id
            FROM reachable
            JOIN engram_phase2_nodes node USING (id)
            WHERE node.updated_at < 75
              AND 1 - (node.embedding <=> $2::vector) > 0.8
            """
        )
        rows = []
        for query in range(args.queries):
            start = query * 7919 % args.nodes
            started = time.perf_counter_ns()
            result = await statement.fetch(
                start, vector_literal(vector(start, args.dimensions))
            )
            rows.append(
                {
                    "product": "postgresql-pgvector",
                    "query": query,
                    "latency_us": (time.perf_counter_ns() - started) / 1000,
                    "results": len(result),
                    "nodes": args.nodes,
                    "dimensions": args.dimensions,
                    "hops": 3,
                    "minimum_cosine": 0.8,
                }
            )
        return rows
    finally:
        await connection.close()


def neo4j_qdrant(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.neo4j_uri or not args.neo4j_password or not args.qdrant_url:
        raise SystemExit(
            "NEO4J_URI, NEO4J_PASSWORD, and QDRANT_URL (or matching flags) are required"
        )
    try:
        from neo4j import GraphDatabase
        from qdrant_client import QdrantClient, models
    except ImportError as error:
        raise SystemExit("install current neo4j and qdrant-client packages") from error

    neo4j = GraphDatabase.driver(
        args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password)
    )
    qdrant = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    collection = "engram_phase2"
    if qdrant.collection_exists(collection):
        qdrant.delete_collection(collection)
    qdrant.create_collection(
        collection,
        vectors_config=models.VectorParams(
            size=args.dimensions, distance=models.Distance.COSINE
        ),
    )
    qdrant.upload_points(
        collection,
        points=[
            models.PointStruct(
                id=item,
                vector=vector(item, args.dimensions),
                payload={"updated_at": item % 100},
            )
            for item in range(args.nodes)
        ],
        wait=True,
    )
    with neo4j.session() as session:
        session.run("MATCH (node) DETACH DELETE node").consume()
        session.run(
            "UNWIND $rows AS row CREATE (:Agent {id: row.id, updated_at: row.updated_at})",
            rows=[
                {"id": item, "updated_at": item % 100} for item in range(args.nodes)
            ],
        ).consume()
        session.run("CREATE INDEX agent_id IF NOT EXISTS FOR (node:Agent) ON (node.id)").consume()
        session.run(
            """
            UNWIND $rows AS row
            MATCH (source:Agent {id: row.source}), (target:Agent {id: row.target})
            CREATE (source)-[:LINK]->(target)
            """,
            rows=[
                {"source": item, "target": (item + step) % args.nodes}
                for item in range(args.nodes)
                for step in (1, 2, 3)
            ],
        ).consume()

    rows = []
    try:
        with neo4j.session() as session:
            for query in range(args.queries):
                start = query * 7919 % args.nodes
                started = time.perf_counter_ns()
                semantic = qdrant.query_points(
                    collection,
                    query=vector(start, args.dimensions),
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="updated_at", range=models.Range(lt=75)
                            )
                        ]
                    ),
                    score_threshold=0.8,
                    limit=args.nodes,
                    with_payload=False,
                ).points
                candidates = [int(point.id) for point in semantic]
                result = session.run(
                    """
                    MATCH (start:Agent {id: $start})
                    MATCH (start)-[:LINK*0..3]->(candidate:Agent)
                    WHERE candidate.id IN $candidates
                    RETURN DISTINCT candidate.id AS id
                    """,
                    start=start,
                    candidates=candidates,
                ).data()
                rows.append(
                    {
                        "product": "neo4j-qdrant",
                        "query": query,
                        "latency_us": (time.perf_counter_ns() - started) / 1000,
                        "results": len(result),
                        "nodes": args.nodes,
                        "dimensions": args.dimensions,
                        "hops": 3,
                        "minimum_cosine": 0.8,
                    }
                )
    finally:
        neo4j.close()
        qdrant.close()
    return rows


def main() -> None:
    args = parse_args()
    if args.product == "postgresql":
        rows = asyncio.run(postgresql(args))
    else:
        rows = neo4j_qdrant(args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} observations to {args.output}")


if __name__ == "__main__":
    main()
