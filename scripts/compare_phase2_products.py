#!/usr/bin/env python3
"""Run the Phase 2 tri-modal workload against external product stacks.

Current client releases are optional and imported only for the selected stack:
  python -m pip install asyncpg neo4j qdrant-client
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import time
from pathlib import Path


def vector_for(node_id: int, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    vector[node_id % min(dimension, 32)] = 1.0
    vector[(node_id * 13 + 7) % dimension] += 0.1
    return vector


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stack", choices=("postgresql", "neo4j-qdrant"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    return parser.parse_args()


def write(path: Path, product: str, rows: list[tuple[int, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "product",
                "query",
                "latency_us",
                "hits",
                "visited_records",
                "physical_block_reads",
                "nodes",
                "dimension",
                "hops",
            ]
        )
        for query, latency, hits in rows:
            writer.writerow([product, query, f"{latency:.3f}", hits, "", "", "", "", 3])


async def postgresql(args: argparse.Namespace) -> list[tuple[int, float, int]]:
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
            f"""CREATE TABLE engram_phase2_nodes(
                id bigint PRIMARY KEY,
                assertion_time bigint NOT NULL,
                valid_from bigint NOT NULL,
                valid_to bigint NOT NULL,
                embedding vector({args.dimension}) NOT NULL
            )"""
        )
        await connection.execute(
            """CREATE TABLE engram_phase2_edges(
                source bigint NOT NULL, target bigint NOT NULL, weight real NOT NULL,
                edge_type smallint NOT NULL
            )"""
        )
        await connection.executemany(
            "INSERT INTO engram_phase2_nodes VALUES($1, $2, 0, 1000, $3)",
            [
                (node, node % 10 + 1, str(vector_for(node, args.dimension)))
                for node in range(args.nodes)
            ],
        )
        await connection.executemany(
            "INSERT INTO engram_phase2_edges VALUES($1, $2, 1.0, 1)",
            [
                (node, (node + step) % args.nodes)
                for node in range(args.nodes)
                for step in range(1, 11)
            ],
        )
        await connection.execute("CREATE INDEX ON engram_phase2_edges(source)")
        rows = []
        statement = await connection.prepare(
            """WITH RECURSIVE reachable(id, depth) AS (
                 SELECT $1::bigint, 0
                 UNION
                 SELECT e.target, r.depth + 1
                 FROM reachable r JOIN engram_phase2_edges e ON e.source = r.id
                 WHERE r.depth < 3
               )
               SELECT n.id
               FROM reachable r JOIN engram_phase2_nodes n ON n.id = r.id
               WHERE n.assertion_time <= 10
                 AND n.valid_from <= 500 AND 500 < n.valid_to
                 AND 1 - (n.embedding <=> $2::vector) >= 0.8"""
        )
        for query_id in range(args.queries):
            root = query_id * 7919 % args.nodes
            started = time.perf_counter_ns()
            result = await statement.fetch(root, str(vector_for(root, args.dimension)))
            rows.append((query_id, (time.perf_counter_ns() - started) / 1000, len(result)))
        return rows
    finally:
        await connection.close()


def neo4j_qdrant(args: argparse.Namespace) -> list[tuple[int, float, int]]:
    if not args.neo4j_uri or not args.neo4j_password or not args.qdrant_url:
        raise SystemExit("NEO4J_URI, NEO4J_PASSWORD, and QDRANT_URL are required")
    try:
        from neo4j import GraphDatabase
        from qdrant_client import QdrantClient, models
    except ImportError as error:
        raise SystemExit("install current neo4j and qdrant-client packages") from error
    graph = GraphDatabase.driver(
        args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password)
    )
    qdrant = QdrantClient(url=args.qdrant_url)
    collection = "engram_phase2"
    qdrant.recreate_collection(
        collection,
        vectors_config=models.VectorParams(
            size=args.dimension, distance=models.Distance.COSINE
        ),
    )
    qdrant.upload_points(
        collection,
        [
            models.PointStruct(
                id=node,
                vector=vector_for(node, args.dimension),
                payload={"assertion_time": node % 10 + 1, "valid_from": 0, "valid_to": 1000},
            )
            for node in range(args.nodes)
        ],
    )
    with graph.session() as session:
        session.run("MATCH (n:EngramPhase2) DETACH DELETE n").consume()
        session.run(
            "UNWIND range(0, $last) AS id CREATE (:EngramPhase2 {id:id})",
            last=args.nodes - 1,
        ).consume()
        session.run(
            """MATCH (n:EngramPhase2)
               UNWIND range(1,10) AS step
               MATCH (m:EngramPhase2 {id:(n.id + step) % $nodes})
               CREATE (n)-[:LINK {weight:1.0, edge_type:1}]->(m)""",
            nodes=args.nodes,
        ).consume()
    rows = []
    try:
        with graph.session() as session:
            for query_id in range(args.queries):
                root = query_id * 7919 % args.nodes
                started = time.perf_counter_ns()
                semantic = qdrant.query_points(
                    collection,
                    query=vector_for(root, args.dimension),
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="assertion_time", range=models.Range(lte=10)
                            ),
                            models.FieldCondition(
                                key="valid_from", range=models.Range(lte=500)
                            ),
                            models.FieldCondition(
                                key="valid_to", range=models.Range(gt=500)
                            ),
                        ]
                    ),
                    score_threshold=0.8,
                    limit=args.nodes,
                ).points
                candidates = [point.id for point in semantic]
                result = session.run(
                    """MATCH (root:EngramPhase2 {id:$root})
                       MATCH (root)-[:LINK*0..3]->(candidate:EngramPhase2)
                       WHERE candidate.id IN $candidates
                       RETURN DISTINCT candidate.id""",
                    root=root,
                    candidates=candidates,
                ).data()
                rows.append(
                    (query_id, (time.perf_counter_ns() - started) / 1000, len(result))
                )
        return rows
    finally:
        graph.close()
        qdrant.delete_collection(collection)


def main() -> None:
    args = arguments()
    if args.stack == "postgresql":
        rows = asyncio.run(postgresql(args))
        product = "postgresql-pgvector"
    else:
        rows = neo4j_qdrant(args)
        product = "neo4j-qdrant"
    write(args.output, product, rows)


if __name__ == "__main__":
    main()
