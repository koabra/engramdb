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


def query_vector_for(node_id: int, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    vector[node_id % min(dimension, 32)] = 1.0
    return vector


def graph_target(node_id: int, step: int, nodes: int) -> int:
    mixed = (
        node_id * 6_364_136_223_846_793_005
        + step * 1_442_695_040_888_963_407
    ) & ((1 << 64) - 1)
    return mixed % nodes


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


def write(
    path: Path,
    product: str,
    rows: list[tuple[int, float, int]],
    nodes: int,
    dimension: int,
) -> None:
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
            writer.writerow(
                [product, query, f"{latency:.3f}", hits, "", "", nodes, dimension, 3]
            )


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
                id bigint NOT NULL,
                assertion_time bigint NOT NULL,
                valid_from bigint NOT NULL,
                valid_to bigint NOT NULL,
                embedding vector({args.dimension}) NOT NULL,
                PRIMARY KEY(id, assertion_time)
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
            ]
            + [
                (node, 20, str(vector_for(node + 1, args.dimension)))
                for node in range(0, args.nodes, 10)
            ],
        )
        await connection.executemany(
            "INSERT INTO engram_phase2_edges VALUES($1, $2, 1.0, 1)",
            [
                (node, graph_target(node, step, args.nodes))
                for node in range(args.nodes)
                for step in range(1, 11)
            ],
        )
        await connection.execute("CREATE INDEX ON engram_phase2_edges(source)")
        rows = []
        statement = await connection.prepare(
            """WITH RECURSIVE latest_nodes AS (
                 SELECT DISTINCT ON (id) *
                 FROM engram_phase2_nodes
                 WHERE assertion_time <= 10
                 ORDER BY id, assertion_time DESC
               ), reachable(id, depth) AS (
                 SELECT $1::bigint, 0
                 UNION
                 SELECT e.target, r.depth + 1
                 FROM reachable r JOIN engram_phase2_edges e ON e.source = r.id
                 WHERE r.depth < 3
               )
               SELECT DISTINCT n.id
               FROM reachable r JOIN latest_nodes n ON n.id = r.id
               WHERE n.valid_from <= 500 AND 500 < n.valid_to
                 AND 1 - (n.embedding <=> $2::vector) >= 0.8"""
        )
        for query_id in range(args.queries):
            root = query_id * 7919 % args.nodes
            started = time.perf_counter_ns()
            result = await statement.fetch(root, str(query_vector_for(root, args.dimension)))
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
    points = [
        models.PointStruct(
            id=node * 2,
            vector=vector_for(node, args.dimension),
            payload={
                "logical_id": node,
                "assertion_time": node % 10 + 1,
                "valid_from": 0,
                "valid_to": 1000,
            },
        )
        for node in range(args.nodes)
    ]
    points.extend(
        models.PointStruct(
            id=node * 2 + 1,
            vector=vector_for(node + 1, args.dimension),
            payload={
                "logical_id": node,
                "assertion_time": 20,
                "valid_from": 0,
                "valid_to": 1000,
            },
        )
        for node in range(0, args.nodes, 10)
    )
    qdrant.upload_points(
        collection,
        points,
    )
    with graph.session() as session:
        session.run("MATCH (n:EngramPhase2) DETACH DELETE n").consume()
        session.run(
            "UNWIND range(0, $last) AS id CREATE (:EngramPhase2 {id:id})",
            last=args.nodes - 1,
        ).consume()
        edges = [
            {"source": node, "target": graph_target(node, step, args.nodes)}
            for node in range(args.nodes)
            for step in range(1, 11)
        ]
        for start in range(0, len(edges), 10_000):
            session.run(
                """UNWIND $edges AS edge
                   MATCH (n:EngramPhase2 {id:edge.source})
                   MATCH (m:EngramPhase2 {id:edge.target})
                   CREATE (n)-[:LINK {weight:1.0, edge_type:1}]->(m)""",
                edges=edges[start : start + 10_000],
            ).consume()
    rows = []
    try:
        with graph.session() as session:
            for query_id in range(args.queries):
                root = query_id * 7919 % args.nodes
                started = time.perf_counter_ns()
                semantic = qdrant.query_points(
                    collection,
                    query=query_vector_for(root, args.dimension),
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
                candidates = list(
                    {
                        point.payload["logical_id"]
                        for point in semantic
                        if point.payload is not None
                    }
                )
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
    write(args.output, product, rows, args.nodes, args.dimension)


if __name__ == "__main__":
    main()
