#!/usr/bin/env python3
"""CPU integrity tiers for KV-cache Flight and inference adapters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

from engramdb import EngramClient, SglangKvCacheAdapter, VllmKvCacheAdapter


def attention(query: list[float], keys: list[list[float]], values: list[list[float]]) -> list[float]:
    scores = [sum(a * b for a, b in zip(query, key)) for key in keys]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denominator = sum(weights)
    return [
        sum(weight * value[column] for weight, value in zip(weights, values))
        / denominator
        for column in range(len(query))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="grpc://127.0.0.1:50051")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = EngramClient(args.location)
    session = client.fork_session(client.main_branch())
    keys = [[0.25, -0.5, 0.75, 1.0], [1.0, 0.5, -0.25, 0.125]]
    values = [[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, 0.5, 0.25]]
    cache = b"".join(
        struct.pack("<f", value)
        for tensor in (keys, values)
        for row in tensor
        for value in row
    )
    fingerprint = list(hashlib.blake2s(b"tiny-llama-test", digest_size=32).digest())
    spec = {
        "model_fingerprint": fingerprint,
        "dtype": "Fp32",
        "layout": "VllmPagedV1",
        "num_layers": 1,
        "num_blocks": 1,
        "num_kv_heads": 1,
        "head_size": 4,
        "block_size": 2,
        "sequence_length": 2,
        "tensor_parallel_rank": 0,
        "tensor_parallel_world": 1,
    }
    vllm = VllmKvCacheAdapter(client)
    vllm.store(session, spec, cache)
    restored, restored_spec = vllm.restore_bytes(session)
    restored_bytes = restored.tobytes()
    byte_equal = restored_bytes == cache and restored_spec == spec
    restored_floats = struct.unpack("<16f", restored_bytes)
    restored_keys = [list(restored_floats[:4]), list(restored_floats[4:8])]
    restored_values = [list(restored_floats[8:12]), list(restored_floats[12:16])]
    query = [0.5, -0.25, 1.0, 0.75]
    expected = attention(query, keys, values)
    actual = attention(query, restored_keys, restored_values)
    logits_equal = b"".join(struct.pack("<f", value) for value in expected) == b"".join(
        struct.pack("<f", value) for value in actual
    )

    sglang = SglangKvCacheAdapter(client)
    destination = bytearray(len(cache))
    sglang_spec = sglang.restore_radix_prefix(session, destination)
    adapter_equal = bytes(destination) == cache and sglang_spec == spec
    ticket = vllm.native_restore_ticket(session)
    capabilities = client.hardware_capabilities()
    (args.output_dir / "capability.json").write_text(
        json.dumps(capabilities, indent=2), encoding="utf-8"
    )
    (args.output_dir / "restore-ticket.json").write_text(
        json.dumps(ticket, indent=2), encoding="utf-8"
    )
    rows = [
        {"tier": "T0-byte-roundtrip", "passed": byte_equal, "hardware": False},
        {"tier": "T1-software-attention", "passed": logits_equal, "hardware": False},
        {"tier": "adapter-vllm-sglang", "passed": adapter_equal, "hardware": False},
        {
            "tier": "T2-vllm-runtime",
            "passed": False,
            "hardware": False,
        },
        {
            "tier": "T3-gpu-gds",
            "passed": capabilities["transfer_path"] == "GdsDirectVerified",
            "hardware": True,
        },
    ]
    with (args.output_dir / "integrity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=("tier", "passed", "hardware"))
        writer.writeheader()
        writer.writerows(rows)
    if not byte_equal or not logits_equal or not adapter_equal:
        raise SystemExit("CPU KV-cache integrity failed")
    client.commit_session(session)


if __name__ == "__main__":
    main()
