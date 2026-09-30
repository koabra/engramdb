#!/usr/bin/env python3
"""Render dependency-free SVG plots for Phase 2 validation metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def svg_page(title: str, body: str, width: int = 800, height: int = 420) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font-family:system-ui,sans-serif;fill:#111827}}.grid{{stroke:#e5e7eb}}</style>
<text x="{width / 2}" y="34" text-anchor="middle" font-size="20" font-weight="600">{title}</text>
{body}
</svg>"""


def recall_plot(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    recall = float(row["recall_at_10"])
    body = f"""
<rect x="120" y="120" width="560" height="54" rx="5" fill="#e5e7eb"/>
<rect x="120" y="120" width="{560 * min(recall, 1):.2f}" height="54" rx="5" fill="#2563eb"/>
<line x1="{120 + 560 * 0.95}" y1="100" x2="{120 + 560 * 0.95}" y2="194" stroke="#dc2626" stroke-width="3"/>
<text x="120" y="215" font-size="13">0.0</text><text x="680" y="215" text-anchor="end" font-size="13">1.0</text>
<text x="400" y="155" text-anchor="middle" font-size="22" font-weight="600">{recall:.4f}</text>
<text x="{120 + 560 * 0.95 - 5}" y="92" text-anchor="end" font-size="12" fill="#dc2626">required &gt; 0.95</text>
<text x="400" y="285" text-anchor="middle" font-size="14">{row['base_vectors']} base vectors · {row['queries']} queries · {row['simd']}</text>
<text x="400" y="315" text-anchor="middle" font-size="13">truth: {row['truth']}</text>"""
    output_path.write_text(svg_page("Quantized HNSW Recall@10", body), encoding="utf-8")


def latency_plot(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    values = [float(row["latency_us"]) for row in rows]
    labels = ["p50", "p90", "p99"]
    samples = [percentile(values, value) for value in (0.5, 0.9, 0.99)]
    maximum = max(samples) * 1.15
    bars = []
    for index, (label, sample) in enumerate(zip(labels, samples)):
        y = 100 + index * 85
        width = sample / maximum * 500
        bars.append(
            f'<text x="90" y="{y + 29}" text-anchor="end" font-size="14">{label}</text>'
            f'<rect x="110" y="{y}" width="500" height="42" rx="4" fill="#e5e7eb"/>'
            f'<rect x="110" y="{y}" width="{width:.2f}" height="42" rx="4" fill="#2563eb"/>'
            f'<text x="625" y="{y + 28}" font-size="14">{sample:.1f} µs</text>'
        )
    body = "\n".join(bars) + (
        f'<text x="400" y="380" text-anchor="middle" font-size="13">'
        f'{len(rows)} three-hop semantic + graph + temporal queries</text>'
    )
    output_path.write_text(svg_page("Tri-modal query latency", body), encoding="utf-8")


def storage_plot(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    sample_gib = int(row["physical_bytes"]) / 2**30
    extrapolated_gib = float(row["extrapolated_10m_bytes"]) / 2**30
    body = f"""
<text x="110" y="110" font-size="14">Measured sample ({int(row['sample_nodes']):,} nodes)</text>
<text x="110" y="155" font-size="34" font-weight="600">{sample_gib:.3f} GiB</text>
<text x="110" y="230" font-size="14">Linear 10M-node projection</text>
<text x="110" y="275" font-size="34" font-weight="600">{extrapolated_gib:.2f} GiB</text>
<text x="110" y="335" font-size="13">{float(row['bytes_per_node']):.1f} bytes/node · {row['dimension']}d · {row['edges']} edges</text>"""
    output_path.write_text(svg_page("Fused-block storage footprint", body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("recall", "latency", "storage"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "recall":
        recall_plot(args.input, args.output)
    elif args.kind == "latency":
        latency_plot(args.input, args.output)
    else:
        storage_plot(args.input, args.output)


if __name__ == "__main__":
    main()
