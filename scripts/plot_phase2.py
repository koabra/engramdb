#!/usr/bin/env python3
"""Render dependency-free SVG plots from Phase 2 benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(index, 0)]


def write_svg(path: Path, body: str, width: int = 860, height: int = 460) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font-family:system-ui,sans-serif;fill:#111827}}.grid{{stroke:#e5e7eb}}.axis{{stroke:#374151;stroke-width:1.5}}</style>
{body}
</svg>""",
        encoding="utf-8",
    )


def latency_plot(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    products = sorted({row["product"] for row in rows})
    summaries = []
    for product in products:
        values = [
            float(row["latency_us"]) for row in rows if row["product"] == product
        ]
        summaries.append(
            (
                product,
                statistics.median(values),
                percentile(values, 0.90),
                percentile(values, 0.99),
            )
        )
    maximum = max(value for summary in summaries for value in summary[1:]) * 1.15
    colors = ("#2563eb", "#dc2626", "#16a34a")
    bars = []
    for product_index, summary in enumerate(summaries):
        product = summary[0]
        for metric_index, (label, value) in enumerate(
            zip(("p50", "p90", "p99"), summary[1:])
        ):
            y = 95 + product_index * 115 + metric_index * 27
            width = value / maximum * 590
            bars.append(
                f'<text x="82" y="{y+16}" text-anchor="end" font-size="12">{product} {label}</text>'
                f'<rect x="95" y="{y}" width="{width:.2f}" height="20" rx="3" fill="{colors[metric_index]}"/>'
                f'<text x="{105+width:.2f}" y="{y+15}" font-size="12">{value:.2f} µs</text>'
            )
    body = (
        '<text x="430" y="34" text-anchor="middle" font-size="20" font-weight="600">'
        "Three-hop tri-modal query latency</text>"
        + "".join(bars)
    )
    write_svg(output_path, body, height=max(280, 130 + len(summaries) * 115))


def recall_plot(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    values = [float(row["recall"]) for row in rows]
    bins = [sum(value >= threshold for value in values) / len(values) for threshold in (0.8, 0.9, 0.95, 1.0)]
    bars = []
    for index, (threshold, fraction) in enumerate(zip((0.8, 0.9, 0.95, 1.0), bins)):
        x = 120 + index * 170
        height = fraction * 300
        bars.append(
            f'<rect x="{x}" y="{380-height:.2f}" width="90" height="{height:.2f}" fill="#2563eb" rx="4"/>'
            f'<text x="{x+45}" y="{400}" text-anchor="middle" font-size="13">≥ {threshold:.2f}</text>'
            f'<text x="{x+45}" y="{365-height:.2f}" text-anchor="middle" font-size="13">{fraction*100:.1f}%</text>'
        )
    body = (
        '<text x="430" y="32" text-anchor="middle" font-size="20" font-weight="600">Recall@10 distribution</text>'
        f'<text x="430" y="58" text-anchor="middle" font-size="13">mean {statistics.mean(values):.4f}; {len(values)} queries</text>'
        '<line class="axis" x1="90" y1="380" x2="800" y2="380"/>'
        + "".join(bars)
    )
    write_svg(output_path, body)


def amplification_plot(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    source_bytes = int(row["source_bytes"])
    encoded_bytes = int(row["encoded_bytes"])
    physical_bytes = int(row["physical_bytes"])
    values = (source_bytes, encoded_bytes, physical_bytes)
    maximum = max(values)
    labels = ("FP32 source", "INT8 encoded", "64 KiB blocks")
    colors = ("#9ca3af", "#16a34a", "#2563eb")
    bars = []
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = 105 + index * 90
        width = value / maximum * 550
        bars.append(
            f'<text x="155" y="{y+25}" text-anchor="end" font-size="13">{label}</text>'
            f'<rect x="170" y="{y}" width="{width:.2f}" height="38" fill="{color}" rx="4"/>'
            f'<text x="{180+width:.2f}" y="{y+25}" font-size="13">{value/1024**3:.2f} GiB</text>'
        )
    body = (
        '<text x="430" y="34" text-anchor="middle" font-size="20" font-weight="600">'
        f'{int(row["nodes"]):,} nodes: storage footprint</text>'
        + "".join(bars)
    )
    write_svg(output_path, body)


def cache_plot(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    maximum = max(int(row["elapsed_ns"]) for row in rows)
    bars = []
    for index, row in enumerate(rows):
        y = 130 + index * 100
        value = int(row["elapsed_ns"])
        width = value / maximum * 550
        bars.append(
            f'<text x="145" y="{y+25}" text-anchor="end" font-size="13">{row["layout"]}</text>'
            f'<rect x="160" y="{y}" width="{width:.2f}" height="38" fill="#2563eb" rx="4"/>'
            f'<text x="{170+width:.2f}" y="{y+25}" font-size="13">{value/1e6:.2f} ms</text>'
        )
    body = (
        '<text x="430" y="34" text-anchor="middle" font-size="20" font-weight="600">'
        "Contiguous fused scan vs pointer layout</text>"
        '<text x="430" y="60" text-anchor="middle" font-size="12">wall-clock proxy; hardware counters reported separately</text>'
        + "".join(bars)
    )
    write_svg(output_path, body, height=390)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("latency", "recall", "amplification", "cache"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    {
        "latency": latency_plot,
        "recall": recall_plot,
        "amplification": amplification_plot,
        "cache": cache_plot,
    }[args.kind](args.input, args.output)


if __name__ == "__main__":
    main()
