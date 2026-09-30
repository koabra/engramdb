#!/usr/bin/env python3
"""Render dependency-free SVG plots from Phase 3 CSV metrics."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def svg(path: Path, body: str, height: int = 440) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="860" height="{height}" viewBox="0 0 860 {height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font-family:system-ui,sans-serif;fill:#111827}}.axis{{stroke:#374151;stroke-width:1.5}}.grid{{stroke:#e5e7eb}}</style>
{body}
</svg>""",
        encoding="utf-8",
    )


def ingestion(source: Path, output: Path) -> None:
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    maximum = max(float(row["wall_s"]) for row in rows) * 1.15
    bars = []
    for index, row in enumerate(rows):
        for metric, color, offset in (
            ("client_cpu_s", "#2563eb", 0),
            ("wall_s", "#16a34a", 34),
        ):
            value = float(row[metric])
            y = 100 + index * 105 + offset
            bars.append(
                f'<text x="165" y="{y+20}" text-anchor="end" font-size="12">{row["product"]} {metric}</text>'
                f'<rect x="175" y="{y}" width="{value/maximum*540:.2f}" height="26" fill="{color}" rx="3"/>'
                f'<text x="{185+value/maximum*540:.2f}" y="{y+19}" font-size="12">{value:.4f}s</text>'
            )
    svg(
        output,
        '<text x="430" y="32" text-anchor="middle" font-size="20" font-weight="600">5,000-document ingestion</text>'
        + "".join(bars),
        max(300, 150 + len(rows) * 105),
    )


def sessions(source: Path, output: Path) -> None:
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    maximum = max(int(row["rss_bytes"]) for row in rows)
    points = []
    labels = []
    for index, row in enumerate(rows):
        x = 100 + index * (650 / max(len(rows) - 1, 1))
        y = 370 - int(row["rss_bytes"]) / maximum * 280
        points.append(f"{x:.2f},{y:.2f}")
        labels.append(
            f'<text x="{x:.2f}" y="400" text-anchor="middle" font-size="12">{int(row["sessions"]):,}</text>'
            f'<text x="{x:.2f}" y="{y-10:.2f}" text-anchor="middle" font-size="11">{int(row["rss_bytes"])/1024**2:.1f} MiB</text>'
        )
    body = (
        '<text x="430" y="32" text-anchor="middle" font-size="20" font-weight="600">Durable speculative sessions</text>'
        '<line class="axis" x1="80" y1="370" x2="780" y2="370"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#2563eb" stroke-width="3"/>'
        + "".join(labels)
    )
    svg(output, body)


def zero_copy(source: Path, output: Path) -> None:
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    maximum = max(
        max(int(row["arrow_bytes"]), int(row["python_peak_bytes"])) for row in rows
    )
    groups = []
    for index, row in enumerate(rows):
        x = 100 + index * 180
        arrow_height = int(row["arrow_bytes"]) / maximum * 280
        python_height = int(row["python_peak_bytes"]) / maximum * 280
        groups.append(
            f'<rect x="{x}" y="{370-arrow_height:.2f}" width="55" height="{arrow_height:.2f}" fill="#16a34a"/>'
            f'<rect x="{x+62}" y="{370-python_height:.2f}" width="55" height="{python_height:.2f}" fill="#dc2626"/>'
            f'<text x="{x+58}" y="398" text-anchor="middle" font-size="12">{row["rows"]} rows</text>'
        )
    body = (
        '<text x="430" y="32" text-anchor="middle" font-size="20" font-weight="600">Arrow payload vs traced Python heap</text>'
        '<text x="650" y="60" font-size="12" fill="#16a34a">green: Arrow bytes</text>'
        '<text x="650" y="78" font-size="12" fill="#dc2626">red: Python peak</text>'
        '<line class="axis" x1="70" y1="370" x2="800" y2="370"/>'
        + "".join(groups)
    )
    svg(output, body)


def latency(source: Path, output: Path) -> None:
    values = sorted(
        float(row["latency_us"])
        for row in csv.DictReader(source.open(encoding="utf-8"))
    )

    def percentile(fraction: float) -> float:
        return values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]

    summary = (
        ("p50", statistics.median(values)),
        ("p90", percentile(0.9)),
        ("p99", percentile(0.99)),
    )
    maximum = max(value for _, value in summary) * 1.15
    bars = "".join(
        f'<text x="130" y="{120+i*80}" text-anchor="end" font-size="13">{name}</text>'
        f'<rect x="145" y="{98+i*80}" width="{value/maximum*550:.2f}" height="32" fill="#2563eb" rx="4"/>'
        f'<text x="{155+value/maximum*550:.2f}" y="{120+i*80}" font-size="12">{value:.1f} µs</text>'
        for i, (name, value) in enumerate(summary)
    )
    svg(
        output,
        '<text x="430" y="32" text-anchor="middle" font-size="20" font-weight="600">Arrow Flight query latency</text>'
        + bars,
        380,
    )


def planner(source: Path, output: Path) -> None:
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    maximum = max(float(row["plans_per_s"]) for row in rows)
    bars = "".join(
        f'<text x="145" y="{130+i*95}" text-anchor="end" font-size="13">{row["plan"]}</text>'
        f'<rect x="160" y="{105+i*95}" width="{float(row["plans_per_s"])/maximum*540:.2f}" height="38" fill="#7c3aed" rx="4"/>'
        f'<text x="{170+float(row["plans_per_s"])/maximum*540:.2f}" y="{130+i*95}" font-size="12">{float(row["plans_per_s"]):,.0f}/s</text>'
        for i, row in enumerate(rows)
    )
    svg(
        output,
        '<text x="430" y="32" text-anchor="middle" font-size="20" font-weight="600">EnQL parse + optimize throughput</text>'
        + bars,
        350,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind", choices=("ingestion", "sessions", "zero-copy", "latency", "planner")
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    {
        "ingestion": ingestion,
        "sessions": sessions,
        "zero-copy": zero_copy,
        "latency": latency,
        "planner": planner,
    }[args.kind](args.input, args.output)


if __name__ == "__main__":
    main()
