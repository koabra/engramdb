#!/usr/bin/env python3
"""Render dependency-free SVG plots from Phase 1 benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


COLORS = {
    "engramdb": "#2563eb",
    "postgresql": "#dc2626",
    "mongodb": "#16a34a",
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(index, 0)]


def branch_plot(input_path: Path, output_path: Path) -> None:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    with input_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            grouped[(row["product"], int(row["concurrency"]))].append(
                float(row["latency_us"])
            )
    points = {
        key: (statistics.median(values), percentile(values, 0.95))
        for key, values in grouped.items()
    }
    if not points:
        raise SystemExit("branch CSV contains no observations")

    width, height = 900, 520
    left, right, top, bottom = 90, 35, 55, 75
    plot_width, plot_height = width - left - right, height - top - bottom
    all_concurrency = sorted({key[1] for key in points})
    all_latency = [latency for pair in points.values() for latency in pair]
    min_x, max_x = min(all_concurrency), max(all_concurrency)
    min_y = max(min(all_latency) * 0.8, 1)
    max_y = max(all_latency) * 1.25

    def x(value: float) -> float:
        if min_x == max_x:
            return left + plot_width / 2
        return left + math.log10(value / min_x) / math.log10(max_x / min_x) * plot_width

    def y(value: float) -> float:
        if min_y == max_y:
            return top + plot_height / 2
        return top + (1 - math.log10(value / min_y) / math.log10(max_y / min_y)) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#111827}.grid{stroke:#e5e7eb}.axis{stroke:#374151;stroke-width:1.5}.p95{stroke-dasharray:6 5}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-weight="600">Durable branch creation latency</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}"/>',
    ]
    for concurrency in all_concurrency:
        position = x(concurrency)
        svg.extend(
            [
                f'<line class="grid" x1="{position:.2f}" y1="{top}" x2="{position:.2f}" y2="{top+plot_height}"/>',
                f'<text x="{position:.2f}" y="{top+plot_height+24}" text-anchor="middle" font-size="12">{concurrency}</text>',
            ]
        )
    ticks = 6
    for index in range(ticks):
        exponent = math.log10(min_y) + index / (ticks - 1) * math.log10(max_y / min_y)
        value = 10**exponent
        position = y(value)
        svg.extend(
            [
                f'<line class="grid" x1="{left}" y1="{position:.2f}" x2="{left+plot_width}" y2="{position:.2f}"/>',
                f'<text x="{left-10}" y="{position+4:.2f}" text-anchor="end" font-size="12">{value:.0f}</text>',
            ]
        )

    products = sorted({key[0] for key in points})
    for product_index, product in enumerate(products):
        color = COLORS.get(product, "#7c3aed")
        samples = sorted(
            (concurrency, *points[(name, concurrency)])
            for name, concurrency in points
            if name == product
        )
        for value_index, (label, css_class) in enumerate((("p50", ""), ("p95", " p95"))):
            coordinates = " ".join(
                f"{x(concurrency):.2f},{y(sample[1+value_index]):.2f}"
                for sample in samples
                for concurrency in [sample[0]]
            )
            svg.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5" class="{css_class.strip()}"/>'
            )
            for sample in samples:
                svg.append(
                    f'<circle cx="{x(sample[0]):.2f}" cy="{y(sample[1+value_index]):.2f}" r="3.5" fill="{color}"/>'
                )
            legend_y = 62 + product_index * 42 + value_index * 17
            svg.extend(
                [
                    f'<line x1="{width-185}" y1="{legend_y}" x2="{width-155}" y2="{legend_y}" stroke="{color}" stroke-width="2.5" class="{css_class.strip()}"/>',
                    f'<text x="{width-147}" y="{legend_y+4}" font-size="12">{product} {label}</text>',
                ]
            )
    svg.extend(
        [
            f'<text x="{left+plot_width/2}" y="{height-20}" text-anchor="middle" font-size="14">Concurrent requests (log scale)</text>',
            f'<text x="20" y="{top+plot_height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {top+plot_height/2})">Latency (µs, log scale)</text>',
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_plot(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    amplification = float(row["write_amplification"])
    ops = float(row["ops_per_s"])
    width, height = 760, 360
    maximum = max(amplification, 1.0) * 1.15
    bar_width = amplification / maximum * 500
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font-family:system-ui,sans-serif;fill:#111827}}</style>
<text x="380" y="34" text-anchor="middle" font-size="20" font-weight="600">1 KiB micro-write result</text>
<text x="100" y="104" font-size="14">Physical / logical bytes</text>
<rect x="100" y="125" width="500" height="40" fill="#e5e7eb" rx="4"/>
<rect x="100" y="125" width="{bar_width:.2f}" height="40" fill="#2563eb" rx="4"/>
<text x="620" y="151" font-size="16" font-weight="600">{amplification:.2f}×</text>
<text x="100" y="220" font-size="14">Sustained committed writes</text>
<text x="100" y="260" font-size="36" font-weight="600">{ops:,.1f} ops/s</text>
<text x="100" y="310" font-size="12" fill="#4b5563">{row['logical_workers']} logical workers on {row['physical_threads']} physical threads</text>
</svg>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("branch", "write"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.kind == "branch":
        branch_plot(args.input, args.output)
    else:
        write_plot(args.input, args.output)


if __name__ == "__main__":
    main()
