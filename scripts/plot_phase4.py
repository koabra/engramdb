#!/usr/bin/env python3
"""Render dependency-free SVG plots for Phase 4 software and hardware tiers."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def write_svg(path: Path, body: str, height: int = 430) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="860" height="{height}" viewBox="0 0 860 {height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font-family:system-ui,sans-serif;fill:#111827}}.axis{{stroke:#374151;stroke-width:1.5}}.grid{{stroke:#e5e7eb}}</style>
{body}
</svg>""",
        encoding="utf-8",
    )


def bandwidth(source: Path, output: Path) -> None:
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    maximum = max(float(row["gb_per_s"]) for row in rows) * 1.15
    bars = "".join(
        f'<text x="150" y="{130+i*95}" text-anchor="end" font-size="13">{row["operation"]} ({row["backend"]})</text>'
        f'<rect x="165" y="{105+i*95}" width="{float(row["gb_per_s"])/maximum*520:.2f}" height="38" fill="#2563eb" rx="4"/>'
        f'<text x="{175+float(row["gb_per_s"])/maximum*520:.2f}" y="{130+i*95}" font-size="12">{float(row["gb_per_s"]):.3f} GB/s</text>'
        for i, row in enumerate(rows)
    )
    write_svg(
        output,
        '<text x="430" y="32" text-anchor="middle" font-size="20" font-weight="600">KV-cache transfer bandwidth</text>'
        '<text x="430" y="58" text-anchor="middle" font-size="12">CPU direct-I/O fallback; not GPU/GDS</text>'
        + bars,
        360,
    )


def ttft(source: Path, output: Path) -> None:
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    modes = sorted({row["mode"] for row in rows})
    colors = ("#dc2626", "#16a34a", "#2563eb")
    tokens = sorted({int(row["tokens"]) for row in rows})
    maximum = max(float(row["elapsed_ms"]) for row in rows)

    def x(value: int) -> float:
        return 100 + math.log10(value / min(tokens)) / math.log10(max(tokens) / min(tokens)) * 650

    def y(value: float) -> float:
        return 360 - value / maximum * 270

    lines = []
    for index, mode in enumerate(modes):
        selected = sorted(
            (int(row["tokens"]), float(row["elapsed_ms"]))
            for row in rows
            if row["mode"] == mode
        )
        points = " ".join(f"{x(token):.2f},{y(value):.2f}" for token, value in selected)
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[index]}" stroke-width="3"/>'
            f'<text x="610" y="{75+index*18}" font-size="12" fill="{colors[index]}">{mode}</text>'
        )
    labels = "".join(
        f'<text x="{x(token):.2f}" y="390" text-anchor="middle" font-size="12">{token:,}</text>'
        for token in tokens
    )
    write_svg(
        output,
        '<text x="430" y="32" text-anchor="middle" font-size="20" font-weight="600">TTFT component simulation</text>'
        '<text x="430" y="55" text-anchor="middle" font-size="12">CPU software only; no GPU/GDS TTFT measured</text>'
        '<line class="axis" x1="80" y1="360" x2="780" y2="360"/>'
        + "".join(lines)
        + labels,
    )


def integrity(source: Path, output: Path) -> None:
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    items = "".join(
        f'<circle cx="125" cy="{100+i*55}" r="11" fill="{"#16a34a" if row["passed"]=="True" else "#9ca3af"}"/>'
        f'<text x="150" y="{106+i*55}" font-size="14">{row["tier"]}: {"pass" if row["passed"]=="True" else "not run / unavailable"}</text>'
        for i, row in enumerate(rows)
    )
    write_svg(
        output,
        '<text x="430" y="32" text-anchor="middle" font-size="20" font-weight="600">KV-cache integrity tiers</text>'
        + items,
        max(360, 140 + len(rows) * 55),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("bandwidth", "ttft", "integrity"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    {"bandwidth": bandwidth, "ttft": ttft, "integrity": integrity}[args.kind](
        args.input, args.output
    )


if __name__ == "__main__":
    main()
