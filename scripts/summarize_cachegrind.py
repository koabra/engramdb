#!/usr/bin/env python3
"""Extract query-kernel cache misses and render a comparison SVG."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def kernel_values(path: Path, function: str) -> list[int]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if function not in line or line.lstrip().startswith(">"):
            continue
        values = []
        for token in line.split(function, 1)[0].split():
            if token[0].isdigit():
                values.append(int(token.replace(",", "")))
        if len(values) >= 9:
            return values[:9]
    raise SystemExit(f"query kernel {function} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    fused = kernel_values(args.fused, "phase2_bench::profile_fused_queries")
    split = kernel_values(args.split, "phase2_bench::profile_split_queries")
    rows = [
        ("fused", fused[3] + fused[6], fused[4] + fused[7], fused[5] + fused[8]),
        ("split", split[3] + split[6], split[4] + split[7], split[5] + split[8]),
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["layout", "data_references", "l1_data_misses", "l2_data_misses"])
        writer.writerows(rows)

    maximum = max(value for _, _, d1, ll in rows for value in (d1, ll))
    bars = []
    colors = {"fused": "#2563eb", "split": "#dc2626"}
    for group, (layout, _, d1, ll) in enumerate(rows):
        for item, (label, value) in enumerate((("L1D", d1), ("L2", ll))):
            y = 100 + group * 125 + item * 48
            width = value / maximum * 480
            bars.append(
                f'<text x="95" y="{y+24}" text-anchor="end" font-size="13">{layout} {label}</text>'
                f'<rect x="110" y="{y}" width="{width:.2f}" height="32" rx="3" fill="{colors[layout]}"/>'
                f'<text x="{120+width:.2f}" y="{y+23}" font-size="12">{value:,}</text>'
            )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="420" viewBox="0 0 800 420">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font-family:system-ui,sans-serif;fill:#111827}}</style>
<text x="400" y="34" text-anchor="middle" font-size="20" font-weight="600">Query-kernel cache misses</text>
{''.join(bars)}
<text x="400" y="385" text-anchor="middle" font-size="12">Cachegrind: 48 KiB L1D, 2 MiB L2, 64-byte lines</text>
</svg>"""
    args.svg.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
