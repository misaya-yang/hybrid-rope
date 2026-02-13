#!/usr/bin/env python3
"""Generate zero-dependency SVG plot for 50M YaRN comparison."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "results" / "50m_yarn_compare_v2" / "results.json"
OUT_DIR = ROOT / "results" / "figures"
OUT_PATH = OUT_DIR / "50m_yarn_compare_v2.svg"


def scale(v, vmin, vmax, pmin, pmax):
    if vmax == vmin:
        return (pmin + pmax) / 2
    return pmin + (v - vmin) * (pmax - pmin) / (vmax - vmin)


def build_polyline(xs, ys, x_min, x_max, y_min, y_max, left, top, width, height):
    pts = []
    for x, y in zip(xs, ys):
        px = scale(x, x_min, x_max, left, left + width)
        py = scale(y, y_min, y_max, top + height, top)
        pts.append((px, py))
    return pts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with IN_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    exps = data["experiments"]
    lengths = [2048, 4096, 8192, 12288, 16384]
    series = {
        "Hybrid (native)": {
            "color": "#1f77b4",
            "values": [exps["hybrid_native"][str(x)] for x in lengths],
        },
        "Geo (native)": {
            "color": "#2ca02c",
            "values": [exps["geo_native"][str(x)] for x in lengths],
        },
        "Geo + YaRN (progressive)": {
            "color": "#d62728",
            "values": [exps["geo_yarn_progressive"][str(x)] for x in lengths],
        },
    }

    all_y = [v for s in series.values() for v in s["values"]]
    x_min, x_max = min(lengths), max(lengths)
    y_min, y_max = min(all_y) * 0.9, max(all_y) * 1.05

    w, h = 1080, 720
    left, right, top = 90, 40, 60
    pw, ph = w - left - right, h - top - bottom

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append('<text x="90" y="30" font-size="24" font-family="Arial" fill="#111">50M PPL Comparison: Hybrid vs Geo vs Geo+YaRN</text>')

    # grid + y ticks
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5
        py = scale(yv, y_min, y_max, top + ph, top)
        svg.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left+pw}" y2="{py:.2f}" stroke="#e6e6e6"/>')
        svg.append(f'<text x="{left-10}" y="{py+4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#444">{yv:.1f}</text>')

    # x ticks
    for xv in lengths:
        px = scale(xv, x_min, x_max, left, left + pw)
        svg.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top+ph}" stroke="#f0f0f0"/>')
        svg.append(f'<text x="{px:.2f}" y="{top+ph+24}" text-anchor="middle" font-size="12" font-family="Arial" fill="#444">{xv}</text>')

    # axes
    svg.append(f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#111" stroke-width="1.5"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#111" stroke-width="1.5"/>')
    svg.append(f'<text x="{left+pw/2:.2f}" y="{h-28}" text-anchor="middle" font-size="14" font-family="Arial" fill="#111">Evaluation Length</text>')
    svg.append(f'<text x="24" y="{top+ph/2:.2f}" transform="rotate(-90 24 {top+ph/2:.2f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#111">Perplexity (lower is better)</text>')

    # lines
    legend_y = top + 8
    legend_x = left + pw - 280
    offset = 0
    for name, conf in series.items():
        pts = build_polyline(lengths, conf["values"], x_min, x_max, y_min, y_max, left, top, pw, ph)
        poly = " ".join([f"{x:.2f},{y:.2f}" for x, y in pts])
        svg.append(f'<polyline points="{poly}" fill="none" stroke="{conf["color"]}" stroke-width="3"/>')
        for x, y in pts:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{conf["color"]}"/>')
        ly = legend_y + offset
        svg.append(f'<rect x="{legend_x}" y="{ly-10}" width="18" height="4" fill="{conf["color"]}"/>')
        svg.append(f'<text x="{legend_x+26}" y="{ly-6}" font-size="13" font-family="Arial" fill="#111">{name}</text>')
        offset += 24

    svg.append('</svg>')

    OUT_PATH.write_text("\n".join(svg), encoding="utf-8")
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
