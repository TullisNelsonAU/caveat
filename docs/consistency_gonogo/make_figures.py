#!/usr/bin/env python3
"""Render the two go/no-go figures from the consistency outputs — pure stdlib, emits SVG so there
is no matplotlib/numpy dependency.

  fig_scatter.svg : GT-free S (mean cavity surprise, log y) vs true post-hoc ECE, one point per
                    binary, colored by group. This is the headline: does S track the drift it
                    cannot see?
  fig_strip.svg   : the residual-cluster strip for one obfuscated binary — per-address surprise
                    outliers along .text, showing they line up in contiguous runs (the spatial
                    signal) rather than scattering as a well-specified model would.

usage: make_figures.py results.csv strip.csv out_dir
"""
import csv
import sys
import math

GROUP_COLOR = {
    "clean_fit": "#4c78a8",
    "clean_holdout": "#54a24b",
    "desync": "#e45756",
    "packed": "#b279a2",
}


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def scatter_svg(rows):
    W, H, PAD = 720, 460, 70
    pts = []
    for r in rows:
        ece = float(r["ece_calibrated"])
        s = float(r["s_glob_surprise"])
        pts.append((ece, s, r["role"]))
    xs = [p[0] for p in pts]
    ys = [max(p[1], 1e-3) for p in pts]
    xmin, xmax = 0.0, max(xs) * 1.05 + 1e-6
    ymin, ymax = math.log10(min(ys) * 0.7), math.log10(max(ys) * 1.4)

    def px(x):
        return PAD + (x - xmin) / (xmax - xmin) * (W - 2 * PAD)

    def py(y):
        ly = math.log10(max(y, 1e-3))
        return H - PAD - (ly - ymin) / (ymax - ymin) * (H - 2 * PAD)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    # axes
    s.append(f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#333"/>')
    s.append(f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#333"/>')
    # x ticks
    for i in range(6):
        x = xmin + (xmax - xmin) * i / 5
        s.append(f'<line x1="{px(x):.1f}" y1="{H-PAD}" x2="{px(x):.1f}" y2="{H-PAD+5}" stroke="#333"/>')
        s.append(f'<text x="{px(x):.1f}" y="{H-PAD+20}" font-size="11" text-anchor="middle">{x:.3f}</text>')
    # y ticks (log decades)
    lo, hi = math.floor(ymin), math.ceil(ymax)
    for d in range(lo, hi + 1):
        yy = 10 ** d
        s.append(f'<line x1="{PAD-5}" y1="{py(yy):.1f}" x2="{PAD}" y2="{py(yy):.1f}" stroke="#333"/>')
        s.append(f'<text x="{PAD-10}" y="{py(yy)+4:.1f}" font-size="11" text-anchor="end">{yy:g}</text>')
    # points
    for ece, y, role in pts:
        s.append(f'<circle cx="{px(ece):.1f}" cy="{py(y):.1f}" r="5" fill="{GROUP_COLOR.get(role,"#888")}" fill-opacity="0.75" stroke="#222" stroke-width="0.5"/>')
    # labels
    s.append(f'<text x="{W/2:.0f}" y="{H-20}" font-size="13" text-anchor="middle">true post-hoc ECE (clean-fit map, needs GT)</text>')
    s.append(f'<text x="20" y="{H/2:.0f}" font-size="13" text-anchor="middle" transform="rotate(-90 20 {H/2:.0f})">GT-free S_glob = mean cavity surprise (log)</text>')
    s.append(f'<text x="{W/2:.0f}" y="26" font-size="15" font-weight="bold" text-anchor="middle">Consistency surprise tracks the drift it cannot see</text>')
    # legend
    ly = PAD + 6
    for role, col in GROUP_COLOR.items():
        if any(p[2] == role for p in pts):
            s.append(f'<circle cx="{W-PAD-110}" cy="{ly}" r="5" fill="{col}"/>')
            s.append(f'<text x="{W-PAD-100}" y="{ly+4}" font-size="11">{role}</text>')
            ly += 18
    s.append("</svg>")
    return "\n".join(s)


def strip_svg(rows, title):
    # Order by offset; draw a thin vertical tick per address, red if a surprise outlier ("event").
    rows = sorted(rows, key=lambda r: int(r["offset"]))
    n = len(rows)
    W, H, PAD = 900, 150, 40
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{W/2:.0f}" y="22" font-size="14" font-weight="bold" text-anchor="middle">{title}</text>')
    s.append(f'<text x="{W/2:.0f}" y="38" font-size="11" text-anchor="middle" fill="#555">each column = one candidate address in .text order; red = within-binary surprise outlier (&gt;μ+2σ)</text>')
    y0, y1 = 50, H - 30
    plot_w = W - 2 * PAD
    for i, r in enumerate(rows):
        if r["event"] == "1":
            x = PAD + (i / max(n - 1, 1)) * plot_w
            s.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y1}" stroke="#e45756" stroke-width="1" stroke-opacity="0.55"/>')
    s.append(f'<line x1="{PAD}" y1="{y1}" x2="{W-PAD}" y2="{y1}" stroke="#333"/>')
    s.append(f'<text x="{PAD}" y="{H-10}" font-size="11">.text start</text>')
    s.append(f'<text x="{W-PAD}" y="{H-10}" font-size="11" text-anchor="end">.text end</text>')
    s.append("</svg>")
    return "\n".join(s)


def main():
    results, strip, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = load_rows(results)
    with open(f"{outdir}/fig_scatter.svg", "w") as f:
        f.write(scatter_svg(rows))
    srows = load_rows(strip)
    name = strip.split("/")[-1].replace(".csv", "")
    with open(f"{outdir}/fig_strip.svg", "w") as f:
        f.write(strip_svg(srows, f"Residual-cluster strip — {name}"))
    print(f"wrote {outdir}/fig_scatter.svg and {outdir}/fig_strip.svg")


if __name__ == "__main__":
    main()
