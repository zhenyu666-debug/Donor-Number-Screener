"""Step 11: Render Top-20 candidate molecules to a color-graded SVG.

Reads the latest top-20 from `results/top20_candidates_5model.csv`
(falls back to bayes / gridsearch versions if missing) and renders
each structure with a colour gradient based on the predicted DN.

Usage:
  python src/11_pretty_top20_svg.py

Output:
  figures/top20_color_graded.svg
  figures/top20_color_graded.png  (rasterized for the README)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FIGURES_DIR, RESULTS_DIR, get_logger

log = get_logger("top20_svg")


def _find_top_csv():
    for p in ("top20_candidates_5model.csv",
              "top20_candidates_bayes.csv",
              "top20_candidates.csv"):
        f = RESULTS_DIR / p
        if f.exists():
            return f
    raise FileNotFoundError("No top-20 file in results/")


def _dn_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.startswith("dn_pred_stack") or c.startswith("dn_pred_ens"):
            return c
    return df.columns[2]


def _norm_color(dn, vmin, vmax):
    """Map a DN value to (R,G,B) using a yellow→orange→red gradient."""
    t = (dn - vmin) / max(vmax - vmin, 1e-9)
    # 3-stop gradient:  light yellow → orange → deep red
    stops = [(0.96, 0.92, 0.55),  # light yellow
             (0.93, 0.60, 0.20),  # orange
             (0.78, 0.10, 0.10)]  # deep red
    idx = t * (len(stops) - 1)
    lo, hi = int(np.floor(idx)), int(np.ceil(idx))
    f = idx - lo
    rgb = tuple(stops[lo][i] * (1 - f) + stops[hi][i] * f for i in range(3))
    return rgb


def main():
    csv = _find_top_csv()
    log.info("Reading %s", csv)
    df = pd.read_csv(csv)
    dn_col = _dn_col(df)
    log.info("Using DN column: %s", dn_col)

    # Rank by predicted DN (descending)
    df = df.sort_values(dn_col, ascending=False).reset_index(drop=True)
    dns = df[dn_col].values
    vmin, vmax = dns.min(), dns.max()

    # Parse molecules
    mols = []
    valid = []
    for smi in df["smiles"]:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
            valid.append(True)
        else:
            mols.append(None)
            valid.append(False)
    df["valid"] = valid
    n_valid = int(df["valid"].sum())
    log.info("Valid molecules: %d / %d", n_valid, len(df))

    # Build 4x5 grid SVG with colour-coded cards
    n = len(mols)
    cell_w, cell_h = 220, 240
    cols = 4
    rows = (n + cols - 1) // cols
    W = cols * cell_w
    H = rows * cell_h + 60  # top header

    # Draw each molecule to its own SVG then paste in
    from io import StringIO
    from xml.etree import ElementTree as ET

    # Build single composite SVG manually so we can colour each card.
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs>',
        '  <linearGradient id="bg" x1="0" x2="0" y1="0" y2="1">',
        '    <stop offset="0" stop-color="#fafafa"/>',
        '    <stop offset="1" stop-color="#e8ecf1"/>',
        '  </linearGradient>',
        '</defs>',
        f'<rect width="{W}" height="{H}" fill="url(#bg)"/>',
        f'<text x="20" y="36" font-family="Segoe UI, Arial" '
        f'font-size="22" font-weight="700" fill="#1a1a1a">'
        f'Top-{n} high-DN additives '
        f'<tspan fill="#6a6a6a" font-size="14" font-weight="400">'
        f'(latest 5-model stacking run)</tspan></text>',
    ]

    drawer = rdMolDraw2D.MolDraw2DSVG(cell_w - 30, cell_h - 70)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 1.5
    opts.padding = 0.05

    for i, (smi, dn, mol) in enumerate(zip(df["smiles"], dns, mols)):
        r, c = i // cols, i % cols
        x = c * cell_w + 15
        y = r * cell_h + 60

        r_, g_, b_ = _norm_color(dn, vmin, vmax)
        # Card background
        parts.append(
            f'<rect x="{x}" y="{y}" width="{cell_w-30}" height="{cell_h-30}" '
            f'fill="rgb({int(r_*255)},{int(g_*255)},{int(b_*255)})" '
            f'stroke="#444" stroke-width="1" rx="6" ry="6"/>'
        )

        # Molecule (or fallback)
        if mol is not None:
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            inner_svg = drawer.GetDrawingText()
            # Strip outer <svg> tags to embed
            start = inner_svg.find(">") + 1
            end = inner_svg.rfind("</svg>")
            inner = inner_svg[start:end]
            parts.append(
                f'<g transform="translate({x+10},{y+10})" '
                f'fill="white" stroke="black">{inner}</g>'
            )
        else:
            parts.append(
                f'<text x="{x+30}" y="{y+80}" font-family="Arial" '
                f'font-size="12" fill="white">parse error</text>'
            )

        # Rank + SMILES + DN
        parts.append(
            f'<text x="{x+10}" y="{y+cell_h-50}" font-family="Arial" '
            f'font-size="13" font-weight="700" fill="white">'
            f'#{i+1}  DN={dn:.2f}</text>'
        )
        smi_disp = smi if len(smi) <= 28 else smi[:25] + "..."
        parts.append(
            f'<text x="{x+10}" y="{y+cell_h-30}" font-family="Consolas, '
            f'monospace" font-size="11" fill="white" opacity="0.95">'
            f'{smi_disp}</text>'
        )

    # Legend gradient bar
    bar_y = H - 30
    bar_x0, bar_x1 = 20, 220
    parts.append(
        f'<rect x="{bar_x0}" y="{bar_y}" width="{bar_x1-bar_x0}" '
        f'height="10" fill="url(#bg)"/>'
    )
    # Simple gradient via 30 rects
    n_steps = 30
    for k in range(n_steps):
        t = k / (n_steps - 1)
        r_, g_, b_ = _norm_color(vmin + t * (vmax - vmin), vmin, vmax)
        parts.append(
            f'<rect x="{bar_x0 + (bar_x1-bar_x0)*k/n_steps}" y="{bar_y}" '
            f'width="{(bar_x1-bar_x0)/n_steps+0.5}" height="10" '
            f'fill="rgb({int(r_*255)},{int(g_*255)},{int(b_*255)})"/>'
        )
    parts.append(
        f'<text x="{bar_x0}" y="{bar_y-3}" font-family="Arial" '
        f'font-size="10" fill="#6a6a6a">DN {vmin:.1f}</text>'
    )
    parts.append(
        f'<text x="{bar_x1-30}" y="{bar_y-3}" font-family="Arial" '
        f'font-size="10" fill="#6a6a6a">DN {vmax:.1f}</text>'
    )

    parts.append('</svg>')
    svg = "\n".join(parts)

    svg_path = FIGURES_DIR / "top20_color_graded.svg"
    svg_path.write_text(svg, encoding="utf-8")
    log.info("Wrote %s", svg_path)

    # Rasterized PNG version (smaller, for README)
    png_path = FIGURES_DIR / "top20_color_graded.png"
    try:
        # Use cairosvg if available, otherwise skip
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                         write_to=str(png_path), output_width=W)
        log.info("Wrote %s", png_path)
    except ImportError:
        # Fall back: matplotlib can't render SVG cleanly; instead
        # produce a PNG via matplotlib by stacking molecule images
        log.warning("cairosvg not available; producing PNG via matplotlib")
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
        axes = axes.flat if rows > 1 else [axes]
        for ax in axes:
            ax.set_axis_off()
        for i, (smi, dn, mol) in enumerate(zip(df["smiles"], dns, mols)):
            ax = axes[i]
            r_, g_, b_ = _norm_color(dn, vmin, vmax)
            ax.set_facecolor((r_, g_, b_))
            if mol is not None:
                img = Draw.MolToImage(mol, size=(200, 180))
                ax.imshow(img)
            ax.set_title(f"#{i+1}  DN={dn:.2f}\n{smi[:25]}",
                         fontsize=8, color="white", weight="bold")
        fig.tight_layout()
        fig.savefig(png_path, dpi=140, bbox_inches="tight",
                    facecolor="#fafafa")
        plt.close(fig)
        log.info("Wrote %s (matplotlib)", png_path)

    print(f"\nSVG: {svg_path}")
    print(f"PNG: {png_path}")
    print(f"Range: DN {vmin:.2f} - {vmax:.2f}")


if __name__ == "__main__":
    main()
