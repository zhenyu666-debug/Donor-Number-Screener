"""p42b_viz_tea.py - Visualization for TEA/LCA comparison.

Generates bar plots comparing DEER vs alternatives on:
  1. Manufacturing cost ($/kg)
  2. Energy consumption (kWh/kg)
  3. GHG emissions (kg CO2-eq / kg)
  4. Capacity recovery (%)
  5. Radar chart: overall multi-dimensional comparison

Requires: matplotlib

Outputs:
  figures/deer_tea_comparison.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
from utils import get_logger, RESULTS_DIR, FIGURES_DIR  # noqa: E402

log = get_logger("p42b_viz_tea")


def load_tea_data() -> list:
    import pandas as pd
    path = RESULTS_DIR / "deer_tea_lca.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run p42_tea_lca.py first: {path}")
    return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip").to_dict("records")


def bar_chart(data: list, metric_col: str, title: str, ylabel: str,
              filename: str, invert: bool = False) -> Path:
    """Simple horizontal bar chart. invert=True means lower is better."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib unavailable — skipping %s", title)
        return Path("")

    names  = [r["pathway"] for r in data]
    values = [r[metric_col] for r in data]
    colors = [r["color"] for r in data]

    if invert:
        # Sort ascending (best = lowest)
        order = sorted(range(len(values)), key=lambda i: values[i])
    else:
        order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)

    names  = [names[i]  for i in order]
    values = [values[i]  for i in order]
    colors = [colors[i]  for i in order]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)

    # Highlight DEER
    for i, (bar, name) in enumerate(zip(bars, names)):
        if name == "DEER":
            bar.set_edgecolor("black")
            bar.set_linewidth(2)

    ax.set_xlabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    if invert:
        ax.axvline(x=values[-1], color="green", linestyle="--", alpha=0.5,
                   label="best")
    fig.tight_layout()

    out = FIGURES_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


def radar_chart(data: list) -> Path:
    """Radar chart comparing all pathways across normalized metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib unavailable — skipping radar chart")
        return Path("")

    # Normalize metrics: higher is better for all (invert cost/energy/ghg)
    pathways = [r["pathway"] for r in data]
    metrics = {
        "cost":       [max(r["cost_usd_kg"] for r in data) / r["cost_usd_kg"] for r in data],
        "energy":     [max(r["energy_kwh_kg"] for r in data) / max(r["energy_kwh_kg"], 0.01)
                       for r in data],
        "ghg":        [max(r["ghg_kg_co2_kg"] for r in data) /
                       max(r["ghg_kg_co2_kg"], 0.01) for r in data],
        "capacity":   [r["capacity_recovery_pct"] / 100.0 for r in data],
    }

    categories = list(metrics.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, pathway in enumerate(pathways):
        values = [metrics[cat][i] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, color=data[i]["color"], linewidth=1.5,
                label=pathway)
        ax.fill(angles, values, color=data[i]["color"], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.set_title("Normalized pathway comparison\n(higher = better)",
                 fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / "deer_tea_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


def comparison_matrix(data: list) -> Path:
    """Print a text-based comparison matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.table import Table
    except ImportError:
        return Path("")

    headers = ["Pathway", "Cost ($/kg)", "Energy\n(kWh/kg)", "GHG\n(kgCO2/kg)", "Recovery (%)"]
    rows = []
    for r in sorted(data, key=lambda x: x["cost_usd_kg"]):
        rows.append([
            r["pathway"].replace("_", "\n"),
            f"${r['cost_usd_kg']:.2f}",
            f"{r['energy_kwh_kg']:.1f}",
            f"{r['ghg_kg_co2_kg']:.1f}",
            f"{r['capacity_recovery_pct']:.0f}%",
        ])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    tbl = Table(ax, bbox=[0, 0, 1, 1])
    col_w = [0.30, 0.18, 0.18, 0.18, 0.16]
    for j, h in enumerate(headers):
        cell = tbl.add_cell(0, j, col_w[j], 0.12, text=h,
                            loc="center", facecolor="#34495e")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_fontsize(9)

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.add_cell(i + 1, j, col_w[j], 0.10,
                                text=val, loc="center",
                                facecolor="#f9f9f9" if i % 2 == 0 else "white")
            cell.get_text().set_fontsize(8)
            if row[0].replace("\n", "") == "DEER":
                cell.set_facecolor("#e8f5e9")

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.8)
    ax.add_table(tbl)
    ax.set_title("DEER TEA/LCA Comparison Summary", fontsize=12,
                 fontweight="bold", pad=12)

    out = FIGURES_DIR / "deer_tea_summary_table.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize DEER TEA/LCA results")
    ap.add_argument("--no-radar", action="store_true")
    args = ap.parse_args()

    log.info("Loading TEA data...")
    data = load_tea_data()

    log.info("Generating charts...")
    bar_chart(data, "cost_usd_kg",
              "Manufacturing Cost by Recycling Pathway",
              "Cost ($/kg recovered material)",
              "deer_tea_cost_bar.png", invert=False)

    bar_chart(data, "energy_kwh_kg",
              "Energy Consumption by Pathway",
              "Energy (kWh/kg)",
              "deer_tea_energy_bar.png", invert=True)

    bar_chart(data, "ghg_kg_co2_kg",
              "GHG Emissions by Pathway",
              "kg CO₂-eq / kg recovered material",
              "deer_tea_ghg_bar.png", invert=True)

    bar_chart(data, "capacity_recovery_pct",
              "Capacity Recovery by Pathway",
              "Capacity recovery (%)",
              "deer_tea_capacity_bar.png", invert=False)

    if not args.no_radar:
        radar_chart(data)

    comparison_matrix(data)

    log.info("Done. All figures in %s", FIGURES_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
