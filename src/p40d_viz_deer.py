"""p40d_viz_deer.py - DEER visualization figures.



Generates 6 publication-quality figures from simulation results:

  1. CV regeneration curves       (regeneration_cv_curves.csv)

  2. LiF cycling stability       (lif_cycling_curve.csv)

  3. Solvent screening results    (solvent_eei_predictions.csv)

  4. Pouch cell scale-up         (pouch_comparison.csv)

  5. TEA/LCA comparison          (deer_tea_lca.csv)

  6. Sensitivity tornado         (deer_sensitivity.json)



Output: figures/deer_*.png

"""

from __future__ import annotations



import json

import sys

from pathlib import Path



import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import numpy as np

import pandas as pd



THIS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(THIS_DIR))

from utils import RESULTS_DIR, FIGURES_DIR  # noqa: E402




# Colour palette

DEER_BLUE   = "#1A6EBF"

DEER_ORANGE = "#E8630A"

FRESH_GRAY  = "#7F7F7F"

MARKER_CAT  = "#2171B5"

MARKER_ANO  = "#D94801"

ACCENT_GOLD = "#F0AD4E"

LABEL_FONT  = 11

TITLE_FONT  = 13

DPI         = 150





# --------------------------------------------------------------------------- #

# Fig 1 – CV regeneration curves (R_EEI vs scan number)

# --------------------------------------------------------------------------- #

def fig_cv_regeneration(out_path: Path) -> None:

    df = pd.read_csv(RESULTS_DIR / "regeneration_cv_curves.csv")

    if df.empty:

        print(f"[p40d_viz_deer] WARNING: {df} is empty, skipping Fig 1")

        return



    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)

    for col, color, label in [

        ("r_eei_cathode_ohm_cm2",   MARKER_CAT, "NMC811 cathode (CEI)"),

        ("r_eei_anode_ohm_cm2",    MARKER_ANO, "Graphite anode (SEI)"),

        ("r_eei_combined_ohm_cm2", DEER_BLUE,  "Combined EEI"),

    ]:

        if col not in df.columns:

            continue

        ax.plot(

            df["scan_number"], df[col],

            marker="o", color=color, label=label, linewidth=2, markersize=5,

        )



    # Read summary for annotations

    try:

        summary = json.loads((RESULTS_DIR / "regeneration_summary.json").read_text())

        k = summary.get("cathode_k_diss_per_scan", 0)

        recovery = summary.get("combined_recovery_pct", 0)

        ax.annotate(

            f"k_diss = {k:.3f}/scan\nRecovery = {recovery:.1f}% (10 scans)",

            xy=(5, df["r_eei_combined_ohm_cm2"].iloc[-1]),

            xytext=(3, df["r_eei_combined_ohm_cm2"].max() * 0.6),

            arrowprops=dict(arrowstyle="->", color=DEER_BLUE),

            fontsize=9, color=DEER_BLUE,

        )

    except Exception:

        pass



    ax.set_xlabel("CV Scan Number", fontsize=LABEL_FONT)

    ax.set_ylabel(r"$R_{\mathrm{EEI}}$ ($\Omega\cdot$cm$^2$)", fontsize=LABEL_FONT)

    ax.set_title("DEER EEI Dissolution — R_EEI vs CV Scan", fontsize=TITLE_FONT)

    ax.legend(fontsize=9)

    ax.set_xlim(-0.5, df["scan_number"].max() + 0.5)

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"[p40d_viz_deer] Fig1 saved: {out_path}")





# --------------------------------------------------------------------------- #

# Fig 2 – LiF cycling stability (capacity retention vs cycles)

# --------------------------------------------------------------------------- #

def fig_lif_cycling(out_path: Path) -> None:

    df = pd.read_csv(RESULTS_DIR / "lif_cycling_curve.csv")

    if df.empty:

        print("[p40d_viz_deer] WARNING: lif_cycling_curve.csv empty, skipping Fig 2")

        return



    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)



    deer_cap  = df["capacity_deer_pct"].values.astype(float)

    fresh_cap = df["capacity_fresh_pct"].values.astype(float)

    cycles    = df["cycle"].values.astype(float)



    ax.plot(cycles, deer_cap,  color=DEER_BLUE,  linewidth=2.5,

            label="DEER cell (LiF residual)")

    ax.plot(cycles, fresh_cap, color=FRESH_GRAY, linewidth=2.5,

            linestyle="--", label="Fresh cell (organic SEI)")



    # 80 % SOH reference

    ax.axhline(80.0, color="red", linestyle=":", linewidth=1.5, label="80% SOH")



    # Find cycles-to-80%

    deer_80  = next((int(c) for c, v in zip(cycles, deer_cap)  if v <= 80.0), len(cycles))

    fresh_80 = next((int(c) for c, v in zip(cycles, fresh_cap) if v <= 80.0), len(cycles))



    ax.axvline(deer_80,  color=DEER_BLUE,  linestyle=":", linewidth=1.2, alpha=0.7)

    ax.axvline(fresh_80, color=FRESH_GRAY, linestyle=":", linewidth=1.2, alpha=0.7)

    ax.annotate(

        f"DEER 80%: {deer_80} cycles\nFresh 80%: {fresh_80} cycles",

        xy=(fresh_80, 80.5), fontsize=9,

        color=FRESH_GRAY,

    )



    gain = deer_80 - fresh_80

    gain_pct = round(gain / max(1, fresh_80) * 100, 1)

    ax.annotate(

        f"+{gain} cycles (+{gain_pct}%)",

        xy=(deer_80 * 0.5, 90),

        fontsize=10, color=DEER_BLUE, fontweight="bold",

    )



    ax.set_xlabel("Cycle Number", fontsize=LABEL_FONT)

    ax.set_ylabel("Relative Capacity (%)", fontsize=LABEL_FONT)

    ax.set_title("LiF Residual Layer — Cycling Stability (DEER vs Fresh)", fontsize=TITLE_FONT)

    ax.legend(fontsize=9)

    ax.set_xlim(0, min(500, cycles.max()))

    ax.set_ylim(75, 102)

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"[p40d_viz_deer] Fig2 saved: {out_path}")





# --------------------------------------------------------------------------- #

# Fig 3 – Solvent screening: bar + scatter + Pareto

# --------------------------------------------------------------------------- #

def fig_solvent_screening(out_path: Path) -> None:

    df = pd.read_csv(RESULTS_DIR / "solvent_eei_predictions.csv")

    if df.empty:

        print("[p40d_viz_deer] WARNING: solvent_eei_predictions.csv empty, skipping Fig 3")

        return



    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=DPI)



    # ── Panel A: Top-15 dissolution bar ──────────────────────────────────

    ax = axes[0]

    top15 = df.nlargest(15, "eei_dissolution_score").iloc[::-1]

    colors = [

        DEER_BLUE if "DMI" in n else DEER_ORANGE if "DMSO" in n else "#AAAAAA"

        for n in top15["name"]

    ]

    y_pos = np.arange(len(top15))

    ax.barh(y_pos, top15["eei_dissolution_score"].astype(float), color=colors, height=0.65)

    ax.set_yticks(y_pos)

    ax.set_yticklabels([n[:30] for n in top15["name"]], fontsize=7)

    ax.set_xlabel("EEI Dissolution Score", fontsize=9)

    ax.set_title("Top-15 Solvents\nby EEI Dissolution", fontsize=10)

    ax.set_xlim(0, 1.05)

    ax.grid(True, axis="x", alpha=0.3)

    dee_patch  = mpatches.Patch(color=DEER_BLUE,  label="DMI (anchor)")

    dmso_patch = mpatches.Patch(color=DEER_ORANGE, label="DMSO")

    gray_patch = mpatches.Patch(color="#AAAAAA",   label="Other")

    ax.legend(handles=[dee_patch, dmso_patch, gray_patch], fontsize=8)



    # ── Panel B: DN vs dissolution scatter ────────────────────────────────

    ax2 = axes[1]

    x = df["dn"].astype(float)

    y = df["eei_dissolution_score"].astype(float)

    ax2.scatter(x, y, alpha=0.6, color="#2171B5", s=40)

    # Label top 3

    for _, row in df.nlargest(3, "eei_dissolution_score").iterrows():

        ax2.annotate(

            row["name"].split(" -")[0][:12],

            (float(row["dn"]), float(row["eei_dissolution_score"])),

            fontsize=7, xytext=(5, 3), textcoords="offset points",

        )

    ax2.set_xlabel("Donor Number (kcal/mol)", fontsize=9)

    ax2.set_ylabel("EEI Dissolution Score", fontsize=9)

    ax2.set_title("DN vs EEI\nDissolution", fontsize=10)

    ax2.grid(True, alpha=0.3)



    # ── Panel C: Pareto front scatter ────────────────────────────────────

    ax3 = axes[2]

    x3 = df["eei_dissolution_score"].astype(float)

    y3 = df["electrode_compat_score"].astype(float)

    composite = df["composite_score"].astype(float)

    q1, q2 = composite.quantile(0.5), composite.quantile(0.75)

    tier_colors = []

    for comp in composite:

        if comp >= q2:

            tier_colors.append("#2171B5")

        elif comp >= q1:

            tier_colors.append("#F0AD4E")

        else:

            tier_colors.append("#7F7F7F")

    ax3.scatter(x3, y3, color=tier_colors, alpha=0.7, s=45)

    for _, row in df.nlargest(3, "composite_score").iterrows():

        ax3.annotate(

            row["name"].split(" -")[0][:10],

            (float(row["eei_dissolution_score"]), float(row["electrode_compat_score"])),

            fontsize=7, xytext=(3, 3), textcoords="offset points",

        )

    ax3.set_xlabel("EEI Dissolution Score", fontsize=9)

    ax3.set_ylabel("Electrode Compat Score", fontsize=9)

    ax3.set_title("Pareto Front\nDissolution vs Compat", fontsize=10)

    ax3.grid(True, alpha=0.3)



    fig.suptitle(

        "DEER Solvent Screening — Phase 1 Results",

        fontsize=TITLE_FONT, fontweight="bold", y=1.01,

    )

    fig.tight_layout()

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"[p40d_viz_deer] Fig3 saved: {out_path}")





# --------------------------------------------------------------------------- #

# Fig 4 – Pouch cell scale-up validation

# --------------------------------------------------------------------------- #

def fig_pouch_validation(out_path: Path) -> None:

    csv_path = RESULTS_DIR / "pouch_comparison.csv"

    if not csv_path.exists():

        print(f"[p40d_viz_deer] WARNING: {csv_path} not found, skipping Fig 4")

        return

    df = pd.read_csv(csv_path)

    json_path = RESULTS_DIR / "pouch_scale_validation.json"

    summary = {}

    if json_path.exists():

        summary = json.loads(json_path.read_text())



    fig, ax = plt.subplots(figsize=(7, 5), dpi=DPI)

    labels  = df["label"].tolist()

    recovery = df["predicted_recovery_pct"].astype(float).tolist()



    colors = [DEER_BLUE if "3 Ah" in lbl else DEER_ORANGE if "1 Ah" in lbl else FRESH_GRAY

               for lbl in labels]

    bars = ax.bar(labels, recovery, color=colors, width=0.5, alpha=0.85)



    # Kalra reference line

    ax.axhline(90.3, color="red", linestyle="--", linewidth=1.5, label="Kalra 2026: 90.3%")



    for bar, val in zip(bars, recovery):

        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,

                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)



    # Scale factors annotation

    if summary:

        f_total = df["f_total"].astype(float).iloc[2] if len(df) > 2 else 1.0

        note = f"Total correction factor = {f_total:.4f}"

        ax.annotate(note, xy=(0.5, -0.12), xycoords="axes fraction",

                    ha="center", fontsize=8, color="#555555")



    ax.set_ylabel("Recovery Efficiency (%)", fontsize=LABEL_FONT)

    ax.set_title("Coin Cell -> Pouch Cell Scale-Up", fontsize=TITLE_FONT)

    ax.set_ylim(0, 105)

    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)

    dee_patch  = mpatches.Patch(color=DEER_BLUE,  label="3 Ah pouch")

    dmso_patch = mpatches.Patch(color=DEER_ORANGE, label="1 Ah pouch")

    gray_patch = mpatches.Patch(color=FRESH_GRAY,  label="Coin cell")

    ax.legend(handles=[dee_patch, dmso_patch, gray_patch], fontsize=8)

    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"[p40d_viz_deer] Fig4 saved: {out_path}")





# --------------------------------------------------------------------------- #

# Fig 5 – TEA/LCA comparison (horizontal bars)

# --------------------------------------------------------------------------- #

def fig_tea_comparison(out_path: Path) -> None:

    csv_path = RESULTS_DIR / "deer_tea_lca.csv"

    if not csv_path.exists():

        print(f"[p40d_viz_deer] WARNING: {csv_path} not found, skipping Fig 5")

        return

    df = pd.read_csv(csv_path, encoding="latin-1")

    if df.empty:

        return



    pathway_col = next((c for c in df.columns if "pathway" in c.lower()), None)

    if pathway_col is None:

        pathway_col = df.columns[0]



    pathways = df[pathway_col].tolist()

    costs    = df["cost_usd_kg"].astype(float).tolist() if "cost_usd_kg" in df.columns else []

    energies = df["energy_kwh_kg"].astype(float).tolist() if "energy_kwh_kg" in df.columns else []

    ghgs     = df["ghg_kgco2_kg"].astype(float).tolist() if "ghg_kgco2_kg" in df.columns else []

    caps     = df["capacity_recovery_pct"].astype(float).tolist() if "capacity_recovery_pct" in df.columns else []



    # Map pathway → bar color

    def bar_color(p: str) -> str:

        p_lower = p.lower()

        if "deer" in p_lower:

            return DEER_BLUE

        if "electrolyte" in p_lower:

            return "#2E8B57"

        if "direct" in p_lower or "cathode" in p_lower:

            return DEER_ORANGE

        return "#7F7F7F"



    _n = len(pathways)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=DPI)



    panels = [

        (axes[0, 0], costs,    "$/kg",              "Manufacturing Cost",             True),

        (axes[0, 1], energies, "kWh/kg",             "Energy Consumption",            False),

        (axes[1, 0], ghgs,     "kg CO$_2$-eq/kg",   "GHG Emissions",                False),

        (axes[1, 1], caps,     "%",                  "Capacity Recovery",             True),

    ]



    for ax, values, unit, title, low_is_good in panels:

        if not values:

            ax.set_title(title)

            continue

        colors = [bar_color(p) for p in pathways]

        sorted_pairs = sorted(zip(pathways, values, colors), key=lambda x: x[1], reverse=low_is_good)

        ys = np.arange(len(sorted_pairs))

        labels, vals, cols = zip(*sorted_pairs)

        short = [p.split("_")[0][:20] for p in labels]

        ax.barh(ys, vals, color=cols, height=0.6)

        ax.set_yticks(ys)

        ax.set_yticklabels(short, fontsize=9)

        ax.set_xlabel(unit, fontsize=9)

        ax.set_title(title, fontsize=10)

        ax.grid(True, axis="x", alpha=0.3)

        if low_is_good:

            ax.invert_yaxis()

        for i, v in enumerate(vals):

            ax.text(v + max(vals) * 0.01, i, f"{v:.2f}", va="center", fontsize=8)



    fig.suptitle(

        "DEER Techno-Economic & Environmental Assessment — 5 Pathways",

        fontsize=TITLE_FONT, fontweight="bold",

    )

    fig.tight_layout()

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"[p40d_viz_deer] Fig5 saved: {out_path}")





# --------------------------------------------------------------------------- #

# Fig 6 – Sensitivity tornado

# --------------------------------------------------------------------------- #

def fig_sensitivity_tornado(out_path: Path) -> None:

    json_path = RESULTS_DIR / "deer_sensitivity.json"

    if not json_path.exists():

        print(f"[p40d_viz_deer] WARNING: {json_path} not found, skipping Fig 6")

        return

    data = json.loads(json_path.read_text())



    tornado = data.get("tornado", [])

    if not tornado:

        print("[p40d_viz_deer] No tornado data, skipping Fig 6")

        return



    base = data.get("base_case", {}).get("total_cost_kg", 3.79)

    entries = sorted(tornado, key=lambda x: abs(x.get("delta_max", 0)), reverse=True)



    labels  = [e["parameter"][:28] for e in entries]

    deltas  = [e.get("delta_max", 0) for e in entries]

    colors  = ["#D94801" if d > 0 else "#2171B5" for d in deltas]



    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)

    ys = np.arange(len(labels))

    ax.barh(ys, deltas, color=colors, height=0.6, alpha=0.85)



    # Zero reference line

    ax.axvline(0, color="black", linewidth=0.8)



    # Labels

    ax.set_yticks(ys)

    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Impact on DEER cost ($/kg)", fontsize=LABEL_FONT)

    ax.set_title(

        f"DEER Cost Sensitivity Analysis  (base = ${base:.2f}/kg)",

        fontsize=TITLE_FONT,

    )



    # Legend

    pos_patch = mpatches.Patch(color="#D94801", label="Higher cost")

    neg_patch = mpatches.Patch(color="#2171B5", label="Lower cost")

    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc="lower right")



    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"[p40d_viz_deer] Fig6 saved: {out_path}")





# --------------------------------------------------------------------------- #

# Main

# --------------------------------------------------------------------------- #

def main() -> int:

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)



    fig_map = [

        (FIGURES_DIR / "deer_cv_regeneration.png",  fig_cv_regeneration),

        (FIGURES_DIR / "deer_lif_cycling.png",      fig_lif_cycling),

        (FIGURES_DIR / "deer_solvent_screening.png", fig_solvent_screening),

        (FIGURES_DIR / "deer_pouch_validation.png",  fig_pouch_validation),

        (FIGURES_DIR / "deer_tea_comparison.png",   fig_tea_comparison),

        (FIGURES_DIR / "deer_sensitivity_tornado.png", fig_sensitivity_tornado),

    ]



    for path, fn in fig_map:

        try:

            fn(path)

        except Exception as exc:

            print(f"[p40d_viz_deer] ERROR in {fn.__name__}: {exc}")



    print("[p40d_viz_deer] Done.")

    return 0





if __name__ == "__main__":

    sys.exit(main())

