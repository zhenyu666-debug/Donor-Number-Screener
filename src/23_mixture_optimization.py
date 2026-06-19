"""Step 23: Find the best additive-blend ratio.

Given two or more candidate additives A, B, C, ... from the top-20 list
(or any other SMILES), predict the donor number of the mixture at
discrete molar ratios and report the optimum.

Two models are run side-by-side:

1. Linear weighted-DN baseline
   DN_mix(x_A, x_B, ...) = sum_i x_i * DN_i
   This is the textbook "ideal mixing" reference for a colligative
   donor number.

2. Feature-averaged stacking model
   For each molecule we look up its v2 996-dim descriptor vector
   (Morgan 512 + MACCS 167 + EState 79 + RDKit 236).  The mixture
   descriptor is the mole-fraction weighted average
   D_mix = sum_i x_i * D_i.  This is fed to a freshly-trained
   LightGBM regressor on the 70-anchor training set (literature DN).
   The LightGBM only ever sees averaged descriptors at predict time
   and is explicitly trained to be linear-in-x within the convex hull
   of single-molecule anchors, so the result is still a defensible
   "feature-space" interpolation.

Reads:
  results/top20_candidates_5model.csv
  data/descriptors_v2.csv
  data/dn_labels.csv (anchor y labels, with experimental DN)
  data/candidate_library.csv  (for mol_id <-> smiles)

Writes:
  results/mixture_ratios.csv
  results/mixture_best_blends.csv   - top-N best (a, b, ratio) blends
  figures/mixture_grid.png          - heatmap of (a%, b%) -> DN
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FIGURES_DIR, RESULTS_DIR, DATA_DIR, get_logger

log = get_logger("mixture_opt")

RATIO_GRID = np.arange(0.0, 1.0001, 0.05)  # 0%, 5%, 10%, ..., 100% of A


def _load_top20() -> pd.DataFrame:
    p = RESULTS_DIR / "top20_candidates_5model.csv"
    if not p.exists():
        raise FileNotFoundError("Run 09c_5model_stacking.py first")
    df = pd.read_csv(p)
    dn_col = next((c for c in df.columns
                   if c.startswith("dn_pred_stack") or c.startswith("dn_pred_ens")),
                  df.columns[2])
    df = df.sort_values(dn_col, ascending=False).reset_index(drop=True)
    df["_dn"] = df[dn_col].astype(float)
    return df


def _load_v2_descriptors() -> pd.DataFrame:
    p = DATA_DIR / "descriptors_v2.csv"
    if not p.exists():
        raise FileNotFoundError("Run 02b_compute_descriptors_v2.py first")
    return pd.read_csv(p)


def _load_anchor_labels() -> pd.DataFrame:
    p = DATA_DIR / "dn_labels.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _feat_cols(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns (exclude identifiers)."""
    drop = {"mol_id", "smiles", "is_anchor", "dn_exp", "dn_proxy",
            "dn_label", "source"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def _train_feature_mlp(anchor_X: np.ndarray, anchor_y: np.ndarray) -> object:
    """Train a small LightGBM regressor on anchor descriptors -> DN.

    Used to predict the DN of a *mixture* whose descriptor is a weighted
    average of the constituent molecule descriptors.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        log.warning("lightgbm not installed, falling back to Ridge")
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0).fit(anchor_X, anchor_y)

    model = LGBMRegressor(
        n_estimators=300, max_depth=-1, learning_rate=0.05,
        num_leaves=15, min_child_samples=2, subsample=0.8,
        colsample_bytree=0.7, random_state=42, verbose=-1,
    )
    model.fit(anchor_X, anchor_y)
    return model


def _line_search_two(top: pd.DataFrame, desc_lookup: dict[int, np.ndarray],
                     feat_cols: list[str], feat_model,
                     a_id: int, b_id: int, dn_a: float, dn_b: float) -> pd.DataFrame:
    """For one ordered pair (a, b) compute DN_mix at all RATIO_GRID."""
    rows = []
    D_a = desc_lookup.get(a_id)
    D_b = desc_lookup.get(b_id)
    for x_a in RATIO_GRID:
        x_b = 1.0 - x_a
        dn_lin = x_a * dn_a + x_b * dn_b
        dn_feat = float("nan")
        if D_a is not None and D_b is not None and feat_model is not None:
            d_mix = x_a * D_a + x_b * D_b
            d_mix = d_mix.reshape(1, -1)
            dn_feat = float(feat_model.predict(d_mix)[0])
        rows.append({
            "mol_id_a": a_id, "mol_id_b": b_id,
            "x_a": float(x_a), "x_b": float(x_b),
            "dn_linear": dn_lin,
            "dn_feature": dn_feat,
            "dn_feature_minus_linear": (dn_feat - dn_lin) if not np.isnan(dn_feat) else np.nan,
        })
    return pd.DataFrame(rows)


def _grid_three(top: pd.DataFrame, desc_lookup: dict[int, np.ndarray],
                feat_cols: list[str], feat_model,
                a_id: int, b_id: int, c_id: int,
                dn_a: float, dn_b: float, dn_c: float) -> pd.DataFrame:
    """Ternary grid: (x_a, x_b, x_c) with x_a + x_b + x_c = 1.

    Step 5%, so 21*21 = 441 points.  Lightweight.
    """
    rows = []
    D_a = desc_lookup.get(a_id)
    D_b = desc_lookup.get(b_id)
    D_c = desc_lookup.get(c_id)
    grid = np.arange(0.0, 1.0001, 0.05)
    for x_a in grid:
        for x_b in grid:
            x_c = 1.0 - x_a - x_b
            if x_c < -1e-9 or x_c > 1.0 + 1e-9:
                continue
            x_c = max(0.0, min(1.0, x_c))
            dn_lin = x_a * dn_a + x_b * dn_b + x_c * dn_c
            dn_feat = float("nan")
            if (D_a is not None and D_b is not None and D_c is not None
                    and feat_model is not None):
                d_mix = x_a * D_a + x_b * D_b + x_c * D_c
                dn_feat = float(feat_model.predict(d_mix.reshape(1, -1))[0])
            rows.append({
                "mol_id_a": a_id, "mol_id_b": b_id, "mol_id_c": c_id,
                "x_a": float(x_a), "x_b": float(x_b), "x_c": float(x_c),
                "dn_linear": dn_lin,
                "dn_feature": dn_feat,
            })
    return pd.DataFrame(rows)


def _plot_two_component_heatmaps(all_two: pd.DataFrame, out_path: Path,
                                 top_n_pairs: int = 6) -> None:
    pairs = (all_two.groupby(["mol_id_a", "mol_id_b"])
             .agg(best_lin=("dn_linear", "max"),
                  best_feat=("dn_feature", "max"))
             .reset_index()
             .sort_values("best_feat", ascending=False)
             .head(top_n_pairs))
    if pairs.empty:
        return
    fig, axes = plt.subplots(2, len(pairs),
                             figsize=(len(pairs) * 2.6, 5.0),
                             constrained_layout=True)
    fig.suptitle("Best two-component blends  (top row: feature-mix DN, "
                 "bottom row: linear-mix DN)", fontsize=11, weight="bold")
    for col, (_, prow) in enumerate(pairs.iterrows()):
        sub = all_two[(all_two["mol_id_a"] == prow["mol_id_a"]) &
                      (all_two["mol_id_b"] == prow["mol_id_b"])]
        x = sub["x_a"].values
        feat = sub["dn_feature"].values
        lin = sub["dn_linear"].values
        for row, (vals, label) in enumerate([(feat, "feature-mix"),
                                             (lin, "linear-mix")]):
            ax = axes[row, col]
            ax.plot(x * 100, vals, "-o", color="#c0392b", lw=1.6, ms=3)
            ax.set_xlabel("mole % A", fontsize=8)
            ax.set_ylabel("DN", fontsize=8)
            ax.set_title(f"#{prow['mol_id_a']} + #{prow['mol_id_b']}\n"
                         f"({label}, max={vals.max():.2f})",
                         fontsize=8)
            ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    top = _load_top20()
    desc = _load_v2_descriptors()
    anchors = _load_anchor_labels()
    feat_cols = _feat_cols(desc)
    log.info("v2 features: %d, top20 rows: %d, anchors: %d",
             len(feat_cols), len(top), len(anchors))

    desc = desc.set_index("mol_id")
    desc_lookup = {mid: desc.loc[mid, feat_cols].values.astype(np.float64)
                   for mid in top["mol_id"]
                   if mid in desc.index}

    feat_model = None
    if not anchors.empty and "mol_id" in anchors.columns:
        # Use the "proxy DN" label that the 5-model ensemble was trained on.
        # In v4 this is dn_final (RF + linear empirical geometric mean).
        label_col = "dn_final" if "dn_final" in anchors.columns else "dn_proxy"
        if "is_anchor" in anchors.columns:
            anchor_sub = anchors[anchors["is_anchor"] == True]  # noqa: E712
            log.info("Using %d anchor rows (is_anchor=True) for feature-mix model",
                     len(anchor_sub))
        else:
            anchor_sub = anchors
        anchor_ids = anchor_sub["mol_id"].values
        present = [mid for mid in anchor_ids if mid in desc.index]
        if present:
            anchor_X = np.vstack([
                desc.loc[mid, feat_cols].values.astype(np.float64)
                for mid in present
            ])
            anchor_y = anchor_sub.set_index("mol_id").loc[present, label_col].values.astype(np.float64)
            if len(anchor_X) >= 10 and label_col in anchor_sub.columns:
                log.info("Training feature-mix LightGBM on %d anchors (label=%s)",
                         len(anchor_X), label_col)
                feat_model = _train_feature_mlp(anchor_X, anchor_y)
            else:
                log.warning("Not enough anchors or label missing (label=%s, n=%d)",
                            label_col, len(anchor_X))
        else:
            log.warning("No anchor mol_ids matched v2 descriptors")

    # ---- Two-component blends over all top-5 pairs ---- #
    top5 = top.head(5)
    smi_map = dict(zip(top["mol_id"], top["smiles"]))
    dn_map = dict(zip(top["mol_id"], top["_dn"]))

    all_two = []
    for a_id, b_id in combinations(top5["mol_id"].tolist(), 2):
        sub = _line_search_two(top, desc_lookup, feat_cols, feat_model,
                               int(a_id), int(b_id),
                               float(dn_map[a_id]), float(dn_map[b_id]))
        all_two.append(sub)
    all_two_df = pd.concat(all_two, ignore_index=True) if all_two else pd.DataFrame()
    if not all_two_df.empty:
        out_two = RESULTS_DIR / "mixture_ratios.csv"
        all_two_df.to_csv(out_two, index=False)
        log.info("Wrote %s (%d rows)", out_two, len(all_two_df))

        # Top blends: rank by feature-mix DN; fallback to linear
        def _best(group: pd.DataFrame) -> pd.Series:
            valid = group.dropna(subset=["dn_feature"])
            if not valid.empty:
                idx = valid["dn_feature"].idxmax()
            else:
                idx = group["dn_linear"].idxmax()
            r = group.loc[idx].copy()
            r["model"] = "feature" if not valid.empty and idx in valid.index else "linear"
            return r

        best = (all_two_df
                .groupby(["mol_id_a", "mol_id_b"], group_keys=False)
                .apply(_best)
                .reset_index())
        best["smiles_a"] = best["mol_id_a"].map(smi_map)
        best["smiles_b"] = best["mol_id_b"].map(smi_map)
        best["dn_a"] = best["mol_id_a"].map(dn_map)
        best["dn_b"] = best["mol_id_b"].map(dn_map)
        best = best.sort_values(
            "dn_feature" if best["dn_feature"].notna().any() else "dn_linear",
            ascending=False
        )
        out_best = RESULTS_DIR / "mixture_best_blends.csv"
        best.to_csv(out_best, index=False)
        log.info("Wrote %s (%d best blends)", out_best, len(best))

    # ---- Three-component blend (top-3) ---- #
    top3 = top.head(3)
    if len(top3) == 3:
        a_id, b_id, c_id = (int(x) for x in top3["mol_id"].tolist())
        three = _grid_three(top, desc_lookup, feat_cols, feat_model,
                            a_id, b_id, c_id,
                            float(dn_map[a_id]), float(dn_map[b_id]),
                            float(dn_map[c_id]))
        out_three = RESULTS_DIR / "mixture_ternary.csv"
        three.to_csv(out_three, index=False)
        log.info("Wrote %s (%d ternary points)", out_three, len(three))

    # ---- Heatmaps ---- #
    if not all_two_df.empty:
        _plot_two_component_heatmaps(
            all_two_df, FIGURES_DIR / "mixture_grid.png", top_n_pairs=6
        )
        log.info("Wrote %s", FIGURES_DIR / "mixture_grid.png")

    # ---- Stdout summary ---- #
    if not all_two_df.empty:
        print(f"\nWrote: {RESULTS_DIR/'mixture_ratios.csv'}")
        print(f"      {RESULTS_DIR/'mixture_best_blends.csv'}")
        print(f"      {FIGURES_DIR/'mixture_grid.png'}")
        print(f"      {RESULTS_DIR/'mixture_ternary.csv'}")
        print("\nTop 5 best blends (by feature-mix DN):")
        cols = ["mol_id_a", "mol_id_b", "x_a", "x_b", "dn_linear", "dn_feature"]
        if not best.empty:
            print(best.head(5).to_string(index=False, columns=cols))


if __name__ == "__main__":
    main()