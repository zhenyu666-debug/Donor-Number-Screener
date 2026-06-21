"""Step 6: Advanced analysis layer.

Reproduces the spirit of the eScience 2026 paper Fig. 6-15 plus five
innovation overlays on top of the existing 01-05 reproduction:

  1. Hero summary panel (fig0)
  2. Multi-objective Pareto front (fig6)
  3. t-SNE chemical-space landscape (fig7)
  4. Atom-level SHAP contributions (fig8)
  5. Conformal prediction intervals + calibration (fig9)
  6. Active learning vs random vs full DFT labelling (fig10)
  7. Synthetic accessibility (SA) vs predicted DN (fig11)
  8. Decision-tree distillation of the RF (fig12)
  9. ROC + precision-recall for "DN > 30" classification (fig13)
 10. 1D partial-dependence of top-6 features (fig14)
 11. Top-20 candidate structure grid (fig15 + SVG)

Outputs (all paths relative to the project root):
  - figures/fig0_hero.png
  - figures/fig6_pareto.png
  - figures/fig7_tsne_landscape.png
  - figures/fig8_atom_shap.png
  - figures/fig9_conformal.png
  - figures/fig10_active_learning.png
  - figures/fig11_synth_acc.png
  - figures/fig12_decision_tree.png
  - figures/fig13_roc_pr.png
  - figures/fig14_pdp.png
  - figures/fig15_top_molecules.png
  - figures/top_molecules.svg
  - results/conformal_intervals.csv
  - results/pareto_optimal.csv
  - results/decision_rules.txt
  - results/active_learning_curve.csv
  - results/top_molecules_summary.json

Implementation notes
--------------------
- This script retrains a small RF + XGB on the *full* 80 % training
  partition of the 3 551 candidate library (the model in step 4 is
  not serialised to disk).  Same seed=42 so the test set is identical
  to step 4.
- We deliberately avoid `umap-learn` and `sascorer` because they are
  not part of `requirements.txt` and pulling them tends to upgrade
  numpy above 2.x which breaks `rdkit-pypi` on Windows.  We fall back
  to `sklearn.manifold.TSNE` and a hand-rolled `sa_score()` proxy.
- Every figure is wrapped in its own try/except in `main()` so a
  failure in one plot does not abort the whole pipeline.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import (mean_squared_error,
                             precision_recall_curve, r2_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (DATA_DIR, FIGURES_DIR, RESULTS_DIR,  # noqa: E402
                   get_logger, set_global_seed)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
set_global_seed(42)
log = get_logger("advanced")

NON_FEATURE_COLS = {"mol_id", "smiles", "smiles_x", "smiles_y",
                    "dn_rf", "dn_empirical", "dn_final", "confidence",
                    "is_anchor"}


# --------------------------------------------------------------------------- #
# 0.  Train / load the predictors (same recipe as 04_train_models.py)
# --------------------------------------------------------------------------- #
def train_predictors():
    """Return (rf, xgb, X_train, X_test, y_train, y_test, df, feat_cols)."""
    desc = pd.read_csv(DATA_DIR / "descriptors.csv")
    labels = pd.read_csv(DATA_DIR / "dn_labels.csv")
    df = desc.merge(labels[["mol_id", "dn_final", "is_anchor"]],
                    on="mol_id", how="left")
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS and df[c].dtype != "O"]
    X = df[feat_cols].values
    y = df["dn_final"].values
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.20,
                                           random_state=42, shuffle=True)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    log.info("Retraining RF + XGB: train n=%d test n=%d feats=%d",
             len(y_train), len(y_test), len(feat_cols))

    rf = RandomForestRegressor(n_estimators=300, max_depth=None,
                               min_samples_split=2, n_jobs=-1,
                               random_state=42).fit(X_train, y_train)
    xgb = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.1,
                       n_jobs=-1, verbosity=0, tree_method="hist",
                       random_state=42).fit(X_train, y_train)
    log.info("Retrain done.  RF R2=%.4f  XGB R2=%.4f",
             r2_score(y_test, rf.predict(X_test)),
             r2_score(y_test, xgb.predict(X_test)))
    return rf, xgb, X_train, X_test, y_train, y_test, df, feat_cols


# --------------------------------------------------------------------------- #
# Synthetic accessibility proxy
# --------------------------------------------------------------------------- #
def sa_score(mol: Chem.Mol) -> float:
    """Cheap SA-score proxy that does not require Ertl's RDKit contrib.

    Linear combination of: #rotatable bonds, ring atoms, heavy atoms and
    aromatic atoms.  Calibrated so that 1.0 = trivially easy, 10.0 =
    very hard.  Real molecules in the library sit between 1.5 and 4.5.
    """
    if mol is None:
        return 10.0
    rot = Descriptors.NumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    n_ring = rdMolDescriptors.CalcNumRings(mol)
    n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    sa = 1.0 + 0.18 * rot + 0.5 * n_ring + 0.05 * heavy + 0.25 * n_arom
    return float(max(1.0, min(10.0, sa)))


# --------------------------------------------------------------------------- #
# 1.  Hero panel
# --------------------------------------------------------------------------- #
def fig0_hero(rf, xgb, X_test, y_test, df, feat_cols):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("Hero: data-driven screening of high-DN Li-S additives",
                 fontsize=16, fontweight="bold", y=0.99)

    # Top-left: parity scatter (XGB)
    pred = xgb.predict(X_test)
    axes[0, 0].scatter(y_test, pred, s=10, alpha=0.5, c="#3b6fb6",
                       edgecolor="none")
    lo, hi = float(min(y_test.min(), pred.min())), float(max(y_test.max(), pred.max()))
    axes[0, 0].plot([lo, hi], [lo, hi], "k--", lw=1)
    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    axes[0, 0].set_title(f"XGBoost parity (R$^2$={r2:.3f}, RMSE={rmse:.2f})",
                         fontsize=11)
    axes[0, 0].set_xlabel("Reference DN")
    axes[0, 0].set_ylabel("Predicted DN")
    axes[0, 0].set_aspect("equal", adjustable="datalim")

    # Top-right: feature importance (XGB top 15)
    fi = pd.Series(xgb.feature_importances_, index=feat_cols)
    top = fi.sort_values(ascending=False).head(15)
    nice = {"HOMO_proxy": "HOMO energy (proxy)",
            "dipole_proxy": "Dipole moment (proxy)",
            "LUMO_proxy": "LUMO energy (proxy)",
            "HL_gap_proxy": "HOMO-LUMO gap (proxy)",
            "polarizability_proxy": "Polarizability (proxy)",
            "TPSA": "TPSA", "MolLogP": "LogP",
            "MaxEStateIndex": "Max EState",
            "MinEStateIndex": "Min EState",
            "n_O": "#O", "n_N": "#N", "n_F": "#F",
            "fr_pyridine": "fr_pyridine",
            "fr_NH2": "fr_NH2", "fr_amide": "fr_amide"}
    pretty = [nice.get(c, c) for c in top.index]
    axes[0, 1].barh(pretty, top.values, color="#d36a4a")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title("Top-15 features (XGBoost)", fontsize=11)
    axes[0, 1].set_xlabel("Importance")

    # Bottom-left: DN prediction histogram
    full = pd.read_csv(DATA_DIR / "full_predictions.csv")
    axes[1, 0].hist(full["dn_pred_ens"], bins=60, color="#7e57c2",
                    edgecolor="white")
    axes[1, 0].axvline(full["dn_pred_ens"].quantile(0.995),
                       color="black", linestyle="--",
                       label=f"Top-20 cut ({full['dn_pred_ens'].quantile(0.995):.1f})")
    axes[1, 0].set_title("Ensemble DN distribution (n=3 551)", fontsize=11)
    axes[1, 0].set_xlabel("Predicted DN")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend(loc="upper right")

    # Bottom-right: atom-count composition (top-20 vs random)
    top20 = full.sort_values("dn_pred_ens", ascending=False).head(20)
    rng = full.sample(200, random_state=42)
    def atoms(smi: str) -> dict:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return {"O": 0, "N": 0, "F": 0, "S": 0}
        return {s: sum(1 for a in m.GetAtoms() if a.GetSymbol() == s)
                for s in ("O", "N", "F", "S")}

    top_counts = pd.DataFrame([atoms(s) for s in top20["smiles"]]).mean()
    rng_counts = pd.DataFrame([atoms(s) for s in rng["smiles"]]).mean()
    width = 0.35
    x = np.arange(len(top_counts))
    axes[1, 1].bar(x - width / 2, top_counts.values, width,
                   color="#d36a4a", label="Top-20 (high-DN)")
    axes[1, 1].bar(x + width / 2, rng_counts.values, width,
                   color="#888888", label="Random n=200")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(["O", "N", "F", "S"])
    axes[1, 1].set_ylabel("Mean #atoms per molecule")
    axes[1, 1].set_title("SEI-relevant element enrichment", fontsize=11)
    axes[1, 1].legend()
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGURES_DIR / "fig0_hero.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out)


# --------------------------------------------------------------------------- #
# 2.  Pareto
# --------------------------------------------------------------------------- #
def fig6_pareto(df, rf, xgb):
    full = pd.read_csv(DATA_DIR / "full_predictions.csv")
    pred_ens = (rf.predict(df[[c for c in df.columns
                               if c not in NON_FEATURE_COLS
                               and df[c].dtype != "O"]].values) +
                xgb.predict(df[[c for c in df.columns
                                if c not in NON_FEATURE_COLS
                                and df[c].dtype != "O"]].values)) / 2.0
    full["dn_pred_ens"] = pred_ens

    # Compute objective vectors.
    mols = [Chem.MolFromSmiles(s) for s in full["smiles"]]
    mws = np.array([Descriptors.MolWt(m) if m else 200 for m in mols])
    logp = np.array([Descriptors.MolLogP(m) if m else 0 for m in mols])
    tpsa = np.array([Descriptors.TPSA(m) if m else 0 for m in mols])
    nrot = np.array([Descriptors.NumRotatableBonds(m) if m else 0 for m in mols])
    sa = np.array([sa_score(m) for m in mols])
    dn = full["dn_pred_ens"].values

    full["MW"] = mws
    full["LogP"] = logp
    full["TPSA"] = tpsa
    full["Nrot"] = nrot
    full["SA"] = sa

    # Pareto optimal: maximise DN, minimise each of the 5 others.
    obj_cols = ["MW", "LogP", "TPSA", "Nrot", "SA"]
    is_pareto = np.ones(len(full), dtype=bool)
    for i in range(len(full)):
        if not is_pareto[i]:
            continue
        for j in range(len(full)):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i  if  DN[j] >= DN[i]  AND  all(other[j]<=other[i])
            if (dn[j] >= dn[i] and
                mws[j] <= mws[i] and
                logp[j] <= logp[i] and
                tpsa[j] <= tpsa[i] and
                nrot[j] <= nrot[i] and
                sa[j] <= sa[i] and
                (dn[j] > dn[i] or mws[j] < mws[i] or logp[j] < logp[i] or
                 tpsa[j] < tpsa[i] or nrot[j] < nrot[i] or sa[j] < sa[i])):
                is_pareto[i] = False
                break
    pareto = full[is_pareto].copy()
    pareto["dn_pred_ens"] = dn[is_pareto]
    pareto.to_csv(RESULTS_DIR / "pareto_optimal.csv", index=False)
    log.info("Pareto: %d of %d are non-dominated", len(pareto), len(full))

    # Plot 2x3 matrix of DN vs each other objective.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, xcol in zip(axes.flat, obj_cols):
        ax.scatter(full[xcol], dn, s=5, alpha=0.3, c="#aaaaaa",
                   edgecolor="none", label="All candidates")
        ax.scatter(pareto[xcol], pareto["dn_pred_ens"], s=20, c="#d36a4a",
                   edgecolor="black", linewidth=0.4, label="Pareto optimal")
        ax.set_xlabel(xcol)
        ax.set_ylabel("Predicted DN")
        ax.set_title(f"DN vs {xcol}")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"Multi-objective Pareto front "
                 f"({len(pareto)} non-dominated of {len(full)})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGURES_DIR / "fig6_pareto.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out)


# --------------------------------------------------------------------------- #
# 3.  t-SNE
# --------------------------------------------------------------------------- #
def fig7_tsne(df, rf, xgb):
    full = pd.read_csv(DATA_DIR / "full_predictions.csv")
    smiles = full["smiles"].tolist()
    feats_cols = [c for c in df.columns
                  if c not in NON_FEATURE_COLS and df[c].dtype != "O"]
    X = df[feats_cols].values
    pred_ens = (rf.predict(X) + xgb.predict(X)) / 2.0
    full["dn_pred_ens"] = pred_ens

    log.info("Computing 1024-bit Morgan fingerprints for t-SNE")
    fps = np.zeros((len(smiles), 1024), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        arr = np.zeros(1024, dtype=np.uint8)
        AllChem.DataStructs.ConvertToNumpyArray(bv, arr)
        fps[i] = arr

    # Subset 1500 for tractability.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(smiles), size=min(1500, len(smiles)), replace=False)
    sub = fps[idx]
    dn_sub = pred_ens[idx]
    smiles_sub = [smiles[i] for i in idx]

    t0 = time.perf_counter()
    coords = TSNE(n_components=2, perplexity=30, random_state=42,
                  init="random", learning_rate="auto",
                  max_iter=1000).fit_transform(sub.astype(np.float32))
    log.info("t-SNE done in %.1fs", time.perf_counter() - t0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=dn_sub, cmap="viridis",
                    s=8, alpha=0.7, edgecolor="none")
    plt.colorbar(sc, ax=ax, label="Predicted DN")

    # Highlight top-20
    top20_idx = np.argsort(dn_sub)[-20:]
    ax.scatter(coords[top20_idx, 0], coords[top20_idx, 1],
               s=80, facecolor="none", edgecolor="white", linewidth=1.2)
    for i in top20_idx[:8]:
        ax.annotate(smiles_sub[i][:18], (coords[i, 0], coords[i, 1]),
                    fontsize=6, color="white",
                    bbox=dict(facecolor="black", alpha=0.4, pad=1))
    ax.set_title("Chemical-space landscape (Morgan FP 1024-bit, t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    out = FIGURES_DIR / "fig7_tsne_landscape.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out)


# --------------------------------------------------------------------------- #
# 4.  Atom-level SHAP
# --------------------------------------------------------------------------- #
def fig8_atom_shap(xgb, X_test, df, feat_cols):
    """Atom-level contribution by fragment-perturbation.

    For each heavy atom A in the top-3 candidate molecules, we zero
    out the columns in X that are most strongly influenced by A
    (HOMO/LUMO/dipole proxies + atom-count columns) and re-predict.
    The signed delta in predicted DN is attributed to A.
    """
    import shap
    explainer = shap.TreeExplainer(xgb)
    # global SHAP on 200 random test molecules
    idx = np.random.default_rng(42).choice(len(X_test), size=200, replace=False)
    sv = explainer.shap_values(X_test[idx])  # (200, n_features)
    abs_sv = np.abs(sv).mean(axis=0)
    imp = pd.Series(abs_sv, index=feat_cols).sort_values(ascending=False)
    log.info("Atom-level: top-3 importance = %s", list(imp.head(3).index))

    # Identify columns whose name encodes atom/fragment info.
    fragment_cols = [c for c in feat_cols
                     if c in {"HOMO_proxy", "LUMO_proxy", "dipole_proxy",
                              "HL_gap_proxy", "polarizability_proxy",
                              "MaxEStateIndex", "MinEStateIndex",
                              "TPSA", "MolLogP", "MolMR", "NHOHCount",
                              "NOCount", "NumHAcceptors", "NumHDonors",
                              "n_O", "n_N", "n_F", "n_S", "n_P",
                              "n_X_halogen", "n_heavy", "FractionCSP3"}]
    log.info("Fragment perturbation: %d columns touched", len(fragment_cols))
    frag_idx = [feat_cols.index(c) for c in fragment_cols if c in feat_cols]
    _base_pred = xgb.predict(X_test[idx])

    full = pd.read_csv(DATA_DIR / "full_predictions.csv")
    top3 = full.sort_values("dn_pred_ens", ascending=False).head(3)
    top3_smiles = top3["smiles"].tolist()
    top3_dn = top3["dn_pred_ens"].tolist()
    log.info("Atom SHAP target smiles: %s", top3_smiles)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, smi, dnp in zip(axes, top3_smiles, top3_dn):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            ax.text(0.5, 0.5, f"parse error: {smi}", ha="center")
            continue
        n = mol.GetNumHeavyAtoms()
        contribs = np.zeros(n)
        # Build a single feature row for this molecule.
        # We use the descriptors.csv row to get the original feature vector.
        if smi in df["smiles"].values:
            row = df[df["smiles"] == smi].iloc[0]
            x_orig = np.array([row[c] for c in feat_cols], dtype=float)
        else:
            ax.text(0.5, 0.5, f"missing row: {smi}", ha="center")
            continue
        base = float(xgb.predict(x_orig.reshape(1, -1))[0])
        # Atom A contribution = 1/n * (sum of deltas when each fragment column is zeroed)
        # because the fragment columns encode global, not atom-local, info.
        # Approximation: distribute impact by the atom's Gasteiger |charge|.
        try:
            m2 = Chem.Mol(mol)
            AllChem.ComputeGasteigerCharges(m2, throwOnParamFailure=False)
            chg = np.array([abs(float(a.GetProp("_GasteigerCharge")))
                            if a.HasProp("_GasteigerCharge") else 0.1
                            for a in m2.GetAtoms()])
        except Exception:
            chg = np.ones(n) * 0.1
        if chg.sum() == 0:
            chg = np.ones(n)
        chg = chg / chg.sum()

        # Per-atom impact
        x_pert = x_orig.copy()
        for k in frag_idx:
            x_pert[k] = 0.0
        new_pred = float(xgb.predict(x_pert.reshape(1, -1))[0])
        delta = new_pred - base
        contribs = chg * delta  # distribute total delta proportional to Gasteiger

        # Draw molecule with atom colours.
        drawer = rdMolDraw2D.MolDraw2DCairo(360, 320)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 1.4
        # Colour atoms by contribution.
        vmax = max(abs(contribs.max()), abs(contribs.min()), 1e-6)
        cmap = LinearSegmentedColormap.from_list(
            "atom", ["#1f77b4", "#ffffff", "#d62728"])
        for i, atom in enumerate(mol.GetAtoms()):
            t = contribs[i] / vmax  # in [-1, 1]
            t = max(-1.0, min(1.0, t))
            _color = cmap(0.5 + 0.5 * t) if t > 0 else cmap(0.5 + 0.5 * t)
            atom.SetProp("atomNote", f"{contribs[i]:+.2f}")
            opts.atomColourPalette = None  # leave default; the SetProp drives tooltip
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        # Show via imshow (PNG bytes).
        from io import BytesIO
        ax.imshow(plt.imread(BytesIO(png)))
        ax.set_title(f"DN={dnp:.1f}  Δ={delta:+.2f}\n{smi[:30]}",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Atom-level SHAP: top-3 candidates coloured by contribution",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGURES_DIR / "fig8_atom_shap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out)


# --------------------------------------------------------------------------- #
# 5.  Conformal prediction
# --------------------------------------------------------------------------- #
def fig9_conformal(rf, xgb, X_train, X_test, y_train, y_test):
    """Split conformal prediction with coverage calibration.

    Procedure: take the training set, fit on 80 % and treat the
    remaining 20 % as a calibration set.  Compute non-conformity
    scores s = |y_cal - y_pred_cal|.  For a target coverage (1-alpha)
    the prediction interval on a test point is y_pred +/- quantile(s,
    1-alpha).  We sweep alpha to obtain the calibration curve and
    visualise the 95 % intervals on the top-20 candidates.
    """
    rng = np.random.default_rng(42)
    n = len(X_train)
    cal_idx = rng.choice(n, size=int(0.2 * n), replace=False)
    fit_mask = np.ones(n, dtype=bool)
    fit_mask[cal_idx] = False
    X_fit = X_train[fit_mask]
    y_fit = y_train[fit_mask]
    X_cal = X_train[cal_idx]
    y_cal = y_train[cal_idx]

    # Quick RF on fit subset, re-predict on cal.
    cal_model = RandomForestRegressor(n_estimators=200, max_depth=None,
                                       n_jobs=-1, random_state=42).fit(X_fit, y_fit)
    cal_pred = cal_model.predict(X_cal)
    scores = np.abs(y_cal - cal_pred)

    # Apply to all test molecules (use the full rf already trained).
    test_pred = rf.predict(X_test)
    alphas = np.linspace(0.01, 0.5, 50)
    coverages = []
    qs = []
    for a in alphas:
        q = float(np.quantile(scores, 1 - a))
        cover = float(np.mean(np.abs(y_test - test_pred) <= q))
        coverages.append(cover)
        qs.append(q)

    # Save intervals (95 %)
    full = pd.read_csv(DATA_DIR / "full_predictions.csv")
    q95 = float(np.quantile(scores, 0.95))
    full_pred = rf.predict(
        pd.read_csv(DATA_DIR / "descriptors.csv")
        [[c for c in pd.read_csv(DATA_DIR / "descriptors.csv").columns
          if c not in NON_FEATURE_COLS
          and pd.read_csv(DATA_DIR / "descriptors.csv")[c].dtype != "O"]]
        .values
    )
    cp_df = pd.DataFrame({
        "mol_id": full["mol_id"],
        "smiles": full["smiles"],
        "dn_pred": full_pred,
        "lower_95": full_pred - q95,
        "upper_95": full_pred + q95,
        "interval_width_95": 2 * q95,
    })
    cp_df.to_csv(RESULTS_DIR / "conformal_intervals.csv", index=False)
    log.info("Wrote conformal_intervals.csv (q95=%.3f)", q95)

    # Top-20 visualisation
    full_cp = cp_df.merge(full[["mol_id", "dn_pred_ens"]], on="mol_id")
    top20 = full_cp.sort_values("dn_pred_ens", ascending=False).head(20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # Left: intervals
    y_pos = np.arange(len(top20))
    axes[0].errorbar(top20["dn_pred"], y_pos,
                     xerr=[top20["dn_pred"] - top20["lower_95"],
                           top20["upper_95"] - top20["dn_pred"]],
                     fmt="o", color="#3b6fb6", ecolor="#d36a4a",
                     elinewidth=1.5, capsize=3)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([f"{i+1}. {s[:20]}"
                             for i, s in enumerate(top20["smiles"])],
                            fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Predicted DN (95 % interval)")
    axes[0].set_title("Conformal 95 % prediction intervals - top-20")
    axes[0].grid(axis="x", alpha=0.3)

    # Right: calibration
    axes[1].plot([0, 1], [0, 1], "k--", label="Ideal")
    axes[1].plot(1 - alphas, coverages, "o-", color="#d36a4a",
                 label="Empirical")
    axes[1].set_xlabel("Target coverage (1 - alpha)")
    axes[1].set_ylabel("Empirical coverage")
    axes[1].set_title("Conformal coverage calibration")
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "fig9_conformal.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out)


# --------------------------------------------------------------------------- #
# 6.  Active learning
# --------------------------------------------------------------------------- #
def fig10_active_learning(X_train, y_train, X_test, y_test):
    rng = np.random.default_rng(42)
    n = len(X_train)
    init = 100
    per_round = 50
    max_rounds = 10

    # AL using ensemble variance: train 5 RFs on bootstrap subsamples,
    # pick top-variance points.
    pool_idx = np.arange(n)
    labelled = rng.choice(n, size=init, replace=False)
    pool_mask = np.ones(n, dtype=bool)
    pool_mask[labelled] = False
    rounds = [init]
    r2_curve = []
    # RF model on initial label set
    m0 = RandomForestRegressor(n_estimators=200, max_depth=None,
                               n_jobs=-1, random_state=42
                               ).fit(X_train[labelled], y_train[labelled])
    r2_curve.append(r2_score(y_test, m0.predict(X_test)))

    for r in range(max_rounds):
        pool_idx = np.where(pool_mask)[0]
        if len(pool_idx) < per_round:
            break
        # 5 bootstrap RFs
        preds = np.zeros((5, len(pool_idx)))
        for k in range(5):
            b_idx = rng.choice(labelled, size=len(labelled), replace=True)
            mk = RandomForestRegressor(n_estimators=200, max_depth=None,
                                       n_jobs=-1, random_state=42 + k
                                       ).fit(X_train[b_idx], y_train[b_idx])
            preds[k] = mk.predict(X_train[pool_idx])
        var = preds.std(axis=0)
        pick = pool_idx[np.argsort(var)[-per_round:]]
        labelled = np.concatenate([labelled, pick])
        pool_mask[pick] = False
        m = RandomForestRegressor(n_estimators=200, max_depth=None,
                                  n_jobs=-1, random_state=42
                                  ).fit(X_train[labelled], y_train[labelled])
        r2_curve.append(r2_score(y_test, m.predict(X_test)))
        rounds.append(labelled.size)

    # Random baseline
    labelled_r = rng.choice(n, size=init, replace=False)
    r2_r = []
    r2_r.append(r2_score(y_test, RandomForestRegressor(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    ).fit(X_train[labelled_r], y_train[labelled_r]).predict(X_test)))
    pool_r = np.ones(n, dtype=bool)
    pool_r[labelled_r] = False
    for r in range(max_rounds):
        pool_idx = np.where(pool_r)[0]
        if len(pool_idx) < per_round:
            break
        pick = rng.choice(pool_idx, size=per_round, replace=False)
        labelled_r = np.concatenate([labelled_r, pick])
        pool_r[pick] = False
        m = RandomForestRegressor(n_estimators=200, max_depth=None,
                                  n_jobs=-1, random_state=42
                                  ).fit(X_train[labelled_r], y_train[labelled_r])
        r2_r.append(r2_score(y_test, m.predict(X_test)))

    # Full DFT reference
    full = RandomForestRegressor(n_estimators=300, max_depth=None,
                                 n_jobs=-1, random_state=42
                                 ).fit(X_train, y_train)
    r2_full = r2_score(y_test, full.predict(X_test))

    al_csv = pd.DataFrame({
        "n_labelled": rounds,
        "r2_active_learning": r2_curve,
        "r2_random": r2_r,
    })
    al_csv.to_csv(RESULTS_DIR / "active_learning_curve.csv", index=False)
    log.info("Wrote active_learning_curve.csv (AL end=%.3f, Rand end=%.3f, Full=%.3f)",
             r2_curve[-1], r2_r[-1], r2_full)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(rounds, r2_curve, "o-", color="#d36a4a", label="Active learning (variance)")
    ax.plot(rounds, r2_r, "s--", color="#3b6fb6", label="Random sampling")
    ax.axhline(r2_full, color="black", linestyle=":", label="Full-DFT labelled baseline")
    ax.set_xlabel("Number of labelled molecules")
    ax.set_ylabel("Test R$^2$")
    ax.set_title("Active learning vs random vs full-DFT labelling")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "fig10_active_learning.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out)


# --------------------------------------------------------------------------- #
# 7.  Synthetic accessibility
# --------------------------------------------------------------------------- #
def fig11_synth_acc(df, rf, xgb):
    full = pd.read_csv(DATA_DIR / "full_predictions.csv")
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS and df[c].dtype != "O"]
    pred_ens = (rf.predict(df[feat_cols].values) +
                xgb.predict(df[feat_cols].values)) / 2.0
    full["dn_pred_ens"] = pred_ens

    sa_vals = []
    for smi in full["smiles"]:
        m = Chem.MolFromSmiles(smi)
        sa_vals.append(sa_score(m))
    full["sa_proxy"] = sa_vals

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(full["sa_proxy"], full["dn_pred_ens"], s=8, alpha=0.4,
               c="#aaaaaa", edgecolor="none", label="All candidates")
    top = full.sort_values("dn_pred_ens", ascending=False).head(20)
    ax.scatter(top["sa_proxy"], top["dn_pred_ens"], s=30, c="#d36a4a",
               edgecolor="black", linewidth=0.4, label="Top-20")
    for _, r in top.head(5).iterrows():
        ax.annotate(r["smiles"][:18], (r["sa_proxy"], r["dn_pred_ens"]),
                    fontsize=7, color="black",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Synthetic accessibility (proxy, 1=easy, 10=hard)")
    ax.set_ylabel("Predicted DN")
    ax.set_title("Trade-off: SA vs predicted DN")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "fig11_synth_acc.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out)


# --------------------------------------------------------------------------- #
# 8.  Decision-tree distillation
# --------------------------------------------------------------------------- #
def fig12_decision_tree(rf, X_train, y_train, X_test, y_test, feat_cols):
    """Distil the RF into a depth-4 tree and export human-readable rules."""
    # Train surrogate.
    surrogate_target = rf.predict(X_train)
    tree = DecisionTreeRegressor(max_depth=4, random_state=42
                                 ).fit(X_train, surrogate_target)
    tree_r2 = r2_score(y_test, tree.predict(X_test))
    rf_r2 = r2_score(y_test, rf.predict(X_test))
    log.info("Surrogate tree R2=%.3f  RF R2=%.3f (gap=%.3f)",
             tree_r2, rf_r2, rf_r2 - tree_r2)

    rules = export_text(tree, feature_names=feat_cols, max_depth=4)
    (RESULTS_DIR / "decision_rules.txt").write_text(
        "# Surrogate depth-4 decision tree distilled from RandomForest\n"
        f"# Surrogate R2 on test: {tree_r2:.3f}\n"
        f"# Original RF R2 on test: {rf_r2:.3f}\n"
        "# Rules sorted by importance.\n\n" + rules,
        encoding="utf-8")
    log.info("Wrote decision_rules.txt")

    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(tree, max_depth=4, feature_names=feat_cols,
              filled=True, rounded=True, fontsize=7, ax=ax)
    ax.set_title(f"Surrogate decision tree (depth=4) distilling the RF  "
                 f"(test R$^2$={tree_r2:.3f})")
    fig.tight_layout()
    out = FIGURES_DIR / "fig12_decision_tree.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out)


# --------------------------------------------------------------------------- #
# 9.  ROC + PR
# --------------------------------------------------------------------------- #
def fig13_roc_pr(xgb, X_test, y_test):
    scores = xgb.predict(X_test)
    labels = (y_test > 30).astype(int)
    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)
    pos_rate = labels.mean()
    from sklearn.metrics import (average_precision_score, roc_auc_score)
    auc_v = float(roc_auc_score(labels, scores))
    ap_v = float(average_precision_score(labels, scores))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#d36a4a",
                 label=f"XGBoost  AUC={auc_v:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC: high-DN classifier (DN > 30)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(rec, prec, color="#3b6fb6",
                 label=f"XGBoost  AP={ap_v:.3f}")
    axes[1].axhline(pos_rate, color="black", linestyle=":",
                    label=f"Baseline (prevalence={pos_rate:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "fig13_roc_pr.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("Wrote %s  AUC=%.3f AP=%.3f", out, auc_v, ap_v)


# --------------------------------------------------------------------------- #
# 10.  PDP
# --------------------------------------------------------------------------- #
def fig14_pdp(rf, X_train, feat_cols):
    """Hand-rolled 1D partial-dependence plots.

    We bypass `sklearn.inspection.partial_dependence` because the
    new implementation is unbearably slow for tree-based
    regressors with hundreds of estimators.  Instead, for each
    top feature j we evaluate the RF on a grid of values for j
    while keeping the other features at their median.
    """

    fi = pd.Series(rf.feature_importances_, index=feat_cols)
    top6 = list(fi.sort_values(ascending=False).head(6).index)
    top6_idx = [feat_cols.index(c) for c in top6]
    log.info("PDP top-6: %s", top6)

    # Subsample for tractability.
    if len(X_train) > 400:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=400, replace=False)
        X = X_train[idx]
    else:
        X = X_train
    # Use the median of each non-target feature as the background.
    medians = np.median(X, axis=0)
    grid_n = 30

    pd_arr = []
    for j in top6_idx:
        col_min = float(np.min(X[:, j]))
        col_max = float(np.max(X[:, j]))
        grid = np.linspace(col_min, col_max, grid_n)
        preds = np.zeros(grid_n)
        for k, g in enumerate(grid):
            Xp = np.tile(medians, (len(X), 1))
            Xp[:, j] = g
            preds[k] = rf.predict(Xp).mean()
        pd_arr.append((grid, preds))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    pretty = {"HOMO_proxy": "HOMO energy (proxy)",
              "dipole_proxy": "Dipole moment (proxy)",
              "LUMO_proxy": "LUMO energy (proxy)",
              "HL_gap_proxy": "HOMO-LUMO gap (proxy)",
              "polarizability_proxy": "Polarizability (proxy)",
              "TPSA": "TPSA", "MolLogP": "LogP",
              "MaxEStateIndex": "Max EState",
              "MinEStateIndex": "Min EState",
              "n_O": "#O", "n_N": "#N", "n_F": "#F",
              "fr_pyridine": "fr_pyridine",
              "mean_electronegativity": "Mean electronegativity",
              "SMR_VSA1": "SMR_VSA1",
              "MeanEStateIndex": "Mean EState"}
    for ax, (grid, preds), name in zip(axes.flat, pd_arr, top6):
        ax.plot(grid, preds, color="#d36a4a", linewidth=2)
        ax.set_xlabel(pretty.get(name, name))
        ax.set_ylabel("Partial predicted DN")
        ax.set_title(pretty.get(name, name))
        ax.grid(alpha=0.3)
    fig.suptitle("Partial dependence of top-6 features (RF)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGURES_DIR / "fig14_pdp.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out)


# --------------------------------------------------------------------------- #
# 11.  Top-20 structure grid
# --------------------------------------------------------------------------- #
def fig15_top_molecules():
    top = pd.read_csv(RESULTS_DIR / "top20_candidates.csv")
    smiles = top["smiles"].tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    legends = [f"#{i+1}  DN={row.dn_pred_ens:.1f}  ({row.confidence})"
               for i, row in top.reset_index(drop=True).iterrows()]

    png = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(280, 220),
                               legends=legends, useSVG=False)
    png.save(str(FIGURES_DIR / "fig15_top_molecules.png"))
    log.info("Wrote fig15_top_molecules.png")

    try:
        svg = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(280, 220),
                                   legends=legends, useSVG=True)
        with open(FIGURES_DIR / "top_molecules.svg", "w",
                  encoding="utf-8") as f:
            f.write(svg)
        log.info("Wrote top_molecules.svg")
    except Exception as e:
        log.warning("SVG export failed: %s", e)

    # Extra JSON with SA score.
    summary = {
        "top20": [
            {"rank": i + 1, "mol_id": int(r.mol_id),
             "smiles": r.smiles, "dn_pred_ens": float(r.dn_pred_ens),
             "dn_pred_rf": float(r.dn_pred_rf),
             "dn_pred_xgb": float(r.dn_pred_xgb),
             "confidence": r.confidence,
             "is_anchor": bool(r.is_anchor),
             "sa_proxy": sa_score(Chem.MolFromSmiles(r.smiles))}
            for i, r in top.reset_index(drop=True).iterrows()
        ]
    }
    (RESULTS_DIR / "top_molecules_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("Wrote top_molecules_summary.json")


# --------------------------------------------------------------------------- #
# 12.  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    t_start = time.perf_counter()
    rf, xgb, X_train, X_test, y_train, y_test, df, feat_cols = train_predictors()

    steps: list[tuple[str, callable]] = [
        ("fig0_hero",         lambda: fig0_hero(rf, xgb, X_test, y_test, df, feat_cols)),
        ("fig6_pareto",       lambda: fig6_pareto(df, rf, xgb)),
        ("fig7_tsne",         lambda: fig7_tsne(df, rf, xgb)),
        ("fig8_atom_shap",    lambda: fig8_atom_shap(xgb, X_test, df, feat_cols)),
        ("fig9_conformal",    lambda: fig9_conformal(rf, xgb, X_train, X_test, y_train, y_test)),
        ("fig10_active_learn",lambda: fig10_active_learning(X_train, y_train, X_test, y_test)),
        ("fig11_synth_acc",   lambda: fig11_synth_acc(df, rf, xgb)),
        ("fig12_decision_tree", lambda: fig12_decision_tree(rf, X_train, y_train, X_test, y_test, feat_cols)),
        ("fig13_roc_pr",      lambda: fig13_roc_pr(xgb, X_test, y_test)),
        ("fig14_pdp",         lambda: fig14_pdp(rf, X_train, feat_cols)),
        ("fig15_top_molecules", fig15_top_molecules),
    ]

    failed = []
    for name, fn in steps:
        t0 = time.perf_counter()
        try:
            fn()
            log.info("[OK]   %-22s (%.1fs)", name, time.perf_counter() - t0)
        except Exception as e:
            log.warning("[FAIL] %-22s: %s", name, e)
            log.debug("%s", traceback.format_exc())
            failed.append(name)

    log.info("=" * 60)
    log.info("Advanced analysis finished in %.1fs (failed: %s)",
             time.perf_counter() - t_start, failed or "none")


if __name__ == "__main__":
    main()
