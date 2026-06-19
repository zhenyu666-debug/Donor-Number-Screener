"""Step 22: Per-molecule structure cards for the Top-20 candidates.

Generates one PNG per molecule (and a combined contact-sheet PNG) using
RDKit's default white-background renderer, decorated with:
  - rank
  - SMILES
  - IUPAC name (best-effort, falls back to the SMILES)
  - predicted DN (5-model stacking)
  - 95% interval from the EBM posterior (if available)

Reads:
  results/top20_candidates_5model.csv   - 5-model stack predictions
  results/ebm_uncertainty.csv           - EBM posterior quantiles (optional)
  data/descriptors_v2.csv               - for v2 descriptor lookups

Writes:
  figures/top20_cards/<rank>_<mol_id>.png   - one per molecule
  figures/top20_cards.png                   - 4x5 contact sheet
  results/top20_cards.csv                   - name + DN + 95% CI table
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FIGURES_DIR, RESULTS_DIR, DATA_DIR, get_logger

log = get_logger("top20_cards")


def _load_top20() -> pd.DataFrame:
    for p in ("top20_candidates_5model.csv",
              "top20_candidates_bayes.csv",
              "top20_candidates.csv"):
        f = RESULTS_DIR / p
        if f.exists():
            log.info("Reading %s", f)
            df = pd.read_csv(f)
            break
    else:
        raise FileNotFoundError("No top-20 file in results/")

    dn_col = next((c for c in df.columns
                   if c.startswith("dn_pred_stack") or c.startswith("dn_pred_ens")),
                  df.columns[2])
    df = df.sort_values(dn_col, ascending=False).reset_index(drop=True)
    df["_dn"] = df[dn_col].astype(float)
    df["_dn_col"] = dn_col
    return df


def _load_ebm_ci() -> dict[int, tuple[float, float, float, float]]:
    """Return {mol_id: (mean, std, q05, q95)} from EBM posterior if available."""
    p = RESULTS_DIR / "ebm_uncertainty.csv"
    if not p.exists():
        return {}
    ebm = pd.read_csv(p)
    out: dict[int, tuple[float, float, float, float]] = {}
    for _, r in ebm.iterrows():
        try:
            mid = int(r["mol_id"])
        except (KeyError, ValueError):
            continue
        out[mid] = (
            float(r.get("ebm_mean", np.nan)),
            float(r.get("ebm_std", np.nan)),
            float(r.get("ebm_q05", np.nan)),
            float(r.get("ebm_q95", np.nan)),
        )
    return out


def _iupac(mol: Chem.Mol | None, smi: str) -> str:
    if mol is None:
        return smi
    try:
        name = Chem.MolToIUPACName(mol)
        if name and name.strip():
            return name
    except Exception:
        pass
    return smi


def _draw_mol_image(mol: Chem.Mol, size: tuple[int, int] = (380, 320)) -> np.ndarray:
    """White-background RDKit render, returning an RGBA numpy array."""
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax.set_axis_off()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    img = Draw.MolToImage(mol, size=size)
    ax.imshow(img)
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    try:
        rgba = np.asarray(fig.canvas.buffer_rgba())
    except AttributeError:
        rgba = np.asarray(fig.canvas.tostring_rgb())
        rgba = np.stack([rgba, np.full(rgba.shape[:2], 255, dtype=np.uint8)], axis=-1)
    plt.close(fig)
    return rgba


def _make_card(smi: str, mol: Chem.Mol, rank: int, dn: float,
               ebm: tuple[float, float, float, float] | None,
               out_path: Path) -> None:
    """Single molecule PNG (320x420), white background, with labels."""
    W, H = 320, 420
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=110)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.02, 0.30, 0.96, 0.66])
    ax.set_axis_off()
    ax.set_facecolor("white")
    if mol is not None:
        img = Draw.MolToImage(mol, size=(280, 240))
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, smi, ha="center", va="center", fontsize=10,
                family="monospace", wrap=True)
    ax.set_xlim(0, 280)
    ax.set_ylim(240, 0)

    # Caption
    iupac = _iupac(mol, smi)
    cap_lines = [
        f"#{rank:>2}   DN = {dn:.2f}",
    ]
    if ebm is not None and all(np.isfinite(v) for v in ebm):
        mean, std, q05, q95 = ebm
        cap_lines.append(f"95% CI  [{q05:.2f}, {q95:.2f}]  (EBM)")
        cap_lines.append(f"posterior std = {std:.2f}")
    cap_lines.append("IUPAC:")
    # word-wrap iupac by ~38 chars
    words = iupac.split()
    wrapped: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > 38:
            wrapped.append(cur.rstrip())
            cur = w + " "
        else:
            cur += w + " "
    if cur:
        wrapped.append(cur.rstrip())
    cap_lines.extend(wrapped[:3])
    cap_lines.append("SMILES:")
    smi_disp = smi if len(smi) <= 38 else smi[:35] + "..."
    cap_lines.append(smi_disp)

    ax_cap = fig.add_axes([0.02, 0.02, 0.96, 0.27])
    ax_cap.set_axis_off()
    ax_cap.set_facecolor("white")
    txt = "\n".join(cap_lines)
    ax_cap.text(0.0, 1.0, txt, ha="left", va="top",
                family="monospace" if "SMILES" in txt else "DejaVu Sans",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#f3f5f8", edgecolor="#888", linewidth=0.6))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _make_contact_sheet(rows: list[dict], out_path: Path) -> None:
    cols = 5
    n = len(rows)
    rows_grid = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols,
                             figsize=(cols * 3.4, rows_grid * 3.7),
                             dpi=110)
    axes = np.atleast_2d(axes).reshape(rows_grid, cols)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_axis_off()
    for i, r in enumerate(rows):
        ax = axes[i // cols, i % cols]
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is not None:
            ax.imshow(Draw.MolToImage(mol, size=(280, 240)))
        ax.set_xlim(0, 280)
        ax.set_ylim(240, 0)
        # overlay caption box
        iupac = _iupac(mol, r["smiles"])
        cap = [f"#{r['rank']}  DN={r['dn_pred']:.2f}"]
        if r.get("ebm") is not None and all(np.isfinite(v) for v in r["ebm"]):
            _, _, q05, q95 = r["ebm"]
            cap.append(f"95% CI [{q05:.2f}, {q95:.2f}]")
        cap.append(iupac[:42] + ("..." if len(iupac) > 42 else ""))
        smi_disp = r["smiles"] if len(r["smiles"]) <= 38 else r["smiles"][:35] + "..."
        cap.append(smi_disp)
        ax.text(0, -8, "\n".join(cap), fontsize=7.5, va="top", ha="left",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#f3f5f8", edgecolor="#888", linewidth=0.5))
    fig.suptitle("Top-20 high-DN additive candidates  (5-model stacking)",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    df = _load_top20()
    ebm_ci = _load_ebm_ci()
    log.info("Loaded %d candidates; EBM CI for %d of them",
             len(df), len(ebm_ci))

    cards_dir = FIGURES_DIR / "top20_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, r in df.iterrows():
        rank_n = int(rank) + 1
        smi = str(r["smiles"])
        mid = int(r["mol_id"])
        mol = Chem.MolFromSmiles(smi)
        ebm = ebm_ci.get(mid)
        out_png = cards_dir / f"{rank_n:02d}_mol{mid}.png"
        _make_card(smi, mol, rank_n, float(r["_dn"]), ebm, out_png)
        iupac = _iupac(mol, smi)
        rows.append({
            "rank": rank_n,
            "mol_id": mid,
            "smiles": smi,
            "iupac": iupac,
            "dn_pred": float(r["_dn"]),
            "dn_pred_col": r["_dn_col"],
            "ebm": ebm,
            "ebm_mean": ebm[0] if ebm else np.nan,
            "ebm_q05": ebm[2] if ebm else np.nan,
            "ebm_q95": ebm[3] if ebm else np.nan,
        })

    sheet = FIGURES_DIR / "top20_cards.png"
    _make_contact_sheet(rows, sheet)
    log.info("Wrote contact sheet %s", sheet)

    out_csv = RESULTS_DIR / "top20_cards.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    print(f"\nCards dir : {cards_dir}")
    print(f"Sheet     : {sheet}")
    print(f"Table     : {out_csv}")


if __name__ == "__main__":
    main()