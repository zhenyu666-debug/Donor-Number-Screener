"""streamlit_app.py - v5: Streamlit demo client for the FastAPI REST server.

Provides an interactive web UI to:
  - Predict DN for a single SMILES
  - Screen a batch of SMILES
  - View reliability / calibration diagram
  - Monitor drift (if drift_baseline.json exists)

Requires the FastAPI server to be running:
  uvicorn src.15_api_server:app --host 0.0.0.0 --port 8000

Or use the built-in simulation mode when the API is unavailable.

Usage:
  pip install streamlit requests pandas plotly
  streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import requests

# Streamlit (graceful fallback if not installed)
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
API_BASE = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _call(endpoint: str, payload: dict, timeout: int = 30) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return None
    except Exception as exc:
        return {"_error": str(exc)}


def predict_single(smiles: str) -> dict:
    result = _call("/predict_smiles", {"smiles": smiles})
    if result is None:
        # Fallback: simulate a response
        try:
            from rdkit import Chem
            from rdkit.Chem.Descriptors import ExactMolWt
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": f"Invalid SMILES: {smiles}"}
            mw = ExactMolWt(mol)
            # Rough DN estimate: MW-based heuristic for demo
            dn_estimate = max(5.0, min(45.0, 25.0 + (mw - 200) * 0.05))
            return {
                "smiles": smiles,
                "dn_pred": round(dn_estimate, 2),
                "dn_lower": round(dn_estimate - 3.0, 2),
                "dn_upper": round(dn_estimate + 3.0, 2),
                "model_std": 1.5,
                "n_models": 5,
                "region_index": 2,
                "region_name": "intermediate_solvent",
                "_simulated": True,
            }
        except ImportError:
            return {"error": "rdkit not available and API server is not running"}
    return result or {"error": "API call failed"}


def screen_batch(smiles_list: list[str], k: int = 20) -> dict:
    result = _call("/screen_top", {"smiles_list": smiles_list, "k": k, "include_pareto": True})
    if result is None:
        return {"error": "API server unavailable"}
    return result


def health_check() -> bool:
    result = _call("/health", {}, timeout=5)
    return result is not None


# ---------------------------------------------------------------------------
# Calibration plot
# ---------------------------------------------------------------------------

def plot_calibration() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        st.info("matplotlib not available for calibration plot")
        return

    metrics_path = PROJECT_ROOT / "results" / "calibration_metrics.json"
    curve_path = PROJECT_ROOT / "results" / "calibration_summary.csv"

    if not metrics_path.exists():
        st.info("Run `python src/p46_calibration.py` first to generate calibration data.")
        return

    import json
    with open(metrics_path) as f:
        metrics = json.load(f)

    cal = metrics.get("calibration", {})

    if curve_path.exists():
        df = pd.read_csv(curve_path)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect")
            if "mean_predicted" in df.columns and "frac_positives" in df.columns:
                mp = df["mean_predicted"]
                fp = df["frac_positives"]
                ax.plot(mp, fp, "ro-", lw=1.5, alpha=0.7, label="Raw")
            if "iso_mean_predicted" in df.columns and "iso_frac_positives" in df.columns:
                iso_mp = df["iso_mean_predicted"]
                iso_fp = df["iso_frac_positives"]
                ax.plot(iso_mp, iso_fp, "g^-", lw=1.5, alpha=0.7, label="Isotonic")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted (normalised)")
            ax.set_ylabel("Fraction of positives")
            ax.set_title("Reliability Diagram — 5-Model DN Ensemble")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.markdown("### ECE Scores")
            for method, vals in cal.items():
                ece = vals.get("ece", 0)
                cov = vals.get("coverage_95", 0)
                st.metric(method.capitalize(), f"{ece:.3f}", f"Coverage {cov:.1%}")

    else:
        st.info("calibration_summary.csv not found")


# ---------------------------------------------------------------------------
# Drift monitor
# ---------------------------------------------------------------------------

def render_drift() -> None:
    st.markdown("### Drift Monitor")

    drift_path = PROJECT_ROOT / "results" / "drift_report.json"
    baseline_path = PROJECT_ROOT / "results" / "drift_baseline.json"

    if not drift_path.exists():
        st.info(
            "No drift report found. Run:\n"
            "```bash\n"
            "python src/18_drift_detect.py --mode baseline\n"
            "python src/18_drift_detect.py --mode batch --input your_new_data.csv\n"
            "```"
        )
        return

    import json
    with open(drift_path) as f:
        drift = json.load(f)

    features = drift.get("per_feature_psi", {})
    if not features:
        st.info("No per-feature PSI data in drift report")
        return

    df = pd.DataFrame([
        {"feature": k, "psi": float(v)}
        for k, v in features.items()
        if isinstance(v, (int, float))
    ])
    df = df.sort_values("psi", ascending=False)

    threshold = 0.20
    drifted = df[df["psi"] > threshold]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            df.head(30),
            use_container_width=True,
            hide_index=True,
        )
    with col2:
        st.metric("Total features", len(df))
        st.metric("Drifted (PSI > 0.20)", len(drifted), delta_color="inverse")
        if not drifted.empty:
            worst = drifted.iloc[0]
            st.metric("Worst feature", worst["feature"], f"PSI={worst['psi']:.3f}")

    if not drifted.empty:
        st.warning(
            f"{len(drifted)} features drifted (PSI > {threshold}). "
            "Consider retraining the model on the new data."
        )


# ---------------------------------------------------------------------------
# SMILES input helpers
# ---------------------------------------------------------------------------

_COMMON_SOLVENTS = [
    ("DMSO", "CS(=O)C"),
    ("DME", "CCOCCOC"),
    ("DOL", "C1CCOC1"),
    ("Acetonitrile (AN)", "CC#N"),
    ("EC", "C1COC(=O)O1"),
    ("PC", "CC(=O)OC1(C)CCCC1"),
    ("DMC", "CNC(=O)OC"),
    ("EMC", "CC(=O)OCC"),
    ("DEC", "CC(=O)OCC"),
    ("FEC", "OCC1OC(=O)OC1(F)F"),
    ("Water", "O"),
    ("Methanol", "CO"),
    ("Ethanol", "CCO"),
    ("Acetone", "CC(=O)C"),
    ("THF", "C1CCOC1"),
    ("DMF", "CN(C)C=O"),
    ("NMP", "CN1CCCC1=O"),
    ("DMI", "Cn1cnc(c1=O)-n(C)C"),
    ("ACN", "CC#N"),
    ("TFEP", "CC(F)(F)F"),
]

_DEMO_SMILES_POOL = [
    "CC1CN(C(C1)(C)C)c1nc(ccc1C(=O)NS(=O)(=O)c1cn(nc1C)C)n1ccc(n1)OCC(C(F)(F)F)(C)C",
    "CC(C)(CO)C1=CC2=CC(NC(=O)C3(CC3)C4=CC=C5OC(O2)(F)F)C(F)=C2N1C[C@@H](O)CO",
    "CC(C)(C)C1=CC(=C(O)C=C1NC(=O)C2=CNC3=CC=CC=C3C2=O)C(C)(C)C",
    "CS(=O)C",
    "CCOCCOC",
    "CC1CCOC1",
    "CC#N",
]


def dn_region_label(idx: int) -> str:
    labels = {
        0: "Very Weak",
        1: "Weak",
        2: "Intermediate",
        3: "Strong",
        4: "Very Strong",
    }
    return labels.get(idx, "Unknown")


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def run_app() -> None:
    st.set_page_config(
        page_title="Donor-Number Screener — Demo",
        page_icon="🧪",
        layout="wide",
    )

    st.title("Donor-Number Screening Pipeline — v5 Demo")
    st.markdown(
        "**v5 features**: SHAP interaction values, calibration reliability diagrams, "
        "and real-time drift monitoring. "
        "Start the API server: `uvicorn src.15_api_server:app --port 8000`"
    )

    tab_single, tab_batch, tab_calibration, tab_drift = st.tabs([
        "Single SMILES", "Batch Screen", "Calibration", "Drift Monitor"
    ])

    # ---- Single SMILES --------------------------------------------------------
    with tab_single:
        col_query, col_result = st.columns([1, 2])

        with col_query:
            smiles_input = st.text_input(
                "SMILES",
                placeholder="e.g. CS(=O)C (DMSO)",
                help="Canonical SMILES for a molecule",
            )
            with st.expander("Common solvents"):
                for name, smi in _COMMON_SOLVENTS:
                    if st.button(f"{name}", key=f"btn_{name}"):
                        smiles_input = smi
                        st.rerun()

            predict_btn = st.button("Predict DN", type="primary", disabled=not smiles_input)

        with col_result:
            if predict_btn and smiles_input:
                with st.spinner("Calling API..."):
                    result = predict_single(smiles_input)

                if "error" in result:
                    st.error(result["error"])
                elif "_error" in result:
                    st.error(result["_error"])
                else:
                    dn = result.get("dn_pred", 0)
                    lo = result.get("dn_lower", 0)
                    hi = result.get("dn_upper", 0)
                    std = result.get("model_std", 0)
                    idx = result.get("region_index", 0)
                    n_models = result.get("n_models", 0)
                    simulated = result.get("_simulated", False)

                    if simulated:
                        st.caption("Simulated response (API offline, rdkit available)")

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("DN Prediction", f"{dn:.2f}")
                    c2.metric("Lower (95% CI)", f"{lo:.2f}")
                    c3.metric("Upper (95% CI)", f"{hi:.2f}")
                    c4.metric("Model Std", f"{std:.2f}")
                    c5.metric("Solvent Region", dn_region_label(idx))

                    if "descriptors" in result:
                        st.dataframe(
                            pd.DataFrame(result["descriptors"]).T,
                            use_container_width=True,
                        )

    # ---- Batch Screen ---------------------------------------------------------
    with tab_batch:
        st.markdown("Paste multiple SMILES (one per line or comma-separated)")
        batch_input = st.text_area(
            "SMILES list",
            placeholder="CS(=O)C\nCCOCCOC\nCC1CCOC1\n...",
            height=200,
        )
        k = st.slider("Top-K results", 5, 50, 20)

        if st.button("Screen", type="primary", disabled=not batch_input):
            lines = [l.strip() for l in batch_input.replace(",", "\n").splitlines() if l.strip()]
            with st.spinner(f"Screening {len(lines)} molecules..."):
                result = screen_batch(lines, k=k)

            if "error" in result:
                st.error(result["error"])
            else:
                candidates = result.get("candidates", [])
                if candidates:
                    df = pd.DataFrame(candidates)
                    st.dataframe(
                        df[["smiles", "dn_pred", "region_name"]].rename(
                            columns={"smiles": "SMILES", "dn_pred": "DN Pred", "region_name": "Region"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No candidates returned")

    # ---- Calibration ----------------------------------------------------------
    with tab_calibration:
        st.markdown("### Reliability Diagram")
        plot_calibration()

    # ---- Drift Monitor --------------------------------------------------------
    with tab_drift:
        render_drift()

    # ---- Footer ---------------------------------------------------------------
    st.divider()
    st.caption(
        "Donor-Number-Screener v5 | "
        "GitHub: zhenyu666-debug/Donor-Number-Screener | "
        "Run `python src/14_shap_explain.py && python src/p46_calibration.py` first"
    )


def main() -> None:
    if not _STREAMLIT_AVAILABLE:
        print("ERROR: streamlit not installed.")
        print("  pip install streamlit")
        print("  streamlit run streamlit_app.py")
        return
    run_app()


if __name__ == "__main__":
    main()
