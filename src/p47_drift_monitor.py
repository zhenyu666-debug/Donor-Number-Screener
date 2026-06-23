"""p47_drift_monitor.py - v5: Real-time drift monitor with webhook + polling.

Provides two operational modes:
  1. Webhook server: exposes POST /drift/webhook to receive new batches
     and immediately computes PSI drift vs the stored baseline.
  2. Polling client: periodically re-runs the drift check using a new
     data source (CSV file, REST endpoint, or file watcher).

Workflow:
  python src/p47_drift_monitor.py --mode webhook --port 8001
  # In another terminal, push batches:
  python src/p47_drift_monitor.py --mode poll --input new_batch.csv --interval 3600

Outputs (under results/):
  drift_alerts.json         - recent drift alerts with severity and features
  drift_history.json        - full drift history across all batches

Usage:
  # Start webhook server
  python src/p47_drift_monitor.py --mode webhook --port 8001

  # Run polling client (every hour)
  python src/p47_drift_monitor.py --mode poll --input new.csv --interval 3600

  # Run once (one-shot)
  python src/p47_drift_monitor.py --mode check --input new.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, RESULTS_DIR, get_logger  # noqa: E402

warnings.filterwarnings("ignore")
log = get_logger("drift_monitor")

NON_FEATURE_COLS = {
    "mol_id", "smiles", "smiles_x", "smiles_y",
    "dn_rf", "dn_empirical", "dn_final", "confidence", "is_anchor",
}

N_BINS = 10
PSI_STABLE = 0.10
PSI_MINOR = 0.20
PSI_MAJOR = 0.25

DRIFT_HISTORY_FILE = RESULTS_DIR / "drift_history.json"
DRIFT_ALERTS_FILE = RESULTS_DIR / "drift_alerts.json"


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def load_baseline() -> tuple[dict[str, np.ndarray], list[str]]:
    """Load the stored drift baseline from results/drift_baseline.json."""
    path = RESULTS_DIR / "drift_baseline.json"
    if not path.exists():
        log.error(
            "drift_baseline.json not found. Run:\n"
            "  python src/18_drift_detect.py --mode baseline"
        )
        raise FileNotFoundError(str(path))

    with open(path) as f:
        data = json.load(f)

    feat_cols: list[str] = data.get("feature_names", [])
    bin_edges: dict[str, list[float]] = data.get("bin_edges", {})
    baselines: dict[str, np.ndarray] = {}
    for feat, edges in bin_edges.items():
        baselines[feat] = np.array(edges)
    return baselines, feat_cols


def compute_psi_col(new_vals: np.ndarray, baseline_edges: np.ndarray) -> float:
    """Compute PSI for a single feature column against baseline bin edges."""
    if len(new_vals) == 0:
        return 0.0
    counts_new, _ = np.histogram(new_vals, bins=baseline_edges)
    probs_new = counts_new / (counts_new.sum() + 1e-12)
    counts_base, _ = np.histogram(new_vals, bins=baseline_edges)
    probs_base = counts_base / (counts_base.sum() + 1e-12)

    probs_base = np.clip(probs_base, 1e-6, None)
    probs_new = np.clip(probs_new, 1e-6, None)
    psi = np.sum((probs_new - probs_base) * np.log(probs_new / probs_base))
    return float(psi)


def compute_drift_report(
    new_df: pd.DataFrame,
    feat_cols: list[str],
    baselines: dict[str, np.ndarray],
) -> dict:
    """Compute per-feature PSI against stored baseline."""
    results: dict[str, float] = {}
    for feat in feat_cols:
        if feat not in baselines:
            continue
        if feat not in new_df.columns:
            continue
        vals = new_df[feat].dropna().values.astype(float)
        if len(vals) == 0:
            continue
        results[feat] = compute_psi_col(vals, baselines[feat])

    # Overall PSI: mean of drifted features
    drifted = {f: p for f, p in results.items() if p > PSI_MINOR}
    overall_psi = float(np.mean(list(results.values()))) if results else 0.0

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_samples": len(new_df),
        "n_features": len(feat_cols),
        "overall_psi": overall_psi,
        "per_feature_psi": results,
        "drifted_features": drifted,
        "severity": _severity(overall_psi, len(drifted)),
    }


def _severity(overall_psi: float, n_drifted: int) -> str:
    if overall_psi > PSI_MAJOR or n_drifted > 50:
        return "CRITICAL"
    if overall_psi > PSI_MINOR or n_drifted > 20:
        return "WARNING"
    if overall_psi > PSI_STABLE or n_drifted > 5:
        return "INFO"
    return "OK"


def _alert_message(report: dict) -> str:
    sev = report["severity"]
    n = len(report.get("drifted_features", {}))
    top = sorted(report["drifted_features"].items(), key=lambda x: x[1], reverse=True)[:5]
    lines = [f"[{sev}] Drift detected: overall PSI={report['overall_psi']:.4f}"]
    if top:
        lines.append("Top drifted features:")
        for feat, psi in top:
            lines.append(f"  {feat}: PSI={psi:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------

def load_history() -> list[dict]:
    if not DRIFT_HISTORY_FILE.exists():
        return []
    with open(DRIFT_HISTORY_FILE) as f:
        return json.load(f)


def save_history(history: list[dict]) -> None:
    DRIFT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def save_alerts(alerts: list[dict]) -> None:
    DRIFT_ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=2)


def record_drift(report: dict) -> None:
    """Append a drift report to history and update alerts."""
    history = load_history()
    history.append(report)
    # Keep last 100 entries
    history = history[-100:]
    save_history(history)

    if report["severity"] in ("WARNING", "CRITICAL"):
        alerts = []
        if DRIFT_ALERTS_FILE.exists():
            with open(DRIFT_ALERTS_FILE) as f:
                alerts = json.load(f)
        alerts.append({
            "timestamp": report["timestamp"],
            "severity": report["severity"],
            "overall_psi": report["overall_psi"],
            "n_drifted": len(report.get("drifted_features", {})),
            "top_drifted": sorted(
                report.get("drifted_features", {}).items(),
                key=lambda x: x[1], reverse=True
            )[:5],
            "message": _alert_message(report),
        })
        alerts = alerts[-50:]  # Keep last 50 alerts
        save_alerts(alerts)
        log.warning(_alert_message(report))
    else:
        log.info("Drift check OK: severity=%s, overall_psi=%.4f",
                 report["severity"], report["overall_psi"])


# ---------------------------------------------------------------------------
# Polling mode
# ---------------------------------------------------------------------------

def poll_mode(input_path: Path, interval: int) -> None:
    """Continuously poll a CSV file and check for drift."""
    baselines, feat_cols = load_baseline()
    last_mtime: float = 0.0
    last_size: int = 0

    log.info("Polling mode: %s every %ds", input_path, interval)
    while True:
        try:
            if not input_path.exists():
                log.warning("Input file not found: %s", input_path)
                time.sleep(interval)
                continue

            mtime = input_path.stat().st_mtime
            size = input_path.stat().st_size

            if mtime == last_mtime and size == last_size:
                time.sleep(min(interval, 30))
                continue

            last_mtime = mtime
            last_size = size

            log.info("New data detected: %s (%.1f KB)", input_path, size / 1024)
            new_df = pd.read_csv(input_path)

            # Align columns to baseline features
            aligned = new_df[[c for c in feat_cols if c in new_df.columns]]
            log.info("Aligned features: %d / %d", aligned.shape[1], len(feat_cols))

            report = compute_drift_report(aligned, feat_cols, baselines)
            record_drift(report)

            log.info("Next check in %ds ...", interval)
            time.sleep(interval)

        except KeyboardInterrupt:
            log.info("Polling stopped by user")
            break
        except Exception as exc:
            log.error("Polling error: %s", exc)
            time.sleep(interval)


def check_mode(input_path: Path) -> None:
    """One-shot drift check against a CSV file."""
    baselines, feat_cols = load_baseline()
    new_df = pd.read_csv(input_path)
    aligned = new_df[[c for c in feat_cols if c in new_df.columns]]
    report = compute_drift_report(aligned, feat_cols, baselines)
    record_drift(report)

    # Pretty print
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Webhook server mode
# ---------------------------------------------------------------------------

def webhook_server(host: str, port: int) -> None:
    """Start a FastAPI webhook server for real-time drift detection."""
    try:
        from fastapi import FastAPI, Request
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        log.error("fastapi or uvicorn not installed. Run: pip install fastapi uvicorn")
        return

    baselines, feat_cols = load_baseline()

    app = FastAPI(title="Drift Monitor Webhook")

    class DriftPayload(BaseModel):
        smiles: Optional[list[str]] = None
        descriptors: Optional[dict[str, list[float]]] = None
        metadata: Optional[dict] = None

    class DriftResponse(BaseModel):
        status: str
        severity: str
        overall_psi: float
        n_drifted: int
        top_drifted: dict[str, float]

    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": "drift_monitor_webhook"}

    @app.post("/drift/webhook", response_model=DriftResponse)
    async def drift_webhook(payload: DriftPayload):
        # Build DataFrame from payload
        if payload.descriptors:
            df = pd.DataFrame(payload.descriptors)
        elif payload.smiles:
            # Compute descriptors on the fly (requires RDKit)
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                rows = []
                for smi in payload.smiles:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    row = {"smiles": smi}
                    for fn in [
                        Descriptors.MolWt, Descriptors.MolLogP,
                        Descriptors.NumHDonors, Descriptors.NumHAcceptors,
                        Descriptors.NumHeteroatoms, Descriptors.TPSA,
                    ]:
                        row[fn.__name__] = float(fn(mol))
                    rows.append(row)
                df = pd.DataFrame(rows)
            except ImportError:
                return DriftResponse(
                    status="error",
                    severity="N/A",
                    overall_psi=0.0,
                    n_drifted=0,
                    top_drifted={},
                )
        else:
            return DriftResponse(
                status="error",
                severity="N/A",
                overall_psi=0.0,
                n_drifted=0,
                top_drifted={},
            )

        aligned = df[[c for c in feat_cols if c in df.columns]]
        report = compute_drift_report(aligned, feat_cols, baselines)
        record_drift(report)

        top = dict(sorted(
            report.get("drifted_features", {}).items(),
            key=lambda x: x[1], reverse=True
        )[:5])

        return DriftResponse(
            status="ok",
            severity=report["severity"],
            overall_psi=report["overall_psi"],
            n_drifted=len(report.get("drifted_features", {})),
            top_drifted=top,
        )

    @app.get("/drift/history")
    async def drift_history():
        return load_history()

    @app.get("/drift/alerts")
    async def drift_alerts():
        if DRIFT_ALERTS_FILE.exists():
            with open(DRIFT_ALERTS_FILE) as f:
                return json.load(f)
        return []

    @app.post("/drift/baseline/refresh")
    async def refresh_baseline():
        """Recompute baseline from the current full library."""
        try:
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "src/18_drift_detect.py", "--mode", "baseline"],
                capture_output=True, text=True,
            )
            return {"status": "ok" if result.returncode == 0 else "error", "output": result.stdout}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    log.info("Starting drift webhook server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="p47 Drift Monitor: webhook server + polling client for real-time PSI drift detection"
    )
    p.add_argument(
        "--mode",
        choices=["webhook", "poll", "check"],
        default="check",
        help="'webhook' starts the server; 'poll' watches a file; 'check' runs once",
    )
    p.add_argument("--port", type=int, default=8001, help="Webhook server port")
    p.add_argument("--host", default="0.0.0.0", help="Webhook server host")
    p.add_argument("--input", type=Path, help="CSV file for poll/check mode")
    p.add_argument("--interval", type=int, default=3600,
                   help="Polling interval in seconds (default: 3600)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    log.info("=" * 60)
    log.info("p47 Drift Monitor (v5)")
    log.info("Mode: %s", args.mode)
    log.info("=" * 60)

    if args.mode == "webhook":
        webhook_server(args.host, args.port)
    elif args.mode == "poll":
        if not args.input:
            log.error("--input required for poll mode")
            return
        poll_mode(args.input, args.interval)
    elif args.mode == "check":
        if not args.input:
            log.error("--input required for check mode")
            return
        check_mode(args.input)


if __name__ == "__main__":
    main()
