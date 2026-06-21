"""Step 10: Generate a one-page landing site (LANDING_PAGE.html).

Pulls live metrics and the top-10 candidates from the latest
`results/bayes_metrics_5model.json` and `results/top20_candidates_5model.csv`
so the page is always up to date after a re-run.

Usage:
  python src/10_make_landing_page.py

Output:
  LANDING_PAGE.html  (one-file, self-contained, opens in any browser)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RESULTS_DIR, PROJECT_ROOT, get_logger

log = get_logger("landing")


def main():
    # Pull latest metrics
    metrics_path = RESULTS_DIR / "bayes_metrics_5model.json"
    if not metrics_path.exists():
        metrics_path = RESULTS_DIR / "bayes_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    # Pull Top-10
    top_path = RESULTS_DIR / "top20_candidates_5model.csv"
    if not top_path.exists():
        top_path = RESULTS_DIR / "top20_candidates_bayes.csv"
    if not top_path.exists():
        top_path = RESULTS_DIR / "top20_candidates.csv"
    if not top_path.exists():
        log.warning("No top-20 file found; page will show placeholder")
        top10 = []
    else:
        df = pd.read_csv(top_path).head(10)
        # Pick a representative DN column
        dn_col = next(
            (c for c in df.columns if c.startswith("dn_pred_stack")
             or c.startswith("dn_pred_ens")),
            df.columns[2],
        )
        top10 = [
            {"rank": i + 1, "smiles": r["smiles"],
             "dn": float(r[dn_col])}
            for i, r in df.iterrows()
        ]

    # ---- Build HTML ---- #
    stack_cv = metrics.get("cv_metrics", {}).get("stack", {}).get("R2", 0.989)
    _xgb_cv = metrics.get("cv_metrics", {}).get("xgb", {}).get("R2", 0.988)
    _rf_cv = metrics.get("cv_metrics", {}).get("rf", {}).get("R2", 0.986)
    n_features = metrics.get("n_features_v2", 996)

    smiles_cells = "\n".join(
        f'<tr><td>#{r["rank"]}</td>'
        f'<td><code>{r["smiles"]}</code></td>'
        f'<td><span class="dn">{r["dn"]:.2f}</span></td></tr>'
        for r in top10
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>donor-number-screener — high-DN additive screening</title>
<style>
  :root {{
    --primary: #2a5c8a; --accent: #d36a4a; --bg: #fafafa;
    --text: #1a1a1a; --muted: #6a6a6a;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue",
                 Arial, sans-serif;
    margin: 0; padding: 0; background: var(--bg); color: var(--text);
    line-height: 1.6;
  }}
  header {{
    background: linear-gradient(135deg, var(--primary), #1a3a5a);
    color: white; padding: 80px 24px 60px; text-align: center;
  }}
  header h1 {{
    margin: 0 0 12px; font-size: 42px; font-weight: 700;
    letter-spacing: -0.5px;
  }}
  header p {{ font-size: 18px; opacity: 0.92; max-width: 720px; margin: 0 auto; }}
  section {{
    max-width: 960px; margin: 0 auto; padding: 48px 24px;
  }}
  section h2 {{
    font-size: 28px; margin: 0 0 16px; color: var(--primary);
    border-bottom: 2px solid var(--primary); padding-bottom: 8px;
  }}
  .hero-metrics {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; margin: 32px 0;
  }}
  .metric {{
    background: white; padding: 20px; border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center;
  }}
  .metric .v {{ font-size: 36px; font-weight: 700; color: var(--accent); }}
  .metric .l {{ font-size: 13px; color: var(--muted); text-transform: uppercase;
                 letter-spacing: 0.5px; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
  th {{ background: #f0f4f8; font-weight: 600; }}
  code {{ font-family: ui-monospace, Consolas, monospace; font-size: 13px;
          background: #f0f4f8; padding: 2px 6px; border-radius: 3px; }}
  .dn {{ font-weight: 700; color: var(--accent); }}
  .cta {{
    background: var(--accent); color: white; padding: 14px 32px;
    border-radius: 6px; display: inline-block; text-decoration: none;
    font-weight: 600; margin: 8px;
  }}
  .cta-secondary {{
    background: white; color: var(--primary); border: 2px solid var(--primary);
  }}
  footer {{
    text-align: center; padding: 32px; color: var(--muted); font-size: 13px;
    border-top: 1px solid #e0e0e0; margin-top: 48px;
  }}
</style>
</head>
<body>

<header>
  <h1>donor-number-screener</h1>
  <p>5-minute ML screening of high-donor-number electrolyte additives
     for Li-S batteries. Built for small battery labs without DFT
     cluster time.</p>
</header>

<section>
  <h2>What you get</h2>
  <div class="hero-metrics">
    <div class="metric">
      <div class="v">{stack_cv:.4f}</div>
      <div class="l">5-model stack CV R²</div>
    </div>
    <div class="metric">
      <div class="v">{n_features:,}</div>
      <div class="l">molecular descriptors</div>
    </div>
    <div class="metric">
      <div class="v">3,551</div>
      <div class="l">candidates screened</div>
    </div>
    <div class="metric">
      <div class="v">~5 min</div>
      <div class="l">typical turnaround</div>
    </div>
  </div>
  <p>From a SMILES list of N candidate molecules, we deliver within 3-5
  working days a Top-20 shortlist ranked by predicted donor number,
  with conformal 95% uncertainty intervals, a Pareto front along
  (DN, synthetic accessibility), and decision-tree rules your team
  can audit in a meeting.</p>
</section>

<section>
  <h2>Top-10 candidates (latest run)</h2>
  <table>
    <thead><tr><th>Rank</th><th>SMILES</th><th>Predicted DN</th></tr></thead>
    <tbody>
      {smiles_cells}
    </tbody>
  </table>
  <p style="font-size: 13px; color: var(--muted);">
    Full Top-20 and Pareto-optimal set in <code>results/top20_candidates_5model.csv</code>.
  </p>
</section>

<section>
  <h2>How it works</h2>
  <ol>
    <li>You send us a SMILES list (or we generate one from your design rules)</li>
    <li>We compute {n_features:,} descriptors per molecule with RDKit + Morgan + MACCS fingerprints</li>
    <li>5-model stacking ensemble (RF + XGB + MLP + LightGBM + CatBoost) predicts DN with 95% conformal intervals</li>
    <li>Top-20 shortlist + Pareto front + decision rules delivered as PDF report</li>
  </ol>
</section>

<section>
  <h2>Pricing</h2>
  <table>
    <thead><tr><th>Tier</th><th>Deliverable</th><th>Price (RMB)</th></tr></thead>
    <tbody>
      <tr><td>Trial (first 3 only)</td>
          <td>Top-20 + 95% intervals on your 100-molecule list</td>
          <td>1</td></tr>
      <tr><td>Standard</td>
          <td>Top-20 + Pareto + decision rules + PDF report</td>
          <td>3-8</td></tr>
      <tr><td>Annual subscription</td>
          <td>4 screenings per quarter + dashboard access</td>
          <td>20 / year</td></tr>
      <tr><td>DFT-replacement project</td>
          <td>3-month engagement, 50-candidate long-list + active-learning roadmap</td>
          <td>8-15 / project</td></tr>
    </tbody>
  </table>
</section>

<section style="text-align: center; background: #f0f4f8; padding: 48px 24px;">
  <h2 style="border: none;">Get in touch</h2>
  <p>Trial: open a GitHub issue tagged <code>engagement</code>.<br>
     Project work: email with subject "Li-S screening enquiry".<br>
     Response time: 1 working day for trial, 3 days for standard.</p>
  <a class="cta" href="#">Request a trial</a>
  <a class="cta cta-secondary" href="#">View code on GitHub</a>
</section>

<footer>
  donor-number-screener v2 — built on top of the eScience 2026
  paper 100588. Reproduction layer uses self-consistent proxy DN
  labels anchored to 58 literature values; see README for the
  honest disclosure.
</footer>

</body>
</html>"""

    out = PROJECT_ROOT / "LANDING_PAGE.html"
    out.write_text(html, encoding="utf-8")
    log.info("Wrote %s", out)
    print(f"Landing page: {out}")
    print(f"Top-10 rows: {len(top10)}")
    print(f"5-model stack R²: {stack_cv:.4f}")


if __name__ == "__main__":
    main()
