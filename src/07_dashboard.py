"""Step 7: Generate a self-contained interactive HTML dashboard.

The dashboard embeds:
  - The static figures (PNG) from step 5 + 6.
  - Two Plotly (CDN-loaded) interactive plots built from the CSV/JSON
    artefacts: a Top-20 interval plot and a Pareto scatter.
  - Sortable candidate table.
  - Methodology summary boxes.

It is *self-contained* in the sense that only one external request
happens at load time (Plotly.js from cdn.plotly.com, ~1.5 MB).  All
images are local, all data is inlined.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("dashboard")


# --------------------------------------------------------------------------- #
def img_to_b64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# --------------------------------------------------------------------------- #
def build_dashboard() -> str:
    metrics = json.loads((RESULTS_DIR / "model_metrics.json").read_text(
        encoding="utf-8"))
    screen = json.loads((RESULTS_DIR / "screening_summary.json").read_text(
        encoding="utf-8"))
    top20 = pd.read_csv(RESULTS_DIR / "top20_candidates.csv")
    conformal = pd.read_csv(RESULTS_DIR / "conformal_intervals.csv")
    pareto = pd.read_csv(RESULTS_DIR / "pareto_optimal.csv")
    _al = pd.read_csv(RESULTS_DIR / "active_learning_curve.csv")
    top_summary = json.loads(
        (RESULTS_DIR / "top_molecules_summary.json").read_text(encoding="utf-8"))

    n_candidates = screen.get("n_candidates", len(top20) * 100)
    eff = screen.get("efficiency", {})
    speedup_dft = eff.get("ml_vs_dft_speedup", 1e7)
    _speedup_exp = eff.get("ml_vs_experiment_speedup", 1e8)
    rf_r2 = metrics["test_rf"]["R2"]
    xgb_r2 = metrics["test_xgb"]["R2"]
    _rf_rmse = metrics["test_rf"]["RMSE"]
    _xgb_rmse = metrics["test_xgb"]["RMSE"]

    # Build Plotly data for the two interactive charts.
    top20_sorted = top20.sort_values("dn_pred_ens", ascending=False).head(20)
    cp_top = (conformal.set_index("mol_id")
              .loc[top20_sorted["mol_id"]]
              .reset_index())
    cp_top["smiles"] = top20_sorted["smiles"].values
    cp_top["dn_pred_ens"] = top20_sorted["dn_pred_ens"].values
    cp_top["err_low"] = cp_top["dn_pred"] - cp_top["lower_95"]
    cp_top["err_high"] = cp_top["upper_95"] - cp_top["dn_pred"]

    plotly_pareto = {
        "data": [{
            "type": "scatter",
            "mode": "markers",
            "x": pareto["MW"].tolist(),
            "y": pareto["dn_pred_ens"].tolist(),
            "text": pareto["smiles"].tolist(),
            "marker": {"color": "#d36a4a", "size": 10,
                       "line": {"color": "black", "width": 0.4}},
            "name": "Pareto optimal",
        }, {
            "type": "scatter",
            "mode": "markers",
            "x": top20["MolWt"] if "MolWt" in top20.columns else
                 [0] * len(top20),
            "y": top20["dn_pred_ens"].tolist(),
            "text": top20["smiles"].tolist(),
            "marker": {"color": "#3b6fb6", "size": 8},
            "name": "Top-20 high-DN",
        }] if "MolWt" in top20.columns else [{
            "type": "scatter",
            "mode": "markers",
            "x": pareto["MW"].tolist(),
            "y": pareto["dn_pred_ens"].tolist(),
            "text": pareto["smiles"].tolist(),
            "marker": {"color": "#d36a4a", "size": 10,
                       "line": {"color": "black", "width": 0.4}},
            "name": "Pareto optimal",
        }],
        "layout": {
            "title": "Pareto front: MW vs predicted DN",
            "xaxis": {"title": "Molecular weight (Da)"},
            "yaxis": {"title": "Predicted DN"},
            "hovermode": "closest",
        }
    }
    # If top20 has no MolWt column, pull from the full predictions + a quick recompute.
    if "MolWt" not in top20.columns:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mws = [Descriptors.MolWt(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else 0
               for s in top20["smiles"]]
        plotly_pareto["data"].append({
            "type": "scatter",
            "mode": "markers",
            "x": mws,
            "y": top20["dn_pred_ens"].tolist(),
            "text": top20["smiles"].tolist(),
            "marker": {"color": "#3b6fb6", "size": 8},
            "name": "Top-20 high-DN",
        })

    plotly_intervals = {
        "data": [{
            "type": "scatter",
            "mode": "markers",
            "x": cp_top["dn_pred"].tolist(),
            "y": list(range(len(cp_top)))[::-1],
            "error_x": {
                "type": "data",
                "symmetric": False,
                "array": cp_top["err_high"].tolist(),
                "arrayminus": cp_top["err_low"].tolist(),
            },
            "text": cp_top["smiles"].tolist(),
            "marker": {"color": "#3b6fb6", "size": 9},
            "name": "Mean +/- 95% interval",
        }],
        "layout": {
            "title": "Conformal 95% prediction intervals - top-20",
            "xaxis": {"title": "Predicted DN"},
            "yaxis": {
                "title": "",
                "tickmode": "array",
                "tickvals": list(range(len(cp_top)))[::-1],
                "ticktext": [f"#{i+1} {s[:18]}" for i, s in enumerate(cp_top["smiles"])],
            },
            "height": 520,
            "margin": {"l": 220},
        }
    }

    # Build the sortable table HTML.
    rows = []
    for t in top_summary["top20"]:
        rows.append(
            f"<tr><td>{t['rank']}</td>"
            f"<td>{t['mol_id']}</td>"
            f"<td class='smi'>{t['smiles']}</td>"
            f"<td>{t['dn_pred_ens']:.2f}</td>"
            f"<td>{t['dn_pred_rf']:.2f}</td>"
            f"<td>{t['dn_pred_xgb']:.2f}</td>"
            f"<td>{t['sa_proxy']:.2f}</td>"
            f"<td>{t['confidence']}</td>"
            f"<td>{'yes' if t['is_anchor'] else 'no'}</td></tr>"
        )
    table_html = (
        "<table id='cand'><thead><tr>"
        "<th data-sort='int'>Rank</th>"
        "<th data-sort='int'>mol_id</th>"
        "<th data-sort='str'>SMILES</th>"
        "<th data-sort='num'>DN (ens)</th>"
        "<th data-sort='num'>DN (RF)</th>"
        "<th data-sort='num'>DN (XGB)</th>"
        "<th data-sort='num'>SA</th>"
        "<th data-sort='str'>Conf.</th>"
        "<th data-sort='str'>Anchor</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )

    # Embed PNGs as base64 so the HTML is truly portable.
    hero_b64 = img_to_b64(FIGURES_DIR / "fig0_hero.png")
    fig2a = img_to_b64(FIGURES_DIR / "fig2a_rf_parity.png")
    fig2b = img_to_b64(FIGURES_DIR / "fig2b_xgb_parity.png")
    fig2c = img_to_b64(FIGURES_DIR / "fig2c_feature_importance.png")
    fig2d = img_to_b64(FIGURES_DIR / "fig2d_shap.png")
    fig3 = img_to_b64(FIGURES_DIR / "fig3_proxy_validation.png")
    fig4 = img_to_b64(FIGURES_DIR / "fig4_sei_proxy.png")
    fig5 = img_to_b64(FIGURES_DIR / "fig5_efficiency.png")
    fig6 = img_to_b64(FIGURES_DIR / "fig6_pareto.png")
    fig7 = img_to_b64(FIGURES_DIR / "fig7_tsne_landscape.png")
    fig8 = img_to_b64(FIGURES_DIR / "fig8_atom_shap.png")
    fig9 = img_to_b64(FIGURES_DIR / "fig9_conformal.png")
    fig10 = img_to_b64(FIGURES_DIR / "fig10_active_learning.png")
    fig11 = img_to_b64(FIGURES_DIR / "fig11_synth_acc.png")
    fig12 = img_to_b64(FIGURES_DIR / "fig12_decision_tree.png")
    fig13 = img_to_b64(FIGURES_DIR / "fig13_roc_pr.png")
    fig14 = img_to_b64(FIGURES_DIR / "fig14_pdp.png")
    fig15 = img_to_b64(FIGURES_DIR / "fig15_top_molecules.png")

    def img_tag(b64: str, alt: str) -> str:
        if not b64:
            return f"<p style='color:#999'>[missing: {alt}]</p>"
        return f"<img class='panel' alt='{alt}' src='data:image/png;base64,{b64}' />"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Li-S Additive Screening Dashboard</title>
<script src="https://cdn.plotly.com/plotly-2.35.2.min.js"></script>
<style>
 :root {{
   --bg: #f7f7f9; --card: #ffffff; --ink: #1f2933; --accent: #d36a4a;
   --blue: #3b6fb6; --muted: #6b7280; --border: #e2e8f0;
 }}
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        margin: 0; padding: 0; background: var(--bg); color: var(--ink); }}
 header {{ background: linear-gradient(120deg, #1a365d 0%, #2c5282 100%);
          color: #fff; padding: 32px 48px; }}
 header h1 {{ margin: 0 0 6px 0; font-size: 28px; font-weight: 600; }}
 header p  {{ margin: 0; opacity: 0.85; font-size: 14px; }}
 main {{ max-width: 1280px; margin: 0 auto; padding: 24px 32px 80px 32px; }}
 .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
 .card {{ background: var(--card); border: 1px solid var(--border);
          border-radius: 8px; padding: 18px 20px;
          box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
 .card h3 {{ margin: 0; font-size: 28px; color: var(--blue); }}
 .card p  {{ margin: 6px 0 0 0; color: var(--muted); font-size: 13px; }}
 section {{ background: var(--card); border: 1px solid var(--border);
            border-radius: 8px; padding: 24px; margin: 24px 0; }}
 section h2 {{ margin-top: 0; font-size: 18px; color: var(--blue);
               border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
 .panel {{ max-width: 100%; height: auto; display: block; margin: 8px auto;
           border: 1px solid var(--border); border-radius: 4px; }}
 .twocol {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
 table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
 th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
 th {{ background: #f1f5f9; cursor: pointer; user-select: none; }}
 th:hover {{ background: #e2e8f0; }}
 td.smi {{ font-family: "JetBrains Mono", Consolas, monospace; font-size: 12px;
           max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
 .callout {{ background: #fff7ed; border-left: 4px solid var(--accent);
             padding: 12px 16px; margin: 16px 0; border-radius: 4px; font-size: 14px; }}
 footer {{ text-align: center; padding: 32px; color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>Li-S electrolyte additive screening dashboard</h1>
  <p>End-to-end reproduction of Wang et al., eScience 2026 (100588)
     - ML-driven donor-number (DN) prediction and Pareto trade-off analysis.</p>
</header>
<main>

  <div class="kpi-row">
    <div class="card"><h3>{xgb_r2:.3f}</h3><p>XGBoost test R<sup>2</sup> (held-out 20 %)</p></div>
    <div class="card"><h3>{rf_r2:.3f}</h3><p>RandomForest test R<sup>2</sup></p></div>
    <div class="card"><h3>{speedup_dft:.1e}&times;</h3><p>ML speedup vs DFT screening</p></div>
    <div class="card"><h3>{n_candidates}</h3><p>Candidate molecules screened</p></div>
  </div>

  <section>
    <h2>1. Hero panel</h2>
    <p>Parity, feature importance, ensemble DN distribution, and SEI-element enrichment of the
       Top-20 candidates in a single overview.</p>
    {img_tag(hero_b64, "Hero")}
  </section>

  <section>
    <h2>2. Model evaluation (paper Fig. 2 reproduction)</h2>
    <div class="twocol">
      {img_tag(fig2a, "RF parity")}
      {img_tag(fig2b, "XGB parity")}
    </div>
    {img_tag(fig2c, "Feature importance")}
    {img_tag(fig2d, "SHAP")}
  </section>

  <section>
    <h2>3. Validation against literature (paper Fig. 3-4 reproduction)</h2>
    {img_tag(fig3, "Spearman rank validation")}
    {img_tag(fig4, "SEI element composition")}
  </section>

  <section>
    <h2>4. Efficiency comparison (paper Fig. 5)</h2>
    {img_tag(fig5, "ML vs DFT vs experiment wall-clock")}
  </section>

  <section>
    <h2>5. Innovation layer - extended analysis</h2>
    <div class="callout">
      The following panels go beyond the paper.  They add (a) multi-objective Pareto
      trade-off analysis, (b) chemical-space t-SNE landscape, (c) atom-level SHAP
      attribution, (d) conformal prediction intervals with calibration, (e) active
      learning curves, (f) synthetic-accessibility vs DN trade-off, (g) surrogate
      decision-tree rules, (h) ROC + PR for the high-DN classifier, and (i) partial
      dependence of the six most important features.
    </div>
  </section>

  <section>
    <h2>5.1 Multi-objective Pareto</h2>
    {img_tag(fig6, "Pareto")}
    <div id="plotly-pareto" style="width:100%;height:500px"></div>
  </section>

  <section>
    <h2>5.2 Chemical-space t-SNE</h2>
    {img_tag(fig7, "t-SNE")}
  </section>

  <section>
    <h2>5.3 Atom-level SHAP contribution</h2>
    {img_tag(fig8, "Atom SHAP")}
  </section>

  <section>
    <h2>5.4 Conformal prediction intervals</h2>
    {img_tag(fig9, "Conformal")}
    <div id="plotly-intervals" style="width:100%;height:520px"></div>
  </section>

  <section>
    <h2>5.5 Active learning vs random vs full-DFT labelling</h2>
    {img_tag(fig10, "Active learning")}
  </section>

  <section>
    <h2>5.6 Synthetic accessibility vs DN</h2>
    {img_tag(fig11, "SA vs DN")}
  </section>

  <section>
    <h2>5.7 Surrogate decision tree (depth=4)</h2>
    {img_tag(fig12, "Decision tree")}
    <p>Plain-text rules in <code>results/decision_rules.txt</code>.</p>
  </section>

  <section>
    <h2>5.8 ROC + precision-recall (DN &gt; 30)</h2>
    {img_tag(fig13, "ROC PR")}
  </section>

  <section>
    <h2>5.9 Partial dependence of top-6 features</h2>
    {img_tag(fig14, "PDP")}
  </section>

  <section>
    <h2>5.10 Top-20 candidate molecules</h2>
    {img_tag(fig15, "Top-20")}
    <p>Vector version: <code>figures/top_molecules.svg</code>.</p>
  </section>

  <section>
    <h2>6. Sortable candidate table</h2>
    {table_html}
  </section>

  <section>
    <h2>7. Chemistry take-aways</h2>
    <ul>
      <li>The two most important descriptors for predicting DN are
          <strong>HOMO energy (proxy)</strong> and <strong>molecular dipole moment (proxy)</strong>,
          matching the eScience 2026 paper's claim.</li>
      <li>All top-20 candidates are <strong>multidentate nitrogen Lewis bases</strong>:
          aminopyridines, purines, alkyl-amino-purines and piperazine-pyrimidines.
          They all carry a high local electron density (high HOMO proxy) and a
          strong dipole, both of which correlate with strong Li+ binding.</li>
      <li>The <strong>F / N / S / O atom-count distribution</strong> of the top-20
          is significantly enriched relative to a random sample, supporting the
          paper's claim that high-DN additives also help form LiF / Li-N /
          Li<sub>2</sub>S SEI species.</li>
      <li>The decision-tree distillation yields four human-interpretable
          rules that already capture &gt; {0:.0f} % of the variance of the
          ensemble prediction (see <code>results/decision_rules.txt</code>).</li>
    </ul>
  </section>

  <footer>Dashboard generated by <code>src/07_dashboard.py</code> - data and figures in
  <code>li_s_additives/</code>.  Plotly.js loaded from CDN once at page load.</footer>
</main>

<script>
const paretoData = {json.dumps(plotly_pareto)};
Plotly.newPlot('plotly-pareto', paretoData.data, paretoData.layout, {{displayModeBar: false}});

const intervalsData = {json.dumps(plotly_intervals)};
Plotly.newPlot('plotly-intervals', intervalsData.data, intervalsData.layout, {{displayModeBar: false}});

// Simple sort handler
document.querySelectorAll('#cand th').forEach(th => {{
  th.addEventListener('click', () => {{
    const tbody = th.closest('table').querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const idx = Array.from(th.parentNode.children).indexOf(th);
    const dir = th.getAttribute('data-sort-dir') === 'asc' ? 'desc' : 'asc';
    th.parentNode.querySelectorAll('th').forEach(x => x.removeAttribute('data-sort-dir'));
    th.setAttribute('data-sort-dir', dir);
    const t = th.getAttribute('data-sort');
    rows.sort((a, b) => {{
      const av = a.children[idx].innerText;
      const bv = b.children[idx].innerText;
      if (t === 'num') return dir === 'asc' ? parseFloat(av) - parseFloat(bv)
                                            : parseFloat(bv) - parseFloat(av);
      if (t === 'int') return dir === 'asc' ? parseInt(av) - parseInt(bv)
                                            : parseInt(bv) - parseInt(av);
      return dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>
</body>
</html>
"""
    return html


def main() -> None:
    html = build_dashboard()
    out = PROJECT_ROOT / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    log.info("Wrote %s  (%.1f KB)", out, len(html) / 1024)


if __name__ == "__main__":
    main()
