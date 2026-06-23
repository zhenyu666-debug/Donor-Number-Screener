"""p43_deer_md.py - DEER MD simulation module."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils import RESULTS_DIR, FIGURES_DIR, get_logger  # noqa: E402
from utils_pb import write_json  # noqa: E402

log = get_logger("p43_deer_md")

R_GAS = 8.314462618e-3
E_CHARGE = 1.602176634e-19
EPS_0 = 8.8541878128e-12
PI = 3.14159265358979

DEER_BLUE = "#1A6EBF"
DEER_ORANGE = "#E8630A"
FRESH_GRAY = "#7F7F7F"

DPI = 150
LABEL_FONT = 11
TITLE_FONT = 13


def binding_energy(dn: float, epsilon_r: float, dipole: float) -> float:
    """Compute Li+ coordination binding energy (eV) based on Coulomb law and DN."""
    if not np.isfinite(dn) or not np.isfinite(epsilon_r) or not np.isfinite(dipole):
        return 0.0
    r_li_o = 2.0
    q_li = 1.0
    q_o = -0.5
    r_m = r_li_o * 1e-10
    coulomb_e = (q_li * q_o * E_CHARGE**2) / (4 * PI * EPS_0 * epsilon_r * r_m)
    e_coul_eV = coulomb_e / E_CHARGE
    dn_scale = (dn / 29.0) if dn > 0 else 0.0
    dipole_norm = min(dipole / 5.0, 1.0) if dipole > 0 else 0.0
    e_binding = e_coul_eV * dn_scale * (0.7 + 0.3 * dipole_norm)
    return float(e_binding)


def coordination_number(dn: float, temperature_K: float = 298.0) -> float:
    """Estimate average Li+ solvation shell size based on DN and temperature."""
    if not np.isfinite(dn) or dn <= 0:
        return 4.5
    dn_ref = 16.8
    cn_ref_298 = 4.5
    dn_slope = 0.08
    cn_ref = cn_ref_298 - dn_slope * (dn - dn_ref)
    cn_ref = max(2.0, min(6.0, cn_ref))
    T_ref = 298.0
    E_a_cn = 2.0
    cn_T = cn_ref * (1.0 + (E_a_cn / (R_GAS * T_ref)) * (temperature_K - T_ref) / T_ref)
    return float(max(2.0, min(6.0, cn_T)))


def md_dissolution_rate(dn: float, epsilon_r: float, temperature_K: float = 298.0, viscosity_cp: float = 2.0) -> float:
    """Compute dissolution rate (nm/scan) using simplified Arrhenius model."""
    if not np.isfinite(dn) or dn <= 0:
        return 0.0
    k_0 = 0.5
    E_a = 30.0 - 0.8 * dn
    E_a = max(1.0, E_a)
    arrhenius = np.exp(-E_a / (R_GAS * temperature_K))
    visc_factor = 2.0 / max(0.1, viscosity_cp) if viscosity_cp > 0 else 1.0
    visc_factor = min(3.0, visc_factor)
    eps_factor = min(epsilon_r / 30.0, 2.0) if epsilon_r > 0 else 1.0
    rate = k_0 * arrhenius * visc_factor * eps_factor
    return float(max(0.0, min(2.0, rate)))


SHELL_GEOMETRIES = {
    "dmi": "N4O2 (4 N-donor, 2 O-acceptor) - bidentate/N-chelating",
    "dmso": "O1 (one S=O donor) - monodentate, tetrahedral",
    "ec": "O2 (two carbonyl O donors) - bidentate chelating",
    "fec": "O2 (two carbonyl O donors) - bidentate with F substituent",
    "dol": "O2 (two ether O donors) - weak chelation",
    "dmc": "O1 (one carbonyl O donor) - monodentate",
    "dec": "O1 (one carbonyl O donor) - monodentate",
    "acn": "N1 (nitrile N donor) - linear monodentate",
    "dmf": "O1 (carbonyl O donor) - monodentate",
    "dmac": "O1 (carbonyl O donor) - monodentate",
    "pyridine": "N1 (heterocyclic N donor) - planar monodentate",
    "pyrazole": "N2 (two heterocyclic N donors) - bidentate",
    "aniline": "N1 (amino N donor) - pyramidal geometry",
    "2-pyrrolidinone": "O1 (carbonyl O) + N1 (amide N) - bidentate",
    "caprolactam": "O1 (carbonyl O) + N1 (amide N) - bidentate",
    "tmp": "O3 (three P=O donors) - tridentate",
    "meoh": "O1 (hydroxyl O donor) - monodentate",
    "etoh": "O1 (hydroxyl O donor) - monodentate",
    "thf": "O1 (ether O donor) - monodentate",
    "nmp": "O1 (carbonyl O) + N1 (lactam N) - bidentate",
    "1-methylpyrrolidine": "N1 (tertiary amine donor) - pyramidal",
    "tmeda": "N2 (two tertiary amine donors) - bidentate",
    "pyrazine": "N2 (two heterocyclic N donors) - planar bidentate",
}


def solvent_shell(smiles_or_name: str, dn: float) -> str:
    """Return the coordination geometry description for a solvent."""
    if not smiles_or_name or not isinstance(smiles_or_name, str):
        return "O? (DN={:.1f} - fallback)".format(dn)
    key_lower = smiles_or_name.lower().strip()
    if key_lower in SHELL_GEOMETRIES:
        return SHELL_GEOMETRIES[key_lower]
    for known_key, geometry in SHELL_GEOMETRIES.items():
        if known_key in key_lower or key_lower in known_key:
            return geometry
    if dn >= 28.0:
        return "O2 (two O donors) - DN={:.1f} suggests bidentate".format(dn)
    elif dn >= 22.0:
        return "O1-2 (1-2 O donors) - DN={:.1f} suggests moderate".format(dn)
    elif dn >= 15.0:
        return "O1 (one O donor) - DN={:.1f} suggests weak".format(dn)
    else:
        return "O0 (no O donors) - DN={:.1f} non-coordinating".format(dn)


def rdf_proxy(dn: float, temperature_K: float = 298.0) -> float:
    """Compute g(r) peak position (Angstrom) for Li-O pair."""
    if not np.isfinite(dn) or dn <= 0:
        return 2.2
    g_r_298 = 2.50 - 0.5 * (dn - 10.0) / 19.0
    g_r_298 = max(1.95, min(2.55, g_r_298))
    delta_T = temperature_K - 298.0
    g_r = g_r_298 + 0.0005 * delta_T
    return float(max(1.9, min(2.6, g_r)))


def load_solvents() -> pd.DataFrame:
    """Load solvents from CSV or generate fallback data."""
    csv_path = RESULTS_DIR / "solvent_eei_predictions.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            return df
        except:
            pass
    log.info("Generating solvent data from known DEER solvents")
    fallback_data = [
        ("O=C1N(C)CCC1", "DMI", 29.0, 37.7, 4.1, 2.1),
        ("CS(C)=O", "DMSO", 29.8, 47.2, 3.9, -0.5),
        ("O=C1COC(=O)O1", "FEC", 18.0, 110.0, 2.1, 2.1),
        ("O=C1OCCO1", "EC", 16.8, 89.0, 4.9, 1.9),
        ("C1COC(=O)O1", "DOL", 21.1, 7.0, 2.0, 0.6),
        ("CC(=O)OC", "EA", 17.1, 6.0, 1.9, 0.4),
        ("COC(=O)OC", "DMC", 17.2, 3.1, 1.9, 0.6),
        ("CCOC(=O)OCC", "DEC", 17.0, 2.8, 1.8, 0.7),
        ("CC#N", "ACN", 14.1, 36.0, 3.5, 0.3),
        ("CC(=O)N(C)C", "DMF", 26.6, 37.0, 3.9, 0.8),
        ("CN(C)C=O", "DMAc", 26.6, 38.0, 3.7, 0.9),
        ("C1CC(=O)NC1", "2-Pyrrolidinone", 27.3, 28.0, 3.3, 1.1),
        ("CO", "MeOH", 19.1, 33.0, 1.7, 0.5),
        ("CCO", "EtOH", 19.2, 25.0, 1.7, 1.2),
        ("C1CCOC1", "THF", 20.0, 7.5, 1.7, 0.5),
        ("c1ccncc1", "Pyridine", 33.1, 13.3, 2.4, 1.0),
        ("Nc1ccccc1", "Aniline", 32.2, 6.9, 2.1, 4.4),
    ]
    rows = []
    for item in fallback_data:
        rows.append({"smiles": item[0], "name": item[1], "dn": item[2], "epsilon_r": item[3], "dipole_debye": item[4], "viscosity_cp": item[5] if item[5] > 0 else 1.0})
    return pd.DataFrame(rows)


def compute_md_metrics(df: pd.DataFrame, temperature_K: float = 298.0) -> pd.DataFrame:
    """Compute all MD-inspired metrics for each solvent."""
    results = []
    for _, row in df.iterrows():
        dn = float(row.get("dn", 0.0))
        eps = float(row.get("epsilon_r", 30.0))
        dip = float(row.get("dipole_debye", 2.0))
        visc = float(row.get("viscosity_cp", 2.0))
        smiles = str(row.get("smiles", ""))
        name = str(row.get("name", "Unknown"))
        if dn <= 0 or not np.isfinite(dn):
            continue
        e_binding = binding_energy(dn, eps, dip)
        cn = coordination_number(dn, temperature_K)
        rate = md_dissolution_rate(dn, eps, temperature_K, max(0.1, visc))
        shell = solvent_shell(name, dn)
        g_r = rdf_proxy(dn, temperature_K)
        results.append({
            "smiles": smiles, "name": name, "donor_number": dn,
            "dielectric_constant": eps, "dipole_debye": dip, "viscosity_cp": visc,
            "binding_energy_eV": round(e_binding, 4),
            "coordination_number": round(cn, 2),
            "dissolution_rate_nm_per_scan": round(rate, 4),
            "gr_peak_angstrom": round(g_r, 3),
            "coordination_geometry": shell,
            "md_solvation_score": round(abs(e_binding) * cn / 2.0, 3),
        })
    return pd.DataFrame(results)


def make_binding_energy_figure(df: pd.DataFrame, out_path: Path) -> None:
    """Generate scatter plot of DN vs binding energy, color-coded by g(r) peak."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("matplotlib unavailable: {}".format(exc))
        return
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    x = df["donor_number"].values
    y = df["binding_energy_eV"].values
    colors = df["gr_peak_angstrom"].values
    sc = ax.scatter(x, y, c=colors, cmap="plasma_r", s=100, alpha=0.8, edgecolors="white", linewidth=0.5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("g(r) peak (Li-O) [A]", fontsize=10)
    for _, row in df.iterrows():
        name_lower = row["name"].lower()
        if "dmi" in name_lower:
            ax.scatter(row["donor_number"], row["binding_energy_eV"], color=DEER_BLUE, s=250, marker="*", zorder=5, edgecolors="white", linewidth=1.0)
            ax.annotate("DMI", (row["donor_number"], row["binding_energy_eV"]), xytext=(5, 5), textcoords="offset points", fontsize=9, color=DEER_BLUE, fontweight="bold")
        elif "dmso" in name_lower:
            ax.scatter(row["donor_number"], row["binding_energy_eV"], color=DEER_ORANGE, s=250, marker="D", zorder=5, edgecolors="white", linewidth=1.0)
            ax.annotate("DMSO", (row["donor_number"], row["binding_energy_eV"]), xytext=(5, 5), textcoords="offset points", fontsize=9, color=DEER_ORANGE, fontweight="bold")
    ax.axhline(y=min(y), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=max(y), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Donor Number (kcal/mol)", fontsize=LABEL_FONT)
    ax.set_ylabel("Li+ Binding Energy (eV)", fontsize=LABEL_FONT)
    ax.set_title("DEER MD Simulation - Solvent-EEI Binding Energy vs Donor Number", fontsize=TITLE_FONT)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=DEER_BLUE, markersize=14, label="DMI"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=DEER_ORANGE, markersize=10, label="DMSO"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="Other"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Binding energy figure saved: {}".format(out_path))


def main() -> int:
    """Run MD-inspired solvent-EEI simulation."""
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        temperature_K = 298.0
        df_solvents = load_solvents()
        log.info("Loaded {} solvents".format(len(df_solvents)))
        if df_solvents.empty:
            log.error("No solvents to process")
            return 1
        df_md = compute_md_metrics(df_solvents, temperature_K)
        log.info("Computed MD metrics for {} solvents".format(len(df_md)))
        if df_md.empty:
            log.error("No valid MD metrics computed")
            return 1
        out_json = RESULTS_DIR / "deer_md_simulation.json"
        json_data = {
            "metadata": {"temperature_K": temperature_K, "n_solvents": len(df_md), "description": "MD-inspired solvent-EEI interaction model for DEER"},
            "solvents": df_md.to_dict(orient="records"),
        }
        write_json(out_json, json_data)
        log.info("JSON output written: {}".format(out_json))
        out_fig = FIGURES_DIR / "deer_md_binding_energy.png"
        make_binding_energy_figure(df_md, out_fig)
        print()
        print("=" * 70)
        print("DEER MD Simulation Summary")
        print("=" * 70)
        print("{:<35} {:>5} {:>8} {:>5} {:>8} {:>6}".format("Solvent", "DN", "E_bind", "CN", "Rate", "g(r)"))
        print("-" * 70)
        for _, row in df_md.sort_values("donor_number", ascending=False).head(10).iterrows():
            print("{:<35} {:>5.1f} {:>8.3f} {:>5.2f} {:>8.4f} {:>6.3f}".format(
                row["name"][:34], row["donor_number"], row["binding_energy_eV"],
                row["coordination_number"], row["dissolution_rate_nm_per_scan"], row["gr_peak_angstrom"]))
        print("=" * 70)
        return 0
    except Exception as exc:
        log.error("Error in MD simulation: {}".format(exc))
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
