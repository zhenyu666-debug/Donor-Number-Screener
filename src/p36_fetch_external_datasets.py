from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

REPO_ROOT = THIS_DIR.parent
DATA_DIR = REPO_ROOT / 'data'

SOURCES = {
    'combat': {
        'url': 'https://raw.githubusercontent.com/rashatwi/combat/main/data.csv',
        'license': 'MIT',
        'doi': '10.5281/zenodo.7830272',
        'description': 'ComBat Li-S electrolyte quantum chemistry and MD database',
    },
    'electrolytomics_cond': {
        'base': 'https://raw.githubusercontent.com/AmanchukwuLab/electrolytomics/main',
        'files': [
            'datasets/raw/conductivity/EDB-1_conductivity.csv',
            'datasets/raw/conductivity/all_multi_cond_add.csv',
            'datasets/raw/conductivity/all_multi_cond_comb_comm.csv',
            'datasets/raw/conductivity/all_multi_cond_comm.csv',
        ],
        'license': 'MIT',
        'doi': '10.1021/acs.chemmater.4c03196',
        'description': 'Electrolytomics ionic conductivity datasets',
    },
    'electrolytomics_oxstab': {
        'base': 'https://raw.githubusercontent.com/AmanchukwuLab/electrolytomics/main',
        'files': [
            'datasets/raw/oxstab/oxstab_train_random.csv',
            'datasets/raw/oxstab/oxstab_test_random.csv',
        ],
        'license': 'MIT',
        'doi': '10.1021/acs.chemmater.4c03196',
        'description': 'Electrolytomics oxidation stability datasets',
    },
    'electrolytomics_ce': {
        'base': 'https://raw.githubusercontent.com/AmanchukwuLab/electrolytomics/main',
        'files': ['datasets/raw/CE/EDB-2_ce.csv'],
        'license': 'MIT',
        'doi': '10.1021/acs.chemmater.4c03196',
        'description': 'Electrolytomics Coulombic efficiency dataset',
    },
    'calisol': {
        'url': 'https://raw.githubusercontent.com/Pele0599/CALiSol-23/main/CALiSol-23%20Dataset.csv',
        'fallback_urls': ['https://raw.githubusercontent.com/Pele0599/CALiSol-23/main/calisolsmile.csv'],
        'license': 'CC BY 4.0',
        'doi': '10.1038/s41597-024-03575-8',
        'description': 'CALiSol-23 experimental ionic conductivity atlas',
    },
    'dn_dft': {
        'url': 'https://raw.githubusercontent.com/mqcomplab/DonorNumberPrediction/master/DN_data.csv',
        'license': 'MIT',
        'doi': '10.5281/zenodo.3998765',
        'description': 'DFT-calculated donor numbers from DonorNumberPrediction',
    },
    'superionic': {
        'url': 'https://data.caltech.edu/records/23mvv-6gk43/files/ionic_conductivity_database.csv',
        'fallback_urls': ['https://data.caltech.edu/records/gz5xf-m5051/files/ionic_conductivity_database.csv'],
        'license': 'CC BY 4.0',
        'doi': '10.22002/23mvv-6gk43',
        'description': 'McHaffie superionic Li conductor database from CaltechDATA',
    },
}

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_csv(path, rows, fieldnames=None):
    ensure_dir(path.parent)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fns = fieldnames or list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fns})

def write_json(path, obj):
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)

def http_get(url, timeout=30.0):
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'PBP-external/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print(f'[fetch] GET {url} failed: {e}')
        return None

def _float(val):
    if val is None:
        return float('nan')
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return float('nan')

def fetch_combat():
    cfg = SOURCES['combat']
    data = http_get(cfg['url'])
    if not data:
        return []
    text = data.decode('utf-8', errors='replace')
    rows = []
    for i, r in enumerate(csv.DictReader(io.StringIO(text))):
        smiles = (r.get('CanonicalSMILES') or r.get('SMILES') or '').strip()
        if not smiles:
            continue
        rows.append({
            'id': f'CB-{i:04d}',
            'source_db': 'ComBat',
            'smiles': smiles,
            'name': r.get('Abbreviation') or r.get('IUPACName') or '',
            'cas': r.get('CAS') or '',
            'dn': _float(r.get('DN')),
            'an': float('nan'),
            'epsilon_r': _float(r.get('De')),
            'viscosity_cp': _float(r.get('Viscosity')),
            'dipole_debye': float('nan'),
            'conductivity_S_cm': float('nan'),
            'oxidation_V': float('nan'),
            'reduction_V': float('nan'),
            'be_litfsi_kcal': _float(r.get('BE_Salt')),
            'be_polysulfide_kcal': _float(r.get('BE_PS')),
            'diffusion': _float(r.get('Diffusion')),
            'density': _float(r.get('Density')),
            'molecular_weight': _float(r.get('MolecularWeight')),
            'melting_point_c': _float(r.get('MeltingPoint')),
            'boiling_point_c': _float(r.get('BoilingPoint')),
            'family': r.get('Type') or 'other',
            'doi': cfg['doi'],
            'license': cfg['license'],
            'notes': 'ComBat Li-S',
        })
    print(f'[fetch] ComBat: {len(rows)} solvent entries')
    return rows

def _fetch_elec(base, paths, prefix, extras=None):
    extras = extras or {}
    rows = []
    for path in paths:
        url = f'{base}/{path}'
        data = http_get(url)
        if not data:
            continue
        text = data.decode('utf-8', errors='replace')
        for i, r in enumerate(csv.DictReader(io.StringIO(text))):
            # Try multiple SMILES column names (Electrolytomics uses solv_X_sm,
            # smiles, or smiles column depending on file).
            smiles = (r.get('SMILES') or r.get('smiles')
                      or r.get('solv_1_sm') or r.get('solvent_1_smiles')
                      or r.get('additive_smiles') or '').strip()
            if not smiles:
                continue
            row = {
                'id': f'{prefix}-{len(rows):05d}',
                'source_db': prefix.replace('-', '_'),
                'smiles': smiles,
                'name': (r.get('Name') or r.get('Solvent') or r.get('solv_1_sm') or
                         r.get('solvent_1_smiles') or r.get('additive_smiles') or ''),
                'cas': '',
                'dn': _float(r.get('DN')),
                'an': float('nan'),
                'epsilon_r': float('nan'),
                'viscosity_cp': float('nan'),
                'dipole_debye': float('nan'),
                'conductivity_S_cm': float('nan'),
                'oxidation_V': float('nan'),
                'reduction_V': float('nan'),
                'family': 'liquid_electrolyte',
                'doi': SOURCES['electrolytomics_cond']['doi'],
                'license': 'MIT',
                'notes': 'Electrolytomics',
            }
            for ok, sk in extras.items():
                row[ok] = _float(r.get(sk))
            rows.append(row)
    print(f'[fetch] {prefix}: {len(rows)} entries')
    return rows

def fetch_electrolytomics_cond():
    cfg = SOURCES['electrolytomics_cond']
    return _fetch_elec(cfg['base'], cfg['files'], 'ELC', {
        'conductivity_S_cm': 'Conductivity',
        'temperature_K': 'Temperature',
        'concentration_m': 'Concentration',
    })

def fetch_electrolytomics_oxstab():
    cfg = SOURCES['electrolytomics_oxstab']
    return _fetch_elec(cfg['base'], cfg['files'], 'ELO', {
        'oxidation_V': 'OxidationStability',
        'reduction_V': 'ReductionStability',
    })

def fetch_electrolytomics_ce():
    cfg = SOURCES['electrolytomics_ce']
    return _fetch_elec(cfg['base'], cfg['files'], 'ELCE', {
        'coulombic_efficiency_pct': 'CoulombicEfficiency',
    })

def fetch_calisol():
    cfg = SOURCES['calisol']
    data = http_get(cfg['url'], timeout=120.0)
    if not data:
        for url in cfg.get('fallback_urls', []):
            data = http_get(url, timeout=60.0)
            if data:
                break
    if not data:
        return []
    text = data.decode('utf-8', errors='replace')
    rows = []
    SOLVENT_COLS = ['EC', 'PC', 'DMC', 'EMC', 'DEC', 'DME', 'DMSO', 'AN', 'MOEMC',
                    'TFP', 'EA', 'MA', 'FEC', 'DOL', '2-MeTHF', 'DMM', 'Freon 11',
                    'Methylene chloride', 'THF', 'Toluene', 'Sulfolane', '2-Glyme',
                    '3-Glyme', '4-Glyme', '3-Me-2-Oxazolidinone', '3-MeSulfolane',
                    'Ethyldiglyme', 'DMF', 'Ethylbenzene', 'Ethylmonoglyme',
                    'Benzene', 'g-Butyrolactone', 'Cumene', 'Propylsulfone',
                    'Pseudocumeme', 'TEOS', 'm-Xylene', 'o-Xylene']
    for i, r in enumerate(csv.DictReader(io.StringIO(text))):
        # Each row is one mixture: extract which solvents are non-zero
        solvents_in_mix = [s for s in SOLVENT_COLS if _float(r.get(s)) > 0]
        mix_label = '+'.join(solvents_in_mix) if solvents_in_mix else ''
        k = _float(r.get('k'))
        if not math.isfinite(k) or k <= 0:
            continue
        rows.append({
            'id': f'CAL-{i:05d}',
            'source_db': 'CALiSol-23',
            'smiles': '',
            'name': mix_label,
            'cas': '',
            'salt': r.get('salt') or '',
            'concentration_m': _float(r.get('c')),
            'temperature_K': _float(r.get('T')),
            'conductivity_S_cm': k,
            'viscosity_cp': float('nan'),
            'source_doi': r.get('doi') or '',
            'dn': float('nan'), 'an': float('nan'), 'epsilon_r': float('nan'),
            'dipole_debye': float('nan'), 'oxidation_V': float('nan'), 'reduction_V': float('nan'),
            'family': 'liquid_electrolyte',
            'doi': cfg['doi'], 'license': cfg['license'],
            'notes': 'CALiSol-23 experimental conductivity',
        })
    print(f'[fetch] CALiSol-23: {len(rows)} entries')
    return rows

def fetch_dn_dft():
    cfg = SOURCES['dn_dft']
    data = http_get(cfg['url'])
    if not data:
        return []
    text = data.decode('utf-8', errors='replace')
    rows = []
    ti = io.StringIO(text)
    try:
        reader = csv.DictReader(ti, delimiter=';')
        first = next(reader, None)
        ti.seek(0)
        if first and 'inert_solvent' not in first:
            ti = io.StringIO(text)
            reader = csv.DictReader(ti, delimiter=';')
    except Exception:
        ti = io.StringIO(text)
        reader = csv.DictReader(ti)
    for i, r in enumerate(reader):
        solvent = (r.get('solvent') or r.get('name') or '').strip()
        if not solvent:
            continue
        rows.append({
            'id': f'DN-{i:04d}',
            'source_db': 'DonorNumberPrediction',
            'smiles': (r.get('smiles') or r.get('SMILES') or '').strip(),
            'name': solvent,
            'cas': r.get('cas') or '',
            'dn': _float(r.get('donor_number') or r.get('DN')),
            'reference_acid': r.get('reference_acid') or '',
            'inert_solvent': r.get('inert_solvent') or '',
            'an': float('nan'), 'epsilon_r': float('nan'),
            'viscosity_cp': float('nan'), 'dipole_debye': float('nan'),
            'conductivity_S_cm': float('nan'),
            'oxidation_V': float('nan'), 'reduction_V': float('nan'),
            'family': 'liquid_electrolyte',
            'doi': cfg['doi'], 'license': cfg['license'],
            'notes': 'DFT-calculated donor number',
        })
    print(f'[fetch] DonorNumberPrediction: {len(rows)} entries')
    return rows

def fetch_superionic():
    cfg = SOURCES['superionic']
    data = http_get(cfg['url'], timeout=60.0)
    if not data:
        for url in cfg.get('fallback_urls', []):
            data = http_get(url, timeout=60.0)
            if data:
                break
    if not data:
        return []
    text = data.decode('utf-8', errors='replace')
    rows = []
    for i, r in enumerate(csv.DictReader(io.StringIO(text))):
        formula = (r.get('formula') or r.get('compound') or '').strip()
        if not formula:
            continue
        rows.append({
            'id': f'SUP-{i:04d}',
            'source_db': 'McHaffie_superionic',
            'smiles': '',
            'name': formula,
            'cas': r.get('CAS') or '',
            'formula': formula,
            'sigma_ion_S_cm': _float(r.get('ionic_conductivity') or r.get('conductivity') or r.get('conductivity_siemens_per_cm')),
            'E_g_eV': _float(r.get('band_gap') or r.get('Eg')),
            'migration_barrier_eV': _float(r.get('migration_barrier')),
            'space_group': r.get('space_group') or '',
            'icsd_id': r.get('icsd_id') or '',
            'dn': float('nan'), 'an': float('nan'), 'epsilon_r': float('nan'),
            'viscosity_cp': float('nan'), 'dipole_debye': float('nan'),
            'conductivity_S_cm': float('nan'),
            'oxidation_V': float('nan'), 'reduction_V': float('nan'),
            'family': 'superionic_conductor',
            'doi': cfg['doi'], 'license': cfg['license'],
            'notes': 'McHaffie superionic conductor dataset',
        })
    print(f'[fetch] McHaffie-superionic: {len(rows)} entries')
    return rows

UNIFIED_COLS = [
    'id', 'source_db', 'smiles', 'name', 'cas',
    'dn', 'an', 'epsilon_r', 'viscosity_cp', 'dipole_debye',
    'conductivity_S_cm', 'oxidation_V', 'reduction_V',
    'family', 'doi', 'license', 'notes',
]

SOURCE_ORDER = {
    'ComBat': 0, 'CALiSol-23': 1, 'Electrolytomics_cond': 2,
    'Electrolytomics_oxstab': 3, 'Electrolytomics_CE': 4,
    'DonorNumberPrediction': 5, 'McHaffie_superionic': 6,
}

def _safe_row(d):
    out = {col: d.get(col, '') for col in UNIFIED_COLS}
    for col in ['dn', 'an', 'epsilon_r', 'viscosity_cp', 'dipole_debye',
                'conductivity_S_cm', 'oxidation_V', 'reduction_V']:
        try:
            out[col] = float(out.get(col, '') or 'nan')
        except (ValueError, TypeError):
            out[col] = float('nan')
    return out

def merge_all(combat, calisol, elec_cond, elec_ox, elec_ce, dn_dft, superionic):
    all_rows = []
    for src in [combat, calisol, elec_cond, elec_ox, elec_ce, dn_dft, superionic]:
        for r in src:
            all_rows.append(_safe_row(r))
    seen = {}
    for r in all_rows:
        key = (r['smiles'].strip(), r['source_db'].strip())
        if not key[0]:
            key = (r['name'].strip(), r['source_db'].strip())
        if not key[0]:
            continue
        if key not in seen:
            seen[key] = r
        else:
            for k, v in r.items():
                if v in ('', None):
                    continue
                try:
                    if isinstance(v, float) and not math.isfinite(v):
                        continue
                except Exception:
                    pass
                ev = seen[key].get(k)
                try:
                    if isinstance(ev, float) and math.isnan(ev):
                        seen[key][k] = v
                except Exception:
                    pass
    merged = list(seen.values())
    merged.sort(key=lambda x: (SOURCE_ORDER.get(x['source_db'], 9), x['source_db'], x['id']))
    return merged

def main():
    p = argparse.ArgumentParser(description='Mirror external battery electrolyte datasets.')
    p.add_argument('--offline', action='store_true', help='Skip HTTP requests.')
    p.add_argument('--out-dir', default=str(DATA_DIR))
    args = p.parse_args()
    ensure_dir(Path(args.out_dir))
    out_dir = Path(args.out_dir)
    t0 = time.time()
    if args.offline:
        print('[mode] OFFLINE')
        combat = calisol = elec_cond = elec_ox = elec_ce = dn_dft = superionic = []
    else:
        print('[fetch] Starting external dataset fetches...')
        combat = fetch_combat()
        time.sleep(0.5)
        elec_cond = fetch_electrolytomics_cond()
        time.sleep(0.5)
        elec_ox = fetch_electrolytomics_oxstab()
        time.sleep(0.5)
        elec_ce = fetch_electrolytomics_ce()
        time.sleep(0.5)
        calisol = fetch_calisol()
        time.sleep(0.5)
        dn_dft = fetch_dn_dft()
        time.sleep(0.5)
        superionic = fetch_superionic()
    write_csv(out_dir / 'external_solvents_combat.csv', combat)
    write_csv(out_dir / 'external_conductivity_calisol.csv', calisol)
    write_csv(out_dir / 'external_electrolytomics_conductivity.csv', elec_cond)
    write_csv(out_dir / 'external_electrolytomics_oxstab.csv', elec_ox)
    write_csv(out_dir / 'external_electrolytomics_ce.csv', elec_ce)
    write_csv(out_dir / 'external_dn_dft.csv', dn_dft)
    write_csv(out_dir / 'external_sse_superionic.csv', superionic)
    merged = merge_all(combat, calisol, elec_cond, elec_ox, elec_ce, dn_dft, superionic)
    write_csv(out_dir / 'merged_electrolyte_library.csv', merged, fieldnames=UNIFIED_COLS)
    elapsed = time.time() - t0
    meta = {
        'fetched_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'elapsed_s': round(elapsed, 2),
        'offline': args.offline,
        'n_total_merged': len(merged),
        'per_source': {
            'combat': len(combat), 'calisol': len(calisol),
            'electrolytomics_cond': len(elec_cond),
            'electrolytomics_oxstab': len(elec_ox),
            'electrolytomics_ce': len(elec_ce),
            'donor_number_dft': len(dn_dft),
            'superionic_mchaffie': len(superionic),
        },
        'sources': {n: {'license': i['license'], 'doi': i['doi'], 'description': i['description']}
                    for n, i in SOURCES.items()},
    }
    write_json(out_dir / 'external_datasets_meta.json', meta)
    print(f'\n[done] {len(merged)} merged entries in {out_dir}')
    for lb, n in [
        ('ComBat', len(combat)), ('CALiSol-23', len(calisol)),
        ('Electrolytomics-cond', len(elec_cond)),
        ('Electrolytomics-oxstab', len(elec_ox)),
        ('Electrolytomics-CE', len(elec_ce)),
        ('DonorNumberPrediction', len(dn_dft)),
        ('McHaffie-superionic', len(superionic)),
    ]:
        print(f'  {lb:25s} {n:>6}')
    print(f'  Elapsed: {elapsed:.1f}s')
    return 0

if __name__ == '__main__':
    sys.exit(main())
