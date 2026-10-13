"""Reproduce manuscript cells from frozen reports and completed evaluations."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = Path('/local-scratch2/mirzaei/PAPER_REVISION_GRAPHVAE_FULL_20260922')
DATA = Path('/local-scratch2/mirzaei')
sources = {}

def read(path):
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

old = read(PREVIOUS / 'TABLE_EVIDENCE.json')
datasets = ['GRID', 'TRIANGULAR_GRID', 'LOBSTER', 'PTC', 'PROTEINS', 'QM9']
methods = ['false', 'true', 'defog']
metrics = ['Degree MMD', 'Clustering MMD', 'Orbit MMD', 'Spectral MMD', 'Diameter MMD', 'Absolute error in mean edge count']
cells = {}
for row in old:
    metric = row['metric']
    if 'edge' in metric.lower() and 'error' in metric.lower():
        metric = metrics[-1]
    for method in methods:
        cells[(row['dataset'], metric, method)] = row[method].replace(' (n=2)', '').replace(' (n=3)', '').replace('**', '')

def summary(values):
    return f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}'

protein_struct = {
    'false': ['0.037520 ± 0.005647', '0.025883 ± 0.000997', '0.024463 ± 0.004194', '0.021577 ± 0.001381', '0.097364 ± 0.010305', '26.362041 ± 1.175713'],
    'true': ['0.021855 ± 0.003040', '0.021611 ± 0.001361', '0.010732 ± 0.002849', '0.013285 ± 0.002163', '0.042071 ± 0.021942', '21.854864 ± 2.840036'],
}
report = PREVIOUS / 'sources/reports/datasets/PROTEINS/RESULTS.md'
sources[str(report)] = hashlib.sha256(report.read_bytes()).hexdigest()
raw_seeds = {}
for method in methods:
    values = []
    structures = []
    for seed in range(3):
        if method == 'defog':
            path = DATA / f'proteins_defog_corrected_eval_20260922/seed_{seed}/common_metrics.json'
            obj = read(path)
            ev = obj['random_gin']
            structures.append(obj['structural_mmd'])
        else:
            label = 'true_full_003' if method == 'true' else 'false'
            path = DATA / f'archive_completion_20260922/PROTEINS/{label}/seed_{seed}/attributed_random_gin.json'
            obj = read(path)
            ev = obj['evaluation']
        values.append(ev['modes']['topology_control']['summary']['f1_pr']['mean'])
    raw_seeds[method] = values
    cells[('PROTEINS', 'F1-PR', method)] = summary(values)
    for metric, key in zip(metrics, ['degree', 'clustering', 'orbit', 'spectral', 'diameter', 'edge_count_absolute_error']):
        cells[('PROTEINS', metric, method)] = summary([s[key] for s in structures]) if method == 'defog' else protein_struct[method][metrics.index(metric)]

def mean(cell):
    return float(cell.split()[0])

labels = {'TRIANGULAR_GRID': 'Triangular grid'}
structural = ['| Dataset / method | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ | Diameter ↓ | Mean-edge error ↓ |', '|---|---:|---:|---:|---:|---:|---:|']
method_labels = {'false': 'GraphVAE', 'true': 'GraphVAE-REQ (full)', 'defog': 'DeFoG'}
for ds in datasets:
    for method in methods:
        row = []
        for metric in metrics:
            cell = cells[(ds, metric, method)]
            if mean(cell) == min(mean(cells[(ds, metric, m)]) for m in methods):
                cell = f'**{cell}**'
            row.append(cell)
        structural.append(f'| {labels.get(ds, ds)} / {method_labels[method]} | ' + ' | '.join(row) + ' |')

gin = ['| Dataset / input | Seeds (GV / REQ / DeFoG) | GraphVAE ↑ | GraphVAE-REQ full ↑ | DeFoG ↑ | REQ vs GV | REQ vs DeFoG |', '|---|:---:|---:|---:|---:|---:|---:|']
for ds in datasets:
    values = [cells[(ds, 'F1-PR', m)] for m in methods]
    numbers = [mean(v) for v in values]
    displayed = [f'**{v}**' if x == max(numbers) else v for v, x in zip(values, numbers)]
    gains = [f'{100 * (numbers[1] - base) / base:+.2f}%' for base in (numbers[0], numbers[2])]
    mode = 'S' if ds not in ('PROTEINS', 'QM9') else 'C'
    gin.append(f'| {labels.get(ds, ds)} / {mode} | {"3 / 3 / 2" if ds == "GRID" else "3 / 3 / 3"} | ' + ' | '.join(displayed + gains) + ' |')

payload = {'sources_sha256': sources, 'cells': [{'dataset': k[0], 'metric': k[1], 'method': k[2], 'value': v} for k, v in cells.items()], 'proteins_f1_per_seed': raw_seeds}
(ROOT / 'evidence.json').write_text(json.dumps(payload, indent=2) + '\n')
(ROOT / 'tables.md').write_text('\n'.join(structural) + '\n\n' + '\n'.join(gin) + '\n')
print('\n'.join(structural) + '\n\n' + '\n'.join(gin))
