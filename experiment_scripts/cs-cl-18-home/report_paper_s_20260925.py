import json
from pathlib import Path
import numpy as np

ROOT = Path('/local-scratch2/mirzaei/PAPER_S_RANDOM_GIN_20260925')
AUDIT = Path('/local-scratch2/mirzaei/PAPER_RESULTS_AUDIT_20260924')
methods = ['motif_false', 'motif_true_full', 'defog']
labels = ['Motif=False', 'Motif=True full', 'DeFoG']

def fmt(values):
    a = np.asarray(values)
    if abs(a.mean()) < 0.00001:
        return '%.4e ± %.4e' % (a.mean(), a.std(ddof=1))
    return '%.6f ± %.6f' % (a.mean(), a.std(ddof=1))

lines = ['# Paper metric audit and unified S-input Random-GIN results', '',
         'Updated 2026-09-25. The manuscript has not been changed.', '',
         '## Corrections beyond GRID', '',
         '- TRIANGULAR_GRID: original GraphVAE and DeFoG results use different reference collections. Use common-reference re-evaluations; document seed inclusion rather than claiming a score-filtered set is unbiased.',
         '- PTC: the old motif=False structural results use seed-dependent references, unlike the motif=True/DeFoG panel. Replace with a common-reference structural evaluation. The matched-reference F1 scores are a separate, already corrected evaluation.',
         '- PROTEINS: original C-input F1 uses different references for motif=False versus True/DeFoG. The S evaluation below uses the same paper motif=True reference for all methods. This fixes evaluation reference identity, not underlying training-split differences.',
         '- QM9: replace C-input F1 with the newly evaluated S-input scores below for uniform input semantics.',
         '- LOBSTER: its archived structural and S-input numbers reproduce; all three methods share the same reference. No numerical correction was identified.', '',
         'The arithmetic audit reproduced all 126 mean/SD cells and 12 improvement percentages in final.md. Numerical reproduction alone does not establish a controlled training comparison.', '',
         '## LOBSTER — archived paper-related metrics', '',
         'Link correlation ON, full-matrix motif=True, seeds 0/1/2 for each method, 20 reference graphs. All three DeFoG seeds are retained. Mean ± sample SD across generator seeds. Random-GIN scores first average ten evaluator initializations per generator. No motif-correlation metrics are included in this paper-focused report.', '']

rows = {m: [] for m in methods}
sources = json.loads((AUDIT / 'synthetic_archive_sources.json').read_text())
for r in sources:
    if r['dataset'] != 'LOBSTER':
        continue
    m = {'false': 'motif_false', 'true': 'motif_true_full'}[r['method']]
    structural = r['files']['final_table2_metrics.json']['data']
    st = dict(structural['metrics'], **structural['extra_metrics'])
    st['edge_count_absolute_error'] = abs(st['generated_edge_count'] - st['reference_edge_count'])
    gin = r['files']['graph_realism_random_gin.json']['data']['metrics']
    rows[m].append({'seed': r['seed'], 'structural': st, 'gin': gin})
for s in range(3):
    p = Path('/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/metrics/lobster_seed%d.json' % s)
    r = json.loads(p.read_text())
    rows['defog'].append({'seed': s, 'structural': r['structural_mmd'], 'gin': r['third_party_random_gin_structural_features']['metrics']})

def header():
    return ['| Metric | ' + ' | '.join(labels) + ' |', '|---|---:|---:|---:|']

lines += ['### Structural metrics', ''] + header()
for key, label in [('degree','Degree MMD ↓'), ('clustering','Clustering MMD ↓'), ('orbit','Orbit MMD ↓'), ('spectral','Spectral MMD ↓'), ('diameter','Diameter MMD ↓'), ('triangle','Triangle MMD ↓'), ('sparsity','Sparsity MMD ↓'), ('edge_count_absolute_error','Mean-edge absolute error ↓'), ('generated_edge_count','Generated mean edge count (reference 45.85)')]:
    lines.append('| %s | %s |' % (label, ' | '.join(fmt([r['structural'][key] for r in rows[m]]) for m in methods)))
lines += ['', '### S-input Random-GIN', ''] + header()
for key, stat, label in [('f1_pr','mean','F1-PR ↑'), ('precision','mean','Precision ↑'), ('recall','mean','Recall ↑'), ('mmd_rbf','mean','RBF MMD ↓'), ('mmd_linear','mean','Linear MMD mean ↓'), ('mmd_linear','median','Linear MMD median ↓'), ('mmd_linear','trimmed_mean','Linear MMD trimmed mean ↓')]:
    lines.append('| %s | %s |' % (label, ' | '.join(fmt([r['gin'][key][stat] for r in rows[m]]) for m in methods)))
lines += ['', '## Newly computed S-input scores: PROTEINS and QM9', '',
          'S means three topology-derived node inputs: degree, clustering coefficient, and square clustering. It does not use native node or edge attributes. The same vendored Random-GIN wrapper is used as in the synthetic S panel: evaluator seeds 0–9, three generator seeds 0/1/2. C means the constant-input control and must not be relabeled S without evaluation.', '',
          'Saved graph collections are re-evaluated, without retraining or resampling. All methods use one fixed reference within each dataset: 209 PROTEINS graphs (paper motif=True reference) and 512 QM9 graphs. PROTEINS DeFoG uses the first 209 of 210 saved graphs, matching the original common evaluator; no quality-based selection is performed. Topology conversion removes self-loops and isolates, retains the largest component, and then applies the S wrapper. No empty graphs were silently dropped.', '']
for ds in ['PROTEINS','QM9']:
    summary = json.loads((ROOT / ds / 'aggregate.json').read_text())
    lines += ['### ' + ds, ''] + header()
    for metric in ['f1_pr','precision','recall','mmd_rbf','mmd_linear']:
        lines.append('| %s | %s |' % (metric, ' | '.join(fmt(summary[m][metric]['per_seed']) for m in methods)))
    lines += ['', '| Method | Seed 0 F1-PR | Seed 1 F1-PR | Seed 2 F1-PR |', '|---|---:|---:|---:|']
    for m, label in zip(methods, labels):
        lines.append('| %s | %s |' % (label, ' | '.join('%.6f' % x for x in summary[m]['f1_pr']['per_seed'])))
    lines += ['']
lines += ['## Evidence and limits', '',
          '- Per-seed JSON files in `PROTEINS/` and `QM9/` contain raw scores for every evaluator initialization, input paths and SHA-256 hashes, shared-reference hashes, and wrapper hashes.',
          '- S evaluation program: `/local-scratch/localhome/mirzaei/evaluate_paper_s_20260925.py`.',
          '- Raw numerical audit: `/local-scratch2/mirzaei/PAPER_RESULTS_AUDIT_20260924/NUMERICAL_AUDIT.json`.',
          '- LOBSTER source snapshot: `/local-scratch2/mirzaei/PAPER_RESULTS_AUDIT_20260924/synthetic_archive_sources.json`; DeFoG per-seed files: `/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/metrics/lobster_seed{0,1,2}.json`.',
          '- Matching evaluation reference sets does not retrospectively equalize different historical training splits or training budgets. PROTEINS still needs that qualification; PTC structural values are not repaired by this S-only run.',
          '- All requested S metrics were finite. Optional TensorFlow-dependent metrics are unavailable and are not claimed here.',
          '- LOBSTER full has higher mean F1-PR and lower diameter MMD than both baselines; the small F1 margin is a numerical comparison, not a significance result.']
(ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
print(ROOT / 'RESULTS.md')
