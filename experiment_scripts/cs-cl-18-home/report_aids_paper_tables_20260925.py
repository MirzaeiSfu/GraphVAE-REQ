"""Build the requested AIDS paper rows from current, independent-seed results."""
import json
from pathlib import Path
import numpy as np

ROOT = Path('/local-scratch2/mirzaei/PAPER_S_RANDOM_GIN_20260925/AIDS')
methods = ['motif_false', 'motif_true_full', 'defog']
labels = ['GraphVAE', 'GraphVAE+RG', 'DeFoG']
keys = ['degree', 'clustering', 'orbit', 'spectral', 'diameter']
summary = {}
source_paths = {}
for method in methods:
    if method == 'defog':
        paths = [Path('/local-scratch2/mirzaei/common_eval_fix_20260923/AIDS/results/defog/seed_%d/structural.json' % s) for s in [3, 4, 5]]
    else:
        setting = {'motif_false': 'false', 'motif_true_full': 'true_full'}[method]
        paths = [Path('/local-scratch2/mirzaei/aids_common_eval_10k_20260917/evaluations/graphvae/%s/seed_%d/common_structural.json' % (setting, s)) for s in range(3)]
    rows = [json.loads(p.read_text()) for p in paths]
    source_paths[method] = [str(p) for p in paths]
    assert all(r['graph_count'] == 400 for r in rows)
    summary[method] = {k: {'mean': float(np.mean([r['structural_mmd'][k] for r in rows])),
                           'sample_sd': float(np.std([r['structural_mmd'][k] for r in rows], ddof=1)),
                           'per_seed': [r['structural_mmd'][k] for r in rows]}
                       for k in rows[0]['structural_mmd']}
    seeds = [3, 4, 5] if method == 'defog' else [0, 1, 2]
    for seed, row in zip(seeds, rows):
        s = json.loads((ROOT / ('%s_seed_%d.json' % (method, seed))).read_text())
        assert s['generated_path'] == row['generated']

gin = json.loads((ROOT / 'aggregate.json').read_text())
f1 = [gin[m]['f1_pr']['mean'] for m in methods]
improvements = [100 * (f1[1] / f1[0] - 1), 100 * (f1[1] / f1[2] - 1)]

def cell(value, best=False):
    txt = '%.4f $\\pm$ %.4f' % (value['mean'], value['sample_sd'])
    return '\\textbf{' + txt + '}' if best else txt

f1row = 'AIDS & S & ' + ' & '.join(cell(gin[m]['f1_pr'], f1[i] == max(f1)) for i, m in enumerate(methods)) + (' & %+.2f\\%% & %+.2f\\%% \\\\' % tuple(improvements))
structrows = []
for i, method in enumerate(methods):
    prefix = '\\multirow{3}{*}{AIDS} ' if i == 0 else ''
    structrows.append(prefix + '& ' + labels[i] + ' & ' + ' & '.join(cell(summary[method][k], summary[method][k]['mean'] == min(summary[m][k]['mean'] for m in methods)) for k in keys) + ' \\\\')

tex = '\n'.join([
    '% Three generator seeds per method; ten evaluator initializations per generator.',
    '% GV/RG seeds 0,1,2; independent DeFoG seeds 3,4,5. S inputs are topology-derived.',
    '% Insert in the Random-GIN F1-PR table:', f1row, '',
    '% Insert in the structural table (degree, clustering, orbit, spectral, diameter):',
    *structrows, ''])
(ROOT / 'AIDS_TABLE_ROWS.tex').write_text(tex)
report = ['# AIDS paper tables', '',
          'GraphVAE without motifs versus full-matrix GraphVAE+RG versus DeFoG. GraphVAE variants use generator seeds 0, 1, 2 from the saved AIDS 10k evaluation campaign; DeFoG uses the newer independent seeds 3, 4, 5. The older duplicate DeFoG seeds are not used.', '',
          'All methods use 400 saved generated graphs per seed and the same 400-graph reference. The normalized reference topology was checked across all six GraphVAE bundles and the DeFoG reference and is identical. S uses degree, clustering and square-clustering inputs, not native node or edge attributes. Ten evaluator initializations (0–9) are averaged within each generator seed; the reported uncertainty is sample SD across three generator seeds. No retraining or graph regeneration was needed.', '',
          '## Random-GIN F1-PR', '',
          '| Dataset | Input | GraphVAE | GraphVAE+RG | DeFoG | RG vs. GV | RG vs. DeFoG |',
          '|---|---|---:|---:|---:|---:|---:|',
          '| AIDS | S | ' + ' | '.join('%.6f ± %.6f' % (gin[m]['f1_pr']['mean'], gin[m]['f1_pr']['sample_sd']) for m in methods) + (' | %+.2f%% | %+.2f%% |' % tuple(improvements)), '',
          '## Structural squared MMD', '',
          '| Method | Degree | Clustering | Orbit | Spectral | Diameter |',
          '|---|---:|---:|---:|---:|---:|']
for m, label in zip(methods, labels):
    report.append('| ' + label + ' | ' + ' | '.join('%.6f ± %.6f' % (summary[m][k]['mean'], summary[m][k]['sample_sd']) for k in keys) + ' |')
report += ['', '## LaTeX rows', '', '```latex', tex.rstrip(), '```', '',
           '## Per-generator-seed S F1-PR', '', '| Method | Seeds | F1-PR values |', '|---|---|---|']
for m, label in zip(methods, labels):
    report.append('| ' + label + ' | ' + ('3, 4, 5' if m == 'defog' else '0, 1, 2') + ' | ' + ', '.join('%.6f' % v for v in gin[m]['f1_pr']['per_seed']) + ' |')
report += ['', '## Sources', '', '- S-input evaluation JSONs and raw repeat values are alongside this report.',
           '- Reference normalized-topology digest (ordered graph collection): `b7123bbc661bd7f9758b833670003552fb67e7e09abc1b9cf51f5f7a0853f6c1`.',
           '- The saved DeFoG configurations specify 1,592 training epochs and seeds 3, 4, 5, with generation seeds 9303, 9304, 9305. Their recorded checkpoint hashes are distinct. This is not an equal-epoch comparison to the GraphVAE+RG 10k evaluation campaign.',
           '- DeFoG configuration and checkpoint-hash records: `/local-scratch2/mirzaei/aids_defog_independent_20260922/seed_{3,4,5}/.hydra/config.yaml` and `checkpoint.sha256`.',
           '- This is a topology-only comparison and does not establish native-attribute quality or chemical validity. Shared evaluation graphs do not by themselves verify all historical training choices.']
for m in methods:
    report += ['- ' + m + ':'] + ['  - `' + p + '`' for p in source_paths[m]]
(ROOT / 'AIDS_PAPER_TABLES.md').write_text('\n'.join(report) + '\n')
(ROOT / 'structural_aggregate.json').write_text(json.dumps({'metrics': summary, 'sources': source_paths}, indent=2, allow_nan=False))
print(tex)
