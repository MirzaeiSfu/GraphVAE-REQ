#!/usr/bin/env python3
"""Fixed, positive retained-path log-count MMD; read-only reuse of count caches."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

SOURCE = Path('/local-scratch2/mirzaei/rule_mmd_20260923')
OUT = Path('/local-scratch2/mirzaei/metapath_instance_mmd_20260923')
METHODS = ('motif_true_full', 'motif_false', 'defog')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def positive_path(entry):
    """At least two positive edge atoms forming one open chain; no extra relations."""
    atoms = []
    for atom, state in zip(entry['rule'], entry['state']):
        match = re.fullmatch(r'([^()]+)\(([^()]*)\)', atom)
        if not match:
            return False
        atoms.append((match[1], tuple(x.strip() for x in match[2].split(',')), state))
    edges = [args for name, args, value in atoms if name.lower() in ('edge', 'edges')]
    if len(edges) < 2 or any(len(e) != 2 or e[0] == e[1] for e in edges):
        return False
    if any(value.lower() not in ('t', 'true', '1') for name, args, value in atoms
           if name.lower() in ('edge', 'edges')):
        return False
    pairs = {frozenset(e) for e in edges}
    if len(pairs) != len(edges):
        return False
    neighbours = {}
    for u, v in edges:
        neighbours.setdefault(u, set()).add(v)
        neighbours.setdefault(v, set()).add(u)
    seen, todo = set(), [next(iter(neighbours))]
    while todo:
        u = todo.pop()
        if u not in seen:
            seen.add(u)
            todo.extend(neighbours[u] - seen)
    if len(seen) != len(neighbours) or sorted(map(len, neighbours.values())) != [1, 1] + [2] * (len(neighbours)-2):
        return False
    for name, args, value in atoms:
        if len(args) == 1 and args[0] not in seen:
            return False
        if len(args) == 2 and frozenset(args) not in pairs:
            return False
        if len(args) not in (1, 2):
            return False
    return True


def load_collection(folder, name, width, columns):
    direct = folder / (name + '.npz')
    paths = [direct] if direct.exists() else sorted(folder.glob(name + '.shard*of*.npz'))
    if not paths:
        raise FileNotFoundError(name)
    matrices, nodes, positions, metadata = [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            a = sparse.csr_matrix(d['counts']) if 'counts' in d else sparse.csr_matrix(
                (d['data'], d['indices'], d['indptr']), shape=tuple(d['shape']))
            assert a.shape[1] == width, (name, a.shape, width)
            meta = json.loads(str(d['meta']))
            matrices.append(a[:, columns].toarray().astype(float))
            nodes.append(d['node_counts'].copy())
            positions.extend(meta.get('positions', range(a.shape[0])))
            metadata.append(dict(meta, count_file=str(path), count_sha256=sha(path)))
    expected = metadata[0].get('shard', [0, 1])[1]
    if not direct.exists():
        assert len(paths) == expected
        assert len(set(positions)) == len(positions)
        assert sorted(positions) == list(range(len(positions)))
    order = np.argsort(positions)
    a, n = np.vstack(matrices)[order], np.concatenate(nodes)[order]
    assert np.isfinite(a).all() and (a >= 0).all() and len(a)
    return np.log1p(a), dict(graph_count=len(a), empty_after_cleanup=int((n == 0).sum()), sources=metadata)


def score(x, y, sigma2):
    xx = np.exp(-cdist(x, x, 'sqeuclidean') / (2*sigma2))
    yy = np.exp(-cdist(y, y, 'sqeuclidean') / (2*sigma2))
    xy = np.exp(-cdist(x, y, 'sqeuclidean') / (2*sigma2))
    biased = float(xx.mean() + yy.mean() - 2*xy.mean())
    assert biased >= -1e-10
    n, m = len(x), len(y)
    unbiased = float((xx.sum()-np.trace(xx))/(n*(n-1)) +
                     (yy.sum()-np.trace(yy))/(m*(m-1)) - 2*xy.mean()) if min(n, m) > 1 else None
    return dict(mmd2_biased=max(0., biased), mmd2_unbiased_diagnostic=unbiased)


def run(dataset):
    out = OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    (out/'status.txt').write_text('RUNNING\n')
    directory = SOURCE/'datasets'/dataset
    selection = json.loads((directory/'selection.json').read_text())
    inputs = json.loads((directory/'inputs.json').read_text())
    assert selection['selections_identical'], 'Per-seed training selections differ; do not silently union'
    layout = selection['layout']
    columns = [c for c in selection['pruned_columns'] if positive_path(layout[c])]
    if not columns:
        raise ValueError('No retained positive multi-edge open-chain states')
    result = dict(dataset=dataset, metric='positive-retained-metapath-log-count-MMD2',
        reference='motif=True training split, existing deterministic cap at 1000',
        descriptor='log1p(count of each positive retained path state per graph); no state or node normalization',
        kernel='exp(-squared Euclidean distance / (2*sigma2)); sigma2 median training pairwise squared distance',
        main_estimator='biased empirical squared MMD, no x100 scaling',
        preprocessing='reuse identical hard counting cleanup: undirected, no loops/isolates, largest component; empty rows retained',
        grounding='unchanged inherited training-counter grounding semantics, not a claim of injective simple paths',
        scope='training fidelity diagnostic; positive path subset only, not all training states',
        path_state_count=len(columns), paths=[layout[c] for c in columns],
        input_path=str(directory/'inputs.json'), input_sha256=sha(directory/'inputs.json'),
        selection_sha256=sha(directory/'selection.json'),
        existing_qualifications=inputs.get('open_issues', []), per_seed=[], missing=[])
    ref, refmeta = load_collection(directory/'counts', 'train', len(layout), columns)
    dist = cdist(ref, ref, 'sqeuclidean')
    values = dist[np.triu_indices(len(ref), 1)]
    sigma2 = float(np.median(values)) if len(values) else 0.
    if sigma2 <= 0:
        positive = values[values > 0]
        sigma2 = float(np.median(positive)) if len(positive) else 1.
        result['bandwidth_fallback'] = 'median positive train distances; 1 if all train descriptors identical'
    result.update(sigma2=sigma2, reference_provenance=refmeta)
    for item in inputs['items']:
        if item['method'] not in METHODS:
            continue
        name = f"{item['method']}_seed_{item['seed']}"
        try:
            gen, meta = load_collection(directory/'counts', name, len(layout), columns)
            row = dict(method=item['method'], seed=item['seed'], provenance=meta, **score(ref, gen, sigma2))
            result['per_seed'].append(row)
        except Exception as e:
            result['missing'].append(dict(collection=name, reason=repr(e)))
    result['aggregates'] = {}
    for method in METHODS:
        rows = [r for r in result['per_seed'] if r['method'] == method]
        v = [r['mmd2_biased'] for r in rows]
        result['aggregates'][method] = dict(n=len(v), seeds=[r['seed'] for r in rows],
            mean=float(np.mean(v)) if v else None, sample_sd=float(np.std(v, ddof=1)) if len(v)>1 else None)
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = [f'# {dataset}: retained positive metapath-instance MMD', '',
        f'{len(columns)} path-state descriptors; training reference {len(ref)} graphs. Lower is better.', '',
        '| Method | Seeds | Mean | Sample SD |', '|---|---|---:|---:|']
    for method, a in result['aggregates'].items():
        lines.append(f"| {method} | {a['seeds']} | {a['mean']} | {a['sample_sd']} |")
    lines += ['', '## Per-seed results', '', '| Method | Seed | Generated graphs | MMD² |', '|---|---:|---:|---:|']
    for r in result['per_seed']:
        lines.append(f"| {r['method']} | {r['seed']} | {r['provenance']['graph_count']} | {r['mmd2_biased']:.9g} |")
    lines += ['', 'The JSON records selected paths, provenance, missing results and inherited protocol limitations.',
              'No model retraining or generated-graph filtering was performed by this scorer.',
              'Source collections can have historical seed-selection/preprocessing restrictions; these are not repaired by rescoring.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'status.txt').write_text(('PARTIAL' if result['missing'] else 'COMPLETE')+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    args = parser.parse_args()
    for dataset in args.datasets:
        try:
            run(dataset)
        except Exception as exc:
            p = OUT/dataset
            p.mkdir(parents=True, exist_ok=True)
            (p/'status.txt').write_text('BLOCKED: '+repr(exc)+'\n')
            import traceback
            traceback.print_exc()
