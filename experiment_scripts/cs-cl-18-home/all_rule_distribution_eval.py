#!/usr/bin/env python3
"""Full unpruned rule-state TV followed by Hellinger; read-only count reuse."""
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
from scipy import sparse
from metapath_instance_mmd import SOURCE, METHODS, sha

OUT = Path('/local-scratch2/mirzaei/all_rule_distributions_20260924')
PREVIOUS = Path('/local-scratch2/mirzaei/rule_gaussian_fidelity_20260924')


def distances(a, b):
    """Normalize collection sums, not individual graphs. Zero mass is explicit."""
    sa, sb = float(a.sum()), float(b.sum())
    if sa == 0 or sb == 0:
        # Add one 'no instances' state to both supports. It has unit mass only
        # for zero-total collections; no smoothing of observed states.
        v = float((sa == 0) != (sb == 0))
        return v, v
    p, q = a / sa, b / sb
    tv = float(0.5 * np.abs(p-q).sum())
    h = float(np.sqrt(0.5 * np.square(np.sqrt(p)-np.sqrt(q)).sum()))
    return tv, h


def collect(folder, name, width, previous):
    direct = folder / (name+'.npz')
    paths = [direct] if direct.exists() else sorted(folder.glob(name+'.shard*of*.npz'))
    if not paths:
        raise FileNotFoundError(name)
    known = {x['count_file']: x['count_sha256'] for x in previous['sources']}
    assert set(map(str, paths)) == set(known), 'Count files differ from audited campaign'
    total = np.zeros(width, dtype=np.float64)
    positions, shard_ids, sources = [], [], []
    n, empty, nshards = 0, 0, None
    for path in paths:
        digest = sha(path)
        assert digest == known[str(path)], 'Count file changed: '+str(path)
        with np.load(path, allow_pickle=False) as d:
            a = sparse.csr_matrix(d['counts'], dtype=np.float64) if 'counts' in d else sparse.csr_matrix(
                (d['data'].astype(np.float64), d['indices'], d['indptr']), shape=tuple(d['shape']))
            meta = json.loads(str(d['meta']))
            assert a.shape[1] == width and a.shape[0] > 0
            assert np.isfinite(a.data).all() and (a.data >= 0).all()
            assert len(d['node_counts']) == a.shape[0]
            assert meta.get('collection', name) == name
            assert meta.get('state_count', width) == width
            pos = meta.get('positions', list(range(a.shape[0])))
            assert len(pos) == a.shape[0]
            positions.extend(pos)
            if not direct.exists():
                sid, size = meta['shard']
                if nshards is None:
                    nshards = size
                assert nshards == size
                shard_ids.append(sid)
            total += np.asarray(a.sum(axis=0)).ravel()
            n += a.shape[0]
            empty += int((d['node_counts'] == 0).sum())
            sources.append(dict(path=str(path), sha256=digest, graph_count=a.shape[0], metadata=meta))
    assert sorted(positions) == list(range(n)), 'Missing or duplicate graph positions'
    if not direct.exists():
        assert sorted(shard_ids) == list(range(nshards)), 'Missing or duplicate shards'
    assert n == previous['graph_count']
    return total, dict(graph_count=n, empty_after_cleanup=empty, sources=sources)


def run(ds):
    out = OUT/ds
    out.mkdir(parents=True, exist_ok=True)
    (out/'status.txt').write_text('RUNNING: loading full-state counts\n')
    directory = SOURCE/'datasets'/ds
    prior = json.loads((PREVIOUS/ds/'results.json').read_text())
    assert sha(directory/'selection.json') == prior['selection_sha256']
    assert sha(directory/'inputs.json') == prior['input_sha256']
    selection = json.loads((directory/'selection.json').read_text())
    inputs = json.loads((directory/'inputs.json').read_text())
    layout = selection['layout']
    width = len(layout)
    assert all(e['column'] == i for i, e in enumerate(layout))
    assert width == selection['full_state_count']
    rule_ids = np.array([e['rule_index'] for e in layout])
    groups = {int(r): np.flatnonzero(rule_ids == r) for r in np.unique(rule_ids)}
    templates = {str(r): dict(rule=layout[int(idx[0])]['rule'], states=len(idx)) for r, idx in groups.items()}
    result = dict(dataset=ds, support='all unpruned states in staged training-counter layout',
        full_state_count=width, retained_state_count=len(selection['pruned_columns']),
        rule_templates=templates, reference='training split of motif=True, existing deterministic cap at 1000',
        input_sha256=prior['input_sha256'], selection_sha256=prior['selection_sha256'],
        existing_qualifications=inputs.get('open_issues', []), per_seed=[], aggregates={}, missing=[])
    ref, rm = collect(directory/'counts', 'train', width, prior['provenance']['train'])
    result['reference_provenance'] = rm
    np.savez_compressed(out/'train_aggregate_counts.npz', counts=ref)
    plan = [x for x in inputs['items'] if x['method'] in METHODS]
    for item in plan:
        name = f"{item['method']}_seed_{item['seed']}"
        print(ds, name, 'full states', width, flush=True)
        try:
            gen, gm = collect(directory/'counts', name, width, prior['provenance'][name])
            np.savez_compressed(out/(name+'_aggregate_counts.npz'), counts=gen)
            per_rule = {}
            for r, idx in groups.items():
                tv, h = distances(ref[idx], gen[idx])
                per_rule[str(r)] = dict(tv=tv, hellinger=h, reference_mass=float(ref[idx].sum()),
                    generated_mass=float(gen[idx].sum()), states=len(idx))
            result['per_seed'].append(dict(method=item['method'], seed=item['seed'],
                tv=float(np.mean([v['tv'] for v in per_rule.values()])),
                hellinger=float(np.mean([v['hellinger'] for v in per_rule.values()])),
                per_rule=per_rule, provenance=gm))
        except Exception as e:
            traceback.print_exc()
            result['missing'].append(dict(collection=name, error=repr(e)))
    for method in METHODS:
        rows = [r for r in result['per_seed'] if r['method'] == method]
        result['aggregates'][method] = dict(n=len(rows), seeds=[r['seed'] for r in rows])
        for metric in ('tv', 'hellinger'):
            vals = [r[metric] for r in rows]
            result['aggregates'][method][metric] = dict(mean=float(np.mean(vals)) if vals else None,
                sample_sd=float(np.std(vals, ddof=1)) if len(vals)>1 else None)
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = [f'# {ds}: all-rule state distributions', '',
        f'{width} unpruned states, {len(groups)} rule templates. Training reference: {rm["graph_count"]} graphs.', '',
        'Lower is better for both metrics. Mean ± sample SD across generator seeds.', '',
        'TV is the earlier custom "motif correlation" proxy, not paper-defined motif correlation.', '',
        '| Method | Seeds | Full-state TV | Full-state Hellinger |', '|---|---|---:|---:|']
    def fmt(x):
        return 'N/A' if x['mean'] is None else f"{x['mean']:.8g} ± {x['sample_sd']:.8g}" if x['sample_sd'] is not None else f"{x['mean']:.8g} (one seed)"
    for m, a in result['aggregates'].items():
        lines.append(f"| {m} | {a['seeds']} | {fmt(a['tv'])} | {fmt(a['hellinger'])} |")
    lines += ['', '## Per-seed results', '', '| Method | Seed | Graphs | TV | Hellinger |', '|---|---:|---:|---:|---:|']
    for r in result['per_seed']:
        lines.append(f"| {r['method']} | {r['seed']} | {r['provenance']['graph_count']} | {r['tv']:.8g} | {r['hellinger']:.8g} |")
    lines += ['', 'See ../README.md for formulas, zero-mass convention and limitations. Per-rule results and source hashes are in results.json.',
        'Existing seed selections, graph preprocessing and split differences remain; rescoring does not resolve them.']
    if result['missing']:
        lines += ['', 'Missing: '+json.dumps(result['missing'])]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'status.txt').write_text(('PARTIAL' if result['missing'] else 'COMPLETE')+'\n')


def self_test():
    a, b = np.array([3., 1.]), np.array([1., 3.])
    assert np.allclose(distances(a, b), [.5, np.sqrt(1-np.sqrt(3)/2)])
    assert distances(a, a) == (0., 0.)
    assert distances(np.array([1., 0.]), np.array([0., 1.])) == (1., 1.)
    assert distances(np.zeros(2), np.zeros(2)) == (0., 0.)
    assert distances(a, np.zeros(2)) == (1., 1.)
    assert np.allclose(distances(a, b), distances(10*a, 2*b))
    print('SELF_TEST_PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('datasets', nargs='*')
    args = p.parse_args()
    self_test()
    for ds in args.datasets:
        try:
            run(ds)
        except Exception as e:
            out = OUT/ds
            out.mkdir(parents=True, exist_ok=True)
            (out/'status.txt').write_text('FAILED: '+repr(e)+'\n')
            traceback.print_exc()
