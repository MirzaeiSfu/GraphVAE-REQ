"""Re-evaluate the three saved PTC motif=False collections on the common reference."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path('/local-scratch2/mirzaei/fb/GraphVAE-REQ')
OUT = Path('/local-scratch2/mirzaei/PTC_MATCHED_STRUCTURAL_20260925')
sys.path.insert(0, str(REPO))
from evaluate_defog_synthetic_20260906 import load_safe_pyg, canonical_graph_digest
from stat_rnn import degree_stats, clustering_stats, orbit_stats_all, spectral_stats, MMD_diam, MMD_triangles, sparsity_stats_all
from scripts.evaluate_graph_realism_batch import load_graph_items, item_to_graph

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metrics(reference, generated):
    result = {}
    for name, fn in [('degree', degree_stats), ('clustering', clustering_stats),
                     ('orbit', orbit_stats_all), ('spectral', spectral_stats),
                     ('diameter', MMD_diam), ('triangle', MMD_triangles)]:
        result[name] = float(fn(reference, generated))
    sparsity, re, ge = sparsity_stats_all(reference, generated)
    result.update(sparsity=float(sparsity), reference_edge_count=float(re),
                  generated_edge_count=float(ge), edge_count_absolute_error=float(abs(re-ge)))
    assert all(np.isfinite(v) for v in result.values())
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    rp = Path('/local-scratch2/mirzaei/defog_ptc_frozen_20260906/artifacts/ptc/real_test_graphs.pt')
    reference, _, _ = load_safe_pyg(rp)
    rd = canonical_graph_digest(reference)
    assert len(reference) == 70
    oldroot = Path('/local-scratch2/mirzaei/defog_ptc_full_metrics_20260907/structural')
    for seed in range(3):
        old = json.loads((oldroot / ('seed_%d.json' % seed)).read_text())
        assert old['reference_canonical_graph_sha256'] == rd
    # Check the implementation against an archived DeFoG result before changing a table.
    old = json.loads((oldroot / 'seed_0.json').read_text())
    graphs, _, _ = load_safe_pyg(Path(old['generated_path']))
    check = metrics(reference, graphs)
    for key, val in check.items():
        assert np.isclose(val, old['structural_mmd'][key], atol=1e-8, rtol=1e-6), (key, val, old['structural_mmd'][key])
    (OUT / 'implementation_validation.json').write_text(json.dumps({'reference_canonical_sha256': rd, 'old_result': str(oldroot / 'seed_0.json'), 'recomputed': check, 'all_match': True}, indent=2))
    results = []
    for seed in range(3):
        print('START seed', seed, flush=True)
        base = Path('/local-scratch2/mirzaei/ptc_matched_reference_eval_20260913')
        gp = base / ('generated/graphvae_motif_false_seed%d.pt' % seed)
        generated, meta, _ = load_safe_pyg(gp)
        assert len(generated) == 70
        # Verify this is the same collection/reference used by the paper's corrected S-input F1.
        npybase = base / ('topology_feature/seed_%d' % seed)
        for name, expected in [('testGraphs_adj_.npy', reference), ('Single_comp_generatedGraphs_adj_final_eval.npy', generated)]:
            converted = [item_to_graph(g) for g in load_graph_items(npybase / name)]
            import networkx as nx
            converted = [nx.convert_node_labels_to_integers(g) for g in converted]
            assert canonical_graph_digest(converted) == canonical_graph_digest(expected), name
        st = metrics(reference, generated)
        payload = dict(dataset='PTC', method='motif_false', seed=seed,
                       reference_path=str(rp), reference_sha256=sha(rp), reference_canonical_sha256=rd,
                       generated_path=str(gp), generated_sha256=sha(gp), metadata=meta,
                       graph_count=70, structural_mmd=st,
                       stat_rnn_sha256=sha(REPO / 'stat_rnn.py'),
                       comparison='Same saved generated collections as matched-reference S-input F1; no regeneration or retraining')
        (OUT / ('seed_%d.json' % seed)).write_text(json.dumps(payload, indent=2, allow_nan=False))
        results.append(st)
        print('DONE', seed, st, flush=True)
    aggregate = {k: {'mean': float(np.mean([r[k] for r in results])), 'sample_sd': float(np.std([r[k] for r in results], ddof=1)), 'per_seed': [r[k] for r in results]} for k in results[0]}
    (OUT / 'aggregate.json').write_text(json.dumps(aggregate, indent=2, allow_nan=False))
    print('COMPLETE', json.dumps(aggregate), flush=True)

if __name__ == '__main__':
    main()
