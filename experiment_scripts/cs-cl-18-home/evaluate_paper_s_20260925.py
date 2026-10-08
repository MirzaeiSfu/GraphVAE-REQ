"""Rescore frozen graph collections with a shared reference and S-input GIN."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import torch

REPO = Path('/local-scratch2/mirzaei/fb/GraphVAE-REQ')
sys.path.insert(0, str(REPO / 'scripts'))
from evaluate_graph_realism_batch import evaluate_graph_collections

ROOT = Path('/local-scratch2/mirzaei/PAPER_S_RANDOM_GIN_20260925')

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    graphs, _ = dgl.load_graphs(str(path))
    output = []
    for i, graph in enumerate(graphs):
        g = nx.Graph(dgl.to_networkx(graph, node_attrs=[], edge_attrs=[]))
        g.remove_edges_from(list(nx.selfloop_edges(g)))
        g.remove_nodes_from(list(nx.isolates(g)))
        if not g.number_of_nodes():
            raise ValueError('Empty graph {} in {}'.format(i, path))
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
        output.append(nx.convert_node_labels_to_integers(g))
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['PROTEINS', 'QM9', 'AIDS', 'OGB'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = ROOT / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset == 'PROTEINS':
        base = Path('/local-scratch2/mirzaei/archive_completion_20260922/PROTEINS')
        ref = base / 'true_full_003/seed_0/reference_attributed_graphs.bin'
        paths = {
            'motif_false': [base / ('false/seed_%d/generated_attributed_graphs.bin' % s) for s in range(3)],
            'motif_true_full': [base / ('true_full_003/seed_%d/generated_attributed_graphs.bin' % s) for s in range(3)],
            'defog': [Path('/local-scratch2/mirzaei/proteins_defog_corrected_eval_20260922/seed_%d/generated.bin' % s) for s in range(3)],
        }
    elif args.dataset == 'OGB':
        base = Path('/local-scratch2/mirzaei/common_eval_fix_20260923/OGB/aligned')
        ref = base / 'reference.bin'
        paths = {
            'motif_false': [base / ('false/seed_%d/generated.bin' % s) for s in range(3)],
            'motif_true_full': [base / ('true_full/seed_%d/generated.bin' % s) for s in range(3)],
            'defog': [base / ('defog/seed_%d/generated.bin' % s) for s in range(3)],
        }
    elif args.dataset == 'AIDS':
        base = Path('/local-scratch2/mirzaei/aids_common_eval_10k_20260917/evaluations/graphvae')
        ref = base / 'false/seed_0/reference_attributed_graphs.bin'
        paths = {
            'motif_false': [base / ('false/seed_%d/generated_attributed_graphs.bin' % s) for s in range(3)],
            'motif_true_full': [base / ('true_full/seed_%d/generated_attributed_graphs.bin' % s) for s in range(3)],
            'defog': [Path('/local-scratch2/mirzaei/common_eval_fix_20260923/AIDS/stage/defog/seed_%d/generated.bin' % s) for s in [3, 4, 5]],
        }
    else:
        base = Path('/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluations')
        ref = base / 'graphvae/true_full/seed_0/reference_attributed_graphs.bin'
        paths = {
            'motif_false': [base / ('graphvae/false/seed_%d/generated_attributed_graphs.bin' % s) for s in range(3)],
            'motif_true_full': [base / ('graphvae/true_full/seed_%d/generated_attributed_graphs.bin' % s) for s in range(3)],
            'defog': [base / ('defog/seed_%d/generated_attributed_graphs.bin' % s) for s in range(3)],
        }
    for p in [ref] + [p for ps in paths.values() for p in ps]:
        assert p.is_file(), str(p)
    reference = load(ref)
    print('REFERENCE', ref, len(reference), flush=True)
    results = {}
    for method, sources in paths.items():
        results[method] = []
        for source_index, src in enumerate(sources):
            seed = source_index + 3 if args.dataset == 'AIDS' and method == 'defog' else source_index
            dest = out / ('%s_seed_%d.json' % (method, seed))
            if dest.is_file():
                existing = json.loads(dest.read_text())
                assert existing['reference_sha256'] == digest(ref)
                assert existing['generated_sha256'] == digest(src)
                results[method].append(existing)
                continue
            print('START', method, seed, flush=True)
            generated = load(src)
            saved_count = len(generated)
            if args.dataset == 'PROTEINS' and method == 'defog':
                # Match the original common evaluator's first-N selection.
                assert saved_count == 210 and len(reference) == 209
                generated = generated[:len(reference)]
            assert len(generated) == len(reference), (len(generated), len(reference))
            result = evaluate_graph_collections(generated, reference, 10, 0, torch.device('cuda'), True)
            assert all(np.isfinite(v) for vals in result['raw_metrics'].values() for v in vals)
            result.update(dataset=args.dataset, method=method, training_seed=seed,
                          generated_path=str(src), generated_sha256=digest(src),
                          reference_path=str(ref), reference_sha256=digest(ref),
                          evaluator_sha256=digest(REPO / 'scripts/evaluate_graph_realism_batch.py'),
                          input_channels=['degree', 'clustering', 'square_clustering'],
                          saved_generated_count=saved_count,
                          selection='first N saved graphs, N=reference size; no score-based selection',
                          preprocessing='undirected simple; remove loops/isolates; largest component; then evaluator self loops',
                          caveat='Common evaluation reference does not change historical training splits.')
            if args.dataset == 'OGB':
                result['selection'] = 'Unchanged pre-aligned 289-graph collections; historical GraphVAE selected terminal chain, nonempty/largest-component filtering and first-N truncation retained.'
                result['alignment_manifest'] = '/local-scratch2/mirzaei/common_eval_fix_20260923/OGB/alignment_manifest.json'
            dest.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            results[method].append(result)
            print('DONE', method, seed, result['metrics']['f1_pr'], flush=True)
    summary = {}
    for method, rows in results.items():
        summary[method] = {}
        for metric in rows[0]['metrics']:
            vals = [r['metrics'][metric]['mean'] for r in rows]
            summary[method][metric] = {'mean': float(np.mean(vals)), 'sample_sd': float(np.std(vals, ddof=1)), 'per_seed': vals, 'n': 3}
    (out / 'aggregate.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('COMPLETE', args.dataset, json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
