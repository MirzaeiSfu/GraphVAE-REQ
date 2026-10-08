"""Full-test GraphVAE generation and exact, bounded-memory structural MMD."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist

ROOT = Path('/local-scratch2/mirzaei/QM9_FULL_TEST_20260925')
OLD = Path('/local-scratch2/mirzaei/qm9_common_eval_20260917')
REPO = OLD / 'source/GraphVAE-REQ'
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
import evaluate_attributed_graph_realism_checkpoints as generation
import stat_rnn

def exact_disc(a, b, kernel, is_parallel=True, *args, **kwargs):
    if kernel.__name__ != 'gaussian_tv' or args or set(kwargs) - {'sigma'}:
        raise ValueError('Unsupported kernel: no silent approximation')
    sigma = kwargs.get('sigma', 1.)
    dim = max(max(len(x) for x in a), max(len(x) for x in b))
    def pack(rows):
        arr = np.zeros((len(rows), dim), dtype=np.float64)
        for i, row in enumerate(rows):
            arr[i, :len(row)] = row
        assert np.isfinite(arr).all()
        return np.unique(arr, axis=0, return_counts=True)
    aa, ac = pack(a)
    bb, bc = pack(b)
    total = 0.
    for i in range(0, len(aa), 256):
        dist = cdist(aa[i:i+256], bb, metric='cityblock') / 2.
        values = np.exp(-dist * dist / (2 * sigma * sigma))
        total += float(ac[i:i+256].astype(float) @ (values @ bc.astype(float)))
    return total / (len(a) * len(b))

def selfcheck():
    rng = np.random.RandomState(123)
    a = [rng.rand(5), rng.rand(3), np.array([1., 0.])]
    a.append(a[0].copy())
    b = [rng.rand(4), np.array([1., 0.]), np.array([1., 0.])]
    for sigma in [1., .1, 30.]:
        expected = sum(stat_rnn.mmd.gaussian_tv(x, y, sigma=sigma) for x in a for y in b) / (len(a)*len(b))
        actual = exact_disc(a, b, stat_rnn.mmd.gaussian_tv, sigma=sigma)
        assert np.isclose(expected, actual, atol=1e-13, rtol=1e-12)

def evaluate(gen_path, ref_path, out):
    import dgl
    import networkx as nx
    def load(path):
        result = []
        for graph in dgl.load_graphs(str(path))[0]:
            g = nx.Graph(dgl.to_networkx(graph, node_attrs=[], edge_attrs=[]))
            g.remove_edges_from(list(nx.selfloop_edges(g)))
            g.remove_nodes_from(list(nx.isolates(g)))
            assert len(g), 'Empty graph: stop instead of silently changing sample count'
            result.append(nx.convert_node_labels_to_integers(g.subgraph(max(nx.connected_components(g), key=len)).copy()))
        return result
    reference, generated = load(ref_path), load(gen_path)
    assert len(reference) == len(generated) == 26167
    selfcheck()
    stat_rnn.mmd.disc = exact_disc
    # Orbit routines catch errors internally: verify all calls completed.
    original_orca = stat_rnn.orca
    orbit_success = []
    def checked_orca(g):
        values = original_orca(g)
        assert len(values) == len(g)
        orbit_success.append(1)
        return values
    stat_rnn.orca = checked_orca
    payload = {'count': len(reference), 'generated': str(gen_path), 'reference': str(ref_path),
               'estimator': 'biased squared MMD; exact weighted duplicate compression and blockwise sums', 'metrics': {}}
    for name, fn in [('degree',stat_rnn.degree_stats), ('clustering',stat_rnn.clustering_stats),
                     ('orbit',stat_rnn.orbit_stats_all), ('spectral',stat_rnn.spectral_stats), ('diameter',stat_rnn.MMD_diam)]:
        start = time.monotonic()
        value = float(fn(reference, generated))
        assert np.isfinite(value)
        if name == 'orbit':
            assert len(orbit_success) == len(reference) + len(generated)
        payload['metrics'][name] = value
        payload[name+'_seconds'] = time.monotonic()-start
        out.write_text(json.dumps(payload, indent=2, allow_nan=False))
        print('MMD_DONE', name, value, payload[name+'_seconds'], flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['false','true_full','external','selfcheck'])
    parser.add_argument('--generated')
    parser.add_argument('--reference')
    parser.add_argument('--output')
    args = parser.parse_args()
    os.chdir(REPO)
    torch.set_num_threads(2)
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.method == 'selfcheck':
        selfcheck()
        print('Exact kernel selfcheck passed')
        return
    if args.method == 'external':
        evaluate(Path(args.generated), Path(args.reference), Path(args.output))
        return
    for seed in range(3):
        out = ROOT / args.method / ('seed_%d'%seed)
        out.mkdir(parents=True, exist_ok=True)
        run = OLD / 'graphvae' / args.method / ('seed_%d'%seed)
        config = generation.load_config(run / 'run_config_used.yaml')
        config['dataset_cache_dir'] = str(OLD / 'cache/dataset')
        cache = generation.load_cached_dataset(config)
        reference = generation.build_reference_graphs(cache, 'test', 0, .5)
        assert len(reference) == 26167
        device = torch.device('cuda')
        checkpoint = generation.resolve_checkpoint(run, None)
        model = generation.build_model(config, cache, device)
        model.load_state_dict(generation._checkpoint_state_dict(checkpoint, device))
        model.eval()
        generation._seed_generation(12345)
        started = time.monotonic()
        print('GENERATING', args.method, seed, len(reference), flush=True)
        graphs, attempts = generation.generate_attributed_graphs(model, len(reference), device, cache, .5, 4, reference[0].edge_feature_dim)
        elapsed = time.monotonic()-started
        mode = 'decoded_node_edge' if reference[0].edge_feature_dim else 'decoded_node'
        exports = generation.save_dgl_graph_collections(out, [generation.to_dgl_graph(g,mode) for g in graphs], [generation.to_dgl_graph(g,mode) for g in reference])
        (out / 'generation.json').write_text(json.dumps({'checkpoint': str(checkpoint), 'count': len(graphs), 'attempts': attempts, 'seconds': elapsed, 'generation_seed':12345, 'exports':exports}, indent=2))
        print('GENERATION_DONE', args.method, seed, elapsed, flush=True)
        del model, graphs, reference, cache
        torch.cuda.empty_cache()
        evaluate(Path(exports['generated']), Path(exports['reference']), out / 'structural.json')
        (out / 'COMPLETE').write_text('Full-test structural evaluation complete\n')

if __name__ == '__main__':
    main()
