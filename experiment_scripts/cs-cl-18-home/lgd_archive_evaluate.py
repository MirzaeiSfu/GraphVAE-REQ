"""Apply the existing synthetic evaluation protocol to LGD samples."""
import sys
import os
import json
import pickle
import importlib.util
from pathlib import Path
import numpy as np
import networkx as nx
import torch

repo = Path('/local-scratch2/mirzaei/fb/GraphVAE-REQ')
os.chdir(repo)
sys.path.insert(0, str(repo))
source = '/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/evaluate_defog_synthetic_20260906.py'
spec = importlib.util.spec_from_file_location('existing_synthetic_eval', source)
existing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(existing)
from scripts.evaluate_graph_realism_batch import evaluate_graph_collections, item_to_graph
import stat_rnn as stats

out = Path(sys.argv[1])
with (out / 'samples.pkl').open('rb') as f:
    data = pickle.load(f)
def normalize(collection):
    return [nx.convert_node_labels_to_integers(item_to_graph(g)) for g in collection]
ref, gen = normalize(data['reference']), normalize(data['generated'])
if any(g.number_of_edges() == 0 for g in ref + gen):
    raise ValueError('Empty graph after benchmark normalization; inspect samples before scoring')
result = dict(dataset=data['dataset'], training_seed=2, checkpoint=data['checkpoint'],
              reference_graph_count=len(ref), generated_graph_count=len(gen),
              protocol='Largest connected component after removing loops and isolates; topology GIN 10 initializations, seed 0; sample SD across training seeds unavailable (n=1)',
              structural_mmd={}, errors={})
def save():
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
for key, fn in [('degree',stats.degree_stats),('clustering',stats.clustering_stats),
                ('orbit',stats.orbit_stats_all),('spectral',stats.spectral_stats),
                ('diameter',stats.MMD_diam),('triangle',stats.MMD_triangles)]:
    try: result['structural_mmd'][key] = float(fn(ref,gen))
    except Exception as e: result['errors'][key] = repr(e)
    save()
try:
    sparsity, r, g = stats.sparsity_stats_all(ref,gen)
    result['structural_mmd'].update(sparsity=float(sparsity),reference_edge_count=float(r),generated_edge_count=float(g),edge_count_absolute_error=float(abs(r-g)))
except Exception as e: result['errors']['sparsity'] = repr(e)
result['pruned_state_tv_test_reference'] = existing.state_distribution_comparison(ref,gen)
result['pruned_state_tv_train_reference'] = existing.state_distribution_comparison(normalize(data['train_reference']),gen)
result['train_reference_note'] = 'Same generated collection as test evaluation; counts summed then normalized. Graph collection sizes may differ. Not a matched-size historical replication.'
save()
try:
    result['topology_random_gin'] = evaluate_graph_collections(generated_graphs=gen,reference_graphs=ref,repeats=10,seed=0,device=torch.device('cpu'),use_structural_features=True)
except Exception as e: result['errors']['random_gin'] = repr(e)
save()
lines = [f"# {data['dataset']} LGD seed 2 evaluation", '', 'One training seed; no across-seed standard deviation. See metrics.json for evaluator repeat statistics and provenance.', '', '## Structural metrics', '', '| Metric | Value |', '| --- | --- |']
lines += [f'| {k} | {v:.8g} |' for k,v in result['structural_mmd'].items()]
lines += ['', '## Topology Random-GIN', '', '```json',json.dumps(result.get('topology_random_gin',{}),indent=2),'```', '', '## Pruned-state motif distribution', '', 'Rule: edges(nodes0,nodes1) AND edges(nodes1,nodes2). Retained states FF, TT, FT, TF. Sum counts over graphs, normalize once, TV = 0.5 sum |p_generated - p_reference|.', '', f"Test-reference TV: {result['pruned_state_tv_test_reference']['total_variation']:.8g}", '', f"Train-reference TV: {result['pruned_state_tv_train_reference']['total_variation']:.8g}", '', result['train_reference_note'], '', 'Errors: '+json.dumps(result['errors'])]
(out / 'LGD_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('EVALUATION COMPLETE',result['errors'],flush=True)
