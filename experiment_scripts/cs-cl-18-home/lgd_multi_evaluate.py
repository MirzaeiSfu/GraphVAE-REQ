"""Evaluate LGD with existing GraphVAE structural/GNN and retained-rule counters."""
import sys,os,json,pickle,importlib.util
from pathlib import Path
import numpy as np
import networkx as nx
import torch,dgl
out=Path(sys.argv[1]); repo=Path('/local-scratch2/mirzaei/fb/GraphVAE-REQ')
sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'scripts'));os.chdir(repo)
with (out/'samples.pkl').open('rb') as f: data=pickle.load(f)
dataset=data['dataset'];meta=data['metadata']
node_dims=meta['node_feature_dims'];edge_dims=[x-1 for x in meta['edge_feature_dims'][1:]]
def collection(records):
    graphs=[]
    for r in records:
        n=r['num_nodes'];edges=r['edge_rows'];src=[];dst=[];ef=[]
        for u,v,features in edges:
            for a,b in [(u,v),(v,u)]: src.append(a);dst.append(b);ef.append(features)
        g=dgl.graph((src,dst),num_nodes=n)
        cols=torch.as_tensor(r['node_categories'],dtype=torch.long)
        g.ndata['attr']=torch.cat([torch.nn.functional.one_hot(cols[:,i],num_classes=w) for i,w in enumerate(node_dims)],dim=1).float()
        if edge_dims:
            cols=torch.as_tensor(ef,dtype=torch.long).reshape(len(ef),len(edge_dims))
            g.edata['attr']=torch.cat([torch.nn.functional.one_hot(cols[:,i],num_classes=w) for i,w in enumerate(edge_dims)],dim=1).float()
        graphs.append(g)
    return graphs
ref=collection(data['reference_records']);gen=collection(data['generated_records']);train=collection(data['train_records'])
for label,graphs in [('reference',ref),('generated',gen),('train',train)]: dgl.save_graphs(str(out/f'{label}.bin'),graphs)
from scripts.evaluate_graph_realism_batch import evaluate_graph_collections,item_to_graph
from eval.attributed_gin import evaluate_dgl_feature_modes
import stat_rnn as stats
def topology(gs):return [nx.convert_node_labels_to_integers(item_to_graph(nx.Graph(dgl.to_networkx(g)))) for g in gs]
r,g=topology(ref),topology(gen)
result=dict(dataset=dataset,seed=data['seed'],checkpoint=data['checkpoint'],metadata=meta,graph_count=len(gen),structural={},errors={})
def save(): (out/'metrics.json').write_text(json.dumps(result,indent=2))
for key,fn in [('degree',stats.degree_stats),('clustering',stats.clustering_stats),('orbit',stats.orbit_stats_all),('spectral',stats.spectral_stats),('diameter',stats.MMD_diam),('triangle',stats.MMD_triangles)]:
    try: result['structural'][key]=float(fn(r,g))
    except Exception as e:result['errors'][key]=repr(e)
    save()
try:
    sp,a,b=stats.sparsity_stats_all(r,g);result['structural'].update(sparsity=float(sp),reference_edge_count=float(a),generated_edge_count=float(b),edge_count_absolute_error=float(abs(a-b)))
except Exception as e:result['errors']['sparsity']=repr(e)
try:result['topology_random_gin']=evaluate_graph_collections(generated_graphs=g,reference_graphs=r,repeats=10,seed=0,device=torch.device('cpu'),use_structural_features=True)
except Exception as e:result['errors']['topology_random_gin']=repr(e)
save()
if not meta['topology_only']:
    try:result['feature_random_gin']=evaluate_dgl_feature_modes(gen,ref,modes=('decoded_node','decoded_node_edge') if edge_dims else ('decoded_node',),repeats=10,seed=0,device=torch.device('cpu'))
    except Exception as e:result['errors']['feature_random_gin']=repr(e)
save()
try:
    import lgd_retained_tv
    result['motif_tv']=lgd_retained_tv.evaluate(dataset,ref,gen,train)
except Exception as e:
    import traceback;traceback.print_exc();result['errors']['motif_tv']=repr(e)
save()
lines=[f'# {dataset} LGD seed {data["seed"]}', '', f'Checkpoint: `{data["checkpoint"]}`', '', 'Structural metrics, topology Random-GIN (10 initializations, evaluator seed 0), native-feature GIN where available, and training-rule TV. Generator seed and evaluator initializations are distinct. No standard deviation across generator seeds is inferred from this one run.', '', '```json',json.dumps(result,indent=2),'```']
(out/'LGD_RESULTS.md').write_text('\n'.join(lines)+'\n')
(out/('COMPLETE' if not result['errors'] else 'PARTIAL')).write_text(json.dumps(result['errors']))
