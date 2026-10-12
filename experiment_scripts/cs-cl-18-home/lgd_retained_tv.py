"""Use existing training-rule selection for attributed LGD evaluations."""
import json,pickle,sys,importlib.util
from pathlib import Path
import numpy as np
import torch
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def summarize(observed,generated,entries):
    groups={}
    for i,e in enumerate(entries):
        if len(e['rule'])>1:groups.setdefault(e['rule_index'],[]).append(i)
    rows=[]
    for ri,indices in groups.items():
        a=observed[:,indices].sum(0);b=generated[:,indices].sum(0)
        tv=None if a.sum()<=0 else 1.0 if b.sum()<=0 else float(.5*np.abs(a/a.sum()-b/b.sum()).sum())
        rows.append(dict(rule=entries[indices[0]]['rule'],retained_state_count=len(indices),tv=tv,reference_counts=a.tolist(),generated_counts=b.tolist()))
    valid=[x['tv'] for x in rows if x['tv'] is not None]
    return dict(mean_rule_tv=float(np.mean(valid)) if valid else None,rules=rows)
def evaluate(dataset,ref,gen,train):
    if dataset in ('LOBSTER','TRIANGULAR_GRID','GRID'):
        import dgl,networkx as nx
        m=module('syntv','/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/evaluate_defog_synthetic_20260906.py')
        from scripts.evaluate_graph_realism_batch import item_to_graph
        conv=lambda gs:[item_to_graph(nx.Graph(dgl.to_networkx(g))) for g in gs]
        return dict(test=m.state_distribution_comparison(conv(ref),conv(gen)),train=m.state_distribution_comparison(conv(train),conv(gen)))
    if dataset=='PTC':
        m=module('ptccount','/local-scratch/localhome/mirzaei/evaluate_defog_ptc_rule_metrics_20260907.py')
        template=Path('/local-scratch2/mirzaei/ptc_rule_metrics_20260906/results/full_matrix_seed0.json')
        entries=json.loads(template.read_text())['motif_entries']
        def count(gs):
            items=[dict(num_nodes=g.num_nodes(),x=g.ndata['attr'],edge_index=torch.stack(g.edges())) for g in gs]
            return m.count_entries(items,entries)[0]
        gc=count(gen)
        return dict(template=str(template),retained_states=len(entries),test=summarize(count(ref),gc,entries),train=summarize(count(train),gc,entries))
    from data import DataWrapper,merge_datasets
    from motif_counting.motif_counter import RelationalMotifCounter
    import evaluate_motif_count_distance as ev
    if dataset=='MUTAG':
        root=Path('/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/GraphVAE-REQ')
        config=Path('/local-scratch2/mirzaei/mutag_edgefeat_motif02_3seed_20260911/runs/full_matrix/seed_0/run_config_used.yaml')
        cache=next(Path('/local-scratch2/mirzaei/mutag_threshold_calibration_20260912/dataset_cache').glob('MUTAG*.pkl'))
        motif=root/'cache_motifs/mutag_edgefeat_campaign_20260910'
    elif dataset=='AIDS':
        root=Path('/local-scratch2/mirzaei/aids_common_eval_10k_20260917')
        config=root/'graphvae/true_full/seed_0/run_config_used.yaml'
        cache=next((root/'cache/dataset').glob('AIDS*.pkl'));motif=root/'cache/motif'
    else:raise ValueError('No verified training-rule selection for '+dataset)
    _,flat=ev.load_yaml(config)
    assert flat.get('rule_prune'), 'Expected pruned training configuration'
    device=torch.device('cpu');args=ev.make_counter_args(flat,motif,device)
    counter=RelationalMotifCounter(str(flat['database_name']),args)
    ev.ensure_global_motif_value_cap(counter,args)
    with cache.open('rb') as f:cached=pickle.load(f)
    wrapper=DataWrapper(merge_datasets(cached['list_graphs']),counter.relation_keys,cached.get('node_onehot_info'),edge_onehot_info=cached.get('edge_onehot_info'),edge_feature_info_mapping=counter.feature_info_mapping,device='cpu')
    converter=module('dglrecords','/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_qm9_motif_tv_dgl.py')
    def count(gs):
        records=[ev.hard_graph_postprocess(x) for x in converter.graph_records(gs,counter.relation_keys[0])]
        return ev.count_exact_graph_records(counter,records,wrapper.feature_onehot_mapping,8,device).numpy()
    entries=ev.motif_entry_metadata(counter);gc=count(gen)
    return dict(config=str(config),database=flat['database_name'],retained_states=len(entries),test=summarize(count(ref),gc,entries),train=summarize(count(train),gc,entries))
