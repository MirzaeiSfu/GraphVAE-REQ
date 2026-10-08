#!/usr/bin/env python3
"""Fresh graph recount plus independent Einstein-sum rule-grounding checks."""
import json
import re
import gc
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
import endpoint_metapath_mmd as ep
from evaluate_motif_count_distance import count_exact_graph_records

ROOT=Path('/local-scratch2/mirzaei/all_rule_distributions_20260924')
OUT=ROOT/'audit'

def brute(record,entry,counter,mapping):
    """Enumerate variable assignments by independent NumPy tensor contraction.

    Same non-injective grounding convention as the training counter. Negative
    edges are 1-A including diagonal. Edge N/A follows counter's any-edge-label
    convention; this does not validate that convention against database SQL.
    """
    names=[];terms=[];operands=[]
    for atom,state in zip(entry['rule'],entry['state']):
        name,argstr=re.fullmatch(r'([^()]+)\(([^()]*)\)',atom).groups()
        args=argstr.split(',')
        for v in args:
            if v not in names:names.append(v)
        terms.append(''.join(chr(97+names.index(v)) for v in args))
        if name in record['adj']:
            a=record['adj'][name].numpy().astype(float)
            arr=1-a if state=='F' else a
        elif len(args)==1:
            found,idx,_=counter._find_feature(name)
            assert found,(name,'unknown node predicate')
            col=mapping.get(idx,{}).get(int(state))
            arr=record['feat_onehot'][:,col].numpy().astype(float) if col is not None else np.zeros(record['feat_onehot'].shape[0])
        else:
            idx=next(k for k,v in counter.feature_info_mapping.items() if v['feature_name']==name)
            a=record['edge'][idx].numpy().astype(float)
            if state=='N/A':arr=a.sum(axis=0)
            else:
                reverse={v:k for k,v in counter.feature_info_mapping[idx]['value_index_mapping'].items()}
                arr=a[reverse[int(state)]]
        operands.append(arr)
    return float(np.einsum(','.join(terms)+'->',*operands,optimize='greedy'))

def cached_samples(directory,name,columns,positions):
    direct=directory/'counts'/(name+'.npz')
    paths=[direct] if direct.exists() else sorted((directory/'counts').glob(name+'.shard*of*.npz'))
    rows={}
    for path in paths:
        with np.load(path,allow_pickle=False) as d:
            meta=json.loads(str(d['meta']))
            a=sparse.csr_matrix(d['counts']) if 'counts' in d else sparse.csr_matrix((d['data'],d['indices'],d['indptr']),shape=tuple(d['shape']))
            for i,p in enumerate(meta.get('positions',range(a.shape[0]))):
                if p in positions:rows[p]=a[i,columns].toarray().ravel()
    return np.stack([rows[p] for p in positions])

def run(ds):
    print('FRESH_RECOUNT',ds,flush=True)
    directory=ep.prior.SOURCE/'datasets'/ds
    sel=json.loads((directory/'selection.json').read_text());layout=sel['layout']
    ids=np.array([e['rule_index'] for e in layout]);rng=np.random.default_rng(20260924)
    refs=np.load(ROOT/ds/'train_aggregate_counts.npz')['counts']
    kept=set(sel['pruned_columns']);cols=set()
    # Fixed, score-independent coverage: every small rule, random states,
    # high-reference-mass states, and retained states for every large rule.
    for k in np.unique(ids):
        idx=np.flatnonzero(ids==k)
        if len(idx)<=10:cols.update(idx.tolist())
        else:
            cols.update(rng.choice(idx,size=3,replace=False).tolist())
            cols.update(idx[np.argsort(refs[idx])[-3:]].tolist())
            retained=[i for i in idx if i in kept]
            cols.update(retained[:2])
    columns=sorted(int(c) for c in cols);selected={}
    for c in columns:
        e=layout[c];selected.setdefault(e['rule_index'],[]).append(e['value_index'])
    counter,_,inputs=ep.source.build_counter(directory,torch.device('cpu'),'unpruned')
    for c in columns:
        e=layout[c]
        assert list(ep.source.state_key(counter,e['rule_index'],counter.values[e['rule_index']][e['value_index']]))==e['state']
        assert list(counter.rules[e['rule_index']])==e['rule']
    cache=ep.source.load_cache(directory)
    _,mapping,_=ep.source.reference_records_mapping(cache,counter)
    plan=[('train',None)]+[(f"{x['method']}_seed_{x['seed']}",x['path']) for x in inputs['items'] if x['method'] in ep.prior.METHODS]
    results=[]
    for name,path in plan:
        if path is None:records,mapping,_=ep.source.reference_records(directory,'train',counter,torch.device('cpu'))
        else:records=ep.source.generated_records(Path(path),counter.relation_keys[0],ep.source.mapping_node_dim(cache),cache.get('edge_onehot_info'),counter.feature_info_mapping)
        positions=sorted(set([0,len(records)//2]))
        hard=[ep.hard_graph_postprocess(records[p]) for p in positions]
        expected=cached_samples(directory,name,columns,positions)
        fresh=count_exact_graph_records(counter,hard,mapping,1,torch.device('cpu'),selected).numpy()
        independent=np.array([[brute(rec,layout[c],counter,mapping) if len(rec['features']) else 0. for c in columns] for rec in hard])
        assert np.allclose(fresh,expected,rtol=1e-6,atol=1e-6),(ds,name,'fresh vs cache',float(np.max(np.abs(fresh-expected))))
        assert np.allclose(independent,expected,rtol=1e-6,atol=1e-6),(ds,name,'independent vs cache',float(np.max(np.abs(independent-expected))))
        results.append(dict(collection=name,positions=positions,states=len(columns),cached_vs_fresh_max=float(np.max(np.abs(fresh-expected))),
            cached_vs_independent_max=float(np.max(np.abs(independent-expected)))))
        print('PASS',ds,name,'graphs',positions,'states',len(columns),flush=True)
        del records,hard
    (OUT/(ds+'_fresh_counts.json')).write_text(json.dumps(dict(dataset=ds,status='PASS',selected_columns=columns,collections=results,
        caveat='Spot-check of saved hard graphs and training-counter grounding semantics, not exhaustive count validation or database SQL validation.'),indent=2)+'\n')
    del counter,cache,sel,layout
    gc.collect()

if __name__=='__main__':
    import sys,traceback
    torch.set_num_threads(2)
    for ds in sys.argv[1:]:
        try:run(ds)
        except Exception as e:
            traceback.print_exc()
            (OUT/(ds+'_fresh_counts.json')).write_text(json.dumps(dict(dataset=ds,status='FAILED',error=repr(e)))+'\n')
