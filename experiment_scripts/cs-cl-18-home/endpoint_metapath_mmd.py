#!/usr/bin/env python3
"""Permutation-invariant distribution of full-matrix endpoint counts."""
import argparse
import contextlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
import metapath_instance_mmd as prior

OUT = Path('/local-scratch2/mirzaei/endpoint_metapath_mmd_20260924')
spec = importlib.util.spec_from_file_location('count_source', prior.SOURCE/'rule_mmd.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
from evaluate_motif_count_distance import TensorGraphBatch, hard_graph_postprocess

# Fixed bins, chosen independently of method scores: zero, 1, 2--3, 4--7, ...,
# 2^19--(2^20-1), >=2^20. Each rule histogram includes all valid endpoints.
BINS = np.array([-np.inf, .5] + [2.**i-.5 for i in range(1,21)] + [np.inf])

def descriptor(values):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all() and (a >= -1e-6).all()
    h = np.histogram(np.maximum(a,0), bins=BINS)[0].astype(float)
    return np.sqrt(h/max(1,h.sum()))

def extract(records, counter, mapping, selected, expected, output):
    rows=[]
    for i, raw in enumerate(records):
        record=hard_graph_postprocess(raw)
        n=int(record['features'].shape[0])
        count=len(expected[i])
        if n:
            wrapper=TensorGraphBatch([record],counter.relation_keys,mapping,torch.device('cpu'))
            with torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
                matrices, mask=counter.count_batch(wrapper,batch_size=1,selected_rules_values=selected,output_mode='full_matrix')
            matrices=matrices[0].cpu().numpy();mask=mask.cpu().numpy().astype(bool)
            if mask.ndim==4:mask=mask[0]
            arrays=[matrices[j][mask[j]] for j in range(count)]
            totals=np.array([a.sum(dtype=np.float64) for a in arrays])
            # Verify full-matrix counting matches the already audited scalar
            # counts on precisely the same graph/retained state columns.
            assert np.allclose(np.log1p(totals),expected[i],rtol=2e-5,atol=2e-5), (i,totals,np.expm1(expected[i]))
            hist=np.concatenate([descriptor(a) for a in arrays])/np.sqrt(count)
        else:
            assert np.allclose(expected[i],0)
            hist=np.zeros(count*(len(BINS)-1))
        rows.append(np.r_[hist, float(n==0)])
        if (i+1)%10==0: print(f'{output.name}: {i+1}/{len(records)} graphs',flush=True)
    features=np.asarray(rows)
    np.savez_compressed(output,features=features)
    return features

def run(ds,smoke=False):
    dest=OUT/ds;dest.mkdir(parents=True,exist_ok=True)
    (dest/'status.txt').write_text('RUNNING\n')
    directory=prior.SOURCE/'datasets'/ds
    saved=json.loads((prior.OUT/ds/'results.json').read_text())
    selection=json.loads((directory/'selection.json').read_text())
    assert prior.sha(directory/'selection.json')==saved['selection_sha256']
    assert prior.sha(directory/'inputs.json')==saved['input_sha256']
    entries=saved['paths']; columns=[e['column'] for e in entries]
    selected={}
    for e in entries:selected.setdefault(e['rule_index'],[]).append(e['value_index'])
    counter,flat,inputs=source.build_counter(directory,torch.device('cpu'),'unpruned')
    actual=source.full_state_layout(counter)
    assert all(actual[c]==selection['layout'][c] for c in columns)
    cache=source.load_cache(directory)
    collections={}; provenance={}
    plan=[('train',None)]+[(f"{x['method']}_seed_{x['seed']}",x['path']) for x in inputs['items'] if x['method'] in prior.METHODS]
    for name,path in plan:
        expected,meta=prior.load_collection(directory/'counts',name,len(selection['layout']),columns)
        if path is None:
            records,mapping,extra=source.reference_records(directory,'train',counter,torch.device('cpu'))
        else:
            if meta['sources'][0].get('source_sha256'):
                assert prior.sha(Path(path))==meta['sources'][0]['source_sha256']
            _,mapping,_=source.reference_records_mapping(cache,counter)
            records=source.generated_records(Path(path),counter.relation_keys[0],source.mapping_node_dim(cache),cache.get('edge_onehot_info'),counter.feature_info_mapping)
        assert len(records)==len(expected)
        output=dest/(name+'.npz')
        if smoke:
            extract(records[:1],counter,mapping,selected,expected[:1],dest/'smoke.npz')
            (dest/'status.txt').write_text('SMOKE_PASS\n');return
        collections[name]=extract(records,counter,mapping,selected,expected,output)
        provenance[name]=meta
    ref=collections['train'];d=cdist(ref,ref,'sqeuclidean');v=d[np.triu_indices(len(ref),1)]
    sigma2=float(np.median(v)) if len(v) else 0.
    if sigma2<=0:sigma2=float(np.median(v[v>0])) if (v>0).any() else 1.
    results=dict(dataset=ds,metric='endpoint-metapath-histogram-MMD2',
        definition='Per retained positive path: histogram of full-matrix valid endpoint counts including zeros; square-root frequencies, concatenate / sqrt(path_count), append empty-graph indicator; reference-calibrated RBF biased MMD2.',
        limitations='Permutation-invariant summary, not full matrix equivalence; discards endpoint identity and cross-path endpoint associations. Training reference; inherited graph cleanup, split and selection qualifications apply.',
        bins=['-inf']+BINS[1:-1].tolist()+['inf'],paths=entries,sigma2=sigma2,
        existing_qualifications=saved['existing_qualifications'],provenance=provenance,per_seed=[],aggregates={})
    for name,x in collections.items():
        if name=='train':continue
        method,seed=name.rsplit('_seed_',1)
        results['per_seed'].append(dict(method=method,seed=int(seed),**prior.score(ref,x,sigma2)))
    for method in prior.METHODS:
        rows=[r for r in results['per_seed'] if r['method']==method];vals=[r['mmd2_biased'] for r in rows]
        results['aggregates'][method]=dict(n=len(vals),seeds=[r['seed'] for r in rows],mean=float(np.mean(vals)),sample_sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None)
    (dest/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=[f'# {ds}: endpoint metapath MMD', '', 'Lower is better; mean ± sample SD across generator seeds.', '', '| Method | Seeds | Mean | SD |','|---|---|---:|---:|']
    for m,a in results['aggregates'].items():lines.append(f"| {m} | {a['seeds']} | {a['mean']:.8g} | {a['sample_sd']} |")
    lines+=['','See results.json for per-seed numbers, retained paths and protocol qualifications. Earlier total-instance MMD remains a separate result.']
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (dest/'status.txt').write_text('COMPLETE\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('datasets',nargs='+');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    for ds in args.datasets:
        try:run(ds,args.smoke)
        except Exception as e:
            target=OUT/ds;target.mkdir(parents=True,exist_ok=True);(target/'status.txt').write_text('FAILED: '+repr(e)+'\n')
            import traceback;traceback.print_exc()
            if args.smoke:raise
