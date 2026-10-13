#!/usr/bin/env python3
"""Fixed reachability and positive-endpoint multiplicity diagnostics."""
import argparse
import contextlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wasserstein_distance
import endpoint_metapath_mmd as endpoint

prior, source = endpoint.prior, endpoint.source
OUT = Path('/local-scratch2/mirzaei/metapath_reachability_w1_20260924')

def quantities(matrix, valid):
    n = matrix.shape[0]
    assert matrix.shape == (n,n) and valid.shape == (n,n)
    a = matrix[valid & ~np.eye(n,dtype=bool)]
    assert np.isfinite(a).all() and (a >= -1e-6).all()
    # Hard counts should be integers, with only floating-point roundoff allowed.
    assert np.allclose(a,np.rint(a),atol=1e-3,rtol=1e-6)
    a = np.maximum(np.rint(a),0)
    positive = a[a>0]
    return [len(positive)/max(n*(n-1),1), float(positive.mean()) if len(positive) else 0.]

def extract(records,counter,mapping,selected,expected,path):
    out=[]
    for i,raw in enumerate(records):
        rec=endpoint.hard_graph_postprocess(raw);n=int(rec['features'].shape[0]);p=len(expected[i])
        if n:
            wrapper=endpoint.TensorGraphBatch([rec],counter.relation_keys,mapping,torch.device('cpu'))
            with torch.no_grad(),contextlib.redirect_stdout(sys.stderr):
                matrices,mask=counter.count_batch(wrapper,batch_size=1,selected_rules_values=selected,output_mode='full_matrix')
            matrices=matrices[0].cpu().numpy();mask=mask.cpu().numpy().astype(bool)
            if mask.ndim==4:mask=mask[0]
            assert matrices.shape==(p,n,n)
            totals=np.array([matrices[j][mask[j]].sum(dtype=np.float64) for j in range(p)])
            assert np.allclose(np.log1p(totals),expected[i],rtol=2e-5,atol=2e-5),f'Count mismatch graph {i}'
            vals=[quantities(matrices[j],mask[j]) for j in range(p)]
        else:
            assert np.allclose(expected[i],0)
            vals=np.zeros((p,2))
        out.append(vals)
        if (i+1)%20==0:print(f'{path.name}: {i+1}/{len(records)}',flush=True)
    out=np.asarray(out);np.savez_compressed(path,reachability=out[:,:,0],multiplicity=out[:,:,1]);return out

def run(ds,smoke=False):
    dest=OUT/ds;dest.mkdir(parents=True,exist_ok=True);(dest/'status.txt').write_text('RUNNING\n')
    directory=prior.SOURCE/'datasets'/ds
    previous=json.loads((prior.OUT/ds/'results.json').read_text())
    assert prior.sha(directory/'selection.json')==previous['selection_sha256']
    assert prior.sha(directory/'inputs.json')==previous['input_sha256']
    selection=json.loads((directory/'selection.json').read_text());entries=previous['paths'];cols=[e['column'] for e in entries]
    selected={}
    for e in entries:selected.setdefault(e['rule_index'],[]).append(e['value_index'])
    counter,_,inputs=source.build_counter(directory,torch.device('cpu'),'unpruned')
    layout=source.full_state_layout(counter);assert all(layout[c]==selection['layout'][c] for c in cols)
    cache=source.load_cache(directory);collections={};provenance={}
    plan=[('train',None)]+[(f"{x['method']}_seed_{x['seed']}",x['path']) for x in inputs['items'] if x['method'] in prior.METHODS]
    for name,path in plan:
        expected,meta=prior.load_collection(directory/'counts',name,len(layout),cols)
        if path is None:records,mapping,_=source.reference_records(directory,'train',counter,torch.device('cpu'))
        else:
            if meta['sources'][0].get('source_sha256'):assert prior.sha(Path(path))==meta['sources'][0]['source_sha256']
            _,mapping,_=source.reference_records_mapping(cache,counter)
            records=source.generated_records(Path(path),counter.relation_keys[0],source.mapping_node_dim(cache),cache.get('edge_onehot_info'),counter.feature_info_mapping)
        assert len(records)==len(expected)
        if smoke:
            extract(records[:1],counter,mapping,selected,expected[:1],dest/'smoke.npz');(dest/'status.txt').write_text('SMOKE_PASS\n');return
        collections[name]=extract(records,counter,mapping,selected,expected,dest/(name+'.npz'));provenance[name]=meta
    ref=collections['train'];result=dict(dataset=ds,paths=entries,reference='motif=True training split; inherited cap 1000',
        definition='R=positive distinct ordered endpoint pairs / n(n-1); L=mean count among positive distinct endpoint pairs, or zero if none. W1 between per-graph R and L distributions, separately per path; macro-average across retained positive path states.',
        reachability_scope='Unconditional over all distinct endpoint pairs, not conditional on eligible endpoint types; includes type prevalence.',
        limitations='Custom diagnostic; not full-matrix equality. Preserve inherited cleanup, counter semantics and split/seed qualifications. No method selection by scores.',
        existing_qualifications=previous['existing_qualifications'],provenance=provenance,per_seed=[],aggregates={})
    for name,gen in collections.items():
        if name=='train':continue
        method,seed=name.rsplit('_seed_',1)
        distances=np.array([[wasserstein_distance(ref[:,j,k],gen[:,j,k]) for k in range(2)] for j in range(len(entries))])
        result['per_seed'].append(dict(method=method,seed=int(seed),reachability_w1=float(distances[:,0].mean()),multiplicity_w1=float(distances[:,1].mean()),
            per_path_w1=distances.tolist(),reference_zero_reach_fraction=(ref[:,:,0]==0).mean(0).tolist(),generated_zero_reach_fraction=(gen[:,:,0]==0).mean(0).tolist()))
    for method in prior.METHODS:
        rows=[r for r in result['per_seed'] if r['method']==method]
        a=dict(n=len(rows),seeds=[r['seed'] for r in rows])
        for metric in ['reachability_w1','multiplicity_w1']:
            v=[r[metric] for r in rows];a[metric]=dict(mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)) if len(v)>1 else None)
        result['aggregates'][method]=a
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[f'# {ds}: metapath reachability and multiplicity', '', 'Wasserstein-1 distance; lower is better. Mean ± sample SD over generator seeds. Scores are separate, not combined.', '',
        '| Method | Seeds | Reachability W1 | Multiplicity W1 |','|---|---|---:|---:|']
    for method,a in result['aggregates'].items():
        cells=[f"{a[k]['mean']:.8g} ± {a[k]['sample_sd']:.8g}" for k in ['reachability_w1','multiplicity_w1']]
        lines.append(f"| {method} | {a['seeds']} | {' | '.join(cells)} |")
    lines+=['','See results.json for per-path and per-seed results, zero-reach frequencies and provenance. Earlier metrics are not superseded.']
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n');(dest/'status.txt').write_text('COMPLETE\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('datasets',nargs='+');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    for ds in args.datasets:
        try:run(ds,args.smoke)
        except Exception as e:
            p=OUT/ds;p.mkdir(parents=True,exist_ok=True);(p/'status.txt').write_text('FAILED: '+repr(e)+'\n')
            import traceback;traceback.print_exc()
            if args.smoke:raise
