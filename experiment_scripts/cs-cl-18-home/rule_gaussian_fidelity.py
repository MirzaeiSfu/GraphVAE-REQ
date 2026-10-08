#!/usr/bin/env python3
"""Diagonal-Gaussian Wasserstein fidelity of retained-rule endpoint histograms.

All retained states, not only positive paths. Not the training NLL, and not
an overall-quality certificate. Reference-only scaling is fixed across methods.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import endpoint_metapath_mmd as ep

OUT=Path('/local-scratch2/mirzaei/rule_gaussian_fidelity_20260924')
FLOOR=1e-6

def gaussian_distance(ref,gen):
    # Squared W2 between diagonal Gaussian moment approximations after
    # reference-only variance scaling. Population variance (ddof=0).
    mr,mg=ref.mean(0),gen.mean(0)
    vr,vg=ref.var(0),gen.var(0)
    location=(mr-mg)**2/(vr+FLOOR)
    spread=(np.sqrt(vr)-np.sqrt(vg))**2/(vr+FLOOR)
    return dict(score=float(np.mean(location+spread)),mean_component=float(location.mean()),spread_component=float(spread.mean()))

def run(ds,smoke=False):
    out=OUT/ds;out.mkdir(parents=True,exist_ok=True);(out/'status.txt').write_text('RUNNING\n')
    directory=ep.prior.SOURCE/'datasets'/ds
    selection=json.loads((directory/'selection.json').read_text())
    assert selection['selections_identical'],'Different retained selections require separate scoring'
    columns=sorted(selection['pruned_columns']);entries=[selection['layout'][c] for c in columns]
    selected={}
    for e in entries:selected.setdefault(e['rule_index'],[]).append(e['value_index'])
    counter,_,inputs=ep.source.build_counter(directory,torch.device('cpu'),'unpruned')
    layout=ep.source.full_state_layout(counter)
    assert all(layout[c]==selection['layout'][c] for c in columns)
    cache=ep.source.load_cache(directory);collections={};provenance={}
    plan=[('train',None)]+[(f"{x['method']}_seed_{x['seed']}",x['path']) for x in inputs['items'] if x['method'] in ep.prior.METHODS]
    width=len(ep.BINS)-1
    for name,path in plan:
        expected,meta=ep.prior.load_collection(directory/'counts',name,len(layout),columns)
        if path is None:records,mapping,_=ep.source.reference_records(directory,'train',counter,torch.device('cpu'))
        else:
            if meta['sources'][0].get('source_sha256'):assert ep.prior.sha(Path(path))==meta['sources'][0]['source_sha256']
            _,mapping,_=ep.source.reference_records_mapping(cache,counter)
            records=ep.source.generated_records(Path(path),counter.relation_keys[0],ep.source.mapping_node_dim(cache),cache.get('edge_onehot_info'),counter.feature_info_mapping)
        assert len(records)==len(expected)
        if smoke:
            ep.extract(records[:1],counter,mapping,selected,expected[:1],out/'smoke.npz');(out/'status.txt').write_text('SMOKE_PASS\n');return
        features=ep.extract(records,counter,mapping,selected,expected,out/(name+'.npz'))
        # Undo concatenation normalization so each state has the same fixed
        # [0,1] square-root histogram coordinate scale for the variance floor.
        features[:,:-1]*=np.sqrt(len(entries))
        collections[name]=features;provenance[name]=meta
    groups={}
    for i,e in enumerate(entries):groups.setdefault(e['rule_index'],[]).extend(range(i*width,(i+1)*width))
    ref=collections['train'];result=dict(dataset=ds,metric='reference-scaled diagonal Gaussian W2 on retained-rule endpoint histograms',
        definition='Fit per-coordinate mean and variance to square-root endpoint-count histograms. Score ((mean_gen-mean_ref)^2+(sd_gen-sd_ref)^2)/(var_ref+1e-6), averaged within each template then equally across templates. Empty-graph indicator discrepancy reported separately.',
        interpretation='Custom rule-fidelity diagnostic, not the actual Gaussian training loss, not an overall graph-quality metric. Gaussian approximation ignores correlations and higher moments.',
        retained_states=len(entries),rule_templates=len(groups),entries=entries,variance_floor=FLOOR,
        reference='motif=True training split; existing cap 1000',provenance=provenance,
        selection_sha256=ep.prior.sha(directory/'selection.json'),input_sha256=ep.prior.sha(directory/'inputs.json'),
        existing_qualifications=inputs.get('open_issues',[]),per_seed=[],aggregates={})
    for name,gen in collections.items():
        if name=='train':continue
        method,seed=name.rsplit('_seed_',1)
        per_rule={str(r):gaussian_distance(ref[:,idx],gen[:,idx]) for r,idx in groups.items()}
        result['per_seed'].append(dict(method=method,seed=int(seed),score=float(np.mean([r['score'] for r in per_rule.values()])),
            per_rule=per_rule,empty_graph_fraction_reference=float(ref[:,-1].mean()),empty_graph_fraction_generated=float(gen[:,-1].mean()),
            empty_indicator_gaussian=gaussian_distance(ref[:,-1:],gen[:,-1:])))
    for method in ep.prior.METHODS:
        rows=[r for r in result['per_seed'] if r['method']==method];vals=[r['score'] for r in rows]
        result['aggregates'][method]=dict(n=len(vals),seeds=[r['seed'] for r in rows],mean=float(np.mean(vals)),sample_sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[f'# {ds}: all-retained-rule Gaussian fidelity','', 'Lower is better. Custom diagnostic; not the training loss. Mean ± sample SD across generator seeds.','', '| Method | Seeds | Score |','|---|---|---:|']
    for m,a in result['aggregates'].items():lines.append(f"| {m} | {a['seeds']} | {a['mean']:.8g} ± {a['sample_sd']:.8g} |")
    lines+=['','Per-rule contributions, empty-graph rates and provenance are in results.json. All earlier metrics remain reportable; do not select a metric based on its winner.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');(out/'status.txt').write_text('COMPLETE\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('datasets',nargs='+');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    for ds in args.datasets:
        try:run(ds,args.smoke)
        except Exception as e:
            d=OUT/ds;d.mkdir(parents=True,exist_ok=True);(d/'status.txt').write_text('FAILED: '+repr(e)+'\n')
            import traceback;traceback.print_exc()
            if args.smoke:raise
