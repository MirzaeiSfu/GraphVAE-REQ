#!/usr/bin/env python3
"""Independent sparse-count/score audit; does not change evaluation outputs."""
import json
from pathlib import Path
import hashlib
import numpy as np
from scipy import sparse

BASE=Path('/local-scratch2/mirzaei/all_rule_distributions_20260924')
DATA=Path('/local-scratch2/mirzaei/rule_mmd_20260923/datasets')
OUT=BASE/'audit'
METHODS=['motif_false','motif_true_full','defog']

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def distance(a,b):
    if a.sum()==0 or b.sum()==0:
        v=float((a.sum()==0)!=(b.sum()==0));return v,v
    p=a/a.sum();q=b/b.sum()
    # Independent overlap / Bhattacharyya implementations.
    return max(0.,1-float(np.minimum(p,q).sum())),np.sqrt(max(0.,1-float(np.sqrt(p*q).sum())))

def run(ds):
    print('AUDIT',ds,flush=True)
    r=json.loads((BASE/ds/'results.json').read_text())
    sel=json.loads((DATA/ds/'selection.json').read_text())
    inputs=json.loads((DATA/ds/'inputs.json').read_text())
    assert not r['missing']
    assert digest(DATA/ds/'inputs.json')==r['input_sha256']
    assert digest(DATA/ds/'selection.json')==r['selection_sha256']
    layout=sel['layout']; width=len(layout)
    ids=np.array([e['rule_index'] for e in layout])
    groups={int(k):np.flatnonzero(ids==int(k)) for k in r['rule_templates']}
    for k,idx in groups.items():
        keys=[tuple(layout[c]['state']) for c in idx]
        assert len(set(keys))==len(keys), 'Duplicate state in rule'
        assert all(layout[c]['rule']==layout[idx[0]]['rule'] for c in idx)
    items={f"{x['method']}_seed_{x['seed']}":x for x in inputs['items']}
    plans=[('train',r['reference_provenance'])]+[(f"{x['method']}_seed_{x['seed']}",x['provenance']) for x in r['per_seed']]
    a={};stats={};source_hashes={}
    for name,provenance in plans:
        aggregate=np.zeros(width);positions=[];empty=0;nodes=[];fractional_max=0.;shards=[]
        for src in provenance['sources']:
            assert digest(src['path'])==src['sha256']
            with np.load(src['path'],allow_pickle=False) as d:
                m=json.loads(str(d['meta']))
                mat=sparse.csr_matrix(d['counts']) if 'counts' in d else sparse.csr_matrix((d['data'],d['indices'],d['indptr']),shape=tuple(d['shape']))
                mat.check_format(full_check=True)
                assert mat.shape[1]==width and np.isfinite(mat.data).all() and np.all(mat.data>=0)
                assert m['collection']==name
                # Independent reduction by state index, not CSR sum(axis=0).
                aggregate+=np.bincount(mat.indices,weights=mat.data.astype(float),minlength=width)
                if mat.nnz: fractional_max=max(fractional_max,float(np.abs(mat.data-np.rint(mat.data)).max()))
                positions.extend(m.get('positions',range(mat.shape[0])))
                nodes.extend(d['node_counts'].tolist());empty+=int((d['node_counts']==0).sum())
                shards.append(m.get('shard',[0,1]))
                if name!='train':
                    if name not in source_hashes:source_hashes[name]=digest(items[name]['path'])
                    assert source_hashes[name]==m['source_sha256']
        assert sorted(positions)==list(range(provenance['graph_count']))
        assert empty==provenance['empty_after_cleanup']
        if len(shards)>1:
            assert len({s[1] for s in shards})==1 and sorted(s[0] for s in shards)==list(range(shards[0][1]))
        with np.load(BASE/ds/(name+'_aggregate_counts.npz')) as d:
            assert np.allclose(aggregate,d['counts'],rtol=1e-12,atol=1e-9)
        a[name]=aggregate
        stats[name]=dict(graph_count=len(nodes),empty=empty,mean_nodes=float(np.mean(nodes)),max_fractional_count_error=fractional_max)
    for m in METHODS:
        hashes=[h for name,h in source_hashes.items() if name.startswith(m+'_seed_')]
        assert len(hashes)==3 and len(set(hashes))==3, 'Duplicate source bytes or wrong seed count'
    maxerr=0.;computed=[]
    kept=np.array(sel['pruned_columns']);keepmask=np.zeros(width,dtype=bool);keepmask[kept]=True
    for row in r['per_seed']:
        name=f"{row['method']}_seed_{row['seed']}"
        per=[];pruned=[]
        for k,idx in groups.items():
            tv,h=distance(a['train'][idx],a[name][idx]);per.append((tv,h))
            saved=row['per_rule'][str(k)]
            maxerr=max(maxerr,abs(saved['tv']-tv),abs(saved['hellinger']-h))
            j=idx[keepmask[idx]]
            if len(j):pruned.append(distance(a['train'][j],a[name][j]))
        score=np.mean(per,axis=0)
        assert np.allclose(score,[row['tv'],row['hellinger']],rtol=1e-8,atol=3e-8)
        computed.append(dict(method=row['method'],seed=row['seed'],tv=float(score[0]),hellinger=float(score[1]),
            pruned_tv=float(np.mean(pruned,axis=0)[0])))
    assert maxerr<3e-8,maxerr
    for method in METHODS:
        rows=[x for x in computed if x['method']==method]
        assert len(rows)==3 and len({x['seed'] for x in rows})==3
        for metric in ('tv','hellinger'):
            vals=[x[metric] for x in rows];saved=r['aggregates'][method][metric]
            assert np.isclose(np.mean(vals),saved['mean'],atol=3e-8,rtol=1e-8)
            assert np.isclose(np.std(vals,ddof=1),saved['sample_sd'],atol=3e-8,rtol=1e-8)
    diag=[]
    for k,idx in groups.items():
        rule=r['rule_templates'][str(k)]['rule']
        means={m:float(np.mean([x['per_rule'][str(k)]['tv'] for x in r['per_seed'] if x['method']==m])) for m in METHODS}
        mass=a['train'][idx].sum()
        row=dict(rule_index=k,rule=rule,states=len(idx),retained_states=int(keepmask[idx].sum()),
            reference_mass=float(mass),reference_retained_mass_fraction=float(a['train'][idx[keepmask[idx]]].sum()/mass) if mass else None,
            means=means,delta_true_minus_false=means['motif_true_full']-means['motif_false'])
        diag.append(row)
    audit=dict(dataset=ds,status='PASS',max_score_disagreement=maxerr,full_states=width,retained_states=len(kept),
        collections=stats,source_hashes=source_hashes,per_rule=diag,per_seed=computed,
        limitations=inputs.get('open_issues',[]),note='Checks count-to-score pipeline, not a proof of the original counter or training provenance. Source byte uniqueness does not prove independent training seeds.')
    (OUT/(ds+'.json')).write_text(json.dumps(audit,indent=2)+'\n')
    print('PASS',ds,'max disagreement',maxerr,flush=True)
    return audit

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    all_results=[run(ds) for ds in ['GRID','TRIANGULAR_GRID','LOBSTER','PTC','PROTEINS','QM9','AIDS']]
    lines=['# Independent all-rule distribution audit','','Read-only audit; original scores unchanged.','','| Dataset | Status | Full states | Retained states | Max numerical discrepancy |','|---|---|---:|---:|---:|']
    for x in all_results:lines.append(f"| {x['dataset']} | {x['status']} | {x['full_states']} | {x['retained_states']} | {x['max_score_disagreement']:.3g} |")
    lines+=['','Checks: sparse count hashes/shard coverage, source graph hashes, unique states, finite nonnegative counts, independent state sums, independent TV/Hellinger formulas, per-seed means and sample SD. Full protocol fairness and counter semantics require additional checks.','', '## QM9 and AIDS rule breakdown']
    for x in all_results[-2:]:
        lines += ['', '### '+x['dataset'],'','| Rule | States | Retained | False TV | True TV | True − False |','|---|---:|---:|---:|---:|---:|']
        for q in x['per_rule']:
            lines.append(f"| {' AND '.join(q['rule'])} | {q['states']} | {q['retained_states']} | {q['means']['motif_false']:.7f} | {q['means']['motif_true_full']:.7f} | {q['delta_true_minus_false']:.7f} |")
    (OUT/'NUMERICAL_AUDIT.md').write_text('\n'.join(lines)+'\n')
