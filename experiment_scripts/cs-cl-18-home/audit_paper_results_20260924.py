#!/usr/bin/env python3
"""Read-only verification of manuscript cells against original metric JSONs."""
import json,hashlib,subprocess,statistics,re
from pathlib import Path
from decimal import Decimal

ROOT=Path('/local-scratch/localhome/mirzaei')
DATA=Path('/local-scratch2/mirzaei')
OUT=DATA/'PAPER_RESULTS_AUDIT_20260924'
OUT.mkdir(exist_ok=True)
sources={}; observations=[]; raw={}

def read(p):
    p=Path(p);b=p.read_bytes();sources[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)

ev=read(ROOT/'paper_revision_20260922/evidence.json')
paper=(ROOT/'final.md').read_text()
tables=(ROOT/'paper_revision_20260922/tables.md').read_text().strip().split('\n\n')
assert all(t in paper for t in tables)
hashchecks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in ev['sources_sha256'].items()}
assert all(hashchecks.values())
cells={(r['dataset'],r['metric'],r['method']):r['value'] for r in ev['cells']}

# Read the archived synthetic per-seed files on system19 without changing them.
remote=r'''
import json,hashlib
from pathlib import Path
base=Path('/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921')
out=[]
for ds in ['GRID','TRIANGULAR_GRID','LOBSTER']:
 for method,sub in [('false','motif_false'),('true','motif_true_full_matrix')]:
  for seed in range(3):
   root=base/ds/'experiments/graphvae'/sub/f'seed_{seed}'
   files={}
   for name in ['final_table2_metrics.json','graph_realism_random_gin.json','real_test_graphs.pt.json']:
    p=root/name
    if p.exists():
     b=p.read_bytes();files[name]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'data':json.loads(b)}
   out.append({'dataset':ds,'method':method,'seed':seed,'files':files})
print(json.dumps(out))
'''
remote_data=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','mirzaei@cs-cl-19.cmpt.sfu.ca','python3','-'],input=remote.encode(),timeout=60))
(OUT/'synthetic_archive_sources.json').write_text(json.dumps(remote_data,indent=2)+'\n')

def add(ds,m,s,struct,gin,provenance):
    if 'metrics' in struct:
        st=dict(struct['metrics']);ex=struct['extra_metrics'];st['edge_count_absolute_error']=abs(ex['reference_edge_count']-ex['generated_edge_count'])
        st['reference_edge_count']=ex['reference_edge_count']
    else:st=struct.get('structural_mmd',struct)
    if 'modes' in gin:
        mode=gin['modes']['topology_control'];f=mode['summary']['f1_pr']['mean'];reps=mode['per_repeat']['f1_pr']
    else:
        f=gin['metrics']['f1_pr']['mean'];reps=gin['raw_metrics']['f1_pr']
    assert len(reps)==10 and abs(statistics.mean(reps)-f)<1e-10
    raw.setdefault((ds,m),[]).append(dict(seed=s,structural=st,f1=float(f),provenance=provenance))

for x in remote_data:
    f=x['files'];add(x['dataset'],x['method'],x['seed'],f['final_table2_metrics.json']['data'],f['graph_realism_random_gin.json']['data'],f)
for ds,stem,seeds in [('GRID','grid',[0,4]),('TRIANGULAR_GRID','triangular_grid',[0,1,3]),('LOBSTER','lobster',[0,1,2])]:
    for s in seeds:
        p=DATA/'defog_synthetic_evaluation_20260906/metrics'/f'{stem}_seed{s}.json';d=read(p)
        add(ds,'defog',s,d,d['third_party_random_gin_structural_features'],dict(path=str(p),reference_hash=d['reference_canonical_graph_sha256']))

for s in range(3):
    p=DATA/f'motif_true_clean_20260906/ptc/full_matrix/seed_{s}'
    add('PTC','true',s,read(p/'final_table2_metrics.json'),read(p/'graph_realism_random_gin.json'),dict(path=str(p)))
    p=Path(f'/local-scratch2/new/gather/datasets/ptc/setting_01/seed_{s}/final_table2_metrics.json')
    f=DATA/f'ptc_matched_reference_eval_20260913/topology_feature/seed_{s}/evaluation.json'
    add('PTC','false',s,read(p),read(f),dict(structural=str(p),gin=str(f)))
    p=DATA/f'defog_ptc_full_metrics_20260907/structural/seed_{s}.json';d=read(p)
    add('PTC','defog',s,d,d['third_party_random_gin_structural_features'],dict(path=str(p)))

for m,sub in [('false','graphvae/false'),('true','graphvae/true_full'),('defog','defog')]:
    for s in range(3):
        p=DATA/f'qm9_common_eval_20260917/evaluations/{sub}/seed_{s}'
        st=read(p/('common_metrics.json' if m=='defog' else 'common_structural.json'))
        gin=st['random_gin'] if m=='defog' else read(p/'attributed_random_gin.json')['evaluation']
        add('QM9',m,s,st,gin,dict(path=str(p),structural_reference=st['reference']))

for m,sub in [('false','motif_false'),('true','motif003'),('defog','defog')]:
    for s in range(3):
        if m=='defog':
            p=DATA/f'proteins_defog_corrected_eval_20260922/seed_{s}/common_metrics.json';st=read(p);gin=st['random_gin']
        else:
            p=DATA/f'proteins_3way_comparison_20260913/structural/{sub}/seed_{s}.json';st=read(p)
            gp=DATA/f'archive_completion_20260922/PROTEINS/{"false" if m=="false" else "true_full_003"}/seed_{s}/attributed_random_gin.json';gin=read(gp)['evaluation']
        add('PROTEINS',m,s,st,gin,dict(path=str(p)))

metric_keys={'Degree MMD':'degree','Clustering MMD':'clustering','Orbit MMD':'orbit','Spectral MMD':'spectral','Diameter MMD':'diameter','Absolute error in mean edge count':'edge_count_absolute_error','F1-PR':'f1'}
checked=[]
for (ds,m),rows in raw.items():
    for metric,key in metric_keys.items():
        vals=[r['f1'] if key=='f1' else r['structural'][key] for r in rows]
        observed=[statistics.mean(vals),statistics.stdev(vals)]
        expected=cells[(ds,metric,m)]
        tokens=expected.split(' ± ')
        tolerance=[float(Decimal(10)**Decimal(Decimal(t).as_tuple().exponent))/2+1e-12 for t in tokens]
        passed=all(abs(v-float(t))<=tol for v,t,tol in zip(observed,tokens,tolerance))
        checked.append(dict(dataset=ds,method=m,metric=metric,paper=expected,mean=observed[0],sd=observed[1],seeds=[r['seed'] for r in rows],rounding_match=passed))

gains=[]
for line in tables[1].splitlines()[2:]:
    cols=[v.strip().replace('**','') for v in line.strip('|').split('|')]
    vals=[float(c.split()[0]) for c in cols[2:5]]
    for val,base in zip(cols[5:7],[vals[0],vals[2]]):
        assert abs(float(val.strip('%'))-100*(vals[1]-base)/base)<0.0051
        gains.append(val)

result=dict(manuscript=str(ROOT/'final.md'),manuscript_sha256=hashlib.sha256(paper.encode()).hexdigest(),
    original_edited_pdf=str(DATA/'PAPER_REVISION_GRAPHVAE_FULL_20260922/PAPER_FIRST_DRAFT_GRAPHVAE_FULL.pdf'),
    numerical_checks=checked,numerical_matches=sum(x['rounding_match'] for x in checked),total_cells=len(checked),
    percentage_checks=len(gains),frozen_source_hash_checks=hashchecks,source_sha256=sources,
    raw_rows=[dict(dataset=k[0],method=k[1],rows=v) for k,v in raw.items()],
    warning='Arithmetic verification is not protocol validation. See AUDIT_REPORT.md for split/reference/selection issues.')
(OUT/'NUMERICAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print('Numerical matches',result['numerical_matches'],'/',len(checked),'percentage checks',len(gains))
for x in checked:
    if not x['rounding_match']:print('MISMATCH',x)
for (ds,m),rows in raw.items():print(ds,m,'reference edges', [r['structural']['reference_edge_count'] for r in rows])
