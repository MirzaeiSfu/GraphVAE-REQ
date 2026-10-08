"""Refresh supplemental per-seed tables without overwriting historical reports."""
import pathlib,json,statistics,math,subprocess,time,datetime
root=pathlib.Path('/local-scratch2/mirzaei/archive_completion_20260922')
def flat(d,p=''):
 out={}
 for k,v in d.items():
  key=p+k
  if isinstance(v,dict):out.update(flat(v,key+'.'))
  elif isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v):out[key]=v
 return out
def report(ds):
 base=root/ds; rows={}
 for path in base.rglob('*.json'):
  if path.name not in ('attributed_random_gin.json','random_gin_all_modes.json','common_metrics.json','structural.json','motif_tv_test.json'):continue
  data=json.loads(path.read_text());method='DeFoG'
  if 'true_full_003' in path.parts:method='Motif=True full alpha=0.03'
  elif 'false' in path.parts:method='Motif=False'
  seed=next((x for x in path.parts if x.startswith('seed_')),'unknown')
  values={}
  ev=data.get('evaluation',data.get('random_gin',{})) or {}
  for mode,m in ev.get('modes',{}).items():
   for name,v in m.get('summary',{}).items():
    if isinstance(v,dict) and isinstance(v.get('mean'),(int,float)):values[f'GIN.{mode}.{name}']=v['mean']
  values.update({'structural.'+k:v for k,v in data.get('structural_mmd',{}).items() if isinstance(v,(float,int))})
  if path.name=='motif_tv_test.json':values['motif.test_retained_state_TV']=data['test']['mean_rule_tv']
  for metric,value in values.items():
   if value is not None and math.isfinite(value):rows.setdefault((metric,method),{})[seed]=value
 lines=[f'# {ds}: new archived evaluation supplement','',f'Updated {datetime.datetime.now().isoformat()}. Partial until all jobs finish. Historical comparison reports remain under `reports/`; this supplement does not replace them.','',
 'Means and sample SD below are across available generator-labelled results, not GIN repeats. PROTEINS historical DeFoG seeds 0/1 have duplicate generated collections, so their nominal SD is not independent-seed uncertainty. MUTAG here includes only corrected DeFoG seeds 1/2. OGB here is DeFoG only; no unverified three-way comparison is claimed.','',
 '| Metric | Method | Per-seed values | Available n | Mean ± sample SD |','|---|---|---|---:|---:|']
 for (metric,method),pairs in sorted(rows.items()):
  vals=list(pairs.values());s='N/A' if len(vals)<2 else f'{statistics.stdev(vals):.7g}'
  lines.append(f'| {metric} | {method} | '+', '.join(f'{k}={v:.7g}' for k,v in sorted(pairs.items()))+f' | {len(vals)} | {statistics.mean(vals):.7g} ± {s} |')
 lines+=['','## Original result files','']+[f'- `{p}`' for p in sorted(base.rglob('*.json'))]
 path=base/f'{ds}_NEW_EVALUATION_SUPPLEMENT_20260922.md';path.write_text('\n'.join(lines)+'\n')
 subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes',str(path),f'mirzaei@cs-cl-19.cmpt.sfu.ca:/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/{ds}/'],check=True)
for attempt in range(180):
 for ds in ('MUTAG','PROTEINS','OGB'):report(ds)
 active=any(subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0 for name in ('proteins_baseline_eval_20260922','ogb_common_eval_20260922','mutag_complete_eval_20260922'))
 if not active:break
 time.sleep(60)
