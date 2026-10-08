#!/usr/bin/env python3
"""Aggregate normalized full-state TV for attributed graph bundles."""
import argparse, json, pickle, sys
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--config', required=True)
p.add_argument('--motif-cache-dir', required=True); p.add_argument('--reference', required=True)
p.add_argument('--generated', required=True); p.add_argument('--output', required=True)
a=p.parse_args(); sys.path.insert(0,a.repo)
from motif_counting.motif_counter import RelationalMotifCounter
from scripts.evaluate_motif_count_distance_correlation import (make_counter_args,load_yaml,count_exact_graph_records,motif_entry_metadata,hard_graph_postprocess)

_,flat=load_yaml(Path(a.config)); dev=torch.device('cpu')
c=RelationalMotifCounter(str(flat['database_name']),make_counter_args(flat,Path(a.motif_cache_dir),dev))

def load(path):
 x=torch.load(path,map_location='cpu')
 return x['graphs'] if isinstance(x,dict) else x

def records(path):
 out=[]
 for g in load(path):
  n=int(g['num_nodes']); x=torch.as_tensor(g['x']).float(); labels=x.argmax(1)+1
  ei=torch.as_tensor(g['edge_index']).long(); adj=torch.zeros(n,n); adj[ei[0],ei[1]]=1
  ea=g.get('edge_attr'); edge=None
  if ea is not None:
   ea=torch.as_tensor(ea).float(); packed=torch.zeros(ea.shape[1],n,n)
   packed[:,ei[0],ei[1]]=ea.T; edge=[packed]
  out.append(hard_graph_postprocess({'features':labels[:,None].float(),'feat_onehot':x,'adj':{c.relation_keys[0]:adj},'edge':edge}))
 return out

mapping={0:{i+1:i for i in range(torch.as_tensor(load(a.reference)[0]['x']).shape[1])}}
ref=count_exact_graph_records(c,records(a.reference),mapping,128,dev,None)
gen=count_exact_graph_records(c,records(a.generated),mapping,128,dev,None)
entries=motif_entry_metadata(c,None); groups={}
for e in entries:
 if len(e.get('rule',[]))>=2: groups.setdefault(int(e['rule_index']),[]).append(int(e['index']))
rows=[]
for ri,idx in sorted(groups.items()):
 o=ref[:,idx].clamp_min(0).sum(0); g=gen[:,idx].clamp_min(0).sum(0)
 tv=None if o.sum()<=0 or g.sum()<=0 else float((.5*((o/o.sum())-(g/g.sum())).abs().sum()).item())
 rows.append({'rule_index':ri,'state_count':len(idx),'tv':tv})
vals=[x['tv'] for x in rows if x['tv'] is not None]
payload={'metric':'aggregate normalized complete-state TV; lower is better','database':flat['database_name'],'reference_graphs':len(load(a.reference)),'generated_graphs':len(load(a.generated)),'score':sum(vals)/len(vals) if vals else None,'rules':rows}
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(payload,indent=2)+'\n'); print(payload)
