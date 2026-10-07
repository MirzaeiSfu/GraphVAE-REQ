#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from ggm_eval.io import save_pyg_collection

p=argparse.ArgumentParser()
p.add_argument('--input',required=True); p.add_argument('--output',required=True)
p.add_argument('--method',required=True); p.add_argument('--seed',type=int,required=True)
p.add_argument('--dataset',default='GRID')
p.add_argument('--feature-dim',type=int,default=1)
p.add_argument('--feature-schema',default='constant-node|export=topology_control')
a=p.parse_args()
items=list(np.load(a.input,allow_pickle=True))
graphs=[]
for item in items:
    x=np.asarray(item.tolist() if getattr(item,'dtype',None)==object else item,dtype=np.float32)
    b=np.logical_or(x>=0.5,x.T>=0.5); np.fill_diagonal(b,False)
    # Inputs already contain the selected largest component; remove isolates defensively.
    keep=np.flatnonzero(np.logical_or(b.any(0),b.any(1)))
    b=b[np.ix_(keep,keep)]
    uv=np.asarray(np.nonzero(b),dtype=np.int64)
    node_values=torch.zeros((len(keep),a.feature_dim),dtype=torch.float32)
    node_values[:,0]=1.0
    graphs.append(Data(x=node_values,edge_index=torch.as_tensor(uv),num_nodes=len(keep)))
meta={
 'dataset':a.dataset.upper(),'split':'generated','generator':a.method,'training_seed':a.seed,
 'feature_mode':'topology_control','feature_schema':a.feature_schema,
 'postprocessing':{'adjacency_threshold':0.5,'symmetry':'logical_or','self_loops':'removed','isolated_nodes':'removed','connected_components':'source_saved_largest_component'},
 'source_path':str(Path(a.input).resolve()),'accepted_count':len(graphs)
}
print(json.dumps(save_pyg_collection(a.output,graphs,metadata=meta,normalize=True),indent=2))
