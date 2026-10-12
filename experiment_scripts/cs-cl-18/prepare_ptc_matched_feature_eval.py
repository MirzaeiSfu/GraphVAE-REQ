#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from ggm_eval.io import load_pyg_collection

root=Path('/local-scratch2/mirzaei/ptc_matched_reference_eval_20260913')
reference=load_pyg_collection('/local-scratch2/mirzaei/defog_ptc_frozen_20260906/artifacts/ptc/real_test_graphs.pt')
ref=[]
for g in reference:
    a=np.zeros((int(g.num_nodes),int(g.num_nodes)),dtype=np.float32)
    e=g.edge_index.detach().cpu().numpy(); a[e[0],e[1]]=1
    ref.append(a)
for seed in range(3):
    out=root/'topology_feature'/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=True)
    src=Path(f'/local-scratch2/new/gather/datasets/ptc/setting_01/seed_{seed}/Single_comp_generatedGraphs_adj_final_eval.npy')
    (out/'Single_comp_generatedGraphs_adj_final_eval.npy').write_bytes(src.read_bytes())
    with (out/'testGraphs_adj_.npy').open('wb') as f: np.save(f,np.asarray(ref,dtype=object),allow_pickle=True)
