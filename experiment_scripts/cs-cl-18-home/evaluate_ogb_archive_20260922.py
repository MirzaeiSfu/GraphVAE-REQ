import pathlib,subprocess,sys,time,json,os
import torch,dgl
root=pathlib.Path('/local-scratch2/mirzaei/archive_completion_20260922/OGB');root.mkdir(parents=True,exist_ok=True)
repo='/local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ'
helpers=pathlib.Path('/local-scratch2/mirzaei/edge_feature_random_gin_20260917')
raw='/local-scratch2/mirzaei/defog_ogbg_3seed_20260920/source/DeFoG/data/ogbg_molbbbp_frozen/raw/ogbg_molbbbp_frozen_graphvae_split.pt'
p=torch.load(raw,map_location='cpu');torch.save(p['splits']['test'],root/'reference.pt')
def call(script,args):subprocess.run([sys.executable,str(script)]+list(map(str,args)),check=True)
call(helpers/'convert_pyg_collection_to_dgl.py',['--input',root/'reference.pt','--output',root/'reference.bin'])
for seed in (0,1,2):
 host='13' if seed==2 else '17';remote=f'/localhome/mirzaei/defog_ogbg_3seed_20260920/archive_evaluation_20260922/seed_{seed}'
 out=root/f'seed_{seed}';out.mkdir(exist_ok=True)
 try:
  while True:
   q=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',f'mirzaei@cs-cl-{host}.cmpt.sfu.ca',f'cat {remote}/status'],capture_output=True,text=True)
   if q.stdout.startswith('COMPLETE'):break
   if q.stdout.startswith('FAILED'):raise RuntimeError('Remote generation failed')
   time.sleep(30)
  subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes',f'mirzaei@cs-cl-{host}.cmpt.sfu.ca:{remote}/',str(out)+'/'],check=True)
  call(helpers/'convert_pyg_collection_to_dgl.py',['--input',out/'generated_graphs.pt','--output',out/'generated.bin'])
  call(helpers/'evaluate_dgl_all_modes.py',['--repo',repo,'--generated',out/'generated.bin','--reference',root/'reference.bin','--output',out/'random_gin_all_modes.json','--label','ogb_defog','--seed',seed,'--device','cpu','--repeats',10])
  call('/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py',['--repo',repo,'--generated',out/'generated.bin','--reference',root/'reference.bin','--output',out/'structural.json','--label','ogb_defog','--seed',seed,'--skip-random-gin'])
  (out/'COMPLETE').write_text('Completed structural and GIN evaluation; motif TV pending verified training-rule selection.')
 except Exception as e:(out/'FAILED').write_text(repr(e));print(repr(e),flush=True)
 finally:
  subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes',str(out)+'/',f'mirzaei@cs-cl-19.cmpt.sfu.ca:/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/evaluations/defog_seed_{seed}/'])
