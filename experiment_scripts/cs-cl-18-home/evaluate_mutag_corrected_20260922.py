import sys, pathlib, subprocess, json, os
import torch,dgl
ROOT=pathlib.Path('/local-scratch2/mirzaei/archive_completion_20260922/MUTAG')
helpers=pathlib.Path('/local-scratch2/mirzaei/edge_feature_random_gin_20260917')
repo='/local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ'
reference=helpers/'work/mutag_reference.bin'
for seed in (1,2):
 out=ROOT/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
 try:
  subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes',f'mirzaei@cs-cl-19.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_defog_independent_20260919/seed_{seed}/generation/generated_graphs.pt',str(out)+'/'],check=True)
  payload=torch.load(out/'generated_graphs.pt',map_location='cpu');graphs=[]
  for x in payload['graphs']:
   ei=x['edge_index']; g=dgl.graph((ei[0],ei[1]),num_nodes=x['num_nodes']);g.ndata['attr']=x['x'].float()
   e=x['edge_attr'];assert e.shape[1]==5 and not e[:,0].any(), 'Unexpected no-edge coding'
   g.edata['attr']=e[:,1:].float();graphs.append(g)
  dgl.save_graphs(str(out/'generated.bin'),graphs)
  (out/'conversion.json').write_text(json.dumps(dict(source=str(out/'generated_graphs.pt'),graphs=len(graphs),edge_conversion='Remove verified-zero no-edge channel; preserve four bond categories',reference=str(reference)),indent=2))
  def call(script,args):subprocess.run([sys.executable,str(script)]+list(map(str,args)),check=True)
  call(helpers/'evaluate_dgl_all_modes.py',['--repo',repo,'--generated',out/'generated.bin','--reference',reference,'--output',out/'random_gin_all_modes.json','--label','mutag_corrected_defog','--seed',seed,'--device','cpu','--repeats',10])
  call('/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py',['--repo',repo,'--generated',out/'generated.bin','--reference',reference,'--output',out/'structural.json','--label','mutag_corrected_defog','--seed',seed,'--skip-random-gin'])
  sys.path[:0]=[repo,repo+'/scripts','/local-scratch/localhome/mirzaei'];os.chdir(repo)
  import lgd_retained_tv
  ref,_=dgl.load_graphs(str(reference)); matched=graphs[:len(ref)]
  tv=lgd_retained_tv.evaluate('MUTAG',ref,matched,ref)
  tv.pop('train',None);tv['reference_scope']='held-out test only';tv['generated_graph_count']=len(matched)
  (out/'motif_tv_test.json').write_text(json.dumps(tv,indent=2));(out/'COMPLETE').write_text('Evaluation complete')
 except Exception as e:
  (out/'FAILED').write_text(repr(e));print(repr(e),flush=True)
 finally:
  subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes',str(out)+'/',f'mirzaei@cs-cl-19.cmpt.sfu.ca:/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/evaluations/corrected_defog_seed_{seed}/'])
