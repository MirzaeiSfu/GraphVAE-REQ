#!/usr/bin/env bash
set -euo pipefail
repo=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/GraphVAE-REQ
py=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
root=/local-scratch2/mirzaei/mutag_threshold_calibration_20260912
cd "$repo"; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
for t in 0.25 0.30 0.35 0.40 0.45 0.50; do
 for s in 0 1 2; do
  out=$root/validation/t_${t}/seed_$s; mkdir -p "$out"
  "$py" scripts/evaluate_attributed_graph_realism_checkpoints.py \
   --run-dir "$root/checkpoints/seed_$s" --config configs/mutag_edgefeat_campaign/full_matrix.yaml \
   --checkpoint "$root/checkpoints/seed_$s/best_validation_mmd_model" --dataset-cache-dir "$root/dataset_cache" \
   --split validation --modes topology_control --max-graphs 18 --generation-batch-size 32 \
   --generation-seed $((25000+s)) --evaluator-seed 0 --repeats 1 --adjacency-threshold "$t" \
   --device cuda:0 --output-dir "$out" --save-samples >/dev/null
 done
done
"$py" - <<"PY"
import numpy as np,glob,json,os
root="/local-scratch2/mirzaei/mutag_threshold_calibration_20260912/validation"
rows=[]
for td in sorted(glob.glob(root+"/t_*")):
 t=float(os.path.basename(td)[2:]); ge=[]; re=[]
 for p in sorted(glob.glob(td+"/seed_*/attributed_graph_samples.npz")):
  x=np.load(p,allow_pickle=True)
  ge += [np.asarray(e).shape[1]/2 for e in x["generated_edges"]]
  re += [np.asarray(e).shape[1]/2 for e in x["reference_edges"]]
 rows.append({"threshold":t,"generated_mean_edges":float(np.mean(ge)),"reference_mean_edges":float(np.mean(re)),"absolute_error":float(abs(np.mean(ge)-np.mean(re)))})
best=min(rows,key=lambda x:(x["absolute_error"],x["threshold"])); json.dump({"selection_rule":"minimum aggregate validation mean-edge absolute error","candidates":rows,"selected":best},open(root+"/selection.json","w"),indent=2); print(json.dumps({"candidates":rows,"selected":best},indent=2))
PY
