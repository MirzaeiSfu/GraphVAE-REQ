#!/usr/bin/env bash
set -euo pipefail
root=/local-scratch2/mirzaei/qm9_common_eval_20260917
py=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
remote=/localhome/mirzaei/qm9_defog_seedfixed250_batch128_20260916
trap 'printf "FAILED line=%s\n" "$LINENO" > "$root/status/complete_three_seed.status"' ERR
printf 'WAITING_FOR_EXPORT\n' > "$root/status/complete_three_seed.status"
while true; do
  state=$(ssh -o BatchMode=yes -o ConnectTimeout=10 mirzaei@cs-cl-17.cmpt.sfu.ca "cat $remote/raw_export_fixed/seed_0.status 2>/dev/null" || true)
  if [[ $state == COMPLETE* ]]; then break; fi
  if [[ $state == FAILED* ]]; then exit 1; fi
  sleep 30
done
rsync -a -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' "mirzaei@cs-cl-17.cmpt.sfu.ca:$remote/raw_export_fixed/seed_0/generated_graphs.pt" "$root/defog/raw/seed_0.pt"
out=$root/evaluations/defog/seed_0
mkdir -p "$out"
printf 'EVALUATING\n' > "$root/status/complete_three_seed.status"
"$py" "$root/convert_defog_raw_to_common_dgl.py" --input "$root/defog/raw/seed_0.pt" --output "$out/generated_attributed_graphs.bin" --metadata "$out/conversion.json" --seed 0
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
repo=$root/source/GraphVAE-REQ
reference=$root/evaluations/graphvae/true_full/seed_0/reference_attributed_graphs.bin
"$py" "$root/evaluate_dgl_common_metrics.py" --repo "$repo" --generated "$out/generated_attributed_graphs.bin" --reference "$reference" --output "$out/common_metrics.json" --label defog --seed 0 --device cuda --repeats 10
"$py" "$root/evaluate_qm9_motif_tv_batch.py" --repo "$repo" --config "$root/graphvae/true_full/seed_0/run_config_used.yaml" --dataset-cache "$root/cache/dataset/QM9_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-default.pkl" --motif-cache-dir "$root/cache/motif" --reference "$reference" --device cpu --batch-size 16 --item defog 0 "$out/generated_attributed_graphs.bin" "$out/motif_tv.json"
"$py" "$root/build_complete_report.py"
archive=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9
ssh -o BatchMode=yes mirzaei@cs-cl-19.cmpt.sfu.ca "mkdir -p $archive/completed_common_eval_20260922"
rsync -a -e 'ssh -o BatchMode=yes' "$root/evaluations/" "mirzaei@cs-cl-19.cmpt.sfu.ca:$archive/completed_common_eval_20260922/"
rsync -a -e 'ssh -o BatchMode=yes' "$root/QM9_THREE_SEED_COMPLETE_COMPARISON_20260922.md" "mirzaei@cs-cl-19.cmpt.sfu.ca:$archive/"
printf 'COMPLETE report_and_metrics_archived\n' > "$root/status/complete_three_seed.status"
