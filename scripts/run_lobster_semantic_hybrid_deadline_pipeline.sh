#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

repo_paths="CLUSTER_REPO_PATHS_LOBSTER_SEMANTIC_HYBRID_DEADLINE.txt"
remote_runs="runs/20260726/lobster_semantic_hybrid_deadline"
remote_selections="runs/20260726/lobster_semantic_hybrid_deadline_validation_selection"
local_runs="collected_runs/20260726/lobster_semantic_hybrid_deadline"
local_selections="collected_runs/20260726/lobster_semantic_hybrid_deadline_validation_selection"
evaluation_dir="collected_runs/20260726/lobster_semantic_hybrid_deadline_heldout_evaluation"
pipeline_log_dir="runs/20260726/lobster_semantic_hybrid_deadline_pipeline"
python_bin="/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python"
generated_filename="Single_comp_generatedGraphs_adj_semantic_hybrid_rollout0.npy"
random_gin_filename="graph_realism_random_gin_semantic_hybrid_rollout0.json"

mkdir -p "$pipeline_log_dir"

while true; do
  ready=0
  while read -r host worker_repo; do
    marker="${worker_repo}/${remote_selections}/${host}/SELECTION_COMPLETE"
    if ssh -n -o ConnectTimeout=10 "$host" "test -f $(printf '%q' "$marker")"; then
      ready=$((ready + 1))
    fi
  done < "$repo_paths"
  echo "[wait] validation selections ready=${ready}/3 $(date -Iseconds)"
  if ((ready == 3)); then
    break
  fi
  sleep 60
done

scripts/cluster_collect_results.sh \
  --repo-paths "$repo_paths" \
  --remote-run-root "$remote_runs" \
  --collect-root collected_runs \
  --date-prefix 20260726
scripts/cluster_collect_results.sh \
  --repo-paths "$repo_paths" \
  --remote-run-root "$remote_selections" \
  --collect-root collected_runs \
  --date-prefix 20260726

"$python_bin" -u scripts/evaluate_lobster_frozen_selections.py \
  --selection-json "$local_selections/cs-cl-18/selection.json" \
  --selection-json "$local_selections/cs-cl-19/selection.json" \
  --selection-json "$local_selections/cs-cl-26/selection.json" \
  --runs-root "$local_runs" \
  --condition lobster_semantic_hybrid_r001_legacy \
  --condition lobster_semantic_hybrid_r001_edgecount01_legacy \
  --output-dir "$evaluation_dir" \
  --expected-runs 6 \
  --test-rollouts 10 \
  --seed 21260726 \
  --device cuda:0 \
  --generated-filename "$generated_filename" \
  --model-filename semantic_hybrid_frozen_validation_model.pt

run_args=()
while IFS= read -r generated_path; do
  run_args+=(--run-dir "$(dirname "$generated_path")")
done < <(find "$local_runs" -type f -name "$generated_filename" | sort)
if ((${#run_args[@]} != 12)); then
  echo "Expected 6 generated run directories, found $((${#run_args[@]} / 2))" >&2
  exit 4
fi

"$python_bin" -u scripts/evaluate_graph_realism_batch.py \
  "${run_args[@]}" \
  --generated-filename "$generated_filename" \
  --reference-filename heldoutTestGraphs_adj_.npy \
  --json-filename "$random_gin_filename" \
  --summary-csv "$evaluation_dir/random_gin_summary.csv" \
  --repeats 10 \
  --max-graphs 1000 \
  --seed 0 \
  --device cpu

printf '%s\n' "completed_at=$(date -Iseconds)" > \
  "$pipeline_log_dir/PIPELINE_COMPLETE"
echo "[complete] semantic hybrid deadline pipeline $(date -Iseconds)"
