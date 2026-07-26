#!/usr/bin/env bash
set -euo pipefail

if (($# != 4)); then
  echo "Usage: $0 REPO_PATH HOST_LABEL SEED GPU" >&2
  exit 2
fi

repo_path="$1"
host_label="$2"
seed="$3"
gpu="$4"
run_root="runs/20260726/lobster_semantic_hybrid_deadline/seed_${seed}"
output_dir="runs/20260726/lobster_semantic_hybrid_deadline_validation_selection/${host_label}"
session="semantic_hybrid_select_${host_label}_20260726"
python_bin=""
for candidate in \
  /local-scratch2/mirzaei/miniconda3/envs/micro/bin/python \
  /localhome/mirzaei/miniconda3/envs/micro/bin/python
do
  if [[ -x "$candidate" ]]; then
    python_bin="$candidate"
    break
  fi
done
if [[ -z "$python_bin" ]]; then
  echo "Could not find the micro environment Python executable" >&2
  exit 3
fi

cd "$repo_path"
mkdir -p "$output_dir"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "already active: $session"
  exit 0
fi

worker_command=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "$repo_path")
while true; do
  completed=\$(find $(printf '%q' "$run_root") -name final_metrics_summary.json | wc -l)
  if ((completed >= 2)); then
    break
  fi
  echo "[wait] completed=\$completed/2 \$(date -Iseconds)"
  sleep 30
done
env CUDA_VISIBLE_DEVICES=$(printf '%q' "$gpu") MPLBACKEND=Agg \
  $(printf '%q' "$python_bin") -u scripts/select_lobster_checkpoints_per_run.py \
  --runs-root $(printf '%q' "$run_root") \
  --output-dir $(printf '%q' "$output_dir") \
  --expected-runs 2 \
  --expected-checkpoints-per-run 5 \
  --validation-rollouts 10 \
  --seed 20260726 \
  --device cuda:0 \
  --skip-materialization
printf '%s\n' "completed_at=\$(date -Iseconds)" > \
  $(printf '%q' "$output_dir/SELECTION_COMPLETE")
EOF
)

tmux new-session -d -s "$session" bash -lc \
  "$worker_command 2>&1 | tee $(printf '%q' "$output_dir/stdout.log")"
echo "launched: $session"
