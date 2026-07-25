#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "Usage: $0 REPO_PATH" >&2
  exit 2
fi

repo_path="$1"
gpu="1"
seed="0"
date_prefix="20260725"
run_root="runs/20260725/lobster_graphvae_mm_fixed_split_controls/seed_0"
predecessor_job="lobster_graphvae_mm_fixed_split_native40_legacy__cs-cl-09_gpu1"
job="lobster_graphvae_mm_fixed_split_matched1_legacy__cs-cl-09_gpu1"
config="configs/matrix_motif/lobster_graphvae_mm_fixed_split_matched1_legacy.yaml"
run_dir="$run_root/$job"
run_label="${date_prefix}_${job}"
session="kiarash_control_seed0_wave2_20260725"
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
mkdir -p "$run_dir"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "already active: $session"
  exit 0
fi

worker_command=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "$repo_path")
predecessor=$(printf '%q' "$run_root/$predecessor_job/seed_0/model_19999_0")
while [[ ! -f "\$predecessor" ]]; do
  echo "[wait] predecessor incomplete \$(date -Iseconds)"
  sleep 30
done
mkdir -p $(printf '%q' "$run_dir")
printf '%s\n' \
  $(printf '%q' "date_prefix=$date_prefix") \
  $(printf '%q' "run_root=$run_root") \
  $(printf '%q' "job_dir=$job") \
  $(printf '%q' "run_dir=$run_dir") \
  $(printf '%q' "run_label=$run_label") \
  $(printf '%q' "config_name=lobster_graphvae_mm_fixed_split_matched1_legacy") \
  $(printf '%q' "config_path=$config") \
  'host=cs-cl-09' \
  'gpu=1' \
  'cuda_visible_devices=1' \
  'device=cuda:0' \
  $(printf '%q' "python_bin=$python_bin") \
  > $(printf '%q' "$run_dir/RUN_INFO.txt")
env CUDA_VISIBLE_DEVICES=$gpu MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
  $(printf '%q' "$python_bin") -u main.py \
  --config $(printf '%q' "$config") \
  --device cuda:0 \
  --graph_save_path $(printf '%q' "$run_dir") \
  --run_label $(printf '%q' "$run_label") \
  --disable_dataset_cache true \
  --seed $seed
EOF
)

tmux new-session -d -s "$session" bash -lc \
  "$worker_command 2>&1 | tee $(printf '%q' "$run_dir/stdout.log")"
echo "launched: $session"
