#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

root=/local-scratch2/mirzaei/proteins_all_pruned_rules_correlation_20260913
py=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
repo=/local-scratch2/mirzaei/eval-count-v5-20260831
graphvae_eval="$repo/scripts/evaluate_motif_count_distance_correlation_all_pruned.py"
external_eval="$repo/scripts/evaluate_external_all_pruned_motif_distribution.py"
shared=/local-scratch2/mirzaei/normalized_rule_correlation_assets/shared
pruned=/local-scratch2/mirzaei/motif_corr_new_20260912/proteins
false_models=/local-scratch2/mirzaei/normalized_rule_correlation_assets/proteins/false
defog=/local-scratch2/mirzaei/proteins_defog_3seed_20260910

mkdir -p "$root"/{motif_true,motif_false,defog}
rm -f "$root/COMPLETE" "$root/FAILED"

run_graphvae_group() {
  local method=$1
  local config checkpoint dataset
  for seed in 0 1 2; do
    if [[ $method == motif_true ]]; then
      config="$pruned/seed_$seed/run_config_used.yaml"
      checkpoint="$pruned/seed_$seed/best_validation_mmd_model"
      dataset="$pruned/seed_$seed/dataset.pkl"
    else
      config="$shared/proteins_false.yaml"
      checkpoint="$false_models/seed_$seed.model"
      dataset="$pruned/seed_0/dataset.pkl"
    fi
    "$py" -u "$graphvae_eval" \
      --config "$config" \
      --motif-selection-config "$pruned/seed_0/run_config_used.yaml" \
      --checkpoint "$checkpoint" \
      --dataset-cache "$dataset" \
      --motif-cache-dir "$pruned/motif_cache" \
      --output "$root/$method/seed_$seed.json" \
      --seed "$seed" --device cpu \
      --generation-batch-size 32 --count-batch-size 128 \
      > "$root/$method/seed_$seed.log" 2>&1 &
  done
  wait
}

run_graphvae_group motif_true
run_graphvae_group motif_false

for seed in 0 1 2; do
  if [[ $seed == 0 ]]; then
    generated="$defog/final_eval/seed_0/generated_graphs.pt"
  else
    generated="$defog/collected_final_eval/seed_$seed/generated_graphs.pt"
  fi
  "$py" -u "$external_eval" \
    --config "$pruned/seed_0/run_config_used.yaml" \
    --dataset-cache "$pruned/seed_0/dataset.pkl" \
    --motif-cache-dir "$pruned/motif_cache" \
    --generated-graphs "$generated" \
    --output "$root/defog/seed_$seed.json" \
    --dataset-label PROTEINS --device cpu --count-batch-size 128 \
    > "$root/defog/seed_$seed.log" 2>&1 &
done
wait

"$py" - "$root" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for method in ("motif_true", "motif_false", "defog"):
    rows = []
    for seed in range(3):
        data = json.loads((root / method / f"seed_{seed}.json").read_text())
        result = (
            data["result"] if method == "defog"
            else data["hard_all_pruned_rules_motif_correlation"]
        )
        state_count = (
            data["pruned_state_count"] if method == "defog"
            else data["motif_entry_count"]
        )
        if state_count != 19 or result["all_pruned_rule_count"] != 5:
            raise RuntimeError(
                f"{method} seed {seed}: expected 19 states/5 rules, "
                f"got {state_count}/{result['all_pruned_rule_count']}"
            )
        score = (
            result["score"] if method == "defog"
            else result["aggregate_state_distribution_tv"]
        )
        informative = (
            result["informative_rules_score"] if method == "defog"
            else result["informative_rules_aggregate_state_distribution_tv"]
        )
        rows.append({"seed": seed, "all_rules_tv": score,
                     "informative_rules_tv": informative})
    summary[method] = {
        "seeds": rows,
        "all_rules_mean": statistics.mean(r["all_rules_tv"] for r in rows),
        "all_rules_sample_std": statistics.stdev(r["all_rules_tv"] for r in rows),
        "informative_rules_mean": statistics.mean(r["informative_rules_tv"] for r in rows),
        "informative_rules_sample_std": statistics.stdev(r["informative_rules_tv"] for r in rows),
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

date -Is > "$root/COMPLETE"
