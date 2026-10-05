#!/usr/bin/env bash
set -euo pipefail

source /local-scratch2/mirzaei/miniconda3/etc/profile.d/conda.sh
conda activate lgd_env

repo_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_dir"

dataset_name="${1:?usage: $0 DATASET}"
graphvae_repo=/local-scratch2/mirzaei/fb/GraphVAE-REQ
case "$dataset_name" in
  GRID)
    exec python prepare_graphvae_req_data.py --graphvae-repo "$graphvae_repo" --cache /local-scratch2/mirzaei/Abdolreza/harvest/cs-cl-19/local-scratch2__mirzaei__fb__GraphVAE-REQ/runs/count_distance_3seed/grid/dataset_cache/GRID_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-default.pkl --dataset GRID --topology-only
    ;;
  LOBSTER)
    exec python prepare_graphvae_req_data.py --graphvae-repo "$graphvae_repo" --cache /local-scratch2/mirzaei/Abdolreza/harvest/cs-cl-16/local-scratch__localhome__mirzaei__fb__GraphVAE-REQ/runs/count_distance_3seed/lobster/dataset_cache/LOBSTER_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-lobster-optimal_v2.pkl --dataset LOBSTER --topology-only
    ;;
  TRIANGULAR_GRID)
    exec python prepare_graphvae_req_data.py --graphvae-repo "$graphvae_repo" --cache /local-scratch2/mirzaei/countv5_tri_cache/TRIANGULAR_GRID_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-default.pkl --dataset TRIANGULAR_GRID --topology-only
    ;;
  PTC)
    exec python prepare_graphvae_req_data.py --graphvae-repo "$graphvae_repo" --cache /local-scratch2/mirzaei/ptc_rule_metrics_20260906/cache/PTC_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-gin-node-label-v2.pkl --dataset PTC
    ;;
  AIDS)
    exec python prepare_graphvae_req_data.py --graphvae-repo "$graphvae_repo" --cache /local-scratch2/mirzaei/aids_common_eval_10k_20260917/cache/dataset/AIDS_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-all_components_features-tu-quantile8-maxall.pkl --dataset AIDS
    ;;
  MUTAG)
    exec python prepare_graphvae_req_data.py --indexed-raw-dir /local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG/data/mutag_edgefeat/raw --raw-prefix mutag --dataset MUTAG
    ;;
  QM9)
    exec python prepare_graphvae_req_data.py --graphvae-repo "$graphvae_repo" --cache /local-scratch2/mirzaei/qm9_common_eval_20260917/cache/dataset/QM9_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-default.pkl --dataset QM9
    ;;
  *)
    echo "Unknown dataset: $dataset_name" >&2
    exit 2
    ;;
esac
