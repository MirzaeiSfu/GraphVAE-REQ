#!/usr/bin/env bash
set -u
root=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC
mkdir -p "$root/manifests" "$root/reports"
exec >> "$root/manifests/transfer.log" 2>&1
failed=0
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-09/PTC
echo START /local-scratch2/mirzaei/PTC
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/PTC/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-09/PTC/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-16/normalized_rule_correlation_ptc
echo START /local-scratch/localhome/mirzaei/normalized_rule_correlation_ptc
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-16.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/normalized_rule_correlation_ptc/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-16/normalized_rule_correlation_ptc/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-16/normalized_rule_correlation_ptc
echo START /localhome/mirzaei/normalized_rule_correlation_ptc
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-16.cmpt.sfu.ca:/localhome/mirzaei/normalized_rule_correlation_ptc/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-16/normalized_rule_correlation_ptc/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-17/defog_ptc_metrics_20260907
echo START /local-scratch/localhome/mirzaei/defog_ptc_metrics_20260907
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/defog_ptc_metrics_20260907/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-17/defog_ptc_metrics_20260907/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-17/defog_ptc_metrics_20260907
echo START /localhome/mirzaei/defog_ptc_metrics_20260907
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/defog_ptc_metrics_20260907/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-17/defog_ptc_metrics_20260907/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_matched_reference_eval_20260913
echo START /local-scratch2/mirzaei/ptc_matched_reference_eval_20260913
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ptc_matched_reference_eval_20260913/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_matched_reference_eval_20260913/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_pruned_state_tv_train_test_20260909
echo START /local-scratch2/mirzaei/ptc_pruned_state_tv_train_test_20260909
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ptc_pruned_state_tv_train_test_20260909/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_pruned_state_tv_train_test_20260909/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_rule_metrics_20260906
echo START /local-scratch2/mirzaei/ptc_rule_metrics_20260906
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ptc_rule_metrics_20260906/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_rule_metrics_20260906/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_matched_reference_eval_20260913_incomplete_20graph_1454
echo START /local-scratch2/mirzaei/ptc_matched_reference_eval_20260913_incomplete_20graph_1454
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ptc_matched_reference_eval_20260913_incomplete_20graph_1454/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_matched_reference_eval_20260913_incomplete_20graph_1454/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_motif_true_vs_baseline_gather_corrected_20260905
echo START /local-scratch2/mirzaei/ptc_motif_true_vs_baseline_gather_corrected_20260905
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ptc_motif_true_vs_baseline_gather_corrected_20260905/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_motif_true_vs_baseline_gather_corrected_20260905/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_matched_reference_eval_20260913_schema_failed_1454
echo START /local-scratch2/mirzaei/ptc_matched_reference_eval_20260913_schema_failed_1454
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ptc_matched_reference_eval_20260913_schema_failed_1454/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_matched_reference_eval_20260913_schema_failed_1454/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/defog_ptc_frozen_20260906
echo START /local-scratch2/mirzaei/defog_ptc_frozen_20260906
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/defog_ptc_frozen_20260906/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/defog_ptc_frozen_20260906/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_pruned_state_tv_soft_20260909
echo START /local-scratch2/mirzaei/ptc_pruned_state_tv_soft_20260909
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ptc_pruned_state_tv_soft_20260909/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_pruned_state_tv_soft_20260909/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/collected_ptc_aids_motif_20260905
echo START /local-scratch2/mirzaei/collected_ptc_aids_motif_20260905
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/collected_ptc_aids_motif_20260905/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/collected_ptc_aids_motif_20260905/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_motif_true_vs_gather_comparison_20260905
echo START /local-scratch2/mirzaei/ptc_motif_true_vs_gather_comparison_20260905
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ptc_motif_true_vs_gather_comparison_20260905/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_motif_true_vs_gather_comparison_20260905/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/defog_ptc_full_metrics_20260907
echo START /local-scratch2/mirzaei/defog_ptc_full_metrics_20260907
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/defog_ptc_full_metrics_20260907/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/defog_ptc_full_metrics_20260907/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_motif_true_vs_baseline_gather_corrected_20260906
echo START /local-scratch2/mirzaei/ptc_motif_true_vs_baseline_gather_corrected_20260906
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ptc_motif_true_vs_baseline_gather_corrected_20260906/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_motif_true_vs_baseline_gather_corrected_20260906/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_full_state_correlation_20260906
echo START /local-scratch2/mirzaei/ptc_full_state_correlation_20260906
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ptc_full_state_correlation_20260906/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_full_state_correlation_20260906/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/mutag_ptc_seed2_configs
echo START /local-scratch/localhome/mirzaei/mutag_ptc_seed2_configs
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/mutag_ptc_seed2_configs/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/mutag_ptc_seed2_configs/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/PTC_verified_20260914
echo START /local-scratch/localhome/mirzaei/PTC_verified_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/PTC_verified_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/PTC_verified_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_metric_configs_20260906
echo START /local-scratch/localhome/mirzaei/ptc_metric_configs_20260906
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/ptc_metric_configs_20260906/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_metric_configs_20260906/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/mutag_ptc_seed2_configs
echo START /localhome/mirzaei/mutag_ptc_seed2_configs
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/localhome/mirzaei/mutag_ptc_seed2_configs/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/mutag_ptc_seed2_configs/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/PTC_verified_20260914
echo START /localhome/mirzaei/PTC_verified_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/localhome/mirzaei/PTC_verified_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/PTC_verified_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_metric_configs_20260906
echo START /localhome/mirzaei/ptc_metric_configs_20260906
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/localhome/mirzaei/ptc_metric_configs_20260906/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-18/ptc_metric_configs_20260906/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-19/defog_ptc_metrics_20260907
echo START /local-scratch2/mirzaei/defog_ptc_metrics_20260907
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/defog_ptc_metrics_20260907/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/cs-cl-19/defog_ptc_metrics_20260907/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/gather
echo START /local-scratch2/new/gather/datasets/ptc
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/new/gather/datasets/ptc/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/sources/gather/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/reports
echo START /local-scratch2/mirzaei/DATASET_REPORTS_ALL3_20260917/PTC_ALL_RESULTS_3SEED_20260917.md
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/DATASET_REPORTS_ALL3_20260917/PTC_ALL_RESULTS_3SEED_20260917.md /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/reports/PTC_ALL_RESULTS_3SEED_20260917.md || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/reports
echo START /local-scratch2/mirzaei/ALL_DATASET_MD_REPORTS_20260914/PTC_ALL_RESULTS_VERIFIED_20260914.md
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ALL_DATASET_MD_REPORTS_20260914/PTC_ALL_RESULTS_VERIFIED_20260914.md /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PTC/reports/PTC_ALL_RESULTS_VERIFIED_20260914.md || failed=$((failed+1))
echo "FAILED_TRANSFERS=$failed"
if [ "$failed" -eq 0 ]; then date -Is > "$root/manifests/TRANSFER_COMPLETE"; else date -Is > "$root/manifests/TRANSFER_INCOMPLETE"; fi
