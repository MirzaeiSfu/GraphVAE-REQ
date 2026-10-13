#!/usr/bin/env bash
set -u
root=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9
mkdir -p "$root/manifests" "$root/reports"
exec >> "$root/manifests/transfer.log" 2>&1
failed=0
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-09/qm9_defog_matched_split_20260914
echo START /local-scratch2/mirzaei/qm9_defog_matched_split_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/qm9_defog_matched_split_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-09/qm9_defog_matched_split_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_lab24_20260914
echo START /local-scratch/localhome/mirzaei/qm9_defog_lab24_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/qm9_defog_lab24_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_lab24_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_seedfixed250_batch128_20260916
echo START /local-scratch/localhome/mirzaei/qm9_defog_seedfixed250_batch128_20260916
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/qm9_defog_seedfixed250_batch128_20260916/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_seedfixed250_batch128_20260916/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_pruned_training_cache_top10_20260914
echo START /local-scratch/localhome/mirzaei/qm9_pruned_training_cache_top10_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/qm9_pruned_training_cache_top10_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_pruned_training_cache_top10_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_matched250_batch128_20260915
echo START /local-scratch/localhome/mirzaei/qm9_defog_matched250_batch128_20260915
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/qm9_defog_matched250_batch128_20260915/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_matched250_batch128_20260915/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_sanity_pruned10_20260914
echo START /local-scratch/localhome/mirzaei/qm9_cp_smoothed_sanity_pruned10_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/qm9_cp_smoothed_sanity_pruned10_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_sanity_pruned10_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_random_sanity_20260914
echo START /local-scratch/localhome/mirzaei/qm9_cp_smoothed_random_sanity_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/qm9_cp_smoothed_random_sanity_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_random_sanity_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_sanity_20260914
echo START /local-scratch/localhome/mirzaei/qm9_cp_smoothed_sanity_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/qm9_cp_smoothed_sanity_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_sanity_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_seed_fix_backup_20260916
echo START /local-scratch/localhome/mirzaei/qm9_defog_seed_fix_backup_20260916
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/qm9_defog_seed_fix_backup_20260916/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_seed_fix_backup_20260916/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_lab24_20260914
echo START /localhome/mirzaei/qm9_defog_lab24_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/qm9_defog_lab24_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_lab24_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_seedfixed250_batch128_20260916
echo START /localhome/mirzaei/qm9_defog_seedfixed250_batch128_20260916
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/qm9_defog_seedfixed250_batch128_20260916/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_seedfixed250_batch128_20260916/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_pruned_training_cache_top10_20260914
echo START /localhome/mirzaei/qm9_pruned_training_cache_top10_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/qm9_pruned_training_cache_top10_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_pruned_training_cache_top10_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_matched250_batch128_20260915
echo START /localhome/mirzaei/qm9_defog_matched250_batch128_20260915
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/qm9_defog_matched250_batch128_20260915/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_matched250_batch128_20260915/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_sanity_pruned10_20260914
echo START /localhome/mirzaei/qm9_cp_smoothed_sanity_pruned10_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/qm9_cp_smoothed_sanity_pruned10_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_sanity_pruned10_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_random_sanity_20260914
echo START /localhome/mirzaei/qm9_cp_smoothed_random_sanity_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/qm9_cp_smoothed_random_sanity_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_random_sanity_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_sanity_20260914
echo START /localhome/mirzaei/qm9_cp_smoothed_sanity_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/qm9_cp_smoothed_sanity_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_cp_smoothed_sanity_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_seed_fix_backup_20260916
echo START /localhome/mirzaei/qm9_defog_seed_fix_backup_20260916
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/qm9_defog_seed_fix_backup_20260916/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-17/qm9_defog_seed_fix_backup_20260916/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_common_eval_20260917
echo START /local-scratch2/mirzaei/qm9_common_eval_20260917
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/qm9_common_eval_20260917/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_common_eval_20260917/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_comparable_20260915
echo START /local-scratch/localhome/mirzaei/qm9_comparable_20260915
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/qm9_comparable_20260915/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_comparable_20260915/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_defog_campaign_20260914
echo START /local-scratch/localhome/mirzaei/qm9_defog_campaign_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/qm9_defog_campaign_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_defog_campaign_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_launcher_fix
echo START /local-scratch/localhome/mirzaei/qm9_launcher_fix
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/qm9_launcher_fix/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_launcher_fix/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_comparable_20260915
echo START /localhome/mirzaei/qm9_comparable_20260915
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/localhome/mirzaei/qm9_comparable_20260915/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_comparable_20260915/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_defog_campaign_20260914
echo START /localhome/mirzaei/qm9_defog_campaign_20260914
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/localhome/mirzaei/qm9_defog_campaign_20260914/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_defog_campaign_20260914/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_launcher_fix
echo START /localhome/mirzaei/qm9_launcher_fix
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/localhome/mirzaei/qm9_launcher_fix/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/QM9/sources/cs-cl-18/qm9_launcher_fix/ || failed=$((failed+1))
echo "FAILED_TRANSFERS=$failed"
if [ "$failed" -eq 0 ]; then date -Is > "$root/manifests/TRANSFER_COMPLETE"; else date -Is > "$root/manifests/TRANSFER_INCOMPLETE"; fi
