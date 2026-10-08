#!/usr/bin/env bash
set -u
root=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS
mkdir -p "$root/manifests" "$root/reports"
exec >> "$root/manifests/transfer.log" 2>&1
failed=0
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-09/aids_defog_independent_20260919
echo START /local-scratch2/mirzaei/aids_defog_independent_20260919
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/aids_defog_independent_20260919/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-09/aids_defog_independent_20260919/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-09/aids_defog_independent_20260919_invalid_pre_seedfix_20260920
echo START /local-scratch2/mirzaei/aids_defog_independent_20260919_invalid_pre_seedfix_20260920
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/aids_defog_independent_20260919_invalid_pre_seedfix_20260920/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-09/aids_defog_independent_20260919_invalid_pre_seedfix_20260920/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-18/aids_common_eval_10k_20260917
echo START /local-scratch2/mirzaei/aids_common_eval_10k_20260917
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/aids_common_eval_10k_20260917/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-18/aids_common_eval_10k_20260917/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-18/aids_false_defog_3seed_20260912
echo START /local-scratch2/mirzaei/aids_false_defog_3seed_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/aids_false_defog_3seed_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-18/aids_false_defog_3seed_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-18/collected_ptc_aids_motif_20260905
echo START /local-scratch2/mirzaei/collected_ptc_aids_motif_20260905
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/collected_ptc_aids_motif_20260905/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-18/collected_ptc_aids_motif_20260905/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-18/protein_aids_campaign_20260910
echo START /local-scratch/localhome/mirzaei/protein_aids_campaign_20260910
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/protein_aids_campaign_20260910/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-18/protein_aids_campaign_20260910/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-18/protein_aids_campaign_20260910
echo START /localhome/mirzaei/protein_aids_campaign_20260910
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/localhome/mirzaei/protein_aids_campaign_20260910/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-18/protein_aids_campaign_20260910/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-19/aids_false_defog_3seed_20260912
echo START /local-scratch2/mirzaei/aids_false_defog_3seed_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/aids_false_defog_3seed_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/cs-cl-19/aids_false_defog_3seed_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/gather
echo START /local-scratch2/new/gather/datasets/aids
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/new/gather/datasets/aids/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/sources/gather/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/reports
echo START /local-scratch2/mirzaei/DATASET_REPORTS_ALL3_20260917/AIDS_TOPOLOGY_ONLY_RESULTS_20260917.md
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/DATASET_REPORTS_ALL3_20260917/AIDS_TOPOLOGY_ONLY_RESULTS_20260917.md /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/reports/AIDS_TOPOLOGY_ONLY_RESULTS_20260917.md || failed=$((failed+1))
echo "FAILED_TRANSFERS=$failed"
if [ "$failed" -eq 0 ]; then date -Is > "$root/manifests/TRANSFER_COMPLETE"; else date -Is > "$root/manifests/TRANSFER_INCOMPLETE"; fi
