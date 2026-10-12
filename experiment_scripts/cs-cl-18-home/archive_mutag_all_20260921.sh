#!/usr/bin/env bash
set -u
root=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG
mkdir -p "$root/manifests" "$root/reports"
exec >> "$root/manifests/transfer.log" 2>&1
failed=0
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_edgefeat_motif025_3seed_20260912
echo START /local-scratch2/mirzaei/mutag_edgefeat_motif025_3seed_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_edgefeat_motif025_3seed_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_edgefeat_motif025_3seed_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_motif_weight_15_20260913
echo START /local-scratch2/mirzaei/mutag_motif_weight_15_20260913
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_motif_weight_15_20260913/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_motif_weight_15_20260913/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_motif_weight_075_20260912
echo START /local-scratch2/mirzaei/mutag_motif_weight_075_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_motif_weight_075_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_motif_weight_075_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_nodefeat_3way_20260910
echo START /local-scratch2/mirzaei/mutag_nodefeat_3way_20260910
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_nodefeat_3way_20260910/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_nodefeat_3way_20260910/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_topology_aware_20260912
echo START /local-scratch2/mirzaei/mutag_topology_aware_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_topology_aware_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_topology_aware_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_edgefeat_motif001_3seed_20260911
echo START /local-scratch2/mirzaei/mutag_edgefeat_motif001_3seed_20260911
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_edgefeat_motif001_3seed_20260911/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_edgefeat_motif001_3seed_20260911/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_edgefeat_motif02_3seed_20260911
echo START /local-scratch2/mirzaei/mutag_edgefeat_motif02_3seed_20260911
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_edgefeat_motif02_3seed_20260911/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_edgefeat_motif02_3seed_20260911/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_edgefeat_campaign_20260910
echo START /local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_edgefeat_campaign_20260910/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_defog_edgefeat_seeds_20260913
echo START /local-scratch2/mirzaei/mutag_defog_edgefeat_seeds_20260913
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_defog_edgefeat_seeds_20260913/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_defog_edgefeat_seeds_20260913/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_edgefeat_weight_sweep_20260911
echo START /local-scratch2/mirzaei/mutag_edgefeat_weight_sweep_20260911
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_edgefeat_weight_sweep_20260911/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_edgefeat_weight_sweep_20260911/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_defog_replacement_seed3_20260912
echo START /local-scratch2/mirzaei/mutag_defog_replacement_seed3_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_defog_replacement_seed3_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-09/mutag_defog_replacement_seed3_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_edgefeat_motif025_3seed_20260912
echo START /local-scratch2/mirzaei/mutag_edgefeat_motif025_3seed_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_edgefeat_motif025_3seed_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_edgefeat_motif025_3seed_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_edgefeat_motif02_3seed_20260911
echo START /local-scratch2/mirzaei/mutag_edgefeat_motif02_3seed_20260911
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_edgefeat_motif02_3seed_20260911/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_edgefeat_motif02_3seed_20260911/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_edgefeat_motif001_3seed_20260911
echo START /local-scratch2/mirzaei/mutag_edgefeat_motif001_3seed_20260911
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_edgefeat_motif001_3seed_20260911/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_edgefeat_motif001_3seed_20260911/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_edgefeat_weight_sweep_20260911
echo START /local-scratch2/mirzaei/mutag_edgefeat_weight_sweep_20260911
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_edgefeat_weight_sweep_20260911/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_edgefeat_weight_sweep_20260911/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_weight_sweep_collected_20260912
echo START /local-scratch2/mirzaei/mutag_weight_sweep_collected_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_weight_sweep_collected_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_weight_sweep_collected_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_defog_edgefeat_seeds_20260913
echo START /local-scratch2/mirzaei/mutag_defog_edgefeat_seeds_20260913
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_defog_edgefeat_seeds_20260913/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_defog_edgefeat_seeds_20260913/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_defog_replacement_seed3_20260912
echo START /local-scratch2/mirzaei/mutag_defog_replacement_seed3_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_defog_replacement_seed3_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_defog_replacement_seed3_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_edgefeat_campaign_20260910
echo START /local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_edgefeat_campaign_20260910/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_nodefeat_3way_20260910
echo START /local-scratch2/mirzaei/mutag_nodefeat_3way_20260910
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_nodefeat_3way_20260910/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_nodefeat_3way_20260910/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_threshold_calibration_20260912
echo START /local-scratch2/mirzaei/mutag_threshold_calibration_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_threshold_calibration_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_threshold_calibration_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_motif025_collected_20260912
echo START /local-scratch2/mirzaei/mutag_motif025_collected_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_motif025_collected_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_motif025_collected_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_new_results_20260911
echo START /local-scratch2/mirzaei/mutag_new_results_20260911
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_new_results_20260911/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_new_results_20260911/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_topology_aware_20260912
echo START /local-scratch2/mirzaei/mutag_topology_aware_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_topology_aware_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_topology_aware_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_defog_edgeaware_correlation_20260913
echo START /local-scratch2/mirzaei/mutag_defog_edgeaware_correlation_20260913
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/mutag_defog_edgeaware_correlation_20260913/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_defog_edgeaware_correlation_20260913/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_ptc_seed2_configs
echo START /local-scratch/localhome/mirzaei/mutag_ptc_seed2_configs
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/mutag_ptc_seed2_configs/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_ptc_seed2_configs/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_outlier_check
echo START /local-scratch/localhome/mirzaei/mutag_outlier_check
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/mutag_outlier_check/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_outlier_check/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_nodefeat_3way_20260910
echo START /local-scratch/localhome/mirzaei/mutag_nodefeat_3way_20260910
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/mutag_nodefeat_3way_20260910/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_nodefeat_3way_20260910/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_ptc_seed2_configs
echo START /localhome/mirzaei/mutag_ptc_seed2_configs
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/localhome/mirzaei/mutag_ptc_seed2_configs/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_ptc_seed2_configs/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_outlier_check
echo START /localhome/mirzaei/mutag_outlier_check
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/localhome/mirzaei/mutag_outlier_check/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_outlier_check/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_nodefeat_3way_20260910
echo START /localhome/mirzaei/mutag_nodefeat_3way_20260910
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/localhome/mirzaei/mutag_nodefeat_3way_20260910/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-18/mutag_nodefeat_3way_20260910/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_defog_independent_20260919_invalid_pre_seedfix_20260920
echo START /local-scratch2/mirzaei/mutag_defog_independent_20260919_invalid_pre_seedfix_20260920
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/mutag_defog_independent_20260919_invalid_pre_seedfix_20260920/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_defog_independent_20260919_invalid_pre_seedfix_20260920/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_edgefeat_motif02_3seed_20260911
echo START /local-scratch2/mirzaei/mutag_edgefeat_motif02_3seed_20260911
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/mutag_edgefeat_motif02_3seed_20260911/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_edgefeat_motif02_3seed_20260911/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_edgefeat_motif001_3seed_20260911
echo START /local-scratch2/mirzaei/mutag_edgefeat_motif001_3seed_20260911
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/mutag_edgefeat_motif001_3seed_20260911/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_edgefeat_motif001_3seed_20260911/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_defog_edgeaware_correlation_20260913
echo START /local-scratch2/mirzaei/mutag_defog_edgeaware_correlation_20260913
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/mutag_defog_edgeaware_correlation_20260913/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_defog_edgeaware_correlation_20260913/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_defog_independent_20260919
echo START /local-scratch2/mirzaei/mutag_defog_independent_20260919
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/mutag_defog_independent_20260919/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_defog_independent_20260919/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_defog_edgefeat_seeds_20260913
echo START /local-scratch2/mirzaei/mutag_defog_edgefeat_seeds_20260913
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/mutag_defog_edgefeat_seeds_20260913/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_defog_edgefeat_seeds_20260913/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_edgefeat_campaign_20260910
echo START /local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_edgefeat_campaign_20260910/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_edgefeat_weight_sweep_20260911
echo START /local-scratch2/mirzaei/mutag_edgefeat_weight_sweep_20260911
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/mutag_edgefeat_weight_sweep_20260911/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_edgefeat_weight_sweep_20260911/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_nodefeat_3way_20260910
echo START /local-scratch2/mirzaei/mutag_nodefeat_3way_20260910
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/mutag_nodefeat_3way_20260910/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_nodefeat_3way_20260910/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_topology_aware_20260912
echo START /local-scratch2/mirzaei/mutag_topology_aware_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/mutag_topology_aware_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_topology_aware_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_edgefeat_motif025_3seed_20260912
echo START /local-scratch2/mirzaei/mutag_edgefeat_motif025_3seed_20260912
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/mutag_edgefeat_motif025_3seed_20260912/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/cs-cl-19/mutag_edgefeat_motif025_3seed_20260912/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/gather
echo START /local-scratch2/new/gather/datasets/mutag
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/new/gather/datasets/mutag/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/sources/gather/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/reports
echo START /local-scratch2/mirzaei/DATASET_REPORTS_ALL3_20260917/MUTAG_ALL_RESULTS_20260917.md
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/DATASET_REPORTS_ALL3_20260917/MUTAG_ALL_RESULTS_20260917.md /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/reports/MUTAG_ALL_RESULTS_20260917.md || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/reports
echo START /local-scratch2/mirzaei/ALL_DATASET_MD_REPORTS_20260914/MUTAG_ALL_RESULTS_VERIFIED_20260914.md
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/ALL_DATASET_MD_REPORTS_20260914/MUTAG_ALL_RESULTS_VERIFIED_20260914.md /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/MUTAG/reports/MUTAG_ALL_RESULTS_VERIFIED_20260914.md || failed=$((failed+1))
echo "FAILED_TRANSFERS=$failed"
if [ "$failed" -eq 0 ]; then date -Is > "$root/manifests/TRANSFER_COMPLETE"; else date -Is > "$root/manifests/TRANSFER_INCOMPLETE"; fi
