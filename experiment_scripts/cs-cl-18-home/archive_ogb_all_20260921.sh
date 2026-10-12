#!/usr/bin/env bash
set -u
root=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB
mkdir -p "$root/manifests" "$root/reports"
exec >> "$root/manifests/transfer.log" 2>&1
failed=0
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-13/defog_ogbg_3seed_20260920
echo START /local-scratch/localhome/mirzaei/defog_ogbg_3seed_20260920
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-13.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/defog_ogbg_3seed_20260920/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-13/defog_ogbg_3seed_20260920/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-13/defog_ogbg_3seed_20260920
echo START /localhome/mirzaei/defog_ogbg_3seed_20260920
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-13.cmpt.sfu.ca:/localhome/mirzaei/defog_ogbg_3seed_20260920/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-13/defog_ogbg_3seed_20260920/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/defog_ogbg_3seed_20260920
echo START /local-scratch/localhome/mirzaei/defog_ogbg_3seed_20260920
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/defog_ogbg_3seed_20260920/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/defog_ogbg_3seed_20260920/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_vram_20260712_083826
echo START /local-scratch/localhome/mirzaei/ogb_vram_20260712_083826
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_vram_20260712_083826/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_table2_20260712
echo START /local-scratch/localhome/mirzaei/ogb_table2_20260712
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/ogb_table2_20260712/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_table2_20260712/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_vram_benchmark
echo START /local-scratch/localhome/mirzaei/ogb_vram_benchmark
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/ogb_vram_benchmark/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_vram_benchmark/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/defog_ogbg_3seed_20260920
echo START /localhome/mirzaei/defog_ogbg_3seed_20260920
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/defog_ogbg_3seed_20260920/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/defog_ogbg_3seed_20260920/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_vram_20260712_083826
echo START /localhome/mirzaei/ogb_vram_20260712_083826
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/ogb_vram_20260712_083826/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_vram_20260712_083826/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_table2_20260712
echo START /localhome/mirzaei/ogb_table2_20260712
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/ogb_table2_20260712/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_table2_20260712/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_vram_benchmark
echo START /localhome/mirzaei/ogb_vram_benchmark
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-17.cmpt.sfu.ca:/localhome/mirzaei/ogb_vram_benchmark/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-17/ogb_vram_benchmark/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-18/defog_ogbg_3seed_20260920
echo START /local-scratch2/mirzaei/defog_ogbg_3seed_20260920
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/defog_ogbg_3seed_20260920/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-18/defog_ogbg_3seed_20260920/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-19/defog_ogbg_3seed_20260920
echo START /local-scratch2/mirzaei/defog_ogbg_3seed_20260920
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ /local-scratch2/mirzaei/defog_ogbg_3seed_20260920/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/cs-cl-19/defog_ogbg_3seed_20260920/ || failed=$((failed+1))
mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/gather
echo START /local-scratch2/new/gather/datasets/ogb
if [ "$(df -Pk /local-scratch2 | awk 'NR==2 {print $4}')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi
rsync -a --partial --stats --exclude=env/ --exclude=envs/ --exclude=venv/ --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ '--exclude=*.tar.gz' --exclude=source/DeFoG/data/ --exclude=source/GraphVAE-REQ/datasets/ --exclude=source/GraphVAE-REQ/cache/ mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/new/gather/datasets/ogb/ /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB/sources/gather/ || failed=$((failed+1))
echo "FAILED_TRANSFERS=$failed"
if [ "$failed" -eq 0 ]; then date -Is > "$root/manifests/TRANSFER_COMPLETE"; else date -Is > "$root/manifests/TRANSFER_INCOMPLETE"; fi
