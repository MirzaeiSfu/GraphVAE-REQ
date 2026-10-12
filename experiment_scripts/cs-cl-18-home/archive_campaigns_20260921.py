"""Discover scoped campaign artifacts and copy them in per-dataset tmux jobs."""
import json, subprocess, shlex
from pathlib import Path
ROOT='/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921'
datasets=['OGB','AIDS','QM9','PTC','MUTAG']
hosts=['09','13','16','17','18','19','26']
inventory=[]
for host in hosts:
    cmd="find /local-scratch2/mirzaei /local-scratch/mirzaei /local-scratch/localhome/mirzaei /localhome/mirzaei -mindepth 1 -maxdepth 1 -type d 2>/dev/null"
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',f'mirzaei@cs-cl-{host}.cmpt.sfu.ca',cmd],capture_output=True,text=True)
    for src in p.stdout.splitlines():
        name=Path(src).name.lower()
        for ds in datasets:
            token=ds.lower()
            if token in name and ('lgd' not in name) and not name.startswith('.'):
                inventory.append(dict(dataset=ds,host=host,source=src,destination=f'{ROOT}/{ds}/sources/cs-cl-{host}/{Path(src).name}'))
# PTC's curated archive is authoritative; the top-level scan includes it.
for ds in datasets:
    gather=f'/local-scratch2/new/gather/datasets/{ds.lower()}'
    if Path(gather).exists(): inventory.append(dict(dataset=ds,host='18',source=gather,destination=f'{ROOT}/{ds}/sources/gather'))
for ds in datasets:
    for directory in ['/local-scratch2/mirzaei/DATASET_REPORTS_ALL3_20260917','/local-scratch2/mirzaei/ALL_DATASET_MD_REPORTS_20260914']:
        for report in Path(directory).glob('*.md'):
            if ds.lower() in report.name.lower(): inventory.append(dict(dataset=ds,host='18',source=str(report),destination=f'{ROOT}/{ds}/reports/{report.name}'))
Path('archive_inventory_20260921.json').write_text(json.dumps(inventory,indent=2))
for ds in datasets:
    rows=[x for x in inventory if x['dataset']==ds]
    script=['#!/usr/bin/env bash','set -u',f'root={ROOT}/{ds}','mkdir -p "$root/manifests" "$root/reports"','exec >> "$root/manifests/transfer.log" 2>&1','failed=0']
    # Avoid duplicating environment installations, raw source datasets and git object stores.
    excludes=['env/','envs/','venv/','.venv/','.git/','__pycache__/','*.tar.gz','source/DeFoG/data/','source/GraphVAE-REQ/datasets/','source/GraphVAE-REQ/cache/']
    for row in rows:
        src=row['source'];dest=row['destination'];isfile=src.endswith('.md')
        script += [f'mkdir -p {shlex.quote(str(Path(dest).parent) if isfile else dest)}',f'echo START {shlex.quote(src)}','if [ "$(df -Pk /local-scratch2 | awk \'NR==2 {print $4}\')" -lt 20971520 ]; then echo LOW_DISK_STOP; exit 2; fi']
        remote=src if row['host']=='19' else f'mirzaei@cs-cl-{row["host"]}.cmpt.sfu.ca:{src}'
        command=['rsync','-a','--partial','--stats']
        command += [f'--exclude={x}' for x in excludes]
        command += [remote+('' if isfile else '/'),dest+('' if isfile else '/')]
        script += [shlex.join(command)+' || failed=$((failed+1))']
    script += ['echo "FAILED_TRANSFERS=$failed"','if [ "$failed" -eq 0 ]; then date -Is > "$root/manifests/TRANSFER_COMPLETE"; else date -Is > "$root/manifests/TRANSFER_INCOMPLETE"; fi']
    Path(f'archive_{ds.lower()}_all_20260921.sh').write_text('\n'.join(script)+'\n')
    print(ds,len(rows))
