"""Non-destructive incremental archive completion; run on system19 in tmux."""
import subprocess, pathlib, json, datetime
ROOT=pathlib.Path('/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921')
def run(args):
    print(' '.join(map(str,args)),flush=True)
    subprocess.run(args,check=True)
def copy(host,src,dst):
    dst.mkdir(parents=True,exist_ok=True)
    if __import__('shutil').disk_usage(ROOT).free < 25*1024**3: raise RuntimeError('Less than 25 GiB free')
    run(['rsync','-a','--partial','--timeout=120','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',
         '--exclude=env/','--exclude=envs/','--exclude=.git/','--exclude=__pycache__/',
         f'mirzaei@cs-cl-{host}.cmpt.sfu.ca:{src}/',str(dst)+'/'])
ds=__import__('sys').argv[1];dest=ROOT/ds
(dest/'manifests').mkdir(parents=True,exist_ok=True)
try:
    if ds=='PROTEINS':
        copy('09','/local-scratch2/mirzaei/PROTEINS',dest)
        paths=subprocess.check_output(['ssh','-o','BatchMode=yes','mirzaei@cs-cl-18.cmpt.sfu.ca',
            "find /local-scratch2/mirzaei -maxdepth 1 -type d -iname 'proteins*'"],text=True).splitlines()
        for src in paths: copy('18',src,dest/'sources'/'cs-cl-18'/pathlib.Path(src).name)
        for host in ['09','19']:
            if host=='19':
                run(['rsync','-a','/local-scratch2/mirzaei/proteins_defog_3seed_20260910/',str(dest/'sources'/'cs-cl-19'/'proteins_defog_3seed_20260910')+'/'])
            else: copy(host,'/local-scratch2/mirzaei/proteins_defog_3seed_20260910',dest/'sources'/f'cs-cl-{host}'/'proteins_defog_3seed_20260910')
        run(['rsync','-a','-e','ssh -o BatchMode=yes','mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/DATASET_REPORTS_ALL3_20260917/PROTEINS_ALL_RESULTS_3SEED_20260917.md',str(dest/'reports')+'/'])
    else:
        run(['bash',str(ROOT/f'archive_{ds.lower()}_all_20260921.sh')])
    (dest/'manifests'/'REFRESH_COMPLETE_20260922').write_text(datetime.datetime.now().isoformat())
except Exception as e:
    (dest/'manifests'/'REFRESH_FAILED_20260922').write_text(repr(e));raise
