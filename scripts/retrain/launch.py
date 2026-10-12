#!/usr/bin/env python3
"""Retrain paper experiments on the system each one originally ran on, from a fresh clone of this repository.

Run on cs-cl-18 (the reproduction package with the data that is not in git is on its external drive):

    python3 scripts/retrain/launch.py <id | prefix | all> [--ref REF] [--dry-run] [--no-start] [--host HOST] [--gpu N]

<id> is a run in scripts/retrain/retrain_plan.yaml, e.g. PTC/graphvae_rg/seed_1; a prefix like PTC/ or GRID/defog/
selects several; `all` selects every run. For each run this script:
  1. resolves --ref (default: paper-reproduction-v1) to a commit and clones the repository at that commit on the
     original system: <workspace>/GraphVAE-REQ_<commit>, reused if it is already there;
  2. copies the data that run needs from the reproduction package to <workspace>/data (rsync; unchanged files skipped);
  3. starts scripts/retrain/run_experiment.py in the clone, with the Python environment the original run used:
     lab machine -> tmux session retrain_<DATASET>_<method>_seed_N on the least-used GPU;
     Solar       -> sbatch with the original partition, node, GPU type, CPUs, memory and time limit.
Outputs go to <workspace>/outputs/<id>/ and logs to <workspace>/logs/ on that system. Each launch is appended to
~/retrain_launch_log.tsv on the machine you launch from.
"""
import argparse, datetime, os, shlex, subprocess, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = yaml.safe_load(open(os.path.join(HERE, "retrain_plan.yaml")))
SOLAR = ["ssh", "-p", "24", "-o", "BatchMode=yes", "mirzaei@solar.cs.sfu.ca"]
LOG = os.path.expanduser("~/retrain_launch_log.tsv")


def ssh(sysd):
    return SOLAR if sysd["type"] == "solar" else ["ssh", "-o", "BatchMode=yes", f"mirzaei@{sysd['host']}.cmpt.sfu.ca"]


def sh(cmd, dry, **kw):
    print("   $", " ".join(shlex.quote(c) for c in cmd)[:400])
    if dry:
        return ""
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout


def system_for(rid, r, args):
    s = dict(r["system"])
    if s["type"] == "unknown":
        if not args.host:
            raise SystemExit(f"{rid}: original system unknown ({s.get('note')}); pass --host cs-cl-XX")
        s = dict(type="lab", host=args.host, gpu=PLAN["lab_systems"][args.host]["gpu"])
    kind = r["kind"]
    if s["type"] == "lab":
        lab = PLAN["lab_systems"][s["host"]]
        s.update(workspace=lab["workspace"], python=lab["python"][kind])
    else:
        s.update(workspace=PLAN["solar"]["workspace"], python=PLAN["solar"]["python"][kind])
    return s


def launch(rid, r, commit, args):
    s = system_for(rid, r, args)
    ws, tag = s["workspace"], "retrain_" + rid.replace("/", "_")
    clone = f"{ws}/GraphVAE-REQ_{commit[:9]}"
    print(f"== {rid} -> {s['type']} {s.get('host')} {s.get('node', '')} ({s.get('gpu')}) | commit {commit[:9]}")
    # 1. clone the code at the pinned commit (LFS objects are not needed for training)
    sh(ssh(s) + [f"mkdir -p {ws}/data {ws}/outputs {ws}/logs {ws}/sbatch && if [ ! -d {clone}/.git ]; then "
                 f"GIT_LFS_SKIP_SMUDGE=1 git clone -q {PLAN['repository']} {clone} && "
                 f"git -C {clone} -c advice.detachedHead=false checkout -q {commit}; fi && "
                 f"test \"$(git -C {clone} rev-parse HEAD)\" = {commit} && echo clone-ok"], args.dry_run)
    # 2. data that is not in git
    pkg = PLAN["data_package"].split(":", 1)[1]
    for name in r["data"]:
        src = f"{pkg}/{PLAN['data_sources'][name]}/"
        dst = f"{ws}/data/{name}/"
        e = ["-e", "ssh -p 24 -o BatchMode=yes"] if s["type"] == "solar" else []
        host = "mirzaei@solar.cs.sfu.ca" if s["type"] == "solar" else f"mirzaei@{s['host']}.cmpt.sfu.ca"
        sh(ssh(s) + [f"mkdir -p {dst}"], args.dry_run)
        sh(["rsync", "-rlt", "--no-perms", "--no-owner", "--no-group"] + e + [src, f"{host}:{dst}"], args.dry_run)
    if args.no_start:
        print("   (--no-start: code and data are in place; not starting)"); return
    out, py = f"{ws}/outputs/{rid}", s["python"]
    run = f"cd {clone} && {py} scripts/retrain/run_experiment.py {rid} --data {ws}/data --out {out}"
    if s["type"] == "lab":
        gpu = args.gpu
        if gpu is None and not args.dry_run:
            rows = sh(ssh(s) + ["nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"], False)
            gpu = sorted((int(m), i.strip()) for i, m in (l.split(",") for l in rows.strip().splitlines()))[0][1]
        inner = (f"export PATH={os.path.dirname(py)}:$PATH CUDA_VISIBLE_DEVICES={gpu or 0}; "
                 f"{{ echo host=$(hostname) gpu={gpu or 0} start=$(date -Is); {run}; echo exit=$? end=$(date -Is); }} "
                 f"2>&1 | tee {ws}/logs/{tag}.log; exec bash")
        sh(ssh(s) + [f"test -x {py} || {{ echo missing {py}; exit 3; }}; tmux new-session -d -s {tag} {shlex.quote(inner)}"], args.dry_run)
        where = f"{s['host']} tmux {tag} gpu {gpu or 0}"
    else:
        lines = ["#!/usr/bin/env bash", f"#SBATCH --job-name={tag[:40]}"]
        if s["partition"] == "cs-gpu-research":
            lines.append("#SBATCH --account=cs-schulte")
        lines += [f"#SBATCH --partition={s['partition']}", f"#SBATCH --nodelist={s['node']}",
                  f"#SBATCH --gres=gpu:{s['gpu']}:1", f"#SBATCH --cpus-per-task={s['cpus']}", f"#SBATCH --mem={s['mem']}",
                  f"#SBATCH --time={s['time']}", f"#SBATCH --output={ws}/logs/%x_%j.out", "set -euo pipefail", f"PY={py}"]
        if r["kind"] == "graphvae":
            lines += ['if [[ ! -x "$PY" ]]; then  # same environment archive the original Solar runs used',
                      '  ENV=/localscratch/mirzaei/miniconda3/envs/micro_env; mkdir -p "$ENV"',
                      f'  [[ -x "$ENV/bin/python" ]] || tar -xzf {PLAN["solar"]["graphvae_env_archive"]} -C "$ENV"; PY="$ENV/bin/python"', "fi"]
        lines += ['export PATH="$(dirname "$PY"):$PATH"', 'echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"',
                  f"cd {clone}", f'"$PY" scripts/retrain/run_experiment.py {rid} --data {ws}/data --out {out}',
                  'echo "end=$(date -Is)"']
        sb = f"{ws}/sbatch/{tag}.sbatch"
        if args.dry_run:
            print("   (sbatch)\n      " + "\n      ".join(lines))
        else:
            subprocess.run(ssh(s) + [f"cat > {sb}"], input="\n".join(lines) + "\n", text=True, check=True)
        where = "solar " + sh(ssh(s) + [f"sbatch {sb}"], args.dry_run).strip()
    if not args.dry_run:
        with open(LOG, "a") as f:
            f.write(f"{datetime.datetime.now():%F %T}\t{rid}\t{commit}\t{where}\t{out}\n")
    print(f"   started: {where}\n   output:  {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("select")
    ap.add_argument("--ref", default="paper-reproduction-v1", help="branch, tag or commit to clone (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="print every step without doing it")
    ap.add_argument("--no-start", action="store_true", help="clone and copy data, but do not start the run")
    ap.add_argument("--host", help="machine for runs whose original system is unknown (PROTEINS GraphVAE)")
    ap.add_argument("--gpu", help="GPU index on a lab machine (default: the least used)")
    args = ap.parse_args()
    out = subprocess.run(["git", "ls-remote", PLAN["repository"], args.ref, f"{args.ref}^{{}}"],
                         text=True, capture_output=True).stdout.split()
    commit = out[-2] if len(out) >= 2 else args.ref  # peeled tag if present, else branch; or a raw commit hash
    if len(commit) != 40:
        sys.exit(f"cannot resolve --ref {args.ref} on {PLAN['repository']}; pass a branch, tag or full commit hash")
    sel = [k for k in PLAN["runs"] if args.select == "all" or k == args.select or k.startswith(args.select)]
    if not sel:
        sys.exit(f"no run matches {args.select!r}")
    for rid in sel:
        launch(rid, PLAN["runs"][rid], commit, args)


if __name__ == "__main__":
    main()
