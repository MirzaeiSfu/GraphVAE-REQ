"""Run one paper experiment from its YAML config, inside a clone of this repository.

usage: python scripts/retrain/run_experiment.py <DATASET>/<method>/seed_N --data DATA_DIR --out OUT_DIR
                                                [--ckpt CHECKPOINT] [--dry-run]

Run it with the Python environment of the method (GraphVAE: `micro`, DeFoG: its venv, LGD: `LGD_FIX_20260924/env`).
DATA_DIR holds the data that is not in git (dataset/motif caches, frozen DeFoG splits, LGD datasets); launch.py fills
it from the reproduction package. Settings come only from the YAML files in configs/paper_runs; the command line
adds nothing but paths.

  GraphVAE:  python main.py --config configs/paper_runs/<id>.yaml  (+ output, cache and data paths)
  DeFoG:     python src/main.py <overrides from training_overrides.yaml>, then the same for generation
  LGD:       pretrain.py --cfg encoder.yaml -> select_best_encoder_ckpt.py -> train_diffusion.py --cfg diffusion.yaml
             -> sample_tu.py --cfg diffusion.yaml
"""
import argparse, glob, os, shlex, subprocess, sys
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(cmd, cwd, env, dry, capture=False):
    print(f"\n$ (cd {cwd}) " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry:
        return ""
    if capture:
        return subprocess.run(cmd, cwd=cwd, env=env, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
    subprocess.run(cmd, cwd=cwd, env=env, check=True)
    return ""


def resolve(value, data, out):
    if value.startswith("@OUT"):
        return out + value[4:]
    if value.startswith("@REPO"):
        return REPO + value[5:]
    return os.path.join(data, value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id"); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", help="DeFoG: generate from this checkpoint instead of the one trained here")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    plan = yaml.safe_load(open(os.path.join(REPO, "scripts", "retrain", "retrain_plan.yaml")))
    r = plan["runs"][a.run_id]
    data, out = os.path.abspath(a.data), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    py, env = sys.executable, dict(os.environ)
    code = os.path.join(REPO, r["code"])
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip()
    print(f"run {a.run_id} | kind {r['kind']} | repository commit {commit} | code {r['code']} | python {py}", flush=True)

    if r["kind"] == "graphvae":
        if r.get("vendor"):  # packages the original Solar runs loaded from vendor/python
            env["PYTHONPATH"] = os.path.join(data, "vendor/python") + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [py, "main.py", "--config", r["config"], "--graph_save_path", out]
        for k, v in r["paths"].items():
            cmd += [f"--{k}", resolve(v, data, out)]
        run(cmd, code, env, a.dry_run)

    elif r["kind"] == "defog":
        # same PYTHONPATH as the original launcher of this run (frozen runs need GraphVAE-REQ's ggm_eval)
        pp = [x.replace("@REPO", REPO).replace("@CODE", code) for x in r["pythonpath"]]
        env["PYTHONPATH"] = os.pathsep.join(pp + [env.get("PYTHONPATH", "")])
        print("PYTHONPATH=" + env["PYTHONPATH"], flush=True)
        for stage, ofile in r["stages"].items():
            overrides = []
            for o in yaml.safe_load(open(os.path.join(REPO, ofile))) or []:
                k = o.split("=", 1)[0]
                if r.get("dataset_key") and k == r["dataset_key"]:
                    o = f"{k}={os.path.join(data, r['dataset_path'])}"
                elif k == "general.test_only":
                    ckpt = a.ckpt or (sorted(glob.glob(f"{out}/training/checkpoints/*/*.ckpt"), key=os.path.getmtime)
                                      or [f"{out}/training/checkpoints/<trained checkpoint>"])[-1]
                    o = f"general.test_only={ckpt}"
                elif k.startswith("hydra.run.dir"):
                    continue
                overrides.append(o)
            run([py, "src/main.py"] + overrides + [f"hydra.run.dir={out}/{stage}"], code, env, a.dry_run)

    elif r["kind"] == "lgd":
        env["PYTHONPATH"] = code + os.pathsep + env.get("PYTHONPATH", "")
        common = ["dataset.dir", os.path.join(data, r["dataset_dir"]), "accelerator", "cuda:0", "wandb.use", "False"]
        run([py, "pretrain.py", "--cfg", os.path.join(REPO, r["encoder_config"]), "--repeat", "1",
             "out_dir", f"{out}/encoder"] + common, code, env, a.dry_run)
        ckpt_dirs = glob.glob(f"{out}/encoder/**/ckpt", recursive=True)
        enc_dir = os.path.dirname(ckpt_dirs[0]) if ckpt_dirs else f"{out}/encoder/<run>"
        enc = run([py, "select_best_encoder_ckpt.py", enc_dir], code, env, a.dry_run, capture=True) or "<selected encoder>"
        print(f"selected encoder checkpoint (lowest validation loss_recon): {enc}", flush=True)
        for script in ("train_diffusion.py", r["sampler"]):
            extra = ["--run-id", str(r["seed"])] if "sample_tu" in script else ["--repeat", "1"]
            run([py, script, "--cfg", os.path.join(REPO, r["diffusion_config"])] + extra +
                ["out_dir", f"{out}/diffusion"] + common + ["diffusion.first_stage_config", enc], code, env, a.dry_run)
    print(f"\ndone: {a.run_id} -> {out}", flush=True)


if __name__ == "__main__":
    main()
