"""Write scripts/retrain/retrain_plan.yaml: for every paper run, its config files, the data it needs that is not in
git, and the system it originally ran on.

usage: python scripts/retrain/make_retrain_plan.py <REPRODUCTION_PACKAGE>/RETRAIN_PLAN.yaml
The original systems are taken from the reproduction package's plan (built from job records, run logs and Solar's
job history). Data paths are relative to the data folder that launch.py fills from the package.
"""
import os, re, sys
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pkg = yaml.safe_load(open(sys.argv[1]))

# Data that is not in git: <name in the data folder> -> <path in the reproduction package>
DATA_SOURCES = {
    "motif_caches/synthetic_9f01785": "code/graphvae/9f01785b4/cache_motifs",
    "motif_caches/ptc_c10bff5": "code/graphvae/c10bff58b/cache_motifs",
    "motif_caches/code_b": "code/graphvae/af56479bb/cache_motifs",
    "vendor/python": "code/graphvae/af56479bb/vendor/python",
    "graphvae/PTC": "data/graphvae/PTC",
    "graphvae/PROTEINS": "data/graphvae/PROTEINS",
    "graphvae/QM9": "data/graphvae/QM9",
    "defog/GRID": "data/defog/GRID",
    "defog/TRIANGULAR_GRID": "data/defog/TRIANGULAR_GRID",
    "defog/LOBSTER": "data/defog/LOBSTER",
    "defog/PTC": "data/defog/PTC",
    "defog/QM9": "data/defog/QM9",
    "lgd/graphvae_req": "code/lgd/datasets/graphvae_req",
}
CODE_B_RUNS = {"PTC/graphvae_rg/seed_1", "PTC/graphvae_rg/seed_2"} | {
    f"{d}/{m}/seed_{s}" for d, m in (("PROTEINS", "graphvae_rg"), ("QM9", "graphvae"), ("QM9", "graphvae_rg")) for s in (0, 1, 2)}
DEFOG_CODE = {"QM9": "third_party/defog", "PROTEINS": "third_party/defog_proteins"}
# PYTHONPATH exactly as each original DeFoG launcher set it (@REPO = repository root, @CODE = the DeFoG folder):
# frozen-protocol runs: GraphVAE-REQ's run_defog_job.py; QM9: qm9_defog_lab24 run script; PROTEINS: launch_defog_proteins.sh
DEFOG_PYTHONPATH = {"QM9": ["@CODE/src", "@CODE"], "PROTEINS": ["@CODE", "@CODE/src"]}
DEFOG_PYTHONPATH_FROZEN = ["@REPO/graph_evaluation/src", "@CODE"]


def graphvae_paths(ds, method, s):
    """Command-line path settings, as {setting: value}; values are data-folder paths, or @OUT / @REPO prefixed."""
    if ds in ("GRID", "TRIANGULAR_GRID", "LOBSTER"):
        sub = "linkcorr_motif_true" if method == "graphvae_rg" else "multihop_smoothed"
        return {"motif_cache_dir": f"motif_caches/synthetic_9f01785/{sub}"}
    if (ds, method) == ("PTC", "graphvae"):
        return {"dataset_cache_dir": f"graphvae/PTC/graphvae/seed_{s}/dataset_cache",
                "motif_cache_dir": "motif_caches/ptc_c10bff5", "model_path": "@OUT/best_model"}
    if (ds, method) == ("PTC", "graphvae_rg"):
        sub = "seed_0" if s == 0 else "seeds_1_2"
        return {"dataset_cache_dir": f"graphvae/PTC/graphvae_rg/{sub}/dataset_cache",
                "motif_cache_dir": f"graphvae/PTC/graphvae_rg/{sub}/motif_cache"}
    if (ds, method) == ("PROTEINS", "graphvae"):
        return {"dataset_cache_dir": "@OUT/dataset_cache"}
    if (ds, method) == ("PROTEINS", "graphvae_rg"):
        return {"dataset_cache_dir": f"graphvae/PROTEINS/graphvae_rg/dataset_cache/seed_{s}",
                "motif_cache_dir": "graphvae/PROTEINS/graphvae_rg/motif_cache"}
    if ds == "QM9":
        return {"dataset_cache_dir": "graphvae/QM9/dataset_cache", "data_dir": "@REPO/data_raw",
                "motif_cache_dir": "graphvae/QM9/graphvae_rg/motif_cache" if method == "graphvae_rg"
                else "motif_caches/code_b/multihop_smoothed"}
    raise KeyError((ds, method))


def data_needed(paths):
    keys = set()
    for v in paths.values():
        for name in DATA_SOURCES:
            if v == name or v.startswith(name + "/"):
                keys.add(name)
    return sorted(keys)


runs = {}
for r in pkg["runs"]:
    rid = r["id"]; ds, method, sd = rid.split("/"); s = int(sd.split("_")[1])
    system = {k: v for k, v in r["original_system"].items() if k not in ("python", "workspace", "python_env_archive")}
    entry = dict(dataset=ds, method=method, seed=s, system=system)
    if method.startswith("graphvae"):
        paths = graphvae_paths(ds, method, s)
        entry.update(kind="graphvae", code=".", config=f"configs/paper_runs/{rid}.yaml",
                     effective_config=f"configs/paper_runs/{rid}.effective.yaml", paths=paths,
                     vendor=rid in CODE_B_RUNS)
        data = data_needed(paths) + (["vendor/python"] if rid in CODE_B_RUNS else [])
    elif method == "defog":
        cdir = f"configs/paper_runs/{rid}"
        files = sorted(os.listdir(os.path.join(REPO, cdir)))
        stages = [st for st in ("training", "generation", "training_and_generation") if f"{st}_overrides.yaml" in files]
        root = {"QM9": f"defog/QM9/seed_{s}", "PROTEINS": None}.get(ds, f"defog/{ds}")
        entry.update(kind="defog", code=DEFOG_CODE.get(ds, "third_party/defog_c631697"),
                     pythonpath=DEFOG_PYTHONPATH.get(ds, DEFOG_PYTHONPATH_FROZEN),
                     stages={st: f"{cdir}/{st}_overrides.yaml" for st in stages},
                     dataset_key="dataset.datadir" if ds == "QM9" else ("dataset.root" if root else None),
                     dataset_path=root)
        data = [root] if root else []
    else:
        cdir = f"configs/paper_runs/{rid}"
        entry.update(kind="lgd", code="third_party/lgd", encoder_config=f"{cdir}/encoder.yaml",
                     diffusion_config=f"{cdir}/diffusion.yaml", dataset_dir="lgd/graphvae_req",
                     sampler="legacy_LGD_3SEED_CAMPAIGN_20260920/sample_tu.py" if rid == "PTC/lgd/seed_2" else "sample_tu.py")
        data = ["lgd/graphvae_req"]
    entry["data"] = sorted(set(data))
    runs[rid] = entry

lab = {h: dict(gpu=v["gpu"], workspace=v["workspace"].replace("REPRO_20261008", "RETRAIN"), python=v["python"])
       for h, v in pkg["lab_systems"].items()}
solar = dict(pkg["solar"]); solar["workspace"] = solar["workspace"].replace("REPRO_20261008", "RETRAIN")
solar["graphvae_env_archive"] = "/project/cs-schulte-lab/ali/micro_env.tar.gz"
plan = dict(repository="https://github.com/MirzaeiSfu/GraphVAE-REQ.git",
            data_package="cs-cl-18:/media/mirzaei/backup/REPRODUCTION_PACKAGE_20261008",
            data_sources=DATA_SOURCES, lab_systems=lab, solar=solar, runs=runs)
out = os.path.join(REPO, "scripts", "retrain", "retrain_plan.yaml")
with open(out, "w") as f:
    f.write("# Every paper run: config files, data not in git, and the system it originally ran on.\n"
            "# Generated by scripts/retrain/make_retrain_plan.py. Used by launch.py and run_experiment.py.\n")
    yaml.safe_dump(plan, f, sort_keys=False, default_flow_style=False)
print(len(runs), "runs written to", out)
