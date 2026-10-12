"""Build one self-contained input YAML per GraphVAE paper run.

input YAML = the run's original base config file (from the exact code version it ran with)
           + the run's original command-line flags (kept as strings, so argparse converts them exactly as on the
             command line; argparse applies `type` to string defaults).

usage: make_input_yamls.py <paper_runs_dir> <package_code_graphvae_dir> <out_dir>
  paper_runs_dir: configs/paper_runs (holds the effective seed_N.yaml files with the original command in the header)
"""
import glob, os, re, shlex, sys
import yaml

RUNS, CODE, OUT = sys.argv[1:4]


def flatten(cfg):
    flat = {}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                assert k2 not in flat, k2
                flat[k2] = v2
        else:
            assert k not in flat, k
            flat[k] = v
    return flat


def option_dests(main_py):
    """Map each command-line flag name to the setting (dest) it sets, from main.py's add_argument calls."""
    src = open(main_py).read(); out = {}
    for m in re.finditer(r"add_argument\(\s*((?:['\"]-[-\w]+['\"]\s*,\s*)+)(.*?)\)\s*\n", src, re.S):
        opts = re.findall(r"['\"]-{1,2}([\w-]+)['\"]", m.group(1))
        dm = re.search(r"dest\s*=\s*['\"](\w+)['\"]", m.group(2))
        long = [o for o in opts if len(o) > 1] or opts
        dest = dm.group(1) if dm else long[0].replace("-", "_")
        for o in opts:
            out[o] = dest
    return out


n = 0
for eff in sorted(glob.glob(os.path.join(RUNS, "*", "graphvae*", "seed_*.yaml"))):
    head = open(eff).read().split("\n")[:8]
    commit = re.search(r"/commit/(\w+)", "\n".join(head)).group(1)
    cmd = [l for l in head if l.startswith("# Original launch command:")][0].split(":", 1)[1].strip()
    toks = shlex.split(cmd)
    assert toks[0] == "main.py"
    flags, i = {}, 1
    while i < len(toks):
        t = toks[i]
        assert t.startswith("--"), (eff, t)
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            flags[t[2:]] = toks[i + 1]; i += 2
        else:
            raise SystemExit(f"value-less flag {t} in {eff}")
    base_rel = flags.pop("config")
    base = flatten(yaml.safe_load(open(os.path.join(CODE, commit, base_rel))))
    base.pop("config", None)
    dests = option_dests(os.path.join(CODE, commit, "main.py"))
    merged = dict(base)
    for k, v in flags.items():
        merged[dests.get(k, k.replace("-", "_"))] = v  # e.g. --model_path sets PATH, --vis_step sets Vis_step  # command-line values override the base config, as on the original command line
    ds, method = eff.split(os.sep)[-3:-1]
    seed = re.search(r"seed_(\d+)", eff).group(1)
    out = os.path.join(OUT, ds, method, f"seed_{seed}.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# INPUT config for {ds} | {method} | seed {seed}.  Run:  python main.py --config configs/paper_runs/{ds}/{method}/seed_{seed}.yaml\n"
                f"# Built from the original base config `{base_rel}` (code {commit}) + the original command-line flags.\n"
                f"# Paths below are those of the original machine; pass --graph_save_path / --dataset_cache_dir / --motif_cache_dir\n"
                f"# (and --data_dir, --model_path where present) on the command line to point them elsewhere.\n"
                f"# Command-line values are kept as strings: main.py converts them exactly as it did on the command line.\n")
        yaml.safe_dump(merged, f, sort_keys=True, default_flow_style=False)
    n += 1
print("input YAMLs written:", n)
