#!/usr/bin/env python3
"""Re-run the frozen-benchmark DeFoG generation for one checkpoint.

This reproduces exactly the command built by
GraphVAE-REQ/baselines/defog/frozen_eval/run_defog_job.py::generate()
(branch feat/defog-fair-benchmark, commit 87c8167; DeFoG commit c631697),
i.e. `src/main.py` with the same Hydra overrides, except for:
  * general.generation_seed         (--gen-seed, original 12345)
  * general.final_model_samples_to_generate (--n, original 20)
  * train.batch_size                (--train-batch-size; ONLY used at test time
                                     as the sampling batch = 2*train.batch_size;
                                     original GRID/TRI 1 -> 2, LOBSTER 4 -> 8)
  * general.name / hydra.run.dir / dataset.root (bookkeeping paths only).

The only instrumentation is a transparent wrapper around
GraphDiscreteFlowModel.sample_batch that appends every raw batch (before the
strict acceptance filter) plus its wall time to `progress_batches.pkl`; it does
not touch any random number generator.
"""
import argparse
import json
import os
import pickle
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--defog-root", required=True)
p.add_argument("--graph-eval-src", required=True)
p.add_argument("--data-root", required=True, help="dir with real_{train,validation,test}_graphs.pt")
p.add_argument("--dataset", required=True, choices=["GRID", "TRIANGULAR_GRID", "LOBSTER"])
p.add_argument("--train-seed", type=int, required=True)
p.add_argument("--ckpt", required=True)
p.add_argument("--gen-seed", type=int, required=True)
p.add_argument("--n", type=int, required=True)
p.add_argument("--train-batch-size", type=int, default=None)
p.add_argument("--out-dir", required=True)
args = p.parse_args()

SPEC = {  # from baselines/defog/frozen_eval/manifest.yaml + campaign.yaml
    "GRID": dict(experiment="grid", batch_size=1,
                 train="c821ad9fa50a3c4e39ff73ab009221dfad88e72b34c5456f94ea2cb2c1c8b74b",
                 val="8099953cc34770cb2b5a9d2af1bd04baaaf61a0a0d78fc538eea2ef55c85e363",
                 ref="d6f00e40b315e8556881844897534f515b14ee74e53f6d320e8c80fbd7e32566"),
    "TRIANGULAR_GRID": dict(experiment="grid", batch_size=1,
                 train="3e66eb440731667af0fb4af361b77ef8291dc8b4951a8e98b083176b236d5874",
                 val="35bd06e8a3528b56cb3a66ee56a47a494286dfe9f5580fa4d4076df6783d3e60",
                 ref="750cb77476e7246171ca59d6fba3794ce891ee41c49049ed9bbdae9f8adfe638"),
    "LOBSTER": dict(experiment="tree", batch_size=4,
                 train="3ddb284ca5ef7cb60065b427b610966c57abc189507a952abf86d157e2b33b6f",
                 val="b8c54f153f989a454079b338ff45071c89d595610b9e3552d6547ccb6e3fdadb",
                 ref="9fabc272cbec32dcc53ca720343635b39c8736e1939fab2585dd574a27f381da"),
}
s = SPEC[args.dataset]
tbs = s["batch_size"] if args.train_batch_size is None else args.train_batch_size
ds_l = args.dataset.lower()
out = os.path.abspath(args.out_dir)
os.makedirs(out, exist_ok=True)
overrides = [
    f"+experiment={s['experiment']}",
    "dataset=frozen_graphvae",
    f"dataset.identity={args.dataset}",
    f"dataset.root={os.path.abspath(args.data_root)}",
    f"dataset.train_sha256={s['train']}",
    f"dataset.validation_sha256={s['val']}",
    f"dataset.reference_sha256={s['ref']}",
    "dataset.feature_mode=topology_control",
    'dataset.feature_schema="constant-node|export=topology_control"',
    f"train.seed={args.train_seed}",
    f"train.batch_size={tbs}",
    "general.wandb=disabled",
    "general.validation_selection=loss",
    f"general.generation_seed={args.gen_seed}",
    "general.strict_generation=true",
    "general.defog_commit=c631697b9cd5a2474d22ba12de33943c6b49e53e",
    f"general.name=generate_{ds_l}_seed_{args.train_seed}",
    f"general.test_only={os.path.abspath(args.ckpt)}",
    f"general.final_model_samples_to_generate={args.n}",
    "general.final_model_samples_to_save=0",
    "general.final_model_chains_to_save=0",
    f"hydra.run.dir={out}",
]
with open(os.path.join(out, "synfix_invocation.json"), "w") as f:
    json.dump({"argv": sys.argv, "overrides": overrides, "sampling_batch": 2 * tbs,
               "host": os.uname().nodename,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "started": time.time()}, f, indent=2)

defog_root = os.path.abspath(args.defog_root)
src = os.path.join(defog_root, "src")
sys.path[:0] = [src, defog_root, os.path.abspath(args.graph_eval_src)]
os.chdir(defog_root)
try:  # same import order as src/main.py
    import graph_tool.all  # noqa: F401
except ImportError:
    pass
import runpy  # noqa: E402
import graph_discrete_flow_model as gdfm  # noqa: E402

_orig = gdfm.GraphDiscreteFlowModel.sample_batch
_progress = os.path.join(out, "progress_batches.pkl")


def _wrapped(self, *a, **k):
    t0 = time.time()
    res = _orig(self, *a, **k)
    dt = time.time() - t0
    mol, _lab = res
    rec = {"batch_id": a[0] if a else k.get("batch_id"), "seconds": dt,
           "samples": [[x.clone(), e.clone()] for x, e in mol]}
    with open(_progress, "ab") as f:
        pickle.dump(rec, f)
    print(f"\n[synfix] batch {rec['batch_id']} size {len(mol)} "
          f"n={[int(x.shape[0]) for x, _ in mol]} {dt:.1f}s", flush=True)
    return res


gdfm.GraphDiscreteFlowModel.sample_batch = _wrapped
sys.argv = [os.path.join(src, "main.py")] + overrides
runpy.run_path(os.path.join(src, "main.py"), run_name="__main__")
