import pathlib
import subprocess
import sys

import dgl
import numpy as np


root = pathlib.Path("/local-scratch2/mirzaei/archive_completion_20260922/OGB")
repo = "/local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ"
helpers = pathlib.Path("/local-scratch2/mirzaei/edge_feature_random_gin_20260917")


def call(script, args):
    subprocess.run([sys.executable, str(script), *map(str, args)], check=True)


for seed in (0, 1, 2):
    host = "13" if seed == 2 else "17"
    remote = (
        "/localhome/mirzaei/defog_ogbg_3seed_20260920/"
        f"archive_evaluation_20260922/seed_{seed}"
    )
    out = root / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for stale in (out / "FAILED", out / "COMPLETE"):
        stale.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                "rsync",
                "-a",
                "-e",
                "ssh -o BatchMode=yes",
                f"mirzaei@cs-cl-{host}.cmpt.sfu.ca:{remote}/",
                f"{out}/",
            ],
            check=True,
        )
        arrays = np.load(out / "generated_adjs.npz")
        graphs = []
        empty_count = 0
        for key in arrays.files:
            adjacency = np.asarray(arrays[key], dtype=bool)
            src, dst = np.nonzero(adjacency)
            if len(src) == 0:
                empty_count += 1
            graphs.append(
                dgl.graph(
                    (src.astype(np.int64), dst.astype(np.int64)),
                    num_nodes=adjacency.shape[0],
                )
            )
        if len(graphs) != 405:
            raise RuntimeError(f"Expected 405 graphs, found {len(graphs)}")
        dgl.save_graphs(str(out / "generated.bin"), graphs)
        (out / "conversion_summary.txt").write_text(
            f"graphs=405\nempty_edge_graphs={empty_count}\nsource=generated_adjs.npz\n"
        )
        call(
            "/local-scratch/localhome/mirzaei/evaluate_dgl_topology_gin_20260922.py",
            [
                "--repo", repo,
                "--generated", out / "generated.bin",
                "--reference", root / "reference.bin",
                "--output", out / "random_gin_all_modes.json",
                "--label", "ogb_defog",
                "--seed", seed,
                "--device", "cpu",
                "--repeats", 10,
            ],
        )
        call(
            "/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py",
            [
                "--repo", repo,
                "--generated", out / "generated.bin",
                "--reference", root / "reference.bin",
                "--output", out / "structural.json",
                "--label", "ogb_defog",
                "--seed", seed,
                "--skip-random-gin",
            ],
        )
        (out / "COMPLETE").write_text(
            "Completed structural and GIN evaluation from all 405 native adjacency outputs; "
            f"{empty_count} generated graphs had no edges. Motif TV pending verified rules.\n"
        )
    except Exception as exc:
        (out / "FAILED").write_text(repr(exc) + "\n")
        print(f"seed {seed}: {exc!r}", flush=True)
    finally:
        subprocess.run(
            [
                "rsync",
                "-a",
                "-e",
                "ssh -o BatchMode=yes",
                f"{out}/",
                f"mirzaei@cs-cl-19.cmpt.sfu.ca:/local-scratch2/mirzaei/"
                f"EXPERIMENT_ARCHIVE_20260921/OGB/evaluations/defog_seed_{seed}/",
            ],
            check=True,
        )
