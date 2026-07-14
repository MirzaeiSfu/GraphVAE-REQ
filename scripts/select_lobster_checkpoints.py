#!/usr/bin/env python3
"""Select a Lobster checkpoint after training using fixed validation graphs.

Every periodic checkpoint is evaluated repeatedly on the validation split.  The
held-out test split is touched only once, after the global winner is selected.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from model import GraphTransformerDecoder_FC  # noqa: E402
from reproduce_table2_grid import (  # noqa: E402
    PAPER_TABLE2_BY_DATASET,
    compute_table2_metrics,
    locked_orca_tmp,
    to_graphs,
)

METRICS = ("degree", "clustering", "orbit", "spectral", "diameter")


def summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "median": float(np.median(values)), "min": float(values.min()),
        "max": float(values.max()),
    }


def checkpoints(run_dir: Path):
    paths = sorted(run_dir.glob("periodic_epoch_*.pt"))
    final = sorted(run_dir.glob("model_*_*"))
    if final:
        paths.append(final[-1])
    return paths


def load_decoder(path: Path, device: torch.device, latent_dim: int):
    state = torch.load(path, map_location="cpu")
    decoder_state = {
        key[len("decode."):]: value
        for key, value in state.items() if key.startswith("decode.")
    }
    bias_keys = [key for key in decoder_state if key.endswith("bias")]
    output_bias = max((decoder_state[key] for key in bias_keys), key=lambda x: x.numel())
    max_nodes = int(round(output_bias.numel() ** 0.5))
    decoder = GraphTransformerDecoder_FC(latent_dim, 256, max_nodes, directed=True).to(device)
    decoder.load_state_dict(decoder_state)
    decoder.eval()
    return decoder


def generate(decoder, count: int, latent_dim: int, device: torch.device, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        latent = torch.randn(count, latent_dim, generator=generator, device=device)
        matrices = (torch.sigmoid(decoder(latent)) >= 0.5).cpu().numpy()
    raw, lcc = [], []
    for matrix in matrices:
        graph = nx.Graph(nx.from_numpy_array(matrix.astype(np.int8)))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph.remove_nodes_from(list(nx.isolates(graph)))
        raw.append(graph)
        if graph.number_of_nodes():
            graph = nx.Graph(graph.subgraph(max(nx.connected_components(graph), key=len)))
        lcc.append(graph)
    return raw, lcc


def evaluate(decoder, refs, rollouts, seed, latent_dim, device, dense_threshold):
    denominators = PAPER_TABLE2_BY_DATASET["LOBSTER"]["GraphVAE"]
    rows = []
    for index in range(rollouts):
        raw, generated = generate(decoder, len(refs), latent_dim, device, seed + index)
        metric = compute_table2_metrics(refs, generated)
        score = float(np.mean([metric[name] / denominators[name] for name in METRICS]))
        edges = [graph.number_of_edges() for graph in raw]
        rows.append({
            "rollout": index, "seed": seed + index, "metrics": metric,
            "normalized_mmd": score, "mean_raw_edges": float(np.mean(edges)),
            "max_raw_edges": int(max(edges, default=0)),
            "dense_rate": float(np.mean([value > dense_threshold for value in edges])),
        })
    scores = [row["normalized_mmd"] for row in rows]
    dense_rates = [row["dense_rate"] for row in rows]
    return {
        "rollouts": rows, "score": summary(scores),
        "dense_rate": float(np.mean(dense_rates)),
        "mean_raw_edges": summary([row["mean_raw_edges"] for row in rows]),
    }


def report_markdown(payload):
    lines = [
        "# Lobster post-training checkpoint selection", "",
        "Lower is better. All candidates use validation only; held-out test is evaluated only for the winner.", "",
        "| Rank | Run | Checkpoint | Validation median | Std | Dense rate | Selection score |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    ranked = sorted(payload["candidates"], key=lambda row: row["selection_score"])
    for rank, row in enumerate(ranked, 1):
        val = row["validation"]
        lines.append(
            f"| {rank} | {row['run']} | {row['checkpoint']} | "
            f"{val['score']['median']:.6f} | {val['score']['std']:.6f} | "
            f"{val['dense_rate']:.2%} | {row['selection_score']:.6f} |"
        )
    winner = payload["winner"]
    lines += ["", "## Winner", "", f"`{winner['run']}/{winner['checkpoint']}`"]
    if "test" in winner:
        lines += ["", f"Held-out test normalized MMD: {winner['test']['score']['mean']:.6f} ± {winner['test']['score']['std']:.6f}."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-rollouts", type=int, default=10)
    parser.add_argument("--test-rollouts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--stability-weight", type=float, default=0.25)
    parser.add_argument("--dense-penalty-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    candidates = []

    with locked_orca_tmp():
        validation_paths = sorted(args.runs_root.glob("*/seed_*/validationGraphs_adj_.npy"))
        validation_paths += sorted(args.runs_root.glob("*/validationGraphs_adj_.npy"))
        for val_path in validation_paths:
            run_dir = val_path.parent
            job_dir = run_dir.parent if run_dir.name.startswith("seed_") else run_dir
            refs = to_graphs(np.load(val_path, allow_pickle=True), keep_largest_component=False)
            reference_edges = np.asarray([graph.number_of_edges() for graph in refs], dtype=float)
            dense_threshold = float(reference_edges.mean() + 3 * reference_edges.std())
            for checkpoint in checkpoints(run_dir):
                run_name = f"{job_dir.name}/{run_dir.name}" if run_dir != job_dir else job_dir.name
                print(f"[validation] {run_name}/{checkpoint.name}", flush=True)
                decoder = load_decoder(checkpoint, device, args.latent_dim)
                validation = evaluate(decoder, refs, args.validation_rollouts, args.seed,
                                      args.latent_dim, device, dense_threshold)
                selection_score = (validation["score"]["median"]
                                   + args.stability_weight * validation["score"]["std"]
                                   + args.dense_penalty_weight * validation["dense_rate"])
                candidates.append({
                    "run": run_name, "artifact_dir": str(run_dir), "checkpoint": checkpoint.name,
                    "checkpoint_path": str(checkpoint), "dense_threshold": dense_threshold,
                    "selection_score": float(selection_score), "validation": validation,
                })
                del decoder
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not candidates:
            raise SystemExit(f"No completed Lobster runs/checkpoints found under {args.runs_root}")
        winner = min(candidates, key=lambda row: row["selection_score"])
        winner_run = Path(winner["artifact_dir"])
        test_path = winner_run / "heldoutTestGraphs_adj_.npy"
        test_refs = to_graphs(np.load(test_path, allow_pickle=True), keep_largest_component=False)
        test_edges = np.asarray([graph.number_of_edges() for graph in test_refs], dtype=float)
        decoder = load_decoder(Path(winner["checkpoint_path"]), device, args.latent_dim)
        winner["test"] = evaluate(decoder, test_refs, args.test_rollouts, args.seed + 1_000_000,
                                  args.latent_dim, device,
                                  float(test_edges.mean() + 3 * test_edges.std()))

    payload = {
        "runs_root": str(args.runs_root), "device": str(device),
        "validation_rollouts": args.validation_rollouts, "test_rollouts": args.test_rollouts,
        "selection_formula": "median_normalized_mmd + stability_weight*std + dense_penalty_weight*dense_rate",
        "stability_weight": args.stability_weight,
        "dense_penalty_weight": args.dense_penalty_weight,
        "candidates": candidates, "winner": winner,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(report_markdown(payload))
    print(f"Selected {winner['run']}/{winner['checkpoint']}")


if __name__ == "__main__":
    main()
