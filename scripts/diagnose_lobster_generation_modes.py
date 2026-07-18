#!/usr/bin/env python3
"""Compare Lobster adjacency post-processing modes on identical latent draws."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model import GraphTransformerDecoder_FC


MODES = ("full_threshold_0p5", "cropped_threshold_0p5", "cropped_calibrated", "cropped_fixed_budget")


def pair_probabilities(probability: np.ndarray, size: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Match nx.Graph's effective OR rule for an asymmetric adjacency matrix."""
    probability = probability[:size, :size]
    rows, cols = np.triu_indices(size, 1)
    return np.maximum(probability[rows, cols], probability[cols, rows]), (rows, cols)


def graph_from_pairs(size: int, pair_probs: np.ndarray, indices, *, threshold=None, budget=None) -> nx.Graph:
    rows, cols = indices
    if budget is not None:
        selected = np.zeros(pair_probs.size, dtype=bool)
        k = min(max(int(budget), 0), pair_probs.size)
        if k:
            selected[np.argpartition(pair_probs, -k)[-k:]] = True
    else:
        selected = pair_probs >= float(threshold)
    graph = nx.Graph()
    graph.add_nodes_from(range(size))
    graph.add_edges_from(zip(rows[selected].tolist(), cols[selected].tolist()))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def largest_component(graph: nx.Graph) -> nx.Graph:
    if not graph:
        return graph.copy()
    return nx.Graph(graph.subgraph(max(nx.connected_components(graph), key=len)))


def summary(values) -> dict:
    array = np.asarray(values, dtype=float)
    return {name: float(value) for name, value in {
        "mean": array.mean(), "std": array.std(), "median": np.median(array),
        "min": array.min(), "q05": np.quantile(array, 0.05),
        "q95": np.quantile(array, 0.95), "max": array.max(),
        "cv": array.std() / array.mean() if array.mean() else 0.0,
    }.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--validation-mean-edges", type=float, default=45.4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    references = np.load(args.references, allow_pickle=True)
    sizes = [int(graph.shape[0]) for graph in references]
    state = torch.load(args.checkpoint, map_location="cpu")
    max_nodes = int(round(state["decode.layers.3.bias"].numel() ** 0.5))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    decoder = GraphTransformerDecoder_FC(args.latent_dim, 256, max_nodes, directed=True).to(device)
    decoder.load_state_dict({key[len("decode."):]: value for key, value in state.items() if key.startswith("decode.")})
    decoder.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    probability_batches = []
    with torch.no_grad():
        for _ in range(args.rollouts):
            latent = torch.randn(len(sizes), args.latent_dim, generator=generator, device=device)
            probability_batches.append(torch.sigmoid(decoder(latent)).cpu().numpy())

    cropped_pairs = [
        pair_probabilities(probability_batches[r][i], size)[0]
        for r in range(args.rollouts) for i, size in enumerate(sizes)
    ]
    all_cropped_pairs = np.concatenate(cropped_pairs)
    desired_edges = args.validation_mean_edges * args.rollouts * len(sizes)
    desired_edges = int(round(min(desired_edges, all_cropped_pairs.size)))
    calibrated_threshold = (
        float(np.partition(all_cropped_pairs, all_cropped_pairs.size - desired_edges)[all_cropped_pairs.size - desired_edges])
        if desired_edges else 1.0
    )
    fixed_budget = int(round(args.validation_mean_edges))

    rows = []
    saved = {mode: [] for mode in MODES}
    near_half = []
    for rollout, probabilities in enumerate(probability_batches):
        rollout_graphs = {mode: [] for mode in MODES}
        for graph_index, (probability, size) in enumerate(zip(probabilities, sizes)):
            full_pairs, full_indices = pair_probabilities(probability, max_nodes)
            crop_pairs, crop_indices = pair_probabilities(probability, size)
            near_half.append(float(np.mean((crop_pairs >= 0.45) & (crop_pairs <= 0.55))))
            graphs = {
                "full_threshold_0p5": graph_from_pairs(max_nodes, full_pairs, full_indices, threshold=0.5),
                "cropped_threshold_0p5": graph_from_pairs(size, crop_pairs, crop_indices, threshold=0.5),
                "cropped_calibrated": graph_from_pairs(size, crop_pairs, crop_indices, threshold=calibrated_threshold),
                "cropped_fixed_budget": graph_from_pairs(size, crop_pairs, crop_indices, budget=fixed_budget),
            }
            for mode, graph in graphs.items():
                lcc = largest_component(graph)
                rollout_graphs[mode].append(graph)
                saved[mode].append(nx.to_numpy_array(graph, dtype=np.int8))
                rows.append({"rollout": rollout, "graph_index": graph_index, "target_nodes": size,
                             "mode": mode, "raw_nodes": graph.number_of_nodes(), "raw_edges": graph.number_of_edges(),
                             "lcc_nodes": lcc.number_of_nodes(), "lcc_edges": lcc.number_of_edges()})

    report = {
        "checkpoint": str(args.checkpoint), "references": str(args.references), "rollouts": args.rollouts,
        "graphs_per_rollout": len(sizes), "seed": args.seed, "device": str(device), "max_decoder_nodes": max_nodes,
        "reference_node_sizes": summary(sizes), "reference_edge_counts": summary([
            nx.from_numpy_array(np.asarray(graph)).number_of_edges() for graph in references]),
        "validation_mean_edges_used_for_calibration": args.validation_mean_edges,
        "calibrated_threshold": calibrated_threshold, "fixed_edge_budget": fixed_budget,
        "cropped_pair_probability_near_0p5_fraction": summary(near_half), "modes": {},
    }
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        rollout_means = [np.mean([row["raw_edges"] for row in mode_rows if row["rollout"] == rollout])
                         for rollout in range(args.rollouts)]
        report["modes"][mode] = {
            "raw_edges_all_graphs": summary([row["raw_edges"] for row in mode_rows]),
            "lcc_edges_all_graphs": summary([row["lcc_edges"] for row in mode_rows]),
            "raw_nodes_all_graphs": summary([row["raw_nodes"] for row in mode_rows]),
            "mean_raw_edges_across_rollouts": summary(rollout_means),
        }
        np.save(args.output_dir / f"{mode}_graphs.npy", np.array(saved[mode], dtype=object), allow_pickle=True)

    with (args.output_dir / "per_graph.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Lobster generation-mode stability diagnostic", "", f"- checkpoint: `{args.checkpoint}`",
             f"- rollouts: {args.rollouts}", f"- calibrated threshold: {calibrated_threshold:.6f}",
             f"- fixed edge budget: {fixed_budget}", "", "| Mode | rollout mean edges ± std | all-graph edge range |", "|---|---:|---:|"]
    for mode in MODES:
        item = report["modes"][mode]
        roll = item["mean_raw_edges_across_rollouts"]; all_edges = item["raw_edges_all_graphs"]
        lines.append(f"| {mode} | {roll['mean']:.2f} ± {roll['std']:.2f} | {all_edges['min']:.0f}–{all_edges['max']:.0f} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
