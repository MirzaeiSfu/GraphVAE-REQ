#!/usr/bin/env python3
"""Decode a best checkpoint and threshold unpadded terminal matrices per rule."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import ReconstructedDataWrapper, _build_fom
from motif_counting.motif_counter import RelationalMotifCounter
from terminal_motif_evaluation import build_chain_metadata, write_json
from evaluate_terminal_motif_mmd_checkpoints import (
    flatten_config,
    load_decoder_modules,
    load_feature_metadata,
    load_yaml,
    resolve_cache_dir,
    resolve_checkpoint,
    sha256,
    training_seed_from_run_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--reference-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gin-checkpoint-name", default="best")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    started = time.time()
    run_dir = cli.run_dir.expanduser().resolve()
    reference_path = cli.reference_archive.expanduser().resolve()
    output_path = cli.output.expanduser().resolve()
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists() and not cli.overwrite:
        print(f"[skip] {output_path}")
        return 0
    os.chdir(REPO_ROOT)

    with np.load(reference_path, allow_pickle=False) as reference:
        selected_indices = reference["selected_chain_indices"].astype(np.int64)
        node_counts = reference["node_counts"].astype(np.int32)
        thresholds = reference["rule_thresholds"].astype(np.float64)
        reference_shape = reference["binary_terminal_matrices"].shape
    reference_metadata_path = reference_path.with_suffix(".json")
    with reference_metadata_path.open("r", encoding="utf-8") as handle:
        reference_metadata = json.load(handle)

    config_path = run_dir / "run_config_used.yaml"
    config_args = flatten_config(load_yaml(config_path))
    # Baseline settings often disable pruning or motif loss entirely. Evaluation
    # must nevertheless use exactly the same rule definition as the matching
    # reference archive, otherwise per-rule comparisons are not aligned.
    motif_definition_path = Path(reference_metadata["config"]).resolve()
    motif_definition = load_yaml(motif_definition_path)
    for key, value in motif_definition.get("motif", {}).items():
        setattr(config_args, key, value)
    motif_data = motif_definition.get("data", {})
    if "database_name" in motif_data:
        config_args.database_name = motif_data["database_name"]
    motif_runtime = motif_definition.get("runtime", {})
    if "motif_cache_dir" in motif_runtime:
        config_args.motif_cache_dir = motif_runtime["motif_cache_dir"]
    device = str(cli.device)
    config_args.device = device
    cache_dir = resolve_cache_dir(config_args)
    config_args.motif_cache_dir = str(cache_dir)
    if not hasattr(config_args, "motif_prune_max_total_values"):
        config_args.motif_prune_max_total_values = None
    if not hasattr(config_args, "motif_prune_max_values_per_rule"):
        config_args.motif_prune_max_values_per_rule = None

    checkpoint = resolve_checkpoint(run_dir, "best")
    directed = bool(getattr(config_args, "directed"))
    decoder, node_decoder, edge_decoder, dimensions = load_decoder_modules(
        checkpoint, directed, device
    )
    if int(node_counts.max()) > int(dimensions["max_nodes"]):
        raise ValueError(
            "Reference true node count exceeds checkpoint decoder size."
        )
    node_onehot_info, edge_onehot_info = load_feature_metadata(config_args)
    counter = RelationalMotifCounter(
        database_name=getattr(config_args, "database_name"),
        args=config_args,
    )

    saved_references = np.load(
        run_dir / "testGraphs_adj_.npy", allow_pickle=True
    )
    saved_node_counts = np.asarray(
        [int(adjacency.shape[0]) for adjacency in saved_references],
        dtype=np.int32,
    )
    if not np.array_equal(saved_node_counts, node_counts):
        raise ValueError(
            "Run test graph sizes do not match the reference archive."
        )

    training_seed = training_seed_from_run_dir(
        run_dir, int(getattr(config_args, "seed", 0))
    )
    generation_seed = training_seed
    torch.manual_seed(generation_seed)
    np.random.seed(generation_seed)
    random.seed(generation_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(generation_seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(generation_seed)

    binary = np.zeros(reference_shape, dtype=np.uint8)
    chain_metadata: List[Dict[str, Any]] | None = None
    observed_selected_indices: np.ndarray | None = None

    with torch.inference_mode():
        for start in range(0, len(node_counts), cli.batch_size):
            stop = min(start + cli.batch_size, len(node_counts))
            count = stop - start
            z = torch.randn(
                count,
                dimensions["latent_dim"],
                generator=generator,
                device=device,
            )
            adj_probs = torch.sigmoid(decoder(z))
            node_logits = node_decoder(z)
            edge_logits = edge_decoder(z) if edge_decoder is not None else None
            if node_logits.dim() == 2:
                node_logits = node_logits.view(
                    count, dimensions["max_nodes"], -1
                )

            batch_counts = node_counts[start:stop]
            for node_count in sorted(np.unique(batch_counts).tolist(), reverse=True):
                local_numpy = np.flatnonzero(batch_counts == node_count)
                local = torch.as_tensor(
                    local_numpy, dtype=torch.long, device=device
                )
                n = int(node_count)
                cropped_adj = adj_probs.index_select(0, local)[:, :n, :n]
                cropped_nodes = node_logits.index_select(0, local)[:, :n, :]
                cropped_edges = (
                    edge_logits.index_select(0, local)[:, :, :n, :n]
                    if edge_logits is not None
                    else None
                )
                wrapper = ReconstructedDataWrapper(
                    reconstructed_adj=cropped_adj,
                    node_feat_logits=cropped_nodes,
                    edge_feat_logits=cropped_edges,
                    relation_keys=counter.relation_keys,
                    node_onehot_info=node_onehot_info,
                    feature_onehot_mapping=_build_fom(node_onehot_info),
                    edge_onehot_info=edge_onehot_info,
                    edge_feature_info_mapping=counter.feature_info_mapping,
                    adj_threshold=0.5,
                    use_soft_adj=False,
                    device=device,
                )
                matrices, valid_mask = counter.count_batch(
                    wrapper,
                    batch_size=len(local_numpy),
                    output_mode="matrix",
                    detach_to_cpu=False,
                )
                if chain_metadata is None:
                    chain_metadata, observed = build_chain_metadata(
                        counter,
                        valid_mask,
                        matrix_size=n,
                        exclude_raw_relation=True,
                    )
                    observed_selected_indices = np.asarray(
                        observed, dtype=np.int64
                    )
                    if not np.array_equal(
                        observed_selected_indices, selected_indices
                    ):
                        raise ValueError(
                            "Generated/reference selected chain indices differ."
                        )

                index_tensor = torch.as_tensor(
                    selected_indices,
                    dtype=torch.long,
                    device=matrices.device,
                )
                selected = matrices.index_select(1, index_tensor)
                selected_mask = valid_mask.index_select(0, index_tensor)
                threshold_tensor = torch.as_tensor(
                    thresholds,
                    dtype=selected.dtype,
                    device=selected.device,
                ).view(1, -1, 1, 1)
                current = (
                    (selected > threshold_tensor)
                    & selected_mask.unsqueeze(0)
                ).to(torch.uint8).cpu().numpy()
                global_indices = start + local_numpy
                binary[global_indices, :, :n, :n] = current

            print(f"[generated] {run_dir}: {stop}/{len(node_counts)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        binary_terminal_matrices=binary,
        selected_chain_indices=selected_indices,
        node_counts=node_counts,
        rule_thresholds=thresholds,
    )
    cache_path = cache_dir / f"{config_args.database_name}.pkl"
    write_json(
        metadata_path,
        {
            "protocol": "reference_mean_per_rule_unpadded_v1",
            "run_dir": str(run_dir),
            "dataset": getattr(config_args, "dataset"),
            "setting": run_dir.parent.name,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "motif_definition_config": str(motif_definition_path),
            "motif_definition_config_sha256": sha256(motif_definition_path),
            "motif_cache": str(cache_path),
            "motif_cache_sha256": sha256(cache_path),
            "reference_archive": str(reference_path),
            "sample_count": int(len(node_counts)),
            "true_node_count_min": int(node_counts.min()),
            "true_node_count_max": int(node_counts.max()),
            "selected_chain_count": int(len(selected_indices)),
            "base_graph_discretization": {
                "adjacency": "sigmoid decoder probability >= 0.5",
                "node_features": "argmax within each categorical group",
                "edge_features": "argmax within each categorical group",
            },
            "terminal_threshold": (
                "terminal_value > reference-only per-rule mean"
            ),
            "padded_nodes_enter_motif_algorithm": False,
            "occurrence_metric_computed": False,
            "model_dimensions": dimensions,
            "elapsed_seconds": time.time() - started,
            "output": str(output_path),
        },
    )
    print(f"[done] {run_dir}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
