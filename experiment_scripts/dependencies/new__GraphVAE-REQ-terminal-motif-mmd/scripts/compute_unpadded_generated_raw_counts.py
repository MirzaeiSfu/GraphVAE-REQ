#!/usr/bin/env python3
"""Generate a training-sized sample and compute raw per-rule motif counts."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--training-count-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    started = time.time()
    run_dir = cli.run_dir.expanduser().resolve()
    training_path = cli.training_count_archive.expanduser().resolve()
    training_metadata_path = training_path.with_suffix(".json")
    output_path = cli.output.expanduser().resolve()
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists() and not cli.overwrite:
        print(f"[skip] {output_path}")
        return 0
    os.chdir(REPO_ROOT)

    with np.load(training_path, allow_pickle=False) as archive:
        selected_indices = archive["selected_chain_indices"].astype(np.int64)
        node_counts = archive["node_counts"].astype(np.int32)
        training_shape = archive["raw_motif_counts"].shape
    with training_metadata_path.open("r", encoding="utf-8") as handle:
        training_metadata = json.load(handle)

    config_path = run_dir / "run_config_used.yaml"
    config_args = flatten_config(load_yaml(config_path))
    # Evaluate every checkpoint with the canonical motif definition used to
    # build its matching training archive, including baseline checkpoints.
    motif_definition_path = Path(training_metadata["config"]).resolve()
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
            "Training true node count exceeds checkpoint decoder size."
        )
    node_onehot_info, edge_onehot_info = load_feature_metadata(config_args)
    counter = RelationalMotifCounter(
        database_name=getattr(config_args, "database_name"),
        args=config_args,
    )

    training_seed = training_seed_from_run_dir(
        run_dir, int(getattr(config_args, "seed", 0))
    )
    generation_seed = (
        int(cli.seed) if cli.seed is not None else int(training_seed)
    )
    torch.manual_seed(generation_seed)
    np.random.seed(generation_seed)
    random.seed(generation_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(generation_seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(generation_seed)

    generated_counts = np.empty(training_shape, dtype=np.float64)
    chain_metadata: List[Dict[str, Any]] | None = None

    with torch.inference_mode():
        for start in range(0, len(node_counts), int(cli.batch_size)):
            stop = min(start + int(cli.batch_size), len(node_counts))
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
            for node_count in sorted(
                np.unique(batch_counts).tolist(), reverse=True
            ):
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
                    if not np.array_equal(
                        np.asarray(observed, dtype=np.int64), selected_indices
                    ):
                        raise ValueError(
                            "Generated/training selected chain indices differ."
                        )
                index = torch.as_tensor(
                    selected_indices,
                    dtype=torch.long,
                    device=matrices.device,
                )
                selected = matrices.index_select(1, index)
                selected_mask = valid_mask.index_select(0, index)
                counts = (
                    selected.to(torch.float64)
                    * selected_mask.unsqueeze(0).to(torch.float64)
                ).sum(dim=(2, 3))
                generated_counts[start + local_numpy] = counts.cpu().numpy()
            print(f"[generated-raw-count] {run_dir}: {stop}/{len(node_counts)}")

    if not np.isfinite(generated_counts).all():
        raise ValueError("Non-finite raw generated motif count detected.")
    integer_deviation = float(
        np.max(np.abs(generated_counts - np.rint(generated_counts)))
    )
    if integer_deviation > 1e-5:
        raise ValueError(
            "Discretized generated graphs produced non-integer raw counts: "
            f"max deviation={integer_deviation}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        raw_motif_counts=generated_counts,
        selected_chain_indices=selected_indices,
        node_counts=node_counts,
    )
    motif_cache_path = cache_dir / f"{config_args.database_name}.pkl"
    write_json(
        metadata_path,
        {
            "protocol": "raw_training_count_mmd_v1",
            "archive_role": "generated",
            "run_dir": str(run_dir),
            "dataset": getattr(config_args, "dataset"),
            "setting": run_dir.parent.name,
            "training_seed": int(training_seed),
            "generation_seed": int(generation_seed),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "motif_definition_config": str(motif_definition_path),
            "motif_definition_config_sha256": sha256(motif_definition_path),
            "motif_cache": str(motif_cache_path),
            "motif_cache_sha256": sha256(motif_cache_path),
            "training_count_archive": str(training_path),
            "training_count_archive_sha256": sha256(training_path),
            "sample_count": int(len(node_counts)),
            "selected_rule_count": int(len(selected_indices)),
            "true_node_count_min": int(node_counts.min()),
            "true_node_count_max": int(node_counts.max()),
            "node_count_schedule": (
                "one generated graph cropped to each exact training-graph "
                "node count"
            ),
            "base_graph_discretization": {
                "adjacency": "sigmoid decoder probability >= 0.5",
                "node_features": "argmax within each categorical group",
                "edge_features": "argmax within each categorical group",
            },
            "count_definition": (
                "sum of every valid raw terminal-matrix cell for each rule; "
                "diagonal included; no threshold and no binarization"
            ),
            "padded_nodes_enter_motif_algorithm": False,
            "occurrence_metric_computed": False,
            "max_integer_deviation": integer_deviation,
            "model_dimensions": dimensions,
            "elapsed_seconds": time.time() - started,
            "output": str(output_path),
        },
    )
    print(
        f"[done] {run_dir}: graphs={len(node_counts)} "
        f"rules={len(selected_indices)} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
