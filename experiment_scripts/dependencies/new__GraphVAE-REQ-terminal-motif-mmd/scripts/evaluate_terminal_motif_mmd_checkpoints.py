#!/usr/bin/env python3
"""Evaluate binarized terminal motif-chain graphs from saved checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import ReconstructedDataWrapper, _build_fom, list_graph_loader
from model import GraphTransformerDecoder_FC
from motif_counting.motif_counter import RelationalMotifCounter
from terminal_motif_evaluation import (
    binarize_terminal_matrices,
    build_chain_metadata,
    evaluate_terminal_adjacencies,
    write_json,
)
from util import EdgeFeatureDecoder, NodeFeatureDecoder, build_onehot_features


CHECKPOINT_NAMES = ("last_model", "best_validation_mmd_model")
OUTPUT_DIRNAME = "terminal_motif_mmd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode saved GraphVAE checkpoints, obtain each motif chain's final "
            "matrix, binarize it, and compute structural MMD against test graphs."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        help="Run directory to evaluate; may be repeated.",
    )
    source.add_argument(
        "--gather-root",
        type=Path,
        help="Recursively discover runs below this gather directory.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last", "auto"),
        default="best",
        help="Checkpoint source (default: best).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for decoding and motif matrix calculation.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarize terminal matrices with value >= threshold (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Post-hoc generation seed; default is each run's training seed.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-save-per-chain",
        action="store_true",
        help="Do not save compressed binarized matrices for each selected chain.",
    )
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help=(
            "Decode and save per-chain matrices/metadata without recomputing or "
            "overwriting the existing union evaluation JSON."
        ),
    )
    parser.add_argument(
        "--include-only-motif-runs",
        action="store_true",
        help="Skip configs where motif_loss is false.",
    )
    parser.add_argument(
        "--motif-definition-config",
        type=Path,
        default=None,
        help=(
            "Take database_name, motif-cache location, and motif rule-selection "
            "flags from this YAML. This is useful for evaluating a baseline "
            "checkpoint with exactly the same terminal chains as a motif run."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--repair-result-metadata",
        action="store_true",
        help=(
            "Update existing result JSON training_seed values from seed_N run "
            "directory names without recomputing decoded graphs or metrics."
        ),
    )
    return parser.parse_args()


def flatten_config(config: Dict[str, Any]) -> SimpleNamespace:
    flat: Dict[str, Any] = {}
    for section in config.values():
        if isinstance(section, dict):
            flat.update(section)
    return SimpleNamespace(**flat)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return loaded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_runs(args: argparse.Namespace) -> List[Path]:
    if args.run_dir:
        runs = [path.expanduser().resolve() for path in args.run_dir]
    else:
        root = args.gather_root.expanduser().resolve()
        runs = sorted(
            config_path.parent
            for config_path in root.rglob("run_config_used.yaml")
            if any((config_path.parent / name).exists() for name in CHECKPOINT_NAMES)
            and (config_path.parent / "testGraphs_adj_.npy").exists()
        )

    if args.include_only_motif_runs:
        runs = [
            run
            for run in runs
            if bool(load_yaml(run / "run_config_used.yaml").get("motif", {}).get("motif_loss"))
        ]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    return runs


def resolve_checkpoint(run_dir: Path, choice: str) -> Path:
    candidates = {
        "best": [run_dir / "best_validation_mmd_model"],
        "last": [run_dir / "last_model"],
        "auto": [run_dir / "last_model", run_dir / "best_validation_mmd_model"],
    }[choice]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {choice} checkpoint found in {run_dir}; tried "
        + ", ".join(str(path.name) for path in candidates)
    )


def training_seed_from_run_dir(run_dir: Path, fallback: int) -> int:
    """Use the run identity when copied YAML still contains its seed-0 template."""

    match = re.fullmatch(r"seed_(\d+)", run_dir.name)
    return int(match.group(1)) if match else int(fallback)


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def infer_model_dimensions(
    state: Dict[str, torch.Tensor], directed: bool
) -> Tuple[int, int, int, Optional[int]]:
    latent_dim = int(state["decode.layers.0.weight"].shape[1])
    adjacency_output = int(state["decode.layers.3.weight"].shape[0])
    if directed:
        max_nodes = math.isqrt(adjacency_output)
        if max_nodes * max_nodes != adjacency_output:
            raise ValueError(
                f"Directed decoder output {adjacency_output} is not a square."
            )
    else:
        # output = N(N+1)/2
        max_nodes = int((math.sqrt(1 + 8 * adjacency_output) - 1) / 2)
        if max_nodes * (max_nodes + 1) // 2 != adjacency_output:
            raise ValueError(
                f"Undirected decoder output {adjacency_output} is not triangular."
            )

    node_output = int(state["node_feature_decoder.net.3.weight"].shape[0])
    if node_output % max_nodes:
        raise ValueError("Node feature decoder output is not divisible by N.")
    node_dim = node_output // max_nodes

    edge_dim = None
    edge_key = "edge_feature_decoder.net.3.weight"
    if edge_key in state:
        edge_output = int(state[edge_key].shape[0])
        if edge_output % (max_nodes * max_nodes):
            raise ValueError("Edge feature decoder output is not divisible by N^2.")
        edge_dim = edge_output // (max_nodes * max_nodes)
    return latent_dim, max_nodes, node_dim, edge_dim


def load_decoder_modules(
    checkpoint: Path, directed: bool, device: str
) -> Tuple[
    GraphTransformerDecoder_FC,
    NodeFeatureDecoder,
    Optional[EdgeFeatureDecoder],
    Dict[str, int],
]:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported checkpoint payload in {checkpoint}.")
    if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        loaded = loaded["state_dict"]
    state = strip_module_prefix(loaded)

    latent_dim, max_nodes, node_dim, edge_dim = infer_model_dimensions(
        state, directed
    )
    decoder = GraphTransformerDecoder_FC(
        input=latent_dim,
        lambdaDim=None,
        SubGraphNodeNum=max_nodes,
        directed=directed,
    )
    node_decoder = NodeFeatureDecoder(latent_dim, max_nodes, node_dim)
    edge_decoder = (
        EdgeFeatureDecoder(latent_dim, max_nodes, edge_dim)
        if edge_dim is not None
        else None
    )

    def subset(prefix: str) -> Dict[str, torch.Tensor]:
        return {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }

    decoder.load_state_dict(subset("decode."), strict=True)
    node_decoder.load_state_dict(subset("node_feature_decoder."), strict=True)
    if edge_decoder is not None:
        edge_decoder.load_state_dict(
            subset("edge_feature_decoder."), strict=True
        )

    decoder.to(device).eval()
    node_decoder.to(device).eval()
    if edge_decoder is not None:
        edge_decoder.to(device).eval()
    return decoder, node_decoder, edge_decoder, {
        "latent_dim": latent_dim,
        "max_nodes": max_nodes,
        "node_feature_dim": node_dim,
        "edge_feature_dim": edge_dim,
    }


def resolve_cache_dir(config_args: SimpleNamespace) -> Path:
    configured = getattr(config_args, "motif_cache_dir", None)
    return (REPO_ROOT / (configured or "cache_motifs")).resolve()


def load_feature_metadata(
    config_args: SimpleNamespace,
) -> Tuple[Dict[int, Dict[str, Any]], Optional[Dict[int, Dict[str, Any]]]]:
    data_dir = getattr(config_args, "data_dir", None)
    if data_dir:
        data_path = Path(data_dir).expanduser()
        if not data_path.is_absolute():
            data_path = REPO_ROOT / data_path
        os.environ["DATA_DIR"] = str(data_path.resolve())
    else:
        os.environ["DATA_DIR"] = str((REPO_ROOT / "data_raw").resolve())

    loaded = list_graph_loader(
        getattr(config_args, "dataset"),
        return_labels=True,
        lobster_feature_schema=getattr(
            config_args, "lobster_feature_schema", "optimal_v2"
        ),
        tu_attribute_bins=int(getattr(config_args, "tu_attribute_bins", 8)),
        tu_max_nodes=getattr(config_args, "tu_max_nodes", None),
    )
    list_adj, _, _, node_features, edge_features, node_info, edge_info = loaded
    _, _, node_onehot_info, edge_onehot_info = build_onehot_features(
        node_features,
        edge_features,
        list_adj,
        node_info,
        edge_info,
    )
    if not node_onehot_info:
        raise RuntimeError(
            f"Dataset {config_args.dataset} has no node-feature metadata."
        )
    return node_onehot_info, edge_onehot_info


def load_existing_metrics(run_dir: Path) -> Dict[str, Any]:
    existing: Dict[str, Any] = {}
    for name in (
        "final_table2_metrics.json",
        "final_table3_metrics.json",
        "final_metrics_summary.json",
    ):
        path = run_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                existing[name] = json.load(handle)
    return existing


def save_chain_archive(
    path: Path,
    binary_batches: List[np.ndarray],
    selected_indices: List[int],
) -> None:
    binary = np.concatenate(binary_batches, axis=0)
    np.savez_compressed(
        path,
        binary_terminal_matrices=binary,
        selected_chain_indices=np.asarray(selected_indices, dtype=np.int64),
    )


def evaluate_run(run_dir: Path, cli: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    output_dir = run_dir / OUTPUT_DIRNAME
    result_path = output_dir / "terminal_motif_mmd.json"
    archive_path = output_dir / "terminal_chain_binary_matrices.npz"
    if cli.archive_only and archive_path.exists() and not cli.overwrite:
        print(f"[skip] {run_dir}: {archive_path.name} already exists")
        with result_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if result_path.exists() and not cli.overwrite and not cli.archive_only:
        print(f"[skip] {run_dir}: {result_path.name} already exists")
        with result_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "run_config_used.yaml"
    config = load_yaml(config_path)
    config_args = flatten_config(config)
    motif_definition_path = None
    if cli.motif_definition_config is not None:
        motif_definition_path = (
            cli.motif_definition_config.expanduser().resolve()
        )
        motif_definition = load_yaml(motif_definition_path)
        for key, value in motif_definition.get("motif", {}).items():
            setattr(config_args, key, value)
        motif_data = motif_definition.get("data", {})
        if "database_name" in motif_data:
            config_args.database_name = motif_data["database_name"]
        motif_runtime = motif_definition.get("runtime", {})
        if "motif_cache_dir" in motif_runtime:
            config_args.motif_cache_dir = motif_runtime["motif_cache_dir"]
    checkpoint = resolve_checkpoint(run_dir, cli.checkpoint)
    reference_path = run_dir / "testGraphs_adj_.npy"
    references = np.load(reference_path, allow_pickle=True)
    sample_count = len(references)
    generation_seed = int(
        cli.seed if cli.seed is not None else getattr(config_args, "seed", 0)
    )

    device = str(cli.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    torch.manual_seed(generation_seed)
    np.random.seed(generation_seed)
    random.seed(generation_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(generation_seed)

    directed = bool(getattr(config_args, "directed"))
    decoder, node_decoder, edge_decoder, dimensions = load_decoder_modules(
        checkpoint, directed, device
    )
    node_onehot_info, edge_onehot_info = load_feature_metadata(config_args)
    if len(node_onehot_info) != dimensions["node_feature_dim"]:
        raise RuntimeError(
            "Loaded node-feature schema does not match checkpoint: "
            f"metadata={len(node_onehot_info)}, "
            f"checkpoint={dimensions['node_feature_dim']}."
        )
    if edge_onehot_info is not None:
        expected_edge_dim = len(edge_onehot_info)
        if expected_edge_dim != dimensions["edge_feature_dim"]:
            raise RuntimeError(
                "Loaded edge-feature schema does not match checkpoint: "
                f"metadata={expected_edge_dim}, "
                f"checkpoint={dimensions['edge_feature_dim']}."
            )

    cache_dir = resolve_cache_dir(config_args)
    config_args.motif_cache_dir = str(cache_dir)
    config_args.device = device
    if not hasattr(config_args, "motif_prune_max_total_values"):
        config_args.motif_prune_max_total_values = None
    if not hasattr(config_args, "motif_prune_max_values_per_rule"):
        config_args.motif_prune_max_values_per_rule = None
    counter = RelationalMotifCounter(
        database_name=getattr(config_args, "database_name"),
        args=config_args,
    )
    cache_path = cache_dir / f"{config_args.database_name}.pkl"

    generator = torch.Generator(device=device)
    generator.manual_seed(generation_seed)
    binary_batches: List[np.ndarray] = []
    union_batches: List[np.ndarray] = []
    chain_metadata: Optional[List[Dict[str, Any]]] = None
    selected_indices: Optional[List[int]] = None

    with torch.inference_mode():
        for start in range(0, sample_count, cli.batch_size):
            count = min(cli.batch_size, sample_count - start)
            z = torch.randn(
                count,
                dimensions["latent_dim"],
                generator=generator,
                device=device,
            )
            reconstructed_adj = torch.sigmoid(decoder(z))
            node_logits = node_decoder(z)
            edge_logits = edge_decoder(z) if edge_decoder is not None else None
            wrapper = ReconstructedDataWrapper(
                reconstructed_adj,
                node_logits,
                edge_logits,
                counter.relation_keys,
                node_onehot_info,
                _build_fom(node_onehot_info),
                edge_onehot_info=edge_onehot_info,
                edge_feature_info_mapping=counter.feature_info_mapping,
                adj_threshold=0.5,
                use_soft_adj=False,
                device=device,
            )
            matrices, valid_mask = counter.count_batch(
                wrapper,
                batch_size=count,
                output_mode="matrix",
                detach_to_cpu=False,
            )

            if chain_metadata is None:
                chain_metadata, selected_indices = build_chain_metadata(
                    counter,
                    valid_mask,
                    matrix_size=dimensions["max_nodes"],
                    exclude_raw_relation=True,
                )
            batch_binary, batch_union = binarize_terminal_matrices(
                matrices,
                valid_mask,
                selected_indices,
                threshold=cli.threshold,
            )
            if not cli.no_save_per_chain:
                binary_batches.append(batch_binary)
            union_batches.append(batch_union)
            print(
                f"[decode] {run_dir}: {start + count}/{sample_count} samples"
            )

            del z, reconstructed_adj, node_logits, edge_logits, matrices, valid_mask

    union_adjacencies = np.concatenate(union_batches, axis=0)
    union_path = output_dir / "terminal_union_adjacencies.npy"
    np.save(union_path, union_adjacencies, allow_pickle=False)
    if not cli.no_save_per_chain:
        save_chain_archive(
            archive_path,
            binary_batches,
            selected_indices,
        )
    write_json(output_dir / "chain_metadata.json", chain_metadata)

    if cli.archive_only:
        print(f"[archive-only] {run_dir}: {archive_path}")
        if result_path.exists():
            with result_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        archive_manifest = {
            "metric_family": "terminal_motif_chain_archive_only",
            "run_dir": str(run_dir),
            "dataset": getattr(config_args, "dataset"),
            "setting": run_dir.parent.name,
            "training_seed": training_seed_from_run_dir(
                run_dir, getattr(config_args, "seed", generation_seed)
            ),
            "generation_seed": generation_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "motif_definition_config": (
                str(motif_definition_path)
                if motif_definition_path
                else str(config_path)
            ),
            "motif_cache": str(cache_path),
            "motif_cache_sha256": sha256(cache_path),
            "sample_count": sample_count,
            "matrix_binarization": {
                "threshold": float(cli.threshold),
                "rule": (
                    f"value >= {float(cli.threshold)} -> 1; otherwise -> 0"
                ),
            },
            "terminal_graph_definition": {
                "selected_chain_count": len(selected_indices),
                "total_chain_count": len(chain_metadata),
                "combination": "archive_only_no_evaluation",
            },
            "model_dimensions": dimensions,
            "outputs": {
                "terminal_union_adjacencies": str(union_path),
                "chain_metadata": str(output_dir / "chain_metadata.json"),
                "terminal_chain_binary_matrices": str(archive_path),
            },
            "evaluation": None,
            "elapsed_seconds": time.time() - started,
        }
        write_json(result_path, archive_manifest)
        return archive_manifest

    # Re-seed the stochastic random-GIN evaluator independently from generation.
    torch.manual_seed(generation_seed)
    np.random.seed(generation_seed)
    random.seed(generation_seed)
    evaluation = evaluate_terminal_adjacencies(union_adjacencies, references)
    result = {
        "metric_family": "binarized_terminal_motif_chain_structural_mmd",
        "run_dir": str(run_dir),
        "dataset": getattr(config_args, "dataset"),
        "setting": run_dir.parent.name,
        "training_seed": training_seed_from_run_dir(
            run_dir, getattr(config_args, "seed", generation_seed)
        ),
        "generation_seed": generation_seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "motif_definition_config": (
            str(motif_definition_path) if motif_definition_path else str(config_path)
        ),
        "motif_definition_config_sha256": sha256(
            motif_definition_path if motif_definition_path else config_path
        ),
        "motif_cache": str(cache_path),
        "motif_cache_sha256": sha256(cache_path),
        "sample_count": sample_count,
        "matrix_binarization": {
            "applied_before_graph_construction_and_metrics": True,
            "threshold": float(cli.threshold),
            "rule": f"value >= {float(cli.threshold)} -> 1; otherwise -> 0",
        },
        "terminal_graph_definition": {
            "selected_terminal_shape": [
                dimensions["max_nodes"],
                dimensions["max_nodes"],
            ],
            "excluded_one_atom_raw_relation": True,
            "combination": "elementwise_binary_OR_over_selected_terminal_chains",
            "self_loops_removed": True,
            "isolates_removed": True,
            "largest_connected_component_only": True,
            "selected_chain_count": len(selected_indices),
            "total_chain_count": len(chain_metadata),
        },
        "model_dimensions": dimensions,
        "outputs": {
            "terminal_union_adjacencies": str(union_path),
            "chain_metadata": str(output_dir / "chain_metadata.json"),
            "terminal_chain_binary_matrices": (
                None
                if cli.no_save_per_chain
                else str(output_dir / "terminal_chain_binary_matrices.npz")
            ),
        },
        "evaluation": evaluation,
        "existing_normal_adjacency_metrics": load_existing_metrics(run_dir),
        "elapsed_seconds": time.time() - started,
    }
    write_json(result_path, result)
    print(f"[done] {run_dir}: {result_path}")
    return result


def main() -> int:
    cli = parse_args()
    if cli.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not math.isfinite(cli.threshold):
        raise ValueError("--threshold must be finite.")
    if cli.archive_only and cli.no_save_per_chain:
        raise ValueError("--archive-only cannot be used with --no-save-per-chain.")

    runs = discover_runs(cli)
    if not runs:
        raise RuntimeError("No evaluable runs were found.")
    print(f"Discovered {len(runs)} run(s).")

    if cli.repair_result_metadata:
        repaired = 0
        for run_dir in runs:
            result_path = run_dir / OUTPUT_DIRNAME / "terminal_motif_mmd.json"
            if not result_path.exists():
                continue
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            canonical_seed = training_seed_from_run_dir(
                run_dir, payload.get("training_seed", 0)
            )
            if payload.get("training_seed") != canonical_seed:
                payload["training_seed"] = canonical_seed
                write_json(result_path, payload)
                repaired += 1
        print(f"Repaired training_seed metadata in {repaired} result file(s).")
        return 0

    failures: List[Dict[str, str]] = []
    results: List[Dict[str, Any]] = []
    for index, run_dir in enumerate(runs, start=1):
        print(f"\n=== Run {index}/{len(runs)}: {run_dir} ===")
        try:
            results.append(evaluate_run(run_dir, cli))
        except Exception as exc:
            failures.append({"run_dir": str(run_dir), "error": repr(exc)})
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)

    summary_root = (
        cli.gather_root.expanduser().resolve()
        if cli.gather_root
        else runs[0].parent
    )
    summary_path = summary_root / "terminal_motif_mmd_batch_summary.json"
    write_json(
        summary_path,
        {
            "completed": len(results),
            "failed": len(failures),
            "results": [
                {
                    "run_dir": result.get("run_dir"),
                    "dataset": result.get("dataset"),
                    "setting": result.get("setting"),
                    "training_seed": result.get("training_seed"),
                    "metrics": (result.get("evaluation") or {}).get("metrics"),
                }
                for result in results
            ],
            "failures": failures,
        },
    )
    print(f"\nBatch summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
