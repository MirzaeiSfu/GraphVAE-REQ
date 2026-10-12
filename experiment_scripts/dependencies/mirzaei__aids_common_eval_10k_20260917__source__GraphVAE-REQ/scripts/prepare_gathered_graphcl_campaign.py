#!/usr/bin/env python3
"""Prepare attributed PyG inputs for every checkpoint in a gathered run tree.

The generated topology, node attributes, and edge attributes are decoded from
the same prior latent sample.  Reference attributes come from the exact
training cache.  PROTEINS is the one exception: its cache was not collected,
so a separately exported real split must be supplied and its adjacency is
checked against every gathered held-out adjacency artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_EVALUATION_SRC = REPO_ROOT / "graph_evaluation" / "src"
for source_path in (REPO_ROOT, REPO_ROOT / "scripts", GRAPH_EVALUATION_SRC):
    source_text = str(source_path)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

from eval.attributed_gin import graph_from_dense_attributes  # noqa: E402
from ggm_eval import load_pyg_collection, save_pyg_collection  # noqa: E402
from ggm_eval.adapters import attributed_arrays_to_pyg  # noqa: E402
from model import GraphTransformerDecoder_FC  # noqa: E402
from resample_grid_checkpoints import load_config  # noqa: E402
from util import EdgeFeatureDecoder, NodeFeatureDecoder  # noqa: E402


DATASET_ORDER = ("aids", "enzymes", "mutag", "ogb", "proteins", "ptc")
CANONICAL_DATASETS = {
    "aids": "AIDS",
    "enzymes": "ENZYMES",
    "mutag": "MUTAG",
    "ogb": "ogbg-molbbbp",
    "proteins": "PROTEINS",
    "ptc": "PTC",
}
FEATURE_SCHEMAS = {
    "aids": "tu-quantile8-maxall|export=decoded_node_edge",
    "enzymes": (
        "tu-quantile8-maxall-src3-4-5-6|export=decoded_node"
    ),
    "mutag": "gin-node-label-v2|export=decoded_node",
    "ogb": "default|export=decoded_node_edge",
    "proteins": "default|export=decoded_node",
    "ptc": "gin-node-label-v2|export=decoded_node",
}
FEATURE_MODES = {
    "aids": "decoded_node_edge",
    "enzymes": "decoded_node",
    "mutag": "decoded_node",
    "ogb": "decoded_node_edge",
    "proteins": "decoded_node",
    "ptc": "decoded_node",
}
DEFAULT_OGB_CACHE = Path(
    "/local-scratch2/new/count_distance_artifacts/ogb/dataset_cache/"
    "ogbg-molbbbp_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_"
    "bfs-legacy_first_component.pkl"
)


def remove_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary cache in {path}.")
    return payload


def cache_for_dataset(gather_root: Path, dataset: str, ogb_cache: Path) -> Path | None:
    if dataset == "proteins":
        return None
    if dataset == "ogb":
        if not ogb_cache.is_file():
            raise FileNotFoundError(f"OGB dataset cache not found: {ogb_cache}")
        return ogb_cache.resolve()
    candidates = sorted(
        (gather_root / "datasets" / dataset).glob(
            "setting_*/seed_*/dataset_cache/*.pkl"
        )
    )
    if not candidates:
        raise FileNotFoundError(f"No gathered dataset cache found for {dataset}.")
    return candidates[0].resolve()


def cache_split(cache: dict, split: str):
    if split == "train":
        dataset = cache["list_graphs"]
        return (
            list(dataset.list_adjs),
            list(dataset.list_node_onehot),
            list(dataset.list_edge_onehot),
        )
    if split == "validation":
        return (
            list(cache["val_adj"]),
            list(cache["list_noh_val"]),
            list(cache["list_eoh_val"]),
        )
    if split == "test":
        return (
            list(cache["test_list_adj"]),
            list(cache["list_noh_test"]),
            list(cache["list_eoh_test"]),
        )
    raise ValueError(f"Unknown split: {split}")


def to_pyg_graphs(
    adjacencies,
    node_values,
    edge_values,
    *,
    node_feature_info,
    edge_feature_info,
    values_are_logits: bool,
    split_name: str,
):
    graphs = []
    rejected = 0
    for index, (adjacency, node_value, edge_value) in enumerate(
        zip(adjacencies, node_values, edge_values)
    ):
        graph = graph_from_dense_attributes(
            adjacency,
            node_value,
            edge_value,
            node_feature_info=node_feature_info,
            edge_feature_info=edge_feature_info,
            values_are_logits=values_are_logits,
            adjacency_threshold=0.5,
        )
        if graph is None:
            rejected += 1
            continue
        graphs.append(
            attributed_arrays_to_pyg(
                graph.edges,
                graph.node_attributes,
                graph.edge_attributes,
                graph.source_node_ids,
                name=f"{split_name} graph {index}",
            )
        )
    if not graphs:
        raise ValueError(f"No non-empty graphs remained for {split_name}.")
    return graphs, rejected


def export_cache_references(
    cache: dict,
    dataset: str,
    output_dir: Path,
) -> dict:
    metadata_base = {
        "dataset": CANONICAL_DATASETS[dataset],
        "feature_mode": FEATURE_MODES[dataset],
        "feature_schema": FEATURE_SCHEMAS[dataset],
        "producer": "scripts/prepare_gathered_graphcl_campaign.py",
        "source": "exact gathered GraphVAE dataset cache",
        "split_mode": "paper_70_10_20",
        "split_seed": 123,
        "train_fraction": 0.7,
        "validation_fraction": 0.1,
        "test_fraction": 0.2,
        "bfs_strategy": "legacy_first_component",
    }
    results = {}
    for split, filename in (
        ("train", "real_train_graphs.pt"),
        ("validation", "real_validation_graphs.pt"),
        ("test", "real_test_graphs.pt"),
    ):
        adjacencies, node_values, edge_values = cache_split(cache, split)
        graphs, rejected = to_pyg_graphs(
            adjacencies,
            node_values,
            edge_values,
            node_feature_info=cache.get("node_onehot_info"),
            edge_feature_info=cache.get("edge_onehot_info"),
            values_are_logits=False,
            split_name=f"{dataset}/{split}",
        )
        destination = output_dir / filename
        manifest = save_pyg_collection(
            destination,
            graphs,
            metadata={**metadata_base, "split": split},
        )
        results[split] = {
            "path": str(destination.resolve()),
            "source_count": len(adjacencies),
            "rejected_empty": rejected,
            "manifest": manifest,
        }
    return results


def dense_adjacency(graph) -> np.ndarray:
    result = np.zeros((int(graph.num_nodes), int(graph.num_nodes)), dtype=np.int8)
    edge_index = graph.edge_index.detach().cpu().numpy()
    if edge_index.size:
        result[edge_index[0], edge_index[1]] = 1
    return result


def adjacency_sequence_digest(adjacencies) -> str:
    digest = hashlib.sha256()
    for adjacency in adjacencies:
        dense = (
            adjacency.toarray()
            if hasattr(adjacency, "toarray")
            else np.asarray(adjacency)
        )
        dense = np.ascontiguousarray((dense >= 0.5).astype(np.int8))
        digest.update(str(dense.shape).encode("ascii"))
        digest.update(dense.tobytes())
    return digest.hexdigest()


def nonempty_adjacencies(adjacencies) -> list:
    retained = []
    for adjacency in adjacencies:
        dense = (
            adjacency.toarray()
            if hasattr(adjacency, "toarray")
            else np.asarray(adjacency)
        )
        binary = dense >= 0.5
        np.fill_diagonal(binary, False)
        if np.any(binary):
            retained.append(adjacency)
    return retained


def validate_proteins_references(
    gather_root: Path,
    proteins_real_dirs: list[Path],
) -> tuple[dict[str, Path], dict]:
    references_by_digest = {}
    real_summaries = {}
    for proteins_real_dir in proteins_real_dirs:
        test_path = proteins_real_dir / "real_test_graphs.pt"
        train_path = proteins_real_dir / "real_train_graphs.pt"
        if not test_path.is_file() or not train_path.is_file():
            raise FileNotFoundError(
                "PROTEINS needs real_train_graphs.pt and real_test_graphs.pt "
                f"in {proteins_real_dir}."
            )
        reference = load_pyg_collection(test_path, normalize=False)
        digest = adjacency_sequence_digest(
            [dense_adjacency(graph) for graph in reference]
        )
        references_by_digest[digest] = (test_path.resolve(), reference)
        real_summaries[digest] = {
            "real_dir": str(proteins_real_dir.resolve()),
            "train": str(train_path.resolve()),
            "test": str(test_path.resolve()),
            "test_graph_count": len(reference),
        }

    reference_by_run = {}
    family_counts = {}
    for adjacency_path in sorted(
        (gather_root / "datasets" / "proteins").glob(
            "setting_*/seed_*/testGraphs_adj_.npy"
        )
    ):
        expected = np.load(adjacency_path, allow_pickle=True)
        retained_expected = nonempty_adjacencies(expected)
        digest = adjacency_sequence_digest(retained_expected)
        if digest not in references_by_digest:
            raise ValueError(
                "No exported attributed PROTEINS reference matches gathered "
                f"{adjacency_path}; adjacency digest={digest}."
            )
        test_path, reference = references_by_digest[digest]
        if len(retained_expected) != len(reference):
            raise ValueError(
                f"PROTEINS test count mismatch for {adjacency_path}: "
                f"{len(retained_expected)} non-empty versus {len(reference)}."
            )
        for index, (graph, expected_adjacency) in enumerate(
            zip(reference, retained_expected)
        ):
            expected_dense = (
                expected_adjacency.toarray()
                if hasattr(expected_adjacency, "toarray")
                else np.asarray(expected_adjacency)
            )
            actual_dense = dense_adjacency(graph)
            if (
                actual_dense.shape != expected_dense.shape
                or not np.array_equal(actual_dense, expected_dense)
            ):
                raise ValueError(
                    "PROTEINS exported reference differs from gathered "
                    f"{adjacency_path} at graph {index}."
                )
        run_dir = adjacency_path.parent.resolve()
        reference_by_run[str(run_dir)] = test_path
        family_counts[digest] = family_counts.get(digest, 0) + 1
    if not reference_by_run:
        raise ValueError("No gathered PROTEINS held-out adjacency artifacts found.")
    summary = {
        "validated_against_run_count": len(reference_by_run),
        "reference_families": real_summaries,
        "run_count_by_reference_family": family_counts,
        "reference_by_run": {
            key: str(value) for key, value in reference_by_run.items()
        },
    }
    return reference_by_run, summary


def checkpoint_state_dict(checkpoint_path: Path) -> dict:
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {checkpoint_path}")
    return payload


def undirected_node_count(output_dim: int) -> int:
    discriminant = 1 + 8 * output_dim
    root = int(math.isqrt(discriminant))
    if root * root != discriminant or (root - 1) % 2:
        raise ValueError(
            f"Cannot infer undirected node count from output dimension {output_dim}."
        )
    node_count = (root - 1) // 2
    if node_count * (node_count + 1) // 2 != output_dim:
        raise ValueError(
            f"Invalid triangular decoder output dimension {output_dim}."
        )
    return node_count


def build_decoders(state: dict, config: dict, device: torch.device):
    directed = bool(config.get("directed", True))
    adjacency_output_dim = int(state["decode.layers.3.bias"].numel())
    if directed:
        node_count = int(math.isqrt(adjacency_output_dim))
        if node_count * node_count != adjacency_output_dim:
            raise ValueError(
                f"Directed decoder output is not square: {adjacency_output_dim}."
            )
    else:
        node_count = undirected_node_count(adjacency_output_dim)
    latent_dim = int(state["decode.layers.0.weight"].shape[1])

    adjacency_decoder = GraphTransformerDecoder_FC(
        latent_dim, 256, node_count, directed
    )
    adjacency_state = {
        remove_prefix(key, "decode."): value
        for key, value in state.items()
        if key.startswith("decode.")
    }
    adjacency_decoder.load_state_dict(adjacency_state, strict=True)

    node_output_dim = int(state["node_feature_decoder.net.3.bias"].numel())
    if node_output_dim % node_count:
        raise ValueError(
            f"Node-head output {node_output_dim} is not divisible by {node_count}."
        )
    node_dim = node_output_dim // node_count
    node_decoder = NodeFeatureDecoder(latent_dim, node_count, node_dim)
    node_state = {
        remove_prefix(key, "node_feature_decoder."): value
        for key, value in state.items()
        if key.startswith("node_feature_decoder.")
    }
    node_decoder.load_state_dict(node_state, strict=True)

    edge_decoder = None
    edge_prefix = "edge_feature_decoder."
    if any(key.startswith(edge_prefix) for key in state):
        edge_output_dim = int(state[f"{edge_prefix}net.3.bias"].numel())
        edge_plane = node_count * node_count
        if edge_output_dim % edge_plane:
            raise ValueError(
                f"Edge-head output {edge_output_dim} is not divisible by "
                f"{edge_plane}."
            )
        edge_dim = edge_output_dim // edge_plane
        edge_decoder = EdgeFeatureDecoder(latent_dim, node_count, edge_dim)
        edge_state = {
            remove_prefix(key, edge_prefix): value
            for key, value in state.items()
            if key.startswith(edge_prefix)
        }
        edge_decoder.load_state_dict(edge_state, strict=True)

    modules = (adjacency_decoder, node_decoder, edge_decoder)
    for module in modules:
        if module is not None:
            module.to(device)
            module.eval()
    return adjacency_decoder, node_decoder, edge_decoder, latent_dim, node_count


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_graphs(
    checkpoint_path: Path,
    config: dict,
    *,
    count: int,
    node_feature_info,
    edge_feature_info,
    device: torch.device,
    seed: int,
    batch_size: int,
):
    state = checkpoint_state_dict(checkpoint_path)
    (
        adjacency_decoder,
        node_decoder,
        edge_decoder,
        latent_dim,
        node_count,
    ) = build_decoders(state, config, device)
    seed_everything(seed)
    generated = []
    attempts = 0
    max_attempts = max(10 * count, batch_size)
    with torch.no_grad():
        while len(generated) < count and attempts < max_attempts:
            current = min(batch_size, max_attempts - attempts)
            latent = torch.randn(current, latent_dim, device=device)
            adjacency = torch.sigmoid(adjacency_decoder(latent)).cpu().numpy()
            node_logits = node_decoder(latent).cpu().numpy()
            edge_logits = (
                None if edge_decoder is None else edge_decoder(latent).cpu().numpy()
            )
            attempts += current
            for index in range(current):
                attributed = graph_from_dense_attributes(
                    adjacency[index],
                    node_logits[index],
                    None if edge_logits is None else edge_logits[index],
                    node_feature_info=node_feature_info,
                    edge_feature_info=edge_feature_info,
                    values_are_logits=True,
                    adjacency_threshold=0.5,
                )
                if attributed is None:
                    continue
                generated.append(
                    attributed_arrays_to_pyg(
                        attributed.edges,
                        attributed.node_attributes,
                        attributed.edge_attributes,
                        attributed.source_node_ids,
                        name=f"generated graph {len(generated)}",
                    )
                )
                if len(generated) == count:
                    break
    if len(generated) != count:
        raise RuntimeError(
            f"Only generated {len(generated)}/{count} non-empty graphs after "
            f"{attempts} attempts for {checkpoint_path}."
        )
    dimensions = {
        "latent_dim": latent_dim,
        "max_nodes": node_count,
        "node_feature_dim": int(generated[0].x.shape[1]),
        "edge_feature_dim": (
            0
            if generated[0].edge_attr is None
            else int(generated[0].edge_attr.shape[1])
        ),
    }
    return generated, attempts, dimensions


def discover_runs(gather_root: Path):
    runs = []
    for dataset in DATASET_ORDER:
        for run_dir in sorted(
            (gather_root / "datasets" / dataset).glob("setting_*/seed_*")
        ):
            if not run_dir.is_dir():
                continue
            config = run_dir / "run_config_used.yaml"
            checkpoint = run_dir / "best_validation_mmd_model"
            if config.is_file() and checkpoint.is_file():
                seed_text = remove_prefix(run_dir.name, "seed_")
                runs.append(
                    {
                        "dataset_token": dataset,
                        "dataset": CANONICAL_DATASETS[dataset],
                        "setting": run_dir.parent.name,
                        "generator_seed": int(seed_text),
                        "run_dir": run_dir.resolve(),
                        "config": config.resolve(),
                        "checkpoint": checkpoint.resolve(),
                    }
                )
    return runs


def write_manifest_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "dataset",
        "dataset_token",
        "setting",
        "generator_seed",
        "status",
        "error",
        "run_dir",
        "config",
        "checkpoint",
        "checkpoint_sha256",
        "generation_seed",
        "reference_graph_count",
        "generation_attempts",
        "feature_mode",
        "feature_schema",
        "latent_dim",
        "max_nodes",
        "node_feature_dim",
        "edge_feature_dim",
        "generated",
        "reference",
        "evaluation_dir",
        "evaluator_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gather-root",
        type=Path,
        default=Path("/local-scratch2/new/gather"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/local-scratch2/new/gather/pretrained_graphcl_evaluation"
        ),
    )
    parser.add_argument("--ogb-cache", type=Path, default=DEFAULT_OGB_CACHE)
    parser.add_argument(
        "--proteins-real-dir",
        type=Path,
        default=Path(
            "/local-scratch2/new/gather/pretrained_graphcl_evaluation/"
            "real/PROTEINS"
        ),
    )
    parser.add_argument(
        "--additional-proteins-real-dir",
        action="append",
        type=Path,
        default=[
            Path(
                "/local-scratch2/new/gather/pretrained_graphcl_evaluation/"
                "real/PROTEINS_selection_seed_1"
            )
        ],
    )
    parser.add_argument("--generation-seed", type=int, default=20260727)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    gather_root = args.gather_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")

    cache_by_dataset = {}
    cache_payload_by_dataset = {}
    reference_by_dataset = {}
    reference_summary = {}
    proteins_reference_by_run = {}
    for dataset in DATASET_ORDER:
        if dataset == "proteins":
            protein_real_dirs = [
                args.proteins_real_dir.expanduser().resolve(),
                *[
                    path.expanduser().resolve()
                    for path in args.additional_proteins_real_dir
                ],
            ]
            (
                proteins_reference_by_run,
                validation,
            ) = validate_proteins_references(
                gather_root, protein_real_dirs
            )
            reference_summary[dataset] = validation
            continue
        cache_path = cache_for_dataset(
            gather_root, dataset, args.ogb_cache.expanduser().resolve()
        )
        cache_by_dataset[dataset] = cache_path
        cache = load_pickle(cache_path)
        cache_payload_by_dataset[dataset] = cache
        real_dir = output_root / "real" / CANONICAL_DATASETS[dataset]
        real_dir.mkdir(parents=True, exist_ok=True)
        exported = export_cache_references(cache, dataset, real_dir)
        reference_by_dataset[dataset] = Path(exported["test"]["path"])
        reference_summary[dataset] = {
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "exports": exported,
        }

    runs = discover_runs(gather_root)
    if not runs:
        raise ValueError(f"No complete gathered runs found below {gather_root}.")
    rows = []
    for run in runs:
        dataset = run["dataset_token"]
        reference_path = (
            proteins_reference_by_run[str(run["run_dir"])]
            if dataset == "proteins"
            else reference_by_dataset[dataset]
        )
        reference_graphs = load_pyg_collection(reference_path)
        run_output = (
            output_root
            / "runs"
            / dataset
            / run["setting"]
            / f"seed_{run['generator_seed']}"
        )
        run_output.mkdir(parents=True, exist_ok=True)
        generated_path = run_output / "generated_attributed_graphs.pt"
        evaluation_dir = run_output / "graphcl_evaluation"
        row = {
            **{key: str(value) if isinstance(value, Path) else value for key, value in run.items()},
            "status": "prepared",
            "error": "",
            "checkpoint_sha256": sha256(run["checkpoint"]),
            "generation_seed": args.generation_seed,
            "reference_graph_count": len(reference_graphs),
            "generation_attempts": "",
            "feature_mode": FEATURE_MODES[dataset],
            "feature_schema": FEATURE_SCHEMAS[dataset],
            "latent_dim": "",
            "max_nodes": "",
            "node_feature_dim": "",
            "edge_feature_dim": "",
            "generated": str(generated_path.resolve()),
            "reference": str(reference_path.resolve()),
            "evaluation_dir": str(evaluation_dir.resolve()),
            "evaluator_source": (
                "fresh_exact_schema_graphcl_3seed"
                if dataset == "enzymes"
                else "bundled_graphcl_gin_3seed"
            ),
        }
        try:
            if args.skip_existing and generated_path.is_file():
                generated_graphs = load_pyg_collection(generated_path)
                manifest = json.loads(
                    generated_path.with_suffix(".pt.json").read_text()
                )
                metadata = manifest.get("metadata", {})
                row["generation_attempts"] = metadata.get(
                    "generation_attempts", ""
                )
                summary = manifest["summary"]
                row["node_feature_dim"] = summary["node_feature_dim"]
                row["edge_feature_dim"] = summary["edge_feature_dim"]
                row["latent_dim"] = metadata.get("latent_dim", "")
                row["max_nodes"] = metadata.get("max_nodes", "")
                if len(generated_graphs) != len(reference_graphs):
                    raise ValueError(
                        "Existing generated/reference counts differ: "
                        f"{len(generated_graphs)} versus {len(reference_graphs)}."
                    )
            else:
                config = load_config(run["config"])
                if dataset == "proteins":
                    node_feature_info = {
                        index: {
                            "feature_name": "node_feature",
                            "value": index,
                        }
                        for index in range(3)
                    }
                    edge_feature_info = None
                else:
                    cache = cache_payload_by_dataset[dataset]
                    node_feature_info = cache.get("node_onehot_info")
                    edge_feature_info = cache.get("edge_onehot_info")
                generated_graphs, attempts, dimensions = generate_graphs(
                    run["checkpoint"],
                    config,
                    count=len(reference_graphs),
                    node_feature_info=node_feature_info,
                    edge_feature_info=edge_feature_info,
                    device=device,
                    seed=args.generation_seed,
                    batch_size=args.batch_size,
                )
                manifest = save_pyg_collection(
                    generated_path,
                    generated_graphs,
                    metadata={
                        "dataset": run["dataset"],
                        "feature_mode": FEATURE_MODES[dataset],
                        "feature_schema": FEATURE_SCHEMAS[dataset],
                        "producer": (
                            "scripts/prepare_gathered_graphcl_campaign.py"
                        ),
                        "run_dir": str(run["run_dir"]),
                        "checkpoint": str(run["checkpoint"]),
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "generation_seed": args.generation_seed,
                        "generation_attempts": attempts,
                        **dimensions,
                    },
                )
                row["generation_attempts"] = attempts
                row.update(dimensions)
                actual_dims = (
                    manifest["summary"]["node_feature_dim"],
                    manifest["summary"]["edge_feature_dim"],
                )
                reference_manifest = json.loads(
                    reference_path.with_suffix(".pt.json").read_text()
                )
                reference_dims = (
                    reference_manifest["summary"]["node_feature_dim"],
                    reference_manifest["summary"]["edge_feature_dim"],
                )
                if actual_dims != reference_dims:
                    raise ValueError(
                        f"Generated/reference dimensions differ: "
                        f"{actual_dims} versus {reference_dims}."
                    )
        except Exception as exc:
            row["status"] = "prepare_failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(
            f"[{row['status']}] {dataset}/{run['setting']}/"
            f"seed_{run['generator_seed']}"
        )
        if row["error"]:
            print(f"  {row['error']}", file=sys.stderr)

    manifest_csv = output_root / "campaign_manifest.csv"
    write_manifest_csv(manifest_csv, rows)
    payload = {
        "schema_version": "gathered-graphcl-campaign-v1",
        "gather_root": str(gather_root),
        "output_root": str(output_root),
        "generation_seed": args.generation_seed,
        "device": str(device),
        "run_count": len(rows),
        "prepared_count": sum(row["status"] == "prepared" for row in rows),
        "failed_count": sum(row["status"] != "prepared" for row in rows),
        "cache_paths": {
            key: str(value) for key, value in cache_by_dataset.items()
        },
        "reference_summary": reference_summary,
        "manifest_csv": str(manifest_csv.resolve()),
    }
    (output_root / "preparation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["failed_count"]:
        raise SystemExit(f"{payload['failed_count']} run(s) failed preparation.")


if __name__ == "__main__":
    main()
