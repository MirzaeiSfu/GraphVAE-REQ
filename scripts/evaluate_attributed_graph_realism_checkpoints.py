#!/usr/bin/env python3
"""Post-hoc feature-aware Random-GIN evaluation for GraphVAE checkpoints.

The primary ``decoded_node_edge`` mode evaluates adjacency together with the
node and edge attributes produced from the *same latent sample*.  The other
modes are matched ablations.  This script deliberately does not construct
degree, clustering, square-clustering, or any other hand-made GIN features.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "graph_evaluation" / "src"))

from eval.attributed_gin import (  # noqa: E402
    FEATURE_MODES,
    AttributedGraph,
    evaluate_dgl_feature_modes,
    graph_from_dense_attributes,
    to_dgl_graph,
)
from resample_grid_checkpoints import (  # noqa: E402
    build_model,
    dataset_cache_path,
    load_cached_dataset,
    load_config,
)
from graphvae_attr_bo_fingerprints import (  # noqa: E402
    feature_schema_fingerprint,
    feature_schema_payload,
    graph_fingerprint,
    sha256_file,
    split_fingerprint,
)
from ggm_eval.adapters import attributed_arrays_to_pyg  # noqa: E402
from ggm_eval.io import save_pyg_collection  # noqa: E402


DEFAULT_CONFIG_FILENAME = "run_config_used.yaml"
DEFAULT_CHECKPOINT_CANDIDATES = (
    "best_validation_mmd_model",
    "model_19999_0",
    "model_9999_0",
)
DEFAULT_OUTPUT_DIRNAME = "attributed_random_gin_eval"
GENERATED_DGL_FILENAME = "generated_attributed_graphs.bin"
REFERENCE_DGL_FILENAME = "reference_attributed_graphs.bin"
GENERATED_PYG_FILENAME = "generated_attributed_graphs.pt"
REFERENCE_PYG_FILENAME = "reference_attributed_graphs.pt"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        required=True,
        help="Trained run directory. May be supplied multiple times.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Configuration used to build the model/cache. By default each run "
            f"uses <run-dir>/{DEFAULT_CONFIG_FILENAME}."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint path or filename inside each run. By default the "
            "best-validation checkpoint is used, with a final-model fallback."
        ),
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=None,
        help="Override dataset_cache_dir recorded in the training config.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Reference split. Default: held-out test.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=FEATURE_MODES,
        default=None,
        help=(
            "Feature ablations to run. Default: all four when edge attributes "
            "exist; otherwise topology_control and decoded_node."
        ),
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=1000,
        help="Maximum accepted reference/generated graphs. Default: 1000.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=16,
        help="Latent samples decoded per generation batch. Default: 16.",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=12345,
        help="Seed for post-hoc latent samples. Default: 12345.",
    )
    parser.add_argument(
        "--evaluator-seed",
        type=int,
        default=0,
        help="Base seed for matched Random-GIN initializations. Default: 0.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of matched Random-GIN initializations. Default: 10.",
    )
    parser.add_argument(
        "--nearest-k",
        type=int,
        default=5,
        help="Neighbourhood size for precision/recall. Default: 5.",
    )
    parser.add_argument(
        "--adjacency-threshold",
        type=float,
        default=0.5,
        help="Sigmoid adjacency probability threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example auto, cpu, cuda, or cuda:1. Default: auto.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <run-dir>/"
            f"{DEFAULT_OUTPUT_DIRNAME}. With multiple runs this acts as a root."
        ),
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Also save the aligned attributed graph collections as a compressed NPZ.",
    )
    parser.add_argument(
        "--save-dgl",
        action="store_true",
        help=(
            "Also save generated and reference DGL files accepted by "
            "evaluate_attributed_dgl_graphs.py."
        ),
    )
    parser.add_argument(
        "--save-pyg",
        action="store_true",
        help=(
            "Also save the exact generated and reference collections in the "
            "restricted tensor-only PyG interchange format."
        ),
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_arg!r} requested but CUDA is unavailable.")
    return torch.device(device_arg)


def resolve_config(run_dir: Path, config_arg: Path | None) -> Path:
    if config_arg is not None:
        path = config_arg.expanduser()
    else:
        path = run_dir / DEFAULT_CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    return path.resolve()


def resolve_checkpoint(run_dir: Path, checkpoint_arg: Path | None) -> Path:
    if checkpoint_arg is not None:
        expanded = checkpoint_arg.expanduser()
        candidates = [expanded]
        if not expanded.is_absolute():
            candidates.insert(0, run_dir / expanded)
    else:
        candidates = [run_dir / name for name in DEFAULT_CHECKPOINT_CANDIDATES]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Checkpoint not found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_output_dir(
    run_dir: Path,
    output_arg: Path | None,
    multiple_runs: bool,
) -> Path:
    if output_arg is None:
        return run_dir / DEFAULT_OUTPUT_DIRNAME
    if multiple_runs:
        return output_arg.expanduser() / run_dir.name
    return output_arg.expanduser()


def _cache_split(cache: dict, split: str) -> tuple[list, list, list | None]:
    if split == "test":
        adjacencies = cache["test_list_adj"]
        node_values = cache.get("list_noh_test")
        edge_values = cache.get("list_eoh_test")
        fallback_dataset = cache.get("list_test_graphs")
    else:
        adjacencies = cache["val_adj"]
        node_values = cache.get("list_noh_val")
        edge_values = cache.get("list_eoh_val")
        fallback_dataset = None

        # In the legacy two-way split, validation is the prefix of training.
        if node_values is None:
            train_values = cache.get("list_noh_train")
            if train_values is not None:
                node_values = train_values[: len(adjacencies)]
        if edge_values is None:
            train_values = cache.get("list_eoh_train")
            if train_values is not None:
                edge_values = train_values[: len(adjacencies)]

    if node_values is None and fallback_dataset is not None:
        node_values = getattr(fallback_dataset, "list_node_onehot", None)
    if edge_values is None and fallback_dataset is not None:
        edge_values = getattr(fallback_dataset, "list_edge_onehot", None)
    if node_values is None:
        raise ValueError(
            f"The cached {split} split has no node attributes. "
            "Feature-aware checkpoint evaluation cannot use topology-created substitutes."
        )
    if len(adjacencies) != len(node_values):
        raise ValueError(
            f"Cached {split} adjacency/node-feature lengths differ: "
            f"{len(adjacencies)} versus {len(node_values)}."
        )
    if edge_values is not None and len(adjacencies) != len(edge_values):
        raise ValueError(
            f"Cached {split} adjacency/edge-feature lengths differ: "
            f"{len(adjacencies)} versus {len(edge_values)}."
        )
    return list(adjacencies), list(node_values), (
        None if edge_values is None else list(edge_values)
    )


def evaluator_input_integrity(cache: dict, config: dict, split: str) -> dict:
    """Fingerprint the exact cache and feature inputs used by this evaluator."""

    adjacencies, node_values, edge_values = _cache_split(cache, split)
    graph_hashes = []
    for index, (adjacency, node_attributes) in enumerate(
        zip(adjacencies, node_values)
    ):
        graph_hashes.append(
            graph_fingerprint(
                adjacency,
                node_attributes,
                None if edge_values is None else edge_values[index],
                relation_axes={
                    "node": cache.get("node_onehot_info"),
                    "edge": cache.get("edge_onehot_info"),
                },
            )
        )
    node_dimension = len(cache.get("node_onehot_info") or {}) or int(
        np.asarray(node_values[0]).shape[-1]
    )
    edge_dimension = (
        0
        if edge_values is None
        else len(cache.get("edge_onehot_info") or {})
        or int(np.asarray(edge_values[0]).shape[-3])
    )
    node_schema = feature_schema_payload(
        cache.get("node_onehot_info"),
        total_dimension=node_dimension,
        dtype=str(np.asarray(node_values[0]).dtype),
    )
    edge_schema = feature_schema_payload(
        cache.get("edge_onehot_info"),
        total_dimension=edge_dimension,
        dtype=("none" if edge_values is None else str(np.asarray(edge_values[0]).dtype)),
    )
    cache_path = dataset_cache_path(config).expanduser().resolve()
    return {
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "split_fingerprint": split_fingerprint(graph_hashes),
        "split_graph_count": len(graph_hashes),
        "node_schema_fingerprint": feature_schema_fingerprint(node_schema),
        "edge_schema_fingerprint": feature_schema_fingerprint(edge_schema),
        "node_schema": node_schema,
        "edge_schema": edge_schema,
    }


def build_reference_graphs(
    cache: dict,
    split: str,
    max_graphs: int,
    adjacency_threshold: float,
) -> list[AttributedGraph]:
    adjacencies, node_values, edge_values = _cache_split(cache, split)
    reference_graphs = []
    for index, (adjacency, node_attributes) in enumerate(
        zip(adjacencies, node_values)
    ):
        if node_attributes is None:
            raise ValueError(f"Reference graph {index} has no node attributes.")
        edge_attributes = None if edge_values is None else edge_values[index]
        graph = graph_from_dense_attributes(
            adjacency,
            node_attributes,
            edge_attributes,
            node_feature_info=cache.get("node_onehot_info"),
            edge_feature_info=cache.get("edge_onehot_info"),
            values_are_logits=False,
            adjacency_threshold=adjacency_threshold,
        )
        if graph is not None:
            reference_graphs.append(graph)
        if max_graphs > 0 and len(reference_graphs) >= max_graphs:
            break
    if len(reference_graphs) < 2:
        raise ValueError(
            f"Only {len(reference_graphs)} non-empty reference graphs were retained."
        )
    return reference_graphs


def _seed_generation(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_attributed_graphs(
    model,
    count: int,
    device: torch.device,
    cache: dict,
    adjacency_threshold: float,
    batch_size: int,
    reference_edge_feature_dim: int,
) -> tuple[list[AttributedGraph], int]:
    if model.node_feature_decoder is None:
        raise ValueError(
            "This model has no node-feature decoder. The attributed evaluator "
            "will not replace it with hand-made topology features."
        )

    generated_graphs: list[AttributedGraph] = []
    attempted = 0
    max_attempts = max(count * 10, batch_size)
    with torch.no_grad():
        while len(generated_graphs) < count and attempted < max_attempts:
            current_batch_size = min(
                batch_size,
                max_attempts - attempted,
                max(count - len(generated_graphs), 1),
            )
            latent = torch.randn(
                current_batch_size,
                model.embeding_dim,
                device=device,
                dtype=torch.float32,
            )
            adjacency_probabilities = torch.sigmoid(model.decode(latent))
            node_logits = model.node_feature_decoder(latent)
            edge_logits = (
                model.edge_feature_decoder(latent)
                if model.edge_feature_decoder is not None
                else None
            )

            adjacency_batch = adjacency_probabilities.detach().cpu().numpy()
            node_batch = node_logits.detach().cpu().numpy()
            edge_batch = (
                None if edge_logits is None else edge_logits.detach().cpu().numpy()
            )
            attempted += current_batch_size

            for batch_index in range(current_batch_size):
                graph = graph_from_dense_attributes(
                    adjacency_batch[batch_index],
                    node_batch[batch_index],
                    None if edge_batch is None else edge_batch[batch_index],
                    node_feature_info=cache.get("node_onehot_info"),
                    edge_feature_info=cache.get("edge_onehot_info"),
                    values_are_logits=True,
                    adjacency_threshold=adjacency_threshold,
                )
                if (
                    graph is not None
                    and edge_batch is None
                    and reference_edge_feature_dim
                ):
                    # Preserve one common GIN architecture for node-only modes.
                    # These zeros are controls, not synthesized edge features.
                    graph = AttributedGraph(
                        edges=graph.edges,
                        node_attributes=graph.node_attributes,
                        edge_attributes=np.zeros(
                            (graph.num_edges, reference_edge_feature_dim),
                            dtype=np.float32,
                        ),
                        source_node_ids=graph.source_node_ids,
                    )
                if graph is not None:
                    generated_graphs.append(graph)
                if len(generated_graphs) >= count:
                    break

    if len(generated_graphs) < count:
        raise RuntimeError(
            f"Only {len(generated_graphs)}/{count} non-empty generated graphs "
            f"were retained after {attempted} latent samples."
        )
    return generated_graphs, attempted


def _checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> dict:
    payload = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload in {checkpoint_path}.")
    return payload


def validate_feature_heads(state_dict: dict):
    has_node_head = any(key.startswith("node_feature_decoder.") for key in state_dict)
    has_edge_head = any(key.startswith("edge_feature_decoder.") for key in state_dict)
    if not has_node_head:
        raise ValueError(
            "Checkpoint has no node_feature_decoder parameters. It cannot be "
            "reevaluated with decoded node attributes."
        )
    return has_node_head, has_edge_head


def select_modes(
    requested_modes: Sequence[str] | None,
    edge_feature_dim: int,
    checkpoint_has_edge_head: bool,
) -> list[str]:
    if requested_modes is None:
        if edge_feature_dim and checkpoint_has_edge_head:
            return list(FEATURE_MODES)
        return ["topology_control", "decoded_node"]

    modes = list(dict.fromkeys(requested_modes))
    edge_modes = {"decoded_edge", "decoded_node_edge"}
    if edge_modes.intersection(modes) and not checkpoint_has_edge_head:
        raise ValueError(
            "An edge-feature mode was requested, but the checkpoint has no "
            "edge_feature_decoder parameters."
        )
    if edge_modes.intersection(modes) and edge_feature_dim == 0:
        raise ValueError(
            "An edge-feature mode was requested, but the reference split has "
            "no edge attributes."
        )
    return modes


def _json_graph_arrays(graphs: Sequence[AttributedGraph]) -> dict[str, np.ndarray]:
    return {
        "edges": np.asarray([graph.edges for graph in graphs], dtype=object),
        "node_attributes": np.asarray(
            [graph.node_attributes for graph in graphs], dtype=object
        ),
        "edge_attributes": np.asarray(
            [graph.edge_attributes for graph in graphs], dtype=object
        ),
        "source_node_ids": np.asarray(
            [graph.source_node_ids for graph in graphs], dtype=object
        ),
    }


def save_graph_collections(
    output_path: Path,
    generated_graphs: Sequence[AttributedGraph],
    reference_graphs: Sequence[AttributedGraph],
):
    generated = _json_graph_arrays(generated_graphs)
    reference = _json_graph_arrays(reference_graphs)
    np.savez_compressed(
        output_path,
        **{f"generated_{key}": value for key, value in generated.items()},
        **{f"reference_{key}": value for key, value in reference.items()},
    )


def save_dgl_graph_collections(
    output_dir: Path,
    generated_graphs: Sequence,
    reference_graphs: Sequence,
) -> dict[str, str]:
    """Save the full-feature DGL collections used by the evaluator."""

    try:
        import dgl
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("DGL is required to save attributed graph files.") from exc

    generated_path = (output_dir / GENERATED_DGL_FILENAME).resolve()
    reference_path = (output_dir / REFERENCE_DGL_FILENAME).resolve()
    dgl.save_graphs(str(generated_path), list(generated_graphs))
    dgl.save_graphs(str(reference_path), list(reference_graphs))
    return {
        "generated": str(generated_path),
        "reference": str(reference_path),
    }


def save_pyg_graph_collections(
    output_dir: Path,
    generated_graphs: Sequence[AttributedGraph],
    reference_graphs: Sequence[AttributedGraph],
    *,
    metadata: dict,
) -> dict[str, dict]:
    """Save the exact attributed collections for matched evaluator reuse."""

    def convert(graphs: Sequence[AttributedGraph], collection_role: str):
        return [
            attributed_arrays_to_pyg(
                graph.edges,
                graph.node_attributes,
                graph.edge_attributes,
                graph.source_node_ids,
                name=f"{collection_role} graph {index}",
            )
            for index, graph in enumerate(graphs)
        ]

    exports = {}
    for role, filename, graphs in (
        ("generated", GENERATED_PYG_FILENAME, generated_graphs),
        ("reference", REFERENCE_PYG_FILENAME, reference_graphs),
    ):
        path = (output_dir / filename).resolve()
        manifest = save_pyg_collection(
            path,
            convert(graphs, role),
            metadata={**metadata, "collection_role": role},
        )
        exports[role] = {
            "path": str(path),
            "manifest": str(path.with_suffix(path.suffix + ".json")),
            "collection_sha256": manifest["collection_sha256"],
            "summary": manifest["summary"],
        }
    return exports


def write_csv(output_path: Path, payload: dict):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("mode", "metric", "mean", "std", "min", "max"),
    )
    writer.writeheader()
    for mode, mode_result in payload["evaluation"]["modes"].items():
        for metric, summary in mode_result["summary"].items():
            writer.writerow({"mode": mode, "metric": metric, **summary})
    _atomic_write_text(output_path, buffer.getvalue())


def write_markdown(output_path: Path, payload: dict):
    lines = [
        "# Attributed Random-GIN Evaluation",
        "",
        (
            "The primary `decoded_node_edge` result consumes adjacency plus node "
            "and edge attributes decoded from the same latent sample. No degree, "
            "clustering, square-clustering, or other hand-made attributes are used."
        ),
        "",
        f"- Run: `{payload['run_dir']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Reference split: `{payload['split']}`",
        f"- Accepted graphs: `{payload['graph_counts']['accepted_per_collection']}`",
        f"- Node feature dimension: `{payload['evaluation']['feature_dimensions']['node']}`",
        f"- Edge feature dimension: `{payload['evaluation']['feature_dimensions']['edge']}`",
        f"- Random-GIN repeats: `{payload['evaluation']['repeats']}`",
        f"- Primary mode: `{payload['primary_mode']}`",
        (
            "- Primary attributed F1-PR: "
            f"`{payload['attributed_f1_pr']['mean']:.6f} ± "
            f"{payload['attributed_f1_pr']['std']:.6f}`"
        ),
        "",
        "Higher is better for F1-PR, precision, and recall. Lower is better for MMD.",
        "",
        "| Mode | F1-PR | Precision | Recall | MMD-RBF | MMD-linear |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, mode_result in payload["evaluation"]["modes"].items():
        summaries = mode_result["summary"]
        cells = []
        for metric in ("f1_pr", "precision", "recall", "mmd_rbf", "mmd_linear"):
            summary = summaries[metric]
            cells.append(f"{summary['mean']:.6f} ± {summary['std']:.6f}")
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- `topology_control`: adjacency only, represented by a fixed node "
                "constant and zero edge attributes in the full feature dimensions."
            ),
            (
                "- `decoded_node`: adjacency plus decoded/reference node attributes; "
                "edge attributes are zeroed."
            ),
            (
                "- `decoded_edge`: adjacency plus decoded/reference edge attributes; "
                "node attributes are fixed constants."
            ),
            (
                "- `decoded_node_edge`: adjacency plus both decoded/reference "
                "attribute types; this is the primary attributed result."
            ),
        ]
    )
    _atomic_write_text(output_path, "\n".join(lines) + "\n")


def evaluate_run(args: argparse.Namespace, run_dir: Path, multiple_runs: bool) -> Path:
    run_dir = run_dir.expanduser().resolve()
    config_path = resolve_config(run_dir, args.config)
    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint)
    output_dir = resolve_output_dir(run_dir, args.output_dir, multiple_runs).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    if args.dataset_cache_dir is not None:
        config["dataset_cache_dir"] = str(args.dataset_cache_dir.expanduser().resolve())
    cache = load_cached_dataset(config)
    integrity = evaluator_input_integrity(cache, config, args.split)

    device = resolve_device(args.device)
    state_dict = _checkpoint_state_dict(checkpoint_path, device)
    _, checkpoint_has_edge_head = validate_feature_heads(state_dict)
    model = build_model(config, cache, device)
    if model.node_feature_decoder is None:
        raise ValueError(
            "The config/cache reconstruction did not create a node-feature decoder, "
            "although the checkpoint contains one. Check that the original config "
            "and matching dataset cache were supplied."
        )
    if checkpoint_has_edge_head != (model.edge_feature_decoder is not None):
        raise ValueError(
            "The checkpoint and reconstructed config/cache disagree about the "
            "presence of the edge-feature decoder. Supply the exact training "
            "config and matching dataset cache."
        )
    model.load_state_dict(state_dict)
    model.eval()

    reference_graphs = build_reference_graphs(
        cache,
        args.split,
        args.max_graphs,
        args.adjacency_threshold,
    )
    edge_dim = reference_graphs[0].edge_feature_dim
    _seed_generation(args.generation_seed)
    generated_graphs, attempted = generate_attributed_graphs(
        model,
        len(reference_graphs),
        device,
        cache,
        args.adjacency_threshold,
        args.generation_batch_size,
        edge_dim,
    )

    modes = select_modes(args.modes, edge_dim, checkpoint_has_edge_head)
    full_feature_mode = "decoded_node_edge" if edge_dim else "decoded_node"
    generated_dgl = [
        to_dgl_graph(graph, full_feature_mode) for graph in generated_graphs
    ]
    reference_dgl = [
        to_dgl_graph(graph, full_feature_mode) for graph in reference_graphs
    ]
    evaluation = evaluate_dgl_feature_modes(
        generated_dgl,
        reference_dgl,
        modes=modes,
        repeats=args.repeats,
        seed=args.evaluator_seed,
        nearest_k=args.nearest_k,
        device=device,
    )
    evaluation["actual_decoder_output_dimensions"] = {
        "node": generated_graphs[0].node_feature_dim,
        "edge": generated_graphs[0].edge_feature_dim,
    }
    evaluation["evaluator_seed"] = args.evaluator_seed
    evaluation["evaluator_seeds"] = [
        args.evaluator_seed + repeat for repeat in range(args.repeats)
    ]
    evaluation["repeats"] = args.repeats
    primary_mode = (
        "decoded_node_edge"
        if "decoded_node_edge" in modes
        else "decoded_node"
        if "decoded_node" in modes
        else modes[0]
    )
    dgl_exports = None
    if args.save_dgl:
        dgl_exports = save_dgl_graph_collections(
            output_dir,
            generated_dgl,
            reference_dgl,
        )
    pyg_exports = None
    if args.save_pyg:
        cache_metadata = cache.get("cache_metadata") or {}
        pyg_exports = save_pyg_graph_collections(
            output_dir,
            generated_graphs,
            reference_graphs,
            metadata={
                "dataset": cache_metadata.get("dataset", config.get("dataset")),
                "feature_mode": full_feature_mode,
                "feature_schema": cache_metadata.get("feature_schema"),
                "split": args.split,
                "test_access": args.split == "test",
                "generation_seed": int(args.generation_seed),
                "source_cache_sha256": integrity["cache_sha256"],
                "split_fingerprint": integrity["split_fingerprint"],
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "producer": "scripts/evaluate_attributed_graph_realism_checkpoints.py",
            },
        )

    payload = {
        "schema_version": "attributed-random-gin-v1",
        "run_dir": str(run_dir),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "dataset_cache_dir": config.get("dataset_cache_dir"),
        "split": args.split,
        "device": str(device),
        "adjacency_threshold": args.adjacency_threshold,
        "generation_seed": args.generation_seed,
        "evaluator_seed": args.evaluator_seed,
        "evaluator_seeds": [
            args.evaluator_seed + repeat for repeat in range(args.repeats)
        ],
        "graph_counts": {
            "accepted_per_collection": len(reference_graphs),
            "generated_accepted": len(generated_graphs),
            "reference_accepted": len(reference_graphs),
            "validation_cache_count": integrity["split_graph_count"],
            "generation_attempts": attempted,
        },
        "feature_source": {
            "generated": (
                "GraphVAE node_feature_decoder and edge_feature_decoder"
                if checkpoint_has_edge_head
                else "GraphVAE node_feature_decoder"
            ),
            "reference": (
                "cached dataset node and edge one-hot attributes"
                if edge_dim
                else "cached dataset node one-hot attributes"
            ),
            "categorical_decoding": "argmax independently within each feature group",
            "hand_made_topology_features": False,
        },
        "implementation": {
            "feature_extractor": "third_party/ggmeval Random-GIN",
            "precision_recall": "third_party/ggmeval prdcEvaluation",
        },
        "primary_mode": primary_mode,
        "integrity": integrity,
        "attributed_f1_pr": evaluation["modes"][primary_mode]["summary"]["f1_pr"],
        "evaluation": evaluation,
    }
    if dgl_exports is not None:
        payload["dgl_exports"] = dgl_exports
    if pyg_exports is not None:
        payload["pyg_exports"] = pyg_exports

    json_path = output_dir / "attributed_random_gin.json"
    _atomic_write_text(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(output_dir / "attributed_random_gin_summary.csv", payload)
    write_markdown(output_dir / "attributed_random_gin_report.md", payload)
    if args.save_samples:
        save_graph_collections(
            output_dir / "attributed_graph_samples.npz",
            generated_graphs,
            reference_graphs,
        )
    return json_path


def main():
    args = parse_args()
    if args.max_graphs in {1, 2} or args.max_graphs < 0:
        raise ValueError("--max-graphs must be 0 (all) or at least 3.")
    if args.generation_batch_size < 1:
        raise ValueError("--generation-batch-size must be positive.")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    if args.nearest_k < 1:
        raise ValueError("--nearest-k must be positive.")
    if not 0.0 <= args.adjacency_threshold <= 1.0:
        raise ValueError("--adjacency-threshold must be in [0, 1].")

    multiple_runs = len(args.run_dir) > 1
    failures = []
    for raw_run_dir in args.run_dir:
        try:
            output_path = evaluate_run(args, raw_run_dir, multiple_runs)
            print(f"[AttributedGIN] Wrote {output_path}")
        except Exception as exc:
            failures.append((raw_run_dir, exc))
            print(f"[AttributedGIN] FAILED {raw_run_dir}: {exc}", file=sys.stderr)

    if failures:
        details = "; ".join(f"{run_dir}: {exc}" for run_dir, exc in failures)
        raise SystemExit(f"{len(failures)} run(s) failed. {details}")


if __name__ == "__main__":
    main()
