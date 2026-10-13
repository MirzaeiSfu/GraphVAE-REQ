#!/usr/bin/env python3
"""Export aligned attributed LOBSTER graphs for trained GraphCL-GIN evaluation.

The generated adjacency, node logits, and edge logits are decoded from the
same latent draw.  The exporter requires the resulting topology to match the
already-frozen rollout-0 adjacency artifact exactly.  It also audits whether
the decoded old_v1 categorical attributes agree with the labels recomputed
from the generated topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import BFS, data_split_three_way, list_graph_loader  # noqa: E402
from dataset_feature_utils import lobster_features  # noqa: E402
from eval.attributed_gin import (  # noqa: E402
    AttributedGraph,
    categorical_groups,
    graph_from_dense_attributes,
)
from model import GraphTransformerDecoder_FC  # noqa: E402
from util import EdgeFeatureDecoder, NodeFeatureDecoder, build_onehot_features  # noqa: E402


FEATURE_SCHEMA = "lobster-old-v1-node15-edge3-decoded-v1"
DEFAULT_CONDITIONS = (
    "lobster_graphvae_mm_fixed_split_native40_legacy",
    "lobster_kiarash_parity_kia40_2000_feature40_legacy",
    "lobster_semantic_hybrid_r001_legacy",
    "lobster_semantic_hybrid_r001_edgecount01_legacy",
)
CONDITION_LABELS = {
    "lobster_graphvae_mm_fixed_split_native40_legacy": "Matched manual Kiarash",
    "lobster_kiarash_parity_kia40_2000_feature40_legacy": "Motif bundle only",
    "lobster_semantic_hybrid_r001_legacy": "Relational 0.01",
    "lobster_semantic_hybrid_r001_edgecount01_legacy": (
        "Relational 0.01 + edge count 0.1"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        action="append",
        type=Path,
        required=True,
        help="Held-out rollout JSON. Repeat for multiple experiment families.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition",
        action="append",
        choices=DEFAULT_CONDITIONS,
        help="Condition to export. Default: all four matched 40/40 conditions.",
    )
    parser.add_argument(
        "--ggm-eval-src",
        type=Path,
        default=ROOT.parent
        / "GraphVAE-REQ-main-evaluation"
        / "graph_evaluation"
        / "src",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adjacency-threshold", type=float, default=0.5)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional original repository root embedded in selection paths.",
    )
    parser.add_argument(
        "--mapped-root",
        type=Path,
        default=None,
        help="Repository root containing copied artifacts from --source-root.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dense(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


def _feature_groups_by_name(
    info: Mapping,
) -> OrderedDict[str, tuple[int, ...]]:
    result: OrderedDict[str, list[int]] = OrderedDict()
    for raw_index, metadata in sorted(info.items(), key=lambda item: int(item[0])):
        result.setdefault(str(metadata["feature_name"]), []).append(int(raw_index))
    return OrderedDict((name, tuple(indices)) for name, indices in result.items())


def _feature_value_lookup(info: Mapping) -> dict[str, dict[int, int]]:
    """Map a feature's raw categorical ID to its one-hot channel."""

    result: dict[str, dict[int, int]] = {}
    for raw_index, metadata in sorted(info.items(), key=lambda item: int(item[0])):
        result.setdefault(str(metadata["feature_name"]), {})[
            int(metadata["value"])
        ] = int(raw_index)
    return result


def attributed_to_pyg(graph: AttributedGraph) -> Data:
    reverse_edges = graph.edges[:, ::-1]
    directed_edges = np.concatenate((graph.edges, reverse_edges), axis=0)
    directed_attributes = np.concatenate(
        (graph.edge_attributes, graph.edge_attributes),
        axis=0,
    )
    result = Data(
        x=torch.as_tensor(graph.node_attributes, dtype=torch.float32),
        edge_index=torch.as_tensor(directed_edges.T, dtype=torch.int64),
        edge_attr=torch.as_tensor(directed_attributes, dtype=torch.float32),
        num_nodes=graph.num_nodes,
    )
    result.source_node_ids = torch.as_tensor(
        graph.source_node_ids,
        dtype=torch.int64,
    )
    return result


def _graph_adjacency(graph: AttributedGraph) -> np.ndarray:
    adjacency = np.zeros((graph.num_nodes, graph.num_nodes), dtype=np.int8)
    if len(graph.edges):
        adjacency[graph.edges[:, 0], graph.edges[:, 1]] = 1
        adjacency[graph.edges[:, 1], graph.edges[:, 0]] = 1
    return adjacency


def _assert_exact_topology(
    graphs: Sequence[AttributedGraph],
    expected_path: Path,
) -> None:
    expected = np.load(expected_path, allow_pickle=True)
    if len(graphs) != len(expected):
        raise RuntimeError(
            f"Generated graph count differs from frozen topology artifact: "
            f"{len(graphs)} versus {len(expected)}."
        )
    for index, (actual_graph, expected_adjacency) in enumerate(
        zip(graphs, expected)
    ):
        actual = _graph_adjacency(actual_graph)
        expected_array = np.asarray(expected_adjacency, dtype=np.int8)
        if not np.array_equal(actual, expected_array):
            raise RuntimeError(
                f"Generated attributed topology {index} does not exactly match "
                f"{expected_path}."
            )


def _load_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload: {path}.")
    return payload


def _prefixed_state(
    state: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def load_decoders(
    checkpoint: Path,
    device: torch.device,
) -> tuple[GraphTransformerDecoder_FC, NodeFeatureDecoder, EdgeFeatureDecoder, dict]:
    state = _load_state(checkpoint)
    adjacency_state = _prefixed_state(state, "decode.")
    node_state = _prefixed_state(state, "node_feature_decoder.")
    edge_state = _prefixed_state(state, "edge_feature_decoder.")
    if not adjacency_state or not node_state or not edge_state:
        raise ValueError(
            f"{checkpoint} must contain adjacency, node, and edge decoders."
        )

    adjacency_output = adjacency_state["layers.3.bias"].numel()
    max_nodes = int(round(math.sqrt(adjacency_output)))
    if max_nodes * max_nodes != adjacency_output:
        raise ValueError("Adjacency decoder output is not a square matrix.")
    latent_dim = int(adjacency_state["layers.0.weight"].shape[1])
    node_output = int(node_state["net.3.bias"].numel())
    edge_output = int(edge_state["net.3.bias"].numel())
    if node_output % max_nodes:
        raise ValueError("Node decoder output does not divide by max_nodes.")
    if edge_output % (max_nodes * max_nodes):
        raise ValueError("Edge decoder output does not divide by max_nodes^2.")
    node_dim = node_output // max_nodes
    edge_dim = edge_output // (max_nodes * max_nodes)

    adjacency_decoder = GraphTransformerDecoder_FC(
        latent_dim,
        256,
        max_nodes,
        directed=True,
    ).to(device)
    node_decoder = NodeFeatureDecoder(
        latent_dim,
        max_nodes,
        node_dim,
    ).to(device)
    edge_decoder = EdgeFeatureDecoder(
        latent_dim,
        max_nodes,
        edge_dim,
    ).to(device)
    adjacency_decoder.load_state_dict(adjacency_state)
    node_decoder.load_state_dict(node_state)
    edge_decoder.load_state_dict(edge_state)
    adjacency_decoder.eval()
    node_decoder.eval()
    edge_decoder.eval()
    return adjacency_decoder, node_decoder, edge_decoder, {
        "latent_dim": latent_dim,
        "max_nodes": max_nodes,
        "node_feature_dim": node_dim,
        "edge_feature_dim": edge_dim,
    }


def decode_graphs(
    checkpoint: Path,
    *,
    frozen_topologies: Sequence[np.ndarray],
    seed: int,
    device: torch.device,
    node_info: Mapping,
    edge_info: Mapping,
    adjacency_threshold: float,
) -> tuple[list[AttributedGraph], dict, dict]:
    adjacency_decoder, node_decoder, edge_decoder, dimensions = load_decoders(
        checkpoint,
        device,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        latent = torch.randn(
            len(frozen_topologies),
            dimensions["latent_dim"],
            generator=generator,
            device=device,
        )
        adjacency = torch.sigmoid(adjacency_decoder(latent)).cpu().numpy()
        node_logits = node_decoder(latent).cpu().numpy()
        edge_logits = edge_decoder(latent).cpu().numpy()

    result: list[AttributedGraph] = []
    redecoded_exact = 0
    redecoded_nonexact_indices = []
    redecoded_edge_symmetric_differences = []
    for index, frozen_topology in enumerate(frozen_topologies):
        redecoded_graph = graph_from_dense_attributes(
            adjacency[index],
            node_logits[index],
            edge_logits[index],
            node_feature_info=node_info,
            edge_feature_info=edge_info,
            values_are_logits=True,
            adjacency_threshold=adjacency_threshold,
        )
        if redecoded_graph is None:
            raise RuntimeError(
                f"Latent sample {index} produced no nontrivial component."
            )
        frozen_topology = np.asarray(frozen_topology, dtype=np.int8)
        redecoded_topology = _graph_adjacency(redecoded_graph)
        if np.array_equal(redecoded_topology, frozen_topology):
            redecoded_exact += 1
            redecoded_edge_symmetric_differences.append(0)
        else:
            redecoded_nonexact_indices.append(index)
            if redecoded_topology.shape == frozen_topology.shape:
                difference = int(
                    np.abs(
                        redecoded_topology.astype(np.int16)
                        - frozen_topology.astype(np.int16)
                    )[np.triu_indices_from(frozen_topology, k=1)].sum()
                )
            else:
                difference = None
            redecoded_edge_symmetric_differences.append(difference)

        # CUDA kernels can differ at the 0.5 adjacency threshold across GPU
        # models.  Keep the already-frozen evaluated topology authoritative,
        # while aligning its local nodes back to the decoder slots recovered
        # from this same latent draw.
        if redecoded_graph.num_nodes != frozen_topology.shape[0]:
            raise RuntimeError(
                f"Frozen/redecoded LCC node counts differ for sample {index}: "
                f"{frozen_topology.shape[0]} versus {redecoded_graph.num_nodes}."
            )
        frozen_full = np.zeros(
            (dimensions["max_nodes"], dimensions["max_nodes"]),
            dtype=np.float32,
        )
        source_ids = redecoded_graph.source_node_ids
        frozen_full[np.ix_(source_ids, source_ids)] = frozen_topology
        graph = graph_from_dense_attributes(
            frozen_full,
            node_logits[index],
            edge_logits[index],
            node_feature_info=node_info,
            edge_feature_info=edge_info,
            values_are_logits=True,
            adjacency_threshold=adjacency_threshold,
        )
        if graph is None:
            raise RuntimeError(f"Frozen topology {index} has no nontrivial component.")
        result.append(graph)
    return result, dimensions, {
        "exact_lcc_count": redecoded_exact,
        "total_count": len(frozen_topologies),
        "nonexact_indices": redecoded_nonexact_indices,
        "edge_symmetric_differences": redecoded_edge_symmetric_differences,
        "export_uses_frozen_topology": True,
    }


def _topology_labels(graph: AttributedGraph) -> tuple[dict[str, np.ndarray], np.ndarray]:
    topology = nx.Graph()
    topology.add_nodes_from(range(graph.num_nodes))
    topology.add_edges_from((int(u), int(v)) for u, v in graph.edges)
    spine_path = lobster_features.find_spine_path(topology)
    spine_nodes = set(spine_path)
    distance = lobster_features.compute_distance_to_spine_labels(
        topology,
        spine_path,
    )
    subtree = lobster_features.compute_branch_component_sizes(
        topology,
        spine_path,
    )
    node_labels = {
        "node_degree": np.asarray(
            [
                lobster_features.compute_node_degree(topology, node)
                for node in topology.nodes()
            ],
            dtype=np.int64,
        ),
        "distance_to_spine": np.asarray(
            [distance[node] for node in topology.nodes()],
            dtype=np.int64,
        ),
        "subtree_size": np.asarray(
            [subtree.get(node, 1) for node in topology.nodes()],
            dtype=np.int64,
        ),
        "eccentricity": np.asarray(
            [
                lobster_features.compute_eccentricity(topology, node)
                for node in topology.nodes()
            ],
            dtype=np.int64,
        ),
    }
    edge_labels = np.asarray(
        [
            lobster_features.compute_edge_type(int(u), int(v), spine_nodes)
            for u, v in graph.edges
        ],
        dtype=np.int64,
    )
    return node_labels, edge_labels


def semantic_consistency(
    graphs: Sequence[AttributedGraph],
    node_info: Mapping,
    edge_info: Mapping,
) -> dict:
    node_lookup = _feature_value_lookup(node_info)
    edge_lookup = _feature_value_lookup(edge_info)
    node_correct = {name: 0 for name in node_lookup}
    node_total = {name: 0 for name in node_lookup}
    node_all_correct = 0
    node_all_total = 0
    edge_correct = 0
    edge_total = 0

    for graph in graphs:
        expected_node, expected_edge = _topology_labels(graph)
        graph_all = np.ones(graph.num_nodes, dtype=bool)
        for name, raw_to_channel in node_lookup.items():
            # A generated topology may create a category absent from the
            # training corpus (notably distance_to_spine=4).  The fixed
            # decoder cannot express that class, so count it as inconsistent
            # instead of silently remapping it.
            expected_channels = np.asarray(
                [raw_to_channel.get(int(value), -1) for value in expected_node[name]],
                dtype=np.int64,
            )
            actual_channels = np.argmax(
                graph.node_attributes[
                    :,
                    np.asarray(tuple(raw_to_channel.values()), dtype=np.int64),
                ],
                axis=1,
            )
            group_channels = np.asarray(
                tuple(raw_to_channel.values()),
                dtype=np.int64,
            )
            actual_global_channels = group_channels[actual_channels]
            correct = actual_global_channels == expected_channels
            node_correct[name] += int(correct.sum())
            node_total[name] += int(len(correct))
            graph_all &= correct
        node_all_correct += int(graph_all.sum())
        node_all_total += int(len(graph_all))

        edge_name = next(iter(edge_lookup))
        raw_to_channel = edge_lookup[edge_name]
        group_channels = np.asarray(tuple(raw_to_channel.values()), dtype=np.int64)
        expected_channels = np.asarray(
            [raw_to_channel[int(value)] for value in expected_edge],
            dtype=np.int64,
        )
        actual_local = np.argmax(
            graph.edge_attributes[:, group_channels],
            axis=1,
        )
        actual_channels = group_channels[actual_local]
        edge_correct += int((actual_channels == expected_channels).sum())
        edge_total += int(len(expected_channels))

    return {
        "node_accuracy": {
            name: node_correct[name] / node_total[name]
            for name in node_correct
        },
        "node_all_features_accuracy": node_all_correct / node_all_total,
        "edge_accuracy": edge_correct / edge_total,
        "node_count": node_all_total,
        "edge_count": edge_total,
    }


def categorical_marginals(
    graphs: Sequence[AttributedGraph],
    node_info: Mapping,
    edge_info: Mapping,
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {"node": {}, "edge": {}}
    node_values = np.concatenate(
        [graph.node_attributes for graph in graphs],
        axis=0,
    )
    edge_values = np.concatenate(
        [graph.edge_attributes for graph in graphs],
        axis=0,
    )
    for name, indices in _feature_groups_by_name(node_info).items():
        counts = node_values[:, indices].sum(axis=0)
        result["node"][name] = (counts / counts.sum()).tolist()
    for name, indices in _feature_groups_by_name(edge_info).items():
        counts = edge_values[:, indices].sum(axis=0)
        result["edge"][name] = (counts / counts.sum()).tolist()
    return result


def _distribution_distances(
    actual: Mapping[str, Mapping[str, Sequence[float]]],
    reference: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict:
    result: dict[str, dict[str, dict[str, float]]] = {"node": {}, "edge": {}}
    epsilon = 1e-12
    for kind in ("node", "edge"):
        for name, raw_actual in actual[kind].items():
            p = np.asarray(raw_actual, dtype=np.float64)
            q = np.asarray(reference[kind][name], dtype=np.float64)
            midpoint = (p + q) / 2.0
            kl_p = np.sum(p * np.log((p + epsilon) / (midpoint + epsilon)))
            kl_q = np.sum(q * np.log((q + epsilon) / (midpoint + epsilon)))
            result[kind][name] = {
                "total_variation": float(0.5 * np.abs(p - q).sum()),
                "jensen_shannon": float(0.5 * (kl_p + kl_q)),
            }
    return result


def build_real_splits() -> tuple[
    dict[str, list[AttributedGraph]],
    Mapping,
    Mapping,
]:
    (
        adjacencies,
        features,
        labels,
        node_features,
        edge_features,
        node_feature_info,
        edge_feature_info,
    ) = list_graph_loader(
        "LOBSTER",
        return_labels=True,
        lobster_feature_schema="old_v1",
        shuffle_seed=0,
    )
    adjacencies, node_features, edge_features = BFS(
        adjacencies,
        node_features,
        edge_features,
    )
    (
        node_onehot,
        edge_onehot,
        node_onehot_info,
        edge_onehot_info,
    ) = build_onehot_features(
        node_features,
        edge_features,
        adjacencies,
        node_feature_info,
        edge_feature_info,
    )
    split = data_split_three_way(
        graph_lis=adjacencies,
        list_x=features,
        list_label=labels,
        list_node_onehot=node_onehot,
        list_edge_onehot=edge_onehot,
        train_fraction=0.7,
        val_fraction=0.1,
        seed=123,
    )
    split_values = {
        "train": (split[0], split[9], split[12]),
        "validation": (split[1], split[10], split[13]),
        "heldout_test": (split[2], split[11], split[14]),
    }
    result: dict[str, list[AttributedGraph]] = {}
    for name, (split_adj, split_node, split_edge) in split_values.items():
        graphs = []
        for index, (adjacency, node_values, edge_values) in enumerate(
            zip(split_adj, split_node, split_edge)
        ):
            graph = graph_from_dense_attributes(
                adjacency,
                node_values,
                edge_values,
                node_feature_info=node_onehot_info,
                edge_feature_info=edge_onehot_info,
                values_are_logits=False,
            )
            if graph is None:
                raise RuntimeError(f"Real {name} graph {index} has no edges.")
            graphs.append(graph)
        result[name] = graphs
    return result, node_onehot_info, edge_onehot_info


def _assert_reference_matches(
    graphs: Sequence[AttributedGraph],
    expected_path: Path,
) -> None:
    expected = np.load(expected_path, allow_pickle=True)
    if len(graphs) != len(expected):
        raise RuntimeError("Reconstructed held-out reference count differs.")
    for index, (graph, adjacency) in enumerate(zip(graphs, expected)):
        if not np.array_equal(_graph_adjacency(graph), np.asarray(adjacency)):
            raise RuntimeError(
                f"Reconstructed held-out reference graph {index} differs."
            )


def _selected_runs(
    paths: Sequence[Path],
    conditions: set[str],
) -> list[dict]:
    by_key: dict[tuple[str, int], dict] = {}
    for path in paths:
        payload = json.loads(path.expanduser().resolve().read_text())
        for row in payload["runs"]:
            condition = str(row["condition"])
            seed = int(row["seed"])
            if condition not in conditions:
                continue
            key = (condition, seed)
            if key in by_key:
                raise ValueError(f"Duplicate selected run: {key}.")
            result = dict(row)
            result["selection_source"] = str(path.expanduser().resolve())
            by_key[key] = result
    expected = {(condition, seed) for condition in conditions for seed in range(3)}
    if set(by_key) != expected:
        raise RuntimeError(
            f"Selected run matrix is incomplete: missing={sorted(expected-set(by_key))}, "
            f"extra={sorted(set(by_key)-expected)}."
        )
    return [by_key[key] for key in sorted(by_key)]


def _resolve_mapped_path(
    raw_path: str,
    source_root: Path | None,
    mapped_root: Path | None,
) -> Path:
    path = Path(raw_path).expanduser()
    if source_root is None and mapped_root is None:
        return path.resolve()
    if source_root is None or mapped_root is None:
        raise ValueError("--source-root and --mapped-root must be supplied together.")
    source = source_root.expanduser().resolve()
    try:
        relative = path.relative_to(source)
    except ValueError as exc:
        raise ValueError(f"{path} is not under mapped source root {source}.") from exc
    return (mapped_root.expanduser().resolve() / relative).resolve()


def _save_collection(path: Path, graphs: Sequence[AttributedGraph], metadata: dict):
    from ggm_eval import save_pyg_collection

    return save_pyg_collection(
        path,
        [attributed_to_pyg(graph) for graph in graphs],
        metadata={
            "dataset": "LOBSTER",
            "feature_schema": FEATURE_SCHEMA,
            "lobster_feature_schema": "old_v1",
            **metadata,
        },
    )


def main() -> None:
    args = parse_args()
    ggm_eval_src = args.ggm_eval_src.expanduser().resolve()
    if not (ggm_eval_src / "ggm_eval").is_dir():
        raise FileNotFoundError(f"ggm_eval package not found: {ggm_eval_src}.")
    sys.path.insert(0, str(ggm_eval_src))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    conditions = set(args.condition or DEFAULT_CONDITIONS)

    real_splits, node_info, edge_info = build_real_splits()
    if len(categorical_groups(node_info, 15)) != 4:
        raise RuntimeError("Expected four old_v1 node feature groups.")
    if len(categorical_groups(edge_info, 3)) != 1:
        raise RuntimeError("Expected one old_v1 edge feature group.")

    selected = _selected_runs(args.selection, conditions)
    first_reference = _resolve_mapped_path(
        selected[0]["reference_path"],
        args.source_root,
        args.mapped_root,
    )
    _assert_reference_matches(real_splits["heldout_test"], first_reference)

    common_real_metadata = {
        "source": "GraphVAE-REQ deterministic LOBSTER loader",
        "dataset_loader_seed": 0,
        "split_mode": "paper_70_10_20",
        "split_seed": 123,
        "bfs_strategy": "legacy_first_component",
        "attributes": "reference_old_v1_onehot",
    }
    manifests = {"real": {}, "generated": {}}
    for split_name, graphs in real_splits.items():
        destination = output_dir / f"real_{split_name}_graphs.pt"
        manifests["real"][split_name] = _save_collection(
            destination,
            graphs,
            {**common_real_metadata, "split": split_name},
        )

    reference_marginals = categorical_marginals(
        real_splits["heldout_test"],
        node_info,
        edge_info,
    )
    reference_consistency = semantic_consistency(
        real_splits["heldout_test"],
        node_info,
        edge_info,
    )
    if any(
        accuracy != 1.0
        for accuracy in reference_consistency["node_accuracy"].values()
    ) or reference_consistency["edge_accuracy"] != 1.0:
        raise RuntimeError(
            "Reference old_v1 labels failed their topology-consistency sanity check."
        )

    audits = []
    for index, row in enumerate(selected, start=1):
        condition = str(row["condition"])
        training_seed = int(row["seed"])
        checkpoint = _resolve_mapped_path(
            row["checkpoint_path"],
            args.source_root,
            args.mapped_root,
        )
        topology_path = _resolve_mapped_path(
            row["generated_rollout0_path"],
            args.source_root,
            args.mapped_root,
        )
        rollout_seed = int(row["rollouts"][0]["seed"])
        expected = np.load(topology_path, allow_pickle=True)
        print(
            f"[{index}/{len(selected)}] {condition} seed={training_seed}",
            flush=True,
        )
        graphs, dimensions, redecoded_topology_audit = decode_graphs(
            checkpoint,
            frozen_topologies=expected,
            seed=rollout_seed,
            device=device,
            node_info=node_info,
            edge_info=edge_info,
            adjacency_threshold=args.adjacency_threshold,
        )
        _assert_exact_topology(graphs, topology_path)
        generated_marginals = categorical_marginals(
            graphs,
            node_info,
            edge_info,
        )
        consistency = semantic_consistency(graphs, node_info, edge_info)
        condition_dir = output_dir / condition / f"seed_{training_seed}"
        destination = condition_dir / "generated_attributed_graphs.pt"
        manifest = _save_collection(
            destination,
            graphs,
            {
                "split": "generated_rollout0",
                "attributes": "decoded_old_v1_argmax",
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "training_seed": training_seed,
                "generation_seed": rollout_seed,
                "checkpoint_sha256": _sha256(checkpoint),
                "topology_source_sha256": _sha256(topology_path),
            },
        )
        audit = {
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "training_seed": training_seed,
            "generation_seed": rollout_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "frozen_topology": str(topology_path),
            "frozen_topology_sha256": _sha256(topology_path),
            "topology_exact_match": True,
            "redecoded_topology_audit": redecoded_topology_audit,
            "output": str(destination),
            "collection_sha256": manifest["collection_sha256"],
            "dimensions": dimensions,
            "semantic_consistency": consistency,
            "categorical_marginals": generated_marginals,
            "marginal_distance_to_heldout": _distribution_distances(
                generated_marginals,
                reference_marginals,
            ),
        }
        condition_dir.mkdir(parents=True, exist_ok=True)
        (condition_dir / "attribute_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audits.append(audit)
        manifests["generated"][f"{condition}/seed_{training_seed}"] = manifest

    campaign = {
        "feature_schema": FEATURE_SCHEMA,
        "conditions": sorted(conditions),
        "condition_labels": CONDITION_LABELS,
        "selection_sources": [
            str(path.expanduser().resolve()) for path in args.selection
        ],
        "node_onehot_info": node_info,
        "edge_onehot_info": edge_info,
        "heldout_reference": str(first_reference),
        "heldout_reference_sha256": _sha256(first_reference),
        "reference_semantic_consistency": reference_consistency,
        "reference_categorical_marginals": reference_marginals,
        "audits": audits,
        "manifests": manifests,
    }
    (output_dir / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote attributed campaign to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
