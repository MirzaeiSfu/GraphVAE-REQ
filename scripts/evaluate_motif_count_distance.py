#!/usr/bin/env python3
"""Evaluate relational motif-count distance on train-sized graph samples.

This preserves the legacy ``count distance`` calculation from the VGAE
repository:

    sqrt(mean((observed_counts - generated_counts) ** 2))

The original implementation compares two motif-count vectors for one graph.
GraphVAE datasets contain many graphs, so this script generates exactly one
graph for every training graph and compares the mean motif-count vectors.  It
also reports the RMSE between aggregate count vectors, which is the literal
VGAE formula after summing the graph dataset.  The aggregate value scales with
the number of graphs; the mean-vector value does not.

The recommended robust distance compares each rule's full per-graph count
distribution with a one-dimensional Wasserstein distance after ``log1p``
compression, then takes an upper-10%-trimmed mean across rules. This prevents a
single high-frequency motif from dominating while still detecting changes in
the shape of the count distribution.

Both views of the generated data are reported:

* ``soft``: explicitly sigmoid-converted decoder probabilities, following the
  VGAE evaluator's soft-count behaviour.
* ``hard``: adjacency thresholding, conversion to an undirected graph, and
  categorical feature argmax, with self-loops and isolated nodes removed and
  only the largest connected component retained on both sides.

Each generated graph is masked to the node count of its corresponding training
graph.  Every graph is counted at its exact, unpadded size so false relation
predicates cannot accidentally count padding.  The hard view may become smaller
after isolated-node removal.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

# Running ``python scripts/...py`` otherwise puts only ``scripts/`` on
# sys.path, while the project modules live one directory above it.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import DataWrapper, ReconstructedDataWrapper, _build_fom, merge_datasets
from model import GraphTransformerDecoder_FC
from motif_counting.motif_counter import RelationalMotifCounter
from motif_counting.motif_objective import (
    build_motif_group_objectives,
    restrict_to_nonzero_weight_motif_groups,
)
from motif_counting.motif_selection_manifest import (
    MOTIF_SELECTION_MANIFEST_FILENAME,
    apply_motif_selection_manifest,
    load_motif_selection_manifest,
)
from motif_counting.motif_representations import canonicalize_motif_output_mode
from util import EdgeFeatureDecoder, NodeFeatureDecoder


class TensorGraphBatch:
    """Minimal exact-size wrapper accepted by RelationalMotifCounter."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        relation_keys: Sequence[str],
        feature_onehot_mapping: Mapping[int, Mapping[int, int]],
        device: torch.device,
    ) -> None:
        if not records:
            raise ValueError("TensorGraphBatch requires at least one graph.")
        sizes = {int(record["features"].shape[0]) for record in records}
        if len(sizes) != 1:
            raise ValueError(f"All graphs in a TensorGraphBatch must match: {sizes}")

        self.device = str(device)
        self.relation_keys = list(relation_keys)
        self.feature_onehot_mapping = dict(feature_onehot_mapping)
        self.num_graphs = len(records)
        self.N_max = sizes.pop()
        self.all_features = torch.stack(
            [record["features"].detach().cpu() for record in records]
        )
        self.all_feat_onehot = torch.stack(
            [record["feat_onehot"].detach().cpu() for record in records]
        )
        self.all_adj = {
            relation: torch.stack(
                [record["adj"][relation].detach().cpu() for record in records]
            )
            for relation in self.relation_keys
        }

        first_edge = records[0]["edge"]
        if first_edge is None:
            self.all_edge = None
        else:
            self.all_edge = [
                torch.stack(
                    [record["edge"][edge_index].detach().cpu() for record in records]
                )
                for edge_index in range(len(first_edge))
            ]

    def get_batch(self, start: int, end: int):
        device = self.device
        features = self.all_features[start:end].to(device)
        feat_onehot = self.all_feat_onehot[start:end].to(device)
        adjacency = {
            relation: self.all_adj[relation][start:end].to(device)
            for relation in self.relation_keys
        }
        edge = (
            [tensor[start:end].to(device) for tensor in self.all_edge]
            if self.all_edge is not None
            else None
        )
        return features, feat_onehot, adjacency, edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one graph per training graph and compare aggregate "
            "relational motif-count vectors."
        )
    )
    parser.add_argument("--config", required=True, help="Run YAML configuration.")
    parser.add_argument(
        "--motif-selection-config",
        default=None,
        help=(
            "Optional motif=True run configuration that defines the exact "
            "training-time rule pruning and active motif groups. Defaults to "
            "--config. Use this when evaluating a motif=False checkpoint against "
            "the rule subset optimized by a corresponding motif=True run."
        ),
    )
    parser.add_argument(
        "--motif-selection-manifest",
        default=None,
        help=(
            "Exact post-pruning motif selection saved by training. When omitted, "
            "the evaluator automatically uses motif_selection_manifest.json next "
            "to the checkpoint if present, otherwise it reconstructs selection "
            "from the YAML and training cache for backward compatibility."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Trained model state dict, normally best_validation_mmd_model.",
    )
    parser.add_argument(
        "--dataset-cache",
        required=True,
        help="Exact processed dataset-cache pickle used by the run.",
    )
    parser.add_argument(
        "--motif-cache-dir",
        required=True,
        help="Directory containing <database_name>.pkl.",
    )
    parser.add_argument("--output", required=True, help="Destination JSON file.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=32,
        help="Latent samples decoded at once.",
    )
    parser.add_argument(
        "--count-batch-size",
        type=int,
        default=256,
        help="Graphs processed per motif-counter batch.",
    )
    parser.add_argument("--adj-threshold", type=float, default=0.5)
    parser.add_argument(
        "--setting",
        default=None,
        help="Optional report label such as 03_graphvae_motif_original_no_temp.",
    )
    parser.add_argument(
        "--dataset-label",
        default=None,
        help="Optional report label; defaults to config dataset.",
    )
    return parser.parse_args()


def flatten_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                if nested_key in flat:
                    raise ValueError(f"Duplicate YAML key after flattening: {nested_key}")
                flat[nested_key] = nested_value
        else:
            if key in flat:
                raise ValueError(f"Duplicate YAML key after flattening: {key}")
            flat[key] = value
    return flat


def load_yaml(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        nested = yaml.safe_load(handle) or {}
    if not isinstance(nested, Mapping):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return dict(nested), flatten_config(nested)


def configure_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")

    if isinstance(loaded, Mapping) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")

    state: Dict[str, torch.Tensor] = {}
    for key, value in loaded.items():
        clean_key = key[7:] if str(key).startswith("module.") else str(key)
        state[clean_key] = value
    return state


def prefixed_state(
    state: Mapping[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    marker = prefix + "."
    return {
        key[len(marker) :]: value
        for key, value in state.items()
        if key.startswith(marker)
    }


def first_non_none(values: Iterable[Any]) -> Optional[Any]:
    return next((value for value in values if value is not None), None)


def build_decoders(
    state: Mapping[str, torch.Tensor],
    flat_config: Mapping[str, Any],
    train_dataset: Any,
    device: torch.device,
) -> Tuple[
    GraphTransformerDecoder_FC,
    NodeFeatureDecoder,
    Optional[EdgeFeatureDecoder],
    int,
]:
    graph_dim = int(flat_config.get("graphEmDim", 1024))
    max_nodes = int(train_dataset.max_num_nodes)
    directed = bool(flat_config.get("directed", False))

    decoder = GraphTransformerDecoder_FC(graph_dim, 256, max_nodes, directed)
    decoder_state = prefixed_state(state, "decode")
    if not decoder_state:
        raise KeyError("Checkpoint has no decode.* parameters.")
    decoder.load_state_dict(decoder_state, strict=True)

    node_target = first_non_none(train_dataset.processed_node_onehot)
    if node_target is None:
        raise RuntimeError(
            "Training cache has no node one-hot features; relational feature "
            "motifs cannot be evaluated for generated graphs."
        )
    node_dim = int(node_target.shape[-1])
    node_decoder = NodeFeatureDecoder(graph_dim, max_nodes, node_dim)
    node_state = prefixed_state(state, "node_feature_decoder")
    if not node_state:
        if bool(flat_config.get("use_feature", True)):
            raise KeyError(
                "Checkpoint has no node_feature_decoder.* parameters; feature-aware "
                "motif counts cannot be generated."
            )
        # No-feature experiments deliberately do not train/save attribute
        # decoders.  Supply constant logits solely to satisfy the reconstructed
        # graph wrapper; relational motif rules do not consume these attributes.
        for parameter in node_decoder.parameters():
            torch.nn.init.zeros_(parameter)
    else:
        node_decoder.load_state_dict(node_state, strict=True)

    edge_decoder: Optional[EdgeFeatureDecoder] = None
    edge_target = first_non_none(train_dataset.processed_edge_onehot)
    edge_state = prefixed_state(state, "edge_feature_decoder")
    if edge_target is not None:
        if not edge_state:
            if bool(flat_config.get("use_feature", True)):
                raise KeyError(
                    "Training cache has edge features but checkpoint has no "
                    "edge_feature_decoder.* parameters."
                )
        edge_dim = int(edge_target.shape[0])
        edge_decoder = EdgeFeatureDecoder(graph_dim, max_nodes, edge_dim)
        if edge_state:
            edge_decoder.load_state_dict(edge_state, strict=True)
        else:
            for parameter in edge_decoder.parameters():
                torch.nn.init.zeros_(parameter)

    decoder.to(device).eval()
    node_decoder.to(device).eval()
    if edge_decoder is not None:
        edge_decoder.to(device).eval()

    return decoder, node_decoder, edge_decoder, graph_dim


def graph_record_from_wrapper(
    wrapper: Any,
    graph_index: int,
    node_count: int,
) -> Dict[str, Any]:
    """Extract one graph at its exact size, excluding every padded position."""
    keep = torch.arange(node_count, device=wrapper.all_features.device)
    return slice_graph_record(
        {
            "features": wrapper.all_features[graph_index],
            "feat_onehot": wrapper.all_feat_onehot[graph_index],
            "adj": {
                relation: wrapper.all_adj[relation][graph_index]
                for relation in wrapper.relation_keys
            },
            "edge": (
                [tensor[graph_index] for tensor in wrapper.all_edge]
                if wrapper.all_edge is not None
                else None
            ),
        },
        keep,
    )


def slice_graph_record(record: Mapping[str, Any], keep: torch.Tensor) -> Dict[str, Any]:
    """Reindex all node- and pair-valued tensors with the same node indices."""
    keep = keep.to(record["features"].device, dtype=torch.long)

    def square_slice(tensor: torch.Tensor) -> torch.Tensor:
        local_keep = keep.to(tensor.device)
        return tensor.index_select(-2, local_keep).index_select(-1, local_keep)

    return {
        "features": record["features"].index_select(0, keep),
        "feat_onehot": record["feat_onehot"].index_select(0, keep),
        "adj": {
            relation: square_slice(tensor)
            for relation, tensor in record["adj"].items()
        },
        "edge": (
            [square_slice(tensor) for tensor in record["edge"]]
            if record["edge"] is not None
            else None
        ),
    }


def hard_graph_postprocess(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply GraphVAE's discrete undirected graph conversion and cleanup."""
    processed = {
        "features": record["features"],
        "feat_onehot": record["feat_onehot"],
        "adj": {key: value.clone() for key, value in record["adj"].items()},
        "edge": (
            [value.clone() for value in record["edge"]]
            if record["edge"] is not None
            else None
        ),
    }
    # nx.from_numpy_array(..., create_using=Graph) creates an undirected edge if
    # either matrix direction is non-zero. Do that conversion explicitly so a
    # checkpoint trained with directed=True is still counted as the same graph
    # that GraphVAE writes/evaluates.
    for relation, adjacency in list(processed["adj"].items()):
        binary = adjacency != 0
        undirected = binary | binary.transpose(0, 1)
        undirected.fill_diagonal_(False)
        processed["adj"][relation] = undirected.to(adjacency.dtype)

    primary = next(iter(processed["adj"].values()))

    # Edge attributes are categorical. Average the two directional score/onehot
    # vectors, choose one label for the undirected pair, mirror it, and mask it
    # to actual edges. Supplying soft edge probabilities before this function
    # gives a deterministic tie-break based on decoder confidence.
    if processed["edge"] is not None:
        edge_mask = (primary != 0).unsqueeze(0)
        symmetric_edges = []
        for edge_feature in processed["edge"]:
            scores = (edge_feature + edge_feature.transpose(-2, -1)) / 2
            label = scores.argmax(dim=0)
            hardened = torch.nn.functional.one_hot(
                label, num_classes=edge_feature.shape[0]
            ).permute(2, 0, 1).to(edge_feature.dtype)
            symmetric_edges.append(hardened * edge_mask.to(hardened.dtype))
        processed["edge"] = symmetric_edges

    incident = primary.abs().sum(dim=0) + primary.abs().sum(dim=1)
    keep = torch.nonzero(incident > 0, as_tuple=False).flatten()
    processed = slice_graph_record(processed, keep)
    if keep.numel() == 0:
        return processed

    primary = next(iter(processed["adj"].values()))
    undirected = ((primary != 0) | (primary.transpose(0, 1) != 0)).detach().cpu()
    unseen = set(range(int(undirected.shape[0])))
    components: List[List[int]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = torch.nonzero(undirected[node], as_tuple=False).flatten().tolist()
            for neighbour in neighbours:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(component)
    largest = max(components, key=lambda values: (len(values), -min(values)))
    component_indices = torch.tensor(
        sorted(largest), dtype=torch.long, device=processed["features"].device
    )
    return slice_graph_record(processed, component_indices)


def count_exact_graph_records(
    counter: RelationalMotifCounter,
    records: Sequence[Mapping[str, Any]],
    feature_onehot_mapping: Mapping[int, Mapping[int, int]],
    batch_size: int,
    device: torch.device,
    selected_rules_values: Optional[Dict[int, List[int]]] = None,
) -> torch.Tensor:
    """Count variable-size graphs without padding by grouping equal sizes."""
    motif_width = (
        sum(len(indices) for indices in selected_rules_values.values())
        if selected_rules_values is not None
        else sum(len(rows) for rows in counter.values)
    )
    output = torch.zeros(len(records), motif_width, dtype=torch.float64)
    size_buckets: Dict[int, List[int]] = {}
    for index, record in enumerate(records):
        size_buckets.setdefault(int(record["features"].shape[0]), []).append(index)

    for node_count, indices in sorted(size_buckets.items()):
        if node_count == 0:
            continue
        wrapper = TensorGraphBatch(
            [records[index] for index in indices],
            relation_keys=counter.relation_keys,
            feature_onehot_mapping=feature_onehot_mapping,
            device=device,
        )
        with torch.no_grad():
            counted = counter.count_batch(
                wrapper,
                batch_size=min(batch_size, len(indices)),
                selected_rules_values=selected_rules_values,
            )
        output[indices] = counted.detach().cpu().to(torch.float64)
    return output


def make_counter_args(
    flat_config: Mapping[str, Any], motif_cache_dir: Path, device: torch.device
) -> SimpleNamespace:
    values = dict(flat_config)
    values["motif_cache_dir"] = str(motif_cache_dir)
    values["device"] = str(device)
    values.setdefault("graph_type", "homogeneous")
    values.setdefault("rule_prune", False)
    values.setdefault("motif_cp_table_source", "cp")
    values.setdefault("use_syntactic_literal_rules", True)
    values.setdefault("syntactic_literal_rule_mode", "both")
    values.setdefault("motif_prune_max_values_per_rule", None)
    if values.get("motif_prune_max_total_values") is not None:
        raise ValueError(
            "motif_prune_max_total_values is not a training option supported "
            "by this checkout, so its selection cannot be reproduced exactly."
        )
    if not bool(values.get("use_syntactic_literal_rules", True)):
        values["syntactic_literal_rule_mode"] = "original"
    return SimpleNamespace(**values)


def _default_motif_weight(flat_config: Mapping[str, Any]) -> float:
    model_name = str(flat_config.get("model", "GraphVAE")).strip().lower()
    return 1.0 if model_name in {
        "graphvae-mm",
        "kernelaugmentedwithtotalnumberoftriangles",
    } else 0.1


def prepare_training_motif_selection(
    counter: RelationalMotifCounter,
    flat_config: Mapping[str, Any],
    pruning_preprocessor: Any,
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    """Reproduce training-time pruning and active motif-group selection.

    ``_CP_smoothed`` pruning is data dependent: training first counts every
    cached value row on the training split, derives local_mult/CP/prior, and
    only then applies the FactorBase pruning score.  Count-distance evaluation
    must execute the same sequence before counting either observed or generated
    graphs.  It must also exclude zero-weight motif groups exactly as training
    does.
    """
    if not bool(flat_config.get("motif_loss", False)):
        raise ValueError(
            "The motif-selection configuration has motif_loss=false, so no "
            "motif rules were optimized during training. Supply "
            "--motif-selection-config from the corresponding motif=True run."
        )

    full_count = sum(len(rows) for rows in counter.values)
    motif_batch_size = int(flat_config.get("motif_batch_size", 50000))
    if motif_batch_size < 1:
        raise ValueError("motif_batch_size must be positive.")
    if counter.requires_data_driven_smoothed_pruning:
        pruning_summary = counter.prepare_data_driven_smoothed_pruning(
            pruning_preprocessor,
            batch_size=motif_batch_size,
        )
    else:
        current_count = sum(len(rows) for rows in counter.values)
        pruning_summary = {
            "full_combinations": current_count,
            "pruned_combinations": current_count,
        }

    motif_output_mode = canonicalize_motif_output_mode(
        flat_config.get("motif_output_mode", "total_count")
    )
    motif_loss_mode = str(flat_config.get("motif_loss_mode", "abs_log_ratio"))
    non_literal_output_mode = canonicalize_motif_output_mode(
        flat_config.get("non_literal_motif_output_mode") or motif_output_mode
    )
    syntactic_output_mode = canonicalize_motif_output_mode(
        flat_config.get("syntactic_literal_motif_output_mode") or motif_output_mode
    )
    unit_output_value = flat_config.get("unit_relation_motif_output_mode")
    unit_output_mode = (
        canonicalize_motif_output_mode(unit_output_value)
        if unit_output_value is not None
        else None
    )
    non_literal_loss_mode = str(
        flat_config.get("non_literal_motif_loss_mode") or motif_loss_mode
    )
    syntactic_loss_mode = str(
        flat_config.get("syntactic_literal_motif_loss_mode") or motif_loss_mode
    )
    unit_loss_mode = (
        str(flat_config.get("unit_relation_motif_loss_mode") or motif_loss_mode)
        if unit_output_mode is not None
        else None
    )

    alpha_motif = flat_config.get("alpha_motif_loss")
    alpha_motif = (
        _default_motif_weight(flat_config)
        if alpha_motif is None
        else float(alpha_motif)
    )
    alpha_syntactic = flat_config.get("alpha_syntactic_literal_motif_loss")
    alpha_syntactic = (
        alpha_motif if alpha_syntactic is None else float(alpha_syntactic)
    )
    alpha_unit = flat_config.get("alpha_unit_relation_motif_loss")
    alpha_unit = (
        0.0
        if unit_output_mode is None
        else (alpha_motif if alpha_unit is None else float(alpha_unit))
    )
    alpha_unit_edge_count = float(
        flat_config.get("alpha_unit_relation_edge_count_loss", 0.0)
    )

    groups = build_motif_group_objectives(
        syntactic_literal_mask=counter.get_syntactic_literal_motif_mask(),
        non_literal_output_mode=non_literal_output_mode,
        non_literal_loss_mode=non_literal_loss_mode,
        non_literal_weight=alpha_motif,
        syntactic_literal_output_mode=syntactic_output_mode,
        syntactic_literal_loss_mode=syntactic_loss_mode,
        syntactic_literal_weight=alpha_syntactic,
        unit_relation_mask=counter.get_unit_relation_motif_mask(),
        unit_relation_output_mode=unit_output_mode,
        unit_relation_loss_mode=unit_loss_mode,
        unit_relation_weight=alpha_unit,
        unit_relation_edge_count_weight=alpha_unit_edge_count,
    )
    active_groups, active_mask = restrict_to_nonzero_weight_motif_groups(groups)
    if not active_groups:
        raise ValueError(
            "The motif-selection configuration has no nonzero-weight motif group."
        )
    selection = counter.select_rule_values_from_motif_mask(active_mask)
    active_count = sum(len(indices) for indices in selection.values())
    summary = {
        "full_combinations": int(full_count),
        "pruned_combinations": int(pruning_summary["pruned_combinations"]),
        "active_combinations": int(active_count),
        "active_groups": [
            {
                "name": group.name,
                "motif_count": group.num_motifs,
                "output_mode": group.output_mode,
                "loss_mode": group.loss_mode,
                "weight": group.weight,
                "edge_count_weight": group.edge_count_weight,
            }
            for group in active_groups
        ],
        "motif_batch_size": motif_batch_size,
    }
    return selection, summary


def count_training_graphs(
    counter: RelationalMotifCounter,
    train_dataset: Any,
    node_onehot_info: Optional[Dict],
    edge_onehot_info: Optional[Dict],
    node_counts: Sequence[int],
    batch_size: int,
    device: torch.device,
    selected_rules_values: Dict[int, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    merged = merge_datasets(train_dataset)
    wrapper = DataWrapper(
        merged,
        counter.relation_keys,
        node_onehot_info,
        edge_onehot_info=edge_onehot_info,
        edge_feature_info_mapping=counter.feature_info_mapping,
        device=str(device),
    )
    records = [
        graph_record_from_wrapper(wrapper, index, node_count)
        for index, node_count in enumerate(node_counts)
    ]
    mapping = wrapper.feature_onehot_mapping
    soft_counts = count_exact_graph_records(
        counter,
        records,
        mapping,
        batch_size,
        device,
        selected_rules_values=selected_rules_values,
    )
    hard_records = [hard_graph_postprocess(record) for record in records]
    hard_counts = count_exact_graph_records(
        counter,
        hard_records,
        mapping,
        batch_size,
        device,
        selected_rules_values=selected_rules_values,
    )
    return soft_counts, hard_counts, [
        int(record["features"].shape[0]) for record in hard_records
    ]


def count_generated_graphs(
    counter: RelationalMotifCounter,
    decoder: GraphTransformerDecoder_FC,
    node_decoder: NodeFeatureDecoder,
    edge_decoder: Optional[EdgeFeatureDecoder],
    graph_dim: int,
    node_counts: Sequence[int],
    node_onehot_info: Optional[Dict],
    edge_onehot_info: Optional[Dict],
    generation_batch_size: int,
    count_batch_size: int,
    adj_threshold: float,
    device: torch.device,
    selected_rules_values: Dict[int, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    soft_batches: List[torch.Tensor] = []
    hard_batches: List[torch.Tensor] = []
    hard_node_counts: List[int] = []

    for start in range(0, len(node_counts), generation_batch_size):
        end = min(start + generation_batch_size, len(node_counts))
        batch_node_counts = node_counts[start:end]
        batch_size = end - start

        # Generate on CPU first so the latent stream is reproducible across GPU types.
        z = torch.randn(batch_size, graph_dim, dtype=torch.float32).to(device)
        with torch.no_grad():
            adjacency_logits = decoder(z)
            # ReconstructedDataWrapper otherwise guesses whether a tensor is
            # logits from its numeric range. Decoder output is known to be
            # logits, so convert explicitly and avoid batch-dependent behavior.
            adjacency_probabilities = torch.sigmoid(adjacency_logits)
            node_logits = node_decoder(z)
            edge_logits = edge_decoder(z) if edge_decoder is not None else None

            wrappers = []
            for use_soft in (True, False):
                wrapper = ReconstructedDataWrapper(
                    reconstructed_adj=adjacency_probabilities,
                    node_feat_logits=node_logits,
                    edge_feat_logits=edge_logits,
                    relation_keys=counter.relation_keys,
                    node_onehot_info=node_onehot_info,
                    feature_onehot_mapping={},
                    edge_onehot_info=edge_onehot_info,
                    edge_feature_info_mapping=counter.feature_info_mapping,
                    adj_threshold=adj_threshold,
                    use_soft_adj=use_soft,
                    prob_temperature=1.0,
                    device=str(device),
                )
                # ReconstructedDataWrapper accepts this mapping separately, but the
                # motif counter consumes the mapping exposed by get_batch().
                if node_onehot_info:
                    wrapper.feature_onehot_mapping = _build_fom(node_onehot_info)
                wrappers.append(wrapper)

            soft_records = [
                graph_record_from_wrapper(wrappers[0], index, node_count)
                for index, node_count in enumerate(batch_node_counts)
            ]
            hard_records = []
            for index, node_count in enumerate(batch_node_counts):
                hard_record = graph_record_from_wrapper(
                    wrappers[1], index, node_count
                )
                if wrappers[0].all_edge is not None:
                    # Use directional soft probabilities when selecting one
                    # categorical label for each undirected edge.
                    soft_record = graph_record_from_wrapper(
                        wrappers[0], index, node_count
                    )
                    hard_record["edge"] = soft_record["edge"]
                hard_records.append(hard_graph_postprocess(hard_record))
            hard_node_counts.extend(
                int(record["features"].shape[0]) for record in hard_records
            )
            mapping = wrappers[0].feature_onehot_mapping
            soft_counts = count_exact_graph_records(
                counter=counter,
                records=soft_records,
                feature_onehot_mapping=mapping,
                batch_size=min(count_batch_size, batch_size),
                device=device,
                selected_rules_values=selected_rules_values,
            )
            hard_counts = count_exact_graph_records(
                counter=counter,
                records=hard_records,
                feature_onehot_mapping=mapping,
                batch_size=min(count_batch_size, batch_size),
                device=device,
                selected_rules_values=selected_rules_values,
            )

        soft_batches.append(soft_counts.detach().cpu().to(torch.float64))
        hard_batches.append(hard_counts.detach().cpu().to(torch.float64))
        print(f"[Generation] {end}/{len(node_counts)} train-matched graphs counted")

        del z, adjacency_logits, adjacency_probabilities, node_logits, edge_logits, wrappers
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return (
        torch.cat(soft_batches, dim=0),
        torch.cat(hard_batches, dim=0),
        hard_node_counts,
    )


def metric_summary(observed: torch.Tensor, generated: torch.Tensor) -> Dict[str, Any]:
    if observed.shape != generated.shape:
        raise ValueError(
            f"Count shape mismatch: observed={tuple(observed.shape)}, "
            f"generated={tuple(generated.shape)}"
        )
    observed_total = observed.sum(dim=0)
    generated_total = generated.sum(dim=0)
    difference = generated_total - observed_total

    observed_mean = observed.mean(dim=0)
    generated_mean = generated.mean(dim=0)
    mean_difference = generated_mean - observed_mean

    aggregate_rmse = torch.sqrt(torch.mean(difference.square()))
    aggregate_mae = torch.mean(difference.abs())
    mean_vector_rmse = torch.sqrt(torch.mean(mean_difference.square()))
    mean_vector_mae = torch.mean(mean_difference.abs())
    observed_mean_rms = torch.sqrt(torch.mean(observed_mean.square()))
    relative_rmse = (
        mean_vector_rmse / observed_mean_rms
        if observed_mean_rms > 0
        else torch.tensor(float("nan"))
    )
    paired_graph_rmse = torch.sqrt(torch.mean((generated - observed).square(), dim=1))

    # Robust companion distances. A one-count floor avoids exploding scores
    # for reference motifs with zero empirical variance. log1p compression
    # and an upper-trimmed Wasserstein mean prevent one frequent motif from
    # dominating the comparison while retaining per-graph distribution shape.
    observed_std = observed.std(dim=0, unbiased=False)
    reference_scale = observed_std.clamp_min(1.0)
    standardized_difference = mean_difference / reference_scale
    standardized_rmse = torch.sqrt(torch.mean(standardized_difference.square()))

    if torch.min(observed) < -1e-8 or torch.min(generated) < -1e-8:
        raise ValueError("Motif counts must be nonnegative for log1p distances.")
    observed_nonnegative = observed.clamp_min(0.0)
    generated_nonnegative = generated.clamp_min(0.0)
    log_mean_difference = torch.log1p(generated_mean.clamp_min(0.0)) - torch.log1p(
        observed_mean.clamp_min(0.0)
    )
    log1p_mean_rmse = torch.sqrt(torch.mean(log_mean_difference.square()))
    observed_sorted = torch.sort(torch.log1p(observed_nonnegative), dim=0).values
    generated_sorted = torch.sort(torch.log1p(generated_nonnegative), dim=0).values
    per_rule_log1p_wasserstein = torch.mean(
        torch.abs(generated_sorted - observed_sorted), dim=0
    )
    wasserstein_sorted = torch.sort(per_rule_log1p_wasserstein).values
    upper_trim_count = int(math.floor(0.1 * wasserstein_sorted.numel()))
    upper_trimmed = (
        wasserstein_sorted[:-upper_trim_count]
        if upper_trim_count > 0
        else wasserstein_sorted
    )
    robust_count_distance = upper_trimmed.mean()

    squared_error = mean_difference.square()
    squared_error_total = squared_error.sum()
    descending_squared_error = torch.sort(squared_error, descending=True).values
    top1_error_share = (
        descending_squared_error[:1].sum() / squared_error_total
        if squared_error_total > 0
        else torch.tensor(0.0, dtype=torch.float64)
    )
    top5_error_share = (
        descending_squared_error[:5].sum() / squared_error_total
        if squared_error_total > 0
        else torch.tensor(0.0, dtype=torch.float64)
    )

    top_count = min(10, int(difference.numel()))
    top_indices = torch.topk(difference.abs(), k=top_count).indices.tolist()

    return {
        "count_distance": float(mean_vector_rmse.item()),
        "aggregate_count_distance": float(aggregate_rmse.item()),
        "mean_vector_count_distance": float(mean_vector_rmse.item()),
        "relative_count_distance": float(relative_rmse.item()),
        "robust_count_distance": float(robust_count_distance.item()),
        "standardized_mean_vector_count_distance": float(
            standardized_rmse.item()
        ),
        "log1p_mean_vector_count_distance": float(log1p_mean_rmse.item()),
        "log1p_wasserstein_mean": float(
            per_rule_log1p_wasserstein.mean().item()
        ),
        "log1p_wasserstein_median": float(
            per_rule_log1p_wasserstein.median().item()
        ),
        "log1p_wasserstein_upper_trimmed_mean": float(
            robust_count_distance.item()
        ),
        "per_rule_log1p_wasserstein": per_rule_log1p_wasserstein.tolist(),
        "reference_count_standard_deviations": observed_std.tolist(),
        "reference_standardization_scales": reference_scale.tolist(),
        "top1_raw_squared_error_share": float(top1_error_share.item()),
        "top5_raw_squared_error_share": float(top5_error_share.item()),
        "aggregate_mean_absolute_count_difference": float(aggregate_mae.item()),
        "mean_vector_absolute_count_difference": float(mean_vector_mae.item()),
        "paired_graph_count_distance_mean": float(paired_graph_rmse.mean().item()),
        "paired_graph_count_distance_median": float(paired_graph_rmse.median().item()),
        "paired_graph_count_distance_std": float(
            paired_graph_rmse.std(unbiased=False).item()
        ),
        "observed_aggregate_counts": observed_total.tolist(),
        "generated_aggregate_counts": generated_total.tolist(),
        "generated_minus_observed": difference.tolist(),
        "observed_mean_counts": observed_mean.tolist(),
        "generated_mean_counts": generated_mean.tolist(),
        "generated_minus_observed_mean": mean_difference.tolist(),
        "top_absolute_error_indices": top_indices,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Decimal):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def node_count_summary(values: Sequence[int]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.int64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty node-count sequence.")
    return {
        "minimum": int(array.min()),
        "maximum": int(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "empty_graphs": int(np.sum(array == 0)),
    }


def motif_entry_metadata(
    counter: RelationalMotifCounter,
    selected_rules_values: Optional[Dict[int, List[int]]] = None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    iteration_plan = (
        [
            (rule_index, value_index, counter.values[rule_index][value_index])
            for rule_index, value_indices in selected_rules_values.items()
            for value_index in value_indices
        ]
        if selected_rules_values is not None
        else [
            (rule_index, value_index, row)
            for rule_index, rows in enumerate(counter.values)
            for value_index, row in enumerate(rows)
        ]
    )
    for rule_index, value_index, row in iteration_plan:
        entries.append(
            {
                "index": len(entries),
                "rule_index": rule_index,
                "value_index": value_index,
                "rule": json_safe(counter.rules[rule_index]),
                "value_row": json_safe(row),
                "rule_source": json_safe(counter.rule_sources[rule_index]),
            }
        )
    return entries


def main() -> None:
    cli = parse_args()
    config_path = Path(cli.config).expanduser().resolve()
    motif_selection_config_path = (
        Path(cli.motif_selection_config).expanduser().resolve()
        if cli.motif_selection_config is not None
        else config_path
    )
    checkpoint_path = Path(cli.checkpoint).expanduser().resolve()
    dataset_cache_path = Path(cli.dataset_cache).expanduser().resolve()
    motif_cache_dir = Path(cli.motif_cache_dir).expanduser().resolve()
    output_path = Path(cli.output).expanduser().resolve()

    if cli.motif_selection_manifest is not None:
        motif_selection_manifest_path = Path(
            cli.motif_selection_manifest
        ).expanduser().resolve()
    else:
        automatic_manifest = checkpoint_path.parent / MOTIF_SELECTION_MANIFEST_FILENAME
        motif_selection_manifest_path = (
            automatic_manifest if automatic_manifest.is_file() else None
        )

    nested_config, flat_config = load_yaml(config_path)
    if motif_selection_config_path == config_path:
        selection_nested_config = nested_config
        selection_flat_config = flat_config
    else:
        selection_nested_config, selection_flat_config = load_yaml(
            motif_selection_config_path
        )
        for key in (
            "dataset",
            "split_mode",
            "split_seed",
            "dataset_loader_seed",
            "train_fraction",
            "val_fraction",
            "bfsOrdering",
            "bfs_strategy",
        ):
            model_value = flat_config.get(key)
            selection_value = selection_flat_config.get(key)
            if (
                model_value is not None
                and selection_value is not None
                and model_value != selection_value
            ):
                raise ValueError(
                    "Model and motif-selection configurations disagree on "
                    f"{key}: {model_value!r} vs {selection_value!r}."
                )
    seed = int(flat_config.get("seed", 0) if cli.seed is None else cli.seed)
    configure_rng(seed)

    requested_device = cli.device or str(flat_config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested_device}")
    device = torch.device(requested_device)

    cache = load_pickle(dataset_cache_path)
    if "list_graphs" not in cache:
        raise KeyError(f"Dataset cache has no list_graphs entry: {dataset_cache_path}")
    train_dataset = cache["list_graphs"]
    node_onehot_info = cache.get("node_onehot_info")
    edge_onehot_info = cache.get("edge_onehot_info")
    node_counts = [int(adjacency.shape[0]) for adjacency in train_dataset.list_adjs]
    if not node_counts:
        raise RuntimeError("Training split contains no graphs.")

    counter_args = make_counter_args(selection_flat_config, motif_cache_dir, device)
    database_name = str(selection_flat_config["database_name"])
    counter = RelationalMotifCounter(database_name=database_name, args=counter_args)

    if motif_selection_manifest_path is not None:
        manifest = load_motif_selection_manifest(motif_selection_manifest_path)
        selected_rules_values = apply_motif_selection_manifest(
            counter,
            manifest,
            database_name=database_name,
            motif_cp_table_source=str(
                selection_flat_config.get("motif_cp_table_source", "cp")
            ),
            rule_prune=bool(selection_flat_config.get("rule_prune", False)),
        )
        motif_selection_summary = {
            "full_combinations": int(manifest["full_combinations"]),
            "pruned_combinations": int(manifest["pruned_combinations"]),
            "active_combinations": int(manifest["active_combinations"]),
            "active_groups": manifest.get("active_groups", []),
            "selection_method": "saved_training_manifest",
        }
    else:
        pruning_wrapper = DataWrapper(
            merge_datasets(train_dataset),
            counter.relation_keys,
            node_onehot_info,
            edge_onehot_info=edge_onehot_info,
            edge_feature_info_mapping=counter.feature_info_mapping,
            device=str(device),
        )
        selected_rules_values, motif_selection_summary = (
            prepare_training_motif_selection(
                counter=counter,
                flat_config=selection_flat_config,
                pruning_preprocessor=pruning_wrapper,
            )
        )
        motif_selection_summary["selection_method"] = "reconstructed_from_training_data"
        del pruning_wrapper
    if device.type == "cuda":
        torch.cuda.empty_cache()

    state = load_state_dict(checkpoint_path)
    decoder, node_decoder, edge_decoder, graph_dim = build_decoders(
        state, flat_config, train_dataset, device
    )

    print(
        f"[CountDistance] dataset={flat_config.get('dataset')} "
        f"database={database_name} train_graphs={len(node_counts)} "
        f"node_range={min(node_counts)}..{max(node_counts)} seed={seed}"
    )
    print(
        "[CountDistance] exact training motif selection: "
        f"full={motif_selection_summary['full_combinations']} "
        f"pruned={motif_selection_summary['pruned_combinations']} "
        f"active={motif_selection_summary['active_combinations']}"
    )
    observed_soft_counts, observed_hard_counts, observed_hard_node_counts = count_training_graphs(
        counter=counter,
        train_dataset=train_dataset,
        node_onehot_info=node_onehot_info,
        edge_onehot_info=edge_onehot_info,
        node_counts=node_counts,
        batch_size=cli.count_batch_size,
        device=device,
        selected_rules_values=selected_rules_values,
    )
    # Decoder construction consumes random numbers during parameter
    # initialisation before checkpoint loading. Reset here so ``--seed``
    # controls only the latent samples and is architecture-independent.
    configure_rng(seed)
    (
        generated_soft_counts,
        generated_hard_counts,
        generated_hard_node_counts,
    ) = count_generated_graphs(
        counter=counter,
        decoder=decoder,
        node_decoder=node_decoder,
        edge_decoder=edge_decoder,
        graph_dim=graph_dim,
        node_counts=node_counts,
        node_onehot_info=node_onehot_info,
        edge_onehot_info=edge_onehot_info,
        generation_batch_size=cli.generation_batch_size,
        count_batch_size=cli.count_batch_size,
        adj_threshold=cli.adj_threshold,
        device=device,
        selected_rules_values=selected_rules_values,
    )

    motif_entries = motif_entry_metadata(counter, selected_rules_values)
    if observed_soft_counts.shape[1] != len(motif_entries):
        raise RuntimeError(
            "Motif metadata length does not match counter output width: "
            f"{len(motif_entries)} vs {observed_soft_counts.shape[1]}"
        )

    payload = {
        "schema_version": "graphvae-motif-count-distance-v5",
        "metric_definition": (
            "Primary: sqrt(mean((mean_generated_counts - "
            "mean_train_counts)^2)); traceability: sqrt(mean((sum_generated_counts "
            "- sum_train_counts)^2)); both over the exact data-pruned, nonzero-"
            "weight relational motif/value combinations used by training"
        ),
        "recommended_metric_definition": (
            "robust_count_distance is the mean per-rule 1D Wasserstein distance "
            "between log1p per-graph motif counts after removing the largest "
            "10% of rule distances (no trimming for fewer than 10 rules)"
        ),
        "dataset": cli.dataset_label or flat_config.get("dataset"),
        "database_name": database_name,
        "setting": cli.setting,
        "seed": seed,
        "graph_count": len(node_counts),
        "generated_graph_count": len(node_counts),
        "node_count_matching": (
            "one generated graph per train graph, cropped to the same pre-cleanup "
            "node count; hard cleanup may reduce it"
        ),
        "soft_protocol": (
            "exact-size train adjacency versus sigmoid decoder probabilities; "
            "active-node diagonals retained to follow the VGAE count-distance input"
        ),
        "hard_protocol": (
            "threshold at adjacency_threshold; convert to undirected graph and "
            "symmetrize categorical edge labels; remove self-loops and isolates; "
            "keep largest connected component on both train and generated graphs"
        ),
        "node_count_summary": node_count_summary(node_counts),
        "hard_postprocess_node_count_summary": {
            "train": node_count_summary(observed_hard_node_counts),
            "generated": node_count_summary(generated_hard_node_counts),
        },
        "motif_entry_count": int(observed_soft_counts.shape[1]),
        "motif_selection": motif_selection_summary,
        "adjacency_threshold": float(cli.adj_threshold),
        "soft": metric_summary(observed_soft_counts, generated_soft_counts),
        "hard": metric_summary(observed_hard_counts, generated_hard_counts),
        "motif_entries": motif_entries,
        "artifacts": {
            "config": str(config_path),
            "motif_selection_config": str(motif_selection_config_path),
            "checkpoint": str(checkpoint_path),
            "dataset_cache": str(dataset_cache_path),
            "motif_cache": str(motif_cache_dir / f"{database_name}.pkl"),
            "motif_selection_manifest": (
                str(motif_selection_manifest_path)
                if motif_selection_manifest_path is not None
                else None
            ),
        },
        "config": nested_config,
        "motif_selection_config": selection_nested_config,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[CountDistance] wrote {output_path}")
    print(
        "[CountDistance] mean-vector "
        f"soft={payload['soft']['mean_vector_count_distance']:.8g} "
        f"hard={payload['hard']['mean_vector_count_distance']:.8g}; "
        "aggregate "
        f"soft={payload['soft']['aggregate_count_distance']:.8g} "
        f"hard={payload['hard']['aggregate_count_distance']:.8g}"
    )
    print(
        "[CountDistance] robust-log1p-wasserstein "
        f"soft={payload['soft']['robust_count_distance']:.8g} "
        f"hard={payload['hard']['robust_count_distance']:.8g}"
    )


if __name__ == "__main__":
    main()
