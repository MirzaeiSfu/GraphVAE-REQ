#!/usr/bin/env python3
"""Evaluate relational motif-count distance on train-sized graph samples.

This adapts the ``count distance`` calculation from the VGAE repository:

    sqrt(mean((observed_counts - generated_counts) ** 2))

The original implementation compares two motif-count vectors for one graph.
GraphVAE datasets contain many graphs, so this script generates exactly one
graph for every training graph and compares the mean motif-count vectors.  It
also reports the RMSE between aggregate count vectors, which is the literal
VGAE formula after summing the graph dataset.  The aggregate value scales with
the number of graphs; the mean-vector value does not.

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
) -> torch.Tensor:
    """Count variable-size graphs without padding by grouping equal sizes."""
    motif_width = sum(len(rows) for rows in counter.values)
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
            counted = counter.count_batch(wrapper, batch_size=min(batch_size, len(indices)))
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
    values.setdefault("use_syntactic_literal_rules", False)
    values.setdefault("syntactic_literal_rule_mode", "original")
    values.setdefault("motif_prune_max_values_per_rule", None)
    values.setdefault("motif_prune_max_total_values", None)
    return SimpleNamespace(**values)


def ensure_global_motif_value_cap(
    counter: RelationalMotifCounter,
    counter_args: SimpleNamespace,
) -> None:
    """Apply the training checkout's global formula-score cap if needed.

    Some older checkouts load ``motif_prune_max_total_values`` from YAML but do
    not apply it inside RelationalMotifCounter.  Evaluation must use exactly the
    motif/value combinations that trained the checkpoint, so reproduce that
    runtime cap here.  Newer counters already apply it; in that case the current
    width is at most the cap and this function is a no-op.
    """
    max_total = getattr(counter_args, "motif_prune_max_total_values", None)
    if max_total is None:
        return
    max_total = int(max_total)
    if not getattr(counter_args, "rule_prune", False):
        raise ValueError("motif_prune_max_total_values requires rule_prune=True")
    if max_total < 1:
        raise ValueError("motif_prune_max_total_values must be positive")
    if sum(len(rows) for rows in counter.values) <= max_total:
        return

    candidates: List[Tuple[float, int, int]] = []
    for rule_index, rows in enumerate(counter.values):
        multiple = counter.multiples[rule_index]
        for row_index, row in enumerate(rows):
            size = len(row)
            try:
                if multiple:
                    local_mult = float(row[size - 4])
                    cp_value = float(row[size - 3])
                    prior_value = float(row[size - 1])
                else:
                    local_mult = float(row[size - 3])
                    cp_value = float(row[size - 5])
                    prior_value = float(row[size - 1])
                if local_mult <= 0 or cp_value <= 0 or prior_value <= 0:
                    continue
                score = (
                    2.0 * local_mult * (math.log(cp_value) - math.log(prior_value))
                    - math.log(local_mult)
                )
            except (IndexError, TypeError, ValueError, ZeroDivisionError):
                continue
            if score > 0:
                candidates.append((score, rule_index, row_index))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = {
        (rule_index, row_index)
        for _score, rule_index, row_index in candidates[:max_total]
    }
    counter.values = [
        [
            row
            for row_index, row in enumerate(rows)
            if (rule_index, row_index) in selected
        ]
        for rule_index, rows in enumerate(counter.values)
    ]
    if hasattr(counter, "_build_syntactic_literal_masks"):
        counter._build_syntactic_literal_masks()
    kept = sum(len(rows) for rows in counter.values)
    print(
        f"  [CountDistance] motif_prune_max_total_values={max_total}: "
        f"using top {kept} formula-scored combinations globally"
    )


def count_training_graphs(
    counter: RelationalMotifCounter,
    train_dataset: Any,
    node_onehot_info: Optional[Dict],
    edge_onehot_info: Optional[Dict],
    node_counts: Sequence[int],
    batch_size: int,
    device: torch.device,
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
        counter, records, mapping, batch_size, device
    )
    hard_records = [hard_graph_postprocess(record) for record in records]
    hard_counts = count_exact_graph_records(
        counter,
        hard_records,
        mapping,
        batch_size,
        device,
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
                counter,
                soft_records,
                mapping,
                min(count_batch_size, batch_size),
                device,
            )
            hard_counts = count_exact_graph_records(
                counter,
                hard_records,
                mapping,
                min(count_batch_size, batch_size),
                device,
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

    top_count = min(10, int(difference.numel()))
    top_indices = torch.topk(difference.abs(), k=top_count).indices.tolist()

    return {
        "count_distance": float(mean_vector_rmse.item()),
        "aggregate_count_distance": float(aggregate_rmse.item()),
        "mean_vector_count_distance": float(mean_vector_rmse.item()),
        "relative_count_distance": float(relative_rmse.item()),
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


def motif_entry_metadata(counter: RelationalMotifCounter) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for rule_index, rows in enumerate(counter.values):
        for value_index, row in enumerate(rows):
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
    checkpoint_path = Path(cli.checkpoint).expanduser().resolve()
    dataset_cache_path = Path(cli.dataset_cache).expanduser().resolve()
    motif_cache_dir = Path(cli.motif_cache_dir).expanduser().resolve()
    output_path = Path(cli.output).expanduser().resolve()

    nested_config, flat_config = load_yaml(config_path)
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

    counter_args = make_counter_args(flat_config, motif_cache_dir, device)
    database_name = str(flat_config["database_name"])
    counter = RelationalMotifCounter(database_name=database_name, args=counter_args)
    ensure_global_motif_value_cap(counter, counter_args)

    state = load_state_dict(checkpoint_path)
    decoder, node_decoder, edge_decoder, graph_dim = build_decoders(
        state, flat_config, train_dataset, device
    )

    print(
        f"[CountDistance] dataset={flat_config.get('dataset')} "
        f"database={database_name} train_graphs={len(node_counts)} "
        f"node_range={min(node_counts)}..{max(node_counts)} seed={seed}"
    )
    observed_soft_counts, observed_hard_counts, observed_hard_node_counts = count_training_graphs(
        counter=counter,
        train_dataset=train_dataset,
        node_onehot_info=node_onehot_info,
        edge_onehot_info=edge_onehot_info,
        node_counts=node_counts,
        batch_size=cli.count_batch_size,
        device=device,
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
    )

    if observed_soft_counts.shape[1] != len(motif_entry_metadata(counter)):
        raise RuntimeError(
            "Motif metadata length does not match counter output width: "
            f"{len(motif_entry_metadata(counter))} vs {observed_soft_counts.shape[1]}"
        )

    payload = {
        "schema_version": "graphvae-motif-count-distance-v3",
        "metric_definition": (
            "Primary: sqrt(mean((mean_generated_counts - "
            "mean_train_counts)^2)); traceability: sqrt(mean((sum_generated_counts "
            "- sum_train_counts)^2)); both over retained relational motif/value "
            "combinations"
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
        "adjacency_threshold": float(cli.adj_threshold),
        "soft": metric_summary(observed_soft_counts, generated_soft_counts),
        "hard": metric_summary(observed_hard_counts, generated_hard_counts),
        "motif_entries": motif_entry_metadata(counter),
        "artifacts": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "dataset_cache": str(dataset_cache_path),
            "motif_cache": str(motif_cache_dir / f"{database_name}.pkl"),
        },
        "config": nested_config,
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


if __name__ == "__main__":
    main()
