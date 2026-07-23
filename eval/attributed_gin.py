"""Feature-aware graph preparation and Random-GIN evaluation utilities.

Unlike the legacy graph-realism evaluator, this module never constructs
degree, clustering, or other topology-derived node features.  It consumes the
node and edge attributes stored on DGL graphs and provides explicit feature
ablations with a fixed GIN input shape.
"""

# Caller-side PyG -> DGL reference only (intentionally not a runtime adapter):
#
#   import dgl
#   def pyg_to_dgl(data):
#       graph = dgl.graph(
#           (data.edge_index[0], data.edge_index[1]),
#           num_nodes=data.num_nodes,
#       )
#       graph.ndata["attr"] = data.x.float()
#       if data.edge_attr is not None:
#           graph.edata["attr"] = data.edge_attr.float()
#       return graph
#
# Convert categorical IDs to one-hot float matrices before assigning them.
# This evaluator deliberately rejects PyG objects so every model enters through
# the same small, auditable DGL contract.

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GGMEVAL_ROOT = REPO_ROOT / "third_party" / "ggmeval"

FEATURE_MODES = (
    "topology_control",
    "decoded_node",
    "decoded_edge",
    "decoded_node_edge",
)


@dataclass(frozen=True)
class AttributedGraph:
    """An undirected graph with attributes aligned to nodes and edges.

    ``edges`` contains each undirected edge once, using local node indices.
    ``source_node_ids`` records the indices in the original dense decoder or
    reference tensors, which makes feature/topology alignment auditable.
    """

    edges: np.ndarray
    node_attributes: np.ndarray
    edge_attributes: np.ndarray
    source_node_ids: np.ndarray

    def __post_init__(self):
        edges = np.asarray(self.edges)
        node_attributes = np.asarray(self.node_attributes)
        edge_attributes = np.asarray(self.edge_attributes)
        source_node_ids = np.asarray(self.source_node_ids)

        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges must have shape (E, 2), got {edges.shape}.")
        if node_attributes.ndim != 2:
            raise ValueError(
                "node_attributes must have shape (N, D), "
                f"got {node_attributes.shape}."
            )
        if edge_attributes.ndim != 2:
            raise ValueError(
                "edge_attributes must have shape (E, C), "
                f"got {edge_attributes.shape}."
            )
        if len(edges) != len(edge_attributes):
            raise ValueError(
                "Each edge must have one attribute row; "
                f"got {len(edges)} edges and {len(edge_attributes)} rows."
            )
        if source_node_ids.shape != (len(node_attributes),):
            raise ValueError(
                "source_node_ids must have one entry per node; "
                f"got {source_node_ids.shape} for {len(node_attributes)} nodes."
            )
        if edges.size:
            if edges.min() < 0 or edges.max() >= len(node_attributes):
                raise ValueError("edges contain a node index outside the local node range.")
            if np.any(edges[:, 0] >= edges[:, 1]):
                raise ValueError("edges must use canonical undirected pairs with u < v.")

        object.__setattr__(self, "edges", edges.astype(np.int64, copy=False))
        object.__setattr__(
            self,
            "node_attributes",
            node_attributes.astype(np.float32, copy=False),
        )
        object.__setattr__(
            self,
            "edge_attributes",
            edge_attributes.astype(np.float32, copy=False),
        )
        object.__setattr__(
            self,
            "source_node_ids",
            source_node_ids.astype(np.int64, copy=False),
        )

    @property
    def num_nodes(self) -> int:
        return int(self.node_attributes.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def node_feature_dim(self) -> int:
        return int(self.node_attributes.shape[1])

    @property
    def edge_feature_dim(self) -> int:
        return int(self.edge_attributes.shape[1])


def as_dense_array(value) -> np.ndarray:
    """Convert scipy-like sparse values and ndarrays to a dense ndarray."""

    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


def categorical_groups(
    feature_info: Mapping | None,
    feature_dim: int,
) -> tuple[tuple[int, ...], ...]:
    """Return one-hot channel groups ordered by first channel index."""

    if feature_dim < 0:
        raise ValueError(f"feature_dim must be non-negative, got {feature_dim}.")
    if feature_dim == 0:
        return ()
    if not feature_info:
        return (tuple(range(feature_dim)),)

    by_name: dict[str, list[int]] = {}
    for raw_index, metadata in sorted(
        feature_info.items(), key=lambda item: int(item[0])
    ):
        index = int(raw_index)
        if index < 0 or index >= feature_dim:
            raise ValueError(
                f"Feature metadata channel {index} is outside dimension {feature_dim}."
            )
        name = str(metadata["feature_name"])
        by_name.setdefault(name, []).append(index)

    covered = sorted(index for indices in by_name.values() for index in indices)
    if covered != list(range(feature_dim)):
        raise ValueError(
            "Feature metadata must cover every channel exactly once; "
            f"covered={covered}, expected={list(range(feature_dim))}."
        )
    return tuple(tuple(indices) for indices in by_name.values())


def grouped_argmax_onehot(
    values: np.ndarray,
    groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Harden categorical logits/probabilities independently per feature."""

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"values must have shape (items, channels), got {values.shape}.")
    output = np.zeros(values.shape, dtype=np.float32)
    for group in groups:
        group_indices = np.asarray(tuple(group), dtype=np.int64)
        if group_indices.size == 0:
            continue
        winners = np.argmax(values[:, group_indices], axis=1)
        output[np.arange(len(values)), group_indices[winners]] = 1.0
    return output


def _largest_component_topology(
    adjacency,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    dense = as_dense_array(adjacency)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError(f"adjacency must be square, got {dense.shape}.")

    binary = dense >= threshold
    binary = np.logical_or(binary, binary.T)
    np.fill_diagonal(binary, False)
    graph = nx.from_numpy_array(binary)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_edges() == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

    component = max(nx.connected_components(graph), key=len)
    source_node_ids = np.asarray(sorted(component), dtype=np.int64)
    source_to_local = {
        int(source_id): local_id
        for local_id, source_id in enumerate(source_node_ids)
    }
    component_graph = graph.subgraph(source_node_ids)
    edges = sorted(
        (
            min(source_to_local[int(source_u)], source_to_local[int(source_v)]),
            max(source_to_local[int(source_u)], source_to_local[int(source_v)]),
        )
        for source_u, source_v in component_graph.edges()
        if source_u != source_v
    )
    return source_node_ids, np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def _edge_rows_from_dense(
    edge_values,
    source_node_ids: np.ndarray,
    local_edges: np.ndarray,
) -> np.ndarray:
    dense = as_dense_array(edge_values)
    if dense.ndim != 3:
        raise ValueError(
            "Dense edge attributes must have shape (C, N, N), "
            f"got {dense.shape}."
        )
    if dense.shape[1] != dense.shape[2]:
        raise ValueError(f"Dense edge attribute matrices must be square, got {dense.shape}.")
    if source_node_ids.size and source_node_ids.max() >= dense.shape[1]:
        raise ValueError(
            "Dense edge attributes do not cover all retained topology node indices."
        )
    if not len(local_edges):
        return np.empty((0, dense.shape[0]), dtype=np.float32)

    source_u = source_node_ids[local_edges[:, 0]]
    source_v = source_node_ids[local_edges[:, 1]]
    forward = dense[:, source_u, source_v].T
    reverse = dense[:, source_v, source_u].T
    return ((forward + reverse) / 2.0).astype(np.float32, copy=False)


def graph_from_dense_attributes(
    adjacency,
    node_values,
    edge_values=None,
    *,
    node_feature_info: Mapping | None = None,
    edge_feature_info: Mapping | None = None,
    values_are_logits: bool,
    adjacency_threshold: float = 0.5,
) -> AttributedGraph | None:
    """Align dense topology, node attributes, and edge attributes.

    Decoder outputs should set ``values_are_logits=True``.  Categorical
    channels are then hardened by argmax *within each original feature group*.
    Reference one-hot tensors may set it to ``False`` and are preserved.
    """

    source_node_ids, local_edges = _largest_component_topology(
        adjacency, adjacency_threshold
    )
    if not len(local_edges):
        return None

    node_values = as_dense_array(node_values)
    if node_values.ndim != 2:
        raise ValueError(
            f"node_values must have shape (N, D), got {node_values.shape}."
        )
    if source_node_ids.max() >= node_values.shape[0]:
        raise ValueError("Node attributes do not cover all retained topology indices.")
    node_rows = node_values[source_node_ids].astype(np.float32, copy=False)
    if values_are_logits or node_feature_info:
        node_rows = grouped_argmax_onehot(
            node_rows,
            categorical_groups(node_feature_info, node_rows.shape[1]),
        )

    if edge_values is None:
        edge_rows = np.empty((len(local_edges), 0), dtype=np.float32)
    else:
        edge_rows = _edge_rows_from_dense(edge_values, source_node_ids, local_edges)
        if values_are_logits or edge_feature_info:
            edge_rows = grouped_argmax_onehot(
                edge_rows,
                categorical_groups(edge_feature_info, edge_rows.shape[1]),
            )

    return AttributedGraph(
        edges=local_edges,
        node_attributes=node_rows,
        edge_attributes=edge_rows,
        source_node_ids=source_node_ids,
    )


def validate_collection_dimensions(
    generated_graphs: Sequence[AttributedGraph],
    reference_graphs: Sequence[AttributedGraph],
) -> tuple[int, int]:
    graphs = list(generated_graphs) + list(reference_graphs)
    if not generated_graphs or not reference_graphs:
        raise ValueError("Generated and reference graph collections must both be non-empty.")
    node_dims = {graph.node_feature_dim for graph in graphs}
    edge_dims = {graph.edge_feature_dim for graph in graphs}
    if len(node_dims) != 1 or len(edge_dims) != 1:
        raise ValueError(
            "All generated and reference graphs must share feature dimensions; "
            f"node_dims={sorted(node_dims)}, edge_dims={sorted(edge_dims)}."
        )
    node_dim = next(iter(node_dims))
    edge_dim = next(iter(edge_dims))
    if node_dim < 1:
        raise ValueError("Attributed Random-GIN needs at least one node feature channel.")
    return node_dim, edge_dim


def _is_pyg_object(value) -> bool:
    return type(value).__module__.startswith("torch_geometric")


def _materialize_dgl_collection(value, *, name: str) -> list:
    if _is_pyg_object(value):
        raise TypeError(
            f"{name} received a PyTorch Geometric object. Convert it to DGL in "
            "the caller and pass a collection of individual DGL graphs."
        )
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a collection of individual DGL graphs.")
    try:
        return list(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a collection of individual DGL graphs."
        ) from exc


def _require_float_feature_matrix(tensor, *, name: str, row_count: int):
    import torch

    if tensor.ndim != 2 or tensor.shape[0] != row_count:
        raise ValueError(
            f"{name} must have shape ({row_count}, D), got {tuple(tensor.shape)}."
        )
    if not torch.is_floating_point(tensor):
        raise TypeError(
            f"{name} must be a floating-point matrix. Convert categorical IDs "
            "to one-hot float features in the caller."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or infinite values.")


def attributed_graph_from_dgl(graph) -> AttributedGraph:
    """Validate and normalize one DGL graph for attributed evaluation.

    The public input contract is a homogeneous, unbatched DGL graph with
    ``ndata["attr"]`` and, when edge features exist, ``edata["attr"]``. Input
    self-loops are ignored. Directed copies of an undirected edge are merged
    only when their attributes agree. Isolates and all but the deterministic
    largest connected component are removed.
    """

    if _is_pyg_object(graph):
        raise TypeError(
            "PyTorch Geometric inputs are not accepted. Convert the graph to "
            "DGL in the caller and store float features in ndata['attr'] and "
            "edata['attr']."
        )

    try:
        import dgl
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("DGL is required for attributed evaluation.") from exc

    if not isinstance(graph, dgl.DGLGraph):
        raise TypeError(
            f"Expected a DGLGraph, got {type(graph).__module__}."
            f"{type(graph).__name__}."
        )
    if not graph.is_homogeneous:
        raise TypeError("Only homogeneous DGL graphs are supported.")
    if int(getattr(graph, "batch_size", 1)) != 1:
        raise ValueError(
            "Batched DGL graphs are not accepted. Call dgl.unbatch and pass the "
            "individual graphs."
        )

    node_count = int(graph.num_nodes())
    edge_count = int(graph.num_edges())
    if "attr" not in graph.ndata:
        raise ValueError(
            "DGL input is missing ndata['attr']; topology-derived replacement "
            "features are intentionally forbidden."
        )
    node_tensor = graph.ndata["attr"]
    _require_float_feature_matrix(
        node_tensor, name="ndata['attr']", row_count=node_count
    )

    edge_tensor = graph.edata.get("attr")
    if edge_tensor is None:
        edge_attributes = np.empty((edge_count, 0), dtype=np.float32)
    else:
        _require_float_feature_matrix(
            edge_tensor, name="edata['attr']", row_count=edge_count
        )
        edge_attributes = edge_tensor.detach().cpu().numpy().astype(
            np.float32, copy=False
        )

    node_attributes = node_tensor.detach().cpu().numpy().astype(
        np.float32, copy=False
    )
    try:
        source_tensor, target_tensor = graph.edges(order="eid")
    except TypeError:  # pragma: no cover - compatibility with older DGL
        source_tensor, target_tensor = graph.edges()
    sources = source_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
    targets = target_tensor.detach().cpu().numpy().astype(np.int64, copy=False)

    edge_rows_by_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    for edge_index, (source, target) in enumerate(zip(sources, targets)):
        source = int(source)
        target = int(target)
        if source == target:
            continue
        pair = (min(source, target), max(source, target))
        edge_rows_by_pair.setdefault(pair, []).append(edge_attributes[edge_index])

    if not edge_rows_by_pair:
        raise ValueError(
            "DGL input has no non-self-loop edges after topology normalization."
        )

    undirected = nx.Graph()
    undirected.add_nodes_from(range(node_count))
    undirected.add_edges_from(edge_rows_by_pair)
    undirected.remove_nodes_from(list(nx.isolates(undirected)))
    components = list(nx.connected_components(undirected))
    component = max(
        components,
        key=lambda node_ids: (len(node_ids), -min(node_ids)),
    )
    source_node_ids = np.asarray(sorted(component), dtype=np.int64)
    source_to_local = {
        int(source_id): local_id
        for local_id, source_id in enumerate(source_node_ids)
    }

    retained_pairs = sorted(
        pair
        for pair in edge_rows_by_pair
        if pair[0] in component and pair[1] in component
    )
    normalized_edges = np.asarray(
        [
            (source_to_local[source], source_to_local[target])
            for source, target in retained_pairs
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    normalized_edge_attributes = []
    for pair in retained_pairs:
        rows = edge_rows_by_pair[pair]
        first = rows[0]
        if any(
            not np.allclose(first, row, rtol=1e-5, atol=1e-7)
            for row in rows[1:]
        ):
            raise ValueError(
                "Conflicting edata['attr'] values for undirected edge "
                f"{pair}. Reverse directions and duplicate edges must carry "
                "the same final attribute."
            )
        normalized_edge_attributes.append(
            np.mean(np.stack(rows, axis=0), axis=0)
        )
    normalized_edge_attributes = np.asarray(
        normalized_edge_attributes, dtype=np.float32
    ).reshape(len(retained_pairs), edge_attributes.shape[1])

    return AttributedGraph(
        edges=normalized_edges,
        node_attributes=node_attributes[source_node_ids],
        edge_attributes=normalized_edge_attributes,
        source_node_ids=source_node_ids,
    )


def feature_view(
    graph: AttributedGraph,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Select decoded features or fixed controls without changing dimensions."""

    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode {mode!r}; expected one of {FEATURE_MODES}.")

    if mode in {"decoded_node", "decoded_node_edge"}:
        node_attributes = graph.node_attributes
    else:
        node_attributes = np.zeros_like(graph.node_attributes)
        node_attributes[:, 0] = 1.0

    if mode in {"decoded_edge", "decoded_node_edge"}:
        edge_attributes = graph.edge_attributes
    else:
        edge_attributes = np.zeros_like(graph.edge_attributes)

    return node_attributes, edge_attributes


def to_dgl_graph(graph: AttributedGraph, mode: str):
    """Convert to DGL, duplicating undirected edges and adding zero-attr loops."""

    try:
        import dgl
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("DGL and PyTorch are required for Random-GIN evaluation.") from exc

    node_attributes, edge_attributes = feature_view(graph, mode)
    if graph.num_edges:
        source = np.concatenate((graph.edges[:, 0], graph.edges[:, 1]))
        target = np.concatenate((graph.edges[:, 1], graph.edges[:, 0]))
        directed_edge_attributes = np.concatenate(
            (edge_attributes, edge_attributes), axis=0
        )
    else:  # pragma: no cover - topology preprocessing rejects edge-empty graphs
        source = np.empty((0,), dtype=np.int64)
        target = np.empty((0,), dtype=np.int64)
        directed_edge_attributes = np.empty(
            (0, graph.edge_feature_dim), dtype=np.float32
        )

    loop_nodes = np.arange(graph.num_nodes, dtype=np.int64)
    source = np.concatenate((source, loop_nodes))
    target = np.concatenate((target, loop_nodes))
    loop_attributes = np.zeros(
        (graph.num_nodes, graph.edge_feature_dim), dtype=np.float32
    )
    directed_edge_attributes = np.concatenate(
        (directed_edge_attributes, loop_attributes), axis=0
    )

    dgl_graph = dgl.graph(
        (
            torch.as_tensor(source, dtype=torch.int64),
            torch.as_tensor(target, dtype=torch.int64),
        ),
        num_nodes=graph.num_nodes,
    )
    dgl_graph.ndata["attr"] = torch.as_tensor(
        node_attributes, dtype=torch.float32
    )
    if graph.edge_feature_dim:
        dgl_graph.edata["attr"] = torch.as_tensor(
            directed_edge_attributes, dtype=torch.float32
        )
    return dgl_graph


def _seed_evaluator(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summarize(values: Sequence[float]) -> dict[str, float]:
    values_array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
    }


def _evaluate_attributed_feature_modes(
    generated_graphs: Sequence[AttributedGraph],
    reference_graphs: Sequence[AttributedGraph],
    *,
    modes: Sequence[str] = FEATURE_MODES,
    repeats: int = 10,
    seed: int = 0,
    nearest_k: int = 5,
    device="cpu",
) -> dict:
    """Internal attributed-array evaluator used by the public DGL API."""

    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}.")
    if nearest_k < 1:
        raise ValueError(f"nearest_k must be positive, got {nearest_k}.")
    if not modes:
        raise ValueError("At least one feature mode is required.")
    if len(generated_graphs) < 3 or len(reference_graphs) < 3:
        raise ValueError(
            "Random-GIN F1-PR needs at least three graphs in each collection."
        )
    node_dim, edge_dim = validate_collection_dimensions(
        generated_graphs, reference_graphs
    )
    for mode in modes:
        if mode not in FEATURE_MODES:
            raise ValueError(f"Unknown feature mode: {mode}.")
        if mode in {"decoded_edge", "decoded_node_edge"} and edge_dim == 0:
            raise ValueError(f"Mode {mode} requires decoded edge attributes.")

    if str(GGMEVAL_ROOT) not in sys.path:
        sys.path.insert(0, str(GGMEVAL_ROOT))
    try:
        from evaluation.gin_evaluation import (
            MMDEvaluation,
            load_feature_extractor,
            prdcEvaluation,
        )
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "The vendored ggmeval dependencies are unavailable. "
            "Use the same PyTorch/DGL environment as training."
        ) from exc

    dgl_collections = {
        mode: (
            [to_dgl_graph(graph, mode) for graph in generated_graphs],
            [to_dgl_graph(graph, mode) for graph in reference_graphs],
        )
        for mode in modes
    }
    effective_nearest_k = min(
        nearest_k, min(len(generated_graphs), len(reference_graphs)) - 2
    )
    raw_results = {
        mode: {
            "f1_pr": [],
            "precision": [],
            "recall": [],
            "mmd_rbf": [],
            "mmd_linear": [],
        }
        for mode in modes
    }

    for repeat_index in range(repeats):
        repeat_seed = seed + repeat_index
        for mode in modes:
            # The same repeat seed gives every ablation the same GIN weights.
            _seed_evaluator(repeat_seed)
            feature_extractor = load_feature_extractor(
                device=device,
                input_dim=node_dim,
                edge_feat_dim=edge_dim,
                node_feat_loc="attr",
                edge_feat_loc="attr",
            )
            generated_dgl, reference_dgl = dgl_collections[mode]
            activation_metric = MMDEvaluation(feature_extractor, kernel="rbf")
            (generated_activations, reference_activations), _ = (
                activation_metric.get_activations(generated_dgl, reference_dgl)
            )

            rbf_result, _ = activation_metric.evaluate(
                generated_activations, reference_activations
            )
            linear_result, _ = MMDEvaluation(
                feature_extractor, kernel="linear"
            ).evaluate(generated_activations, reference_activations)
            pr_result, _ = prdcEvaluation(
                feature_extractor, use_pr=True
            ).evaluate(
                generated_activations,
                reference_activations,
                nearest_k=effective_nearest_k,
            )

            raw_results[mode]["mmd_rbf"].append(float(rbf_result["mmd_rbf"]))
            raw_results[mode]["mmd_linear"].append(
                float(linear_result["mmd_linear"])
            )
            raw_results[mode]["f1_pr"].append(float(pr_result["f1_pr"]))
            raw_results[mode]["precision"].append(float(pr_result["precision"]))
            raw_results[mode]["recall"].append(float(pr_result["recall"]))

    return {
        "feature_dimensions": {
            "node": node_dim,
            "edge": edge_dim,
        },
        "repeats": repeats,
        "base_seed": seed,
        "nearest_k": effective_nearest_k,
        "modes": {
            mode: {
                "summary": {
                    metric: _summarize(values)
                    for metric, values in mode_results.items()
                },
                "per_repeat": mode_results,
            }
            for mode, mode_results in raw_results.items()
        },
    }


def evaluate_dgl_feature_modes(
    generated_graphs: Sequence,
    reference_graphs: Sequence,
    *,
    modes: Sequence[str] = FEATURE_MODES,
    repeats: int = 10,
    seed: int = 0,
    nearest_k: int = 5,
    device="cpu",
) -> dict:
    """Evaluate generated and reference DGL graphs with attributed Random-GIN.

    This is the primary evaluator API. Both collections must contain individual
    homogeneous DGL graphs following the same ``attr`` feature contract.
    """

    generated_inputs = _materialize_dgl_collection(
        generated_graphs, name="generated_graphs"
    )
    reference_inputs = _materialize_dgl_collection(
        reference_graphs, name="reference_graphs"
    )
    normalized = {}
    for collection_name, inputs in (
        ("generated", generated_inputs),
        ("reference", reference_inputs),
    ):
        converted = []
        for graph_index, graph in enumerate(inputs):
            try:
                converted.append(attributed_graph_from_dgl(graph))
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"{collection_name} graph {graph_index}: {exc}"
                ) from exc
        normalized[collection_name] = converted

    result = _evaluate_attributed_feature_modes(
        normalized["generated"],
        normalized["reference"],
        modes=modes,
        repeats=repeats,
        seed=seed,
        nearest_k=nearest_k,
        device=device,
    )
    result["input_contract"] = {
        "graph_type": "DGLGraph",
        "node_features": "ndata['attr'] float matrix",
        "edge_features": "edata['attr'] float matrix or absent",
        "generated_graphs": len(generated_inputs),
        "reference_graphs": len(reference_inputs),
        "pyg_inputs_accepted": False,
    }
    return result
