"""Independent structural MMD evaluation for terminal motif-matrix chains.

Each selected rule/value chain is evaluated as its own generated graph set.
Matrices from different chains are never added, averaged, or OR-combined.
The statistics and Gaussian-TV kernels match ``stat_rnn.mmd_eval``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from stat_rnn import orca, spectral_worker


METRIC_SPECS = {
    "degree": {"is_hist": True, "sigma": 1.0},
    "clustering": {"is_hist": True, "sigma": 0.1},
    "sparsity": {"is_hist": False, "sigma": 30.0},
    "orbit": {"is_hist": False, "sigma": 30.0},
    "spectral": {"is_hist": True, "sigma": 1.0},
    "triangle": {"is_hist": False, "sigma": 30.0},
    "diameter": {"is_hist": False, "sigma": 1.0},
}


@dataclass(frozen=True)
class PreparedGraph:
    """A canonical graph signature and the graph used for statistics."""

    signature: bytes
    graph: nx.Graph


def _canonical_signature(graph: nx.Graph) -> bytes:
    graph = nx.convert_node_labels_to_integers(
        graph, ordering="sorted", label_attribute=None
    )
    adjacency = nx.to_numpy_array(graph, dtype=np.uint8)
    packed = np.packbits(adjacency.reshape(-1), bitorder="little").tobytes()
    return int(graph.number_of_nodes()).to_bytes(4, "little") + packed


def prepare_generated_adjacency(adjacency: np.ndarray) -> PreparedGraph | None:
    """Binarize, remove self-loops/isolates, and retain the largest component."""

    array = (np.asarray(adjacency) >= 0.5).astype(np.uint8, copy=False)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected square adjacency, got {array.shape}.")
    array = array.copy()
    np.fill_diagonal(array, 0)
    graph = nx.from_numpy_array(array, create_using=nx.Graph)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None
    largest_nodes = max(nx.connected_components(graph), key=len)
    graph = nx.Graph(graph.subgraph(largest_nodes))
    graph = nx.convert_node_labels_to_integers(
        graph, ordering="sorted", label_attribute=None
    )
    return PreparedGraph(_canonical_signature(graph), graph)


def prepare_reference_adjacency(adjacency: np.ndarray) -> PreparedGraph | None:
    """Use the same reference cleanup as ``stat_rnn.mmd_eval``."""

    array = np.asarray(adjacency)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected square adjacency, got {array.shape}.")
    graph = nx.from_numpy_array(array, create_using=nx.Graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if graph.number_of_nodes() == 0:
        return None
    graph = nx.convert_node_labels_to_integers(
        graph, ordering="sorted", label_attribute=None
    )
    return PreparedGraph(_canonical_signature(graph), graph)


def _graph_features(prepared: PreparedGraph) -> Tuple[bytes, Dict[str, Any]]:
    graph = prepared.graph
    node_count = graph.number_of_nodes()
    features: Dict[str, Any] = {
        "degree": np.asarray(nx.degree_histogram(graph), dtype=float),
        "clustering": np.histogram(
            list(nx.clustering(graph).values()),
            bins=100,
            range=(0.0, 1.0),
            density=False,
        )[0].astype(float),
        "sparsity": np.asarray(
            [(node_count**2 - graph.number_of_edges()) / node_count**2],
            dtype=float,
        ),
        "spectral": np.asarray(spectral_worker(graph), dtype=float),
        "triangle": np.asarray(
            [np.sum(list(nx.triangles(graph).values())) / node_count],
            dtype=float,
        ),
        "edge_count": float(graph.number_of_edges()),
        "diameter": None,
        "orbit": None,
        "orbit_error": None,
    }
    try:
        if nx.is_connected(graph):
            diameter = nx.diameter(graph)
        else:
            # Terminal-rule graphs can legitimately be disconnected or entirely
            # edgeless. Use the largest component diameter so every real graph
            # remains evaluable without dropping its isolated real nodes.
            diameter = max(
                nx.diameter(graph.subgraph(component))
                for component in nx.connected_components(graph)
            )
        features["diameter"] = np.asarray([diameter], dtype=float)
    except (nx.NetworkXError, ValueError):
        pass
    try:
        orbit_counts = orca(graph)
        features["orbit"] = (
            np.sum(orbit_counts, axis=0).astype(float) / node_count
        )
    except Exception as exc:  # Match stat_rnn.orbit_stats_all's skip behavior.
        features["orbit_error"] = repr(exc)
    return prepared.signature, features


def compute_feature_cache(
    prepared_graphs: Iterable[PreparedGraph],
    *,
    workers: int = 1,
) -> Dict[bytes, Dict[str, Any]]:
    """Compute each unique graph's statistics once."""

    unique: Dict[bytes, PreparedGraph] = {}
    for prepared in prepared_graphs:
        unique.setdefault(prepared.signature, prepared)

    if workers <= 1:
        return dict(_graph_features(graph) for graph in unique.values())

    cache: Dict[bytes, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_graph_features, graph) for graph in unique.values()
        ]
        for future in as_completed(futures):
            signature, features = future.result()
            cache[signature] = features
    return cache


def _as_matrix(samples: Sequence[np.ndarray]) -> np.ndarray:
    max_width = max(np.asarray(sample).reshape(-1).shape[0] for sample in samples)
    matrix = np.zeros((len(samples), max_width), dtype=float)
    for row, sample in enumerate(samples):
        flattened = np.asarray(sample, dtype=float).reshape(-1)
        matrix[row, : flattened.shape[0]] = flattened
    return matrix


def _normalize_histograms(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError("Histogram statistic contains an all-zero sample.")
    return matrix / totals


def gaussian_tv_mean(
    samples1: Sequence[np.ndarray],
    samples2: Sequence[np.ndarray],
    *,
    sigma: float,
    is_hist: bool,
) -> float:
    """Mean Gaussian-TV kernel, vectorized but algebraically equal to mmd_rnn."""

    if not samples1 or not samples2:
        raise ValueError("Gaussian-TV requires two nonempty sample sets.")
    width = max(
        max(np.asarray(sample).reshape(-1).shape[0] for sample in samples1),
        max(np.asarray(sample).reshape(-1).shape[0] for sample in samples2),
    )

    def padded(samples: Sequence[np.ndarray]) -> np.ndarray:
        matrix = np.zeros((len(samples), width), dtype=float)
        for row, sample in enumerate(samples):
            flattened = np.asarray(sample, dtype=float).reshape(-1)
            matrix[row, : flattened.shape[0]] = flattened
        return _normalize_histograms(matrix) if is_hist else matrix

    first = padded(samples1)
    second = padded(samples2)
    tv = cdist(first, second, metric="cityblock") / 2.0
    return float(np.exp(-(tv**2) / (2.0 * sigma**2)).mean())


def gaussian_tv_mmd(
    reference: Sequence[np.ndarray],
    generated: Sequence[np.ndarray],
    *,
    sigma: float,
    is_hist: bool,
    reference_self_kernel: float | None = None,
) -> Tuple[float, float]:
    """Biased MMD used by ``mmd_rnn.compute_mmd``."""

    ref_self = (
        gaussian_tv_mean(reference, reference, sigma=sigma, is_hist=is_hist)
        if reference_self_kernel is None
        else reference_self_kernel
    )
    generated_self = gaussian_tv_mean(
        generated, generated, sigma=sigma, is_hist=is_hist
    )
    cross = gaussian_tv_mean(
        reference, generated, sigma=sigma, is_hist=is_hist
    )
    return float(ref_self + generated_self - 2.0 * cross), float(ref_self)


def feature_samples(
    signatures: Sequence[bytes],
    cache: Mapping[bytes, Mapping[str, Any]],
    metric: str,
) -> List[np.ndarray]:
    return [
        np.asarray(cache[signature][metric], dtype=float)
        for signature in signatures
        if cache[signature].get(metric) is not None
    ]


def reference_metric_context(
    reference_signatures: Sequence[bytes],
    cache: Mapping[bytes, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Precompute reference statistics and their invariant self-kernel."""

    context: Dict[str, Dict[str, Any]] = {}
    for metric, spec in METRIC_SPECS.items():
        samples = feature_samples(reference_signatures, cache, metric)
        if not samples:
            raise RuntimeError(f"No valid reference samples for {metric}.")
        ref_self = gaussian_tv_mean(
            samples,
            samples,
            sigma=float(spec["sigma"]),
            is_hist=bool(spec["is_hist"]),
        )
        context[metric] = {
            "samples": samples,
            "self_kernel": ref_self,
            "sample_count": len(samples),
        }
    return context


def evaluate_signature_chain(
    generated_signatures: Sequence[bytes],
    *,
    reference_context: Mapping[str, Mapping[str, Any]],
    cache: Mapping[bytes, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate one chain without consulting any other chain."""

    metrics: Dict[str, float | None] = {}
    metric_sample_counts: Dict[str, int] = {}
    for metric, spec in METRIC_SPECS.items():
        generated = feature_samples(generated_signatures, cache, metric)
        metric_sample_counts[metric] = len(generated)
        if not generated:
            metrics[metric] = None
            continue
        reference = reference_context[metric]["samples"]
        value, _ = gaussian_tv_mmd(
            reference,
            generated,
            sigma=float(spec["sigma"]),
            is_hist=bool(spec["is_hist"]),
            reference_self_kernel=float(
                reference_context[metric]["self_kernel"]
            ),
        )
        metrics[metric] = value

    generated_edges = [
        float(cache[signature]["edge_count"]) for signature in generated_signatures
    ]
    reference_edge_values = [
        float(value)
        for value in reference_context["edge_count"]["samples"]
    ] if "edge_count" in reference_context else []
    return {
        "metrics": metrics,
        "metric_generated_sample_counts": metric_sample_counts,
        "generated_edge_count": (
            float(np.mean(generated_edges)) if generated_edges else None
        ),
        "reference_edge_count": (
            float(np.mean(reference_edge_values))
            if reference_edge_values
            else None
        ),
    }
