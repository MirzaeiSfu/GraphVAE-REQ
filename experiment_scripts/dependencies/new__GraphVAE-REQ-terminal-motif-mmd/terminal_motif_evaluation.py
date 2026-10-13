"""Structural MMD evaluation for terminal motif matrix-chain outputs.

The motif counter's matrix mode returns the final matrix from every retained
rule/value chain.  Not every final matrix is an adjacency matrix: unary rules
can end in N x 1, 1 x N, or 1 x 1 outputs.  This module keeps only N x N
outputs, binarizes them, removes the one-atom raw-relation chain, and takes
their elementwise union to form one terminal motif graph per generated sample.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch


MMD_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MMD_RESULT_LABELS = {
    "degree": "degree",
    "clustering": "clustering",
    "sparsity": "sparsity",
    "orbit": "orbits",
    "spectral": "Spec",
    "triangle": "Tri",
    "diameter": "diameter",
    "mmd_rbf": "mmd_rbf",
    "mmd_rbf_std": "mmd_rbf_std",
    "precision": "precision",
    "precision_std": "precision_std",
    "recall": "recall",
    "recall_std": "recall_std",
    "f1_pr": "f1_pr",
    "f1_pr_std": "f1_pr_std",
    "ggm_gnn_runs": "ggm_gnn_runs",
}
MMD_EDGE_COUNT_LABELS = {
    "reference_edge_count": "average edge # in test set",
    "generated_edge_count": "average edge # in grnrated set",
}


def _json_safe(value: Any) -> Any:
    """Convert cache metadata into deterministic JSON-compatible objects."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path | str, payload: Dict[str, Any] | List[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)


def actual_matrix_shape(mask: torch.Tensor | np.ndarray) -> Tuple[int, int]:
    """Recover the unpadded H x W shape encoded by a validity mask."""

    mask_array = (
        mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    ).astype(bool, copy=False)
    occupied_rows = np.flatnonzero(mask_array.any(axis=1))
    occupied_cols = np.flatnonzero(mask_array.any(axis=0))
    height = int(occupied_rows[-1] + 1) if occupied_rows.size else 0
    width = int(occupied_cols[-1] + 1) if occupied_cols.size else 0
    return height, width


def is_raw_relation_chain(counter: Any, rule_index: int) -> bool:
    """Identify the one-atom adjacency relation, not a composed motif chain."""

    rule = counter.rules[rule_index]
    states = counter.states[rule_index]
    if len(rule) != 1 or list(states) != [2]:
        return False
    relation_names = tuple(f"{name}(" for name in counter.relation_keys)
    return any(str(rule[0]).startswith(prefix) for prefix in relation_names)


def build_chain_metadata(
    counter: Any,
    valid_mask: torch.Tensor,
    *,
    matrix_size: int,
    exclude_raw_relation: bool = True,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Describe every retained chain and select adjacency-shaped motif outputs."""

    entries: List[Dict[str, Any]] = []
    selected_indices: List[int] = []
    chain_index = 0

    for rule_index, rows in enumerate(counter.values):
        for value_index, value in enumerate(rows):
            height, width = actual_matrix_shape(valid_mask[chain_index])
            raw_relation = is_raw_relation_chain(counter, rule_index)

            exclusion_reason = None
            if height != matrix_size or width != matrix_size:
                exclusion_reason = "terminal_output_is_not_N_by_N"
            elif exclude_raw_relation and raw_relation:
                exclusion_reason = "one_atom_raw_relation_is_not_a_motif_chain"
            else:
                selected_indices.append(chain_index)

            entries.append(
                {
                    "chain_index": chain_index,
                    "rule_index": rule_index,
                    "value_index": value_index,
                    "rule": _json_safe(counter.rules[rule_index]),
                    "states": _json_safe(counter.states[rule_index]),
                    "value": _json_safe(value),
                    "rule_source": _json_safe(counter.rule_sources[rule_index]),
                    "terminal_shape": [height, width],
                    "is_raw_relation_chain": raw_relation,
                    "selected_for_terminal_graph": exclusion_reason is None,
                    "exclusion_reason": exclusion_reason,
                }
            )
            chain_index += 1

    if chain_index != int(valid_mask.shape[0]):
        raise RuntimeError(
            "Motif iteration metadata does not match returned matrix count: "
            f"metadata={chain_index}, matrices={valid_mask.shape[0]}."
        )
    if not selected_indices:
        raise RuntimeError(
            "No adjacency-shaped terminal motif chains remain after filtering."
        )
    return entries, selected_indices


def binarize_terminal_matrices(
    matrices: torch.Tensor,
    valid_mask: torch.Tensor,
    selected_indices: Sequence[int],
    *,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binarize selected matrices and return per-chain and union adjacencies.

    Binarization is deliberately performed before any graph construction or
    structural metric:

        value >= threshold -> 1
        value <  threshold -> 0

    The diagonal is cleared after the chain union, matching the main evaluator's
    removal of self-loops.
    """

    if matrices.ndim != 4 or valid_mask.ndim != 3:
        raise ValueError(
            "Expected matrices (B,M,N,N) and valid_mask (M,N,N), got "
            f"{tuple(matrices.shape)} and {tuple(valid_mask.shape)}."
        )

    index = torch.as_tensor(
        list(selected_indices), dtype=torch.long, device=matrices.device
    )
    selected = matrices.index_select(1, index)
    selected_mask = valid_mask.to(matrices.device).index_select(0, index)
    binary = (selected >= float(threshold)) & selected_mask.unsqueeze(0)
    union = binary.any(dim=1)

    diagonal = torch.arange(union.shape[-1], device=union.device)
    union[:, diagonal, diagonal] = False

    return (
        binary.to(torch.uint8).detach().cpu().numpy(),
        union.to(torch.uint8).detach().cpu().numpy(),
    )


def terminal_adjacencies_to_graphs(
    adjacencies: Sequence[np.ndarray],
) -> Tuple[List[nx.Graph], Dict[str, int]]:
    """Apply the same graph cleanup used by ``main.EvalTwoSet``."""

    graphs: List[nx.Graph] = []
    empty_count = 0
    for adjacency in adjacencies:
        graph = nx.from_numpy_array(np.asarray(adjacency), create_using=nx.Graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph.remove_nodes_from(list(nx.isolates(graph)))
        if nx.is_empty(graph):
            empty_count += 1
            continue
        largest_nodes = max(nx.connected_components(graph), key=len)
        graphs.append(nx.Graph(graph.subgraph(largest_nodes)))

    return graphs, {
        "input_graph_count": len(adjacencies),
        "nonempty_largest_component_count": len(graphs),
        "empty_graph_count": empty_count,
    }


def reference_adjacencies_to_graphs(
    adjacencies: Sequence[np.ndarray],
) -> List[nx.Graph]:
    """Match main.py's test-reference conversion without extra filtering."""

    return [
        nx.from_numpy_array(np.asarray(adjacency), create_using=nx.Graph)
        for adjacency in adjacencies
    ]


def parse_graph_quality_result(mmd_result: str) -> Dict[str, float | None]:
    metrics: Dict[str, float | None] = {}
    for metric_name, result_label in MMD_RESULT_LABELS.items():
        match = re.search(
            rf"{re.escape(result_label)}\s*:\s*({MMD_FLOAT_PATTERN})",
            str(mmd_result),
        )
        metrics[metric_name] = float(match.group(1)) if match else None
    for metric_name, result_label in MMD_EDGE_COUNT_LABELS.items():
        match = re.search(
            rf"{re.escape(result_label)}\s*:\s*({MMD_FLOAT_PATTERN})",
            str(mmd_result),
        )
        metrics[metric_name] = float(match.group(1)) if match else None
    return metrics


def evaluate_terminal_adjacencies(
    generated_adjacencies: Sequence[np.ndarray],
    reference_adjacencies: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """Compute the repository's complete structural MMD result."""

    from stat_rnn import mmd_eval

    generated_graphs, cleanup = terminal_adjacencies_to_graphs(
        generated_adjacencies
    )
    if not generated_graphs:
        raise RuntimeError(
            "Every binarized terminal motif graph is empty; MMD is undefined."
        )
    reference_graphs = reference_adjacencies_to_graphs(reference_adjacencies)
    raw_result = mmd_eval(generated_graphs, reference_graphs, diam=True)
    return {
        "metrics": parse_graph_quality_result(raw_result),
        "raw_eval_result": raw_result,
        "graph_cleanup": cleanup,
    }
