"""Shared helpers for the LGD common evaluation (2026-09-25).

Converts LGD's sampled NetworkX pickles into the exact on-disk formats consumed
by the existing corrected evaluators, and verifies that LGD's pickled reference
equals the common reference used for the other methods.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np

GGM_SRC = "/local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ/graph_evaluation/src"
if GGM_SRC not in sys.path:
    sys.path.insert(0, GGM_SRC)

BASE = Path(__import__("os").environ.get("LGD_EVAL_BASE", "/local-scratch2/mirzaei/lgd_common_eval_20260925"))
LGD_REPO = Path("/local-scratch2/mirzaei/LGD_FIX_20260924/repo")

# The common references (read only) used by the corrected evaluations.
REFERENCES = {
    "PTC": ("pyg", "/local-scratch2/mirzaei/defog_ptc_frozen_20260906/artifacts/ptc/real_test_graphs.pt"),
    "TRIANGULAR_GRID": ("pyg", "/local-scratch2/mirzaei/common_eval_fix_20260923/TRIANGULAR_GRID/stage/defog/triangular_grid/real_test_graphs.pt"),
    "GRID": ("pyg", "/local-scratch2/mirzaei/common_eval_fix_20260923/GRID/stage/defog/grid/real_test_graphs.pt"),
    "LOBSTER": ("pyg", "/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/artifacts/lobster/real_test_graphs.pt"),
    "PROTEINS": ("dgl", "/local-scratch2/mirzaei/common_eval_fix_20260923/PROTEINS/stage/reference.bin"),
    "QM9": ("dgl", "/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluations/graphvae/true_full/seed_0/reference_attributed_graphs.bin"),
}
# One-hot widths of each categorical node field, concatenated in this order.
NODE_FIELDS = {"PTC": [19], "PROTEINS": [3], "TRIANGULAR_GRID": [1], "GRID": [1], "LOBSTER": [1], "QM9": [5, 4]}
NODE_DIMS = {k: sum(v) for k, v in NODE_FIELDS.items()}
# Edge one-hot width written to DGL edata['attr'] (only QM9 references carry edge attributes;
# they are not consumed by the topology_control / decoded_node modes).
EDGE_DIMS = {"QM9": 3}
# Generated graphs staged per seed (the evaluator itself uses gen[:len(reference)]).
MAX_GENERATED = {"QM9": 512}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_lgd_pickle(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def simple_topology(graph: nx.Graph) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(graph.nodes())
    g.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    return g


def largest_component_ids(graph: nx.Graph):
    """Mirror ggm_eval.contract._largest_component_node_ids on a relabelled graph."""
    comps = [sorted(c) for c in nx.connected_components(graph) if len(c) > 1]
    if not comps:
        return None
    return sorted(max(comps, key=lambda nodes: (len(nodes), -min(nodes))))


def pyg_to_nx(data) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(int(data.num_nodes)))
    e = data.edge_index.numpy()
    g.add_edges_from((int(a), int(b)) for a, b in zip(e[0], e[1]) if a != b)
    return g


def dgl_to_nx(graph) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(int(graph.num_nodes())))
    s, d = graph.edges()
    g.add_edges_from((int(a), int(b)) for a, b in zip(s.tolist(), d.tolist()) if a != b)
    return g


def lcc_relabel(g: nx.Graph) -> nx.Graph:
    g = simple_topology(nx.convert_node_labels_to_integers(g, ordering="sorted"))
    keep = largest_component_ids(g)
    if keep is None:
        return nx.Graph()
    return nx.convert_node_labels_to_integers(g.subgraph(keep).copy(), ordering="sorted")


def load_reference(dataset):
    kind, path = REFERENCES[dataset]
    if kind == "pyg":
        from ggm_eval.io import load_pyg_collection
        graphs = load_pyg_collection(path)
        return [pyg_to_nx(g) for g in graphs], graphs, path
    import dgl
    graphs, _ = dgl.load_graphs(path)
    return [dgl_to_nx(g) for g in graphs], graphs, path


def compare_collections(a, b):
    """Index-aligned exact edge-set equality, then isomorphism, then multiset."""
    out = {"count_a": len(a), "count_b": len(b)}
    if len(a) != len(b):
        out["index_aligned"] = False
    exact = iso = 0
    mismatches = []
    for i, (x, y) in enumerate(zip(a, b)):
        ex = (x.number_of_nodes() == y.number_of_nodes()
              and {tuple(sorted(e)) for e in x.edges()} == {tuple(sorted(e)) for e in y.edges()})
        if ex:
            exact += 1
            iso += 1
            continue
        if x.number_of_nodes() == y.number_of_nodes() and x.number_of_edges() == y.number_of_edges() and nx.is_isomorphic(x, y):
            iso += 1
        else:
            mismatches.append(i)
    out.update(exact_index_aligned=exact, isomorphic_index_aligned=iso, mismatched_indices=mismatches[:20],
               n_mismatched=len(mismatches))
    ha = sorted(nx.weisfeiler_lehman_graph_hash(g) for g in a)
    hb = sorted(nx.weisfeiler_lehman_graph_hash(g) for g in b)
    out["wl_multiset_equal"] = ha == hb
    return out


def onehot(labels, dim):
    x = np.zeros((len(labels), dim), dtype=np.float32)
    for row, value in enumerate(labels):
        if not 0 <= int(value) < dim:
            raise ValueError(f"label {value} outside [0,{dim})")
        x[row, int(value)] = 1.0
    return x


def onehot_fields(rows, fields):
    """rows: list of per-node lists of categorical values, one per field."""
    rows = [list(r) if isinstance(r, (list, tuple, np.ndarray)) else [r] for r in rows]
    parts = [onehot([r[i] for r in rows], width) for i, width in enumerate(fields)]
    return np.concatenate(parts, axis=1) if parts else np.zeros((len(rows), 0), np.float32)


def lgd_graph_arrays(graph: nx.Graph, fields, topology_only: bool, edge_dim: int = 0):
    """Return (edges u<v, node one-hot, per-node label lists, edge one-hot or None) for one
    LGD graph, relabelled 0..n-1 by sorted node id."""
    nodes = sorted(graph.nodes())
    mapping = {node: i for i, node in enumerate(nodes)}
    if topology_only:
        labels = [[0] for _ in nodes]
    else:
        labels = []
        for node in nodes:
            cats = graph.nodes[node].get("categorical_features")
            if cats is None:
                cats = [int(graph.nodes[node]["label"])]
            cats = [int(c) for c in cats]
            if len(cats) != len(fields):
                raise ValueError(f"node has {len(cats)} categorical fields, expected {len(fields)}")
            if len(cats) == 1 and "label" in graph.nodes[node] and int(graph.nodes[node]["label"]) != cats[0]:
                raise ValueError("label and categorical_features disagree")
            labels.append(cats)
    edge_map = {}
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        key = tuple(sorted((mapping[u], mapping[v])))
        feats = data.get("categorical_features") or ([int(data["label"])] if "label" in data else [0])
        edge_map[key] = int(feats[0]) if feats else 0
    edges = sorted(edge_map)
    edge_x = onehot([edge_map[e] for e in edges], edge_dim) if edge_dim else None
    return (np.asarray(edges, dtype=np.int64).reshape(-1, 2), onehot_fields(labels, fields), labels, edge_x)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
