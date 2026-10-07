#!/usr/bin/env python3
"""Stage LGD sampled graphs in the exact input format of the corrected evaluators.

Usage: stage_lgd.py DATASET SEED [SEED ...]

PTC / TRIANGULAR_GRID / GRID -> ggm-eval-pyg-tensors collections
    <BASE>/<DATASET>/stage/lgd/<slug>/real_test_graphs.pt   (byte copy of the common reference)
    <BASE>/<DATASET>/stage/lgd/<slug>/generated/seed_<k>/generated_graphs.pt
PROTEINS -> DGL binary with ndata['attr'] one-hot (3 columns)
    <BASE>/PROTEINS/stage/lgd/seed_<k>/generated.bin

Normalization is the common policy used for every other method: remove self-loops,
keep the deterministic largest connected component (ggm_eval.contract rule), drop
isolated nodes.  LGD categorical class k maps to one-hot column k (verified in
reference_verification.json / RESULTS.md).  Edgeless samples have no component with
an edge; they are replaced in place by a single edge (K2) carrying the sample's first
node label(s) so that the collection size and the common reference remain unchanged.
Every such substitution is recorded in stage_manifest.json.
"""
import shutil
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lgd_common import (BASE, EDGE_DIMS, MAX_GENERATED, NODE_FIELDS, REFERENCES, largest_component_ids, lgd_graph_arrays,
                        load_lgd_pickle, sha256_file, write_json)

SLUG = {"PTC": "ptc", "TRIANGULAR_GRID": "triangular_grid", "GRID": "grid", "LOBSTER": "lobster"}
SCHEMA = {  # must equal the reference collection's declared schema (ggm_eval checks it)
    "PTC": {"dataset": "PTC", "feature_schema": "gin-node-label-v2|export=decoded_node", "feature_mode": "decoded_node"},
    "TRIANGULAR_GRID": {"dataset": "TRIANGULAR_GRID", "feature_mode": "topology_control"},
    "GRID": {"dataset": "GRID", "feature_mode": "topology_control"},
    "LOBSTER": {"dataset": "LOBSTER", "feature_mode": "topology_control"},
}


def normalized_graphs(generated, dataset):
    fields = NODE_FIELDS[dataset]
    dim = sum(fields)
    edge_dim = EDGE_DIMS.get(dataset, 0)
    topology_only = dim == 1
    records, log = [], []
    for index, graph in enumerate(generated):
        edges, x, labels, edge_x = lgd_graph_arrays(graph, fields, topology_only, edge_dim)
        n = x.shape[0]
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(map(tuple, edges.tolist()))
        keep = largest_component_ids(g)
        entry = {"index": index, "raw_nodes": n, "raw_edges": int(len(edges)),
                 "raw_components_with_edges": sum(1 for c in nx.connected_components(g) if len(c) > 1),
                 "raw_isolated_nodes": nx.number_of_isolates(g)}
        if keep is None:
            # K2 carrying the labels of the sample's first two nodes (or its only node twice).
            src_rows = (list(range(x.shape[0])) * 2)[:2] if x.shape[0] else None
            x = x[src_rows] if src_rows else np.eye(1, dim, dtype=np.float32).repeat(2, 0)
            edges = np.asarray([[0, 1]], dtype=np.int64)
            edge_x = np.eye(1, edge_dim, dtype=np.float32) if edge_dim else None
            entry["substitution"] = "edgeless sample replaced by K2"
        else:
            local = {node: i for i, node in enumerate(keep)}
            x = x[keep]
            rows = [i for i, (u, v) in enumerate(edges.tolist()) if u in local and v in local]
            edges = np.asarray([(local[u], local[v]) for u, v in edges[rows].tolist()], dtype=np.int64).reshape(-1, 2)
            edge_x = edge_x[rows] if edge_dim else None
        entry.update(kept_nodes=int(x.shape[0]), kept_edges=int(len(edges)))
        records.append((edges, x, edge_x))
        log.append(entry)
    return records, log


def stage_pyg(dataset, seed, generated):
    from ggm_eval.adapters import attributed_arrays_to_pyg
    from ggm_eval.io import save_pyg_collection
    root = BASE / dataset / "stage" / "lgd" / SLUG[dataset]
    ref_src = Path(REFERENCES[dataset][1])
    ref_dst = root / "real_test_graphs.pt"
    if not ref_dst.exists():
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ref_src, ref_dst)
        if (ref_src.parent / (ref_src.name + ".json")).exists():
            shutil.copy2(ref_src.parent / (ref_src.name + ".json"), root / "real_test_graphs.pt.json")
    assert sha256_file(ref_dst) == sha256_file(ref_src)
    records, log = normalized_graphs(generated, dataset)
    graphs = [attributed_arrays_to_pyg(e, x, ex, np.arange(x.shape[0]), name=f"lgd-{dataset}-{seed}-{i}")
              for i, (e, x, ex) in enumerate(records)]
    out = root / "generated" / f"seed_{seed}" / "generated_graphs.pt"
    meta = dict(SCHEMA[dataset])
    meta.update(generator="LGD", training_seed=seed, split="generated", accepted_count=len(graphs),
                checkpoint="epoch_1999", substitutions=sum("substitution" in e for e in log),
                postprocessing={"self_loops": "removed", "connected_components": "deterministic_largest_component",
                                "isolated_nodes": "removed", "edgeless_samples": "replaced_by_K2"})
    save_pyg_collection(out, graphs, metadata=meta)
    return out, log


def stage_dgl(dataset, seed, generated):
    import dgl
    import torch
    records, log = normalized_graphs(generated, dataset)
    graphs = []
    for edges, x, edge_x in records:
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])
        g = dgl.graph((torch.as_tensor(src), torch.as_tensor(dst)), num_nodes=x.shape[0])
        g.ndata["attr"] = torch.as_tensor(x, dtype=torch.float32)
        if edge_x is not None:
            g.edata["attr"] = torch.as_tensor(np.concatenate([edge_x, edge_x], axis=0), dtype=torch.float32)
        graphs.append(g)
    out = BASE / dataset / "stage" / "lgd" / f"seed_{seed}" / "generated.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(out), graphs)
    # dgl.save_graphs is not byte-deterministic; record a content digest of what was written.
    import hashlib
    reloaded, _ = dgl.load_graphs(str(out))
    h = hashlib.sha256()
    for g in reloaded:
        s_, d_ = g.edges(order="eid")
        for t in (s_, d_, g.ndata["attr"], g.edata.get("attr", torch.zeros(0))):
            h.update(t.numpy().tobytes())
        h.update(str(g.num_nodes()).encode())
    log.append({"content_sha256": h.hexdigest()})
    return out, log


def main():
    dataset = sys.argv[1]
    for seed in [int(s) for s in sys.argv[2:]]:
        # final-epoch sample pickle (epoch_1999 for 2000-epoch runs, epoch_299 for QM9)
        pkl = sorted((BASE / "raw_pickles" / f"{dataset}_seed{seed}").glob("epoch_*_graphs.pkl"))[-1]
        generated = load_lgd_pickle(pkl)["generated"]
        if dataset in MAX_GENERATED:
            generated = generated[:MAX_GENERATED[dataset]]
        if dataset in SLUG:
            out, log = stage_pyg(dataset, seed, generated)
        elif dataset in ("PROTEINS", "QM9"):
            out, log = stage_dgl(dataset, seed, generated)
        else:
            raise SystemExit(f"no staging rule for {dataset}")
        content = [e for e in log if "content_sha256" in e]
        log = [e for e in log if "content_sha256" not in e]
        summary = {
            "dataset": dataset, "seed": seed, "source_pickle": str(pkl), "source_sha256": sha256_file(pkl),
            "output": str(out), "output_sha256": sha256_file(out),
            "content_sha256": content[0]["content_sha256"] if content else None, "generated_count": len(log),
            "edgeless_substituted_indices": [e["index"] for e in log if "substitution" in e],
            "disconnected_samples": sum(e["raw_components_with_edges"] > 1 for e in log),
            "samples_with_isolates": sum(e["raw_isolated_nodes"] > 0 for e in log),
            "raw_nodes_total": sum(e["raw_nodes"] for e in log),
            "kept_nodes_total": sum(e["kept_nodes"] for e in log),
            "raw_edges_total": sum(e["raw_edges"] for e in log),
            "kept_edges_total": sum(e["kept_edges"] for e in log),
            "per_graph": log,
        }
        write_json(out.parent / "stage_manifest.json", summary)
        print({k: v for k, v in summary.items() if k != "per_graph"})


if __name__ == "__main__":
    main()
