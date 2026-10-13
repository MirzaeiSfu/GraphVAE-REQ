#!/usr/bin/env python3
"""Convert re-generated DeFoG samples to the Rule-MMD DGL format.

Pipeline per raw DeFoG sample [node_labels, E_dense] (exactly as the frozen benchmark):
  1. DeFoG GraphDiscreteFlowModel._sample_to_pyg(node_only=True) equivalent:
     x = one_hot(node_labels, 1) (topology_control => ones[n,1]),
     undirected edges {(i,j): i<j, E[i,j] > 0}, stored (i,j) block then (j,i) block.
  2. Strict acceptance (graph_discrete_flow_model.sample, strict_generation=true):
     ggm_eval.contract.normalize_pyg_graph(candidate) must not raise TypeError/ValueError.
  3. Export = ggm_eval.io.save_pyg_collection(normalize=True) => prepare_collection(
     mode='decoded_node_edge') => deterministic largest connected component.
  4. DGL: dgl.graph(edge_index) (both directions, no self loops), ndata['attr'] zeros
     [n, D_node], edata['attr'] zeros [E, D_edge] (topology-only, like the existing
     datasets/<DS>/generated/defog/seed_k/generated.bin).

Modes:
  --selftest ORIG_PKL ORIG_PT : run steps 1-3 on the original 20-sample pkl and check
                                the collection digest equals the original generated_graphs.pt.
  default                      : build generated_500.{bin,json} from stream dirs.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

GE_SRC = "/local-scratch2/mirzaei/synfix_defog_work/bundle/graph_evaluation/src"
sys.path.insert(0, GE_SRC)
from torch_geometric.data import Data  # noqa: E402
from ggm_eval.contract import normalize_pyg_graph, collection_digest  # noqa: E402
from ggm_eval.io import save_pyg_collection, load_pyg_collection_with_metadata  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sample_to_pyg(sample):
    node_labels, edge_labels = sample
    node_labels = torch.as_tensor(node_labels, dtype=torch.long)
    assert int(node_labels.max()) == 0 and int(node_labels.min()) == 0, "topology-only expected"
    n = int(node_labels.shape[0])
    x = F.one_hot(node_labels, num_classes=1).to(torch.float32)
    E = torch.as_tensor(edge_labels, dtype=torch.long)
    iu = torch.triu_indices(n, n, offset=1)
    keep = E[iu[0], iu[1]] > 0
    und = iu[:, keep]  # row-major order (src asc, dst asc) == DeFoG double loop order
    if und.shape[1]:
        edge_index = torch.cat([und, und.flip(0)], dim=1).contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    g = Data(x=x, edge_index=edge_index, edge_attr=None, num_nodes=n)
    g.source_node_ids = torch.arange(n, dtype=torch.long)
    return g


def accept(sample):
    cand = sample_to_pyg(sample)
    try:
        normalize_pyg_graph(cand, name="generated candidate")
    except (TypeError, ValueError):
        return None
    return cand


def read_progress(path):
    recs = []
    with open(path, "rb") as f:
        while True:
            try:
                recs.append(pickle.load(f))
            except EOFError:
                break
            except Exception:  # truncated trailing record (file still being appended)
                break
    return recs


def selftest(orig_pkl, orig_pt, out_pt):
    samples = pickle.load(open(orig_pkl, "rb"))
    graphs = [sample_to_pyg(s) for s in samples]
    man = save_pyg_collection(out_pt, graphs, metadata={"selftest": True})
    orig = json.load(open(orig_pt + ".json"))
    ok = man["collection_sha256"] == orig["collection_sha256"]
    print(json.dumps({"n": len(samples), "mine": man["collection_sha256"],
                      "orig": orig["collection_sha256"], "match": ok}))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", nargs=2)
    ap.add_argument("--dataset")
    ap.add_argument("--train-seed", type=int)
    ap.add_argument("--streams", nargs="*", help="stream run dirs, in order of use")
    ap.add_argument("--n", type=int)
    ap.add_argument("--ckpt")
    ap.add_argument("--orig-ckpt-path")
    ap.add_argument("--out-dir")
    ap.add_argument("--work-dir", default="/local-scratch2/mirzaei/synfix_defog_work/convert")
    a = ap.parse_args()
    if a.selftest:
        os.makedirs(a.work_dir, exist_ok=True)
        ok = selftest(a.selftest[0], a.selftest[1], os.path.join(a.work_dir, "selftest.pt"))
        sys.exit(0 if ok else 1)

    import dgl
    if a.orig_ckpt_path is None:
        a.orig_ckpt_path = json.load(open("/local-scratch2/mirzaei/synfix_defog_work/orig_ckpt_paths.json"))[f"{a.dataset}_{a.train_seed}"]
    sys.path.insert(0, "/local-scratch2/mirzaei/rule_mmd_20260923/source/GraphVAE-REQ")
    cache_path = f"/local-scratch2/mirzaei/rule_mmd_20260923/datasets/{a.dataset}/dataset_cache.pkl"
    cache = pickle.load(open(cache_path, "rb"))
    Dn = len(cache["node_onehot_info"] or {})
    De = len(cache["edge_onehot_info"] or {})

    raw_accepted, per_stream = [], []
    need = a.n
    for sd in a.streams:
        inv = json.load(open(os.path.join(sd, "synfix_invocation.json")))
        gen_seed = int([o for o in inv["overrides"] if o.startswith("general.generation_seed=")][0].split("=")[1])
        recs = read_progress(os.path.join(sd, "progress_batches.pkl"))
        attempted = accepted = rejected = used = 0
        secs = 0.0
        nb = 0
        for r in recs:
            if need <= 0:
                break
            nb += 1
            secs += r["seconds"]
            for s in r["samples"]:
                attempted += 1
                c = accept(s)
                if c is None:
                    rejected += 1
                    continue
                accepted += 1
                if need > 0:
                    raw_accepted.append((s, c, sd, gen_seed))
                    used += 1
                    need -= 1
        per_stream.append({"stream_dir": sd, "host": inv["host"], "generation_seed": gen_seed,
                           "sampling_batch": inv["sampling_batch"],
                           "cuda_visible_devices": inv["cuda_visible_devices"],
                           "batches_used": nb, "batches_available": len(recs),
                           "attempted_in_used_batches": attempted, "accepted_in_used_batches": accepted,
                           "rejected_in_used_batches": rejected, "used": used,
                           "sampling_seconds_used_batches": secs})
    assert len(raw_accepted) == a.n, f"only {len(raw_accepted)} accepted < {a.n}"

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.work_dir, exist_ok=True)
    tag = f"{a.dataset}_seed_{a.train_seed}"
    pt_path = os.path.join(a.work_dir, f"{tag}_generated_graphs.pt")
    man = save_pyg_collection(pt_path, [c for _, c, _, _ in raw_accepted], metadata={
        "generator": "DeFoG", "dataset": a.dataset, "feature_mode": "topology_control",
        "feature_schema": "constant-node|export=topology_control", "split": "generated",
        "training_seed": a.train_seed, "checkpoint_sha256": sha(a.ckpt),
        "generation_seeds": sorted({g for *_, g in raw_accepted}),
        "accepted_count": a.n})
    payload = torch.load(pt_path)
    recs = payload["graphs"]
    assert len(recs) == a.n

    out, nodes, dedges, lcc_dropped = [], [], [], 0
    for i, (rec, (s, cand, _, _)) in enumerate(zip(recs, raw_accepted)):
        n = int(rec["num_nodes"])
        ei = rec["edge_index"].long()
        assert int((ei[0] == ei[1]).sum()) == 0
        pairs = set(map(tuple, ei.t().tolist()))
        assert len(pairs) == ei.shape[1] and all((v, u) in pairs for u, v in pairs)
        G = dgl.graph((ei[0], ei[1]), num_nodes=n)
        G.ndata["attr"] = torch.zeros(n, Dn)
        if De:
            G.edata["attr"] = torch.zeros(ei.shape[1], De)
        out.append(G)
        nodes.append(n)
        dedges.append(int(ei.shape[1]))
        if n != int(s[0].shape[0]):
            lcc_dropped += 1
    bin_path = os.path.join(a.out_dir, "generated_500.bin")
    dgl.save_graphs(bin_path, out, {"has_native_features": torch.zeros(a.n, dtype=torch.int64),
                                    "graph_index": torch.arange(a.n, dtype=torch.int64)})

    # Verification against the raw samples: reload and compare edge sets with raw E
    # (restricted to the deterministic largest component, mapped via source_node_ids).
    back, _ = dgl.load_graphs(bin_path)
    mism = 0
    for G, rec, (s, _, _, _) in zip(back, recs, raw_accepted):
        src_ids = rec.get("source_node_ids")
        src_ids = torch.arange(G.num_nodes()) if src_ids is None else src_ids.long()
        E = torch.as_tensor(s[1])
        sub = E[src_ids][:, src_ids] > 0
        sub.fill_diagonal_(False)
        u, v = G.edges()
        A = torch.zeros_like(sub)
        A[u.long(), v.long()] = True
        if not torch.equal(A, sub) or not torch.equal(A, A.t()):
            mism += 1
        # LCC must be connected & contain every raw edge among retained nodes
    assert mism == 0, f"{mism} graphs differ from raw samples"

    total_att = sum(p["attempted_in_used_batches"] for p in per_stream)
    total_rej = sum(p["rejected_in_used_batches"] for p in per_stream)
    und = np.array(dedges) // 2
    info = {
        "dataset": a.dataset, "method": "defog", "training_seed": a.train_seed,
        "N_accepted": a.n, "N_attempted_in_used_batches": total_att,
        "N_rejected_in_used_batches": total_rej,
        "graphs_with_nodes_removed_by_largest_component": lcc_dropped,
        "checkpoint": {"path_used": os.path.abspath(a.ckpt), "sha256": sha(a.ckpt),
                       "original_path": a.orig_ckpt_path, "selection": "best_validation (val_loss)"},
        "generation_seeds": [p["generation_seed"] for p in per_stream],
        "streams": per_stream,
        "sampling_config": {
            "entrypoint": "DeFoG c631697 src/main.py (test_only) via scripts_defog/run_defog_sampling.py",
            "overrides": json.load(open(os.path.join(a.streams[0], "synfix_invocation.json")))["overrides"],
            "sample_steps": 1000, "time_distortion": "polydec",
            "eta_omega": "GRID/TRI eta=50 omega=0.05 (experiment=grid); LOBSTER eta=0 omega=0 (experiment=tree)",
            "node_count_distribution": "unconditional, empirical train+validation node counts (FrozenGraphVAEInfos.node_counts)",
            "sampling_batch": "2*train.batch_size (unchanged from original)",
            "acceptance": "strict_generation: reject if ggm_eval normalize_pyg_graph raises; export keeps deterministic largest connected component",
        },
        "pyg_collection": {"path": pt_path, "collection_sha256": man["collection_sha256"], "sha256": sha(pt_path)},
        "output": {"path": bin_path, "sha256": sha(bin_path), "format": "dgl.save_graphs",
                   "ndata_attr": f"zeros float32 [n,{Dn}] (width of dataset_cache node_onehot_info; topology-only)",
                   "edata_attr": (f"zeros float32 [E,{De}] (width of edge_onehot_info)" if De else None),
                   "labels": "has_native_features=0, graph_index=0..N-1",
                   "directed_edges_both_directions": True, "self_loops": False},
        "summary": {"total_nodes": int(sum(nodes)), "total_directed_edges": int(sum(dedges)),
                    "nodes_mean": float(np.mean(nodes)), "nodes_min": int(min(nodes)), "nodes_max": int(max(nodes)),
                    "undirected_edges_mean": float(und.mean()), "undirected_edges_min": int(und.min()),
                    "undirected_edges_max": int(und.max())},
        "verification": {"edges_match_raw_samples_on_lcc": True, "graphs_checked": a.n},
        "dataset_cache": {"path": cache_path, "sha256": sha(cache_path)},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    json.dump(info, open(os.path.join(a.out_dir, "generated_500.json"), "w"), indent=1)
    print(json.dumps({k: info[k] for k in ["dataset", "training_seed", "N_accepted", "N_attempted_in_used_batches",
                                          "N_rejected_in_used_batches", "summary"]}))


if __name__ == "__main__":
    main()
