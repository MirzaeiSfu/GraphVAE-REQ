#!/usr/bin/env python3
"""Verify that LGD's pickled reference equals the common reference, and that LGD's
categorical node classes map to the same one-hot columns as the common reference.

Usage: verify_reference.py DATASET SEED [SEED ...]
Writes <BASE>/<DATASET>/reference_verification.json.  Exit code 0 always; the
verdict fields (reference_equal, label_mapping.identity) are what matters and are
also printed.  A non-identity label mapping aborts (exit 3) because staging assumes it.
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lgd_common import (BASE, LGD_REPO, NODE_FIELDS, compare_collections, largest_component_ids,
                        lcc_relabel, load_lgd_pickle, load_reference, sha256_file, write_json)


def ref_node_matrix(dataset, graph):
    return graph.ndata["attr"].numpy() if hasattr(graph, "ndata") else graph.x.numpy()


def main():
    dataset = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]]
    fields = NODE_FIELDS[dataset]
    ref_nx, ref_raw, ref_path = load_reference(dataset)
    n_ref = len(ref_nx)
    report = {"dataset": dataset, "common_reference": ref_path,
              "common_reference_sha256": sha256_file(ref_path),
              "common_reference_count": n_ref, "seeds": {}}
    verdicts = []
    raw_first = None
    for seed in seeds:
        pkl = sorted((BASE / "raw_pickles" / f"{dataset}_seed{seed}").glob("epoch_*_graphs.pkl"))[-1]
        data = load_lgd_pickle(pkl)
        if raw_first is None:
            raw_first = data["reference"]
        lgd_ref = [lcc_relabel(g) for g in data["reference"]]
        empty = [i for i, g in enumerate(lgd_ref) if g.number_of_edges() == 0]
        kept = [g for g in lgd_ref if g.number_of_edges() > 0][:n_ref]
        cmp = compare_collections(kept, ref_nx)
        equal = cmp["exact_index_aligned"] == n_ref == len(kept)
        verdicts.append(equal)
        report["seeds"][seed] = {
            "pickle": str(pkl), "pickle_sha256": sha256_file(pkl),
            "lgd_reference_count": len(lgd_ref), "lgd_generated_count": len(data["generated"]),
            "lgd_reference_edgeless_indices": empty[:50], "n_edgeless": len(empty),
            "comparison_first_n_nonedgeless_vs_common": cmp, "reference_equal": equal,
        }
    report["reference_equal_all_seeds"] = all(verdicts)
    # LGD's own frozen test split (source of LGD's categorical classes).
    test = torch.load(LGD_REPO / "datasets/graphvae_req" / dataset / "test.pt")
    limit = max(4 * n_ref, 1000)  # enough to cover the first n_ref non-edgeless graphs
    test_nx, test_labels = [], []
    for d in test[:limit]:
        g = nx.Graph()
        g.add_nodes_from(range(int(d.num_nodes)))
        e = d.edge_index.numpy()
        g.add_edges_from((int(a), int(b)) for a, b in zip(e[0], e[1]) if a != b)
        test_nx.append(g)
        test_labels.append(d.x.long().numpy())
    head = [nx.convert_node_labels_to_integers(g, ordering="sorted") for g in raw_first[:limit]]
    report["lgd_test_pt_vs_pickle_reference"] = compare_collections(head, test_nx[:len(head)])
    if sum(fields) > 1:
        keep_idx = [i for i, g in enumerate(test_nx) if largest_component_ids(g) is not None][:n_ref]
        confusions = [np.zeros((w + 1, w), dtype=np.int64) for w in fields]
        size_mismatch = 0
        for ref_i, test_i in enumerate(keep_idx):
            keep = largest_component_ids(test_nx[test_i])
            ref_x = ref_node_matrix(dataset, ref_raw[ref_i])
            if len(keep) != ref_x.shape[0]:
                size_mismatch += 1
                continue
            offset = 0
            for f, width in enumerate(fields):
                block = ref_x[:, offset:offset + width]
                for local, node in enumerate(keep):
                    confusions[f][min(int(test_labels[test_i][node, f]), width), int(block[local].argmax())] += 1
                offset += width
        off = [int(c.sum() - np.trace(c[:c.shape[1]])) for c in confusions]
        report["label_mapping"] = {
            "rows": "LGD categorical class (test.pt x[:, field])",
            "cols": "argmax column within the field's block of the common reference one-hot",
            "fields": fields, "confusion": [c.tolist() for c in confusions],
            "size_mismatched_graphs": size_mismatch, "off_diagonal_total": off,
            "diagonal_total": [int(np.trace(c[:c.shape[1]])) for c in confusions],
            "identity": all(o == 0 for o in off) and sum(int(np.trace(c[:c.shape[1]])) for c in confusions) > 0,
        }
    out = BASE / dataset / "reference_verification.json"
    write_json(out, report)
    for seed, item in report["seeds"].items():
        c = item["comparison_first_n_nonedgeless_vs_common"]
        print(dataset, "seed", seed, "lgd_ref", item["lgd_reference_count"], "edgeless", item["n_edgeless"],
              "exact", c["exact_index_aligned"], "iso", c["isomorphic_index_aligned"], "/", n_ref,
              "wl_multiset_equal", c["wl_multiset_equal"], "REFERENCE_EQUAL" if item["reference_equal"] else "REFERENCE_DIFFERS")
    t = report["lgd_test_pt_vs_pickle_reference"]
    print("test.pt vs pickle reference: exact", t["exact_index_aligned"], "/", t["count_a"])
    if "label_mapping" in report:
        lm = report["label_mapping"]
        print("label mapping diag", lm["diagonal_total"], "offdiag", lm["off_diagonal_total"],
              "size mismatched", lm["size_mismatched_graphs"], "IDENTITY" if lm["identity"] else "NOT_IDENTITY")
        if not report["reference_equal_all_seeds"]:
            print("NOTE: reference differs from LGD's test split, so this index-aligned label check is not "
                  "meaningful; verify the mapping against LGD's source cache (verify_label_mapping_cache.py).")
        elif not lm["identity"]:
            print("ABORT: LGD class -> one-hot column mapping is not the identity", file=sys.stderr)
            sys.exit(3)
    print(out)


if __name__ == "__main__":
    main()
