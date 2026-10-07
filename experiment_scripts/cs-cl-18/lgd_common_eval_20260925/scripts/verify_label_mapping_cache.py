#!/usr/bin/env python3
"""Check LGD categorical class -> one-hot column mapping against LGD's source dataset cache
(GraphVAE-REQ dataset-cache pickle recorded in repo/datasets/graphvae_req/<DS>/metadata.json).
Used when the common reference is not LGD's test split (PROTEINS).  Also records the cache's
node_onehot_info and that of the cache the common reference came from.

Usage: verify_label_mapping_cache.py DATASET [REFERENCE_SOURCE_CACHE]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

GV = "/local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ"
sys.path.insert(0, GV)  # the cache pickles reference GraphVAE-REQ's `data` module
BASE = Path(__import__("os").environ.get("LGD_EVAL_BASE", "/local-scratch2/mirzaei/lgd_common_eval_20260925"))
LGD = Path("/local-scratch2/mirzaei/LGD_FIX_20260924/repo/datasets/graphvae_req")

dataset = sys.argv[1]
meta = json.loads((LGD / dataset / "metadata.json").read_text())
cache = pickle.load(open(meta["source"], "rb"))
fields = meta["node_feature_dims"]
out = {"dataset": dataset, "lgd_source_cache": meta["source"],
       "lgd_source_node_onehot_info": {str(k): v for k, v in (cache.get("node_onehot_info") or {}).items()}}
conf = np.zeros((sum(fields), sum(fields)), dtype=np.int64)
for split, key in (("train", "list_noh_train"), ("test", "list_noh_test")):
    data = torch.load(LGD / dataset / f"{split}.pt")
    for i, g in enumerate(data):
        n = int(g.num_nodes)
        X = np.asarray(cache[key][i])[:n]
        offset = 0
        for f, w in enumerate(fields):
            for a, b in zip(g.x[:, f].tolist(), X[:, offset:offset + w].argmax(1).tolist()):
                conf[offset + a, offset + b] += 1
            offset += w
out["confusion_lgd_class_vs_cache_column"] = conf.tolist()
out["identity"] = bool(conf.sum() == np.trace(conf))
if len(sys.argv) > 2:
    other = pickle.load(open(sys.argv[2], "rb"))
    out["reference_source_cache"] = sys.argv[2]
    out["reference_source_node_onehot_info"] = {str(k): v for k, v in (other.get("node_onehot_info") or {}).items()}
    out["same_column_semantics"] = out["reference_source_node_onehot_info"] == out["lgd_source_node_onehot_info"]
(BASE / dataset / "label_mapping_source_cache.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items()}, indent=1))
