#!/usr/bin/env bash
# Supplementary (LGD only, no comparators): PROTEINS LGD scored against LGD's OWN held-out
# test split (= motif=True split, dataset_cache.pkl), 209 non-edgeless graphs, same evaluator.
set -euo pipefail
gpu=${1:-0}
B=/local-scratch2/mirzaei/lgd_common_eval_20260925
PY=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
root=$B/PROTEINS/supplementary_lgd_own_test_split
mkdir -p "$root/results"
exec > >(tee -a "$root/pipeline.log") 2>&1
"$PY" - <<'PYEOF'
import sys; sys.path.insert(0, "/local-scratch2/mirzaei/lgd_common_eval_20260925/scripts")
from lgd_common import *
import dgl, torch, numpy as np, networkx as nx
test = torch.load(LGD_REPO / "datasets/graphvae_req/PROTEINS/test.pt")
graphs = []
for d in test:
    g = nx.Graph(); g.add_nodes_from(range(int(d.num_nodes)))
    e = d.edge_index.numpy(); g.add_edges_from((int(a), int(b)) for a, b in zip(e[0], e[1]) if a != b)
    keep = largest_component_ids(g)
    if keep is None:
        continue  # the single edgeless test graph (index 3), as in the common reference policy
    local = {n: i for i, n in enumerate(keep)}
    ed = np.asarray(sorted((local[u], local[v]) for u, v in g.edges() if u in local and v in local))
    x = onehot(d.x[keep, 0].tolist(), 3)
    dg = dgl.graph((torch.as_tensor(np.r_[ed[:, 0], ed[:, 1]]), torch.as_tensor(np.r_[ed[:, 1], ed[:, 0]])), num_nodes=len(keep))
    dg.ndata["attr"] = torch.as_tensor(x)
    graphs.append(dg)
out = BASE / "PROTEINS/supplementary_lgd_own_test_split/reference_lgd_test.bin"
dgl.save_graphs(str(out), graphs); print(out, len(graphs))
PYEOF
for seed in 0 1 2; do
  mkdir -p "$root/results/lgd/seed_$seed"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" /local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py \
    --repo /local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ \
    --generated "$B/PROTEINS/stage/lgd/seed_$seed/generated.bin" --reference "$root/reference_lgd_test.bin" \
    --output "$root/results/lgd/seed_$seed/common_metrics.json" --label proteins_lgd_own_split --seed "$seed" --device cuda --repeats 10
done
echo COMPLETE > "$root/status"
