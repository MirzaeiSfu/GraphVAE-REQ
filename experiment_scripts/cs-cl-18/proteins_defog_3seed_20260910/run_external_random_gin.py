#!/usr/bin/env python3
"""Evaluate a new DeFoG PROTEINS collection with the pinned external Random-GIN."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path("/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ")
sys.path.insert(0, str(REPO / "graph_evaluation" / "src"))
from ggm_eval.runner import evaluate_legacy_random_gin

p = argparse.ArgumentParser()
p.add_argument("--generated", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", default="cpu")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
result = evaluate_legacy_random_gin(
    generated=a.generated,
    reference=a.reference,
    legacy_repository=REPO,
    output_dir=a.output,
    python_executable=sys.executable,
    modes=["decoded_node", "topology_control"],
    repeats=10,
    evaluator_seed=0,
    nearest_k=5,
    max_graphs=0,
    device=a.device,
    trusted_input=True,
)
(a.output / "evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True))
print(json.dumps(result, indent=2, sort_keys=True))
