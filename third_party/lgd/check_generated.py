#!/usr/bin/env python3
"""Sanity gate for sampled LGD graphs: empty / collapsed / over-dense output.

Reads the pickle written by sample_tu.py ({"reference": [nx], "generated": [nx]})
and writes GATE.json with edge statistics. Status:
  PASS  - <=10% edgeless graphs and mean edges within [0.5x, 2x] of reference
  WARN  - otherwise, but some structure is generated
  FAIL  - >=50% edgeless graphs or mean edges > 5x reference
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np


def edges(graph):
    return sum(1 for u, v in graph.edges() if u != v)


def main() -> None:
    path = Path(sys.argv[1])
    data = pickle.loads(path.read_bytes())
    ref = np.array([edges(g) for g in data["reference"]], dtype=float)
    gen = np.array([edges(g) for g in data["generated"]], dtype=float)
    gen_nodes = np.array([g.number_of_nodes() for g in data["generated"]], dtype=float)
    ref_nodes = np.array([g.number_of_nodes() for g in data["reference"]], dtype=float)
    density = np.array([2 * e / max(n * (n - 1), 1) for e, n in zip(gen, gen_nodes)])
    ref_density = np.array([2 * e / max(n * (n - 1), 1) for e, n in zip(ref, ref_nodes)])
    empty = float((gen == 0).mean()) if len(gen) else 1.0
    ratio = float(gen.mean() / max(ref.mean(), 1e-9)) if len(gen) else 0.0
    if empty >= 0.5 or ratio > 5:
        status = "FAIL"
    elif empty <= 0.1 and 0.5 <= ratio <= 2:
        status = "PASS"
    else:
        status = "WARN"
    report = {
        "status": status, "source": str(path), "generated": len(gen), "reference": len(ref),
        "edgeless_fraction": empty, "mean_edges_generated": float(gen.mean()) if len(gen) else 0.0,
        "mean_edges_reference": float(ref.mean()), "edge_ratio": ratio,
        "mean_density_generated": float(density.mean()) if len(gen) else 0.0,
        "mean_density_reference": float(ref_density.mean()),
        "mean_nodes_generated": float(gen_nodes.mean()) if len(gen) else 0.0,
        "mean_nodes_reference": float(ref_nodes.mean()),
    }
    out = path.parent / "GATE.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
