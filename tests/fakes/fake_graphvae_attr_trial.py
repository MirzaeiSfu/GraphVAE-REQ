#!/usr/bin/env python3
"""Deterministic subprocess fake for GraphVAE Attr-F1PR failure injection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import tempfile
import time
from pathlib import Path


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def payload(args):
    result = {
        "schema_version": "attributed-random-gin-v1",
        "split": "test" if args.mode == "wrong-split" else "validation",
        "primary_mode": (
            "topology_control" if args.mode == "topology-only" else "decoded_node_edge"
        ),
        "generation_seed": args.generation_seed,
        "evaluator_seed": args.evaluator_seed,
        "graph_counts": {
            "accepted_per_collection": args.graph_count,
            "generated_accepted": args.graph_count,
            "reference_accepted": args.graph_count,
        },
        "feature_source": {
            "generated": "GraphVAE node_feature_decoder and edge_feature_decoder",
            "reference": "cached dataset node and edge one-hot attributes",
            "hand_made_topology_features": False,
        },
        "integrity": {
            "cache_sha256": args.cache_sha256,
            "split_fingerprint": args.split_fingerprint,
            "node_schema_fingerprint": args.node_schema_fingerprint,
            "edge_schema_fingerprint": args.edge_schema_fingerprint,
        },
        "evaluation": {
            "feature_dimensions": {
                "node": 0 if args.mode == "missing-node" else 4,
                "edge": 0 if args.mode == "missing-edge" else 3,
            },
            "actual_decoder_output_dimensions": {"node": 4, "edge": 3},
            "repeats": args.repeats,
            "modes": {
                "decoded_node_edge": {
                    "summary": {
                        "f1_pr": {
                            "mean": float("nan") if args.mode == "non-finite" else args.score
                        },
                        "precision": {"mean": min(1.0, args.score + 0.01)},
                        "recall": {"mean": max(0.0, args.score - 0.01)},
                    }
                }
            },
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "success", "training-failure", "evaluation-failure", "timeout",
            "malformed-json", "non-finite", "wrong-split", "topology-only",
            "missing-node", "missing-edge", "post-write-corruption",
        ),
        default="success",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--score", type=float, default=0.75)
    parser.add_argument("--graph-count", type=int, default=8)
    parser.add_argument("--generation-seed", type=int, default=123)
    parser.add_argument("--evaluator-seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache-sha256", default="cache")
    parser.add_argument("--split-fingerprint", default="split")
    parser.add_argument("--node-schema-fingerprint", default="node")
    parser.add_argument("--edge-schema-fingerprint", default="edge")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "training-failure":
        print("synthetic training failure", file=sys.stderr)
        return 21
    if args.mode == "timeout":
        time.sleep(args.sleep if args.sleep > 0 else 3600)
    if args.sleep:
        time.sleep(args.sleep)
    checkpoint = args.output_dir / "checkpoint"
    checkpoint.write_bytes(b"node_feature_decoder\0edge_feature_decoder\0fake")
    if args.mode == "evaluation-failure":
        print("synthetic evaluation failure", file=sys.stderr)
        return 22
    output = args.output_dir / "attributed_random_gin.json"
    if args.mode == "malformed-json":
        output.write_text("{malformed", encoding="utf-8")
        return 0
    atomic_json(output, payload(args))
    if args.mode == "post-write-corruption":
        output.write_bytes(output.read_bytes() + b"corruption")
    atomic_json(
        args.output_dir / "fake_result.json",
        {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "evaluator_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
