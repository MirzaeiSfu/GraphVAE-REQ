#!/usr/bin/env python3
"""Evaluate generated and held-out DGL graphs with attributed Random-GIN.

Both files must be written with ``dgl.save_graphs``. Every graph must store its
final float node features in ``ndata["attr"]`` and, when applicable, its final
float edge features in ``edata["attr"]``. PyG objects are not accepted.
"""

# Caller-side PyG conversion reference (not part of this evaluator):
#
#   def pyg_to_dgl(data):
#       graph = dgl.graph(
#           (data.edge_index[0], data.edge_index[1]),
#           num_nodes=data.num_nodes,
#       )
#       graph.ndata["attr"] = data.x.float()
#       if data.edge_attr is not None:
#           graph.edata["attr"] = data.edge_attr.float()
#       return graph
#
# One-hot encode categorical IDs before saving. Then call:
#   dgl.save_graphs("defog_generated.bin", generated_dgl_graphs)

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.attributed_gin import (  # noqa: E402
    FEATURE_MODES,
    evaluate_dgl_feature_modes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dgl",
        type=Path,
        required=True,
        help="DGL file containing generated graphs.",
    )
    parser.add_argument(
        "--reference-dgl",
        type=Path,
        required=True,
        help="DGL file containing the fixed held-out reference graphs.",
    )
    parser.add_argument(
        "--model-name",
        default="external_model",
        help="Label recorded in reports, for example DeFoG, GRAN, or GraphRNN.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=FEATURE_MODES,
        default=None,
        help="Feature ablations. Default: every mode supported by the inputs.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=1000,
        help="Maximum graphs per collection; 0 uses the full reference file.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Matched Random-GIN initializations. Default: 10.",
    )
    parser.add_argument(
        "--evaluator-seed",
        type=int,
        default=0,
        help="Base Random-GIN seed. Default: 0.",
    )
    parser.add_argument(
        "--nearest-k",
        type=int,
        default=5,
        help="Neighbourhood size for precision/recall. Default: 5.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, or cuda:N. Default: auto.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("attributed_dgl_eval"),
        help="Report directory. Default: ./attributed_dgl_eval.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device_arg!r} requested but CUDA is unavailable."
        )
    return torch.device(device_arg)


def load_dgl_graphs(path: Path) -> list:
    try:
        import dgl
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("DGL is required for attributed evaluation.") from exc

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DGL graph file not found: {path}")
    try:
        graphs, _ = dgl.load_graphs(str(path))
    except Exception as exc:
        raise ValueError(
            f"Could not read {path} as a DGL graph file. Inputs must be saved "
            "with dgl.save_graphs; PyG files and objects are not accepted."
        ) from exc
    if not graphs:
        raise ValueError(f"DGL graph file is empty: {path}")
    return list(graphs)


def edge_feature_dim(graph) -> int:
    tensor = graph.edata.get("attr")
    if tensor is None:
        return 0
    if tensor.ndim != 2:
        raise ValueError(
            f"edata['attr'] must be rank 2, got shape {tuple(tensor.shape)}."
        )
    return int(tensor.shape[1])


def choose_modes(requested_modes, graph) -> list[str]:
    edge_dim = edge_feature_dim(graph)
    if requested_modes is None:
        return (
            list(FEATURE_MODES)
            if edge_dim
            else ["topology_control", "decoded_node"]
        )
    modes = list(dict.fromkeys(requested_modes))
    if not edge_dim and any(
        mode in {"decoded_edge", "decoded_node_edge"} for mode in modes
    ):
        raise ValueError(
            "An edge-feature mode was requested, but edata['attr'] is absent."
        )
    return modes


def write_csv(path: Path, payload: dict):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "mode", "metric", "mean", "std", "min", "max"),
        )
        writer.writeheader()
        for mode, result in payload["evaluation"]["modes"].items():
            for metric, summary in result["summary"].items():
                writer.writerow(
                    {
                        "model": payload["model_name"],
                        "mode": mode,
                        "metric": metric,
                        **summary,
                    }
                )


def write_markdown(path: Path, payload: dict):
    primary = payload["attributed_f1_pr"]
    lines = [
        f"# {payload['model_name']} Attributed Random-GIN Evaluation",
        "",
        "Both graph collections entered through the strict DGL feature API.",
        "",
        f"- Generated DGL: `{payload['generated_dgl']}`",
        f"- Reference DGL: `{payload['reference_dgl']}`",
        f"- Graphs per collection: `{payload['graph_count']}`",
        f"- Primary mode: `{payload['primary_mode']}`",
        f"- Primary F1-PR: `{primary['mean']:.6f} ± {primary['std']:.6f}`",
        "",
        "| Mode | F1-PR | Precision | Recall | MMD-RBF | MMD-linear |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, result in payload["evaluation"]["modes"].items():
        summary = result["summary"]
        values = [
            f"{summary[metric]['mean']:.6f} ± {summary[metric]['std']:.6f}"
            for metric in (
                "f1_pr",
                "precision",
                "recall",
                "mmd_rbf",
                "mmd_linear",
            )
        ]
        lines.append(f"| {mode} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.max_graphs < 0 or args.max_graphs in {1, 2}:
        raise ValueError("--max-graphs must be 0 (all) or at least 3.")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    if args.nearest_k < 1:
        raise ValueError("--nearest-k must be positive.")

    generated_path = args.generated_dgl.expanduser().resolve()
    reference_path = args.reference_dgl.expanduser().resolve()
    generated = load_dgl_graphs(generated_path)
    reference = load_dgl_graphs(reference_path)
    if args.max_graphs:
        reference = reference[: args.max_graphs]
    if len(reference) < 3:
        raise ValueError("At least three reference graphs are required.")
    if len(generated) < len(reference):
        raise ValueError(
            f"Generated file has {len(generated)} graphs but the selected "
            f"reference set has {len(reference)}. Export more generated graphs."
        )
    generated = generated[: len(reference)]

    modes = choose_modes(args.modes, reference[0])
    device = resolve_device(args.device)
    evaluation = evaluate_dgl_feature_modes(
        generated,
        reference,
        modes=modes,
        repeats=args.repeats,
        seed=args.evaluator_seed,
        nearest_k=args.nearest_k,
        device=device,
    )
    primary_mode = (
        "decoded_node_edge"
        if "decoded_node_edge" in modes
        else "decoded_node"
        if "decoded_node" in modes
        else modes[0]
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "attributed-dgl-random-gin-v1",
        "model_name": args.model_name,
        "generated_dgl": str(generated_path),
        "reference_dgl": str(reference_path),
        "device": str(device),
        "graph_count": len(reference),
        "input_contract": {
            "accepted": "individual homogeneous DGLGraph objects",
            "node_features": "ndata['attr'] float matrix",
            "edge_features": "edata['attr'] float matrix or absent",
            "pyg_objects": False,
            "plain_tensor_dictionaries": False,
            "hand_made_topology_features": False,
        },
        "normalization": {
            "undirected": True,
            "input_self_loops_ignored": True,
            "largest_connected_component": True,
            "conflicting_reverse_edge_attributes": "error",
        },
        "implementation": {
            "feature_extractor": "third_party/ggmeval Random-GIN",
            "precision_recall": "third_party/ggmeval prdcEvaluation",
        },
        "primary_mode": primary_mode,
        "attributed_f1_pr": evaluation["modes"][primary_mode]["summary"]["f1_pr"],
        "evaluation": evaluation,
    }
    (output_dir / "attributed_dgl_random_gin.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "attributed_dgl_random_gin_summary.csv", payload)
    write_markdown(output_dir / "attributed_dgl_random_gin_report.md", payload)

    print(
        "[AttributedDGL] "
        f"{args.model_name}: {primary_mode} F1-PR="
        f"{payload['attributed_f1_pr']['mean']:.6f} ± "
        f"{payload['attributed_f1_pr']['std']:.6f}"
    )
    print(f"[AttributedDGL] Wrote {output_dir}")


if __name__ == "__main__":
    main()
