#!/usr/bin/env python3
"""Resample saved Grid GraphVAE checkpoints after training.

This script loads already-trained checkpoint state_dict files, generates
multiple graph sets from each checkpoint, and reports Table 2 MMD statistics
plus dense-edge outlier rates. It is intentionally post-training only, so it
does not add cost to the long training run.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.chdir(REPO_ROOT)

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime environment guard
    raise SystemExit("PyYAML is required. Install it or use the micro env.") from exc

from GlobalProperties import kernel  # noqa: E402
from model import AveEncoder, GraphTransformerDecoder_FC, kernelGVAE  # noqa: E402
from reproduce_table2_grid import (  # noqa: E402
    PAPER_TABLE2_BY_DATASET,
    compute_table2_metrics,
    locked_orca_tmp,
    to_graphs,
)
from util import EdgeFeatureDecoder, NodeFeatureDecoder  # noqa: E402


TABLE2_METRICS = ("degree", "clustering", "orbit", "spectral", "diameter")
DENSE_DEFINITIONS = ("twice_mean", "mean_plus_3std", "max_reference")


def flatten_config(config_data: dict) -> dict:
    flat = {}
    for key, value in config_data.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if nested_key in flat:
                    raise ValueError(f"Duplicate config key: {nested_key}")
                flat[nested_key] = nested_value
        else:
            if key in flat:
                raise ValueError(f"Duplicate config key: {key}")
            flat[key] = value
    return flat


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return flatten_config(yaml.safe_load(handle) or {})


def normalize_model_name(model_name: str) -> str:
    aliases = {
        "graphvae": "kipf",
        "graphvae-mm": "GraphVAE-MM",
        "kernelaugmentedwithtotalnumberoftriangles": "GraphVAE-MM",
    }
    normalized = str(model_name).strip()
    return aliases.get(normalized.lower(), normalized)


def dataset_cache_path(config: dict) -> Path:
    cache_root = Path(config.get("dataset_cache_dir") or "cache_datasets").expanduser()
    dataset = config["dataset"]
    split_mode = config.get("split_mode", "legacy_80_20")
    bfs_strategy = config.get("bfs_strategy", "all_components")
    cache_name = f"{dataset}.pkl"
    if split_mode != "legacy_80_20" or bfs_strategy != "all_components":
        cache_name = f"{dataset}_{split_mode}_{bfs_strategy}.pkl"
    return cache_root / cache_name


def load_cached_dataset(config: dict) -> dict:
    cache_path = dataset_cache_path(config)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Dataset cache not found: {cache_path}. Run training once to create it."
        )
    with cache_path.open("rb") as handle:
        return pickle.load(handle)


def build_model(config: dict, cache: dict, device: torch.device) -> kernelGVAE:
    dataset = config["dataset"]
    model_name = normalize_model_name(config.get("model", "GraphVAE"))
    graph_em_dim = int(config.get("graphEmDim", 1024))
    decoder_type = config.get("decoder", "FC")
    encoder_type = config.get("encoder_type", "AvePool")
    directed = bool(config.get("directed", True))
    beta = config.get("beta")
    tiny_overfit = bool(config.get("tiny_overfit", False))

    if model_name in {"KernelAugmentedWithTotalNumberOfTriangles", "GraphVAE-MM"}:
        kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist", "TotalNumberOfTriangles"]
        step_num = 5
        if dataset == "QM9":
            step_num = 2
    elif model_name in {"kipf", "graphVAE"}:
        kernl_type = []
        step_num = 0
    else:
        raise ValueError(f"Unsupported model for resampling: {model_name}")

    autoencoder = bool(tiny_overfit)
    if beta is not None:
        # The model architecture is unchanged; beta only affects training loss.
        pass

    list_graphs = cache["list_graphs"]
    if not list_graphs.node_onehot_s or not list_graphs.edge_onehot_s:
        list_graphs.processALL(self_for_none=cache.get("self_for_none", True))
    subgraph_node_num = list_graphs.max_num_nodes
    in_feature_dim = list_graphs.feature_size
    node_num = list_graphs.max_num_nodes

    degree_center = torch.tensor([[x] for x in range(0, subgraph_node_num, 1)])
    degree_width = torch.tensor([[0.1] for _ in range(0, subgraph_node_num, 1)])
    bin_center = torch.tensor([[x] for x in range(0, subgraph_node_num, 1)])
    bin_width = torch.tensor([[1] for _ in range(0, subgraph_node_num, 1)])

    kernel_model = kernel(
        device=device,
        kernel_type=kernl_type,
        step_num=step_num,
        bin_width=bin_width,
        bin_center=bin_center,
        degree_bin_center=degree_center,
        degree_bin_width=degree_width,
    )

    if encoder_type != "AvePool":
        raise ValueError(f"Unsupported encoder for resampling: {encoder_type}")
    encoder = AveEncoder(in_feature_dim, [256], graph_em_dim)

    if decoder_type != "FC":
        raise ValueError(f"Unsupported decoder for resampling: {decoder_type}")
    decoder = GraphTransformerDecoder_FC(graph_em_dim, 256, node_num, directed)

    alpha_node_feat = float(config.get("alpha_node_feat", 0.0))
    alpha_edge_feat = float(config.get("alpha_edge_feat", 0.0))
    use_motif_loss = bool(config.get("motif_loss", False))

    has_node_targets = bool(list_graphs.node_onehot_s) and list_graphs.node_onehot_s[0] is not None
    has_edge_targets = bool(list_graphs.edge_onehot_s) and list_graphs.edge_onehot_s[0] is not None
    node_feature_decoder = None
    edge_feature_decoder = None
    if (alpha_node_feat > 0 or use_motif_loss) and has_node_targets:
        node_feature_decoder = NodeFeatureDecoder(
            graph_em_dim,
            list_graphs.max_num_nodes,
            list_graphs.node_onehot_s[0].shape[-1],
        )
    if (alpha_edge_feat > 0 or use_motif_loss) and has_edge_targets:
        edge_feature_decoder = EdgeFeatureDecoder(
            graph_em_dim,
            list_graphs.max_num_nodes,
            list_graphs.edge_onehot_s[0].shape[0],
        )

    model = kernelGVAE(
        kernel_model,
        encoder,
        decoder,
        autoencoder,
        graphEmDim=graph_em_dim,
        node_feature_decoder=node_feature_decoder,
        edge_feature_decoder=edge_feature_decoder,
    )
    model.to(device)
    model.eval()
    return model


def largest_component(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph
    return nx.Graph(graph.subgraph(max(nx.connected_components(graph), key=len)))


def generate_graphs(model: kernelGVAE, count: int, device: torch.device) -> dict[str, list[nx.Graph]]:
    raw_graphs = []
    largest_component_graphs = []
    with torch.no_grad():
        for _ in range(count):
            z = torch.randn(1, model.embeding_dim, device=device)
            adj_logit = model.decode(z.float())
            sample_graph = torch.sigmoid(adj_logit)[0].detach().cpu().numpy()
            sample_graph = (sample_graph >= 0.5).astype(np.int8)
            graph = nx.from_numpy_array(sample_graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            graph.remove_nodes_from(list(nx.isolates(graph)))
            if graph.number_of_nodes() > 0:
                raw_graph = nx.Graph(graph)
                raw_graphs.append(raw_graph)
                largest_component_graphs.append(largest_component(raw_graph))
    return {
        "raw": raw_graphs,
        "largest_component": largest_component_graphs,
    }


def compute_dense_edge_threshold(reference_graphs: list[nx.Graph], dense_definition: str) -> float:
    edge_counts = np.array([float(graph.number_of_edges()) for graph in reference_graphs])
    if edge_counts.size == 0:
        return 0.0
    if dense_definition == "twice_mean":
        return float(2.0 * np.mean(edge_counts))
    if dense_definition == "mean_plus_3std":
        return float(np.mean(edge_counts) + 3.0 * np.std(edge_counts))
    if dense_definition == "max_reference":
        return float(np.max(edge_counts))
    raise ValueError(f"Unknown dense definition: {dense_definition}")


def edge_stats(graphs: list[nx.Graph], threshold: float) -> dict:
    edge_counts = [float(graph.number_of_edges()) for graph in graphs]
    node_counts = [int(graph.number_of_nodes()) for graph in graphs]
    dense_flags = [edge_count > threshold for edge_count in edge_counts]
    if not edge_counts:
        return {
            "count": 0,
            "edge_counts": [],
            "node_counts": [],
            "dense_count": 0,
            "dense_rate": None,
            "max_edges": 0.0,
            "mean_edges": 0.0,
            "median_edges": 0.0,
        }
    return {
        "count": len(edge_counts),
        "edge_counts": edge_counts,
        "node_counts": node_counts,
        "dense_count": int(sum(dense_flags)),
        "dense_rate": float(sum(dense_flags) / len(edge_counts)),
        "max_edges": float(max(edge_counts)),
        "mean_edges": float(np.mean(edge_counts)),
        "median_edges": float(np.median(edge_counts)),
    }


def normalized_table2_score(metrics: dict[str, float], dataset: str) -> float:
    paper = PAPER_TABLE2_BY_DATASET[dataset]["GraphVAE"]
    return float(np.mean([metrics[metric] / paper[metric] for metric in TABLE2_METRICS]))


def summarize_values(values: list[float]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_edge_stat_samples(samples: list[dict], field_name: str) -> dict:
    total_graphs = sum(sample[field_name]["count"] for sample in samples)
    total_dense = sum(sample[field_name]["dense_count"] for sample in samples)
    samples_with_dense = sum(sample[field_name]["dense_count"] > 0 for sample in samples)
    mean_edge_counts = [sample[field_name]["mean_edges"] for sample in samples]
    max_edge_counts = [sample[field_name]["max_edges"] for sample in samples]
    return {
        "mean_edge_count": summarize_values(mean_edge_counts),
        "max_edge_count": summarize_values(max_edge_counts),
        "dense_graph_rate": float(total_dense / total_graphs) if total_graphs else None,
        "samples_with_dense_rate": float(samples_with_dense / len(samples)) if samples else None,
    }


def summarize_samples(samples: list[dict]) -> dict:
    scores = [sample["score"] for sample in samples]
    metric_summary = {
        metric: summarize_values([sample["metrics"][metric] for sample in samples])
        for metric in TABLE2_METRICS
    }
    largest_component_edge_summary = summarize_edge_stat_samples(samples, "edge_stats")
    raw_edge_summary = summarize_edge_stat_samples(samples, "raw_edge_stats")
    return {
        "score": summarize_values(scores),
        "metrics": metric_summary,
        "largest_component_edge_summary": largest_component_edge_summary,
        "raw_edge_summary": raw_edge_summary,
        "mean_edge_count": largest_component_edge_summary["mean_edge_count"],
        "max_edge_count": largest_component_edge_summary["max_edge_count"],
        "dense_graph_rate": largest_component_edge_summary["dense_graph_rate"],
        "samples_with_dense_rate": largest_component_edge_summary["samples_with_dense_rate"],
        "worst_sample_index": int(max(range(len(samples)), key=lambda idx: samples[idx]["score"])),
        "best_sample_index": int(min(range(len(samples)), key=lambda idx: samples[idx]["score"])),
    }


def write_markdown_report(output_dir: Path, payload: dict):
    lines = [
        "# Grid Checkpoint Resampling",
        "",
        "Lower is better for all MMD metrics and normalized scores.",
        "",
        f"- config: `{payload['config']}`",
        f"- run_dir: `{payload['run_dir']}`",
        f"- samples_per_checkpoint_split: `{payload['samples_per_split']}`",
        f"- dense_definition: `{payload['dense_definition']}`",
        f"- dense_edge_threshold: `{payload['dense_edge_threshold']}`",
        f"- selection_split: `{payload['selection']['split']}`",
        f"- dense_penalty_weight: `{payload['selection']['dense_penalty_weight']}`",
        f"- selected_checkpoint: `{payload['selection']['selected_checkpoint']}`",
        "",
        (
            "MMD scores use largest connected components of generated graphs for Table 2 compatibility. "
            "Raw dense statistics are computed before largest-component filtering."
        ),
        "",
        "Checkpoint selection uses validation only. Test metrics and test dense rates are reported after selection.",
        "",
        "## Score Summary",
        "",
        "| Checkpoint | Split | Median | Mean | Std | Worst | LCC Median Mean Edges | LCC Worst Max Edges | LCC Dense Rate | Raw Median Mean Edges | Raw Worst Max Edges | Raw Dense Rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for checkpoint_name, checkpoint_result in payload["checkpoints"].items():
        for split_name, split_result in checkpoint_result["splits"].items():
            summary = split_result["summary"]
            score = summary["score"]
            lcc_edges = summary["largest_component_edge_summary"]
            raw_edges = summary["raw_edge_summary"]
            lines.append(
                "| {checkpoint} | {split} | {median:.6f} | {mean:.6f} | {std:.6f} | {worst:.6f} | {lcc_median_mean_edges:.2f} | {lcc_worst_max_edges:.2f} | {lcc_dense:.2%} | {raw_median_mean_edges:.2f} | {raw_worst_max_edges:.2f} | {raw_dense:.2%} |".format(
                    checkpoint=checkpoint_name,
                    split=split_name,
                    median=score["median"],
                    mean=score["mean"],
                    std=score["std"],
                    worst=score["max"],
                    lcc_median_mean_edges=lcc_edges["mean_edge_count"]["median"],
                    lcc_worst_max_edges=lcc_edges["max_edge_count"]["max"],
                    lcc_dense=lcc_edges["dense_graph_rate"] or 0.0,
                    raw_median_mean_edges=raw_edges["mean_edge_count"]["median"],
                    raw_worst_max_edges=raw_edges["max_edge_count"]["max"],
                    raw_dense=raw_edges["dense_graph_rate"] or 0.0,
                )
            )

    lines.extend([
        "",
        "## Selection Candidates",
        "",
        "| Checkpoint | Median Validation MMD | Validation LCC Dense Rate | Validation Raw Dense Rate | Selection Score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for checkpoint_name, candidate in payload["selection"]["candidates"].items():
        lines.append(
            "| {checkpoint} | {median:.6f} | {lcc_dense:.2%} | {raw_dense:.2%} | {score:.6f} |".format(
                checkpoint=checkpoint_name,
                median=candidate["median_normalized_validation_mmd"],
                lcc_dense=candidate["validation_lcc_dense_graph_rate"],
                raw_dense=candidate["validation_raw_dense_graph_rate"],
                score=candidate["selection_score"],
            )
        )

    selected_checkpoint = payload["selection"]["selected_checkpoint"]
    selected_result = payload["checkpoints"].get(selected_checkpoint) if selected_checkpoint else None
    if selected_result and "test" in selected_result["splits"]:
        test_summary = selected_result["splits"]["test"]["summary"]
        score = test_summary["score"]
        lcc_edges = test_summary["largest_component_edge_summary"]
        raw_edges = test_summary["raw_edge_summary"]
        lines.extend([
            "",
            "## Selected Final Test Summary",
            "",
            f"Selected by validation: `{selected_checkpoint}`",
            "",
            "| Median | Mean | Std | Worst | LCC Dense Rate | Raw Dense Rate | Raw Worst Max Edges |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| {median:.6f} | {mean:.6f} | {std:.6f} | {worst:.6f} | {lcc_dense:.2%} | {raw_dense:.2%} | {raw_worst_max_edges:.2f} |".format(
                median=score["median"],
                mean=score["mean"],
                std=score["std"],
                worst=score["max"],
                lcc_dense=lcc_edges["dense_graph_rate"] or 0.0,
                raw_dense=raw_edges["dense_graph_rate"] or 0.0,
                raw_worst_max_edges=raw_edges["max_edge_count"]["max"],
            ),
        ])

    lines.extend(["", "## Metric Summary", ""])
    for checkpoint_name, checkpoint_result in payload["checkpoints"].items():
        for split_name, split_result in checkpoint_result["splits"].items():
            lines.extend([
                f"### {checkpoint_name} / {split_name}",
                "",
                "| Metric | Median | Mean | Std | Worst |",
                "| --- | ---: | ---: | ---: | ---: |",
            ])
            for metric in TABLE2_METRICS:
                stats = split_result["summary"]["metrics"][metric]
                lines.append(
                    "| {metric} | {median:.6f} | {mean:.6f} | {std:.6f} | {worst:.6f} |".format(
                        metric=metric,
                        median=stats["median"],
                        mean=stats["mean"],
                        std=stats["std"],
                        worst=stats["max"],
                    )
                )
            lines.append("")

    (output_dir / "resampling_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_checkpoint_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, Path(path)
    path = Path(value)
    return path.name, path


def discover_checkpoints(run_dir: Path, checkpoint_glob: str) -> list[tuple[str, Path]]:
    checkpoint_paths = sorted(run_dir.glob(checkpoint_glob))
    if checkpoint_paths:
        return [(path.stem, path) for path in checkpoint_paths]

    fallback = [
        ("best_validation_mmd_model", run_dir / "best_validation_mmd_model"),
        ("model_19999_0", run_dir / "model_19999_0"),
    ]
    return [(name, path) for name, path in fallback if path.exists()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dense-definition",
        choices=DENSE_DEFINITIONS,
        default="twice_mean",
        help=(
            "Dense outlier rule: twice_mean, mean_plus_3std, or max_reference. "
            "Validation dense rates are used for checkpoint selection; test dense rates are reporting-only."
        ),
    )
    parser.add_argument(
        "--dense-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Optional penalty added to validation median score: "
            "weight * validation raw dense graph rate. Default 0 selects by median validation MMD only."
        ),
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="checkpoint_epoch_*.pt",
        help="Checkpoint glob searched in --run-dir when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint as name=path. Overrides --checkpoint-glob discovery when provided.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = config["dataset"].upper()
    if dataset != "GRID":
        raise ValueError("This post-training script currently supports GRID only.")

    if args.output_dir is None:
        output_dir = args.run_dir / "resampling_eval"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device or config.get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    cache = load_cached_dataset(config)

    if args.checkpoint:
        checkpoints = [parse_checkpoint_arg(checkpoint) for checkpoint in args.checkpoint]
    else:
        checkpoints = discover_checkpoints(args.run_dir, args.checkpoint_glob)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {args.run_dir} with glob {args.checkpoint_glob}."
        )

    splits = {
        "validation": to_graphs(cache["val_adj"], keep_largest_component=False),
        "test": to_graphs(cache["test_list_adj"], keep_largest_component=False),
    }
    dense_edge_thresholds = {
        split_name: compute_dense_edge_threshold(graphs, args.dense_definition)
        for split_name, graphs in splits.items()
    }
    reference_edge_mean = {
        split_name: float(np.mean([graph.number_of_edges() for graph in graphs]))
        for split_name, graphs in splits.items()
    }

    payload = {
        "config": str(args.config),
        "run_dir": str(args.run_dir),
        "samples_per_split": args.samples,
        "seed": args.seed,
        "device": str(device),
        "score_mode": "normalized_table2",
        "score_denominators": PAPER_TABLE2_BY_DATASET[dataset]["GraphVAE"],
        "dense_definition": args.dense_definition,
        "dense_edge_threshold": dense_edge_thresholds,
        "reference_edge_mean": reference_edge_mean,
        "selection": {
            "split": "validation",
            "dense_penalty_weight": args.dense_penalty_weight,
            "selected_checkpoint": None,
            "selected_score": None,
            "candidates": {},
        },
        "checkpoints": {},
    }

    with locked_orca_tmp():
        for checkpoint_name, checkpoint_path in checkpoints:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            print(f"[Resample] Loading {checkpoint_name}: {checkpoint_path}")
            model = build_model(config, cache, device)
            state_dict = torch.load(str(checkpoint_path), map_location=device)
            model.load_state_dict(state_dict)
            model.eval()

            checkpoint_payload = {
                "path": str(checkpoint_path),
                "splits": {},
            }
            for split_index, (split_name, reference_graphs) in enumerate(splits.items()):
                split_samples = []
                for sample_index in range(args.samples):
                    sample_seed = args.seed + split_index * 100000 + sample_index
                    random.seed(sample_seed)
                    np.random.seed(sample_seed)
                    torch.manual_seed(sample_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sample_seed)

                    generated = generate_graphs(model, len(reference_graphs), device)
                    largest_component_graphs = generated["largest_component"]
                    raw_graphs = generated["raw"]
                    metrics = compute_table2_metrics(reference_graphs, largest_component_graphs)
                    score = normalized_table2_score(metrics, dataset)
                    split_samples.append({
                        "sample_index": sample_index,
                        "seed": sample_seed,
                        "metrics": metrics,
                        "score": score,
                        "edge_stats": edge_stats(
                            largest_component_graphs,
                            dense_edge_thresholds[split_name],
                        ),
                        "raw_edge_stats": edge_stats(raw_graphs, dense_edge_thresholds[split_name]),
                    })
                    print(
                        f"[Resample] {checkpoint_name}/{split_name} sample {sample_index + 1}/{args.samples}: "
                        f"score={score:.6f}"
                    )

                checkpoint_payload["splits"][split_name] = {
                    "samples": split_samples,
                    "summary": summarize_samples(split_samples),
                }

            payload["checkpoints"][checkpoint_name] = checkpoint_payload
            validation_summary = checkpoint_payload["splits"]["validation"]["summary"]
            validation_lcc_dense_rate = (
                validation_summary["largest_component_edge_summary"]["dense_graph_rate"] or 0.0
            )
            validation_raw_dense_rate = validation_summary["raw_edge_summary"]["dense_graph_rate"] or 0.0
            selection_score = (
                validation_summary["score"]["median"]
                + args.dense_penalty_weight * validation_raw_dense_rate
            )
            payload["selection"]["candidates"][checkpoint_name] = {
                "median_normalized_validation_mmd": validation_summary["score"]["median"],
                "validation_lcc_dense_graph_rate": validation_lcc_dense_rate,
                "validation_raw_dense_graph_rate": validation_raw_dense_rate,
                "selection_score": selection_score,
            }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if payload["selection"]["candidates"]:
        selected_checkpoint, selected_candidate = min(
            payload["selection"]["candidates"].items(),
            key=lambda item: item[1]["selection_score"],
        )
        payload["selection"]["selected_checkpoint"] = selected_checkpoint
        payload["selection"]["selected_score"] = selected_candidate["selection_score"]

    (output_dir / "resampling_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(output_dir, payload)
    print(f"Wrote {output_dir / 'resampling_metrics.json'}")
    print(f"Wrote {output_dir / 'resampling_report.md'}")


if __name__ == "__main__":
    main()
