#!/usr/bin/env python3
"""Generate coarse-to-fine loss-weight sweep configs with PyYAML."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import math
import re
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by environment.
    raise SystemExit(
        "PyYAML is required. Run this with the micro environment or install "
        "PyYAML, e.g. `pip install PyYAML`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEMPLATES = (
    "configs/cluster_tests/grid_table2_05_graphvae_motif_both_no_temp.yaml",
    "configs/cluster_tests/grid_table2_11_graphvae_mm_motif_both_no_temp.yaml",
    "configs/cluster_tests/lobster_table2_05_graphvae_motif_both_no_temp.yaml",
    "configs/cluster_tests/lobster_table2_11_graphvae_mm_motif_both_no_temp.yaml",
)
DEFAULT_SLOTS_FILE = "CLUSTER_GPU_CONFIGS_MOTIF_SAMPLE.txt"
DEFAULT_COARSE_MOTIF_WEIGHTS = (0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_COARSE_FEATURE_WEIGHTS = (0.0, 0.1, 1.0)
DEFAULT_FINE_FACTORS = (1.0 / 3.0, 1.0 / math.sqrt(3.0), 1.0, math.sqrt(3.0), 3.0)
DEFAULT_ZERO_CENTER_VALUES = (0.0, 0.01, 0.03, 0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GraphVAE loss-weight sweep configs and, optionally, a "
            "cluster schedule. Coarse mode explores log-spaced motif and "
            "feature weights; fine mode explores multiplicative neighbors "
            "around a selected center."
        )
    )
    parser.add_argument("--stage", choices=("coarse", "fine"), default="coarse")
    parser.add_argument(
        "--template",
        action="append",
        default=[],
        help=(
            "Template config to copy. Can be passed multiple times. Defaults "
            "to GRID/LOBSTER GraphVAE+Motif-both and GraphVAE-MM+Motif-both."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where generated YAML configs are written. Default: "
            "configs/loss_weight_sweeps/<stage>_grid_lobster_both."
        ),
    )
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument(
        "--coarse-motif-weights",
        nargs="+",
        type=float,
        default=list(DEFAULT_COARSE_MOTIF_WEIGHTS),
    )
    parser.add_argument(
        "--coarse-feature-weights",
        nargs="+",
        type=float,
        default=list(DEFAULT_COARSE_FEATURE_WEIGHTS),
    )
    parser.add_argument("--coarse-node-weights", nargs="+", type=float, default=None)
    parser.add_argument("--coarse-edge-weights", nargs="+", type=float, default=None)
    parser.add_argument(
        "--untie-node-edge",
        action="store_true",
        help=(
            "Use the full node x edge Cartesian product. By default node and "
            "edge weights are tied to keep the coarse sweep compact."
        ),
    )
    parser.add_argument("--center-config", type=Path, default=None)
    parser.add_argument("--center-motif", type=float, default=None)
    parser.add_argument("--center-node", type=float, default=None)
    parser.add_argument("--center-edge", type=float, default=None)
    parser.add_argument(
        "--fine-factors",
        nargs="+",
        type=float,
        default=list(DEFAULT_FINE_FACTORS),
    )
    parser.add_argument(
        "--zero-center-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_ZERO_CENTER_VALUES),
    )
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=None)
    parser.add_argument(
        "--syntactic-literal-ratio",
        type=float,
        default=1.0,
        help=(
            "alpha_syntactic_literal_motif_loss is motif_weight multiplied by "
            "this ratio. Default: 1."
        ),
    )
    parser.add_argument(
        "--graph-save-root",
        default=None,
        help=(
            "Root written into runtime.graph_save_path. Default: "
            "runs/loss_weight_sweeps/<sweep-name>."
        ),
    )
    parser.add_argument("--epoch-number", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--motif-batch-size", type=int, default=None)
    parser.add_argument(
        "--keep-best-validation-mmd",
        choices=("preserve", "true", "false"),
        default="preserve",
    )
    parser.add_argument("--best-validation-mmd-metric", default=None)
    parser.add_argument(
        "--third-party-eval",
        choices=("preserve", "true", "false"),
        default="preserve",
    )
    parser.add_argument("--manifest-file", type=Path, default=None)
    parser.add_argument("--schedule-file", type=Path, default=None)
    parser.add_argument("--slots-from", type=Path, default=Path(DEFAULT_SLOTS_FILE))
    parser.add_argument("--max-configs", type=int, default=None)
    return parser.parse_args()


def resolve_repo_path(path: Path | str) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def path_for_config(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return config


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def section(config: dict, name: str) -> dict:
    value = config.setdefault(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return value


def format_float(value: float) -> str:
    if value == 0:
        return "0.0"
    text = f"{float(value):.6g}"
    if "e" not in text and "." not in text:
        text += ".0"
    return text


def slug_float(value: float) -> str:
    text = format_float(value)
    text = text.rstrip("0").rstrip(".") if "." in text else text
    return text.replace("-", "m").replace(".", "p") or "0"


def sanitize_token(value: object) -> str:
    token = str(value).strip().lower().replace("+", "plus")
    token = re.sub(r"[^a-z0-9_.-]+", "-", token).strip("-_.")
    return token or "unknown"


def clamp_weight(value: float, min_weight: float, max_weight: float | None) -> float:
    value = max(value, min_weight)
    if max_weight is not None:
        value = min(value, max_weight)
    return 0.0 if abs(value) < 1e-12 else value


def unique_sorted(values: list[float]) -> list[float]:
    return sorted({round(float(value), 12) for value in values})


def fine_values(
    center: float,
    factors: list[float],
    zero_center_values: list[float],
    min_weight: float,
    max_weight: float | None,
) -> list[float]:
    raw_values = zero_center_values if center <= 0 else [center * factor for factor in factors]
    return unique_sorted([
        clamp_weight(float(value), min_weight, max_weight)
        for value in raw_values
    ])


def centers_from_config(path: Path) -> dict[str, float]:
    loss = section(load_config(path), "loss")
    return {
        "node": float(loss.get("alpha_node_feat", 0.0) or 0.0),
        "edge": float(loss.get("alpha_edge_feat", 0.0) or 0.0),
        "motif": float(loss.get("alpha_motif_loss", 0.0) or 0.0),
    }


def combinations_for_config(
    config: dict,
    args: argparse.Namespace,
    center_values: dict[str, float] | None,
) -> list[tuple[float, float, float]]:
    if args.stage == "coarse":
        node_weights = args.coarse_node_weights or args.coarse_feature_weights
        edge_weights = args.coarse_edge_weights or args.coarse_feature_weights
        motif_weights = args.coarse_motif_weights
        if args.untie_node_edge:
            return [
                (float(node), float(edge), float(motif))
                for node, edge, motif in itertools.product(
                    node_weights, edge_weights, motif_weights
                )
            ]
        return [
            (float(feature), float(feature), float(motif))
            for feature, motif in itertools.product(node_weights, motif_weights)
        ]

    loss = section(config, "loss")
    node_center = args.center_node
    edge_center = args.center_edge
    motif_center = args.center_motif
    if center_values is not None:
        node_center = center_values["node"] if node_center is None else node_center
        edge_center = center_values["edge"] if edge_center is None else edge_center
        motif_center = center_values["motif"] if motif_center is None else motif_center

    node_center = float(loss.get("alpha_node_feat", 0.0) if node_center is None else node_center)
    edge_center = float(loss.get("alpha_edge_feat", 0.0) if edge_center is None else edge_center)
    motif_center = float(loss.get("alpha_motif_loss", 0.0) if motif_center is None else motif_center)

    node_weights = fine_values(
        node_center,
        args.fine_factors,
        args.zero_center_values,
        args.min_weight,
        args.max_weight,
    )
    edge_weights = fine_values(
        edge_center,
        args.fine_factors,
        args.zero_center_values,
        args.min_weight,
        args.max_weight,
    )
    motif_weights = fine_values(
        motif_center,
        args.fine_factors,
        args.zero_center_values,
        args.min_weight,
        args.max_weight,
    )
    if args.untie_node_edge:
        return [
            (node, edge, motif)
            for node, edge, motif in itertools.product(node_weights, edge_weights, motif_weights)
        ]
    return [
        (feature, feature, motif)
        for feature, motif in itertools.product(node_weights, motif_weights)
    ]


def parse_slots(path: Path) -> list[tuple[str, str]]:
    slots: list[tuple[str, str]] = []
    if not path.exists():
        return slots
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            slots.append((parts[0], parts[1]))
    return slots


def write_schedule(path: Path, configs: list[Path], slots: list[tuple[str, str]]) -> None:
    if not slots:
        raise ValueError("No host/GPU slots were found for schedule generation.")
    lines = [
        "# Loss-weight sweep cluster schedule.",
        "# Generated by scripts/generate_loss_weight_sweep.py.",
        "",
    ]
    for index, config_path in enumerate(configs):
        host, gpu = slots[index % len(slots)]
        lines.append(f"{host} {gpu} {path_for_config(config_path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_config(
    template_path: Path,
    template_config: dict,
    output_dir: Path,
    sweep_name: str,
    graph_save_root: str,
    node_weight: float,
    edge_weight: float,
    motif_weight: float,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, object]]:
    config = copy.deepcopy(template_config)
    data = section(config, "data")
    model_section = section(config, "model")
    motif = section(config, "motif")
    loss = section(config, "loss")
    runtime = section(config, "runtime")

    dataset = sanitize_token(data.get("dataset", "dataset"))
    model = sanitize_token(model_section.get("model", "model"))
    rule_mode = sanitize_token(motif.get("syntactic_literal_rule_mode", "rules"))
    stem = (
        f"{dataset}_{model}_{rule_mode}_"
        f"n{slug_float(node_weight)}_e{slug_float(edge_weight)}_m{slug_float(motif_weight)}"
    )
    output_path = output_dir / f"{stem}.yaml"

    literal_motif_weight = motif_weight * args.syntactic_literal_ratio
    loss["alpha_node_feat"] = float(node_weight)
    loss["alpha_edge_feat"] = float(edge_weight)
    loss["alpha_motif_loss"] = float(motif_weight)
    loss["alpha_syntactic_literal_motif_loss"] = float(literal_motif_weight)

    run_label = f"{sweep_name}-{stem}"
    graph_save_path = f"{graph_save_root.rstrip('/')}/{stem}"
    runtime["run_label"] = run_label
    runtime["graph_save_path"] = graph_save_path
    if args.keep_best_validation_mmd != "preserve":
        runtime["keep_best_validation_mmd"] = args.keep_best_validation_mmd == "true"
    if args.best_validation_mmd_metric is not None:
        runtime["best_validation_mmd_metric"] = args.best_validation_mmd_metric
    if args.third_party_eval != "preserve":
        runtime["third_party_eval"] = args.third_party_eval == "true"

    if args.epoch_number is not None:
        section(config, "experiment")["epoch_number"] = int(args.epoch_number)
    if args.train_batch_size is not None:
        section(config, "experiment")["train_batch_size"] = int(args.train_batch_size)
    if args.motif_batch_size is not None:
        motif["motif_batch_size"] = int(args.motif_batch_size)

    write_config(output_path, config)
    row = {
        "config": path_for_config(output_path),
        "template": path_for_config(template_path),
        "stage": args.stage,
        "dataset": data.get("dataset", ""),
        "model": model_section.get("model", ""),
        "syntactic_literal_rule_mode": motif.get("syntactic_literal_rule_mode", ""),
        "alpha_node_feat": format_float(node_weight),
        "alpha_edge_feat": format_float(edge_weight),
        "alpha_motif_loss": format_float(motif_weight),
        "alpha_syntactic_literal_motif_loss": format_float(literal_motif_weight),
        "run_label": run_label,
        "graph_save_path": graph_save_path,
    }
    return output_path, row


def main() -> int:
    args = parse_args()
    templates = args.template or list(DEFAULT_TEMPLATES)
    output_dir = resolve_repo_path(
        args.output_dir or Path(f"configs/loss_weight_sweeps/{args.stage}_grid_lobster_both")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_name = args.sweep_name or output_dir.name
    graph_save_root = args.graph_save_root or f"runs/loss_weight_sweeps/{sweep_name}"
    manifest_file = resolve_repo_path(args.manifest_file or (output_dir / "manifest.csv"))

    center_values = None
    if args.center_config is not None:
        center_values = centers_from_config(resolve_repo_path(args.center_config))

    config_paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []

    for template in templates:
        template_path = resolve_repo_path(template)
        template_config = load_config(template_path)
        for node_weight, edge_weight, motif_weight in combinations_for_config(
            template_config,
            args,
            center_values,
        ):
            config_path, row = make_config(
                template_path=template_path,
                template_config=template_config,
                output_dir=output_dir,
                sweep_name=sweep_name,
                graph_save_root=graph_save_root,
                node_weight=node_weight,
                edge_weight=edge_weight,
                motif_weight=motif_weight,
                args=args,
            )
            config_paths.append(config_path)
            manifest_rows.append(row)
            if args.max_configs is not None and len(config_paths) >= args.max_configs:
                break
        if args.max_configs is not None and len(config_paths) >= args.max_configs:
            break

    fieldnames = [
        "config",
        "template",
        "stage",
        "dataset",
        "model",
        "syntactic_literal_rule_mode",
        "alpha_node_feat",
        "alpha_edge_feat",
        "alpha_motif_loss",
        "alpha_syntactic_literal_motif_loss",
        "run_label",
        "graph_save_path",
    ]
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    if args.schedule_file is not None:
        schedule_path = resolve_repo_path(args.schedule_file)
        write_schedule(schedule_path, config_paths, parse_slots(resolve_repo_path(args.slots_from)))
        print(f"Wrote schedule: {path_for_config(schedule_path)}")

    print(f"Wrote {len(config_paths)} configs to {path_for_config(output_dir)}")
    print(f"Wrote manifest: {path_for_config(manifest_file)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
