# | Action               | Shortcut            |
# | -------------------- | ------------------- |
# | Fold current block   | `Ctrl + Shift + [`  |
# | Unfold current block | `Ctrl + Shift + ]`  |
# | Fold all             | `Ctrl + K Ctrl + 0` |
# | Unfold all           | `Ctrl + K Ctrl + J` |


#====================================================================================
# region imports
import logging
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import hashlib
import math
from pathlib import Path
import plotter
import torch.nn.functional as F
import argparse
import re
import subprocess
import sys
try:
    import yaml
except ImportError:
    yaml = None
from model import *
from data import *
import pickle
import random as random
from GlobalProperties import *
from stat_rnn import mmd_eval
import time
import timeit
import dgl
from util import *
from motif_counting.motif_store import RuleBasedMotifStore, get_motif_pickle_path
from motif_counting.motif_counter import RelationalMotifCounter
from motif_counting.motif_loss_utils import (
    compute_hard_motif_metrics,
    get_motif_temperature,
    get_reconstructed_adj_probs,
    summarize_hard_motif_threshold_sweep,
    summarize_single_graph_motif_counts,
)
from motif_counting.motif_objective import (
    MOTIF_LOSS_MODES,
    NON_LITERAL_MOTIF_GROUP,
    SYNTACTIC_LITERAL_MOTIF_GROUP,
    UNIT_RELATION_MOTIF_GROUP,
    build_motif_group_objectives,
    calibrate_group_histogram_specs,
    compute_grouped_motif_loss,
)
from motif_counting.motif_representations import (
    MOTIF_OUTPUT_MODE_CHOICES,
    canonicalize_motif_output_mode,
    represent_full_motif_matrices,
)
from motif_counting.sanity_check_compare import (
    compare_aggregated_counts_to_factorbase_detailed,
)
from ranking_score import (
    BEST_VALIDATION_MMD_SCORE_MODES,
    TABLE2_VALIDATION_MMD_KEYS,
    compute_validation_mmd_score,
    score_components_for_mode,
    score_denominators_for_mode,
    score_metrics_for_mode,
    score_weights_for_mode,
)
from loss_weight_utils import apply_kia_bce_kl_weights
#endregion
#====================================================================================

subgraphSize = None
keepThebest = False

# Choose which BFS reordering to use before training/counting.
# False -> legacy BFS from node 0 only
# True  -> BFS over all connected components (safe for disconnected graphs)
USE_ALL_COMPONENTS_BFS = True

#====================================================================================
#region arguments
def str2bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, received '{value}'."
    )


def ensure_deterministic_python_hash_seed(seed, deterministic):
    """Restart once with PYTHONHASHSEED set before Python creates hash state."""
    if not deterministic:
        return

    seed_text = str(int(seed))
    if os.environ.get("PYTHONHASHSEED") == seed_text:
        return

    reexec_marker = "GRAPHVAE_DETERMINISTIC_REEXEC"
    if os.environ.get(reexec_marker) == seed_text:
        raise RuntimeError(
            "Deterministic mode requires PYTHONHASHSEED="
            f"{seed_text}, but the restarted process still has "
            f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')!r}."
        )

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = seed_text
    env[reexec_marker] = seed_text
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def configure_reproducibility(seed, deterministic=True, deterministic_warn_only=False):
    """Seed every RNG used by this script and ask PyTorch for deterministic ops."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(dgl, "seed"):
        dgl.seed(seed)
    if hasattr(dgl, "random") and hasattr(dgl.random, "seed"):
        dgl.random.seed(seed)

    if deterministic:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(
                    True,
                    warn_only=bool(deterministic_warn_only),
                )
            except TypeError:
                torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)


def _flatten_config_sections(config_data):
    flat_config = {}
    for key, value in config_data.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if nested_key in flat_config:
                    raise ValueError(
                        f"Duplicate config key '{nested_key}' found while flattening sections."
                    )
                flat_config[nested_key] = nested_value
        else:
            if key in flat_config:
                raise ValueError(f"Duplicate config key '{key}' found in config file.")
            flat_config[key] = value
    return flat_config


def load_config_defaults(config_path, valid_keys):
    if yaml is None:
        raise ImportError(
            "PyYAML is required for --config support. Install it with 'pip install PyYAML'."
        )

    resolved_path = Path(config_path).expanduser()
    with resolved_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    if not isinstance(config_data, dict):
        raise ValueError(
            f"Config file '{resolved_path}' must contain a YAML mapping at the top level."
        )

    flat_config = _flatten_config_sections(config_data)
    unknown_keys = sorted(set(flat_config) - set(valid_keys))
    if unknown_keys:
        raise ValueError(
            f"Unknown config keys in '{resolved_path}': {', '.join(unknown_keys)}"
        )

    return flat_config


DATASET_CACHE_SCHEMA_VERSION = "dataset-cache-v3"
DEFAULT_SPLIT_SEED = 123
DEFAULT_LEGACY_TRAIN_FRACTION = 0.8
DEFAULT_PAPER_TRAIN_FRACTION = 0.7
DEFAULT_PAPER_VAL_FRACTION = 0.1
DEFAULT_TEMP_ANNEAL_GUARD_RATIO = 2.0


def _sanitize_cache_component(value):
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "none"


def _format_cache_float(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def resolve_split_plan(split_mode, train_fraction_arg, val_fraction_arg, split_seed):
    split_seed = int(split_seed)

    if split_mode == "paper_70_10_20":
        train_fraction = (
            DEFAULT_PAPER_TRAIN_FRACTION
            if train_fraction_arg is None
            else float(train_fraction_arg)
        )
        val_fraction = (
            DEFAULT_PAPER_VAL_FRACTION
            if val_fraction_arg is None
            else float(val_fraction_arg)
        )
        split_kind = "three_way"
    else:
        train_fraction = (
            DEFAULT_LEGACY_TRAIN_FRACTION
            if train_fraction_arg is None
            else float(train_fraction_arg)
        )
        val_fraction = 0.0 if val_fraction_arg is None else float(val_fraction_arg)
        if val_fraction != 0.0:
            raise ValueError(
                "--val_fraction is only supported with split_mode=paper_70_10_20."
            )
        split_kind = "two_way"

    test_fraction = 1.0 - train_fraction - val_fraction
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")
    if test_fraction <= 0.0:
        raise ValueError(
            "Split fractions must leave a positive test fraction; "
            f"got train={train_fraction}, val={val_fraction}, test={test_fraction}."
        )

    return {
        "split_kind": split_kind,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "split_seed": split_seed,
    }


def build_dataset_cache_metadata(
    dataset,
    split_mode,
    bfs_strategy,
    split_plan,
    feature_schema="default",
):
    return {
        "cache_schema_version": DATASET_CACHE_SCHEMA_VERSION,
        "dataset": dataset,
        "feature_schema": feature_schema,
        "split_mode": split_mode,
        "bfs_strategy": bfs_strategy,
        "split_kind": split_plan["split_kind"],
        "train_fraction": float(split_plan["train_fraction"]),
        "val_fraction": float(split_plan["val_fraction"]),
        "test_fraction": float(split_plan["test_fraction"]),
        "split_seed": int(split_plan["split_seed"]),
    }


def build_dataset_cache_name(cache_metadata):
    return (
        f"{_sanitize_cache_component(cache_metadata['dataset'])}"
        f"_split-{_sanitize_cache_component(cache_metadata['split_mode'])}"
        f"_train{_format_cache_float(cache_metadata['train_fraction'])}"
        f"_val{_format_cache_float(cache_metadata['val_fraction'])}"
        f"_test{_format_cache_float(cache_metadata['test_fraction'])}"
        f"_seed{cache_metadata['split_seed']}"
        f"_bfs-{_sanitize_cache_component(cache_metadata['bfs_strategy'])}"
        f"_features-{_sanitize_cache_component(cache_metadata['feature_schema'])}.pkl"
    )


def _metadata_values_match(expected_value, cached_value):
    if isinstance(expected_value, float):
        try:
            return math.isclose(
                expected_value, float(cached_value), rel_tol=0.0, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            return False
    return expected_value == cached_value


def validate_dataset_cache_metadata(cache_payload, expected_metadata, cache_path):
    cached_metadata = cache_payload.get("cache_metadata")
    if cached_metadata is None:
        raise ValueError(
            "Dataset cache is missing cache_metadata and may come from an older "
            f"split definition. Delete/regenerate this cache: {cache_path}"
        )

    mismatches = []
    for key, expected_value in expected_metadata.items():
        cached_value = cached_metadata.get(key)
        if not _metadata_values_match(expected_value, cached_value):
            mismatches.append((key, expected_value, cached_value))

    if mismatches:
        details = "; ".join(
            f"{key}: expected {expected!r}, cached {cached!r}"
            for key, expected, cached in mismatches
        )
        raise ValueError(
            f"Dataset cache metadata mismatch for {cache_path}. {details}. "
            "Delete/regenerate the cache or use a different --dataset_cache_dir."
        )


MODEL_NAME_ALIASES = {
    "graphvae": "kipf",
    "graphvae-mm": "GraphVAE-MM",
    "kernelaugmentedwithtotalnumberoftriangles": "GraphVAE-MM",
}

def normalize_model_name(model_name):
    if model_name is None:
        return model_name

    normalized = str(model_name).strip()
    return MODEL_NAME_ALIASES.get(normalized.lower(), normalized)


def default_feature_loss_weight(model_name):
    """Default feature-decoder supervision: stronger for GraphVAE-MM."""
    return 40.0 if normalize_model_name(model_name) == "GraphVAE-MM" else 1.0


def default_motif_loss_weight(model_name, use_motif_loss):
    """Use a graph-statistic-like motif weight for GraphVAE-MM and a smaller one for kipf."""
    if not use_motif_loss:
        return 0.0
    return 1.0 if normalize_model_name(model_name) == "GraphVAE-MM" else 0.1


def resolve_loss_weight(value, default_value):
    return float(default_value if value is None else value)


MMD_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MMD_RESULT_LABELS = {
    "degree": "degree",
    "clustering": "clustering",
    "orbit": "orbits",
    "spectral": "Spec",
    "triangle": "Tri",
    "sparsity": "sparsity",
    "diameter": "diameter",
    "mmd_rbf": "mmd_rbf",
    "mmd_rbf_std": "mmd_rbf_std",
    "precision": "precision",
    "precision_std": "precision_std",
    "recall": "recall",
    "recall_std": "recall_std",
    "f1_pr": "f1_pr",
    "f1_pr_std": "f1_pr_std",
}
MMD_EDGE_COUNT_LABELS = {
    "reference_edge_count": "average edge # in test set",
    "generated_edge_count": "average edge # in grnrated set",
}
THIRD_PARTY_EVAL_GENERATED_FILENAME = "Single_comp_generatedGraphs_adj_final_eval.npy"
THIRD_PARTY_EVAL_REFERENCE_FILENAME = "testGraphs_adj_.npy"
THIRD_PARTY_EVAL_JSON_FILENAME = "graph_realism_random_gin.json"
THIRD_PARTY_EVAL_SUMMARY_FILENAME = "graph_realism_batch_summary.csv"
THIRD_PARTY_EVAL_LOG_FILENAME = "third_party_eval.log"
FINAL_TABLE2_METRICS_FILENAME = "final_table2_metrics.json"
FINAL_TABLE3_METRICS_FILENAME = "final_table3_metrics.json"
FINAL_METRICS_SUMMARY_FILENAME = "final_metrics_summary.json"


def parse_graph_quality_result(mmd_result):
    metrics = {}
    for metric_name, result_label in MMD_RESULT_LABELS.items():
        match = re.search(
            rf"{re.escape(result_label)}\s*:\s*({MMD_FLOAT_PATTERN})",
            str(mmd_result),
        )
        metrics[metric_name] = float(match.group(1)) if match else None
    for metric_name, result_label in MMD_EDGE_COUNT_LABELS.items():
        match = re.search(
            rf"{re.escape(result_label)}\s*:\s*({MMD_FLOAT_PATTERN})",
            str(mmd_result),
        )
        metrics[metric_name] = float(match.group(1)) if match else None
    return metrics


def parse_table2_mmd_result(mmd_result):
    metrics = parse_graph_quality_result(mmd_result)
    return {
        metric_name: metrics.get(metric_name)
        for metric_name in TABLE2_VALIDATION_MMD_KEYS
    }


def write_json_file(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_best_validation_mmd_metadata(metadata_path, metadata):
    write_json_file(metadata_path, metadata)


def table2_metrics_from_parsed(metrics):
    return {
        metric_name: metrics.get(metric_name)
        for metric_name in TABLE2_VALIDATION_MMD_KEYS
    }


def table3_metrics_from_parsed(metrics):
    return {
        metric_name: metrics.get(metric_name)
        for metric_name in (
            "mmd_rbf",
            "mmd_rbf_std",
            "precision",
            "precision_std",
            "recall",
            "recall_std",
            "f1_pr",
            "f1_pr_std",
        )
    }


def load_third_party_metrics(third_party_json_path):
    if third_party_json_path is None:
        return None
    third_party_json_path = Path(third_party_json_path)
    if not third_party_json_path.is_file():
        return None
    with third_party_json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_final_metric_summaries(
    run_dir,
    final_mmd_result,
    third_party_json_path=None,
    model_source="final_epoch",
):
    run_dir = Path(run_dir)
    parsed_metrics = parse_graph_quality_result(final_mmd_result)
    common_payload = {
        "split": "test",
        "model_source": model_source,
        "generated_graphs": str(run_dir / THIRD_PARTY_EVAL_GENERATED_FILENAME),
        "reference_graphs": str(run_dir / THIRD_PARTY_EVAL_REFERENCE_FILENAME),
        "raw_eval_result": str(final_mmd_result),
    }
    table2_payload = {
        **common_payload,
        "metric_family": "table2_structural_mmd",
        "metrics": table2_metrics_from_parsed(parsed_metrics),
        "extra_metrics": {
            "triangle": parsed_metrics.get("triangle"),
            "sparsity": parsed_metrics.get("sparsity"),
            "reference_edge_count": parsed_metrics.get("reference_edge_count"),
            "generated_edge_count": parsed_metrics.get("generated_edge_count"),
        },
    }
    third_party_metrics = load_third_party_metrics(third_party_json_path)
    table3_payload = {
        **common_payload,
        "metric_family": "table3_gnn",
        "local_eval_metrics": table3_metrics_from_parsed(parsed_metrics),
        "third_party_eval_json": (
            str(Path(third_party_json_path))
            if third_party_json_path is not None
            else None
        ),
        "third_party_eval_metrics": third_party_metrics,
    }
    summary_payload = {
        "split": "test",
        "model_source": model_source,
        "table2_metrics_file": str(run_dir / FINAL_TABLE2_METRICS_FILENAME),
        "table3_metrics_file": str(run_dir / FINAL_TABLE3_METRICS_FILENAME),
        "table2": table2_payload,
        "table3": table3_payload,
    }
    write_json_file(run_dir / FINAL_TABLE2_METRICS_FILENAME, table2_payload)
    write_json_file(run_dir / FINAL_TABLE3_METRICS_FILENAME, table3_payload)
    write_json_file(run_dir / FINAL_METRICS_SUMMARY_FILENAME, summary_payload)
    return summary_payload


def _run_git_command(git_args):
    try:
        result = subprocess.run(
            ["git", *git_args],
            cwd=Path(__file__).resolve().parent,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        return f"<git command failed: {exc}>"

    output = result.stdout.strip()
    if result.returncode != 0:
        error_output = result.stderr.strip()
        return f"<git {' '.join(git_args)} failed: {error_output}>"
    return output


def _file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_motif_cache_reproducibility_metadata(args):
    metadata = {
        "enabled": bool(getattr(args, "motif_loss", False)),
    }
    if not metadata["enabled"]:
        return metadata

    motif_pickle_path = get_motif_pickle_path(args.database_name, args)
    metadata.update(
        {
            "database_name": args.database_name,
            "syntactic_literal_rule_mode": getattr(
                args, "syntactic_literal_rule_mode", None
            ),
            "motif_cache_dir": str(motif_pickle_path.parent),
            "motif_pickle_path": str(motif_pickle_path),
            "motif_pickle_exists_at_run_start": motif_pickle_path.exists(),
        }
    )

    if motif_pickle_path.exists():
        stat = motif_pickle_path.stat()
        metadata.update(
            {
                "motif_pickle_size_bytes": stat.st_size,
                "motif_pickle_mtime": stat.st_mtime,
                "motif_pickle_sha256": _file_sha256(motif_pickle_path),
            }
        )

    return metadata


def write_run_reproducibility_files(run_dir, args, run_label):
    run_dir = Path(run_dir)
    config_path = Path(args.config).expanduser() if args.config else None
    git_commit = _run_git_command(["rev-parse", "HEAD"])
    git_describe = _run_git_command(["describe", "--tags", "--always", "--dirty"])
    git_tags_at_head = _run_git_command(["tag", "--points-at", "HEAD"])
    git_status = _run_git_command(["status", "--short"])
    git_diff = _run_git_command(["diff", "HEAD", "--"])

    command = " ".join(sys.argv)
    metadata = {
        "run_label": run_label,
        "command": command,
        "config_path": str(config_path) if config_path is not None else None,
        "git_commit": git_commit,
        "git_describe": git_describe,
        "git_tags_at_head": git_tags_at_head.splitlines() if git_tags_at_head else [],
        "git_status_short": git_status.splitlines() if git_status else [],
        "motif_cache": build_motif_cache_reproducibility_metadata(args),
        "args": vars(args),
    }

    (run_dir / "RUN_LABEL.txt").write_text((run_label or "unlabeled") + "\n", encoding="utf-8")
    (run_dir / "reproducibility.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "git_status.txt").write_text((git_status or "clean") + "\n", encoding="utf-8")
    (run_dir / "git_diff.patch").write_text((git_diff or "") + "\n", encoding="utf-8")

    if config_path is not None and config_path.exists():
        (run_dir / "run_config_used.yaml").write_text(
            config_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    reproduce_lines = [
        "# Run Reproducibility",
        "",
        f"- run_label: `{run_label or 'unlabeled'}`",
        f"- git_commit: `{git_commit}`",
        f"- git_describe: `{git_describe}`",
        f"- config: `{config_path}`" if config_path is not None else "- config: CLI/defaults",
        "",
        "```bash",
        command,
        "```",
        "",
        "Use `run_config_used.yaml` and `reproducibility.json` in this folder",
        "to recover the exact config and git state recorded when the run started.",
    ]
    (run_dir / "REPRODUCE.md").write_text("\n".join(reproduce_lines) + "\n", encoding="utf-8")


def resolve_third_party_eval_device(device_arg, training_device):
    if device_arg != "same":
        return device_arg
    return "cuda" if str(training_device).startswith("cuda") else "cpu"


def build_third_party_eval_env(device_arg, training_device):
    env = os.environ.copy()
    device_text = str(training_device)
    if device_arg == "same" and "CUDA_VISIBLE_DEVICES" not in env:
        match = re.fullmatch(r"cuda:(\d+)", device_text)
        if match:
            env["CUDA_VISIBLE_DEVICES"] = match.group(1)
    return env


def run_third_party_graph_realism_eval(run_dir, args, training_device):
    run_dir = Path(run_dir)
    generated_path = run_dir / THIRD_PARTY_EVAL_GENERATED_FILENAME
    reference_path = run_dir / THIRD_PARTY_EVAL_REFERENCE_FILENAME
    if not generated_path.is_file() or not reference_path.is_file():
        missing = [
            str(path)
            for path in (generated_path, reference_path)
            if not path.is_file()
        ]
        raise FileNotFoundError(
            "Third-party graph realism evaluation requires generated and "
            f"reference graph files. Missing: {missing}"
        )

    repo_root = Path(__file__).resolve().parent
    evaluator_script = repo_root / "scripts" / "evaluate_graph_realism_batch.py"
    eval_device = resolve_third_party_eval_device(
        args.third_party_eval_device,
        training_device,
    )
    summary_csv_path = run_dir / THIRD_PARTY_EVAL_SUMMARY_FILENAME
    log_path = run_dir / THIRD_PARTY_EVAL_LOG_FILENAME
    command = [
        sys.executable,
        str(evaluator_script),
        "--run-dir",
        str(run_dir),
        "--generated-filename",
        THIRD_PARTY_EVAL_GENERATED_FILENAME,
        "--reference-filename",
        THIRD_PARTY_EVAL_REFERENCE_FILENAME,
        "--json-filename",
        args.third_party_eval_json_filename,
        "--summary-csv",
        str(summary_csv_path),
        "--repeats",
        str(args.third_party_eval_repeats),
        "--max-graphs",
        str(args.third_party_eval_max_graphs),
        "--seed",
        str(args.third_party_eval_seed),
        "--device",
        eval_device,
    ]
    if not args.third_party_eval_structural_features:
        command.append("--no-structural-features")

    message = "Running third-party graph realism evaluation: " + " ".join(command)
    print(message)
    logging.info(message)

    result = subprocess.run(
        command,
        cwd=str(repo_root),
        env=build_third_party_eval_env(args.third_party_eval_device, training_device),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(result.stdout or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        logging.info(result.stdout)

    if result.returncode != 0:
        failure_message = (
            "Third-party graph realism evaluation failed with exit code "
            f"{result.returncode}. See {log_path}"
        )
        if args.third_party_eval_strict:
            raise RuntimeError(failure_message)
        print("Warning: " + failure_message)
        logging.warning(failure_message)
        return None

    success_message = (
        "Third-party graph realism evaluation completed. "
        f"JSON={run_dir / args.third_party_eval_json_filename}, "
        f"summary_csv={summary_csv_path}, log={log_path}"
    )
    print(success_message)
    logging.info(success_message)
    return run_dir / args.third_party_eval_json_filename


parser = argparse.ArgumentParser(description='Kernel VGAE')

#===============================
# Config file
#===============================
parser.add_argument(
    '--config',
    type=str,
    default=None,
    help='Path to a single YAML config file.'
)

#===============================
# Data arguments
#===============================
parser.add_argument(
    '--dataset',
    dest="dataset",
    default="TRIANGULAR_GRID",
    help="possible choices include AIDS, ENZYMES, MUTAG, PTC, PROTEINS, QM9, ogbg-molbbbp, GRID, TRIANGULAR_GRID, and LOBSTER"
)
parser.add_argument(
    '-f',
    '--use_feature',
    dest="use_feature",
    default=True,
    type=str2bool,
    help="either use features or identity matrix"
)
parser.add_argument(
    '--bfs_ordering',
    dest="bfsOrdering",
    default=True,
    type=str2bool,
    help="use bfs for graph permutations"
)
parser.add_argument(
    '--bfs_strategy',
    type=str,
    default='legacy_first_component',
    choices=['all_components', 'legacy_first_component'],
    help='BFS ordering strategy. all_components preserves current behavior; legacy_first_component matches the original paper code path.'
)
parser.add_argument(
    '--directed',
    dest="directed",
    default=True,
    type=str2bool,
    help="is the dataset directed?!"
)
parser.add_argument(
    '--database_name',
    type=str,
    default='grid_undir_feat_snap_7a58e6'
)  # qm9_experiment, ogbg-molbbbp_experiment, PTC_experiment, MUTAG_experiment, PVGAErandomGraphs_experiment, FIRSTMM_DB_experiment, DD_experiment, GRID_experiment, PROTEINS_experiment, lobster_experiment, wheel_graph_experiment, TRIANGULAR_GRID_experiment, tree_experiment
parser.add_argument(
    '--graph_type',
    type=str,
    default='homogeneous',
    choices=['homogeneous', 'heterogeneous']
)
parser.add_argument(
    '--graph_index_start',
    type=int,
    default=None,
    help='First graph index to count (inclusive). Only valid when dataset has more than one graph.'
)
parser.add_argument(
    '--graph_index_end',
    type=int,
    default=None,
    help='Last graph index to count (inclusive). Only valid when dataset has more than one graph.'
)
parser.add_argument(
    '--data_dir',
    type=str,
    default=None,
    help='Optional raw dataset root. If set, main.py exports DATA_DIR for data.py; otherwise data.py uses DATA_DIR or local data_raw/.'
)
parser.add_argument(
    '--lobster_feature_schema',
    choices=['old_v1', 'optimal_v2'],
    default='optimal_v2',
    help=(
        "LOBSTER feature definition. old_v1 reproduces the "
        "lobster_undir_feat_snap_85093d experiments; optimal_v2 uses the "
        "newer best_lobster.py schema."
    ),
)
parser.add_argument(
    '--tu_attribute_bins',
    type=int,
    default=8,
    help='Quantile bins per continuous AIDS/ENZYMES node attribute.',
)
parser.add_argument(
    '--tu_max_nodes',
    type=int,
    default=None,
    help='Optionally exclude AIDS/ENZYMES graphs above this node count.',
)
parser.add_argument(
    '--split_mode',
    type=str,
    default='legacy_80_20',
    choices=['legacy_80_20', 'paper_70_10_20'],
    help='Dataset split protocol. legacy_80_20 preserves current behavior; paper_70_10_20 is opt-in for Table 2 reproduction.'
)
parser.add_argument(
    '--split_seed',
    type=int,
    default=DEFAULT_SPLIT_SEED,
    help='Random seed used when shuffling graphs before train/validation/test splitting.'
)
parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help='Global random seed for model initialization, training shuffles, VAE sampling, generation, and evaluation.'
)
parser.add_argument(
    '--deterministic',
    type=str2bool,
    default=True,
    help='Enable strict deterministic execution where PyTorch supports it.'
)
parser.add_argument(
    '--deterministic_warn_only',
    type=str2bool,
    default=False,
    help='If true, warn instead of failing when PyTorch detects a nondeterministic operation.'
)
parser.add_argument(
    '--train_fraction',
    type=float,
    default=None,
    help='Optional override for the training split fraction. Defaults to 0.8 for legacy_80_20 and 0.7 for paper_70_10_20.'
)
parser.add_argument(
    '--val_fraction',
    type=float,
    default=None,
    help='Optional validation split fraction for paper_70_10_20. The test fraction is 1 - train_fraction - val_fraction.'
)

#===============================
# Model arguments
#===============================
parser.add_argument(
    '--model',
    dest="model",
    default="GraphVAE",
    help="Model name. Accepted aliases: GraphVAE/kipf for the baseline, GraphVAE-MM/KernelAugmentedWithTotalNumberOfTriangles for the kernel-augmented variant."
)
parser.add_argument(
    '--encoder',
    dest="encoder_type",
    default="AvePool",
    help="the encoder: only option in this rep is 'AvePool'"
)  # only option in this rep is "AvePool"
parser.add_argument(
    '--decoder',
    dest="decoder",
    default="FC",
    help="the decoder type, FC is only option in this rep"
)
parser.add_argument(
    '--graph_em_dim',
    dest="graphEmDim",
    default=1024,
    type=int,
    help="the dimention of graph Embeding LAyer; z"
)
parser.add_argument(
    '--beta',
    dest="beta",
    default=None,
    help="beta coefiicieny",
    type=float
)
parser.add_argument(
    '--correct_reparameterization',
    type=str2bool,
    default=False,
    help=(
        "Opt in to the correct VAE sample z=mean+eps*std. The default false "
        "preserves Kia's legacy z=mean+eps*variance behavior for baseline "
        "reproduction; pass true only for corrected model runs."
    ),
)

#===============================
# Experiment arguments
#===============================
parser.add_argument(
    '-e',
    '--epoch_number',
    dest="epoch_number",
    default=20000,
    type=int,
    help="Number of Epochs to train the model"
)
parser.add_argument(
    '-v',
    '--vis_step',
    dest="Vis_step",
    default=1000,
    type=int,
    help="at every Vis_step 'minibatch' the plots will be updated"
)
parser.add_argument(
    '--redraw',
    dest="redraw",
    default=False,
    type=str2bool,
    help="either update the log plot each step"
)
parser.add_argument(
    '--lr',
    dest="lr",
    default=0.0003,
    type=float,
    help="model learning rate"
)
parser.add_argument(
    '-b',
    '--train_batch_size',
    dest="train_batch_size",
    default=200,
    type=int,
    help="training mini-batch size"
)
parser.add_argument(
    '--task',
    dest="task",
    default="graphGeneration",
    help="only option in this rep is graphGeneration"
)

#===============================
# Motif arguments
#===============================
parser.add_argument('--motif_loss', type=str2bool, default=False)
parser.add_argument(
    '--motif_output_mode',
    type=str,
    default='total_count',
    choices=sorted(MOTIF_OUTPUT_MODE_CHOICES),
    help=(
        'Motif statistic representation: full_matrix retains every valid matrix '
        'entry; row_column_marginals retains both marginals for NxN results and '
        'the non-singleton marginal for 1xN/Nx1 results; marginal_histogram '
        'forms permutation-invariant soft histograms of those marginals; '
        'degree_histogram applies GraphVAE-MM soft degree bins to row sums of '
        'natural NxN matrices; '
        'total_count sums all entries. Legacy aliases: matrix and count.'
    ),
)
parser.add_argument(
    '--non_literal_motif_output_mode',
    type=str,
    default=None,
    choices=sorted(MOTIF_OUTPUT_MODE_CHOICES),
    help=(
        'Representation for original/non-literal relational motifs. '
        'Defaults to motif_output_mode.'
    ),
)
parser.add_argument(
    '--syntactic_literal_motif_output_mode',
    type=str,
    default=None,
    choices=sorted(MOTIF_OUTPUT_MODE_CHOICES),
    help=(
        'Representation for syntactic-literal motifs. '
        'Defaults to motif_output_mode.'
    ),
)
parser.add_argument(
    '--unit_relation_motif_output_mode',
    type=str,
    default=None,
    choices=sorted(MOTIF_OUTPUT_MODE_CHOICES),
    help=(
        'Optional separate representation for positive bare binary-relation motifs. '
        'When set, these motifs are removed from the non-literal group.'
    ),
)
parser.add_argument(
    '--motif_histogram_num_bins',
    type=int,
    default=16,
    help='Number of soft bins per marginal histogram; must be at least 2.',
)
parser.add_argument(
    '--motif_histogram_smoothing',
    type=float,
    default=0.25,
    help=(
        'Histogram sigmoid-boundary temperature as a fraction of each '
        'log-count bin width; must be greater than zero.'
    ),
)
parser.add_argument(
    '--use_syntactic_literal_rules',
    type=str2bool,
    default=True,
    help='Enable the synthetic literal-derived motif rules and literal-value metadata.'
)
parser.add_argument(
    '--syntactic_literal_rule_mode',
    type=str,
    default='both',
    choices=['original', 'literals', 'both'],
    help=(
        'Motif rule scope when syntactic literals are enabled: original uses only '
        'FactorBase DB rules, literals uses only node/edge feature literal rules, '
        'and both uses DB rules plus injected literal rules with split loss weights.'
    )
)
# The default motif loss is symmetric: zero-observed motifs are included through
# Laplace smoothing so extra motifs in the reconstruction are penalized too.
# `calibrated_gaussian` switches to a Kia-MM style Gaussian NLL where sigma for
# each motif column is estimated from the minibatch RMSE.
parser.add_argument(
    '--motif_loss_mode',
    type=str,
    default='abs_log_ratio',
    choices=sorted(MOTIF_LOSS_MODES),
    help=(
        'Motif loss variant: symmetric abs(log-ratio), squared log-ratio, '
        'or calibrated_gaussian for Kia-MM style Gaussian NLL '
        'with per-motif sigma estimated from minibatch RMSE. Structured output '
        'modes require calibrated_gaussian.'
    )
)
parser.add_argument(
    '--non_literal_motif_loss_mode',
    type=str,
    default=None,
    choices=sorted(MOTIF_LOSS_MODES),
    help='Loss for original/non-literal motifs. Defaults to motif_loss_mode.',
)
parser.add_argument(
    '--syntactic_literal_motif_loss_mode',
    type=str,
    default=None,
    choices=sorted(MOTIF_LOSS_MODES),
    help='Loss for syntactic-literal motifs. Defaults to motif_loss_mode.',
)
parser.add_argument(
    '--unit_relation_motif_loss_mode',
    type=str,
    default=None,
    choices=sorted(MOTIF_LOSS_MODES),
    help=(
        'Loss for the optional unit-relation motif group. Defaults to '
        'motif_loss_mode when that group is enabled.'
    ),
)
# Motif-temperature annealing only affects motif counting, not the main
# reconstruction loss. Keep start=end=1.0 to disable it, or use a schedule like
# start=1.0, end=0.5, start_frac=0.5 to keep training smooth early and sharpen
# the motif probabilities during the second half of training.
parser.add_argument(
    '--motif_temperature_start',
    type=float,
    default=1.0,
    help='Starting temperature for motif-count probabilities; lower than 1 sharpens logits.'
)
parser.add_argument(
    '--motif_temperature_end',
    type=float,
    default=0.5,
    help='Final temperature for motif-count probabilities after annealing.'
)
parser.add_argument(
    '--motif_temperature_anneal_start_frac',
    type=float,
    default=0.5,
    help='Fraction of training to keep the starting motif temperature before annealing.'
)
parser.add_argument(
    '--motif_temperature_guard_ratio',
    type=float,
    default=None,
    help=(
        'If > 0, accept an annealed motif temperature only when the weighted '
        'motif term is no more than this ratio times the other weighted loss '
        'terms; otherwise adaptively relax the effective temperature for that '
        f'batch. Default: {DEFAULT_TEMP_ANNEAL_GUARD_RATIO} when motif '
        'temperature annealing sharpens logits, otherwise 0. Set 0 to disable.'
    )
)
parser.add_argument(
    '--motif_temperature_guard_relax_factor',
    type=float,
    default=1.05,
    help=(
        'When the motif-temperature guard fires, multiply the adaptive effective '
        'temperature by this factor, capped at motif_temperature_start.'
    )
)
parser.add_argument(
    '--motif_temperature_guard_sharpen_factor',
    type=float,
    default=0.995,
    help=(
        'When the motif-temperature guard does not fire, multiply the adaptive '
        'effective temperature by this factor, floored at the scheduled temperature.'
    )
)
parser.add_argument('--rule_prune', type=str2bool, default=False)
parser.add_argument(
    '--protect_unit_relation_motifs_from_pruning',
    type=str2bool,
    default=False,
    help=(
        'With rule_prune=true, restore full cached value rows only for bare '
        'binary-relation rules so their positive adjacency-like motif matrices remain '
        'available as a separate objective group.'
    ),
)
parser.add_argument(
    '--motif_prune_max_values_per_rule',
    type=int,
    default=None,
    help=(
        'Optional extra cap for rule-pruned motif values. When set, each '
        'multi-atom rule keeps only the top N rows according to the pruning '
        'score. Single-atom rules always keep all rows, and full rows are still '
        'cached for rule_prune=False.'
    )
)
parser.add_argument(
    '--motif_batch_size',
    type=int,
    dest="motif_batch_size",
    default=50000,
    help='motif-counting batch size. Only used for multi-graph datasets (QM9). Tune to your VRAM: 8 GB -> 2000 | 16 GB -> 5000 | 24 GB+ -> 30000.'
)
parser.add_argument(
    '--prepare_motif_cache_only',
    type=str2bool,
    default=False,
    help=(
        'Initialize/cache motif rules from the configured FactorBase/MySQL '
        'database and exit before dataset loading, model creation, or training.'
    )
)

#===============================
# Loss arguments
#===============================
parser.add_argument(
    '--alpha_kernel_cost',
    type=float,
    default=1.0,
    help='Deprecated. Reference-compatible training always includes kernel_cost with weight 1.0.'
)
parser.add_argument(
    '--alpha_node_feat',
    type=float,
    default=None,
    help='Weight for node feature reconstruction loss. Defaults: GraphVAE-MM=40, kipf/GraphVAE=1.'
)
parser.add_argument(
    '--alpha_edge_feat',
    type=float,
    default=None,
    help='Weight for edge feature reconstruction loss. Defaults: GraphVAE-MM=40, kipf/GraphVAE=1.'
)
parser.add_argument(
    '--alpha_motif_loss',
    type=float,
    default=None,
    help='Weight for motif loss. Defaults when motif_loss=true: GraphVAE-MM=1, kipf/GraphVAE=0.1; otherwise 0.'
)
parser.add_argument(
    '--use_graphvae_mm_bce_kl_weights',
    type=str2bool,
    default=False,
    help=(
        "Use Kia's GraphVAE-MM dataset-specific adjacency-BCE and KL weights "
        "without enabling GraphVAE-MM graph-statistics kernels. This supports "
        "plain GraphVAE experiments that replace statistics with motif losses."
    )
)
parser.add_argument(
    '--alpha_syntactic_literal_motif_loss',
    type=float,
    default=None,
    help='Optional separate weight for motifs belonging to synthetic-literal rule shapes.'
)
parser.add_argument(
    '--alpha_unit_relation_motif_loss',
    type=float,
    default=None,
    help=(
        'Weight for the optional unit-relation motif group. Defaults to '
        'alpha_motif_loss when that group is enabled.'
    ),
)
parser.add_argument(
    '--alpha_adj_recon',
    type=float,
    default=0.0,
    help='Deprecated. Reference-compatible training already includes adjacency reconstruction inside kernel_cost.'
)

#===============================
# Runtime, output, and evaluation arguments
#===============================
parser.add_argument(
    '--graph_save_path',
    dest="graph_save_path",
    default=None,
    help="the direc to save generated synthatic graphs"
)
parser.add_argument(
    '--run_label',
    type=str,
    default=None,
    help='Readable experiment label written into the run folder.'
)
parser.add_argument(
    '--dataset_cache_dir',
    type=str,
    default=None,
    help='Directory for processed dataset cache files. Defaults to DATASET_CACHE_DIR or cache_datasets/.'
)
parser.add_argument(
    '--disable_dataset_cache',
    type=str2bool,
    default=False,
    help='Read/process raw dataset files without loading or saving processed dataset cache pickles.'
)
parser.add_argument(
    '--motif_cache_dir',
    type=str,
    default=None,
    help='Directory for motif cache pickle files. Defaults to MOTIF_CACHE_DIR or cache_motifs/.'
)
parser.add_argument(
    '-p',
    '--model_path',
    dest="PATH",
    default="model",
    help="a string which determine the path in wich model will be saved"
)
parser.add_argument(
    '--use_gpu',
    dest="UseGPU",
    default=True,
    type=str2bool,
    help="either use GPU or not if availabel"
)
parser.add_argument(
    '--device',
    dest="device",
    default="cuda",
    help="Which device should be used, e.g. cuda, cuda:0, cpu"
)
parser.add_argument(
    '--plot_test_graphs',
    dest="plot_testGraphs",
    default=True,
    type=str2bool,
    help="shall the test set be printed"
)
parser.add_argument(
    '--ideal_evaluation',
    dest="ideal_Evalaution",
    default=False,
    type=str2bool,
    help="if you want to compare the 50%%50 subset of dataset comparison"
)
parser.add_argument(
    '--keep_best_validation_mmd',
    default=True,
    type=str2bool,
    help='Save and use the checkpoint with the lowest validation ranking score.'
)
parser.add_argument(
    '--best_validation_mmd_metric',
    default='table3_priority',
    choices=BEST_VALIDATION_MMD_SCORE_MODES,
    help=(
        'Validation checkpoint score. normalized_table2 averages each Table 2 '
        'metric after dividing by the dataset GraphVAE paper value; '
        'normalized_table2_table3 also includes mmd_rbf divided by the '
        'dataset paper GraphVAE-MM mmd_rbf and (1 - f1_pr) / 0.05; '
        'table3_priority uses 40%% mmd_rbf, 40%% 1 - f1_pr, and '
        '20%% normalized Table 2 metrics; '
        'normalized components use a small denominator floor and cap; '
        'raw_mean averages raw Table 2 MMDs; a metric name tracks only that '
        'metric.'
    )
)
parser.add_argument(
    '--save_validation_checkpoints',
    default=False,
    type=str2bool,
    help='Save a model state_dict at each validation step for post-training resampling.'
)
parser.add_argument(
    '--checkpoint_interval_epochs',
    default=1000,
    type=int,
    help='Save a model state_dict every N epochs; set to 0 to disable periodic epoch checkpoints.'
)
parser.add_argument(
    '--third_party_eval',
    default=True,
    type=str2bool,
    help='Run vendored Random-GIN third-party evaluation after final graph generation.'
)
parser.add_argument(
    '--third_party_eval_repeats',
    type=int,
    default=10,
    help='Number of Random-GIN evaluator repeats for automatic third-party evaluation.'
)
parser.add_argument(
    '--third_party_eval_max_graphs',
    type=int,
    default=1000,
    help='Maximum generated/reference graphs used by automatic third-party evaluation.'
)
parser.add_argument(
    '--third_party_eval_seed',
    type=int,
    default=0,
    help='Random seed used by automatic third-party evaluation.'
)
parser.add_argument(
    '--third_party_eval_device',
    choices=['same', 'auto', 'cpu', 'cuda'],
    default='same',
    help='Device for automatic third-party evaluation. same maps the training device to cpu/cuda.'
)
parser.add_argument(
    '--third_party_eval_structural_features',
    default=True,
    type=str2bool,
    help='Use Kia-style structural node features in automatic third-party evaluation.'
)
parser.add_argument(
    '--third_party_eval_json_filename',
    default=THIRD_PARTY_EVAL_JSON_FILENAME,
    help='Per-run JSON filename written by automatic third-party evaluation.'
)
parser.add_argument(
    '--third_party_eval_strict',
    default=True,
    type=str2bool,
    help='Fail the run if automatic third-party evaluation fails.'
)
parser.set_defaults(tiny_overfit=False)
parser.add_argument(
    '--tiny_overfit',
    dest='tiny_overfit',
    action='store_true',
    help='Use a tiny fixed training subset, disable shuffling, and train with one fixed batch.'
)
parser.add_argument(
    '--no-tiny_overfit',
    dest='tiny_overfit',
    action='store_false',
    help='Disable tiny-overfit debug mode and run full training.'
)
parser.add_argument(
    '--tiny_overfit_size',
    type=int,
    default=32,
    help='Number of training graphs to keep in --tiny_overfit mode.'
)
parser.add_argument('--interactive', action='store_true', default=False)
parser.add_argument(
    '--sanity_check',
    action='store_true',
    default=True,
    help='Run sanity check and print readable results.'
)
parser.add_argument(
    '--sanity_check_only',
    action='store_true',
    default=True,
    help='Run sanity check and exit before training.'
)


config_args, _ = parser.parse_known_args()
if config_args.config is not None:
    valid_config_keys = {action.dest for action in parser._actions}
    parser.set_defaults(**load_config_defaults(config_args.config, valid_config_keys))

args = parser.parse_args()
ensure_deterministic_python_hash_seed(args.seed, args.deterministic)
configure_reproducibility(
    args.seed,
    deterministic=args.deterministic,
    deterministic_warn_only=args.deterministic_warn_only,
)
args.model = normalize_model_name(args.model)

#===============================
# Data settings
#===============================
dataset = args.dataset  # possible choices are: cora, citeseer, karate, pubmed, DBIS
use_feature = args.use_feature
bfs_ordering = args.bfsOrdering
bfs_strategy = args.bfs_strategy
directed = args.directed
database_name = args.database_name
graph_type = args.graph_type
graph_index_start = args.graph_index_start
graph_index_end = args.graph_index_end
data_dir = args.data_dir
lobster_feature_schema = args.lobster_feature_schema
tu_attribute_bins = int(args.tu_attribute_bins)
tu_max_nodes = args.tu_max_nodes
if tu_attribute_bins < 2:
    raise ValueError("tu_attribute_bins must be at least 2.")
if tu_max_nodes is not None and tu_max_nodes < 1:
    raise ValueError("tu_max_nodes must be positive when provided.")
split_mode = args.split_mode
split_plan = resolve_split_plan(
    split_mode=split_mode,
    train_fraction_arg=args.train_fraction,
    val_fraction_arg=args.val_fraction,
    split_seed=args.split_seed,
)
split_seed = split_plan["split_seed"]
split_train_fraction = split_plan["train_fraction"]
split_val_fraction = split_plan["val_fraction"]

#===============================
# Model settings
#===============================
model_name = args.model
encoder_type = args.encoder_type
graphEmDim = args.graphEmDim
decoder_type = args.decoder
beta = args.beta
correct_reparameterization = args.correct_reparameterization

#===============================
# Experiment settings
#===============================
visulizer_step = args.Vis_step
redraw = args.redraw
task = args.task
epoch_number = args.epoch_number
lr = args.lr
train_batch_size = args.train_batch_size

#===============================
# Motif settings
#===============================
use_motif_loss = args.motif_loss
motif_output_mode = canonicalize_motif_output_mode(args.motif_output_mode)
args.motif_output_mode = motif_output_mode
motif_loss_mode = args.motif_loss_mode
non_literal_motif_output_mode = canonicalize_motif_output_mode(
    args.non_literal_motif_output_mode or motif_output_mode
)
syntactic_literal_motif_output_mode = canonicalize_motif_output_mode(
    args.syntactic_literal_motif_output_mode or motif_output_mode
)
unit_relation_motif_output_mode = (
    canonicalize_motif_output_mode(args.unit_relation_motif_output_mode)
    if args.unit_relation_motif_output_mode is not None
    else None
)
non_literal_motif_loss_mode = (
    args.non_literal_motif_loss_mode or motif_loss_mode
)
syntactic_literal_motif_loss_mode = (
    args.syntactic_literal_motif_loss_mode or motif_loss_mode
)
unit_relation_motif_loss_mode = (
    args.unit_relation_motif_loss_mode or motif_loss_mode
    if unit_relation_motif_output_mode is not None
    else None
)
args.non_literal_motif_output_mode = non_literal_motif_output_mode
args.syntactic_literal_motif_output_mode = syntactic_literal_motif_output_mode
args.non_literal_motif_loss_mode = non_literal_motif_loss_mode
args.syntactic_literal_motif_loss_mode = syntactic_literal_motif_loss_mode
args.unit_relation_motif_output_mode = unit_relation_motif_output_mode
args.unit_relation_motif_loss_mode = unit_relation_motif_loss_mode
motif_histogram_num_bins = int(args.motif_histogram_num_bins)
motif_histogram_smoothing = float(args.motif_histogram_smoothing)
if motif_histogram_num_bins < 2:
    raise ValueError("motif_histogram_num_bins must be at least 2.")
if motif_histogram_smoothing <= 0.0:
    raise ValueError("motif_histogram_smoothing must be greater than zero.")
motif_temperature_start = max(float(args.motif_temperature_start), 1e-3)
motif_temperature_end = max(float(args.motif_temperature_end), 1e-3)
motif_temperature_anneal_start_frac = min(
    max(float(args.motif_temperature_anneal_start_frac), 0.0), 1.0
)
motif_temperature_sharpens = (
    use_motif_loss
    and motif_temperature_end < motif_temperature_start - 1e-12
    and motif_temperature_anneal_start_frac < 1.0
)
if args.motif_temperature_guard_ratio is None:
    motif_temperature_guard_ratio = (
        DEFAULT_TEMP_ANNEAL_GUARD_RATIO if motif_temperature_sharpens else 0.0
    )
else:
    motif_temperature_guard_ratio = max(float(args.motif_temperature_guard_ratio), 0.0)
args.motif_temperature_guard_ratio = motif_temperature_guard_ratio
motif_temperature_guard_relax_factor = max(
    float(args.motif_temperature_guard_relax_factor),
    1.0,
)
motif_temperature_guard_sharpen_factor = min(
    max(float(args.motif_temperature_guard_sharpen_factor), 1e-6),
    1.0,
)
args.motif_temperature_guard_relax_factor = motif_temperature_guard_relax_factor
args.motif_temperature_guard_sharpen_factor = motif_temperature_guard_sharpen_factor
rule_prune = args.rule_prune
motif_batch_size = args.motif_batch_size
prepare_motif_cache_only = args.prepare_motif_cache_only
syntactic_literal_rule_mode = (
    args.syntactic_literal_rule_mode
    if args.use_syntactic_literal_rules
    else 'original'
)
args.syntactic_literal_rule_mode = syntactic_literal_rule_mode
use_syntactic_literal_rules = syntactic_literal_rule_mode != 'original'
args.use_syntactic_literal_rules = use_syntactic_literal_rules

#===============================
# Loss settings
#===============================
alpha_kernel_cost = args.alpha_kernel_cost
alpha_node_feat = resolve_loss_weight(
    args.alpha_node_feat,
    default_feature_loss_weight(model_name),
)
alpha_edge_feat = resolve_loss_weight(
    args.alpha_edge_feat,
    default_feature_loss_weight(model_name),
)
alpha_motif_loss = resolve_loss_weight(
    args.alpha_motif_loss,
    default_motif_loss_weight(model_name, use_motif_loss),
)
alpha_syntactic_literal_motif_loss = (
    alpha_motif_loss
    if args.alpha_syntactic_literal_motif_loss is None
    else float(args.alpha_syntactic_literal_motif_loss)
)
alpha_unit_relation_motif_loss = (
    0.0
    if unit_relation_motif_output_mode is None
    else (
        alpha_motif_loss
        if args.alpha_unit_relation_motif_loss is None
        else float(args.alpha_unit_relation_motif_loss)
    )
)
if (
    unit_relation_motif_output_mode is None
    and args.alpha_unit_relation_motif_loss is not None
):
    raise ValueError(
        "alpha_unit_relation_motif_loss requires "
        "unit_relation_motif_output_mode."
    )
alpha_adj_recon = args.alpha_adj_recon
use_graphvae_mm_bce_kl_weights = args.use_graphvae_mm_bce_kl_weights
args.alpha_node_feat = alpha_node_feat
args.alpha_edge_feat = alpha_edge_feat
args.alpha_motif_loss = alpha_motif_loss
args.alpha_syntactic_literal_motif_loss = alpha_syntactic_literal_motif_loss
args.alpha_unit_relation_motif_loss = alpha_unit_relation_motif_loss

#===============================
# Runtime, output, and evaluation settings
#===============================
device = args.device
use_gpu = args.UseGPU
graph_save_path = args.graph_save_path
run_label = args.run_label
dataset_cache_dir = args.dataset_cache_dir
disable_dataset_cache = args.disable_dataset_cache
motif_cache_dir = args.motif_cache_dir
PATH = args.PATH  # the dir to save the with the best performance on validation data
plot_testGraphs = args.plot_testGraphs
ideal_Evalaution = args.ideal_Evalaution
keep_best_validation_mmd = args.keep_best_validation_mmd
best_validation_mmd_metric = args.best_validation_mmd_metric
save_validation_checkpoints = args.save_validation_checkpoints
checkpoint_interval_epochs = max(0, int(args.checkpoint_interval_epochs))
third_party_eval = args.third_party_eval
interactive = args.interactive
sanity_check = args.sanity_check
sanity_check_only = args.sanity_check_only
# endregion
#====================================================================================

if data_dir is not None:
    os.environ["DATA_DIR"] = str(Path(data_dir).expanduser())
if dataset_cache_dir is not None:
    os.environ["DATASET_CACHE_DIR"] = str(Path(dataset_cache_dir).expanduser())
if motif_cache_dir is not None:
    os.environ["MOTIF_CACHE_DIR"] = str(Path(motif_cache_dir).expanduser())
if dataset == "TRIANGULAR_GRID" and database_name == "triangular_grid_undir_feat_snap_ce92ed":
    os.environ["TRIANGULAR_GRID_FEATURE_SCHEMA"] = "legacy"

if prepare_motif_cache_only:
    motif_pickle_path = get_motif_pickle_path(database_name, args)
    print("[PrepareMotifCache] Preparing motif cache only.")
    print(f"[PrepareMotifCache] database_name={database_name}")
    print(f"[PrepareMotifCache] syntactic_literal_rule_mode={syntactic_literal_rule_mode}")
    print(f"[PrepareMotifCache] motif_cache_path={motif_pickle_path}")
    RuleBasedMotifStore(database_name=database_name, args=args)
    print(f"[PrepareMotifCache] Done: {motif_pickle_path}")
    sys.exit(0)


#====================================================================================
# region Tiny overfit debug mode
tiny_overfit = args.tiny_overfit
tiny_overfit_size = args.tiny_overfit_size
if tiny_overfit:
    # Tiny overfit is a deterministic debug preset for checking whether the
    # current loss can be overfit on a tiny fixed subset. The model is later
    # switched to AutoEncoder=True below, so latent sampling is disabled here.
    tiny_overfit_size = 1
    epoch_number = min(int(epoch_number), 1000)
    visulizer_step = min(int(visulizer_step), 100)

    use_motif_loss = True
    args.motif_loss = True
    ideal_Evalaution = False
    args.ideal_Evalaution = False
    plot_testGraphs = False
    args.plot_testGraphs = False
    redraw = False
    task = 'debug'
    args.task = task
    args.tiny_overfit_size = tiny_overfit_size
    args.epoch_number = epoch_number
    args.Vis_step = visulizer_step
    print(f"[TinyOverfit] Auto preset: size={tiny_overfit_size}, epochs={epoch_number}, "
          f"vis_step={visulizer_step}, motif_loss={use_motif_loss}, task={task}")
    logging.info(f"[TinyOverfit] Auto preset: size={tiny_overfit_size}, epochs={epoch_number}, "
                 f"vis_step={visulizer_step}, motif_loss={use_motif_loss}, task={task}")
# end of tiny overfit debug mode
# endregion
#====================================================================================


if graph_save_path is None:
    run_name = "MMD_" + encoder_type + "_" + decoder_type + "_" + dataset + "_" + task + "_" + model_name + "BFS" + str(
        bfs_ordering) + str(epoch_number) + str(time.time())
    graph_save_dir = Path("runs") / run_name
else:
    graph_save_dir = Path(graph_save_path)
    run_name = graph_save_dir.name if graph_save_dir.name else "run_" + str(int(time.time()))

seed_dir_name = f"seed_{args.seed}"
if graph_save_dir.name != seed_dir_name:
    graph_save_dir = graph_save_dir / seed_dir_name
    run_name = f"{run_name}_{seed_dir_name}"

graph_save_dir.mkdir(parents=True, exist_ok=True)
graph_save_path = str(graph_save_dir) + "/"
generated_graph_train_dir = graph_save_dir / "generated_graph_train"
generated_graph_train_dir.mkdir(parents=True, exist_ok=True)
run_log_path = graph_save_dir / "train.log"
run_mmd_log_path = graph_save_dir / "mmd.log"
best_validation_mmd_model_path = graph_save_dir / "best_validation_mmd_model"
best_validation_mmd_metadata_path = graph_save_dir / "best_validation_mmd.json"
best_validation_mmd_score = float("inf")
best_validation_mmd_metadata = None
write_run_reproducibility_files(graph_save_dir, args, run_label)

# maybe to the beest way
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=str(run_log_path), filemode='w', level=logging.INFO)

# **********************************************************************
# setting; general setting and hyper-parameters for each dataset
# region general settings
print("KernelVGAE SETING: " + str(args))
logging.info("KernelVGAE SETING: " + str(args))

kernl_type = []
#---------------------------------------------------------------------
if model_name == "KernelAugmentedWithTotalNumberOfTriangles" or model_name == "GraphVAE-MM":
    kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist", "TotalNumberOfTriangles"]
    if dataset=="mnist":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 10, 50]
        step_num = 5
    if dataset=="zinc":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 10, 50]
        step_num = 5
    if dataset == "large_grid":
        step_num = 5 # s in s-step transition
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 20, 100]
    elif dataset == "ogbg-molbbbp":
        # leision study
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 1500]
        alpha = [0, 0, 0, 0, 0, 1, 1, 0, 40, 1500]
        alpha = [0, 0, 0, 0, 0, 0, 0, 1, 40, 1500]
        # -----------------------------------------
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 40, 1500]
        step_num = 5
    elif dataset == "IMDBBINARY":
        alpha = [ 1, 1, 1, 1, 1, 1, 2, 50]
        step_num = 5
    elif dataset == "QM9":
        step_num = 2
        alpha = [ 1, 1, 1, 1, 1, 20, 200]
    elif dataset == "PTC":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    elif dataset =="MUTAG":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 4, 60]
    elif dataset =="PVGAErandomGraphs":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 4, 1]
    elif dataset == "FIRSTMM_DB":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 100]
    elif dataset == "DD":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 1000]
    elif dataset == "GRID":
        #first 8 values, all 1: weights for the GraphVAE-MM graph-statistic/kernel losses
        #50: weight for adjacency reconstruction BCE
        #2000: weight for KL divergence
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]
    elif dataset in {"PROTEINS", "AIDS", "ENZYMES", "ENZYMEZ"}:
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]

    elif dataset == "LOBSTER":
        step_num = 5
        # leision study
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 2000]  # degree
        alpha = [0, 0, 0, 0, 0, 1, 1, 0, 40, 2000]  # degree
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 2000]
        # -------------------------------------------------
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 40, 2000]
    elif dataset == "wheel_graph":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 3000000, 20000 * 50000]
    elif dataset == "TRIANGULAR_GRID":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]
    elif dataset == "tree":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]
#---------------------------------------------------------------------

elif model_name == "kipf" or model_name == "graphVAE":
    alpha = [1, 1]
    step_num = 0

# Preserve Kia's dataset-specific base-VAE regularization while allowing the
# GraphVAE-MM statistics above to be replaced by motif/feature objectives.
# This modifies only adjacency BCE and KL; kernl_type remains empty for kipf.
alpha = apply_kia_bce_kl_weights(
    alpha,
    dataset,
    use_graphvae_mm_bce_kl_weights,
)

AutoEncoder = False

# Make sure if we are using tiny overfit debug mode, we are actually training an autoencoder (no kernel loss).
if tiny_overfit:
    AutoEncoder = True

if AutoEncoder == True:
    alpha[-1] = 0

if beta != None:
    alpha[-1] = beta

latent_mode = "AE" if AutoEncoder else "VAE"
print("latent_mode:" + latent_mode)
print(
    "reparameterization:"
    + ("correct_std" if correct_reparameterization else "legacy_variance")
)
print("kernl_type:" + str(kernl_type))
print("alpha: " + str(alpha) + " num_step:" + str(step_num))
print(
    "loss_weights:"
    + " base=kernel_cost,"
      f" node_feat={alpha_node_feat},"
      f" edge_feat={alpha_edge_feat},"
      f" motif={alpha_motif_loss},"
      f" syntactic_literal_motif={alpha_syntactic_literal_motif_loss},"
      f" unit_relation_motif={alpha_unit_relation_motif_loss},"
      f" kia_bce_kl={use_graphvae_mm_bce_kl_weights},"
      f" adjacency_bce={alpha[-2]},"
      f" kl={alpha[-1]}"
)
print("motif_loss_mode:" + str(motif_loss_mode))
print(
    "motif_group_objectives:"
    + f" non_literal={non_literal_motif_output_mode}/{non_literal_motif_loss_mode},"
      f" syntactic_literal={syntactic_literal_motif_output_mode}/"
      f"{syntactic_literal_motif_loss_mode},"
      f" unit_relation={unit_relation_motif_output_mode}/"
      f"{unit_relation_motif_loss_mode}"
)
print("syntactic_literal_rule_mode:" + str(syntactic_literal_rule_mode))
print(
    "motif_temperature_anneal:"
    + f"start={motif_temperature_start}, end={motif_temperature_end}, "
      f"start_frac={motif_temperature_anneal_start_frac}, "
      f"guard_ratio={motif_temperature_guard_ratio}, "
      f"guard_relax_factor={motif_temperature_guard_relax_factor}, "
      f"guard_sharpen_factor={motif_temperature_guard_sharpen_factor}"
)

logging.info("latent_mode:" + latent_mode)
logging.info(
    "reparameterization:"
    + ("correct_std" if correct_reparameterization else "legacy_variance")
)
logging.info("kernl_type:" + str(kernl_type))
logging.info("alpha: " + str(alpha) + " num_step:" + str(step_num))
logging.info(
    "loss_weights:"
    + " base=kernel_cost,"
      f" node_feat={alpha_node_feat},"
      f" edge_feat={alpha_edge_feat},"
      f" motif={alpha_motif_loss},"
      f" syntactic_literal_motif={alpha_syntactic_literal_motif_loss},"
      f" unit_relation_motif={alpha_unit_relation_motif_loss},"
      f" kia_bce_kl={use_graphvae_mm_bce_kl_weights},"
      f" adjacency_bce={alpha[-2]},"
      f" kl={alpha[-1]}"
)
logging.info("motif_loss_mode:" + str(motif_loss_mode))
logging.info(
    "motif_group_objectives:"
    + f" non_literal={non_literal_motif_output_mode}/{non_literal_motif_loss_mode},"
      f" syntactic_literal={syntactic_literal_motif_output_mode}/"
      f"{syntactic_literal_motif_loss_mode},"
      f" unit_relation={unit_relation_motif_output_mode}/"
      f"{unit_relation_motif_loss_mode}"
)
logging.info("syntactic_literal_rule_mode:" + str(syntactic_literal_rule_mode))
logging.info(
    "motif_temperature_anneal:"
    + f"start={motif_temperature_start}, end={motif_temperature_end}, "
      f"start_frac={motif_temperature_anneal_start_frac}, "
      f"guard_ratio={motif_temperature_guard_ratio}, "
      f"guard_relax_factor={motif_temperature_guard_relax_factor}, "
      f"guard_sharpen_factor={motif_temperature_guard_sharpen_factor}"
)

  # with is propertion to revese of this value;

device = torch.device(device if torch.cuda.is_available() and use_gpu else "cpu")
print("the selected device is :", device)
logging.info("the selected device is :" + str(device))

# setting the plots legend
functions = ["Accuracy", "loss"]
if model_name == "kernel" or model_name == "KernelAugmentedWithTotalNumberOfTriangles" or model_name == "GraphVAE-MM":
    functions.extend(["Kernel" + str(i) for i in range(step_num)])
    functions.extend(kernl_type[1:])

if model_name == "TrianglesOfEachNode":
    functions.extend(kernl_type)

if model_name == "ThreeStepPath":
    functions.extend(kernl_type)

if model_name == "TotalNumberOfTriangles":
    functions.extend(kernl_type)

functions.append("Binary_Cross_Entropy")
functions.append("KL-D")
#endregion
# ========================================================================


pltr = plotter.Plotter(save_to_filepath="kernelVGAE_Log", functions=functions)

synthesis_graphs = {"wheel_graph", "star", "TRIANGULAR_GRID", "DD", "ogbg-molbbbp", "GRID", "small_lobster",
                    "small_grid", "community", "LOBSTER", "ego", "one_grid", "IMDBBINARY", ""}


# region Modules for latent space transformation and upsampling (not used in the current model, but can be useful for future extensions)
class NodeUpsampling(torch.nn.Module):
    def __init__(self, InNode_num, outNode_num, InLatent_dim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num * InLatent_dim, InLatent_dim * outNode_num)

    def forward(self, inTensor, activation=torch.nn.LeakyReLU(0.001)):
        Z = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(Z.reshpe(inTensor.shape[0], -1).permute(0, 2, 1), inTensor)

        return activation(Z)


class LatentMtrixTransformer(torch.nn.Module):
    def __init__(self, InNode_num, InLatent_dim=None, OutLatentDim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num * InLatent_dim, InNode_num * OutLatentDim)

    def forward(self, inTensor, activation=torch.nn.LeakyReLU(0.001)):
        Z = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(inTensor, Z.reshpe(inTensor.shape[-1], -1))

        return activation(Z)
#endregion

# ============================================================================

#region Testing and evaluation and helper functions
def test_(number_of_samples, model, graph_size, path_to_save_g, remove_self=True, save_graphs=True):
    import os
    if not os.path.exists(path_to_save_g):
        os.makedirs(path_to_save_g)
    # model.eval()
    generated_graph_list = []
    if not os.path.isdir(path_to_save_g):
        os.makedirs(path_to_save_g)
    k = 0
    for g_size in graph_size:
        for j in range(number_of_samples):
            z = torch.tensor(numpy.random.normal(size=[1, model.embeding_dim]))
            z = torch.randn_like(z)
            start_time = time.time()

            adj_logit = model.decode(z.to(device).float())
            print("--- %s seconds ---" % (time.time() - start_time))
            logging.info("--- %s seconds ---" % (time.time() - start_time))
            reconstructed_adj = torch.sigmoid(adj_logit)
            sample_graph = reconstructed_adj[0].cpu().detach().numpy()
            # sample_graph = sample_graph[:g_size,:g_size]
            sample_graph[sample_graph >= 0.5] = 1
            sample_graph[sample_graph < 0.5] = 0
            G = nx.from_numpy_array(sample_graph)
            # generated_graph_list.append(G)
            f_name = path_to_save_g + str(k) + str(g_size) + str(j) + dataset
            k += 1
            # plot and save the generated graph
            # plotter.plotG(G, "generated" + dataset, file_name=f_name)
            if remove_self:
                G.remove_edges_from(nx.selfloop_edges(G))

            G.remove_nodes_from(list(nx.isolates(G)))
            generated_graph_list.append(G)
            if save_graphs:
                plotter.plotG(G, "generated" + dataset, file_name=f_name + "_ConnectedComponnents")
    # ======================================================
    # save nx files
    if save_graphs:
        nx_f_name = path_to_save_g + "_" + dataset + "_" + decoder_type + "_" + model_name + "_" + task
        with open(nx_f_name, 'wb') as f:
            pickle.dump(generated_graph_list, f)
    # # ======================================================
    return generated_graph_list


def EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated=True, _f_name=None, onlyTheBigestConCom = True):
    generated_graphs = test_(1, model, [x.shape[0] for x in test_list_adj], graph_save_path, save_graphs=Save_generated)
    graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs]
    if (onlyTheBigestConCom==False):
        if Save_generated:
            np.save(graph_save_path + 'generatedGraphs_adj_' + str(_f_name) + '.npy',
                    np.array(graphs_to_writeOnDisk, dtype=object),
                    allow_pickle=True)


            logging.info(mmd_eval(generated_graphs, [nx.from_numpy_array(graph.toarray()) for graph in test_list_adj]))
    print("====================================================")
    logging.info("====================================================")

    print("result for subgraph with maximum connected componnent")
    logging.info("result for subgraph with maximum connected componnent")
    generated_graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in generated_graphs if
                        not nx.is_empty(G)]

    statistic_   = mmd_eval(generated_graphs, [nx.from_numpy_array(graph.toarray()) for graph in test_list_adj], diam=True)
    # if writeThem_in!=None:
    #     with open(writeThem_in+'MMD.log', 'w') as f:
    #         f.write(statistic_)
    logging.info(statistic_)
    if Save_generated:
        graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs]
        np.save(graph_save_path + 'Single_comp_generatedGraphs_adj_' + str(_f_name) + '.npy',
                np.array(graphs_to_writeOnDisk, dtype=object),
                allow_pickle=True)

        graphs_to_writeOnDisk = [G.toarray() for G in test_list_adj]
        np.save(graph_save_path + 'testGraphs_adj_.npy',
                np.array(graphs_to_writeOnDisk, dtype=object),
                allow_pickle=True)
    return  statistic_


def get_subGraph_features(org_adj, subgraphs_indexes, kernel_model):
    subgraphs = []
    target_kelrnel_val = None

    for i in range(len(org_adj)):
        subGraph = org_adj[i]
        if subgraphs_indexes != None:
            subGraph = subGraph[:, subgraphs_indexes[i]]
            subGraph = subGraph[subgraphs_indexes[i], :]
        # Converting sparse matrix to sparse tensor
        subGraph = torch.tensor(subGraph.todense())
        subgraphs.append(subGraph)
    subgraphs = torch.stack(subgraphs).to(device)

    if kernel_model != None:
        target_kelrnel_val = kernel_model(subgraphs)
        target_kelrnel_val = [val.to("cpu") for val in target_kelrnel_val]
    subgraphs = subgraphs.to("cpu")
    torch.cuda.empty_cache()
    return target_kelrnel_val, subgraphs


# the code is a hard copy of https://github.com/orybkin/sigma-vae-pytorch
def log_guss(mean, log_std, samples):
    return 0.5 * torch.pow((samples - mean) / log_std.exp(), 2) + log_std + 0.5 * np.log(2 * np.pi)


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor
#endregion

#============================================================================
# region Loss functions for node and edge feature decoders
def compute_true_node_feat_loss(node_feat_logits, target_node_onehot, true_node_num):
    """
    Compute node feature BCE loss on real nodes only (ignore padded nodes).

    Expected shapes:
    - node_feat_logits:  (B, N_max, D)
    - target_node_onehot: (B, N_max, D)
    - true_node_num: length-B iterable/tensor with true node counts per graph

    Returns:
    - Scalar tensor: mean BCE over valid node-feature entries only.
    """
    # Validate tensor ranks early so a shape mismatch fails with a clear error.
    if node_feat_logits.ndim != 3 or target_node_onehot.ndim != 3:
        raise ValueError(
            "node_feat_logits and target_node_onehot must be 3D tensors with shape (B, N_max, D)."
        )

    # Validate that predicted and target tensors are aligned elementwise.
    if node_feat_logits.shape != target_node_onehot.shape:
        raise ValueError(
            f"Shape mismatch: logits {tuple(node_feat_logits.shape)} vs targets {tuple(target_node_onehot.shape)}."
        )

    # Read batch and padded-node dimensions.
    batch_size, max_nodes, node_feat_dim = node_feat_logits.shape

    # Convert node counts to a tensor on the same device as logits.
    # This supports Python lists, NumPy arrays, or torch tensors as input.
    true_node_num = torch.as_tensor(true_node_num, device=node_feat_logits.device)

    # Ensure we have exactly one node-count value per graph in the batch.
    if true_node_num.ndim != 1 or true_node_num.numel() != batch_size:
        raise ValueError(
            f"true_node_num must be 1D with length {batch_size}; got shape {tuple(true_node_num.shape)}."
        )

    # Node counts are indices for masking, so cast to long and clamp to valid bounds.
    # Clamping prevents accidental out-of-range values from breaking mask creation.
    true_node_num = true_node_num.long().clamp(min=0, max=max_nodes)

    # Build a boolean mask of shape (B, N_max):
    # True for real nodes [0, true_node_num[b]) and False for padded rows.
    node_positions = torch.arange(max_nodes, device=node_feat_logits.device).unsqueeze(0)
    valid_node_mask = node_positions < true_node_num.unsqueeze(1)

    # Compute elementwise BCE loss (no reduction) so we can apply the node mask manually.
    # Targets are cast to logits dtype/device to avoid mixed-type issues.
    per_entry_loss = F.binary_cross_entropy_with_logits(
        node_feat_logits,
        target_node_onehot.to(device=node_feat_logits.device, dtype=node_feat_logits.dtype),
        reduction='none'
    )

    # Expand node mask to feature dimension: (B, N_max) -> (B, N_max, 1).
    # Multiply to zero out padded-node contributions.
    valid_entry_mask = valid_node_mask.unsqueeze(-1).to(per_entry_loss.dtype)
    masked_loss_sum = (per_entry_loss * valid_entry_mask).sum()

    # Number of valid entries is (#valid nodes) * D.
    # Clamp denominator to avoid division-by-zero if a degenerate batch has zero valid nodes.
    valid_entry_count = (valid_entry_mask.sum() * node_feat_dim).clamp(min=1.0)

    # Return mean loss over valid node-feature entries only.
    return masked_loss_sum / valid_entry_count
def compute_true_edge_feat_loss(edge_feat_logits, target_edge_onehot, true_node_num):
    """
    Compute edge-feature loss using one-hot edge labels on real, existing edges only.

    Why this function exists:
    - Edge feature targets are one-hot encoded over C edge classes.
    - Padded nodes and non-edge pairs should not contribute to edge-feature loss.
    - We therefore treat edge feature prediction as multi-class classification
      (CrossEntropy), but only on positions where an edge actually exists.

    Expected shapes:
    - edge_feat_logits:   (B, C, N_max, N_max)
    - target_edge_onehot: (B, C, N_max, N_max) with one-hot labels on true edges
    - true_node_num:      length-B iterable/tensor with true node counts per graph

    Returns:
    - Scalar tensor: mean cross-entropy on valid existing edges only.
    """
    # Validate tensor ranks and alignment first for clearer runtime errors.
    if edge_feat_logits.ndim != 4 or target_edge_onehot.ndim != 4:
        raise ValueError(
            "edge_feat_logits and target_edge_onehot must be 4D tensors with shape (B, C, N_max, N_max)."
        )
    if edge_feat_logits.shape != target_edge_onehot.shape:
        raise ValueError(
            f"Shape mismatch: logits {tuple(edge_feat_logits.shape)} vs targets {tuple(target_edge_onehot.shape)}."
        )

    # Unpack dimensions.
    batch_size, _, max_nodes, _ = edge_feat_logits.shape

    # Convert and validate true node counts.
    true_node_num = torch.as_tensor(true_node_num, device=edge_feat_logits.device)
    if true_node_num.ndim != 1 or true_node_num.numel() != batch_size:
        raise ValueError(
            f"true_node_num must be 1D with length {batch_size}; got shape {tuple(true_node_num.shape)}."
        )
    true_node_num = true_node_num.long().clamp(min=0, max=max_nodes)

    # Build node-valid mask (B, N_max), then pair-valid mask (B, N_max, N_max).
    # This removes any contribution from padded node rows/cols.
    node_positions = torch.arange(max_nodes, device=edge_feat_logits.device).unsqueeze(0)
    valid_node_mask = node_positions < true_node_num.unsqueeze(1)
    valid_pair_mask = valid_node_mask.unsqueeze(1) & valid_node_mask.unsqueeze(2)

    # Use the one-hot target to detect where an edge actually exists.
    # For non-edges, target one-hot channels are all zeros.
    target_edge_onehot = target_edge_onehot.to(
        device=edge_feat_logits.device,
        dtype=edge_feat_logits.dtype
    )
    edge_exists_mask = target_edge_onehot.sum(dim=1) > 0

    # Final supervision mask: only real-node pairs that correspond to real edges.
    supervision_mask = valid_pair_mask & edge_exists_mask

    # Convert one-hot labels -> class indices for cross-entropy.
    # Cross-entropy is applied per (i, j) pair, then masked.
    target_edge_class = target_edge_onehot.argmax(dim=1).long()
    per_pair_loss = F.cross_entropy(edge_feat_logits, target_edge_class, reduction='none')

    # Average only over supervised edge positions.
    supervision_mask_f = supervision_mask.to(per_pair_loss.dtype)
    masked_loss_sum = (per_pair_loss * supervision_mask_f).sum()
    supervised_count = supervision_mask_f.sum().clamp(min=1.0)

    return masked_loss_sum / supervised_count


#endregion
#
def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, targert_adj, target_kernel_val, log_std, mean, alpha,
                 reconstructed_adj_logit, pos_wight, norm):
    loss = norm * torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(),
                                                                       targert_adj.float(), pos_weight=pos_wight)

    norm = mean.shape[0] * mean.shape[1]
    kl = (1 / norm) * -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(log_std).pow(2))

    acc = (reconstructed_adj.round() == targert_adj).sum() / float(
        reconstructed_adj.shape[0] * reconstructed_adj.shape[1] * reconstructed_adj.shape[2])
    kernel_diff = 0
    each_kernel_loss = []
    log_sigma_values = []
    for i in range(len(target_kernel_val)):
        log_sigma = ((reconstructed_kernel_val[i] - target_kernel_val[i]) ** 2).mean().sqrt().log()
        log_sigma = softclip(log_sigma, -6)
        log_sigma_values.append(log_sigma.detach().cpu().item())
        step_loss = log_guss(target_kernel_val[i], log_sigma, reconstructed_kernel_val[i]).mean()
        each_kernel_loss.append(step_loss.cpu().detach().numpy() * alpha[i])
        kernel_diff += step_loss * alpha[i]

    kernel_diff += loss * alpha[-2]
    kernel_diff += kl * alpha[-1]
    each_kernel_loss.append((loss * alpha[-2]).item())
    each_kernel_loss.append((kl * alpha[-1]).item())
    return kl, loss, acc, kernel_diff, each_kernel_loss,log_sigma_values


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


# test_(5, "results/multiple graph/cora/model" , [x**2 for x in range(5,10)])





#====================================================================================
# load the data
#region Load the data

dataset_cache_metadata = build_dataset_cache_metadata(
    dataset=dataset,
    split_mode=split_mode,
    bfs_strategy=bfs_strategy,
    split_plan=split_plan,
    feature_schema=(
        f"lobster-{lobster_feature_schema}"
        if dataset == "LOBSTER"
        else (
            f"tu-quantile{tu_attribute_bins}-max{tu_max_nodes or 'all'}"
            if dataset.upper() in {"AIDS", "ENZYMES", "ENZYMEZ"}
            else (
                "gin-node-label-v2"
                if dataset.upper() in {"MUTAG", "PTC"}
                else "default"
            )
        )
    ),
)
use_cache = not disable_dataset_cache
cache_path = None
if use_cache:
    dataset_cache_root = Path(
        os.environ.get("DATASET_CACHE_DIR", "cache_datasets")
    ).expanduser()
    dataset_cache_root.mkdir(parents=True, exist_ok=True)
    cache_name = build_dataset_cache_name(dataset_cache_metadata)
    cache_path = dataset_cache_root / cache_name
else:
    print("[Cache] Dataset cache disabled. Running data pipeline from raw data.")
    logging.info("[Cache] Dataset cache disabled. Running data pipeline from raw data.")

self_for_none = True
if (decoder_type) in ("FCdecoder"): 
    self_for_none = True
    
if use_cache and cache_path.exists():
    print(f"[Cache] Loading '{dataset}' from {cache_path}")
    logging.info(f"[Cache] Loading '{dataset}' from {cache_path}")
    with open(cache_path, "rb") as _f:
        _cache = pickle.load(_f)
    validate_dataset_cache_metadata(_cache, dataset_cache_metadata, cache_path)

    list_adj          = _cache["list_adj"]
    list_x            = _cache["list_x"]
    list_label        = _cache["list_label"]
    list_node_feature = _cache["list_node_feature"]
    list_edge_feature = _cache["list_edge_feature"]
    node_feature_info = _cache["node_feature_info"]
    edge_feature_info = _cache["edge_feature_info"]
    list_node_onehot  = _cache["list_node_onehot"]
    list_edge_onehot  = _cache["list_edge_onehot"]
    node_onehot_info  = _cache["node_onehot_info"]
    edge_onehot_info  = _cache["edge_onehot_info"]
    split_mode        = _cache.get("split_mode", split_mode)
    args.split_mode   = split_mode

    test_list_adj     = _cache["test_list_adj"]
    val_adj           = _cache["val_adj"]
    list_graphs       = _cache["list_graphs"]
    list_test_graphs  = _cache["list_test_graphs"]

    if not _cache["single_graph"]:
        list_x_train     = _cache["list_x_train"]
        list_x_test      = _cache["list_x_test"]
        list_label_train = _cache["list_label_train"]
        list_label_test  = _cache["list_label_test"]
        list_noh_train   = _cache["list_noh_train"]
        list_noh_test    = _cache["list_noh_test"]
        list_eoh_train   = _cache["list_eoh_train"]
        list_eoh_test    = _cache["list_eoh_test"]

else:
    print(f"[Cache] No cache found for '{dataset}'. Running data pipeline ...")
    logging.info(f"[Cache] No cache found for '{dataset}'. Running data pipeline ...")

    (list_adj, list_x, list_label,
     list_node_feature, list_edge_feature,
     node_feature_info, edge_feature_info) = list_graph_loader(
         dataset,
         return_labels=True,
         lobster_feature_schema=lobster_feature_schema,
         tu_attribute_bins=tu_attribute_bins,
         tu_max_nodes=tu_max_nodes,
     )

    # list_adj   = list_adj[:400]
    # list_x     = list_x[:400]
    # list_label = list_label[:400]

    if bfs_ordering:
        bfs_reorder_fn = BFS if bfs_strategy == "legacy_first_component" else BFS_all_components
        print("[BFS] Using {} ordering.".format(
            "legacy single-component BFS" if bfs_strategy == "legacy_first_component" else "all-components BFS"
        ))

        list_adj, list_node_feature, list_edge_feature = bfs_reorder_fn(
            list_adj, list_node_feature, list_edge_feature
        )

    list_node_onehot, list_edge_onehot, node_onehot_info, edge_onehot_info = \
        build_onehot_features(list_node_feature, list_edge_feature, list_adj,
                              node_feature_info, edge_feature_info)

    # list_adj, list_x, list_label = list_graph_loader(dataset, return_labels=True, _max_list_size=80)
    # list_adj, _ = permute(list_adj, None)

    is_single_graph = len(list_adj) == 1

    if is_single_graph:
        test_list_adj = list_adj.copy()
        val_adj       = test_list_adj.copy()
        list_graphs   = Datasets(list_adj, self_for_none, list_x, None)
        list_test_graphs = Datasets(test_list_adj, self_for_none, list_x, None,
                                    Max_num=list_graphs.max_num_nodes,
                                    set_diag_of_isol_Zer=False)
    else:
        # MUTAG/PTC can place their single largest graph in the held-out split
        # (PTC reaches 109 nodes while this seed's training maximum is 23).
        # Use the full-dataset maximum only for these newly integrated datasets
        # so held-out feature tensors always fit without changing legacy runs.
        max_size = (
            max(int(adjacency.shape[0]) for adjacency in list_adj)
            if dataset.upper() in {"MUTAG", "PTC"}
            else None
        )
        # list_label = None

        if split_mode == "paper_70_10_20":
            (list_adj,         val_adj,         test_list_adj,
             list_x_train,     list_x_val,      list_x_test,
             list_label_train, list_label_val,  list_label_test,
             list_noh_train,   list_noh_val,    list_noh_test,
             list_eoh_train,   list_eoh_val,    list_eoh_test) = data_split_three_way(
                graph_lis        = list_adj,
                list_x           = list_x,
                list_label       = list_label,
                list_node_onehot = list_node_onehot,
                list_edge_onehot = list_edge_onehot,
                train_fraction   = split_train_fraction,
                val_fraction     = split_val_fraction,
                seed             = split_seed,
            )
        else:
            (list_adj,         test_list_adj,
             list_x_train,     list_x_test,
             list_label_train, list_label_test,
             list_noh_train,   list_noh_test,
             list_eoh_train,   list_eoh_test) = data_split(
                graph_lis        = list_adj,
                list_x           = list_x,
                list_label       = list_label,
                list_node_onehot = list_node_onehot,
                list_edge_onehot = list_edge_onehot,
                train_fraction   = split_train_fraction,
                seed             = split_seed,
            )
            list_x_val = None
            list_label_val = None
            list_noh_val = None
            list_eoh_val = None
            val_adj = list_adj[:int(len(test_list_adj))]

        labels_for_train = list_label_train if split_mode == "paper_70_10_20" else list_label
        list_graphs = Datasets(list_adj, self_for_none, list_x_train, labels_for_train,
                               Max_num=max_size, set_diag_of_isol_Zer=False,
                               list_node_onehot=list_noh_train,
                               list_edge_onehot=list_eoh_train)
        list_test_graphs = Datasets(test_list_adj, self_for_none, list_x_test, list_label_test,
                                    Max_num=list_graphs.max_num_nodes, set_diag_of_isol_Zer=False,
                                    list_node_onehot=list_noh_test,
                                    list_edge_onehot=list_eoh_test)
        if plot_testGraphs:
            print("printing the test set...")
            # for i, G in enumerate(test_list_adj):
            #     G = nx.from_numpy_array(G.toarray())
            #     plotter.plotG(G, graph_save_path+"_test_graph" + str(i))

    _cache = {
        "list_adj":          list_adj,
        "list_x":            list_x,
        "list_label":        list_label,
        "list_node_feature": list_node_feature,
        "list_edge_feature": list_edge_feature,
        "node_feature_info": node_feature_info,
        "edge_feature_info": edge_feature_info,
        "list_node_onehot":  list_node_onehot,
        "list_edge_onehot":  list_edge_onehot,
        "node_onehot_info":  node_onehot_info,
        "edge_onehot_info":  edge_onehot_info,
        "single_graph":      is_single_graph,
        "test_list_adj":     test_list_adj,
        "val_adj":           val_adj,
        "list_graphs":       list_graphs,
        "list_test_graphs":  list_test_graphs,
        "self_for_none":     self_for_none,
        "split_mode":        split_mode,
        "cache_metadata":    dataset_cache_metadata,
    }

    if not is_single_graph:
        _cache.update({
            "list_x_train":     list_x_train,
            "list_x_test":      list_x_test,
            "list_label_train": list_label_train,
            "list_label_test":  list_label_test,
            "list_x_val":       list_x_val,
            "list_label_val":   list_label_val,
            "list_noh_train":   list_noh_train,
            "list_noh_test":    list_noh_test,
            "list_noh_val":     list_noh_val,
            "list_eoh_train":   list_eoh_train,
            "list_eoh_test":    list_eoh_test,
            "list_eoh_val":     list_eoh_val,
        })

    if use_cache:
        print(f"[Cache] Saving to {cache_path} ...")
        logging.info(f"[Cache] Saving to {cache_path} ...")
        with open(cache_path, "wb") as _f:
            pickle.dump(_cache, _f)
        print("[Cache] Saved successfully.")
        logging.info("[Cache] Saved successfully.")
    else:
        print("[Cache] Disabled; processed dataset was not saved.")
        logging.info("[Cache] Disabled; processed dataset was not saved.")

    # Keep fixed validation and test references distinct in the run folder.
    # EvalTwoSet historically writes both under testGraphs_adj_.npy, so final
    # test evaluation otherwise overwrites the validation data required for
    # leakage-free post-training checkpoint selection.
    np.save(
        graph_save_path + "validationGraphs_adj_.npy",
        np.array([graph.toarray() for graph in val_adj], dtype=object),
        allow_pickle=True,
    )
    np.save(
        graph_save_path + "heldoutTestGraphs_adj_.npy",
        np.array([graph.toarray() for graph in test_list_adj], dtype=object),
        allow_pickle=True,
    )

#endregion
#====================================================================================


#====================================================================================
# Tiny-overfit mode: keep only a small fixed training subset.
# region Tiny-overfit mode
if tiny_overfit:
    keep_n = max(1, min(int(tiny_overfit_size), len(list_graphs.list_adjs)))
    list_graphs = Datasets(
        list_graphs.list_adjs[:keep_n],
        self_for_none,
        list_graphs.list_Xs[:keep_n] if list_graphs.list_Xs is not None else None,
        list_graphs.labels[:keep_n] if list_graphs.labels is not None else None,
        Max_num=list_graphs.max_num_nodes,
        set_diag_of_isol_Zer=list_graphs.set_diag_of_isol_Zer,
        list_node_onehot=(list_graphs.list_node_onehot[:keep_n]
                          if list_graphs.list_node_onehot is not None else None),
        list_edge_onehot=(list_graphs.list_edge_onehot[:keep_n]
                          if list_graphs.list_edge_onehot is not None else None),
    )
    train_batch_size = keep_n
    print(f"[TinyOverfit] Enabled: using {keep_n} fixed training graphs, "
          f"train_batch_size={train_batch_size}, shuffle=off")
    logging.info(f"[TinyOverfit] Enabled: using {keep_n} fixed training graphs, "
                 f"train_batch_size={train_batch_size}, shuffle=off")
else:
    print(f"[TrainingData] Full training set enabled: using {len(list_graphs.list_adjs)} graphs, "
          f"train_batch_size={train_batch_size}, shuffle=on")
    logging.info(f"[TrainingData] Full training set enabled: using {len(list_graphs.list_adjs)} graphs, "
                 f"train_batch_size={train_batch_size}, shuffle=on")
#endregion
#====================================================================================


#====================================================================================
#region Motif Loss Setup: build motif store and canonical full-matrix targets
# Every group-specific representation is derived later from these same targets.
motif_group_objectives = []
motif_training_uses_only_total_counts = False
if use_motif_loss:
    # Initializes the motif rule store (RuleBasedMotifStore).
    RuleBasedMotifStore(database_name=database_name, args=args) 

    # Builds the dataset to count on (train only, or train+test in sanity mode).
    if sanity_check or sanity_check_only:
        remove_self_loops(list_graphs)
        remove_self_loops(list_test_graphs)
        dataa = merge_datasets(list_graphs, list_test_graphs)  
    else :
        dataa = merge_datasets(list_graphs)

    # Creates a relational motif counter and wraps data for counting on CUDA.
    motif_counter = RelationalMotifCounter(database_name=database_name, args=args)
    if use_syntactic_literal_rules:
        print(
            "SYNTACTIC LITERAL MOTIF MASK:"
            + f" rules={len(motif_counter.syntactic_literal_rule_indices)},"
              f" motif_entries={motif_counter.num_syntactic_literal_motifs}/"
              f"{motif_counter.num_syntactic_literal_motifs + motif_counter.num_non_syntactic_literal_motifs}"
        )
        logging.info(
            "SYNTACTIC LITERAL MOTIF MASK:"
            + f" rules={len(motif_counter.syntactic_literal_rule_indices)},"
              f" motif_entries={motif_counter.num_syntactic_literal_motifs}/"
              f"{motif_counter.num_syntactic_literal_motifs + motif_counter.num_non_syntactic_literal_motifs}"
        )
    if unit_relation_motif_output_mode is not None:
        unit_group_summary = (
            "UNIT RELATION MOTIF MASK:"
            f" rules={len(motif_counter.unit_relation_rule_indices)},"
            f" motif_entries={motif_counter.num_unit_relation_motifs}/"
            f"{motif_counter.get_unit_relation_motif_mask().numel()}"
        )
        print(unit_group_summary)
        logging.info(unit_group_summary)
    wrapper = DataWrapper(
        dataa,
        motif_counter.relation_keys,
        node_onehot_info,
        edge_onehot_info=edge_onehot_info,
        edge_feature_info_mapping=motif_counter.feature_info_mapping,
        device='cuda',
    )

    # Resolve and validate active group objectives before the potentially
    # expensive full-matrix target count.
    motif_group_objectives = build_motif_group_objectives(
        syntactic_literal_mask=motif_counter.get_syntactic_literal_motif_mask(),
        non_literal_output_mode=non_literal_motif_output_mode,
        non_literal_loss_mode=non_literal_motif_loss_mode,
        non_literal_weight=alpha_motif_loss,
        syntactic_literal_output_mode=syntactic_literal_motif_output_mode,
        syntactic_literal_loss_mode=syntactic_literal_motif_loss_mode,
        syntactic_literal_weight=alpha_syntactic_literal_motif_loss,
        unit_relation_mask=motif_counter.get_unit_relation_motif_mask(),
        unit_relation_output_mode=unit_relation_motif_output_mode,
        unit_relation_loss_mode=unit_relation_motif_loss_mode,
        unit_relation_weight=alpha_unit_relation_motif_loss,
    )
    motif_full_matrices, motif_full_matrix_mask = motif_counter.count_batch(
        wrapper,
        batch_size=motif_batch_size,
        output_mode='full_matrix',
        detach_to_cpu=True,
    )
    list_graphs.motif_full_matrices = motif_full_matrices
    list_graphs.motif_full_matrix_mask = motif_full_matrix_mask
    full_target_summary = (
        "MOTIF CANONICAL FULL-MATRIX TARGETS:"
        f" values={tuple(motif_full_matrices.shape)},"
        f" mask={tuple(motif_full_matrix_mask.shape)}"
    )
    print(full_target_summary)
    logging.info(full_target_summary)

    motif_group_objectives = calibrate_group_histogram_specs(
        observed_full_matrices=motif_full_matrices,
        full_matrix_mask=motif_full_matrix_mask,
        groups=motif_group_objectives,
        histogram_num_bins=motif_histogram_num_bins,
        histogram_smoothing=motif_histogram_smoothing,
    )
    motif_training_uses_only_total_counts = bool(motif_group_objectives) and all(
        group.output_mode == 'total_count'
        for group in motif_group_objectives
    )
    for group in motif_group_objectives:
        group_summary = (
            f"MOTIF GROUP {group.name}: motifs={group.num_motifs}, "
            f"representation={group.output_mode}, loss={group.loss_mode}, "
            f"weight={group.weight}"
        )
        print(group_summary)
        logging.info(group_summary)

    # FactorBase comparison always uses total counts derived from the canonical
    # matrices, regardless of the representations selected for training.
    if sanity_check or sanity_check_only:
        counts, _, _ = represent_full_motif_matrices(
            full_matrices=motif_full_matrices,
            matrix_mask=motif_full_matrix_mask,
            output_mode='total_count',
        )
        list_graphs.motif_counts = counts
        # Previous sanity-check output:
        # aggregated = counts.sum(0)
        # print(aggregated)
        aggregated = motif_counter.aggregate_motif_counts(counts)
        print("\n" + "=" * 80)
        print("SANITY CHECK: AGGREGATED MOTIF COUNTS")
        print("=" * 80)
        print(aggregated)

        motif_counter.display_rules_and_motifs(aggregated)

        try:
            matches_factorbase, mismatches = (
                compare_aggregated_counts_to_factorbase_detailed(
                    aggregated_counts=aggregated,
                    motif_counter=motif_counter,
                    database_name=database_name,
                )
            )
            print("\n" + "=" * 80)
            print("FACTORBASE LOCAL_MULT COMPARISON")
            print("=" * 80)
            print(f"Counts match database local_mult values: {matches_factorbase}")
            if not matches_factorbase:
                print("First mismatches:")
                for mismatch in mismatches[:20]:
                    print(f"  {mismatch}")
                if len(mismatches) > 20:
                    print(f"  ... and {len(mismatches) - 20} more mismatches")
        except Exception as exc:
            print("\n[SanityCheck] FactorBase comparison could not be completed:")
            print(f"  {exc}")

        if dataset == "PROTEINS":
            print("\nCompare these counts against FactorBase local_mult columns in:")
            print("  proteins_experiment_BN.`edges(nodes0,nodes1)_CP`")
            print("  proteins_experiment_BN.`node_feature(nodes0)_CP`")
            print("  proteins_experiment_BN.`node_feature(nodes1)_CP`")

    if sanity_check_only:
        print("\nSanity-check-only mode enabled; exiting before model training.")
        raise SystemExit(0)
#endregion
#====================================================================================




print("#------------------------------------------------------")
if ideal_Evalaution:
    if split_mode == "paper_70_10_20":
        fifty_fifty_dataset = list_adj + val_adj + test_list_adj
    else:
        fifty_fifty_dataset = list_adj + test_list_adj

    fifty_fifty_dataset = [nx.from_numpy_array(graph.toarray()) for graph in fifty_fifty_dataset]
    random.shuffle(fifty_fifty_dataset)
    print("50%50 Evalaution of dataset")
    logging.info(mmd_eval(fifty_fifty_dataset[:int(len(fifty_fifty_dataset)/2)],fifty_fifty_dataset[int(len(fifty_fifty_dataset)/2):],diam=True))

    graphs_to_writeOnDisk = [nx.to_numpy_array(G) for  G in fifty_fifty_dataset]
    np.save(graph_save_path+dataset+'_dataset.npy',
            np.array(graphs_to_writeOnDisk, dtype=object),
            allow_pickle=True)
print("#------------------------------------------------------")

SubGraphNodeNum = subgraphSize if subgraphSize != None else list_graphs.max_num_nodes
in_feature_dim = list_graphs.feature_size  # ToDo: consider none Synthasis data
nodeNum = list_graphs.max_num_nodes

degree_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
degree_width = torch.tensor([[.1] for x in range(0, SubGraphNodeNum,1)])  # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
# ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly

bin_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
bin_width = torch.tensor([[1] for x in range(0, SubGraphNodeNum, 1)])

kernel_model = kernel(device=device, kernel_type=kernl_type, step_num=step_num,
                      bin_width=bin_width, bin_center=bin_center, degree_bin_center=degree_center,
                      degree_bin_width=degree_width)

if encoder_type == "AvePool":
    encoder = AveEncoder(in_feature_dim, [256], graphEmDim)
else:
    print("requested encoder is not implemented")
    exit(1)

if decoder_type == "FC":
    decoder = GraphTransformerDecoder_FC(graphEmDim, 256, nodeNum, directed)
else:
    print("requested decoder is not implemented")
    exit(1)


if (subgraphSize == None):
    list_graphs.processALL(self_for_none=self_for_none)
    adj_list = list_graphs.get_adj_list()
    graphFeatures, _ = get_subGraph_features(adj_list, None, kernel_model)
    list_graphs.set_features(graphFeatures)


#====================================================================================
# %% Node and edge feature decoders
# region Node and edge feature decoders
# Added for feature decoding, I implemented it as a simple MLP that takes the graph
# embedding as input and outputs the node and edge features, and then I added a loss
# term to the total loss.
#
# Keep these heads optional so the plain GraphVAE baseline can still be trained with
# adjacency reconstruction only, which matches the paper's non-MM setting.
has_node_feature_targets = (
    bool(list_graphs.node_onehot_s) and list_graphs.node_onehot_s[0] is not None
)
has_edge_feature_targets = (
    bool(list_graphs.edge_onehot_s) and list_graphs.edge_onehot_s[0] is not None
)

use_node_feature_decoder = (
    (alpha_node_feat > 0 or use_motif_loss) and has_node_feature_targets
)
use_edge_feature_decoder = (
    (alpha_edge_feat > 0 or use_motif_loss) and has_edge_feature_targets
)

if alpha_node_feat > 0 and not has_node_feature_targets:
    print("[FeatureLoss] Node feature loss requested but no node one-hot targets are available. Disabling node feature decoder/loss.")
    logging.info("[FeatureLoss] Node feature loss requested but no node one-hot targets are available. Disabling node feature decoder/loss.")
if alpha_edge_feat > 0 and not has_edge_feature_targets:
    print("[FeatureLoss] Edge feature loss requested but no edge one-hot targets are available. Disabling edge feature decoder/loss.")
    logging.info("[FeatureLoss] Edge feature loss requested but no edge one-hot targets are available. Disabling edge feature decoder/loss.")
if use_motif_loss and not has_node_feature_targets:
    raise RuntimeError(
        "Motif loss requires node one-hot targets so reconstructed node "
        "features can be counted. Disable motif_loss or use a dataset with "
        "node feature targets."
    )

node_feat_decoder = None
edge_feat_decoder = None
if use_node_feature_decoder:
    node_onehot_dim = list_graphs.node_onehot_s[0].shape[-1]
    node_feat_decoder = NodeFeatureDecoder(
        graphEmDim, list_graphs.max_num_nodes, node_onehot_dim
    )
if use_edge_feature_decoder:
    edge_onehot_dim = list_graphs.edge_onehot_s[0].shape[0]
    edge_feat_decoder = EdgeFeatureDecoder(
        graphEmDim, list_graphs.max_num_nodes, edge_onehot_dim
    )
#endregion
#====================================================================================

#====================================================================================
model = kernelGVAE(kernel_model, encoder, decoder, AutoEncoder, graphEmDim=graphEmDim,
                   node_feature_decoder=node_feat_decoder,
                   edge_feature_decoder=edge_feat_decoder,
                   correct_reparameterization=correct_reparameterization)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000,6000,7000,8000,9000], gamma=0.5)
# A simple schedule helps the tiny motif-only run keep improving after the
# fast early drop. These milestones all occur within the 1000-epoch debug run.
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#    optimizer, milestones=[300, 600, 900], gamma=0.5
#)

# pos_wight = torch.true_divide((list_graphs.max_num_nodes**2*len(list_graphs.processed_adjs)-list_graphs.toatl_num_of_edges),
#                               list_graphs.toatl_num_of_edges) # addrressing imbalance data problem: ratio between positve to negative instance
# pos_wight = torch.tensor(40.0)
# pos_wight/=10
num_nodes = list_graphs.max_num_nodes
# ToDo Check the effect of norm and pos weight

# target_kelrnel_val = kernel_model(target_adj)

if not tiny_overfit:
    list_graphs.shuffle()
start = timeit.default_timer()
# Parameters
step = 0
swith = False
print(model)
logging.info(model.__str__())
min_loss = float('inf')
adaptive_motif_temperature = motif_temperature_start


# 50%50 Evaluation

#region model loading
load_model = False
if load_model == True:  # I used this in line code to load a model #TODO: fix it
    # ========================================
    model_dir = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/MMD_AvePool_FC_DD_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue100001651364417.4785793/"
    model.load_state_dict(torch.load(model_dir + "model_9999_3"))
    # EvalTwoSet(model, test_list_adj, model_dir+"/", Save_generated= False, )

# model_dir1 = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/FinalResultHopefully/"
# model.load_state_dict(torch.load(model_dir1+"model_9999_3"))
# EvalTwoSet(model, test_list_adj, model_dir+"/", Save_generated= False, )
#endregion

#=========================================================================================
# %% Training loop
#region Training loop
for epoch in range(epoch_number):

    if not tiny_overfit:
        list_graphs.shuffle()
    batch = 0
    for iter in range(
        0,
        max(int(len(list_graphs.list_adjs) / train_batch_size), 1) * train_batch_size,
        train_batch_size,
    ):
        from_ = iter
        to_ = train_batch_size * (batch + 1)
        # for iter in range(0, len(list_graphs.list_adjs), train_batch_size):
        #     from_ = iter
        #     to_= train_batch_size*(batch+1) if train_batch_size*(batch+2)<len(list_graphs.list_adjs) else len(list_graphs.list_adjs)

        if subgraphSize == None:
            org_adj, x_s, node_num, subgraphs_indexes, target_kelrnel_val = list_graphs.get__(from_, to_, self_for_none,
                                                                                              bfs=subgraphSize)
        else:
            org_adj, x_s, node_num, subgraphs_indexes = list_graphs.get__(from_, to_, self_for_none, bfs=subgraphSize)

        # Keep an immutable copy of real node counts for masked node-feature loss.
        # `node_num` may be overwritten below for decoder-specific behavior.
        true_node_num = list(node_num)
        target_node_onehots, target_edge_onehots, batch_end = (
            list_graphs.get_feature_targets(from_, from_ + len(true_node_num))
        )
        if batch_end - from_ != len(true_node_num):
            raise RuntimeError(
                "Feature target batch does not match the graph batch: "
                f"graphs={len(true_node_num)}, targets={batch_end - from_}."
            )

        if (type(decoder)) in [GraphTransformerDecoder_FC]:  #
            node_num = len(node_num) * [list_graphs.max_num_nodes]

        x_s = torch.cat(x_s)
        x_s = x_s.reshape(-1, x_s.shape[-1])

        model.train()
        if subgraphSize == None:
            _, subgraphs = get_subGraph_features(org_adj, None, None)
        else:
            target_kelrnel_val, subgraphs = get_subGraph_features(org_adj, subgraphs_indexes, kernel_model)

        # target_kelrnel_val = kernel_model(org_adj, node_num)

        # batchSize = [org_adj.shape[0], org_adj.shape[1]]

        batchSize = [len(org_adj), org_adj[0].shape[0]]

        # org_adj_dgl = [dgl.from_scipy(sp.csr_matrix(graph.cpu().detach().numpy())) for graph in org_adj]
        [graph.setdiag(1) for graph in org_adj]
        org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
        org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
        pos_wight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(), subgraphs.sum())
        
        # added for feature decoding 
        # reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit = model(
        (reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit,
            node_feat_logits, edge_feat_logits) = model(
            org_adj_dgl.to(device), x_s.to(device), batchSize, subgraphs_indexes)
        kl_loss, reconstruction_loss, acc, kernel_cost, each_kernel_loss,log_sigma_values = OptimizerVAE(reconstructed_adj,
                                                                                        generated_kernel_val,
                                                                                        subgraphs.to(device),
                                                                                        [val.to(device) for val in
                                                                                         target_kelrnel_val],
                                                                                        post_log_std, post_mean, alpha,
                                                                                        reconstructed_adj_logit,
                                                                                        pos_wight, 2)

        # Added loss for feature decoding ============================================        
        #=============================================================================
        node_feat_loss = reconstructed_adj_logit.new_tensor(0.0)
        edge_feat_loss = reconstructed_adj_logit.new_tensor(0.0)

        if alpha_node_feat > 0 and node_feat_logits is not None:
            if target_node_onehots is None or any(t is None for t in target_node_onehots):
                raise RuntimeError(
                    "Node feature decoder is enabled, but this batch has no "
                    "padded node one-hot targets."
                )
            target_node_oh = torch.stack(
                [torch.as_tensor(t) for t in target_node_onehots]
            ).to(device)
            node_feat_loss = compute_true_node_feat_loss(
                node_feat_logits=node_feat_logits,
                target_node_onehot=target_node_oh,
                true_node_num=true_node_num
            )

        if alpha_edge_feat > 0 and edge_feat_logits is not None:
            if target_edge_onehots is None or any(t is None for t in target_edge_onehots):
                raise RuntimeError(
                    "Edge feature decoder is enabled, but this batch has no "
                    "padded edge one-hot targets."
                )
            target_edge_oh = torch.stack(
                [torch.as_tensor(t) for t in target_edge_onehots]
            ).to(device)
            # Edge-feature supervision now treats channels as one-hot classes and
            # ignores padded/non-edge positions; only real existing edges contribute.
            edge_feat_loss = compute_true_edge_feat_loss(
                edge_feat_logits=edge_feat_logits,
                target_edge_onehot=target_edge_oh,
                true_node_num=true_node_num
            )
        # These hard metrics are evaluation-only diagnostics. They answer a
        # stricter question than the soft training loss: after discretizing the
        # current reconstruction, do the motif counts still match exactly?
        hard_motif_loss = torch.tensor(0.0, device=device)
        hard_motif_exact_zero = torch.tensor(False, device=device)
        hard_motif_exact_zero_per_graph = torch.zeros(
            len(true_node_num), dtype=torch.bool, device=device
        )
        syntactic_literal_motif_loss = torch.tensor(0.0, device=device)
        non_literal_motif_loss = torch.tensor(0.0, device=device)
        unit_relation_motif_loss = torch.tensor(0.0, device=device)
        weighted_motif_loss_term = torch.tensor(0.0, device=device)
        hard_threshold_sweep_summary = None
        motif_temperature = get_motif_temperature(
            epoch=epoch,
            total_epochs=epoch_number,
            start_temp=motif_temperature_start,
            end_temp=motif_temperature_end,
            anneal_start_frac=motif_temperature_anneal_start_frac,
        )
        motif_temperature_scheduled = motif_temperature
        motif_temperature_guard_triggered = False
        motif_temperature_guard_base = None
        motif_temperature_guard_limit = None
        motif_temperature_guard_proposed = motif_temperature
        if use_motif_loss:
            observed_motif_counts = None
            observed_motif_full_matrices = (
                list_graphs.motif_full_matrices[from_:batch_end].to(device)
            )
            observed_motif_full_matrix_mask = (
                list_graphs.motif_full_matrix_mask.to(device)
            )
            if motif_training_uses_only_total_counts:
                observed_motif_counts, _, _ = represent_full_motif_matrices(
                    full_matrices=observed_motif_full_matrices,
                    matrix_mask=observed_motif_full_matrix_mask,
                    output_mode='total_count',
                )

            def motif_losses_at_temperature(current_motif_temperature):
                current_recon_wrapper = ReconstructedDataWrapper(
                    reconstructed_adj=reconstructed_adj_logit,
                    node_feat_logits=node_feat_logits,
                    edge_feat_logits=edge_feat_logits,
                    relation_keys=motif_counter.relation_keys,
                    node_onehot_info=node_onehot_info,
                    feature_onehot_mapping=wrapper.feature_onehot_mapping,
                    edge_onehot_info=edge_onehot_info,
                    edge_feature_info_mapping=motif_counter.feature_info_mapping,
                    use_soft_adj=True,
                    prob_temperature=current_motif_temperature,
                    device=device,
                )
                (
                    current_recon_full_matrices,
                    current_recon_full_matrix_mask,
                ) = motif_counter.count_batch(
                    current_recon_wrapper,
                    batch_size=motif_batch_size,
                    output_mode='full_matrix',
                )
                if not torch.equal(
                    observed_motif_full_matrix_mask,
                    current_recon_full_matrix_mask,
                ):
                    raise RuntimeError(
                        "Observed and reconstructed full motif matrix masks "
                        "do not match."
                    )
                grouped_loss = compute_grouped_motif_loss(
                    observed_full_matrices=observed_motif_full_matrices,
                    predicted_full_matrices=current_recon_full_matrices,
                    full_matrix_mask=current_recon_full_matrix_mask,
                    groups=motif_group_objectives,
                    histogram_num_bins=motif_histogram_num_bins,
                    histogram_smoothing=motif_histogram_smoothing,
                )
                zero = reconstructed_adj_logit.new_zeros(())
                return (
                    grouped_loss.loss,
                    grouped_loss.group_losses.get(
                        NON_LITERAL_MOTIF_GROUP,
                        zero,
                    ),
                    grouped_loss.group_losses.get(
                        SYNTACTIC_LITERAL_MOTIF_GROUP,
                        zero,
                    ),
                    grouped_loss.group_losses.get(
                        UNIT_RELATION_MOTIF_GROUP,
                        zero,
                    ),
                    grouped_loss.weighted_loss,
                )

            if (
                motif_temperature_guard_ratio > 0.0
                and motif_temperature_scheduled < motif_temperature_start - 1e-12
            ):
                motif_temperature = max(
                    motif_temperature_scheduled,
                    adaptive_motif_temperature * motif_temperature_guard_sharpen_factor,
                )
                motif_temperature = min(motif_temperature, motif_temperature_start)
                motif_temperature_guard_proposed = motif_temperature
            else:
                motif_temperature = motif_temperature_scheduled
                motif_temperature_guard_proposed = motif_temperature
                adaptive_motif_temperature = motif_temperature

            (
                motif_loss,
                non_literal_motif_loss,
                syntactic_literal_motif_loss,
                unit_relation_motif_loss,
                weighted_motif_loss_term,
            ) = motif_losses_at_temperature(motif_temperature)

            if (
                motif_temperature_guard_ratio > 0.0
                and motif_temperature_scheduled < motif_temperature_start - 1e-12
            ):
                non_motif_loss_term = (
                    kernel_cost.detach()
                    + (alpha_node_feat * node_feat_loss).detach()
                    + (alpha_edge_feat * edge_feat_loss).detach()
                )
                non_motif_loss_term = torch.clamp(non_motif_loss_term, min=1e-12)
                guard_limit = motif_temperature_guard_ratio * non_motif_loss_term
                motif_term_for_guard = weighted_motif_loss_term.detach()
                motif_temperature_guard_base = float(non_motif_loss_term.cpu().item())
                motif_temperature_guard_limit = float(guard_limit.cpu().item())
                if (
                    bool(torch.isfinite(motif_term_for_guard).item())
                    and bool(torch.isfinite(guard_limit).item())
                    and bool((motif_term_for_guard > guard_limit).item())
                ):
                    motif_temperature_guard_triggered = True
                    motif_temperature = min(
                        motif_temperature_start,
                        adaptive_motif_temperature * motif_temperature_guard_relax_factor,
                    )
                    (
                        motif_loss,
                        non_literal_motif_loss,
                        syntactic_literal_motif_loss,
                        unit_relation_motif_loss,
                        weighted_motif_loss_term,
                    ) = motif_losses_at_temperature(motif_temperature)
                adaptive_motif_temperature = motif_temperature

            if motif_training_uses_only_total_counts:
                # The hard wrapper thresholds adjacency and converts categorical
                # predictions to one-hot assignments, so these metrics reflect
                # the discrete graph you would inspect after training.
                # Structured modes skip them because representation-specific
                # hard metrics have not been defined yet.
                with torch.no_grad():
                    hard_recon_wrapper = ReconstructedDataWrapper(
                        reconstructed_adj=reconstructed_adj_logit.detach(),
                        node_feat_logits=node_feat_logits.detach(),
                        edge_feat_logits=edge_feat_logits.detach() if edge_feat_logits is not None else None,
                        relation_keys=motif_counter.relation_keys,
                        node_onehot_info=node_onehot_info,
                        feature_onehot_mapping=wrapper.feature_onehot_mapping,
                        edge_onehot_info=edge_onehot_info,
                        edge_feature_info_mapping=motif_counter.feature_info_mapping,
                        use_soft_adj=False,
                        prob_temperature=motif_temperature,
                        device=device,
                    )
                    hard_recon_counts = motif_counter.count_batch(
                        hard_recon_wrapper,
                        batch_size=motif_batch_size,
                    )
                    (hard_motif_loss,
                     hard_motif_exact_zero,
                     hard_motif_exact_zero_per_graph) = compute_hard_motif_metrics(
                        observed_counts=observed_motif_counts,
                        hard_predicted_counts=hard_recon_counts,
                    )

                    should_report_hard_sweep = (tiny_overfit and (step % 10 == 0)) or \
                        ((step + 1) % visulizer_step == 0) or (epoch_number == epoch + 1)
                    if should_report_hard_sweep:
                        hard_threshold_sweep_summary = summarize_hard_motif_threshold_sweep(
                            observed_counts=observed_motif_counts,
                            adj_probs=get_reconstructed_adj_probs(
                                reconstructed_adj_logit,
                                prob_temperature=motif_temperature,
                            ),
                            hard_recon_wrapper=hard_recon_wrapper,
                            motif_counter=motif_counter,
                            batch_size=motif_batch_size,
                        )
            #m_loss = motif_loss * alpha_motif_loss


        else:
            #m_loss = torch.tensor(0.0, device=device)
            motif_loss=torch.tensor(0.0, device=device)
#====================-------=-==-=-=-===-*****%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@

        loss = kernel_cost + \
            alpha_node_feat * node_feat_loss +\
            alpha_edge_feat * edge_feat_loss+\
            weighted_motif_loss_term

        hard_exact_match_count = int(hard_motif_exact_zero_per_graph.sum().item())
        hard_exact_match_total = int(hard_motif_exact_zero_per_graph.numel())
        detailed_hard_motif_counts = None
        should_report_detailed_hard_counts = (
            tiny_overfit
            and motif_training_uses_only_total_counts
            and hard_exact_match_total == 1
            and ((step + 1) % visulizer_step == 0 or (epoch_number == epoch + 1))
        )
        if should_report_detailed_hard_counts and use_motif_loss:
            detailed_hard_motif_counts = summarize_single_graph_motif_counts(
                observed_counts=observed_motif_counts,
                hard_predicted_counts=hard_recon_counts,
            )

        motif_temperature_guard_status = ""
        if motif_temperature_guard_ratio > 0.0:
            motif_temperature_guard_status = (
                f"| motif_temp_scheduled: {motif_temperature_scheduled:.3f} "
                f"| motif_temp_proposed: {motif_temperature_guard_proposed:.3f} "
                f"| motif_temp_guard: {int(motif_temperature_guard_triggered)} "
            )
            if motif_temperature_guard_base is not None:
                motif_temperature_guard_status += (
                    f"| motif_guard_base: {motif_temperature_guard_base:05f} "
                    f"| motif_guard_limit: {motif_temperature_guard_limit:05f} "
                )

        if tiny_overfit and (step % 10 == 0):
            print(f"[TinyOverfit] step={step} total={loss.item():.6f} "
                  f"motif={motif_loss.item():.6f} hard_motif={hard_motif_loss.item():.6f} "
                  f"regular_motif={non_literal_motif_loss.item():.6f} "
                  f"syntactic_literal_motif={syntactic_literal_motif_loss.item():.6f} "
                  f"unit_relation_motif={unit_relation_motif_loss.item():.6f} "
                  f"motif_temp={motif_temperature:.3f} "
                  f"{motif_temperature_guard_status}"
                  f"hard_exact_all={bool(hard_motif_exact_zero.item())} "
                  f"hard_exact_graphs={hard_exact_match_count}/{hard_exact_match_total} "
                  f"recon={reconstruction_loss.item():.6f} "
                  f"node={node_feat_loss.item():.6f} edge={edge_feat_loss.item():.6f}")
    #    
    #   loss = kernel_cost  # Graph generation loss without feature decoding
 
        tmp = [None for x in range(len(functions))]
        pltr.add_values(step, [acc.cpu().item(), loss.cpu().item(), *each_kernel_loss], tmp,
                        redraw=redraw)  # ["Accuracy", "loss", "AUC"])

        step += 1
        optimizer.zero_grad()
        loss.backward()

        if keepThebest and min_loss > loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "model")
        # torch.nn.utils.clip_grad_norm(model.parameters(),  1.0044e-05)
        optimizer.step()

        if (step + 1) % visulizer_step == 0 or epoch_number == epoch + 1:
            model.eval()
            if not tiny_overfit:
                pltr.redraw()
            if not tiny_overfit:
                rnd_indx = random.randint(0, len(node_num) - 1)
                sample_graph = reconstructed_adj[rnd_indx].cpu().detach().numpy()
                sample_graph = sample_graph[:node_num[rnd_indx], :node_num[rnd_indx]]
                sample_graph[sample_graph >= 0.5] = 1
                sample_graph[sample_graph < 0.5] = 0


                G = nx.from_numpy_array(sample_graph)
                plotter.plotG(G, "generated" + dataset,
                              file_name=str(generated_graph_train_dir / f"generatedSample_At_epoch{epoch}"))
                print("reconstructed graph vs Validation:")
                logging.info("reconstructed graph vs Validation:")
                reconstructed_adj = reconstructed_adj.cpu().detach().numpy()
                reconstructed_adj[reconstructed_adj >= 0.5] = 1
                reconstructed_adj[reconstructed_adj < 0.5] = 0
                reconstructed_adj = [nx.from_numpy_array(reconstructed_adj[i]) for i in range(reconstructed_adj.shape[0])]
                reconstructed_adj = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                                    reconstructed_adj if not nx.is_empty(G)]

                target_set = [nx.from_numpy_array(val_adj[i].toarray()) for i in range(len(val_adj))]
                target_set = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in target_set if
                            not nx.is_empty(G)]
                reconstruc_MMD_loss = mmd_eval(reconstructed_adj, target_set[:len(reconstructed_adj)], diam=True)
                logging.info(reconstruc_MMD_loss)

            #todo: instead of printing diffrent level of logging shoud be used
            model.eval()
            if (not tiny_overfit) and task == "graphGeneration":
                # print("generated vs Validation:")
                mmd_res= EvalTwoSet(model, val_adj[:1000], graph_save_path, Save_generated=True, _f_name=epoch)
                with open(run_mmd_log_path, 'a') as f:
                        f.write(str(step)+" @ loss @ , "+str(loss.item())+" , @ Reconstruction @ , "+reconstruc_MMD_loss+" , @ Val @ , " +mmd_res+"\n")

                if keep_best_validation_mmd:
                    validation_mmd_metrics = parse_graph_quality_result(mmd_res)
                    validation_mmd_score = compute_validation_mmd_score(
                        validation_mmd_metrics,
                        best_validation_mmd_metric,
                        dataset,
                    )
                    if validation_mmd_score is None:
                        logging.warning(
                            "Could not compute %s validation MMD checkpoint score from: %s",
                            best_validation_mmd_metric,
                            mmd_res,
                        )
                    elif validation_mmd_score < best_validation_mmd_score:
                        best_validation_mmd_score = validation_mmd_score
                        best_validation_mmd_metadata = {
                            "epoch": int(epoch),
                            "epoch_1_based": int(epoch + 1),
                            "batch": int(batch),
                            "step": int(step),
                            "loss": float(loss.item()),
                            "score": float(validation_mmd_score),
                            "score_mode": best_validation_mmd_metric,
                            "score_metrics": score_metrics_for_mode(best_validation_mmd_metric),
                            "score_components": score_components_for_mode(
                                validation_mmd_metrics,
                                best_validation_mmd_metric,
                                dataset,
                            ),
                            "score_denominators": score_denominators_for_mode(
                                best_validation_mmd_metric,
                                dataset,
                            ),
                            "score_weights": score_weights_for_mode(
                                best_validation_mmd_metric
                            ),
                            "metrics": validation_mmd_metrics,
                            "table2_metrics": table2_metrics_from_parsed(
                                validation_mmd_metrics
                            ),
                            "table3_metrics": table3_metrics_from_parsed(
                                validation_mmd_metrics
                            ),
                            "model_path": str(best_validation_mmd_model_path),
                            "validation_generated_graphs": (
                                graph_save_path
                                + "Single_comp_generatedGraphs_adj_"
                                + str(epoch)
                                + ".npy"
                            ),
                            "validation_mmd_result": mmd_res,
                        }
                        torch.save(model.state_dict(), str(best_validation_mmd_model_path))
                        write_best_validation_mmd_metadata(
                            best_validation_mmd_metadata_path,
                            best_validation_mmd_metadata,
                        )
                        best_message = (
                            "New best validation MMD checkpoint: "
                            f"score={validation_mmd_score:.6f}, "
                            f"mode={best_validation_mmd_metric}, epoch={epoch + 1}"
                        )
                        print(best_message)
                        logging.info(best_message)

                if save_validation_checkpoints:
                    checkpoint_path = graph_save_dir / f"checkpoint_epoch_{epoch + 1:05d}_batch_{batch}.pt"
                    torch.save(model.state_dict(), str(checkpoint_path))
                    checkpoint_message = f"Saved validation checkpoint: {checkpoint_path}"
                    print(checkpoint_message)
                    logging.info(checkpoint_message)

                if ((step + 1) % visulizer_step * 2):
                    torch.save(model.state_dict(), graph_save_path + "model_" + str(epoch) + "_" + str(batch))
            stop = timeit.default_timer()
            # print("trainning time at this epoch:", str(stop - start))
            model.train()
            # if reconstruction_loss.item()<0.051276 and not swith:
            #     alpha[-1] *=2
            #     swith = True
        k_loss_str = ""
        for indx, l in enumerate(each_kernel_loss):
            k_loss_str += functions[indx + 2] + ":"
            k_loss_str += str(l) + ".   "

        epoch_status = (
            f"Epoch: {epoch + 1:03d} |Batch: {batch:03d} | latent_mode: {latent_mode} "
            f"| loss: {loss.item():05f} | motif_loss: {motif_loss.item():05f} "
            f"| regular_motif_loss: {non_literal_motif_loss.item():05f} "
            f"| syntactic_literal_motif_loss: {syntactic_literal_motif_loss.item():05f} "
            f"| unit_relation_motif_loss: {unit_relation_motif_loss.item():05f} "
            f"| motif_temp: {motif_temperature:.3f} "
            f"{motif_temperature_guard_status}"
            f"| node_feat_loss: {node_feat_loss.item():05f} "
            f"| edge_feat_loss: {edge_feat_loss.item():05f} "
            f"| hard_motif_loss: {hard_motif_loss.item():05f} "
            f"| hard_exact_all: {int(bool(hard_motif_exact_zero.item()))} "
            f"| hard_exact_graphs: {hard_exact_match_count}/{hard_exact_match_total} "
            f"| reconstruction_loss: {reconstruction_loss.item():05f} "
            f"| weighted_components: kernel={float(kernel_cost.detach().cpu().item()):05f},"
            f" node={float((alpha_node_feat * node_feat_loss).detach().cpu().item()):05f},"
            f" edge={float((alpha_edge_feat * edge_feat_loss).detach().cpu().item()):05f},"
            f" motif={float(weighted_motif_loss_term.detach().cpu().item()):05f} "
            f"| z_kl_loss: {kl_loss.item():05f} | accu: {(acc.item() if torch.is_tensor(acc) else float(acc)):03f}"
        )
        print(epoch_status, k_loss_str)
        logging.info(epoch_status + " " + str(k_loss_str))
        if hard_threshold_sweep_summary is not None:
            print(hard_threshold_sweep_summary)
            logging.info(hard_threshold_sweep_summary)
        if detailed_hard_motif_counts is not None:
            for detail_line in detailed_hard_motif_counts:
                print(detail_line)
                logging.info(detail_line)
        # print(log_sigma_values)
        log_std = ""
        for indx, l in enumerate(log_sigma_values):
            log_std += "log_std " + functions[indx + 2] + ":"
            log_std += str(l) + ".   "
        print(log_std)
        logging.info(log_std)
        batch += 1
        # scheduler.step()
        # scheduler.step()
    if (
        not tiny_overfit
        and checkpoint_interval_epochs > 0
        and (epoch + 1) % checkpoint_interval_epochs == 0
    ):
        periodic_checkpoint_path = graph_save_dir / f"periodic_epoch_{epoch + 1:05d}.pt"
        torch.save(model.state_dict(), str(periodic_checkpoint_path))
        periodic_checkpoint_message = (
            f"Saved periodic epoch checkpoint: {periodic_checkpoint_path}"
        )
        print(periodic_checkpoint_message)
        logging.info(periodic_checkpoint_message)
model.eval()
if not tiny_overfit:
    torch.save(model.state_dict(), graph_save_path + "model_" + str(epoch) + "_" + str(batch))
#endregion
#=========================================================================================

stop = timeit.default_timer()
print("trainning time:", str(stop - start))
logging.info("trainning time: " + str(stop - start))
# save the train loss for comparing the convergence
import json

file_name = graph_save_path + "_" + encoder_type + "_" + decoder_type + "_" + dataset + "_" + task + "_" + model_name + "_elbo_loss.txt"

if not tiny_overfit:
    with open(file_name, "w") as fp:
        json.dump(list(np.array(pltr.values_train[-2]) + np.array(pltr.values_train[-1])), fp)

# with open(file_name + "/_CrossEntropyLoss.txt", "w") as fp:
#     json.dump(list(np.array(pltr.values_train[-2])), fp)
#
# with open(file_name + "/_train_loss.txt", "w") as fp:
#     json.dump(pltr.values_train[1], fp)

# save the log plot on the current directory
if not tiny_overfit:
    pltr.save_plot(graph_save_path + "KernelVGAE_log_plot")


#==========================================================================================
#   %% Evaluation of the model on graph generation task
# region graph generation task
if task == "graphGeneration":
    final_eval_model_source = "final_epoch"
    if keep_best_validation_mmd and best_validation_mmd_model_path.exists():
        model.load_state_dict(
            torch.load(str(best_validation_mmd_model_path), map_location=device)
        )
        final_eval_model_source = "best_validation_mmd_model"
        best_eval_message = (
            "Loaded best validation MMD checkpoint for final evaluation: "
            f"{best_validation_mmd_model_path}"
        )
        if best_validation_mmd_metadata is not None:
            best_eval_message += (
                f" (score={best_validation_mmd_metadata['score']:.6f}, "
                f"epoch={best_validation_mmd_metadata['epoch_1_based']})"
            )
        print(best_eval_message)
        logging.info(best_eval_message)
    final_mmd_res = EvalTwoSet(
        model,
        test_list_adj,
        graph_save_path,
        Save_generated=True,
        _f_name="final_eval",
    )
    third_party_json_path = None
    if third_party_eval:
        third_party_json_path = run_third_party_graph_realism_eval(graph_save_dir, args, device)
    final_metric_summary = write_final_metric_summaries(
        graph_save_dir,
        final_mmd_res,
        third_party_json_path=third_party_json_path,
        model_source=final_eval_model_source,
    )
    final_metric_message = (
        "Saved final Table 2/Table 3 metric summaries: "
        f"{final_metric_summary['table2_metrics_file']}, "
        f"{final_metric_summary['table3_metrics_file']}"
    )
    print(final_metric_message)
    logging.info(final_metric_message)
# endregion
#==========================================================================================

#==========================================================================================
# %% Evaluation of the model on graph generation task
# region graph Classification task
# if task == "graphClasssification":
#
#
#     org_adj,x_s, node_num, subgraphs_indexes,  labels = list_graphs.adj_s, list_graphs.x_s, list_graphs.num_nodes, list_graphs.subgraph_indexes, list_graphs.labels
#
#     if(type(decoder))in [  GraphTransformerDecoder_FC]: #
#         node_num = len(node_num)*[list_graphs.max_num_nodes]
#
#     x_s = torch.cat(x_s)
#     x_s = x_s.reshape(-1, x_s.shape[-1])
#
#     model.eval()
#     # if subgraphSize == None:
#     #     _, subgraphs = get_subGraph_features(org_adj, None, None)
#
#     batchSize = [len(org_adj), org_adj[0].shape[0]]
#
#     [graph.setdiag(1) for graph in org_adj]
#     org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
#
#     org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
#     mean, std = model.encode(org_adj_dgl.to(device), x_s.to(device), batchSize)
#
#     prior_samples = model.reparameterize(mean, std)
#     # model.encode(org_adj_dgl.to(device), x_s.to(device), batchSize)
#     # _, prior_samples, _, _, _,_ = model(org_adj_dgl.to(device), x_s.to(device), node_num, batchSize, subgraphs_indexes)
#
#
#
#     import classification as CL
#
#     # NN Classifier
#     labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report  = CL.NN(prior_samples.cpu().detach(), labels)
#
#     print("Accuracy:{}".format(accuracy),
#           "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
#           "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
#           "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
#           "confusion matrix:{}".format(conf_matrix))
#
#     # KNN clasiifier
#     labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report  = CL.knn(prior_samples.cpu().detach(), labels)
#     print("Accuracy:{}".format(accuracy),
#           "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
#           "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
#           "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
#           "confusion matrix:{}".format(conf_matrix))
# # evaluatin graph statistics in graph generation tasks
# endregion
#==========================================================================================

#==========================================================================================
# %% Evaluation of the model on graph representation learning task
# region graph representation learning task
# if task == "GraphRepresentation":
#
#     list_test_graphs.processALL(self_for_none=self_for_none)
#
#     test_adj_list = list_test_graphs.get_adj_list()
#     graphFeatures, _ = get_subGraph_features(test_adj_list, None, kernel_model)
#     list_test_graphs.set_features(graphFeatures)
#
#     from_ = 0
#     ro = [-1]
#     org_adj = list_test_graphs.adj_s[from_:to_]
#     x_s = list_test_graphs.x_s[from_:to_]
#     # test_adj_list.num_nodes[from_:to_]
#     labels = list_test_graphs.labels
#
#     x_s = torch.cat(x_s)
#     x_s = x_s.reshape(-1, x_s.shape[-1])
#
#     model.eval()
#     # if subgraphSize == None:
#     #     _, subgraphs = get_subGraph_features(org_adj, None, None)
#     # else:
#     #     target_kelrnel_val, subgraphs = get_subGraph_features(org_adj, subgraphs_indexes, kernel_model)
#
#     # target_kelrnel_val = kernel_model(org_adj, node_num)
#
#     # batchSize = [org_adj.shape[0], org_adj.shape[1]]
#
#     batchSize = [len(org_adj), org_adj[0].shape[0]]
#
#     # org_adj_dgl = [dgl.from_scipy(sp.csr_matrix(graph.cpu().detach().numpy())) for graph in org_adj]
#     [graph.setdiag(1) for graph in org_adj]
#     org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
#     org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
#     pos_wight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(), subgraphs.sum())
#
#     reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit = model(
#         org_adj_dgl.to(device), x_s.to(device), batchSize, subgraphs_indexes)
#
#     i = 0
#     dic = {}
#     digit_labels = []
#     for labl in labels:
#         if labl not in dic:
#             dic[labl] = i
#             i += 1
#         digit_labels.append(dic[labl])
#
#     plotter.featureVisualizer(prior_samples.detach().cpu().numpy(), digit_labels)
#endregion
#==========================================================================================
