#!/usr/bin/env python3
"""Train and generate one frozen DeFoG dataset/training-seed job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GRAPH_EVAL_SRC = ROOT / "graph_evaluation" / "src"
if str(GRAPH_EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(GRAPH_EVAL_SRC))

try:  # direct script execution
    from verify_campaign import load_yaml, verify_generated_artifact  # type: ignore
except ImportError:  # package import in tests
    from .verify_campaign import load_yaml, verify_generated_artifact


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quoted_override(key: str, value: str) -> str:
    # JSON string syntax is accepted by Hydra and protects '=' and '|'.
    return f"{key}={json.dumps(value)}"


def run_command(command: list[str], *, cwd: Path, env: dict) -> None:
    print("COMMAND", json.dumps(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def common_overrides(
    *, dataset: str, seed: int, spec: dict, schedule: dict, artifact_root: Path,
    defog_commit: str,
) -> list[str]:
    dataset_root = artifact_root / dataset.lower()
    return [
        f"+experiment={schedule['experiment']}",
        "dataset=frozen_graphvae",
        f"dataset.identity={dataset}",
        f"dataset.root={dataset_root}",
        f"dataset.train_sha256={spec['collection_sha256']['train']}",
        f"dataset.validation_sha256={spec['collection_sha256']['validation']}",
        f"dataset.reference_sha256={spec['collection_sha256']['reference']}",
        f"dataset.feature_mode={spec['feature_mode']}",
        quoted_override("dataset.feature_schema", spec["feature_schema"]),
        f"train.seed={seed}",
        f"train.batch_size={int(schedule['batch_size'])}",
        "general.wandb=disabled",
        "general.validation_selection=loss",
        "general.generation_seed=12345",
        "general.strict_generation=true",
        f"general.defog_commit={defog_commit}",
    ]


def train(
    *, python: Path, defog_root: Path, job_root: Path, overrides: list[str],
    schedule: dict,
) -> Path:
    training_dir = job_root / "training"
    seed_name = job_root.name[len("seed_") :] if job_root.name.startswith("seed_") else job_root.name
    name = f"frozen_{job_root.parent.name}_seed_{seed_name}"
    command = [
        str(python),
        "src/main.py",
        *overrides,
        f"general.name={name}",
        f"train.n_epochs={int(schedule['epochs'])}",
        f"general.check_val_every_n_epochs={int(schedule['validate_every_epochs'])}",
        f"hydra.run.dir={training_dir}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(GRAPH_EVAL_SRC), str(defog_root), env.get("PYTHONPATH", "")]
    )
    run_command(command, cwd=defog_root, env=env)
    checkpoints = sorted(training_dir.glob("checkpoints/*/best-*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one best-validation checkpoint, found {checkpoints}"
        )
    alias = job_root / "best_validation.ckpt"
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    alias.symlink_to(checkpoints[0].resolve())
    return checkpoints[0].resolve()


def generate(
    *, python: Path, defog_root: Path, job_root: Path, overrides: list[str],
    expected_count: int, checkpoint: Path,
) -> Path:
    generation_dir = job_root / "generation"
    alias = job_root / "best_validation.ckpt"
    if not alias.exists():
        alias.symlink_to(checkpoint.resolve())
    command = [
        str(python),
        "src/main.py",
        *overrides,
        f"general.name=generate_{job_root.parent.name}_{job_root.name}",
        f"general.test_only={alias}",
        f"general.final_model_samples_to_generate={expected_count}",
        "general.final_model_samples_to_save=0",
        "general.final_model_chains_to_save=0",
        f"hydra.run.dir={generation_dir}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(GRAPH_EVAL_SRC), str(defog_root), env.get("PYTHONPATH", "")]
    )
    run_command(command, cwd=defog_root, env=env)
    generated = generation_dir / "generated_graphs.pt"
    if not generated.is_file() or not generated.with_suffix(".pt.json").is_file():
        raise RuntimeError(f"DeFoG did not create a safe generated collection: {generated}")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--defog-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stage", choices=("all", "train", "generate"), default="all")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    campaign_path = args.campaign.expanduser().resolve()
    manifest = load_yaml(manifest_path)
    campaign = load_yaml(campaign_path)
    dataset = args.dataset.upper()
    seed = int(args.seed)
    if dataset not in manifest["datasets"] or dataset not in campaign["datasets"]:
        raise ValueError(f"Dataset is outside the frozen campaign: {dataset}")
    if seed not in [int(value) for value in manifest["protocol"]["training_seeds"]]:
        raise ValueError(f"Training seed is outside the frozen campaign: {seed}")
    defog_root = args.defog_root.expanduser().resolve()
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=defog_root, text=True
    ).strip()
    expected_commit = campaign["defog_commit"]
    if actual_commit != expected_commit:
        raise RuntimeError(f"DeFoG commit {actual_commit}; expected {expected_commit}")
    if manifest["repositories"]["defog_benchmark_commit"] != expected_commit:
        raise RuntimeError("Manifest and worker campaign pin different DeFoG commits")

    artifact_root = args.artifact_root.expanduser().resolve()
    job_root = args.run_root.expanduser().resolve() / dataset.lower() / f"seed_{seed}"
    job_root.mkdir(parents=True, exist_ok=True)
    spec = manifest["datasets"][dataset]
    schedule = campaign["datasets"][dataset]
    overrides = common_overrides(
        dataset=dataset,
        seed=seed,
        spec=spec,
        schedule=schedule,
        artifact_root=artifact_root,
        defog_commit=expected_commit,
    )
    record = {
        "schema_version": campaign["schema_version"],
        "dataset": dataset,
        "training_seed": seed,
        "generation_seed": manifest["protocol"]["generation_seed"],
        "defog_commit": expected_commit,
        "manifest_sha256": sha256_file(manifest_path),
        "campaign_sha256": sha256_file(campaign_path),
        "schedule": schedule,
        "host": os.uname().nodename,
        "started_at": utc_now(),
        "status": "running",
    }
    record_path = job_root / "job_record.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    try:
        checkpoint = None if args.checkpoint is None else args.checkpoint.expanduser().resolve()
        if args.stage in ("all", "train"):
            checkpoint = train(
                python=args.python.expanduser().resolve(),
                defog_root=defog_root,
                job_root=job_root,
                overrides=overrides,
                schedule=schedule,
            )
        if args.stage in ("all", "generate"):
            if checkpoint is None:
                alias = job_root / "best_validation.ckpt"
                if not alias.exists():
                    raise RuntimeError("Generation requires --checkpoint or a completed train stage")
                checkpoint = alias.resolve()
            generated = generate(
                python=args.python.expanduser().resolve(),
                defog_root=defog_root,
                job_root=job_root,
                overrides=overrides,
                expected_count=int(spec["accepted_counts"]["reference"]),
                checkpoint=checkpoint,
            )
            destination = (
                artifact_root / dataset.lower() / "generated" / f"seed_{seed}"
            )
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(generated, destination / "generated_graphs.pt")
            shutil.copy2(
                generated.with_suffix(".pt.json"),
                destination / "generated_graphs.pt.json",
            )
            verification_spec = {**spec, "defog_commit": expected_commit}
            verified = verify_generated_artifact(
                destination / "generated_graphs.pt",
                dataset=dataset,
                training_seed=seed,
                spec=verification_spec,
                protocol=manifest["protocol"],
            )
            record["generated"] = verified
        record["checkpoint"] = str(checkpoint) if checkpoint else None
        record["checkpoint_sha256"] = sha256_file(checkpoint) if checkpoint else None
        record["status"] = "complete"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_at"] = utc_now()
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
