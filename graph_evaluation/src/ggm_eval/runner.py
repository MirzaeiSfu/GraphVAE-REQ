"""Parent-process orchestration for collision-free evaluator execution.

The two research repositories both expose a top-level package named
``evaluation``.  They must never be imported into one interpreter.  This
module starts a fresh worker process for each engine, pins its repository
path, captures logs, and combines only JSON results in the parent process.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from .reporting import (
    aggregate_checkpoint_results,
    write_contrastive_markdown,
    write_json,
    write_legacy_markdown,
)
from .upstreams import (
    validate_contrastive_upstream,
    validate_legacy_repository,
)


PACKAGE_SRC = Path(__file__).resolve().parents[1]


def _worker_environment() -> dict:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    parts = [str(PACKAGE_SRC)]
    if existing:
        parts.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(parts)
    return environment


def _run_worker(
    arguments: List[str],
    *,
    python_executable,
    runtime_dir: Path,
):
    runtime_dir.mkdir(parents=True, exist_ok=True)
    requested_python = str(python_executable)
    resolved_python = (
        shutil.which(requested_python)
        if not Path(requested_python).expanduser().is_absolute()
        else str(Path(requested_python).expanduser().resolve())
    )
    if not resolved_python:
        raise FileNotFoundError(
            f"Python interpreter not found: {python_executable}"
        )
    command = [
        resolved_python,
        "-m",
        "ggm_eval.worker",
        *arguments,
    ]
    result = subprocess.run(
        command,
        cwd=str(runtime_dir),
        env=_worker_environment(),
        text=True,
        capture_output=True,
    )
    (runtime_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (runtime_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"Evaluator worker failed with exit code {result.returncode}. "
            f"See {runtime_dir / 'stderr.log'}."
        )


def train_contrastive_encoders(
    *,
    graphs,
    upstream_repository,
    output_dir,
    encoder: str,
    feature_mode: str,
    seeds: Iterable[int],
    python_executable=sys.executable,
    device: str = "cpu",
    num_layers: int = 3,
    hidden_dim: int = 32,
    epochs: int = 100,
    init: str = "orthogonal",
    limit_lipschitz: bool = True,
    lipschitz_factor: float = 1.0,
    trusted_input: bool = False,
    allow_unpinned_upstream: bool = False,
) -> dict:
    """Train one upstream encoder checkpoint for each independent seed."""

    upstream = validate_contrastive_upstream(
        upstream_repository,
        allow_unpinned=allow_unpinned_upstream,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("At least one encoder seed is required.")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("Encoder seeds must be unique.")
    manifests = []
    for raw_seed in seed_values:
        seed = int(raw_seed)
        seed_dir = destination / f"seed_{seed}"
        result_path = seed_dir / "training.json"
        arguments = [
            "contrastive-train",
            "--graphs",
            str(Path(graphs).expanduser().resolve()),
            "--upstream-repo",
            upstream["checkout"],
            "--output-dir",
            str(seed_dir),
            "--encoder",
            encoder,
            "--feature-mode",
            feature_mode,
            "--seed",
            str(seed),
            "--device",
            device,
            "--num-layers",
            str(num_layers),
            "--hidden-dim",
            str(hidden_dim),
            "--epochs",
            str(epochs),
            "--init",
            init,
            "--lipschitz-factor",
            str(lipschitz_factor),
        ]
        if limit_lipschitz:
            arguments.append("--limit-lipschitz")
        if trusted_input:
            arguments.append("--trusted-input")
        if allow_unpinned_upstream:
            arguments.append("--allow-unpinned-upstream")
        _run_worker(
            arguments,
            python_executable=python_executable,
            runtime_dir=seed_dir / "runtime",
        )
        manifests.append(json.loads(result_path.read_text(encoding="utf-8")))

    payload = {
        "engine": "contrastive-pyg-upstream",
        "encoder": encoder,
        "feature_mode": feature_mode,
        "upstream": upstream,
        "seeds": seed_values,
        "checkpoints": [item["checkpoint"] for item in manifests],
        "training_runs": manifests,
    }
    write_json(destination / "training_summary.json", payload)
    return payload


def evaluate_contrastive_checkpoints(
    *,
    generated,
    reference,
    checkpoints: Iterable,
    upstream_repository,
    output_dir,
    python_executable=sys.executable,
    device: str = "cpu",
    nearest_k: int = 5,
    max_graphs: int = 0,
    trusted_input: bool = False,
    allow_unpinned_upstream: bool = False,
) -> dict:
    """Evaluate the same collections with one or more frozen encoders."""

    upstream = validate_contrastive_upstream(
        upstream_repository,
        allow_unpinned=allow_unpinned_upstream,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    checkpoint_paths = [
        Path(checkpoint).expanduser().resolve() for checkpoint in checkpoints
    ]
    if not checkpoint_paths:
        raise ValueError("At least one --checkpoint is required.")
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise ValueError(
            "Checkpoint paths must be unique; repeating a checkpoint does "
            "not measure encoder uncertainty."
        )

    for index, checkpoint in enumerate(checkpoint_paths):
        checkpoint_dir = destination / f"checkpoint_{index:03d}"
        result_path = checkpoint_dir / "evaluation.json"
        arguments = [
            "contrastive-evaluate",
            "--generated",
            str(Path(generated).expanduser().resolve()),
            "--reference",
            str(Path(reference).expanduser().resolve()),
            "--checkpoint",
            str(checkpoint),
            "--upstream-repo",
            upstream["checkout"],
            "--output",
            str(result_path),
            "--device",
            device,
            "--nearest-k",
            str(nearest_k),
            "--max-graphs",
            str(max_graphs),
        ]
        if trusted_input:
            arguments.append("--trusted-input")
        if allow_unpinned_upstream:
            arguments.append("--allow-unpinned-upstream")
        _run_worker(
            arguments,
            python_executable=python_executable,
            runtime_dir=checkpoint_dir / "runtime",
        )
        results.append(json.loads(result_path.read_text(encoding="utf-8")))

    payload = aggregate_checkpoint_results(results)
    payload["upstream"] = upstream
    payload["generated"] = str(Path(generated).expanduser().resolve())
    payload["reference"] = str(Path(reference).expanduser().resolve())
    write_json(destination / "evaluation.json", payload)
    write_contrastive_markdown(destination / "evaluation.md", payload)
    return payload


def evaluate_legacy_random_gin(
    *,
    generated,
    reference,
    legacy_repository,
    output_dir,
    python_executable=sys.executable,
    modes: Iterable[str] | None = None,
    repeats: int = 10,
    evaluator_seed: int = 0,
    nearest_k: int = 5,
    max_graphs: int = 0,
    device: str = "cpu",
    trusted_input: bool = False,
) -> dict:
    """Run the existing DGL Random-GIN through a PyG-to-DGL adapter."""

    legacy = validate_legacy_repository(legacy_repository)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "evaluation.json"
    arguments = [
        "legacy-evaluate",
        "--generated",
        str(Path(generated).expanduser().resolve()),
        "--reference",
        str(Path(reference).expanduser().resolve()),
        "--legacy-repo",
        legacy["checkout"],
        "--output",
        str(result_path),
        "--repeats",
        str(repeats),
        "--evaluator-seed",
        str(evaluator_seed),
        "--nearest-k",
        str(nearest_k),
        "--max-graphs",
        str(max_graphs),
        "--device",
        device,
    ]
    if modes:
        arguments.extend(["--modes", *list(modes)])
    if trusted_input:
        arguments.append("--trusted-input")
    _run_worker(
        arguments,
        python_executable=python_executable,
        runtime_dir=destination / "runtime",
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["legacy_repository"] = legacy
    write_json(result_path, payload)
    write_legacy_markdown(destination / "evaluation.md", payload)
    return payload
