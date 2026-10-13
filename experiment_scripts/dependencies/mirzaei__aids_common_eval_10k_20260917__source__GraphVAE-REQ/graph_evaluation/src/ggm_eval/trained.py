"""Convenience access to the bundled, contrastively trained GraphCL-GINs.

The checkpoint bundle is a small, immutable model registry.  This module
normalizes dataset names, verifies checkpoint sizes and SHA-256 digests, and
then delegates evaluation to :func:`ggm_eval.runner.evaluate_contrastive_checkpoints`.
It does not copy, import, or modify the research evaluator implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .datasets import normalize_dataset_name
from .runner import evaluate_contrastive_checkpoints
from .upstreams import CONTRASTIVE_REQUIRED_FILES


TRAINED_MODELS_DIR = Path(__file__).resolve().parent / "trained_models"
TRAINED_MODELS_MANIFEST = TRAINED_MODELS_DIR / "manifest.json"
CONTRASTIVE_REPOSITORY_ENV = "GGM_EVAL_CONTRASTIVE_REPO"
CONTRASTIVE_CHECKOUT_NAME = "Self-Supervised-Models-for-GGM-Evaluation"


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    """Load and minimally validate the installed checkpoint manifest."""

    if not TRAINED_MODELS_MANIFEST.is_file():
        raise FileNotFoundError(
            "Bundled checkpoint manifest is missing: "
            f"{TRAINED_MODELS_MANIFEST}. Reinstall graph-ggm-evaluation "
            "with package data enabled."
        )
    payload = json.loads(
        TRAINED_MODELS_MANIFEST.read_text(encoding="utf-8")
    )
    if payload.get("format") != "ggm-eval-trained-model-bundle":
        raise ValueError(
            f"Unsupported trained-model manifest format in "
            f"{TRAINED_MODELS_MANIFEST}."
        )
    if payload.get("version") != 1:
        raise ValueError(
            "Unsupported trained-model manifest version "
            f"{payload.get('version')!r}; expected 1."
        )
    if payload.get("encoder") != "graphcl" or payload.get("architecture") != "gin":
        raise ValueError(
            "The installed manifest is not the expected GraphCL-GIN bundle."
        )
    if not isinstance(payload.get("datasets"), dict):
        raise ValueError("The trained-model manifest has no dataset registry.")
    return payload


def available_trained_datasets() -> Tuple[str, ...]:
    """Return canonical dataset names with bundled GraphCL-GIN checkpoints."""

    return tuple(_load_manifest()["datasets"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_trained_checkpoints(
    dataset: str,
    *,
    seeds: Optional[Iterable[int]] = None,
) -> Tuple[Path, ...]:
    """Resolve and integrity-check bundled GraphCL-GIN checkpoints.

    All available independent seeds are selected when ``seeds`` is omitted.
    The returned paths point into the installed package and can be passed
    directly to the existing isolated evaluator runner.
    """

    canonical = normalize_dataset_name(dataset)
    manifest = _load_manifest()
    dataset_entry = manifest["datasets"].get(canonical)
    if dataset_entry is None:
        raise ValueError(
            f"No trained GraphCL-GIN is bundled for {canonical}. Available "
            f"datasets: {list(available_trained_datasets())}."
        )

    entries_by_seed = {
        int(entry["seed"]): entry
        for entry in dataset_entry.get("checkpoints", ())
    }
    if not entries_by_seed:
        raise ValueError(f"The checkpoint registry for {canonical} is empty.")

    if seeds is None:
        selected_seeds = list(entries_by_seed)
    else:
        selected_seeds = [int(seed) for seed in seeds]
        if not selected_seeds:
            raise ValueError("At least one trained encoder seed is required.")
        if len(set(selected_seeds)) != len(selected_seeds):
            raise ValueError("Trained encoder seeds must be unique.")

    missing = [
        seed for seed in selected_seeds if seed not in entries_by_seed
    ]
    if missing:
        raise ValueError(
            f"No {canonical} checkpoint is bundled for seed(s) {missing}. "
            f"Available seeds: {list(entries_by_seed)}."
        )

    root = TRAINED_MODELS_DIR.resolve()
    resolved = []
    for seed in selected_seeds:
        entry = entries_by_seed[seed]
        checkpoint = (TRAINED_MODELS_DIR / entry["path"]).resolve()
        if root != checkpoint and root not in checkpoint.parents:
            raise ValueError(
                f"Checkpoint path escapes the trained-model bundle: "
                f"{entry['path']!r}."
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Bundled checkpoint is missing: {checkpoint}. Reinstall "
                "graph-ggm-evaluation with package data enabled."
            )
        actual_size = checkpoint.stat().st_size
        expected_size = int(entry["size_bytes"])
        if actual_size != expected_size:
            raise ValueError(
                f"Bundled checkpoint size mismatch for {checkpoint}: "
                f"expected {expected_size}, got {actual_size}."
            )
        actual_digest = _sha256(checkpoint)
        if actual_digest != entry["sha256"]:
            raise ValueError(
                f"Bundled checkpoint SHA-256 mismatch for {checkpoint}: "
                f"expected {entry['sha256']}, got {actual_digest}."
            )
        resolved.append(checkpoint)
    return tuple(resolved)


def _looks_like_contrastive_checkout(path: Path) -> bool:
    return path.is_dir() and all(
        (path / relative).is_file()
        for relative in CONTRASTIVE_REQUIRED_FILES
    )


def resolve_contrastive_upstream(upstream_repository=None) -> Path:
    """Resolve the external research checkout used by the frozen encoders.

    Resolution order is an explicit argument, the
    ``GGM_EVAL_CONTRASTIVE_REPO`` environment variable, then common sibling
    checkout locations.  The existing runner performs the authoritative
    source-file, Git revision, and dirty-worktree validation.
    """

    if upstream_repository is not None:
        return Path(upstream_repository).expanduser().resolve()

    configured = os.environ.get(CONTRASTIVE_REPOSITORY_ENV)
    if configured:
        return Path(configured).expanduser().resolve()

    source_repository = Path(__file__).resolve().parents[3]
    candidates = (
        Path.cwd() / CONTRASTIVE_CHECKOUT_NAME,
        Path.cwd().parent / CONTRASTIVE_CHECKOUT_NAME,
        source_repository.parent / CONTRASTIVE_CHECKOUT_NAME,
        source_repository.parent / "upstreams" / CONTRASTIVE_CHECKOUT_NAME,
    )
    visited = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in visited:
            continue
        visited.add(resolved)
        if _looks_like_contrastive_checkout(resolved):
            return resolved

    raise FileNotFoundError(
        "Could not find the external contrastive evaluator checkout. Pass "
        "upstream_repository=..., set "
        f"{CONTRASTIVE_REPOSITORY_ENV}, or place "
        f"{CONTRASTIVE_CHECKOUT_NAME} beside the project checkout."
    )


def evaluate_with_trained_gnns(
    *,
    dataset: str,
    generated,
    reference,
    output_dir,
    upstream_repository=None,
    seeds: Optional[Iterable[int]] = None,
    python_executable=sys.executable,
    device: str = "auto",
    nearest_k: int = 5,
    max_graphs: int = 0,
    trusted_input: bool = False,
    allow_unpinned_upstream: bool = False,
) -> dict:
    """Evaluate generated graphs with the bundled trained GraphCL-GINs.

    The dataset selects the matching feature schema and all three independent
    encoder seeds by default.  Graph loading, schema validation, embedding,
    metrics, aggregation, and report writing remain in the existing evaluator
    worker and runner.
    """

    canonical = normalize_dataset_name(dataset)
    checkpoints = resolve_trained_checkpoints(canonical, seeds=seeds)
    upstream = resolve_contrastive_upstream(upstream_repository)
    return evaluate_contrastive_checkpoints(
        generated=generated,
        reference=reference,
        checkpoints=checkpoints,
        upstream_repository=upstream,
        output_dir=output_dir,
        python_executable=python_executable,
        device=device,
        nearest_k=nearest_k,
        max_graphs=max_graphs,
        trusted_input=trusted_input,
        allow_unpinned_upstream=allow_unpinned_upstream,
    )
