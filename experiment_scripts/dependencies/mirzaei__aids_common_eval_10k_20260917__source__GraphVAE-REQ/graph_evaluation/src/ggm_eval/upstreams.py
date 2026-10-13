"""Pinned upstream repository discovery.

The contrastive NeurIPS implementation is intentionally not copied into this
repository.  Its public GitHub repository currently has no explicit license.
Users provide a checkout, and the adapter verifies the expected source files
and revision before executing it in a separate interpreter.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


CONTRASTIVE_REPOSITORY = (
    "https://github.com/hamed1375/"
    "Self-Supervised-Models-for-GGM-Evaluation"
)
CONTRASTIVE_PIN = "fb6bc26237eb21d7617fd41b22b4bb26ab29bf95"
CONTRASTIVE_REQUIRED_FILES = (
    "GIN_train_pyg.py",
    "data_utils.py",
    "evaluation/gin_evaluation.py",
    "evaluation/models/gin/gin_pyg.py",
)


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_is_dirty(path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def validate_contrastive_upstream(
    path,
    *,
    allow_unpinned: bool = False,
) -> dict:
    """Validate the external checkout and return provenance metadata."""

    repository = Path(path).expanduser().resolve()
    if not repository.is_dir():
        raise FileNotFoundError(
            f"Contrastive evaluator checkout not found: {repository}"
        )
    missing = [
        relative
        for relative in CONTRASTIVE_REQUIRED_FILES
        if not (repository / relative).is_file()
    ]
    if missing:
        raise ValueError(
            f"{repository} is not the expected evaluator repository; missing "
            f"{missing}."
        )

    revision = _git_revision(repository)
    worktree_dirty = _git_is_dirty(repository)
    if (
        revision != CONTRASTIVE_PIN or worktree_dirty is True
    ) and not allow_unpinned:
        raise ValueError(
            "Contrastive evaluator checkout is not the clean pinned source. "
            f"Expected {CONTRASTIVE_PIN}, got {revision or 'unknown'}, "
            f"worktree_dirty={worktree_dirty}. Use "
            "--allow-unpinned-upstream only for an intentional experiment."
        )
    return {
        "repository": CONTRASTIVE_REPOSITORY,
        "checkout": str(repository),
        "revision": revision,
        "expected_revision": CONTRASTIVE_PIN,
        "revision_matches": revision == CONTRASTIVE_PIN,
        "worktree_dirty": worktree_dirty,
    }


def validate_legacy_repository(path) -> dict:
    """Validate a GraphVAE-REQ checkout containing the legacy evaluator."""

    repository = Path(path).expanduser().resolve()
    required = (
        repository / "eval" / "attributed_gin.py",
        repository
        / "third_party"
        / "ggmeval"
        / "evaluation"
        / "gin_evaluation.py",
    )
    missing = [
        str(item.relative_to(repository))
        for item in required
        if not item.is_file()
    ]
    if missing:
        raise ValueError(
            f"{repository} does not contain the legacy evaluator; missing "
            f"{missing}."
        )
    return {
        "checkout": str(repository),
        "revision": _git_revision(repository),
        "worktree_dirty": _git_is_dirty(repository),
    }
