#!/usr/bin/env python3
"""Fail-closed checks for the shared GraphVAE/DeFoG evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


class ProtocolError(RuntimeError):
    """Raised when an input does not satisfy the frozen protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected a YAML mapping: {path}")
    return payload


def nested_value(payload: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for component in dotted_key.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ProtocolError(f"Missing configuration key: {dotted_key}")
        value = value[component]
    return value


def split_payload(protocol: Mapping[str, Any]) -> dict[str, list[int]]:
    dataset = protocol["dataset"]
    split = protocol["split"]
    graph_ids = list(range(int(dataset["total_graphs_after_filter"])))
    random.Random(int(split["seed"])).shuffle(graph_ids)
    train_count = int(split["counts"]["train"])
    validation_count = int(split["counts"]["validation"])
    return {
        "train": graph_ids[:train_count],
        "validation": graph_ids[train_count : train_count + validation_count],
        "test": graph_ids[train_count + validation_count :],
    }


def split_digest(protocol: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        split_payload(protocol), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ProtocolError(f"Git check failed for {repo}: {detail}")
    return result.stdout.strip()


def verify_repository(
    repo: Path,
    specification: Mapping[str, Any],
    *,
    allow_dirty: bool,
) -> str:
    repo = repo.resolve()
    if not (repo / ".git").exists():
        raise ProtocolError(f"Not a Git repository: {repo}")
    head = git(repo, "rev-parse", "HEAD")

    required_head = specification.get("required_head")
    if required_head and head != required_head:
        raise ProtocolError(
            f"Wrong commit for {repo}: expected {required_head}, got {head}"
        )
    required_ancestor = specification.get("required_ancestor")
    if required_ancestor:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                str(required_ancestor),
                head,
            ],
            check=False,
        )
        if result.returncode:
            raise ProtocolError(
                f"Required ancestor {required_ancestor} is absent from {repo}"
            )

    dirty = git(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty and not allow_dirty:
        raise ProtocolError(f"Repository is dirty: {repo}")

    for record in specification.get("files", []):
        candidate = repo / record["path"]
        if not candidate.is_file():
            raise ProtocolError(f"Required file is missing: {candidate}")
        actual = sha256_file(candidate)
        if actual != record["sha256"]:
            raise ProtocolError(
                f"File digest mismatch for {candidate}: "
                f"expected {record['sha256']}, got {actual}"
            )
    return head


def verify_native_configs(
    graphvae_repo: Path,
    defog_repo: Path,
    protocol: Mapping[str, Any],
) -> None:
    for name, specification in protocol["native_configs"].items():
        base = graphvae_repo if name == "graphvae" else defog_repo
        path = base / specification["path"]
        payload = load_yaml(path)
        for dotted_key, expected in specification["expected"].items():
            actual = nested_value(payload, dotted_key)
            if actual != expected:
                raise ProtocolError(
                    f"Unexpected {path}:{dotted_key}: "
                    f"expected {expected!r}, got {actual!r}"
                )


def artifact_manifest(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        raise ProtocolError(f"Artifact or manifest is missing: {path}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if payload.get("format") != "ggm-eval-pyg-tensors":
        raise ProtocolError(f"Unsupported graph artifact format: {sidecar}")
    return payload


def verify_artifact(
    path: Path,
    protocol: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    manifest = artifact_manifest(path)
    metadata = manifest.get("metadata") or {}
    summary = manifest.get("summary") or {}
    expected_metadata = {
        "dataset": protocol["dataset"]["name"],
        "feature_schema": protocol["dataset"]["feature_schema"],
        "split": split,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ProtocolError(
                f"Unexpected {key} in {path}: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )
    expected_summary = {
        "graph_count": protocol["generation"]["graph_count"],
        "node_feature_dim": protocol["dataset"]["node_feature_dim"],
        "edge_feature_dim": protocol["dataset"]["edge_feature_dim"],
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ProtocolError(
                f"Unexpected {key} in {path}: "
                f"expected {expected!r}, got {summary.get(key)!r}"
            )
    if not manifest.get("collection_sha256"):
        raise ProtocolError(f"Collection digest is missing: {path}")
    return manifest


def verify_loaded_artifact(path: Path, expected_digest: str) -> None:
    graph_evaluation_src = Path(__file__).resolve().parents[2] / "graph_evaluation" / "src"
    sys.path.insert(0, str(graph_evaluation_src))
    from ggm_eval import collection_digest, load_pyg_collection  # type: ignore

    graphs = load_pyg_collection(path)
    actual = collection_digest(graphs)
    if actual != expected_digest:
        raise ProtocolError(
            f"Loaded collection digest mismatch for {path}: "
            f"expected {expected_digest}, got {actual}"
        )


def parse_args() -> argparse.Namespace:
    graphvae_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("protocol.yaml"),
    )
    parser.add_argument("--graphvae-repo", type=Path, default=graphvae_default)
    parser.add_argument("--defog-repo", type=Path, default=graphvae_default.parent / "DeFoG")
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Diagnostic only; final evaluations must use clean repositories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_yaml(args.protocol.resolve())
    graphvae_repo = args.graphvae_repo.resolve()
    defog_repo = args.defog_repo.resolve()

    expected_split_digest = protocol["split"]["index_sha256"]
    actual_split_digest = split_digest(protocol)
    if actual_split_digest != expected_split_digest:
        raise ProtocolError(
            "Protocol split digest is internally inconsistent: "
            f"expected {expected_split_digest}, got {actual_split_digest}"
        )

    graphvae_head = verify_repository(
        graphvae_repo,
        protocol["repositories"]["graphvae"],
        allow_dirty=args.allow_dirty,
    )
    defog_head = verify_repository(
        defog_repo,
        protocol["repositories"]["defog"],
        allow_dirty=args.allow_dirty,
    )
    verify_native_configs(graphvae_repo, defog_repo, protocol)

    artifacts: dict[str, Any] = {}
    for label, path, split in (
        ("generated", args.generated, "generated"),
        ("reference", args.reference, "test"),
    ):
        if path is None:
            continue
        manifest = verify_artifact(path.resolve(), protocol, split=split)
        verify_loaded_artifact(path.resolve(), manifest["collection_sha256"])
        artifacts[label] = manifest["collection_sha256"]

    print(
        json.dumps(
            {
                "protocol": protocol["name"],
                "graphvae_commit": graphvae_head,
                "defog_commit": defog_head,
                "split_sha256": actual_split_digest,
                "artifacts": artifacts,
                "status": "verified",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtocolError as error:
        print(f"protocol verification failed: {error}", file=sys.stderr)
        raise SystemExit(2)
