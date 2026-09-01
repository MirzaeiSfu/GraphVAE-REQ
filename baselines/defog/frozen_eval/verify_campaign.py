#!/usr/bin/env python3
"""Fail-closed verification for frozen DeFoG benchmark inputs and outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GRAPH_EVAL_SRC = ROOT / "graph_evaluation" / "src"
if str(GRAPH_EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(GRAPH_EVAL_SRC))

from ggm_eval import load_pyg_collection_with_metadata  # noqa: E402
from ggm_eval.contract import collection_digest, validate_collection  # noqa: E402


class CampaignVerificationError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignVerificationError(f"Invalid manifest: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_evaluator_files(manifest: dict) -> None:
    for record in manifest["evaluator"]["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            raise CampaignVerificationError(f"Evaluator file is missing: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise CampaignVerificationError(
                f"Evaluator hash mismatch for {record['path']}: {actual}"
            )


def verify_graphvae_revision(manifest: dict) -> str:
    required = manifest["repositories"]["graphvae_required_ancestor"]
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", required, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode:
        raise CampaignVerificationError(
            f"GraphVAE HEAD does not contain required commit {required}"
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def artifact_path(artifact_root: Path, dataset: str, split: str) -> Path:
    filenames = {
        "train": "real_train_graphs.pt",
        "validation": "real_validation_graphs.pt",
        "reference": "real_test_graphs.pt",
    }
    return artifact_root / dataset.lower() / filenames[split]


def verify_reference_artifact(
    path: Path, dataset: str, split: str, spec: dict
) -> dict:
    graphs, metadata = load_pyg_collection_with_metadata(path, normalize=True)
    summary = validate_collection(graphs, name=f"{dataset} {split}").to_dict()
    actual_digest = collection_digest(graphs)
    expected_digest = spec["collection_sha256"][split]
    if actual_digest != expected_digest:
        raise CampaignVerificationError(
            f"{dataset} {split} collection digest {actual_digest}; "
            f"expected {expected_digest}"
        )
    expected = {
        "dataset": dataset,
        "split": split,
        "feature_mode": spec["feature_mode"],
        "feature_schema": spec["feature_schema"],
        "source_cache_sha256": spec["source_cache_sha256"],
        "source_split_fingerprint": spec["split_fingerprints"][split],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise CampaignVerificationError(
                f"{dataset} {split} metadata {key}={metadata.get(key)!r}; "
                f"expected {value!r}"
            )
    if summary["graph_count"] != int(spec["accepted_counts"][split]):
        raise CampaignVerificationError(f"{dataset} {split} count changed")
    if summary["node_feature_dim"] != int(spec["node_feature_dim"]):
        raise CampaignVerificationError(f"{dataset} node feature width changed")
    if summary["edge_feature_dim"] != 0:
        raise CampaignVerificationError(f"{dataset} unexpectedly has edge features")
    rejected = metadata.get("rejected")
    if not isinstance(rejected, list):
        raise CampaignVerificationError(f"{dataset} {split} lacks rejection records")
    if len(rejected) != int(spec["raw_counts"][split]) - summary["graph_count"]:
        raise CampaignVerificationError(f"{dataset} {split} rejection count changed")
    return {
        "path": str(path),
        "collection_sha256": actual_digest,
        "summary": summary,
        "rejected": rejected,
    }


def generated_path(artifact_root: Path, dataset: str, seed: int) -> Path:
    return (
        artifact_root
        / dataset.lower()
        / "generated"
        / f"seed_{seed}"
        / "generated_graphs.pt"
    )


def verify_generated_artifact(
    path: Path,
    *,
    dataset: str,
    training_seed: int,
    spec: dict,
    protocol: dict,
) -> dict:
    graphs, metadata = load_pyg_collection_with_metadata(path, normalize=True)
    summary = validate_collection(graphs, name=f"{dataset} generated seed {training_seed}").to_dict()
    expected_count = int(spec["accepted_counts"]["reference"])
    required = {
        "dataset": dataset,
        "split": "generated",
        "feature_mode": spec["feature_mode"],
        "feature_schema": spec["feature_schema"],
        "training_seed": training_seed,
        "generation_seed": int(protocol["generation_seed"]),
        "checkpoint_selection": "best_validation",
        "defog_commit": spec["defog_commit"],
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise CampaignVerificationError(
                f"{path} metadata {key}={metadata.get(key)!r}; expected {expected!r}"
            )
    for key in (
        "checkpoint_sha256",
        "generation_attempts",
        "accepted_count",
        "rejected_count",
    ):
        if key not in metadata:
            raise CampaignVerificationError(f"{path} lacks provenance field {key}")
    if summary["graph_count"] != expected_count:
        raise CampaignVerificationError(
            f"{path} has {summary['graph_count']} graphs; expected {expected_count}"
        )
    if int(metadata["accepted_count"]) != expected_count:
        raise CampaignVerificationError(f"{path} accepted_count is inconsistent")
    checkpoint_sha256 = metadata["checkpoint_sha256"]
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise CampaignVerificationError(f"{path} has an invalid checkpoint SHA-256")
    attempts = int(metadata["generation_attempts"])
    rejected = int(metadata["rejected_count"])
    if attempts != expected_count + rejected:
        raise CampaignVerificationError(
            f"{path} attempts must equal accepted plus rejected"
        )
    if summary["node_feature_dim"] != int(spec["node_feature_dim"]):
        raise CampaignVerificationError(f"{path} node feature width changed")
    if summary["edge_feature_dim"] != 0:
        raise CampaignVerificationError(f"{path} unexpectedly has edge features")
    return {
        "path": str(path),
        "collection_sha256": collection_digest(graphs),
        "summary": summary,
        "metadata": metadata,
    }


def verify_campaign(
    manifest_path: Path,
    artifact_root: Path,
    *,
    references_only: bool,
    datasets: list[str] | None = None,
) -> dict:
    manifest = load_yaml(manifest_path)
    verify_evaluator_files(manifest)
    graphvae_commit = verify_graphvae_revision(manifest)
    selected = datasets or list(manifest["datasets"])
    results = {}
    for dataset in selected:
        if dataset not in manifest["datasets"]:
            raise CampaignVerificationError(f"Unknown dataset: {dataset}")
        spec = manifest["datasets"][dataset]
        spec = {
            **spec,
            "defog_commit": manifest["repositories"]["defog_benchmark_commit"],
        }
        references = {
            split: verify_reference_artifact(
                artifact_path(artifact_root, dataset, split), dataset, split, spec
            )
            for split in ("train", "validation", "reference")
        }
        generated = {}
        if not references_only:
            for seed in manifest["protocol"]["training_seeds"]:
                generated[str(seed)] = verify_generated_artifact(
                    generated_path(artifact_root, dataset, int(seed)),
                    dataset=dataset,
                    training_seed=int(seed),
                    spec=spec,
                    protocol=manifest["protocol"],
                )
        results[dataset] = {"references": references, "generated": generated}
    return {
        "status": "verified",
        "schema_version": manifest["schema_version"],
        "graphvae_commit": graphvae_commit,
        "references_only": references_only,
        "datasets": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()
    result = verify_campaign(
        args.manifest.expanduser().resolve(),
        args.artifact_root.expanduser().resolve(),
        references_only=args.references_only,
        datasets=None if args.dataset is None else [value.upper() for value in args.dataset],
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
