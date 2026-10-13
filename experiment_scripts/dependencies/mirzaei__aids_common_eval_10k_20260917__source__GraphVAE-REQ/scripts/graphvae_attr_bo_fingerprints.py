#!/usr/bin/env python3
"""Canonical fingerprints for distributed GraphVAE Attr-F1PR studies.

The serialization in this module is deliberately versioned and framed.  A
change to any encoding rule must use a new ``FINGERPRINT_SCHEMA_VERSION`` so a
running study cannot silently mix old and new identities.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FINGERPRINT_SCHEMA_VERSION = "graphvae-attr-bo-fingerprint-v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _frame(tag: str, payload: bytes) -> bytes:
    tag_bytes = tag.encode("utf-8")
    return (
        len(tag_bytes).to_bytes(4, "big")
        + tag_bytes
        + len(payload).to_bytes(8, "big")
        + payload
    )


def framed_sha256(domain: str, fields: Iterable[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    digest.update(_frame("schema", FINGERPRINT_SCHEMA_VERSION.encode("utf-8")))
    digest.update(_frame("domain", domain.encode("utf-8")))
    for tag, payload in fields:
        digest.update(_frame(tag, payload))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_dense_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Object arrays have no canonical binary fingerprint.")
    dtype = array.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        dtype = dtype.newbyteorder("<")
        array = array.astype(dtype, copy=False)
    elif dtype.byteorder == "=":
        dtype = dtype.newbyteorder("<")
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def array_fingerprint(value: Any) -> str:
    """Fingerprint a dense or scipy-like sparse array canonically."""

    if hasattr(value, "tocoo"):
        coo = value.tocoo(copy=True)
        if hasattr(coo, "sum_duplicates"):
            coo.sum_duplicates()
        if hasattr(coo, "eliminate_zeros"):
            coo.eliminate_zeros()
        rows = np.asarray(coo.row, dtype=np.int64)
        cols = np.asarray(coo.col, dtype=np.int64)
        data = _normalized_dense_array(coo.data)
        order = np.lexsort((cols, rows))
        rows = np.ascontiguousarray(rows[order].astype("<i8", copy=False))
        cols = np.ascontiguousarray(cols[order].astype("<i8", copy=False))
        data = np.ascontiguousarray(data[order])
        return framed_sha256(
            "sparse-array",
            (
                ("format", b"coo-lexicographic"),
                ("dtype", data.dtype.str.encode("ascii")),
                ("shape", canonical_json_bytes(list(coo.shape))),
                ("rows", rows.tobytes(order="C")),
                ("cols", cols.tobytes(order="C")),
                ("data", data.tobytes(order="C")),
            ),
        )

    array = _normalized_dense_array(value)
    return framed_sha256(
        "dense-array",
        (
            ("dtype", array.dtype.str.encode("ascii")),
            ("shape", canonical_json_bytes(list(array.shape))),
            ("data", array.tobytes(order="C")),
        ),
    )


def graph_fingerprint(
    adjacency: Any,
    node_attributes: Any,
    edge_attributes: Any,
    *,
    relation_axes: Any = None,
) -> str:
    fields = [
        ("adjacency", array_fingerprint(adjacency).encode("ascii")),
        ("node_attributes", array_fingerprint(node_attributes).encode("ascii")),
    ]
    if edge_attributes is None:
        fields.append(("edge_attributes", b"none"))
    else:
        fields.append(
            ("edge_attributes", array_fingerprint(edge_attributes).encode("ascii"))
        )
    fields.append(("relation_axes", canonical_json_bytes(_json_safe(relation_axes))))
    return framed_sha256("cached-graph", fields)


def split_fingerprint(graph_hashes: Sequence[str]) -> str:
    return framed_sha256(
        "cached-graph-split",
        (("graph", value.encode("ascii")) for value in graph_hashes),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def feature_schema_payload(
    onehot_info: Mapping[Any, Any] | None,
    *,
    total_dimension: int,
    dtype: str,
) -> dict[str, Any]:
    """Normalize ordered one-hot channel metadata into the contract schema."""

    channels = []
    for raw_index, raw_metadata in sorted(
        (onehot_info or {}).items(), key=lambda item: int(item[0])
    ):
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {
            "meaning": raw_metadata
        }
        channels.append(
            {
                "channel": int(raw_index),
                "group": str(metadata.get("feature_name", metadata.get("group", "unknown"))),
                "meaning": _json_safe(
                    metadata.get("value", metadata.get("meaning", metadata.get("label")))
                ),
                "metadata": _json_safe(metadata),
            }
        )
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "encoding": "ordered-grouped-one-hot",
        "dtype": str(dtype),
        "total_dimension": int(total_dimension),
        "channels": channels,
    }


def feature_schema_fingerprint(schema: Mapping[str, Any]) -> str:
    return framed_sha256(
        "feature-schema", (("canonical-json", canonical_json_bytes(schema)),)
    )


def deployment_manifest(
    repo_root: Path,
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Hash tracked deployment files in the same order on every host."""

    root = Path(repo_root).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status_output = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1"], text=True
    )
    if require_clean and status_output:
        raise RuntimeError("Refusing deployment fingerprint for a dirty Git worktree.")
    raw_paths = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"]
    )
    paths = sorted(
        (
            part.decode("utf-8")
            for part in raw_paths.split(b"\0")
            if part
            and part.decode("utf-8").split("/", 1)[0]
            not in {
                "data_raw",
                "cache_datasets",
                "cache_motifs",
                "cache_motifs_archive",
                "cache_motifs_old_lobster",
                "collected_runs",
                "graph_evaluation_inputs",
                "runs",
                "results",
                "reports",
            }
        ),
        key=lambda value: value.encode("utf-8"),
    )
    fields = []
    files = []
    for relative in paths:
        normalized = Path(relative).as_posix()
        path = root / normalized
        if path.is_symlink():
            raise RuntimeError(f"Deployment manifest rejects symlink: {normalized}")
        if not path.is_file():
            raise RuntimeError(f"Tracked deployment file is missing: {normalized}")
        content = path.read_bytes()
        executable = bool(path.stat().st_mode & stat.S_IXUSR)
        record = {
            "path": normalized,
            "executable": executable,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        files.append(record)
        fields.extend(
            (
                ("path", normalized.encode("utf-8")),
                ("executable", b"1" if executable else b"0"),
                ("content", content),
            )
        )
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "git_commit": commit,
        "clean_worktree": not bool(status_output),
        "tree_sha256": framed_sha256("deployment-tree", fields),
        "files": files,
    }


def verify_deployment_manifest(repo_root: Path, manifest: Mapping[str, Any]) -> None:
    root = Path(repo_root).resolve()
    fields = []
    for record in manifest.get("files", []):
        relative = str(record["path"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Deployment path escapes root: {relative}") from exc
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Deployment file missing or symlinked: {relative}")
        content = path.read_bytes()
        executable = bool(path.stat().st_mode & stat.S_IXUSR)
        if (
            len(content) != int(record["size"])
            or hashlib.sha256(content).hexdigest() != record["sha256"]
            or executable != bool(record["executable"])
        ):
            raise RuntimeError(f"Deployment file mismatch: {relative}")
        fields.extend(
            (
                ("path", relative.encode("utf-8")),
                ("executable", b"1" if executable else b"0"),
                ("content", content),
            )
        )
    actual = framed_sha256("deployment-tree", fields)
    if actual != manifest.get("tree_sha256"):
        raise RuntimeError("Deployment tree fingerprint mismatch.")


def sampler_seed(study_seed: int, dispatch_sequence: int) -> int:
    material = (
        f"graphvae-attr-f1pr-sampler-v1\0{int(study_seed)}\0"
        f"{int(dispatch_sequence)}"
    )
    return int.from_bytes(
        hashlib.sha256(material.encode("utf-8")).digest()[:4], "big"
    )


def output_root_fingerprint(path: Path) -> str:
    return framed_sha256(
        "controller-output-root",
        (("absolute-path", os.fsencode(str(Path(path).resolve()))),),
    )


__all__ = [
    "FINGERPRINT_SCHEMA_VERSION",
    "array_fingerprint",
    "canonical_json_bytes",
    "deployment_manifest",
    "feature_schema_fingerprint",
    "feature_schema_payload",
    "framed_sha256",
    "graph_fingerprint",
    "output_root_fingerprint",
    "sampler_seed",
    "sha256_file",
    "split_fingerprint",
    "verify_deployment_manifest",
]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verify-manifest", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.verify_manifest is not None:
        payload = json.loads(args.verify_manifest.read_text(encoding="utf-8"))
        verify_deployment_manifest(args.deployment_root, payload)
        return 0
    if args.output is None:
        parser.error("--output is required when creating a deployment manifest")
    payload = deployment_manifest(args.deployment_root, require_clean=not args.allow_dirty)
    # Keep this standalone helper independent of the Optuna module.
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
