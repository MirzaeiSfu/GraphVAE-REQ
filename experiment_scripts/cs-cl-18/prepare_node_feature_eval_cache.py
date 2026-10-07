#!/usr/bin/env python3
"""Create a compatibility copy of a legacy GraphVAE dataset cache."""

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--schema-version", default="dataset-cache-v4")
    parser.add_argument("--feature-schema", default=None)
    args = parser.parse_args()

    with args.source.open("rb") as handle:
        payload = pickle.load(handle)

    metadata = payload.setdefault("cache_metadata", {})
    metadata["cache_schema_version"] = args.schema_version
    if args.feature_schema is not None:
        metadata["feature_schema"] = args.feature_schema

    # Some legacy caches encode the absence of edge features as one None per
    # graph.  The attributed evaluator expects a single None for that case.
    for key in ("list_eoh_train", "list_eoh_val", "list_eoh_test"):
        values = payload.get(key)
        if isinstance(values, (list, tuple)) and all(value is None for value in values):
            payload[key] = None

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
