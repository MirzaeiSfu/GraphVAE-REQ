#!/usr/bin/env python3
"""Create a small deterministic CP_smoothed cache for QM9 sanity checking."""

import argparse
import json
import pickle
import random
from pathlib import Path


def normalized(value):
    if hasattr(value, "item"):
        value = value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("manifest")
    parser.add_argument("--per-rule", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    with open(args.source, "rb") as handle:
        data = pickle.load(handle)

    rng = random.Random(args.seed)
    sampled_sets = []
    manifest = []
    for index, rule in enumerate(data["rules"]):
        ordinary = list(data["values_full"][index])
        smoothed = data["values_smoothed_full"][index]
        cp_columns = list(data["value_columns"][index])
        smoothed_columns = data["value_smoothed_columns"][index]

        if data["rule_sources"][index] != "factorbase" or smoothed is None:
            sampled_sets.append(smoothed)
            continue

        count = min(args.per_rule, len(ordinary))
        chosen = rng.sample(ordinary, count)
        smoothed_lookup = {
            tuple(normalized(row[smoothed_columns.index(atom)]) for atom in rule): row
            for row in smoothed
        }
        selected = []
        for row in chosen:
            key = tuple(normalized(row[cp_columns.index(atom)]) for atom in rule)
            selected.append(smoothed_lookup[key])
            manifest.append({
                "rule_index": index,
                "rule": rule,
                "state": list(key),
                "factorbase_local_mult": float(row[cp_columns.index("local_mult")]),
            })
        sampled_sets.append(selected)

    data["values_smoothed_full"] = sampled_sets
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest)} sampled FactorBase states to {destination}")


if __name__ == "__main__":
    main()
