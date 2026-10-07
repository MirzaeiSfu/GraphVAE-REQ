#!/usr/bin/env python3
"""Compare every numeric leaf of a stored result JSON with a re-run JSON.

Usage: compare_sanity.py STORED.json RERUN.json [--subtree a.b] [--rerun-subtree x.y] [--out report.json]
"""
import argparse
import json
import math


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield prefix, float(obj)


def sub(obj, path):
    for part in [p for p in (path or "").split(".") if p]:
        obj = obj[part]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stored")
    ap.add_argument("rerun")
    ap.add_argument("--subtree", default="")
    ap.add_argument("--rerun-subtree", default=None)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--out")
    a = ap.parse_args()
    s = dict(leaves(sub(json.load(open(a.stored)), a.subtree)))
    r = dict(leaves(sub(json.load(open(a.rerun)), a.rerun_subtree if a.rerun_subtree is not None else a.subtree)))
    common = sorted(set(s) & set(r))
    worst, fails = 0.0, []
    for k in common:
        d = abs(s[k] - r[k])
        rel = d / max(abs(s[k]), 1e-300)
        worst = max(worst, rel if abs(s[k]) > a.atol else 0.0)
        if not (d <= a.atol + a.rtol * abs(s[k]) or (math.isnan(s[k]) and math.isnan(r[k]))):
            fails.append((k, s[k], r[k]))
    report = {"stored": a.stored, "rerun": a.rerun, "subtree": a.subtree, "compared_leaves": len(common),
              "only_in_stored": sorted(set(s) - set(r))[:50], "only_in_rerun": sorted(set(r) - set(s))[:50],
              "max_relative_difference": worst, "failures": fails[:50], "n_failures": len(fails),
              "PASS": not fails and len(common) > 0}
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ("only_in_stored", "only_in_rerun")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
