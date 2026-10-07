#!/usr/bin/env python3
"""Build RULE_MMD_COMPARISON_5DATASETS.md tables from rule_mmd_summary.json / results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import os
DATASETS = [("GRID", "GRID"), ("TRIANGULAR_GRID", "TRI_GRID"), ("LOBSTER", "LOBSTER"), ("PTC", "PTC"),
            ("PROTEINS", "PROTEINS"), ("QM9", "QM9"), ("AIDS", "AIDS")]
METHODS = [("motif_true_full", "Motif True"), ("motif_false", "Motif False"), ("defog", "DeFoG")]
summary = json.loads((ROOT / "rule_mmd_summary.json").read_text())


def floor(ds, block, view):
    res = json.loads((ROOT / "datasets" / ds / "rule_mmd_results.json").read_text())
    return res["views"][view][block]["scores"]["test"]["train"]["mmd2_unbiased"]


def table(block, view, ref):
    head = "| Dataset | " + " | ".join(label for _, label in METHODS) + " | Real test vs real train |"
    lines = [head, "|---|" + "---|" * (len(METHODS) + 1)]
    for ds, name in DATASETS:
        cells = summary[ds][block][view][ref]
        best = min(METHODS, key=lambda m: cells[m[0]]["mean"])[0]
        row = [name]
        for key, _ in METHODS:
            c = cells[key]
            text = f"{100 * c['mean']:.2f} ± {100 * c['std']:.2f}"
            row.append(f"**{text}**" if key == best else text)
        row.append(f"{100 * floor(ds, block, view):.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def seeds_table(block, view, ref):
    lines = ["| Dataset | Method | Seed values (×100) |", "|---|---|---|"]
    for ds, name in DATASETS:
        for key, label in METHODS:
            vals = summary[ds][block][view][ref][key]["seeds"]
            lines.append(f"| {name} | {label} | " + ", ".join(f"s{s}: {100 * v:.2f}" for s, v in vals) + " |")
    return "\n".join(lines)


def verdict(block, view, ref):
    out = {}
    for ds, name in DATASETS:
        c = summary[ds][block][view][ref]
        mt, df = c["motif_true_full"], c["defog"]
        mt_seeds = [v for _, v in mt["seeds"]]
        df_seeds = [v for _, v in df["seeds"]]
        out[name] = {"mt": mt["mean"], "defog": df["mean"], "mt_better": mt["mean"] < df["mean"],
                     "all_mt_seeds_below_all_defog": max(mt_seeds) < min(df_seeds)}
    return out


def wins(block, ref, other):
    """Motif True vs `other`: '++' all 3 MT seeds better than all other seeds, '+' better mean, '-' worse mean,
    '--' all MT seeds worse."""
    row = []
    for ds, name in DATASETS:
        c = summary[ds][block]["unpruned"][ref]
        mt = [v for _, v in c["motif_true_full"]["seeds"]]
        ot = [v for _, v in c[other]["seeds"]]
        if max(mt) < min(ot):
            mark = "++"
        elif min(mt) > max(ot):
            mark = "--"
        else:
            mark = "+" if c["motif_true_full"]["mean"] < c[other]["mean"] else "-"
        row.append(mark)
    return row


def win_table():
    head = "| Comparison | Metric | Reference | " + " | ".join(n for _, n in DATASETS) + " | MT better (mean) |"
    lines = [head, "|---|---|---|" + "---|" * (len(DATASETS) + 1)]
    for other, olabel in (("defog", "vs DeFoG"), ("motif_false", "vs Motif False")):
        for block, blabel in (("state", "Rule-state"), ("density", "Rule density")):
            for ref in ("train", "test"):
                marks = wins(block, ref, other)
                n = sum(m.startswith("+") for m in marks)
                lines.append(f"| Motif True {olabel} | {blabel} | {ref} | " + " | ".join(marks) + f" | {n}/{len(marks)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    key = sys.argv[1]
    if key == "wins":
        print(win_table())
    elif key == "verdicts":
        print(json.dumps({f"{b}/{v}/{r}": verdict(b, v, r) for b in ("state", "density")
                          for v in ("unpruned",) for r in ("train", "test")}, indent=1))
    else:
        block, view, ref, kind = key.split(":")
        print(table(block, view, ref) if kind == "mean" else seeds_table(block, view, ref))
