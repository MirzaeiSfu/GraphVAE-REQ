#!/usr/bin/env python3
"""Aggregate datasets/*/rule_mmd_results.json into RULE_MMD_REPORT.md (+ summary json)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = [("motif_true_full", "Motif True (full)"), ("motif_false", "Motif False"), ("defog", "DeFoG")]
ORDER = ["GRID", "TRIANGULAR_GRID", "LOBSTER", "PTC", "PROTEINS", "QM9", "AIDS"]
NOTES = {
    "GRID": "Edge-existence rules only (2-edge chain + single edge); no attribute atoms. 20 generated graphs per run; DeFoG seeds 0/4/5.",
    "TRIANGULAR_GRID": "Edge-existence rules only; no attribute atoms. 20 generated graphs per run; DeFoG seeds 0/1/3.",
    "LOBSTER": "Edge-existence rules only; no attribute atoms. 20 generated graphs per run; pruned set from the saved training selection manifest.",
    "PTC": ("240 graphs per run, regenerated from each saved checkpoint (the reported 70-graph sets are missing for "
            "Motif True seeds 1/2 and all Motif False seeds). Caveat: Motif False was trained on an older split in which "
            "43-52 of the 70 test graphs were training graphs, so its test-reference scores are optimistic."),
    "PROTEINS": ("209 graphs per run (same collections as the structural tables). Caveat: the three methods were trained on "
                 "different splits; references here come from the Motif True split, which partly overlaps the Motif False "
                 "and DeFoG training sets."),
    "QM9": "512 graphs per run. Unpruned = full 97,223-state space; pruned = the top-10-per-rule states Motif True trained on (53).",
    "AIDS": ("400 graphs per run; DeFoG seeds 3/4/5. Caveat: DeFoG trained on its own split (~279 of its training graphs are "
             "in the GraphVAE test split), so its test-reference score is partly measured on its training graphs."),
}


def per_method(scores, ref, key):
    grouped = {}
    for name, refs in scores.items():
        match = re.fullmatch(r"(.+)_seed_(\d+)", name)
        if not match or ref not in refs:
            continue
        grouped.setdefault(match.group(1), []).append((int(match.group(2)), refs[ref][key]))
    return {m: sorted(v) for m, v in grouped.items()}


def fmt(values):
    arr = np.array([v for _, v in values])
    return f"{arr.mean():.4g} ± {arr.std(ddof=0):.2g}" if len(arr) > 1 else f"{arr.mean():.4g}"


def main():
    summary, lines = {}, []
    lines += ["# Rule-MMD: rule-instance fidelity of generated graphs", "",
              "Unbiased squared MMD (Gaussian kernel, median-heuristic bandwidth from the train reference, shared by all "
              "methods of a dataset) between per-graph rule-state frequency vectors of generated and reference graphs. "
              "Lower is better; values near 0 (can be slightly negative) mean indistinguishable from the reference. Mean ± std over the three training seeds. **Bold** marks the best method for each column. "
              "'Noise floor' is the train-vs-test MMD of real graphs; a generator at the floor reproduces the rules "
              "as well as a fresh sample of real data.", "",
              "- **Unpruned:** every state of every FactorBase rule (`_CP_smoothed` full state space).",
              "- **Pruned:** only the states that motif=True training optimized (training-time data-driven pruning).",
              "- **Train / Test:** reference collection (train subset capped at 1000 graphs; test split of the same dataset cache).",
              "- All graphs (real and generated) pass the same GraphVAE hard cleanup: undirected, no self-loops/isolates, largest connected component.", ""]
    for ds in ORDER:
        path = ROOT / "datasets" / ds / "rule_mmd_results.json"
        if not path.exists():
            lines += [f"## {ds}", "", "_Not available yet._", ""]
            continue
        res = json.loads(path.read_text())
        summary[ds] = {}
        lines += [f"## {ds}", "",
                  f"Database `{res['database_name']}`: {res['full_state_count']} unpruned states, "
                  f"{res['pruned_state_count']} pruned states.", "", NOTES.get(ds, ""), ""]
        for block, title in (("state", "Rule-state Rule-MMD"), ("density", "Rule-density MMD (secondary)")):
            lines += [f"**{title}**", "", "| Method | Unpruned / Train | Unpruned / Test | Pruned / Train | Pruned / Test |",
                      "|---|---|---|---|---|"]
            cells, best = {}, {}
            for view in ("unpruned", "pruned"):
                scores = res["views"][view][block]["scores"]
                for ref in ("train", "test"):
                    grouped = per_method(scores, ref, "mmd2_unbiased")
                    means = {m: np.mean([v for _, v in vals]) for m, vals in grouped.items()}
                    if means:
                        best[(view, ref)] = min(means, key=means.get)
                    for m, vals in grouped.items():
                        cells[(m, view, ref)] = fmt(vals)
                        summary[ds].setdefault(block, {}).setdefault(view, {}).setdefault(ref, {})[m] = {
                            "mean": float(np.mean([v for _, v in vals])), "std": float(np.std([v for _, v in vals])),
                            "seeds": vals}
            for m, label in METHODS:
                row = [label]
                for view in ("unpruned", "pruned"):
                    for ref in ("train", "test"):
                        c = cells.get((m, view, ref), "n/a")
                        row.append(f"**{c}**" if best.get((view, ref)) == m else c)
                lines.append("| " + " | ".join(row) + " |")
            floor = []
            for view in ("unpruned", "pruned"):
                s = res["views"][view][block]["scores"]
                v = s.get("test", {}).get("train", {}).get("mmd2_unbiased")
                floor += [f"{v:.4g}" if v is not None else "n/a", "—"]
            lines += ["| Noise floor (test vs train) | " + " | ".join(floor) + " |", ""]
        lines += [""]
    lines += ["## OGB (ogbg-molbbbp): not evaluated", "",
              "The saved GraphVAE OGB outputs are adjacency-only, while 2,948 of the 2,949 OGB rule states involve atom or "
              "bond features, so a like-for-like comparison is impossible. No full-matrix Motif True OGB run exists either "
              "(the reported setting 03 uses the older total-count loss).", ""]
    (ROOT / "RULE_MMD_REPORT.md").write_text("\n".join(lines) + "\n")
    (ROOT / "rule_mmd_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(ROOT / "RULE_MMD_REPORT.md")


if __name__ == "__main__":
    main()
