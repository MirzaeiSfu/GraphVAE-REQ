#!/usr/bin/env python3
"""Build an aggressively pruned QM9 CP_smoothed cache from verified CP counts."""

import argparse
import json
import math
import pickle
from pathlib import Path


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output_dir")
    ap.add_argument("--top-per-multi-rule", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.source, "rb") as fh:
        data = pickle.load(fh)

    report = {"source": args.source, "top_per_multi_rule": args.top_per_multi_rule,
              "rules": [], "usage": {"motif_cp_table_source": "cp_smoothed",
                                       "rule_prune": False}}
    derived_smoothed = []
    for idx, rule in enumerate(data["rules"]):
        cp_rows = list(data["values_full"][idx])
        cp_cols = list(data["value_columns"][idx])
        sm_rows = data["values_smoothed_full"][idx]
        sm_cols = data["value_smoothed_columns"][idx]
        source = data["rule_sources"][idx]
        if source != "factorbase" or sm_rows is None:
            chosen = list(data["values_pruned"][idx])
            derived_smoothed.append(sm_rows)
            report["rules"].append({"index": idx, "rule": rule, "source": source,
                                    "retained": len(chosen), "states": []})
            continue

        scored = []
        if len(rule) == 1:
            # Match GraphVAE's pruning contract: unary state inventories are
            # never score-pruned because they define categorical marginals.
            scored = [(None, row) for row in cp_rows]
        else:
            for row in cp_rows:
                lm = float(row[cp_cols.index("local_mult")])
                cp = float(row[cp_cols.index("CP")])
                prior = float(row[cp_cols.index("prior")])
                if lm <= 0 or cp <= 0 or prior <= 0:
                    continue
                score = 2.0 * lm * (math.log(cp) - math.log(prior)) - math.log(lm)
                if score > 0:
                    scored.append((score, row))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            scored = scored[:args.top_per_multi_rule]

        lookup = {tuple(scalar(r[sm_cols.index(atom)]) for atom in rule): r for r in sm_rows}
        selected_smoothed = []
        states = []
        for score, cp_row in scored:
            key = tuple(scalar(cp_row[cp_cols.index(atom)]) for atom in rule)
            sm_row = list(lookup[key])
            for metadata in ("local_mult", "CP", "prior", "ParentSum", "likelihood"):
                if metadata in sm_cols and metadata in cp_cols:
                    sm_row[sm_cols.index(metadata)] = scalar(cp_row[cp_cols.index(metadata)])
            selected_smoothed.append(sm_row)
            states.append({"values": list(key), "score": score,
                           "local_mult": float(cp_row[cp_cols.index("local_mult")]),
                           "CP": float(cp_row[cp_cols.index("CP")]),
                           "prior": float(cp_row[cp_cols.index("prior")])})
        derived_smoothed.append(selected_smoothed)
        report["rules"].append({"index": idx, "rule": rule, "source": source,
                                "full_smoothed_states": len(sm_rows),
                                "observed_cp_states": len(cp_rows),
                                "retained": len(selected_smoothed), "states": states})

    data["values_smoothed_full"] = derived_smoothed
    cache_path = out / "qm9_dir_feat_snap_02f6a6_multi.pkl"
    with cache_path.open("wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "qm9_pruned_rules_and_states.json").write_text(json.dumps(report, indent=2) + "\n")
    md = ["# QM9 aggressively pruned rules", "",
          f"Top cap per multi-atom rule: **{args.top_per_multi_rule}**", "",
          "Use this already-pruned cache with `motif_cp_table_source=cp_smoothed` and "
          "`rule_prune=false`; false here prevents a second pruning pass.", ""]
    for item in report["rules"]:
        md += [f"## Rule {item['index'] + 1}", "", "`" + " AND ".join(item["rule"]) + "`", "",
               f"Retained states: **{item['retained']}**", ""]
    (out / "QM9_PRUNED_RULES.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"cache": str(cache_path),
                      "retained_factorbase_states": sum(x["retained"] for x in report["rules"]
                                                          if x["source"] == "factorbase")}, indent=2))


if __name__ == "__main__":
    main()
