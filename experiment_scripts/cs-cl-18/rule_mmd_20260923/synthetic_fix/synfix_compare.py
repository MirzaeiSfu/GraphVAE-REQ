"""Synthetic fix: scores vs the new reference per method, plus real-data null ranges at each sample size."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.argv = ["x"]
import rule_mmd  # noqa: E402

rng = np.random.default_rng(7)
out = {}
for ds in ["GRID", "TRIANGULAR_GRID", "LOBSTER"]:
    d = Path("datasets_fix") / ds
    r = json.loads((d / "rule_mmd_results.json").read_text())
    layout = json.loads((d / "selection.json").read_text())["layout"]
    counts = rule_mmd.load_counts(d / "counts")
    out[ds] = {}
    for block_index, block in enumerate(["state", "density"]):
        scores = r["views"]["unpruned"][block]["scores"]
        grouped = {}
        for name, refs in scores.items():
            match = re.fullmatch(r"(.+)_seed_(\d+)", name)
            if not match or "reference_new_seed_0" not in refs or match.group(1) == "reference_new":
                continue
            grouped.setdefault(match.group(1), []).append(
                (int(match.group(2)), 100 * refs["reference_new_seed_0"]["mmd2_unbiased"]))
        ref_x = rule_mmd.features(*counts["reference_new_seed_0"][:2], layout)[block_index]
        sigma = r["views"]["unpruned"][block]["bandwidth"]
        if ds == "LOBSTER":  # independent real graphs: the original 100 (train + test) vs the fresh 500
            import scipy.sparse as sp
            parts = [rule_mmd.features(*counts[n][:2], layout)[block_index] for n in ("train", "test")]
            pool = sp.vstack(parts).tocsr() if sp.issparse(parts[0]) else np.vstack(parts)
            sizes, replace = (20, 80), False
        else:  # grids: the reference IS the population; draw real samples from it with replacement
            pool, sizes, replace = ref_x, (20, 100, 500), True
        nulls = {}
        for k in sizes:
            values = []
            for _ in range(300):
                idx = rng.choice(pool.shape[0], k, replace=replace)
                values.append(100 * rule_mmd.mmd2(pool[idx], ref_x, sigma)["mmd2_unbiased"])
            nulls[k] = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        out[ds][block] = {"methods": grouped, "null95": nulls}
        print(ds, block)
        for method, vals in sorted(grouped.items()):
            v = [x for _, x in sorted(vals)]
            print(f"   {method:16s} mean {np.mean(v):7.2f} std {np.std(v):5.2f}  seeds {['%.2f' % x for x in v]}")
        print("   null95", {k: [round(a, 2), round(b, 2)] for k, (a, b) in nulls.items()})
Path("synthetic_fix/compare_vs_new_reference.json").write_text(json.dumps(out, indent=1))
