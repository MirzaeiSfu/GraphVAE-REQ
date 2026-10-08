import glob, json, statistics

ROOT = "/tmp/grid_matched_compare.7L3E6T"
OUT = "/local-scratch/localhome/mirzaei/GRID_MATCHED_MOTIF_COMPARISON.md"
METRICS = [
    ("Degree MMD", "table2.metrics.degree", "lower"),
    ("Clustering MMD", "table2.metrics.clustering", "lower"),
    ("Orbit MMD", "table2.metrics.orbit", "lower"),
    ("Spectral MMD", "table2.metrics.spectral", "lower"),
    ("Diameter MMD", "table2.metrics.diameter", "lower"),
    ("Sparsity MMD", "table2.extra_metrics.sparsity", "lower"),
    ("Triangle MMD", "table2.extra_metrics.triangle", "lower"),
    ("Generated edge count", "table2.extra_metrics.generated_edge_count", "neutral"),
    ("Reference edge count", "table2.extra_metrics.reference_edge_count", "neutral"),
    ("Local MMD-RBF", "table3.local_eval_metrics.mmd_rbf", "lower"),
    ("Local precision", "table3.local_eval_metrics.precision", "higher"),
    ("Local recall", "table3.local_eval_metrics.recall", "higher"),
    ("Local F1-PR", "table3.local_eval_metrics.f1_pr", "higher"),
    ("Third-party MMD-RBF", "table3.third_party_eval_metrics.metrics.mmd_rbf.mean", "lower"),
    ("Third-party MMD-linear", "table3.third_party_eval_metrics.metrics.mmd_linear.mean", "lower"),
    ("Third-party MMD-linear trimmed mean", "table3.third_party_eval_metrics.metrics.mmd_linear.trimmed_mean", "lower"),
    ("Third-party precision", "table3.third_party_eval_metrics.metrics.precision.mean", "higher"),
    ("Third-party recall", "table3.third_party_eval_metrics.metrics.recall.mean", "higher"),
    ("Third-party F1-PR", "table3.third_party_eval_metrics.metrics.f1_pr.mean", "higher"),
]

def load(v): return [json.load(open(p)) for p in sorted(glob.glob(f"{ROOT}/{v}/*.json"))]
def get(d,p):
    for k in p.split("."): d=d[k]
    return float(d)
def values(ds,p): return [get(d,p) for d in ds]
def edge_error(ds): return [abs(get(d,"table2.extra_metrics.generated_edge_count")-get(d,"table2.extra_metrics.reference_edge_count")) for d in ds]
def fmt(x): return f"{x:.3e}" if x and abs(x)<1e-4 else f"{x:.6f}"
def summary(v): return f"{fmt(statistics.mean(v))} ± {fmt(statistics.stdev(v))}"
def improvement(b,c,d):
    bm,cm=statistics.mean(b),statistics.mean(c)
    if d=="neutral" or bm==0:return "—"
    good=bm-cm if d=="lower" else cm-bm
    return f"{good/abs(bm)*100:+.2f}%"

docs={v:load(v) for v in ("false","total_count","full_matrix")}
refs={v:values(ds,"table2.extra_metrics.reference_edge_count") for v,ds in docs.items()}
aligned=len({tuple(round(x,10) for x in a) for a in refs.values()})==1
L=[]; add=L.append
add("# Matched GRID Motif=False vs. Motif=True Results\n")
add("This report compares the new GRID motif=False baseline with the motif=True `total_count` and `full_matrix` experiments. Each setting contains seeds 0, 1, and 2. Aggregate values are mean ± sample standard deviation across seeds.\n")
add("The runs are matched on the featureless GRID database, model, 70/10/20 split, split/loader seed 123, 20,000 epochs, learning rate, training batch size, model selection, and evaluation protocol. The intended training difference is topology-based motif loss.\n")
add(f"Reference alignment: **{'passed' if aligned else 'failed'}**. The mean reference edge count is {statistics.mean(refs['false']):.6f} for motif=False, {statistics.mean(refs['total_count']):.6f} for total-count, and {statistics.mean(refs['full_matrix']):.6f} for full-matrix.\n")
add("Lower is better for MMD/distance metrics; higher is better for precision, recall, and F1. Positive improvement favors motif=True. Edge counts are descriptive.\n")
add("## Aggregate comparison\n")
add("| Metric | Better | Motif=False | True total_count | Δ mean | Improvement | True full_matrix | Δ mean | Improvement |")
add("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for name,path,direction in METRICS+[("Edge-count absolute error",None,"lower")]:
    base=edge_error(docs["false"]) if path is None else values(docs["false"],path)
    row=[name,"Descriptive" if direction=="neutral" else direction.title(),summary(base)]
    for variant in ("total_count","full_matrix"):
        cand=edge_error(docs[variant]) if path is None else values(docs[variant],path)
        row += [summary(cand),fmt(statistics.mean(cand)-statistics.mean(base)),improvement(base,cand,direction)]
    add("| "+" | ".join(row)+" |")
add("")
add("## Per-seed overview\n")
add("| Setting | Seed | Degree | Clustering | Orbit | Spectral | Diameter | Local MMD-RBF | Local F1-PR | Third-party MMD-RBF | Third-party F1-PR | Edge abs. error |")
add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
paths=["table2.metrics.degree","table2.metrics.clustering","table2.metrics.orbit","table2.metrics.spectral","table2.metrics.diameter","table3.local_eval_metrics.mmd_rbf","table3.local_eval_metrics.f1_pr","table3.third_party_eval_metrics.metrics.mmd_rbf.mean","table3.third_party_eval_metrics.metrics.f1_pr.mean"]
labels={"false":"Motif=False","total_count":"True total_count","full_matrix":"True full_matrix"}
for variant in ("false","total_count","full_matrix"):
    errs=edge_error(docs[variant])
    for seed,d in enumerate(docs[variant]): add("| "+" | ".join([labels[variant],str(seed),*[fmt(get(d,p)) for p in paths],fmt(errs[seed])])+" |")
add("")
scored={}
for name,path,direction in METRICS:
    if direction=="neutral":continue
    means={v:statistics.mean(values(ds,path)) for v,ds in docs.items()}
    scored[name]=min(means,key=means.get) if direction=="lower" else max(means,key=means.get)
counts={v:list(scored.values()).count(v) for v in docs}
add("## Summary\n")
add(f"Across the {len(scored)} directionally scored metrics, motif=False is best on **{counts['false']}**, motif=True total-count on **{counts['total_count']}**, and motif=True full-matrix on **{counts['full_matrix']}** metrics.\n")
add("Edge-count absolute error is `abs(generated_edge_count - reference_edge_count)`; it is not the FactorBase motif count-distance objective.\n")
add("## Sources\n")
add("- Motif=False: `/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/grid/seed_<N>` on cs-cl-19")
add("- Motif=True: `/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/grid/{total_count,full_matrix}/seed_<N>` on cs-cl-19")
open(OUT,"w").write("\n".join(L)+"\n")
