import glob, json, statistics

ROOT = "/tmp/all_valid_compare.NCRh06"
OUT = "/local-scratch/localhome/mirzaei/ALL_VALID_MOTIF_TRUE_FALSE_RESULTS.md"
DATASETS = ["mutag", "ptc", "grid", "lobster", "triangular_grid"]
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

def load(ds,v): return [json.load(open(p)) for p in sorted(glob.glob(f"{ROOT}/{ds}/{v}/*.json"))]
def get(d,p):
    for k in p.split("."): d=d[k]
    return float(d)
def vals(ds,p): return [get(d,p) for d in ds]
def edge_errors(ds): return [abs(get(d,"table2.extra_metrics.generated_edge_count")-get(d,"table2.extra_metrics.reference_edge_count")) for d in ds]
def fmt(x): return f"{x:.3e}" if x and abs(x)<1e-4 else f"{x:.6f}"
def stat(v): return f"{fmt(statistics.mean(v))} ± {fmt(statistics.stdev(v))} (n={len(v)})"
def improve(b,c,d):
    bm,cm=statistics.mean(b),statistics.mean(c)
    if d=="neutral" or bm==0:return "—"
    favorable=bm-cm if d=="lower" else cm-bm
    return f"{favorable/abs(bm)*100:+.2f}%"

groups={ds:{v:load(ds,v) for v in ("false","total_count","full_matrix")} for ds in DATASETS}
L=[]; add=L.append
add("# All Valid Motif=False vs. Motif=True Results\n")
add("This consolidated report contains every dataset that currently has final motif=False and motif=True result files suitable for comparison: MUTAG, PTC, GRID, LOBSTER, and TRIANGULAR_GRID. Each dataset has its own section.\n")
add("For MUTAG and PTC, motif=False uses three gather seeds while each motif=True mode has two seeds. For the synthetic datasets, all settings use three seeds. Aggregate values are mean ± sample standard deviation over the available training seeds; `n` is shown in every cell.\n")
add("For MMD/distance metrics lower is better. For precision, recall, and F1 higher is better. A positive improvement percentage favors motif=True; a negative percentage favors motif=False. Edge counts are descriptive.\n")
add("## Dataset coverage\n")
add("| Dataset | Motif=False | True total_count | True full_matrix | Status |")
add("|---|---:|---:|---:|---|")
for ds in DATASETS:
    g=groups[ds]
    note="Ready; unequal seed counts" if ds in ("mutag","ptc") else "Fully matched three-seed comparison"
    add(f"| {ds.upper()} | {len(g['false'])} | {len(g['total_count'])} | {len(g['full_matrix'])} | {note} |")
add("")

winner_summary=[]
for ds in DATASETS:
    g=groups[ds]
    add(f"## {ds.upper()}\n")
    ref={v:vals(d,"table2.extra_metrics.reference_edge_count") for v,d in g.items()}
    add("Reference edge-count means: " + ", ".join(f"{v}={statistics.mean(a):.6f} (n={len(a)})" for v,a in ref.items()) + ".")
    if ds in ("mutag","ptc"):
        add("\nSeed-count caution: motif=False is aggregated over three seeds, whereas each motif=True mode is aggregated over two seeds. Standard deviations are therefore based on different sample sizes.\n")
        ref_means = [statistics.mean(ref[v]) for v in ("false", "total_count", "full_matrix")]
        if max(ref_means) - min(ref_means) > 1e-9:
            add("Reference-set caution: the average reference edge counts differ between the gather baseline and smoothed runs. This comparison is descriptive and is not as tightly matched as the synthetic comparisons.\n")
    else:
        aligned=len({tuple(round(x,10) for x in a) for a in ref.values()})==1
        add(f"\nMatched reference alignment check: **{'passed' if aligned else 'failed'}**.\n")
    add("### Aggregate metrics\n")
    add("| Metric | Better | Motif=False | True total_count | Δ mean | Improvement | True full_matrix | Δ mean | Improvement |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    scored={}
    for name,path,direction in METRICS+[("Edge-count absolute error",None,"lower")]:
        base=edge_errors(g["false"]) if path is None else vals(g["false"],path)
        row=[name,"Descriptive" if direction=="neutral" else direction.title(),stat(base)]
        candidate_means={"false":statistics.mean(base)}
        for v in ("total_count","full_matrix"):
            c=edge_errors(g[v]) if path is None else vals(g[v],path)
            candidate_means[v]=statistics.mean(c)
            row += [stat(c),fmt(statistics.mean(c)-statistics.mean(base)),improve(base,c,direction)]
        add("| "+" | ".join(row)+" |")
        if direction!="neutral" and path is not None:
            scored[name]=min(candidate_means,key=candidate_means.get) if direction=="lower" else max(candidate_means,key=candidate_means.get)
    counts={v:list(scored.values()).count(v) for v in ("false","total_count","full_matrix")}
    winner_summary.append((ds,counts,len(scored)))
    add("")
    add("### Per-seed overview\n")
    add("| Setting | Seed | Degree | Clustering | Orbit | Spectral | Diameter | Local MMD-RBF | Local F1-PR | Third-party MMD-RBF | Third-party F1-PR | Edge abs. error |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    paths=["table2.metrics.degree","table2.metrics.clustering","table2.metrics.orbit","table2.metrics.spectral","table2.metrics.diameter","table3.local_eval_metrics.mmd_rbf","table3.local_eval_metrics.f1_pr","table3.third_party_eval_metrics.metrics.mmd_rbf.mean","table3.third_party_eval_metrics.metrics.f1_pr.mean"]
    labels={"false":"Motif=False","total_count":"True total_count","full_matrix":"True full_matrix"}
    for v in ("false","total_count","full_matrix"):
        errors=edge_errors(g[v])
        for seed,d in enumerate(g[v]): add("| "+" | ".join([labels[v],str(seed),*[fmt(get(d,p)) for p in paths],fmt(errors[seed])])+" |")
    add("")

add("## Cross-dataset summary\n")
add("The count below reports which setting has the best aggregate mean for each of the 17 directionally scored metrics. It is a compact summary, not a statistical significance test.\n")
add("| Dataset | Motif=False best | True total_count best | True full_matrix best | Scored metrics |")
add("|---|---:|---:|---:|---:|")
for ds,c,n in winner_summary:add(f"| {ds.upper()} | {c['false']} | {c['total_count']} | {c['full_matrix']} | {n} |")
add("")
add("## Excluded because results are not final\n")
add("- PROTEINS: all six motif=True trainings reached epoch 20,000, but final PyG export failed on a 210-versus-209 collection-size mismatch. Only provisional log metrics exist.")
add("- ENZYMES: current motif=True runs are incomplete.")
add("- QM9: current motif=True runs are incomplete and gather has no matching QM9 motif=False baseline.")
add("")
add("## Metric note\n")
add("Edge-count absolute error is `abs(generated_edge_count - reference_edge_count)`. It is not the FactorBase motif count-distance training objective. Third-party Random-GIN uses topology-derived node features (degree, clustering, and square clustering), not semantic dataset node/edge attributes.\n")
add("## Result sources\n")
add("- Gather motif=False baselines: `/local-scratch2/new/gather/datasets/{mutag,ptc}/setting_01/seed_<N>` on cs-cl-18.")
add("- MUTAG/PTC motif=True: distributed under `runs/multihop_smoothed/` on cs-cl-13, cs-cl-17, cs-cl-18, and cs-cl-19.")
add("- Synthetic motif=False: `runs/motif_false_3seed/<dataset>/seed_<N>` on cs-cl-16/cs-cl-19.")
add("- Synthetic motif=True: `runs/multihop_smoothed_3seed/<dataset>/{total_count,full_matrix}/seed_<N>` on cs-cl-16/cs-cl-19.")
open(OUT,"w").write("\n".join(L)+"\n")
