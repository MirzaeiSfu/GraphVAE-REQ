#!/usr/bin/env python3
"""Add the verified three-seed GraphVAE motif=False baselines to the synthetic report."""

from pathlib import Path


P = Path("/local-scratch/localhome/mirzaei/DEFOG_VS_GRAPHVAE_MOTIF_TRUE_SYNTHETIC_20260906.md")

# display string, numeric mean
GIN = {
    "GRID": {
        "F1-PR": ("0.392779 ± 0.142617", .392779367781), "Precision": ("0.341667 ± 0.089768", .341666666667),
        "Recall": ("0.775000 ± 0.368273", .775), "MMD-RBF": ("0.806865 ± 0.153998", .806864627202),
        "MMD-linear mean": ("7074.556824 ± 4922.887057", 7074.55682373), "MMD-linear median": ("1323.220754 ± 693.530730", 1323.22075399),
        "MMD-linear 10%-trimmed": ("2016.097254 ± 1311.313696", 2016.09725444),
    },
    "LOBSTER": {
        "F1-PR": ("0.752521 ± 0.071307", .752520886116), "Precision": ("0.620000 ± 0.087178", .62),
        "Recall": ("0.993333 ± 0.011547", .993333333333), "MMD-RBF": ("0.173007 ± 0.053576", .173007262275),
        "MMD-linear mean": ("107.271007 ± 50.055881", 107.271006838), "MMD-linear median": ("88.327766 ± 47.911309", 88.3277664185),
        "MMD-linear 10%-trimmed": ("102.277561 ± 46.873139", 102.277561347),
    },
    "TRIANGULAR_GRID": {
        "F1-PR": ("0.903451 ± 0.023010", .903451271673), "Precision": ("0.828333 ± 0.036171", .828333333333),
        "Recall": ("1.000000 ± 0.000000", 1.), "MMD-RBF": ("0.206531 ± 0.066011", .206531054775),
        "MMD-linear mean": ("137.946756 ± 40.902745", 137.946756363), "MMD-linear median": ("81.556480 ± 26.040791", 81.5564804077),
        "MMD-linear 10%-trimmed": ("96.052583 ± 29.855321", 96.0525832176),
    },
}

DIRECT = {
    "GRID": {
        "Degree MMD": ("0.135944 ± 0.066671", .135944231945), "Clustering MMD": ("0.120726 ± 0.027632", .120726224059),
        "Orbit MMD": ("0.761679 ± 0.165471", .761678829748), "Spectral MMD": ("0.021480 ± 0.004223", .0214800977487),
        "Diameter MMD": ("0.207147 ± 0.070912", .207146955119), "Triangle MMD": ("7.944124e-06 ± 2.232988e-06", 7.9441240155e-6),
        "Sparsity MMD": ("2.315187e-10 ± 3.281282e-10", 2.31518730113e-10), "Edge-count absolute error": ("226.483333 ± 33.954614", 226.483333333),
    },
    "LOBSTER": {
        "Degree MMD": ("0.058797 ± 0.027880", .058796650807), "Clustering MMD": ("0.418071 ± 0.134877", .418071075806),
        "Orbit MMD": ("0.184583 ± 0.066306", .18458278086), "Spectral MMD": ("0.027167 ± 0.008180", .0271674968084),
        "Diameter MMD": ("0.096861 ± 0.009322", .0968614631676), "Triangle MMD": ("1.733973e-05 ± 2.269611e-06", 1.733972941e-5),
        "Sparsity MMD": ("6.331957e-09 ± 8.587780e-09", 6.3319566627e-9), "Edge-count absolute error": ("21.950000 ± 5.975157", 21.95),
    },
    "TRIANGULAR_GRID": {
        "Degree MMD": ("0.008932 ± 0.008190", .00893170936523), "Clustering MMD": ("0.137583 ± 0.074300", .137583041194),
        "Orbit MMD": ("0.077736 ± 0.057113", .0777357610739), "Spectral MMD": ("0.018747 ± 0.003187", .0187470023246),
        "Diameter MMD": ("0.081433 ± 0.027672", .0814328906972), "Triangle MMD": ("6.223435e-05 ± 6.006294e-05", 6.22343463152e-5),
        "Sparsity MMD": ("1.493624e-09 ± 7.156814e-10", 1.49362411328e-9), "Edge-count absolute error": ("72.716667 ± 8.589868", 72.7166666667),
    },
}

EDGES = {
    "GRID": "GraphVAE false 571.483333 ± 33.954614 / 345.000000 ± 0.000000; ",
    "LOBSTER": "GraphVAE false 67.800000 ± 5.975157 / 45.850000 ± 0.000000; ",
    "TRIANGULAR_GRID": "GraphVAE false 373.016667 ± 8.589868 / 300.300000 ± 0.000000; ",
}

TV_FALSE = {
    "GRID": "0.012910 ± 0.003124",
    "LOBSTER": "0.032247 ± 0.002789",
    "TRIANGULAR_GRID": "0.018245 ± 0.003490",
}


def advantage(defog, false, better):
    signed = false - defog if better == "↑" else defog - false
    return f"{100 * signed / abs(defog):+.2f}%"


text = P.read_text(encoding="utf-8")
if "False advantage vs DeFoG" in text:
    raise SystemExit("report already updated")

text = text.replace(
    "# DeFoG versus GraphVAE-REQ motif=True on synthetic datasets",
    "# DeFoG versus GraphVAE-REQ motif=True and motif=False on synthetic datasets",
).replace(
    "and compares them with the earlier controlled three-seed GraphVAE-REQ LinkCorrelation motif=True total-count and full-matrix runs.",
    "and compares them with the earlier controlled three-seed GraphVAE-REQ motif=False baseline and LinkCorrelation motif=True total-count and full-matrix runs.",
)
text = text.replace(
    "| Dataset | DeFoG epochs / batch | DeFoG seeds used | GraphVAE epochs | GraphVAE seeds | GraphVAE motif setting |",
    "| Dataset | DeFoG epochs / batch | DeFoG seeds used | GraphVAE epochs | GraphVAE false seeds | GraphVAE true seeds | GraphVAE motif setting |",
).replace(
    "| --- | ---: | ---: | ---: | ---: | --- |",
    "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    1,
).replace("| GRID | 10,000 / 1 | 0, 1 (seed 2 running) | 20,000 | 0, 1, 2 |", "| GRID | 10,000 / 1 | 0, 1 (seed 2 running) | 20,000 | 0, 1, 2 | 0, 1, 2 |")
text = text.replace("| LOBSTER | 1,000 / 4 | 0, 1, 2 | 20,000 | 0, 1, 2 |", "| LOBSTER | 1,000 / 4 | 0, 1, 2 | 20,000 | 0, 1, 2 | 0, 1, 2 |")
text = text.replace("| TRIANGULAR_GRID | 10,000 / 1 | 0, 1, 2 | 20,000 | 0, 1, 2 |", "| TRIANGULAR_GRID | 10,000 / 1 | 0, 1, 2 | 20,000 | 0, 1, 2 | 0, 1, 2 |")

lines = text.splitlines()
out = []
dataset = None
family = None
for line in lines:
    if line.startswith("### GRID —"):
        dataset = "GRID"
    elif line.startswith("### LOBSTER —"):
        dataset = "LOBSTER"
    elif line.startswith("### TRIANGULAR_GRID —"):
        dataset = "TRIANGULAR_GRID"
    if "third-party Random-GIN" in line and line.startswith("###"):
        family = "gin"
    elif "direct structural metrics" in line and line.startswith("###"):
        family = "direct"
    if line == "| Metric | Better | DeFoG | GraphVAE total | GraphVAE advantage | GraphVAE full | GraphVAE advantage |":
        out.append("| Metric | Better | DeFoG | GraphVAE false | False advantage vs DeFoG | GraphVAE total | Total advantage vs DeFoG | GraphVAE full | Full advantage vs DeFoG |")
        continue
    if line == "| --- | :---: | ---: | ---: | ---: | ---: | ---: |" and dataset and family:
        out.append("| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        continue
    if dataset and family and line.startswith("|") and not line.startswith("| Metric") and not line.startswith("| ---"):
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        source = GIN if family == "gin" else DIRECT
        if len(cells) == 7 and cells[0] in source.get(dataset, {}):
            display, fval = source[dataset][cells[0]]
            dval = float(cells[2].split("±")[0].strip())
            cells = cells[:3] + [display, advantage(dval, fval, cells[1])] + cells[3:]
            line = "| " + " | ".join(cells) + " |"
    if dataset in EDGES and line.startswith("Generated/reference mean edge counts:"):
        line = line.replace("Generated/reference mean edge counts: ", "Generated/reference mean edge counts: " + EDGES[dataset])
    out.append(line)

text = "\n".join(out) + "\n"

# Add motif=False to the correlation aggregate with an explicit protocol warning.
old_h = "| Dataset | DeFoG | GraphVAE total | GraphVAE advantage | GraphVAE full | GraphVAE advantage | Reference identity |"
new_h = "| Dataset | DeFoG | GraphVAE false (historical) | False comparison | GraphVAE total | Total advantage | GraphVAE full | Full advantage | Reference identity |"
text = text.replace(old_h, new_h).replace(
    "| --- | ---: | ---: | ---: | ---: | ---: | --- |\n| GRID |",
    "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |\n| GRID |",
)
for ds, value in TV_FALSE.items():
    token = f"| {ds} |"
    for line in text.splitlines():
        if line.startswith(token) and "Exact match" in line or line.startswith(token) and "Different held-out identities" in line:
            cells = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(cells) == 7:
                replacement = "| " + " | ".join(cells[:2] + [value, "N/C (different collection protocol)"] + cells[2:]) + " |"
                text = text.replace(line, replacement, 1)
            break

note = "\n**Motif=False TV protocol note:** the added motif=False values are the verified three-seed historical training-collection aggregates (70 generated/reference graphs per seed), whereas this DeFoG/true table otherwise uses 20 held-out graphs. They are included for completeness but must not be used for a numeric advantage claim against the 20-graph columns.\n"
text = text.replace("### Per-seed motif TV", note + "\n### Per-seed motif TV")

text = text.replace(
    "- GraphVAE comparison copies:",
    "- Motif=False source runs: GRID and TRIANGULAR_GRID on system 19 under `/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/<dataset>/seed_<seed>`; LOBSTER on system 16 under `/local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/lobster/seed_<seed>`. These are the three-seed baseline artifacts originally gathered for the comparison.\n- GraphVAE comparison copies:",
)
P.write_text(text, encoding="utf-8")
print(P)
