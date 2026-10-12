"""Build reports/EXPERIMENT_CODE_PROVENANCE.md and .csv: which code produced each archived run.

usage: build_provenance.py <archive_root> <graphvae_repo> <runs_archive.json> <out_dir>
"""
import collections, csv, datetime, hashlib, json, os, re, subprocess, sys

ARCHIVE, REPO, RUNS, OUT = sys.argv[1:5]
GH = "https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/"
DEFOG_GH = "https://github.com/MirzaeiSfu/defog/commit/"

# Code that ran but was never committed, located on disk by exact argparse fingerprint (see report text).
UNCOMMITTED = {
    "9c2290e55f": ("cs-cl-18 and cs-cl-19:/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/GraphVAE-REQ/ "
                   "(identical main.py also in cs-cl-18 aids_common_eval_10k_20260917/source/GraphVAE-REQ/)",
                   "adds --alpha_degree_distribution_loss and --alpha_edge_density_loss"),
    "493cf42979": ("commit 3fb44e6 plus the git_diff.patch saved in each run folder; full tree also in "
                   "cs-cl-18 and cs-cl-19:/local-scratch2/new/deploy_alpha005_20260720/GraphVAE-REQ/",
                   "adds --motif_prune_max_total_values"),
}


# Code that was not committed at launch but has since been committed from the launch folder,
# byte-identical (branch recovered/paper-code-20260904).
RECOVERED = {
    "643ccccbb9": ("f75f761fd", "code A: launch folder cs-cl-17 ali/GraphVAE-REQ-kia-motif-20260904-cl17 "
                                "(adds --motif_prune_score_threshold)"),
    "e1a5cc9dfa": ("af56479bb", "code B: launch folder Solar /project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831 "
                                "(adds --resume_from_latest_checkpoint)"),
}
# Runs whose launch folder was found and is byte-identical to a commit.
VERIFIED = [(re.compile(r"^(GRID|LOBSTER|TRIANGULAR_GRID)/experiments/graphvae/"), "9f01785b4",
             "launched from fb/GraphVAE-REQ (cs-cl-16/17/18/19): commit 946146f plus 6 changed files, "
             "byte-identical to 9f01785 (committed 2026-08-26 05:07, after the first runs started)")]


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout


# argparse fingerprint of main.py for every commit on every branch
pat = re.compile(r"add_argument\(\s*['\"](?:-\w\s*['\"],\s*['\"])?--?([\w-]+)['\"](.*?)\)", re.S)
blob_fp, commit_info = {}, {}
for line in git("log", "--all", "--format=%H\t%ct\t%s").splitlines():
    h, ct, subj = line.split("\t", 2)
    blob = git("rev-parse", f"{h}:main.py").strip()
    if not blob or blob.startswith(h + ":"):
        continue
    if blob not in blob_fp:
        names = set()
        for m in pat.finditer(git("cat-file", "-p", blob)):
            dm = re.search(r"dest\s*=\s*['\"](\w+)['\"]", m.group(2))
            names.add(dm.group(1) if dm else m.group(1).replace("-", "_"))
        blob_fp[blob] = hashlib.sha1(",".join(sorted(names)).encode()).hexdigest()[:10]
    commit_info[h] = (int(ct), subj, blob_fp[blob])
by_fp = collections.defaultdict(list)
for h, (ct, subj, fp) in commit_info.items():
    by_fp[fp].append((ct, h, subj))
for v in by_fp.values():
    v.sort()
branches_containing = {}
main_commits = set(git("rev-list", "origin/main").split())


def on_branches(h):
    if h not in branches_containing:
        out = git("branch", "-r", "--contains", h).split()
        branches_containing[h] = ", ".join(b.replace("origin/", "") for b in out if b != "->" and "HEAD" not in b)[:80]
    return branches_containing[h]


def day(ts):
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


rows = []
for r in json.load(open(RUNS)):
    parts = r["path"].split("/")
    ds = parts[0]
    if r["kind"] == "graphvae":
        start = datetime.datetime.strptime(r["start"], "%Y-%m-%d %H:%M").timestamp() if r["start"] else None
        if not start:
            rd = os.path.join(ARCHIVE, r["path"])
            ts = [os.path.getmtime(os.path.join(rd, f)) for f in os.listdir(rd) if os.path.isfile(os.path.join(rd, f))]
            start = min(ts) if ts else None
            r["start"] = day(start) if start else ""
        method = "GraphVAE (motif=False)" if str(r["motif"]) in ("False", "") or not r["motif"] else \
            f"GraphVAE+RG motif=True {r['mode'] or '(mode option not yet in code)'} (lambda={r['alpha']})"
        row = dict(repo="GraphVAE-REQ", dataset=ds, model="GraphVAE-REQ", method=method, label=r["label"], seed=r["seed"],
                   started=r["start"], archive_path=r["path"])
        ver = next(((h, why) for rx, h, why in VERIFIED if rx.match(r["path"])), None)
        if ver:
            row.update(code_commit=ver[0], confidence="verified: launch folder byte-identical to this commit",
                       note=ver[1], branches=on_branches(ver[0]))
        elif r["argkeys"] in RECOVERED:
            h, why = RECOVERED[r["argkeys"]]
            row.update(code_commit=h, confidence="recovered: launch folder committed afterwards, byte-identical",
                       note=why, branches="recovered/paper-code-20260904")
        elif r["commit"]:
            dirty = r["dirty"] or 0
            row.update(code_commit=r["commit"][:9], confidence="exact (recorded at launch)",
                       note=(f"working tree had {dirty} uncommitted change(s); see git_diff.patch in the run folder"
                             if dirty else "clean working tree"), branches=on_branches(r["commit"]))
            if r["argkeys"] in UNCOMMITTED:
                row["note"] += "; " + UNCOMMITTED[r["argkeys"]][1] + " (not in any commit)"
        elif r["argkeys"] in UNCOMMITTED:
            loc, what = UNCOMMITTED[r["argkeys"]]
            row.update(code_commit="NOT IN GIT", confidence="exact argparse match to an on-disk copy",
                       note=f"{what}; code location: {loc}", branches="")
        else:
            cands = by_fp.get(r["argkeys"], [])
            before = [c for c in cands if start is not None and c[0] <= start]
            on_main = [c for c in before if c[1] in main_commits]
            pick = (on_main or before or [None])[-1]
            if pick is None and cands:
                pick = cands[0]
            if pick:
                first, last = cands[0], cands[-1]
                row.update(code_commit=pick[1][:9],
                           confidence=("inferred: newest commit before the run with an identical main.py argument set"
                                       if pick in before else "inferred (low): run predates every commit with this main.py, so the code was committed after the run; first such commit shown"),
                           note=(f"run started from a copy without .git; main.py arguments identical in "
                                 f"{len(cands)} commits from {first[1][:9]} ({day(first[0])}) to {last[1][:9]} "
                                 f"({day(last[0])}); other modules not verified"),
                           branches=on_branches(pick[1]))
            else:
                row.update(code_commit="UNKNOWN", confidence="no matching commit", note="", branches="")
        rows.append(row)
    else:  # DeFoG Hydra run: look for the pinned DeFoG commit recorded nearby
        run_dir = os.path.join(ARCHIVE, os.path.dirname(r["path"]))
        commit, src = "", ""
        d = run_dir
        for _ in range(4):
            jr = os.path.join(d, "job_record.json")
            if os.path.exists(jr):
                try:
                    commit = json.load(open(jr)).get("defog_commit", ""); src = "job_record.json"
                except Exception:
                    pass
                break
            d = os.path.dirname(d)
        if not commit:
            try:
                txt = " ".join(open(os.path.join(run_dir, f), errors="ignore").read()
                               for f in ("config.yaml", "overrides.yaml", "hydra.yaml"))
                m = re.search(r"\b(c631697|474f940|a35327f)[0-9a-f]*", txt)
                if m:
                    commit, src = m.group(0), "Hydra config"
            except Exception:
                pass
        rows.append(dict(repo="defog", dataset=ds, model="DeFoG", method="DeFoG", label="/".join(parts[1:-1])[:90], seed="",
                         started=r["start"], archive_path=r["path"].rsplit("/.hydra", 1)[0],
                         code_commit=(commit[:9] if commit else "c631697 (campaign pin, not recorded per run)"),
                         confidence=("exact (" + src + ")") if commit else "inferred from campaign",
                         note="MirzaeiSfu/defog fork; QM9 campaigns also used uncommitted changes to src/datasets/qm9_dataset.py "
                              "(and, in some, src/main.py and gin.py), now in third_party/defog" if ds == "QM9" else "MirzaeiSfu/defog fork",
                         branches="feat/frozen-graphvae-benchmark"))

# LGD runs in the archive (old campaign) — one row per run folder
for ds in sorted(os.listdir(ARCHIVE)):
    p = os.path.join(ARCHIVE, ds, "experiments", "lgd")
    if os.path.isdir(p):
        for name in sorted(os.listdir(p)):
            rows.append(dict(repo="", dataset=ds, model="LGD", method="LGD (LGD_3SEED_CAMPAIGN_20260920)", label=name, seed="",
                             started="2026-09-20", archive_path=f"{ds}/experiments/lgd/{name}",
                             code_commit="zhouc20/LatentGraphDiffusion f597e1d + local changes",
                             confidence="exact (campaign repo state)",
                             note="OLD campaign with the epoch-0 encoder-selection bug; superseded by LGD_FIX_20260924",
                             branches=""))

rows.sort(key=lambda x: (x["dataset"], x["model"], x["method"], x["label"], str(x["seed"]), x["archive_path"]))
cols = ["repo", "dataset", "model", "method", "label", "seed", "started", "code_commit", "confidence", "branches", "note", "archive_path"]
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "experiment_code_provenance.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

# ---- markdown ----
c = collections.Counter
g = [x for x in rows if x["model"] == "GraphVAE-REQ"]
n_exact = sum(1 for x in g if x["confidence"].startswith("exact (recorded"))
n_inf = sum(1 for x in g if x["confidence"].startswith("inferred"))
n_git = sum(1 for x in g if x["code_commit"] == "NOT IN GIT")
n_ver = sum(1 for x in g if x["confidence"].startswith("verified"))
n_rec = sum(1 for x in g if x["confidence"].startswith("recovered"))


def link(h, repo):
    base = {"GraphVAE-REQ": GH, "defog": DEFOG_GH}.get(repo)
    return f"[`{h}`]({base}{h})" if base and re.fullmatch(r"[0-9a-f]{7,9}", h) else f"`{h}`"


md = []
md.append("# Experiment code provenance\n")
md.append(f"Generated {datetime.date.today()} from `EXPERIMENT_ARCHIVE_20260921` "
          "(copy on the cs-cl-18 external drive, `/media/mirzaei/backup/`). "
          "One row per archived run is in [`experiment_code_provenance.csv`](experiment_code_provenance.csv).\n")
md.append("## Summary\n")
md.append(f"- **GraphVAE-REQ runs:** {len(g)}\n"
          f"  - {n_exact} recorded their commit at launch (exact).\n"
          f"  - {n_ver} were launched from a folder whose code is byte-identical to a commit (verified).\n"
          f"  - {n_rec} ran code that was committed only afterwards, from their launch folders, on branch "
          "`recovered/paper-code-20260904` (recovered).\n"
          f"  - {n_inf} were launched from copies of the repo without `.git`, so no commit was recorded. "
          "Their commit is inferred (see Method).\n"
          f"  - **{n_git} ran code that is still not in git.** The code exists on lab disks; see "
          "[Code that is not in git](#code-that-is-not-in-git).\n"
          f"- **DeFoG runs:** {sum(1 for x in rows if x['model']=='DeFoG')}, all on the "
          f"[`MirzaeiSfu/defog`]({DEFOG_GH}c631697b9cd5a2474d22ba12de33943c6b49e53e) fork at `c631697`. "
          "That code, plus 3 uncommitted QM9/GRID changes, is in [`third_party/defog/`](../third_party/defog/).\n"
          f"- **LGD runs in the archive:** {sum(1 for x in rows if x['model']=='LGD')}, all from the old "
          "`LGD_3SEED_CAMPAIGN_20260920` (encoder-selection bug). The fixed `LGD_FIX_20260924` runs behind the "
          "paper's LGD numbers are not in this archive. Their code is in [`third_party/lgd/`](../third_party/lgd/).\n")
md.append("## Method\n")
md.append("1. **Exact:** `reproducibility.json` in the run folder has a `git_commit`. If the working tree was dirty, "
          "the run folder also has `git_diff.patch`.\n"
          "2. **Inferred:** without a recorded commit, the run's argument names (from `reproducibility.json`) are "
          "compared with the `argparse` options of `main.py` in every commit on every branch. The newest matching "
          "commit made before the run started is reported, along with the full range of matching commits. "
          "This pins `main.py` exactly but does not verify other modules (`motif_counting/`, `data.py`), so treat "
          "it as the most likely commit, not a proof.\n"
          "3. **Verified:** the folder the runs were launched from was found and its code files are byte-identical "
          "to the commit shown.\n"
          "4. **Recovered:** the code was not committed at launch; the launch folder was later committed "
          "byte-identical on branch `recovered/paper-code-20260904` (`f75f761` code A, `af56479` code B).\n"
          "5. **Not in git:** the run's argument set matches no commit, but it does match a `main.py` found on disk. "
          "That copy is listed as the code location.\n")

md.append("## Per dataset\n")
for ds in sorted({x["dataset"] for x in rows}):
    sub = [x for x in rows if x["dataset"] == ds]
    md.append(f"### {ds}\n")
    md.append("| Model / method | Runs | Code commit | Confidence |\n|---|---:|---|---|")
    agg = collections.OrderedDict()
    for x in sub:
        key = (x["model"], x["method"], x["code_commit"], x["confidence"], x["repo"])
        agg[key] = agg.get(key, 0) + 1
    for (model, method, commit, conf, repo), n in agg.items():
        md.append(f"| {method} | {n} | {link(commit, repo)} | {conf} |")
    md.append("")

md.append("## Code that is not in git\n")
md.append("These GraphVAE-REQ runs used `main.py` options that exist in no commit on any branch. "
          "Exact copies were found on disk:\n")
md.append("| Argument fingerprint | Runs | What the code adds | Where it is |\n|---|---:|---|---|")
fp_of = {r["path"]: r["argkeys"] for r in json.load(open(RUNS)) if r["kind"] == "graphvae"}
for fp, (loc, what) in UNCOMMITTED.items():
    n = sum(1 for x in g if fp_of.get(x["archive_path"]) == fp)
    md.append(f"| `{fp}` | {n} | {what} | {loc} |")
md.append("\nAffected runs, by label:\n")
for fp in UNCOMMITTED:
    labs = c(x["dataset"] + " " + x["label"] for x in g if fp_of.get(x["archive_path"]) == fp)
    md.append(f"- `{fp}`: " + "; ".join(f"{k} (x{v})" if v > 1 else k for k, v in sorted(labs.items())))
md.append("\nTo put this code under version control, commit each copy as its own branch "
          "(for example `recovered/code-<name>`) and then regenerate this report.\n")

md.append("## Baselines\n")
md.append("- **DeFoG**: fork `MirzaeiSfu/defog`, branch `feat/frozen-graphvae-benchmark`, commit `c631697` "
          "(2026-09-01, *Add frozen GraphVAE benchmark adapter*). The frozen-benchmark runs record "
          "`defog_commit` in `job_record.json`. The QM9 campaigns (`qm9_defog_campaign_20260914`, "
          "`qm9_defog_lab24_20260914`, `QM9_FULL_TEST_20260925`) and `GRID_DEFOG_SEED6_20260924` ran with 3 "
          "uncommitted files, which are included in `third_party/defog/`: `src/main.py`, "
          "`src/datasets/qm9_dataset.py`, `src/analysis/ggmeval/evaluation/models/gin/gin.py`.\n"
          "- **LGD**: upstream `zhouc20/LatentGraphDiffusion` commit `f597e1d` (2024-12-17) plus local changes "
          "(GraphVAE-REQ dataset export and loader, TU generation and sampling, configs). `third_party/lgd/` holds "
          "the fixed `LGD_FIX_20260924` state: encoder checkpoint chosen by validation `loss_recon`, never epoch 0. "
          "The old `LGD_3SEED_CAMPAIGN_20260920` differs in `sample_tu.py` and `select_best_encoder_ckpt.py`; "
          "those versions are in `third_party/lgd/legacy_LGD_3SEED_CAMPAIGN_20260920/`.\n")
md.append("## Fixed LGD runs (not in the archive)\n")
md.append("The LGD numbers in the paper come from `LGD_FIX_20260924`, which ran after the archive was made. Code: "
          "`third_party/lgd/` (upstream `f597e1d` plus the fixed local changes). Run folders, now also copied to "
          "`POST_ARCHIVE_RESULTS_20261005/` on the external drive:\n\n"
          "| Where | Datasets / seeds |\n|---|---|\n"
          "| Solar `~/LGD_FIX_20260924/results/` | GRID 0-2, PROTEINS 0-2, QM9 0-2, TRIANGULAR_GRID b16 seed 1 and b32 seeds 0/2 |\n"
          "| cs-cl-18, cs-cl-19, cs-cl-09, cs-cl-16 `LGD_FIX_20260924/` | PTC, LOBSTER, TRIANGULAR_GRID seeds 0-5 |\n"
          "| cs-cl-18 `lgd_common_eval_20260925*/` | common-reference evaluation of all of the above |\n")
md.append("## Evaluation code\n")
md.append("The paper's common-reference evaluation numbers were produced by scripts in campaign folders such as "
          "`aids_common_eval_10k_20260917/scripts/`, `qm9_common_eval_20260917/`, `lgd_common_eval_20260925/scripts/` "
          "and `rule_mmd_20260923/`, which are not in git. Copies are on the external drive under "
          "`POST_ARCHIVE_RESULTS_20261005/`.\n")
open(os.path.join(OUT, "EXPERIMENT_CODE_PROVENANCE.md"), "w").write("\n".join(md) + "\n")
print("rows", len(rows), c(x["model"] for x in rows), "exact", n_exact, "inferred", n_inf, "not-in-git", n_git)
