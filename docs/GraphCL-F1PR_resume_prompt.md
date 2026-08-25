# Restart prompt for the LOBSTER GraphCL-F1PR campaign

Copy the text below into a new Codex chat if the current chat stops. Paste it
without adding credentials or storage URLs.

---

Continue implementing and executing
`docs/GraphCL-F1PR_roadmap.md` faithfully in:

```text
/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ
```

Start by reading the entire roadmap and its **Current checkpoint** section,
then inspect `git status --short --branch`, recent commits, all tracked and
untracked changes, current processes, remote worker state, and existing study
lifecycles. Do not redo a completed gate or replace truthful failure evidence.

Primary scientific question: does a fresh LOBSTER BO search using a frozen,
training-only GraphCL-GIN ensemble improve GraphVAE over uniform reconstruction
weights under a separately frozen matched-seed validation confirmation?

Non-negotiable contract:

- dataset is LOBSTER; never run QM9 BO;
- exact cache path, size, SHA-256, 70/10/20 split, 14/11 feature dimensions,
  and `optimal_v2` schema are frozen in the roadmap;
- GraphCL encoders train only on the 70 training graphs;
- BO and confirmation use only the 10 validation graphs;
- `test_access` remains false and held-out/test evaluation is not authorized;
- both GraphVAE feature decoders and GraphCL `decoded_node_edge` mode are
  required;
- train five encoders at seeds `101,202,303,404,505` from the pinned clean
  contrastive upstream revision recorded in the roadmap;
- objective is finite `summary.f1_pr.mean` from five frozen GraphCL-GIN
  checkpoints, with all identity/schema/split assertions;
- preserve exact reservations, grouped multi-seed candidate semantics, failure
  consumption, heartbeat/grace, bounded concurrency, deterministic dispatch,
  atomic artifacts, portable restoration, and no replacement;
- never blindly duplicate ambiguous work;
- do not claim that the old Random-GIN study optimized GraphCL;
- do not run adaptive BO unless the fixed anchor/fidelity gate passes its
  predeclared stability and rank-transfer thresholds;
- do not rerank after confirmation.

Protected PostgreSQL material is outside the repository. Never print, cat,
echo, log, commit, or expose credential contents or an unredacted storage URL.
Use protected mode-0600 environment/pgpass/CA files, `PGPASSFILE`, and
`sslmode=verify-full`. Keep credentials outside source, cache, checkpoint,
collection, and artifact roots.

Use the committed dedicated `cs-cl-09` GraphCL-F1PR mappings and never reuse
Gate 4/5/6, AIDS, or earlier LOBSTER study roots. Candidate concurrency is two
homogeneous GTX TITAN X slots and remains subject to the roadmap's fresh
hardware/timing gate. Preserve old studies and failures.

Work gate by gate. Run focused tests after implementation changes, then the
full GraphCL, non-PostgreSQL distributed BO, and isolated PostgreSQL suites at
the prescribed checkpoints. Audit/freeze/restore every real study.

The user requires a commit and push after each completed gate. Use the
commit-summary skill every time. Before each commit inspect:

```text
git status --short --branch
git log -5 --pretty=format:%s
git diff --stat
git diff --cached --stat
```

Also inspect actual tracked/untracked changes, announce the commit label and
detailed bullets, stage only intended files, inspect staged diff/check, then
commit and push. Never commit credentials, URLs, caches, model/checkpoint
binaries, dependency bundles, ignored run trees, or generated graph artifacts.

Continue autonomously within this roadmap. Give concise progress updates at
least once per minute during long training, remote, or test operations. If a
qualification gate fails, preserve the evidence, report
`qualification_failed`, commit it, and do not force the BO run.

Current execution checkpoint: Gates 1--3 are complete. The exact split export,
five immutable LOBSTER GraphCL encoders, and validation-only same-latent
GraphVAE evaluator are qualified on `cs-cl-09`; read the roadmap's Current
checkpoint and the three qualification JSON files for exact hashes, metrics,
and the preserved initial dependency failure. The next step is Gate 4
distributed controller/worker integration, including grouped GraphVAE seeds
and bounded mock concurrency/failure/recovery tests. Do not regenerate a split,
retrain an encoder, treat the integration metric as BO evidence, or access
held-out/test data.

---
