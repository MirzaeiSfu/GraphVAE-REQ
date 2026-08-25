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

Current execution checkpoint: Gates 1--3 are complete, and every individual
Gate 4 bounded lifecycle case is frozen and restored. This includes exact-two
and exact-three simultaneous workers, definite prelaunch handling, ambiguous
post-ack handling without duplicate dispatch, and grouped stale-worker process
recovery. Preserve the non-overlapping first mock3 attempt and every consumed
failure exactly as recorded.

The stale qualification is complete in
`lobster_graphcl_f1pr_stale_20260825a`. The exact seed-0 process group was
recovered after its parent died, an unrelated process survived, the sole
reservation became PostgreSQL `FAIL` through native heartbeat/stale handling,
and no replacement or partial objective was created. Finalization retained both
interrupted result layers in the tombstone, the final probe is
`RECONCILED_TERMINAL`, and an all-failed portable restore passed. Exact evidence
is in `lobster_graphcl_f1pr_stale_qualification.json`; 77 focused tests pass.

Gate 4 now passes its exit check: `151/151` current non-PostgreSQL distributed,
attribute-BO, and GraphCL tests plus all `19/19` protected isolated PostgreSQL
tests passed, with zero residual test studies. Exact evidence is in
`lobster_graphcl_f1pr_gate4_exit_qualification.json`.

Next begin Gate 5 with a fresh fixed-plan real LOBSTER study containing exactly
the six predeclared anchors at 2,000 epochs and GraphVAE seeds 0 and 1. First
inspect GPU availability and freeze the exact study contract, timing limits,
thresholds, and two physical-slot schedule before study creation. Adaptive BO
is not yet authorized. CPU/mock slots do not authorize a third physical GPU.
Do not regenerate a split, retrain an encoder, treat any mock metric as BO
evidence, or access held-out/test data.

---
