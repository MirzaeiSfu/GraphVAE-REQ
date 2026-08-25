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

Current execution checkpoint: Gates 1--3 and the Gate 4 grouped GraphCL backend
implementation are complete. The exact split export, five immutable LOBSTER
GraphCL encoders, validation-only same-latent GraphVAE evaluator, and frozen
two-worker PostgreSQL lifecycle mock are qualified on `cs-cl-09`; read the
roadmap's Current checkpoint and qualification JSON files for exact hashes,
metrics, and preserved failed attempts. The next step is the fresh
three-reservation CPU/mock concurrency qualification using
`CLUSTER_GRAPHVAE_GRAPHCL_F1PR_LOBSTER_MOCK3_SLOTS.txt`, followed by Gate 4's
remaining ambiguous-launch and stale-worker/recovery cases. The first
three-reservation attempt `lobster_graphcl_f1pr_mock3_20260825a` is frozen and
restored but did not show interval overlap because its mock bodies were too
short; preserve it and never rerun its consumed reservations. Fresh attempt
`lobster_graphcl_f1pr_mock3_20260825b` passes with PostgreSQL RUNNING=3 and a
3.776814222-second common interval, and is frozen/restored with exact evidence
in `lobster_graphcl_f1pr_mock3_qualification.json`; that qualification is
already pushed. The ambiguous case is now also complete in frozen/restored
study
`lobster_graphcl_f1pr_ambiguous_20260825a`: one injected post-ack SSH error was
probed first as `ACTIVE_AMBIGUOUS/retry_safe=false` with its exact reserved DB
row RUNNING, then as `RECONCILED_TERMINAL` after the same worker completed. No
duplicate was dispatched. The qualification file
`lobster_graphcl_f1pr_ambiguous_qualification.json` and its checkpoint updates
are already pushed. Before the live stale-worker case, GraphCL grouped recovery
support was added locally: immutable mock-only child lifetime, required
contracted `--training-seed` path selection, and dual grouped/replicate
interrupted-result tombstone retention. Seventy-six focused tests pass. Commit
and deploy these changes first, then initialize a fresh exact-one-reservation
study with short test-only heartbeat/grace and a bounded mock child. No GraphCL
recovery worker has been killed yet. CPU/mock slots are for immutable mock
studies only and do not authorize or claim a third GPU. Do not regenerate a
split, retrain an encoder, treat any mock metric as BO evidence, or access
held-out/test data.

---
