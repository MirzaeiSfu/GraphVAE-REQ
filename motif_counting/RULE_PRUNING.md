# Rule Pruning Methods

This document describes how motif value rows (CP-table rows) are pruned when
`--rule_prune true` is set, and gives the mathematical derivation of the
two available methods:

- **Method 1** (`--rule_prune_method 1`, default): the legacy per-row
  one-sided BIC-style likelihood-ratio test.
- **Method 2** (`--rule_prune_method 2`): a two-stage prune — a
  per-parent-configuration G² dependence test followed by two-sided
  effect-size and support filters.

Both methods are implemented in
[`rule_pruning.py`](rule_pruning.py) and applied **at cache load time** by
`RelationalMotifCounter` from the complete `values_full` set stored in the
motif pickle. The pickle always contains every rule and every value
combination, so switching `--rule_prune`, `--rule_prune_method`, or the
method-2 thresholds never requires regenerating the cache.

**Literal rules are never pruned** by either method. A rule is exempt when it
is (a) a unary rule, (b) a synthetic literal rule injected by the cache
builder, or (c) a FactorBase rule shaped like a literal (a lone relation atom,
or an edge-feature atom paired with its relation atom). These rules encode the
marginal (prior) distributions of node/edge features and the edge count; for
parentless rules `CP = prior` by construction, so any significance test on
them is degenerate and would remove all of their rows.

## Notation

FactorBase learns a Bayesian network over first-order functors. Every child
node together with its parent set defines a **rule**

$$R:\; C \mid P_1, \dots, P_m,$$

and the conditional probability table `⟨C⟩_CP` stores one row per observed
value combination $(c, p_1, \dots, p_m)$. Each row provides:

| Column | Symbol | Meaning |
|---|---|---|
| `local_mult` | $n$ | number of groundings satisfying $C{=}c \wedge P_1{=}p_1 \wedge \dots \wedge P_m{=}p_m$ |
| `ParentSum` | $N_{\text{cfg}}$ | number of groundings satisfying the parent configuration $\text{cfg} = (p_1, \dots, p_m)$ |
| `CP` | $\hat{p} = n / N_{\text{cfg}}$ | maximum-likelihood estimate of $P(C{=}c \mid \text{cfg})$ |
| `prior` | $p_0$ | marginal estimate of $P(C{=}c)$ |

Each kept row becomes one **motif column** in the `(B, num_motifs)` count
tensor, i.e. one differentiable bmm chain per training batch. Pruning
therefore trades off (i) motif-counting compute, (ii) dilution of the motif
loss (which averages over motif columns), and (iii) loss of structural signal.

## Method 1 (legacy): per-row one-sided BIC-style LR test

Keep row $(c, \text{cfg})$ iff

$$2\,n \left(\ln \hat{p} - \ln p_0\right) - \ln n \;>\; 0.$$

The first term is the (signed) contribution of this cell to the
log-likelihood-ratio statistic comparing the conditional model
$P(C \mid \text{cfg})$ against the independence model $P(C) = p_0$; the
$\ln n$ term is a BIC-flavored complexity penalty.

Properties worth knowing (all observable on the
`triangular_grid_undir_feat_snap_ce92ed_cfg410000000` CP tables):

1. **One-sided.** If $\hat{p} < p_0$ the statistic is negative and the row is
   always dropped, no matter how strong the evidence. Under-represented
   combinations ("this pattern is rare/forbidden given these parents") carry
   exactly the kind of constraint a generator needs, e.g.
   `num_3cycles(n0)=7` given `edge_direction=2, n3c(n1)=3, nhex(n1)=2` has
   $\hat{p}=0.034$ vs $p_0=0.75$ ($|\ln \hat{p}/p_0| = 3.1$) and is dropped.
2. **Sample size beats effect size.** The statistic grows linearly in $n$
   while the penalty grows as $\ln n$, so the keep condition is equivalent to
   $\ln(\hat{p}/p_0) > \ln(n)/(2n)$ — a threshold that vanishes for large
   $n$. With $n = 16850$ a lift of $1.0015$ (pure `float(7,6)` rounding noise
   in the stored prior) passes confidently, while a genuine 40% depletion at
   $n = 100$ is dropped.
3. **Cell-wise testing.** Dependence between $C$ and its parents is a
   property of the whole conditional distribution under a configuration, not
   of a single cell; testing cells independently with a 1-dof-like penalty
   over-counts evidence.

## Method 2: two-stage prune

### Stage 1 — dependence test per parent configuration

For a parent configuration $\text{cfg}$ with observed child values
$c_1, \dots, c_K$, test

$$H_0:\; P(C \mid \text{cfg}) = P(C) \quad\text{(child independent of parents under cfg)}$$

with the likelihood-ratio statistic

$$G^2(\text{cfg}) \;=\; 2 \sum_{k=1}^{K} n_k \,\ln \frac{\hat{p}_k}{p_{0,k}}
\;\;\overset{H_0}{\sim}\;\; \chi^2_{K-1}.$$

Drop **all** rows of $\text{cfg}$ iff
$G^2(\text{cfg}) \le \chi^2_{K-1,\,1-\alpha}$ (default $\alpha = 0.05$;
critical values are tabulated for small dof and approximated by
Wilson–Hilferty, $\chi^2_{k,1-\alpha} \approx k\,(1 - \tfrac{2}{9k} +
z_{1-\alpha}\sqrt{\tfrac{2}{9k}})^3$, beyond).

This is the statistically correct unit of testing: it aggregates the evidence
across all child values of one configuration with the right degrees of
freedom. On the triangular-grid database it is what removes the
`edge_direction` configurations `num_3cycles(n1) ∈ {3, 7}`, where
$\hat{p} \equiv 1/3 \approx p_0$ for all three directions — rows that carry
81.5% of that rule's groundings and contain no information beyond the
marginal (which the exempt literal rules already provide).

### Stage 2 — effect size and support within surviving configurations

Significance alone is not discriminating here: grounding counts run from
hundreds to tens of thousands, so almost any nonzero deviation is
"significant" (a two-sided χ² test keeps 476/479 multi-rule rows on the
triangular-grid database). What makes a motif worth its compute is a **large**
deviation backed by **enough mass**. Keep row $(c, \text{cfg})$ iff both:

$$\underbrace{\left|\ln \frac{\hat{p}}{p_0}\right| \;\ge\; \tau}_{\text{effect size (two-sided)}}
\qquad\text{and}\qquad
\underbrace{n \;\ge\; f_{\min} \cdot N_R}_{\text{support}},$$

where $N_R = \sum_{\text{rows of } R} n$ is the rule's total groundings.
Defaults: $\tau = 0.15$ (`--rule_prune_tau`), $f_{\min} = 0.0025$
(`--rule_prune_min_support_frac`).

Rationale for the defaults:

- $\tau = 0.15$ means CP must differ from the prior by at least ~16% in
  ratio. It sits far above the noise floor introduced by the `float(7,6)`
  rounding of `CP`/`prior` in the database (worst case
  $|\ln \text{lift}| \approx 0.005$), so rounding artifacts can never
  survive, and it is two-sided, so strong *under*-representation
  ($\hat{p} \ll p_0$) is kept as an "avoid this pattern" motif.
- $f_{\min} = 0.0025$ (0.25% of the rule's groundings) removes cells whose
  observed per-graph counts are so small that, after Laplace smoothing in the
  motif loss, they contribute mostly noise — while keeping rare deterministic
  cells that still occur consistently. Lower it (e.g. `0.001`) if you want
  more of the rare $\hat{p} = 1$ structural constraints and can afford the
  compute; raise it to cut motif-counting cost further.

### Interpretation of what survives

Because the filter is two-sided, surviving rows split into three useful
families:

1. **Deterministic constraints** ($\hat{p} \approx 1$, $p_0$ small): hard
   structural implications of the dataset (dominant in `edge_hexagons`,
   `edge_triangle_count`).
2. **Enriched patterns** ($\hat{p} > p_0$): soft positive dependencies.
3. **Depleted patterns** ($\hat{p} < p_0$): soft negative constraints that
   method 1 discards by construction.

### Measured effect (triangular grid, `..._ce92ed_cfg410000000`)

With defaults ($\alpha=0.05$, $\tau=0.15$, $f_{\min}=0.0025$):

| Rule (child) | rows | method 1 keeps | method 2 keeps | notes on method 2 |
|---|---|---|---|---|
| `distance_to_boundary(n0)` | 145 | 130 | 76 | 46 deterministic, 9 depleted kept |
| `edge_direction(n0,n1)` | 15 | 5 | 7 | stage 1 drops 2 independent configs (6 rows) |
| `edge_hexagons(n0,n1)` | 98 | 98 | 48 | all kept rows are $\hat{p}=1$ |
| `edge_triangle_count(n0,n1)` | 26 | 26 | 9 | all kept rows are $\hat{p}=1$ |
| `num_3cycles(n0)` | 66 | 48 | 35 | 9 depleted kept |
| `num_hexagons(n0)` | 129 | 106 | 57 | |
| literal/unary rules | 17 | 17 | 17 | always exempt |
| **total** | **496** | **430** | **249** | ~2× fewer bmm chains |

(Method-1 numbers are what the formula yields now that the old feature-rule
bypass is removed; before the fix, `rule_prune=true` was a no-op on this
database because every rule mentions a node/edge feature functor and was
therefore exempted wholesale.)

## Practical notes

- The pickle stores `values_full` (everything) plus `values_pruned`
  (method 1, literal-exempt) for inspection; the counter recomputes the
  selection from `values_full` at load, so old caches pick up the fixed
  exemption logic without regeneration.
- All thresholds are runtime arguments (`--rule_prune_method`,
  `--rule_prune_tau`, `--rule_prune_min_support_frac`, `--rule_prune_alpha`)
  and are recorded in `reproducibility.json` under `motif_cache`.
- $G^2$ uses the stored MLE `CP`; rows with zero `local_mult`, `CP`, or
  `prior` are unusable for the log-ratio and are dropped by both methods
  (FactorBase CP tables normally do not contain them — zero-count
  combinations only appear in the `_CP_pairs` / `_CP_smoothed` side tables).
