# motif_counting/rule_pruning.py
"""
Value-row pruning methods for FactorBase CP tables.

Both RuleBasedMotifStore (cache build) and RelationalMotifCounter (cache load)
call into this module so the two stay consistent. Literal rules are never
passed here — callers exempt them before pruning (see the callers'
prune-exemption checks).

CP-table row layouts (after `SELECT *` from `<child>_CP`):

  multi-atom rule (multiples == 1):
      [MULT, child_value, parent_values..., ParentSum, local_mult, CP,
       likelihood, prior]

  unary rule (multiples == 0):
      [child_value, CP, MULT, local_mult, likelihood, prior]

See motif_counting/RULE_PRUNING.md for the mathematical description of both
methods.
"""

from math import log
from typing import List, Optional, Tuple

# Method 2 defaults. tau is the minimum |ln(CP/prior)| effect size, the
# support fraction is relative to the rule's total groundings, and alpha is
# the significance level of the per-parent-configuration G^2 test.
DEFAULT_RULE_PRUNE_TAU = 0.15
DEFAULT_RULE_PRUNE_MIN_SUPPORT_FRAC = 0.0025
DEFAULT_RULE_PRUNE_ALPHA = 0.05

# chi-square critical values at alpha=0.05 for small dof; Wilson-Hilferty
# approximation covers larger dof without a scipy dependency.
_CHI2_95 = {
    1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
    6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
}
_CHI2_99 = {
    1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086,
    6: 16.812, 7: 18.475, 8: 20.090, 9: 21.666, 10: 23.209,
}
_Z_95 = 1.6449
_Z_99 = 2.3263


def chi2_critical(dof: int, alpha: float = DEFAULT_RULE_PRUNE_ALPHA) -> float:
    """Upper-tail chi-square critical value for the group G^2 test."""
    dof = max(int(dof), 1)
    if alpha <= 0.01:
        table, z = _CHI2_99, _Z_99
    else:
        table, z = _CHI2_95, _Z_95
    if dof in table:
        return table[dof]
    # Wilson-Hilferty: chi2_{alpha,k} ~ k * (1 - 2/(9k) + z * sqrt(2/(9k)))^3
    frac = 2.0 / (9.0 * dof)
    return dof * (1.0 - frac + z * (frac ** 0.5)) ** 3


def _row_stats(row, multiples: int) -> Optional[Tuple[float, float, float]]:
    """Extract (local_mult, CP, prior) from a CP row; None if unusable."""
    size = len(row)
    try:
        if multiples:
            n = float(row[size - 4])
            cp = float(row[size - 3])
            prior = float(row[size - 1])
        else:
            n = float(row[size - 3])
            cp = float(row[size - 5])
            prior = float(row[size - 1])
    except (TypeError, ValueError):
        return None
    if n <= 0 or cp <= 0 or prior <= 0:
        return None
    return n, cp, prior


def _parent_config(row, multiples: int) -> Tuple:
    """Parent value assignment of a multi-atom CP row (empty for unary)."""
    if not multiples:
        return ()
    # [MULT, child, parents..., ParentSum, local_mult, CP, likelihood, prior]
    return tuple(row[2:len(row) - 5])


def prune_value_rows_method1(value_rows: List, multiples: int) -> List:
    """
    Method 1 (legacy): per-row one-sided BIC-style log-likelihood-ratio test.

    Keep a row iff  2 * n * (ln CP - ln prior) - ln n > 0,
    where n = local_mult. Rows with CP < prior can never pass (one-sided);
    rows with zero counts or probabilities are dropped.
    """
    pruned = []
    for row in value_rows:
        stats = _row_stats(row, multiples)
        if stats is None:
            continue
        n, cp, prior = stats
        if 2.0 * n * (log(cp) - log(prior)) - log(n) > 0:
            pruned.append(row)
    return pruned


def prune_value_rows_method2(
    value_rows: List,
    multiples: int,
    tau: float = DEFAULT_RULE_PRUNE_TAU,
    min_support_frac: float = DEFAULT_RULE_PRUNE_MIN_SUPPORT_FRAC,
    alpha: float = DEFAULT_RULE_PRUNE_ALPHA,
) -> List:
    """
    Method 2: two-stage prune.

    Stage 1 — per parent configuration, a G^2 likelihood-ratio test of
    "child independent of parents given this configuration":
        G^2(cfg) = sum_over_child_values  2 * n * ln(CP / prior)
    compared against the chi-square critical value with K-1 degrees of
    freedom (K = number of child values observed under cfg). Configurations
    that fail are dropped entirely — their conditional distribution is not
    distinguishable from the prior.

    Stage 2 — within surviving configurations, keep a row iff BOTH
        |ln(CP / prior)| >= tau                      (two-sided effect size)
        n >= min_support_frac * total_groundings     (support)
    where total_groundings is the sum of local_mult over all usable rows of
    the rule.
    """
    usable = []
    total_groundings = 0.0
    for row in value_rows:
        stats = _row_stats(row, multiples)
        if stats is None:
            continue
        n, cp, prior = stats
        usable.append((row, n, cp, prior))
        total_groundings += n

    if not usable:
        return []

    # Stage 1: group rows by parent configuration and test each group.
    groups = {}
    for row, n, cp, prior in usable:
        groups.setdefault(_parent_config(row, multiples), []).append((row, n, cp, prior))

    min_support = min_support_frac * total_groundings
    pruned = []
    for group_rows in groups.values():
        g2 = sum(2.0 * n * log(cp / prior) for _, n, cp, prior in group_rows)
        dof = max(len(group_rows) - 1, 1)
        if g2 <= chi2_critical(dof, alpha):
            continue
        # Stage 2: effect size + support within the surviving configuration.
        for row, n, cp, prior in group_rows:
            if abs(log(cp / prior)) >= tau and n >= min_support:
                pruned.append(row)
    return pruned


def prune_value_rows(
    value_rows: List,
    multiples: int,
    method: int = 1,
    tau: float = DEFAULT_RULE_PRUNE_TAU,
    min_support_frac: float = DEFAULT_RULE_PRUNE_MIN_SUPPORT_FRAC,
    alpha: float = DEFAULT_RULE_PRUNE_ALPHA,
) -> List:
    """Dispatch to the selected pruning method."""
    method = int(method)
    if method == 1:
        return prune_value_rows_method1(value_rows, multiples)
    if method == 2:
        return prune_value_rows_method2(
            value_rows, multiples,
            tau=tau, min_support_frac=min_support_frac, alpha=alpha,
        )
    raise ValueError(f"Unknown rule_prune_method: {method} (expected 1 or 2)")
