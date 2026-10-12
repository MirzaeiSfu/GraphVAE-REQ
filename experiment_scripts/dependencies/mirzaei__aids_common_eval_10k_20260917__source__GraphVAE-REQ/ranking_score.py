"""Validation ranking scores for best-checkpoint selection."""

from __future__ import annotations

import math


TABLE2_VALIDATION_MMD_KEYS = ("degree", "clustering", "orbit", "spectral", "diameter")
TABLE3_VALIDATION_METRIC_KEYS = ("mmd_rbf", "f1_pr")

FALLBACK_TABLE2_DATASET = "GRID"
FALLBACK_TABLE2_ROW = "GraphVAE"

PAPER_TABLE2_BY_DATASET = {
    "TRIANGULAR_GRID": {
        "50/50 split": {
            "degree": 3e-5,
            "clustering": 0.002,
            "orbit": 8e-5,
            "spectral": 0.004,
            "diameter": 0.014,
        },
        "GraphVAE": {
            "degree": 0.082,
            "clustering": 0.442,
            "orbit": 0.421,
            "spectral": 0.020,
            "diameter": 0.152,
        },
        "GraphVAE-MM": {
            "degree": 0.001,
            "clustering": 0.093,
            "orbit": 0.001,
            "spectral": 0.013,
            "diameter": 0.133,
        },
    },
    "GRID": {
        "50/50 split": {
            "degree": 1e-5,
            "clustering": 0.0,
            "orbit": 2e-5,
            "spectral": 0.004,
            "diameter": 0.014,
        },
        "GraphVAE": {
            "degree": 0.062,
            "clustering": 0.055,
            "orbit": 0.515,
            "spectral": 0.018,
            "diameter": 0.143,
        },
        "GraphVAE-MM": {
            "degree": 5e-4,
            "clustering": 0.0,
            "orbit": 0.001,
            "spectral": 0.014,
            "diameter": 0.065,
        },
    },
    "LOBSTER": {
        "50/50 split": {
            "degree": 0.002,
            "clustering": 0.0,
            "orbit": 0.002,
            "spectral": 0.005,
            "diameter": 0.032,
        },
        "GraphVAE": {
            "degree": 0.081,
            "clustering": 0.739,
            "orbit": 0.372,
            "spectral": 0.056,
            "diameter": 0.129,
        },
        "GraphVAE-MM": {
            "degree": 2e-4,
            "clustering": 0.0,
            "orbit": 0.008,
            "spectral": 0.017,
            "diameter": 0.187,
        },
    },
    "PROTEINS": {
        "50/50 split": {
            "degree": 4e-5,
            "clustering": 0.004,
            "orbit": 5e-4,
            "spectral": 4e-4,
            "diameter": 0.003,
        },
        "GraphVAE": {
            "degree": 0.022,
            "clustering": 0.108,
            "orbit": 0.577,
            "spectral": 0.016,
            "diameter": 0.080,
        },
        "GraphVAE-MM": {
            "degree": 0.006,
            "clustering": 0.059,
            "orbit": 0.152,
            "spectral": 0.007,
            "diameter": 0.091,
        },
    },
    "ogbg-molbbbp": {
        "50/50 split": {
            "degree": 2e-4,
            "clustering": 2e-5,
            "orbit": 9e-5,
            "spectral": 5e-4,
            "diameter": 0.002,
        },
        "GraphVAE": {
            "degree": 0.028,
            "clustering": 0.442,
            "orbit": 0.047,
            "spectral": 0.015,
            "diameter": 0.055,
        },
        "GraphVAE-MM": {
            "degree": 0.001,
            "clustering": 0.005,
            "orbit": 8e-4,
            "spectral": 0.005,
            "diameter": 0.018,
        },
    },
}

TABLE3_GRAPHVAE_MM_PAPER_MMD_RBF_BY_DATASET = {
    "TRIANGULAR_GRID": 0.17,
    "LOBSTER": 0.10,
    "GRID": 0.13,
    "ogbg-molbbbp": 0.02,
    "PROTEINS": 0.03,
}

VALIDATION_SCORE_F1_PR_ERROR_DENOMINATOR = 0.05
NORMALIZED_MMD_DENOMINATOR_FLOOR = 1e-3
NORMALIZED_SCORE_COMPONENT_CAP = 10.0
TABLE3_PRIORITY_SCORE_WEIGHTS = {
    "mmd_rbf": 0.40,
    "f1_pr_error": 0.40,
    "degree": 0.04,
    "clustering": 0.04,
    "orbit": 0.04,
    "spectral": 0.04,
    "diameter": 0.04,
}
BEST_VALIDATION_MMD_SCORE_MODES = (
    "normalized_table2",
    "normalized_table2_table3",
    "table3_priority",
    "raw_mean",
    "raw_mean_table2_table3",
    "table3",
    *TABLE2_VALIDATION_MMD_KEYS,
    *TABLE3_VALIDATION_METRIC_KEYS,
)


def _dataset_key(dataset_name):
    if dataset_name is None:
        return None
    dataset_key = str(dataset_name).strip()
    if dataset_key in PAPER_TABLE2_BY_DATASET or dataset_key in TABLE3_GRAPHVAE_MM_PAPER_MMD_RBF_BY_DATASET:
        return dataset_key
    upper_key = dataset_key.upper()
    if upper_key in PAPER_TABLE2_BY_DATASET or upper_key in TABLE3_GRAPHVAE_MM_PAPER_MMD_RBF_BY_DATASET:
        return upper_key
    return dataset_key


def table2_denominators(dataset_name=None, paper_row=FALLBACK_TABLE2_ROW):
    dataset_key = _dataset_key(dataset_name)
    dataset_rows = PAPER_TABLE2_BY_DATASET.get(dataset_key, {})
    denominators = dataset_rows.get(paper_row)
    if denominators is not None:
        return denominators
    return PAPER_TABLE2_BY_DATASET[FALLBACK_TABLE2_DATASET][FALLBACK_TABLE2_ROW]


def table3_mmd_rbf_denominator(dataset_name=None):
    dataset_key = _dataset_key(dataset_name)
    return TABLE3_GRAPHVAE_MM_PAPER_MMD_RBF_BY_DATASET.get(dataset_key, 1.0)


def _valid_mmd_value(metrics, metric_name):
    value = metrics.get(metric_name)
    if value is None or not math.isfinite(value) or value < 0:
        return None
    return value


def _valid_probability_metric(metrics, metric_name):
    value = metrics.get(metric_name)
    if value is None or not math.isfinite(value):
        return None
    return min(max(float(value), 0.0), 1.0)


def validation_score_components(metrics, score_mode, dataset_name=None):
    if score_mode in TABLE2_VALIDATION_MMD_KEYS:
        value = _valid_mmd_value(metrics, score_mode)
        return None if value is None else {score_mode: value}

    if score_mode == "mmd_rbf":
        value = _valid_mmd_value(metrics, "mmd_rbf")
        return None if value is None else {"mmd_rbf": value}

    if score_mode == "f1_pr":
        value = _valid_probability_metric(metrics, "f1_pr")
        return None if value is None else {"f1_pr_error": 1.0 - value}

    raw_table2_values = {}
    for metric_name in TABLE2_VALIDATION_MMD_KEYS:
        value = _valid_mmd_value(metrics, metric_name)
        if value is None:
            return None
        raw_table2_values[metric_name] = value

    if score_mode == "raw_mean":
        return raw_table2_values

    table2_paper = table2_denominators(dataset_name)
    if score_mode == "normalized_table2":
        return {
            metric_name: min(
                metrics[metric_name]
                / max(table2_paper[metric_name], NORMALIZED_MMD_DENOMINATOR_FLOOR),
                NORMALIZED_SCORE_COMPONENT_CAP,
            )
            for metric_name in TABLE2_VALIDATION_MMD_KEYS
        }

    mmd_rbf = _valid_mmd_value(metrics, "mmd_rbf")
    f1_pr = _valid_probability_metric(metrics, "f1_pr")
    if mmd_rbf is None or f1_pr is None:
        return None
    table3_values = {
        "mmd_rbf": mmd_rbf,
        "f1_pr_error": 1.0 - f1_pr,
    }

    if score_mode == "table3":
        return table3_values

    if score_mode == "raw_mean_table2_table3":
        return {**raw_table2_values, **table3_values}

    if score_mode == "normalized_table2_table3":
        normalized_table2_values = {
            metric_name: min(
                metrics[metric_name]
                / max(table2_paper[metric_name], NORMALIZED_MMD_DENOMINATOR_FLOOR),
                NORMALIZED_SCORE_COMPONENT_CAP,
            )
            for metric_name in TABLE2_VALIDATION_MMD_KEYS
        }
        normalized_table3_values = {
            "mmd_rbf": min(
                mmd_rbf
                / max(
                    table3_mmd_rbf_denominator(dataset_name),
                    NORMALIZED_MMD_DENOMINATOR_FLOOR,
                ),
                NORMALIZED_SCORE_COMPONENT_CAP,
            ),
            # A 5 percentage-point F1-PR error counts as one unit of badness.
            "f1_pr_error": min(
                (1.0 - f1_pr) / VALIDATION_SCORE_F1_PR_ERROR_DENOMINATOR,
                NORMALIZED_SCORE_COMPONENT_CAP,
            ),
        }
        return {**normalized_table2_values, **normalized_table3_values}

    if score_mode == "table3_priority":
        normalized_table2_values = {
            metric_name: min(
                metrics[metric_name]
                / max(table2_paper[metric_name], NORMALIZED_MMD_DENOMINATOR_FLOOR),
                NORMALIZED_SCORE_COMPONENT_CAP,
            )
            for metric_name in TABLE2_VALIDATION_MMD_KEYS
        }
        return {**table3_values, **normalized_table2_values}

    raise ValueError(f"Unknown best validation MMD score mode: {score_mode}")


def compute_validation_mmd_score(metrics, score_mode, dataset_name=None):
    components = validation_score_components(metrics, score_mode, dataset_name)
    if components is None:
        return None
    if score_mode == "table3_priority":
        return sum(
            components[metric_name] * TABLE3_PRIORITY_SCORE_WEIGHTS[metric_name]
            for metric_name in TABLE3_PRIORITY_SCORE_WEIGHTS
        )
    return sum(components.values()) / len(components)


def score_components_for_mode(metrics, score_mode, dataset_name=None):
    components = validation_score_components(metrics, score_mode, dataset_name)
    return components or {}


def score_denominators_for_mode(score_mode, dataset_name=None):
    if score_mode == "normalized_table2":
        return {
            metric_name: max(denominator, NORMALIZED_MMD_DENOMINATOR_FLOOR)
            for metric_name, denominator in table2_denominators(dataset_name).items()
        }
    if score_mode in ("normalized_table2_table3", "table3_priority"):
        return {
            **{
                metric_name: max(denominator, NORMALIZED_MMD_DENOMINATOR_FLOOR)
                for metric_name, denominator in table2_denominators(dataset_name).items()
            },
            "mmd_rbf": max(
                table3_mmd_rbf_denominator(dataset_name),
                NORMALIZED_MMD_DENOMINATOR_FLOOR,
            ),
            "f1_pr_error": VALIDATION_SCORE_F1_PR_ERROR_DENOMINATOR,
        }
    return None


def score_weights_for_mode(score_mode):
    if score_mode == "table3_priority":
        return dict(TABLE3_PRIORITY_SCORE_WEIGHTS)
    return None


def score_metrics_for_mode(score_mode):
    if score_mode in (*TABLE2_VALIDATION_MMD_KEYS, *TABLE3_VALIDATION_METRIC_KEYS):
        return [score_mode]
    if score_mode == "table3":
        return list(TABLE3_VALIDATION_METRIC_KEYS)
    if score_mode in ("raw_mean_table2_table3", "normalized_table2_table3", "table3_priority"):
        return list(TABLE2_VALIDATION_MMD_KEYS) + list(TABLE3_VALIDATION_METRIC_KEYS)
    return list(TABLE2_VALIDATION_MMD_KEYS)
