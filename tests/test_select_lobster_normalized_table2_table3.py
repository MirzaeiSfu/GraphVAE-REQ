import importlib.util
from pathlib import Path


def load_selector_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "select_lobster_normalized_table2_table3.py"
    )
    spec = importlib.util.spec_from_file_location("lobster_normalized_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_domain_clamps_numerical_estimator_excursions():
    selector = load_selector_module()
    metrics = {
        "degree": 0.01,
        "clustering": 0.02,
        "orbit": -1.4386529309629026e-05,
        "spectral": 0.03,
        "diameter": 0.04,
        "mmd_rbf": 0.20,
        "precision": 1.0,
        "recall": 1.0,
        "f1_pr": 1.00001,
    }

    clamped, adjustments = selector.clamp_metric_domains(metrics)

    assert clamped["orbit"] == 0.0
    assert clamped["f1_pr"] == 1.0
    assert metrics["orbit"] < 0.0
    assert metrics["f1_pr"] > 1.0
    assert set(adjustments) == {"orbit", "f1_pr"}
    components = selector.score_components_for_mode(
        clamped, selector.SCORE_MODE, selector.DATASET
    )
    assert len(components) == 7

