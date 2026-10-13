"""Lightweight tests for the reusable command-line surface."""

from ggm_eval.cli import build_parser


def test_cli_exposes_primary_and_legacy_engines():
    parser = build_parser()

    train = parser.parse_args(
        [
            "train",
            "--graphs",
            "train.pt",
            "--encoder",
            "graphcl",
            "--seeds",
            "0",
            "1",
            "--output-dir",
            "encoders",
            "--upstream-repo",
            "upstream",
        ]
    )
    legacy = parser.parse_args(
        [
            "evaluate-legacy",
            "--generated",
            "generated.pt",
            "--reference",
            "reference.pt",
            "--legacy-repo",
            "graphvae",
            "--output-dir",
            "report",
        ]
    )
    trained = parser.parse_args(
        [
            "evaluate-trained",
            "--dataset",
            "protein",
            "--generated",
            "generated.pt",
            "--reference",
            "reference.pt",
            "--output-dir",
            "trained-report",
        ]
    )

    assert train.encoder == "graphcl"
    assert train.seeds == [0, 1]
    assert train.limit_lipschitz is True
    assert trained.command == "evaluate-trained"
    assert trained.upstream_repo is None
    assert trained.device == "auto"
    assert trained.seeds is None
    assert legacy.command == "evaluate-legacy"
