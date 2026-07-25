"""Command-line interface for the reusable PyG-first evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import (
    convert_dgl_file_to_pyg,
    convert_pyg_file_to_dgl,
)
from .contract import FEATURE_MODES, collection_digest, validate_collection
from .io import load_pyg_collection
from .runner import (
    evaluate_contrastive_checkpoints,
    evaluate_legacy_random_gin,
    train_contrastive_encoders,
)
from .trained import evaluate_with_trained_gnns
from .upstreams import validate_contrastive_upstream


def _print(payload):
    print(json.dumps(payload, indent=2, sort_keys=True))


def _validate(args):
    graphs = load_pyg_collection(
        args.graphs,
        trusted=args.trusted_input,
        normalize=True,
    )
    summary = validate_collection(graphs, name=str(args.graphs))
    _print(
        {
            "graphs": str(Path(args.graphs).expanduser().resolve()),
            "summary": summary.to_dict(),
            "collection_sha256": collection_digest(graphs),
        }
    )


def _convert_dgl(args):
    metadata = {
        key: value
        for key, value in {
            "dataset": args.dataset,
            "feature_schema": args.feature_schema,
        }.items()
        if value is not None
    }
    _print(
        convert_dgl_file_to_pyg(
            args.input,
            args.output,
            metadata=metadata,
        )
    )


def _convert_pyg(args):
    _print(
        convert_pyg_file_to_dgl(
            args.input,
            args.output,
            trusted=args.trusted_input,
        )
    )


def _inspect_upstream(args):
    _print(
        validate_contrastive_upstream(
            args.upstream_repo,
            allow_unpinned=args.allow_unpinned_upstream,
        )
    )


def _train(args):
    _print(
        train_contrastive_encoders(
            graphs=args.graphs,
            upstream_repository=args.upstream_repo,
            output_dir=args.output_dir,
            encoder=args.encoder,
            feature_mode=args.feature_mode,
            seeds=args.seeds,
            python_executable=args.python,
            device=args.device,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            init=args.init,
            limit_lipschitz=args.limit_lipschitz,
            lipschitz_factor=args.lipschitz_factor,
            trusted_input=args.trusted_input,
            allow_unpinned_upstream=args.allow_unpinned_upstream,
        )
    )


def _evaluate(args):
    _print(
        evaluate_contrastive_checkpoints(
            generated=args.generated,
            reference=args.reference,
            checkpoints=args.checkpoint,
            upstream_repository=args.upstream_repo,
            output_dir=args.output_dir,
            python_executable=args.python,
            device=args.device,
            nearest_k=args.nearest_k,
            max_graphs=args.max_graphs,
            trusted_input=args.trusted_input,
            allow_unpinned_upstream=args.allow_unpinned_upstream,
        )
    )


def _evaluate_trained(args):
    _print(
        evaluate_with_trained_gnns(
            dataset=args.dataset,
            generated=args.generated,
            reference=args.reference,
            output_dir=args.output_dir,
            upstream_repository=args.upstream_repo,
            seeds=args.seeds,
            python_executable=args.python,
            device=args.device,
            nearest_k=args.nearest_k,
            max_graphs=args.max_graphs,
            trusted_input=args.trusted_input,
            allow_unpinned_upstream=args.allow_unpinned_upstream,
        )
    )


def _evaluate_legacy(args):
    _print(
        evaluate_legacy_random_gin(
            generated=args.generated,
            reference=args.reference,
            legacy_repository=args.legacy_repo,
            output_dir=args.output_dir,
            python_executable=args.python,
            modes=args.modes,
            repeats=args.repeats,
            evaluator_seed=args.evaluator_seed,
            nearest_k=args.nearest_k,
            max_graphs=args.max_graphs,
            device=args.device,
            trusted_input=args.trusted_input,
        )
    )


def _add_common_input_flag(parser):
    parser.add_argument(
        "--trusted-input",
        action="store_true",
        help=(
            "Allow raw torch.save(PyG Data) pickle input. Omit for the "
            "restricted tensor-only format produced by ggm_eval."
        ),
    )


def _add_python_flag(parser):
    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python interpreter used for the isolated worker. It must contain "
            "the selected upstream evaluator's dependencies."
        ),
    )


def _add_upstream_flags(parser, *, required=True):
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        required=required,
        help=(
            "Checkout of Self-Supervised-Models-for-GGM-Evaluation. "
            + (
                ""
                if required
                else "If omitted, use GGM_EVAL_CONTRASTIVE_REPO or discover "
                "a nearby checkout."
            )
        ),
    )
    parser.add_argument(
        "--allow-unpinned-upstream",
        action="store_true",
        help="Allow a checkout other than the documented pinned commit.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Validate and hash a PyG interchange collection.",
    )
    validate.add_argument("--graphs", type=Path, required=True)
    _add_common_input_flag(validate)
    validate.set_defaults(handler=_validate)

    dgl_to_pyg = subparsers.add_parser(
        "dgl-to-pyg",
        help="Convert an existing dgl.save_graphs file to the PyG contract.",
    )
    dgl_to_pyg.add_argument("--input", type=Path, required=True)
    dgl_to_pyg.add_argument("--output", type=Path, required=True)
    dgl_to_pyg.add_argument(
        "--dataset",
        help="Stable dataset identity stored in artifact metadata.",
    )
    dgl_to_pyg.add_argument(
        "--feature-schema",
        help="Stable node/edge channel-schema identity.",
    )
    dgl_to_pyg.set_defaults(handler=_convert_dgl)

    pyg_to_dgl = subparsers.add_parser(
        "pyg-to-dgl",
        help="Create a legacy dgl.save_graphs file from PyG.",
    )
    pyg_to_dgl.add_argument("--input", type=Path, required=True)
    pyg_to_dgl.add_argument("--output", type=Path, required=True)
    _add_common_input_flag(pyg_to_dgl)
    pyg_to_dgl.set_defaults(handler=_convert_pyg)

    inspect_upstream = subparsers.add_parser(
        "inspect-upstream",
        help="Validate the contrastive evaluator checkout and revision.",
    )
    _add_upstream_flags(inspect_upstream)
    inspect_upstream.set_defaults(handler=_inspect_upstream)

    train = subparsers.add_parser(
        "train",
        help="Train matched random, GraphCL, or InfoGraph encoder checkpoints.",
    )
    train.add_argument("--graphs", type=Path, required=True)
    train.add_argument(
        "--encoder",
        choices=("gin-random", "graphcl", "infograph"),
        required=True,
    )
    train.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="decoded_node_edge",
    )
    train.add_argument("--seeds", type=int, nargs="+", required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--num-layers", type=int, default=3)
    train.add_argument("--hidden-dim", type=int, default=32)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument(
        "--init", choices=("default", "orthogonal"), default="orthogonal"
    )
    lipschitz = train.add_mutually_exclusive_group()
    lipschitz.add_argument(
        "--limit-lipschitz",
        dest="limit_lipschitz",
        action="store_true",
    )
    lipschitz.add_argument(
        "--no-limit-lipschitz",
        dest="limit_lipschitz",
        action="store_false",
    )
    train.set_defaults(limit_lipschitz=True)
    train.add_argument("--lipschitz-factor", type=float, default=1.0)
    _add_upstream_flags(train)
    _add_common_input_flag(train)
    _add_python_flag(train)
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate generated/reference PyG sets with frozen checkpoints.",
    )
    evaluate.add_argument("--generated", type=Path, required=True)
    evaluate.add_argument("--reference", type=Path, required=True)
    evaluate.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Frozen encoder checkpoint. Repeat for independent encoder seeds.",
    )
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--nearest-k", type=int, default=5)
    evaluate.add_argument(
        "--max-graphs",
        type=int,
        default=0,
        help="Zero uses every reference graph.",
    )
    _add_upstream_flags(evaluate)
    _add_common_input_flag(evaluate)
    _add_python_flag(evaluate)
    evaluate.set_defaults(handler=_evaluate)

    trained = subparsers.add_parser(
        "evaluate-trained",
        help="Evaluate with bundled, dataset-matched GraphCL-GIN checkpoints.",
    )
    trained.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset identity or alias: PROTEINS, MUTAG, PTC, AIDS, "
            "ENZYMES, QM9, or ogbg-molbbbp."
        ),
    )
    trained.add_argument("--generated", type=Path, required=True)
    trained.add_argument("--reference", type=Path, required=True)
    trained.add_argument("--output-dir", type=Path, required=True)
    trained.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional subset of bundled seeds; defaults to 0 1 2.",
    )
    trained.add_argument("--device", default="auto")
    trained.add_argument("--nearest-k", type=int, default=5)
    trained.add_argument(
        "--max-graphs",
        type=int,
        default=0,
        help="Zero uses every reference graph.",
    )
    _add_upstream_flags(trained, required=False)
    _add_common_input_flag(trained)
    _add_python_flag(trained)
    trained.set_defaults(handler=_evaluate_trained)

    legacy = subparsers.add_parser(
        "evaluate-legacy",
        help="Run existing DGL Random-GIN through the PyG-to-DGL adapter.",
    )
    legacy.add_argument("--generated", type=Path, required=True)
    legacy.add_argument("--reference", type=Path, required=True)
    legacy.add_argument(
        "--legacy-repo",
        type=Path,
        required=True,
        help=(
            "GraphVAE-REQ checkout containing eval/attributed_gin.py and "
            "the vendored GGM-metrics subset."
        ),
    )
    legacy.add_argument("--output-dir", type=Path, required=True)
    legacy.add_argument("--modes", nargs="+", choices=FEATURE_MODES)
    legacy.add_argument("--repeats", type=int, default=10)
    legacy.add_argument("--evaluator-seed", type=int, default=0)
    legacy.add_argument("--nearest-k", type=int, default=5)
    legacy.add_argument("--max-graphs", type=int, default=0)
    legacy.add_argument("--device", default="cpu")
    _add_common_input_flag(legacy)
    _add_python_flag(legacy)
    legacy.set_defaults(handler=_evaluate_legacy)
    return parser


def main():
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
