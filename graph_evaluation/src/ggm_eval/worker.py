"""Isolated worker entrypoint for upstream evaluator repositories.

Do not import this module from application code.  :mod:`ggm_eval.runner`
starts it in a fresh interpreter so the upstream ``evaluation`` module is
resolved from exactly one repository.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import random
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from . import __version__ as ADAPTER_VERSION
from .adapters import pyg_to_dgl
from .contract import (
    FEATURE_MODES,
    collection_digest,
    prepare_collection,
    validate_collection,
)
from .io import load_pyg_collection_with_metadata
from .reporting import write_json
from .upstreams import (
    validate_contrastive_upstream,
    validate_legacy_repository,
)


CHECKPOINT_FORMAT = "ggm-eval-upstream-gconv"
CHECKPOINT_VERSION = 1


def _runtime_versions() -> dict:
    """Collect installed distribution versions without importing backends."""

    versions = {
        "adapter": ADAPTER_VERSION,
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }
    distributions = (
        ("torch-geometric", "torch_geometric"),
        ("PyGCL", "pygcl"),
        ("scikit-learn", "scikit_learn"),
        ("scipy", "scipy"),
        ("dgl", "dgl"),
    )
    for distribution, output_name in distributions:
        try:
            versions[output_name] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {raw!r} requested but unavailable.")
    return torch.device(raw)


def _activate_repository(path: Path):
    """Make one upstream checkout own top-level imports in this process."""

    resolved = str(path.resolve())
    if resolved in sys.path:
        sys.path.remove(resolved)
    sys.path.insert(0, resolved)
    for module_name in tuple(sys.modules):
        if module_name == "evaluation" or module_name.startswith("evaluation."):
            del sys.modules[module_name]


def _install_upstream_import_shims():
    """Satisfy unused imports in the released PyG code.

    ``data_utils.py`` imports DGL and Ray at module import time, although the
    native-PyG, non-parallel path used here needs neither package.  Installing
    narrowly scoped stand-ins only when an import is unavailable keeps the
    PyG evaluator independent of those optional backends.  Every callable
    that would indicate accidental use fails loudly, except Ray's synchronous
    ``remote``/``get`` pair, which preserves the published non-distributed
    semantics.
    """

    try:
        importlib.import_module("dgl")
    except ImportError:
        sys.modules.pop("dgl", None)
        dgl_stub = ModuleType("dgl")

        class UnavailableDGLGraph:
            """Sentinel used only by upstream ``isinstance`` checks."""

        def unavailable_to_networkx(*_args, **_kwargs):
            raise RuntimeError(
                "The PyG evaluator unexpectedly entered an upstream DGL path."
            )

        dgl_stub.DGLGraph = UnavailableDGLGraph
        dgl_stub.DGLHeteroGraph = UnavailableDGLGraph
        dgl_stub.to_networkx = unavailable_to_networkx
        sys.modules["dgl"] = dgl_stub

    try:
        importlib.import_module("ray")
    except ImportError:
        sys.modules.pop("ray", None)
        ray_stub = ModuleType("ray")

        class SynchronousRemote:
            def __init__(self, function):
                self.function = function

            def remote(self, *arguments, **keywords):
                return self.function(*arguments, **keywords)

        ray_stub.remote = lambda function: SynchronousRemote(function)
        ray_stub.get = lambda values: values
        sys.modules["ray"] = ray_stub


def _mark_features_ready(graphs):
    """Tell upstream that the adapter already selected the model features."""

    for graph in graphs:
        graph.added_struct_feats = True
    return graphs


def _install_pyg_compatibility():
    """Expose input width expected by newer PyG ``GINEConv`` releases.

    The upstream MLP supports ``mlp[0].in_features``.  PyG 2.6 instead checks
    ``mlp.in_features`` when constructing the edge projection.  The property
    is metadata only; forward computation and state-dict keys are unchanged.
    """

    from evaluation.models.gin import gin_pyg

    if not hasattr(gin_pyg.MLP, "in_features"):
        gin_pyg.MLP.in_features = property(
            lambda model: model.linears[0].in_features
        )


def _install_metric_compatibility():
    """Load the sklearn namespace expected by the released PR/DC code."""

    try:
        importlib.import_module("sklearn.metrics")
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation requires scikit-learn for precision, recall, "
            "density, coverage, and MMD."
        ) from exc


def _validate_positive(value: int | float, *, name: str):
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_schema_identity(**collections: dict) -> dict:
    """Require declared dataset/schema identities to agree."""

    identity = {}
    for field in ("dataset", "feature_schema"):
        values = {
            name: metadata.get(field)
            for name, metadata in collections.items()
        }
        declared = [value for value in values.values() if value is not None]
        if declared and (
            len(declared) != len(values)
            or any(value != declared[0] for value in declared[1:])
        ):
            raise ValueError(
                f"Collection metadata field {field!r} differs: {values}."
            )
        identity[field] = None if not declared else declared[0]
    return identity


def _model_args(
    *,
    input_dim: int,
    edge_dim: int,
    num_layers: int,
    hidden_dim: int,
    init: str,
    limit_lipschitz: bool,
    lipschitz_factor: float,
    device: torch.device,
    epochs: int = 0,
    encoder: str = "gin-random",
    seed: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_dim=input_dim,
        edge_dim=None if edge_dim == 0 else edge_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        init=init,
        limit_lip=limit_lipschitz,
        lip_factor=lipschitz_factor,
        device=device,
        epochs=epochs,
        feature_extractor=encoder,
        seed=seed,
        is_parallel=False,
        deg_feats=False,
        clus_feats=False,
        orbit_feats=False,
        model_name=f"ggm_eval_{encoder}_seed_{seed}",
    )


def _safe_torch_load(path: Path):
    parameters = inspect.signature(torch.load).parameters
    kwargs = {"map_location": "cpu"}
    if "weights_only" in parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def _train(args):
    _validate_positive(args.num_layers, name="num_layers")
    _validate_positive(args.hidden_dim, name="hidden_dim")
    _validate_positive(args.lipschitz_factor, name="lipschitz_factor")
    if args.encoder != "gin-random":
        _validate_positive(args.epochs, name="epochs")
    upstream = validate_contrastive_upstream(
        args.upstream_repo,
        allow_unpinned=args.allow_unpinned_upstream,
    )
    _activate_repository(Path(upstream["checkout"]))
    raw_graphs, training_metadata = load_pyg_collection_with_metadata(
        args.graphs, trusted=args.trusted_input, normalize=True
    )
    graphs = prepare_collection(
        raw_graphs,
        mode=args.feature_mode,
        name="encoder training graphs",
        minimum_graphs=2,
    )
    summary = validate_collection(graphs, name="encoder training graphs")
    device = _resolve_device(args.device)
    _seed_everything(args.seed)
    model_args = _model_args(
        input_dim=summary.node_feature_dim,
        edge_dim=summary.edge_feature_dim,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        init=args.init,
        limit_lipschitz=args.limit_lipschitz,
        lipschitz_factor=args.lipschitz_factor,
        device=device,
        epochs=args.epochs,
        encoder=args.encoder,
        seed=args.seed,
    )

    started = time.time()
    _install_pyg_compatibility()
    if args.encoder == "gin-random":
        from evaluation.models.gin.gin_pyg import GConv

        model = GConv(model_args).to(device)
        training_loss = None
    else:
        _install_upstream_import_shims()
        try:
            from GIN_train_pyg import get_model
        except ImportError as exc:
            raise RuntimeError(
                "GraphCL/InfoGraph training requires the upstream dependencies, "
                "including PyGCL (import name GCL)."
            ) from exc
        # The released GraphCL loop does not move batches to the configured
        # device. Supplying device-resident Data objects preserves its code.
        upstream_graphs = [graph.clone().to(device) for graph in graphs]
        model, training_loss = get_model(upstream_graphs, model_args)

    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "checkpoint.pt"
    model_config = {
        "input_dim": summary.node_feature_dim,
        "edge_dim": summary.edge_feature_dim,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "init": args.init,
        "limit_lipschitz": args.limit_lipschitz,
        "lipschitz_factor": args.lipschitz_factor,
    }
    training_config = {
        "trained": args.encoder != "gin-random",
        "epochs": 0 if args.encoder == "gin-random" else args.epochs,
    }
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "state_dict": model.state_dict(),
            "model": model_config,
            "training": training_config,
            "training_metadata": training_metadata,
            "encoder": args.encoder,
            "feature_mode": args.feature_mode,
            "seed": args.seed,
            "upstream_revision": upstream["revision"],
            "adapter_version": ADAPTER_VERSION,
        },
        checkpoint_path,
    )
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_version": CHECKPOINT_VERSION,
        "encoder": args.encoder,
        "feature_mode": args.feature_mode,
        "seed": args.seed,
        "model": model_config,
        "training": training_config,
        "training_metadata": training_metadata,
        "training_graphs": summary.to_dict(),
        "training_collection_sha256": collection_digest(
            raw_graphs, mode=args.feature_mode
        ),
        "training_loss": (
            None if training_loss is None else float(training_loss)
        ),
        "elapsed_seconds": float(time.time() - started),
        "upstream": upstream,
        "versions": _runtime_versions(),
    }
    write_json(destination / "training.json", manifest)


def _load_encoder(checkpoint_path: Path, *, upstream: dict, device):
    checkpoint = _safe_torch_load(checkpoint_path)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")
    if int(checkpoint.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version: {checkpoint.get('version')}."
        )
    if checkpoint.get("upstream_revision") != upstream["revision"]:
        raise ValueError(
            "Checkpoint/upstream revision mismatch: checkpoint uses "
            f"{checkpoint.get('upstream_revision')!r}, evaluator checkout "
            f"uses {upstream['revision']!r}."
        )
    _activate_repository(Path(upstream["checkout"]))
    _install_pyg_compatibility()
    from evaluation.models.gin.gin_pyg import GConv

    config = checkpoint["model"]
    model_args = _model_args(
        input_dim=int(config["input_dim"]),
        edge_dim=int(config["edge_dim"]),
        num_layers=int(config["num_layers"]),
        hidden_dim=int(config["hidden_dim"]),
        init=str(config["init"]),
        limit_lipschitz=bool(config["limit_lipschitz"]),
        lipschitz_factor=float(config["lipschitz_factor"]),
        device=device,
    )
    model = GConv(model_args)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.device = device
    model.eval()
    return model.to(device), checkpoint


def _metric_value(result: dict) -> dict:
    converted = {name: float(value) for name, value in result.items()}
    nonfinite = [
        name for name, value in converted.items() if not np.isfinite(value)
    ]
    if nonfinite:
        raise RuntimeError(
            f"Upstream metrics produced non-finite values: {nonfinite}."
        )
    return converted


def _evaluate_contrastive(args):
    _validate_positive(args.nearest_k, name="nearest_k")
    if args.max_graphs < 0 or args.max_graphs in {1, 2}:
        raise ValueError("max_graphs must be zero or at least three.")
    upstream = validate_contrastive_upstream(
        args.upstream_repo,
        allow_unpinned=args.allow_unpinned_upstream,
    )
    device = _resolve_device(args.device)
    model, checkpoint = _load_encoder(
        Path(args.checkpoint).expanduser().resolve(),
        upstream=upstream,
        device=device,
    )
    mode = checkpoint["feature_mode"]
    generated_raw, generated_metadata = load_pyg_collection_with_metadata(
        args.generated, trusted=args.trusted_input, normalize=True
    )
    reference_raw, reference_metadata = load_pyg_collection_with_metadata(
        args.reference, trusted=args.trusted_input, normalize=True
    )
    schema_identity = _validate_schema_identity(
        training=checkpoint["training_metadata"],
        generated=generated_metadata,
        reference=reference_metadata,
    )
    generated = prepare_collection(
        generated_raw,
        mode=mode,
        name="generated graphs",
        minimum_graphs=3,
    )
    reference = prepare_collection(
        reference_raw,
        mode=mode,
        name="reference graphs",
        minimum_graphs=3,
    )
    _mark_features_ready(generated)
    _mark_features_ready(reference)
    if args.max_graphs:
        reference = reference[: args.max_graphs]
    if len(generated) < len(reference):
        raise ValueError(
            f"Generated collection has {len(generated)} graphs but evaluation "
            f"selected {len(reference)} reference graphs."
        )
    generated = generated[: len(reference)]

    generated_summary = validate_collection(generated, name="generated graphs")
    reference_summary = validate_collection(reference, name="reference graphs")
    expected = checkpoint["model"]
    actual_dims = (
        generated_summary.node_feature_dim,
        generated_summary.edge_feature_dim,
    )
    expected_dims = (int(expected["input_dim"]), int(expected["edge_dim"]))
    if actual_dims != expected_dims:
        raise ValueError(
            f"Checkpoint expects feature dimensions {expected_dims}, got "
            f"{actual_dims}."
        )
    if (
        generated_summary.node_feature_dim
        != reference_summary.node_feature_dim
        or generated_summary.edge_feature_dim
        != reference_summary.edge_feature_dim
    ):
        raise ValueError("Generated/reference feature dimensions differ.")

    _install_upstream_import_shims()
    _install_metric_compatibility()
    from evaluation import gin_evaluation

    feature_args = {"d": False, "c": False, "o": False, "is_parallel": False}
    activation_metric = gin_evaluation.FIDEvaluation(
        model=model, feature_adder_args=feature_args
    )
    (generated_activations, reference_activations), activation_seconds = (
        activation_metric.get_activations(generated, reference)
    )
    effective_k = min(
        args.nearest_k, min(len(generated), len(reference)) - 2
    )

    metrics = {}
    fid, _ = activation_metric.evaluate(
        generated_activations, reference_activations
    )
    metrics.update(_metric_value(fid))
    precision_recall, _ = gin_evaluation.prdcEvaluation(
        model=model, feature_adder_args=feature_args, use_pr=True
    ).evaluate(
        generated_activations,
        reference_activations,
        nearest_k=effective_k,
    )
    metrics.update(_metric_value(precision_recall))
    density_coverage, _ = gin_evaluation.prdcEvaluation(
        model=model, feature_adder_args=feature_args, use_pr=False
    ).evaluate(
        generated_activations,
        reference_activations,
        nearest_k=effective_k,
    )
    metrics.update(_metric_value(density_coverage))
    mmd_rbf, _ = gin_evaluation.MMDEvaluation(
        model=model,
        feature_adder_args=feature_args,
        kernel="rbf",
        sigma="range",
        multiplier="mean",
    ).evaluate(generated_activations, reference_activations)
    metrics.update(_metric_value(mmd_rbf))
    mmd_linear, _ = gin_evaluation.MMDEvaluation(
        model=model,
        feature_adder_args=feature_args,
        kernel="linear",
    ).evaluate(generated_activations, reference_activations)
    metrics.update(_metric_value(mmd_linear))

    payload = {
        "engine": "contrastive-pyg-upstream",
        "encoder": checkpoint["encoder"],
        "feature_mode": mode,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_seed": int(checkpoint["seed"]),
        "model": dict(checkpoint["model"]),
        "training": dict(checkpoint["training"]),
        "training_metadata": dict(checkpoint["training_metadata"]),
        "generated_metadata": generated_metadata,
        "reference_metadata": reference_metadata,
        "schema_identity": schema_identity,
        "upstream_revision": checkpoint["upstream_revision"],
        "metrics": metrics,
        "nearest_k": effective_k,
        "activation_seconds": float(activation_seconds),
        "activation_dim": int(generated_activations.shape[1]),
        "generated_graphs": generated_summary.to_dict(),
        "reference_graphs": reference_summary.to_dict(),
        "generated_sha256": collection_digest(generated, mode=mode),
        "reference_sha256": collection_digest(reference, mode=mode),
        "upstream": upstream,
        "versions": _runtime_versions(),
    }
    write_json(args.output, payload)


def _evaluate_legacy(args):
    _validate_positive(args.repeats, name="repeats")
    _validate_positive(args.nearest_k, name="nearest_k")
    if args.max_graphs < 0 or args.max_graphs in {1, 2}:
        raise ValueError("max_graphs must be zero or at least three.")
    legacy = validate_legacy_repository(args.legacy_repo)
    generated_raw, generated_metadata = load_pyg_collection_with_metadata(
        args.generated, trusted=args.trusted_input, normalize=True
    )
    reference_raw, reference_metadata = load_pyg_collection_with_metadata(
        args.reference, trusted=args.trusted_input, normalize=True
    )
    schema_identity = _validate_schema_identity(
        generated=generated_metadata,
        reference=reference_metadata,
    )
    if args.max_graphs:
        reference_raw = reference_raw[: args.max_graphs]
    if len(generated_raw) < len(reference_raw):
        raise ValueError(
            f"Generated collection has {len(generated_raw)} graphs but "
            f"{len(reference_raw)} are required."
        )
    generated_raw = generated_raw[: len(reference_raw)]
    generated_dgl = [
        pyg_to_dgl(graph, name=f"generated[{index}]")
        for index, graph in enumerate(generated_raw)
    ]
    reference_dgl = [
        pyg_to_dgl(graph, name=f"reference[{index}]")
        for index, graph in enumerate(reference_raw)
    ]

    repository = Path(legacy["checkout"])
    sys.path.insert(0, str(repository))
    from eval.attributed_gin import evaluate_dgl_feature_modes

    modes = args.modes
    if modes is None:
        modes = (
            [
                "topology_control",
                "decoded_node",
                "decoded_edge",
                "decoded_node_edge",
            ]
            if getattr(reference_raw[0], "edge_attr", None) is not None
            else ["topology_control", "decoded_node"]
        )
    evaluation = evaluate_dgl_feature_modes(
        generated_dgl,
        reference_dgl,
        modes=modes,
        repeats=args.repeats,
        seed=args.evaluator_seed,
        nearest_k=args.nearest_k,
        device=_resolve_device(args.device),
    )
    write_json(
        args.output,
        {
            "engine": "legacy-dgl-random-gin",
            "adapter": "strict-pyg-to-dgl",
            "generated": str(Path(args.generated).expanduser().resolve()),
            "reference": str(Path(args.reference).expanduser().resolve()),
            "generated_sha256": collection_digest(generated_raw),
            "reference_sha256": collection_digest(reference_raw),
            "generated_metadata": generated_metadata,
            "reference_metadata": reference_metadata,
            "schema_identity": schema_identity,
            "versions": _runtime_versions(),
            "evaluation": evaluation,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("contrastive-train")
    train.add_argument("--graphs", required=True)
    train.add_argument("--upstream-repo", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument(
        "--encoder",
        choices=("gin-random", "graphcl", "infograph"),
        required=True,
    )
    train.add_argument("--feature-mode", choices=FEATURE_MODES, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--num-layers", type=int, default=3)
    train.add_argument("--hidden-dim", type=int, default=32)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument(
        "--init",
        choices=("default", "orthogonal"),
        default="orthogonal",
    )
    train.add_argument("--limit-lipschitz", action="store_true")
    train.add_argument("--lipschitz-factor", type=float, default=1.0)
    train.add_argument("--trusted-input", action="store_true")
    train.add_argument("--allow-unpinned-upstream", action="store_true")
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser("contrastive-evaluate")
    evaluate.add_argument("--generated", required=True)
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--upstream-repo", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--nearest-k", type=int, default=5)
    evaluate.add_argument("--max-graphs", type=int, default=0)
    evaluate.add_argument("--trusted-input", action="store_true")
    evaluate.add_argument("--allow-unpinned-upstream", action="store_true")
    evaluate.set_defaults(handler=_evaluate_contrastive)

    legacy = subparsers.add_parser("legacy-evaluate")
    legacy.add_argument("--generated", required=True)
    legacy.add_argument("--reference", required=True)
    legacy.add_argument("--legacy-repo", required=True)
    legacy.add_argument("--output", required=True)
    legacy.add_argument("--modes", nargs="+", choices=FEATURE_MODES)
    legacy.add_argument("--repeats", type=int, default=10)
    legacy.add_argument("--evaluator-seed", type=int, default=0)
    legacy.add_argument("--nearest-k", type=int, default=5)
    legacy.add_argument("--max-graphs", type=int, default=0)
    legacy.add_argument("--device", default="cpu")
    legacy.add_argument("--trusted-input", action="store_true")
    legacy.set_defaults(handler=_evaluate_legacy)
    return parser


def main():
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
