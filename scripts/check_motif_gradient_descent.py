#!/usr/bin/env python3
"""Verify that motif loss alone backpropagates through GraphVAE decoders.

The diagnostic loads a real graph and the real FactorBase motif cache, then
optimizes the actual AveEncoder, adjacency decoder, and node-feature decoder
using motif loss alone. No reconstruction, KL, kernel, or
feature-reconstruction loss is included.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import dgl


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import (  # noqa: E402
    BFS,
    DataWrapper,
    Datasets,
    ReconstructedDataWrapper,
    list_graph_loader,
    merge_datasets,
)
from model import AveEncoder, GraphTransformerDecoder_FC  # noqa: E402
from motif_counting.motif_counter import RelationalMotifCounter  # noqa: E402
from motif_counting.motif_loss_utils import compute_motif_loss  # noqa: E402
from util import (  # noqa: E402
    NodeFeatureDecoder,
    build_onehot_features,
    remove_self_loops,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("MUTAG", "PTC"))
    parser.add_argument("--database-name", required=True)
    parser.add_argument("--motif-cache-dir", type=Path, default=Path("cache_motifs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument(
        "--loss-mode",
        choices=("abs_log_ratio", "squared_log_ratio", "calibrated_gaussian"),
        default="calibrated_gaussian",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def parameter_grad_norm(module: torch.nn.Module) -> float:
    squared_norm = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        squared_norm += float(parameter.grad.detach().pow(2).sum().cpu().item())
    return math.sqrt(squared_norm)


def finite_parameter_gradients(module: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )


def main() -> None:
    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    (
        all_adjs,
        _all_x,
        _all_labels,
        all_node_features,
        all_edge_features,
        node_feature_info,
        edge_feature_info,
    ) = list_graph_loader(cli.dataset, return_labels=True)
    all_adjs, all_node_features, all_edge_features = BFS(
        all_adjs,
        all_node_features,
        all_edge_features,
    )
    node_onehots, edge_onehots, node_onehot_info, edge_onehot_info = (
        build_onehot_features(
            all_node_features,
            all_edge_features,
            all_adjs,
            node_feature_info,
            edge_feature_info,
        )
    )

    # Use a real largest graph so the diagnostic has no padded nodes. This
    # prevents padding probabilities from becoming an unintended motif target.
    graph_index = max(range(len(all_adjs)), key=lambda index: all_adjs[index].shape[0])
    adjacency = all_adjs[graph_index]
    node_onehot = node_onehots[graph_index]
    edge_onehot = edge_onehots[graph_index]
    max_nodes = int(adjacency.shape[0])

    dataset = Datasets(
        [adjacency],
        True,
        [None],
        None,
        Max_num=max_nodes,
        set_diag_of_isol_Zer=False,
        list_node_onehot=[node_onehot],
        list_edge_onehot=[edge_onehot],
    )
    remove_self_loops(dataset)

    counter_args = SimpleNamespace(
        motif_cache_dir=str(cli.motif_cache_dir),
        use_syntactic_literal_rules=False,
        syntactic_literal_rule_mode="original",
        rule_prune=False,
        motif_prune_max_values_per_rule=None,
        device=str(device),
    )
    motif_counter = RelationalMotifCounter(cli.database_name, counter_args)
    observed_wrapper = DataWrapper(
        merge_datasets(dataset),
        motif_counter.relation_keys,
        node_onehot_info=node_onehot_info,
        edge_onehot_info=edge_onehot_info,
        edge_feature_info_mapping=motif_counter.feature_info_mapping,
        device=str(device),
    )
    observed_counts = motif_counter.count_batch(observed_wrapper, batch_size=1).detach()

    node_dim = int(node_onehot.shape[-1])
    encoder_adjacency = dataset.processed_adjs[0].copy().tolil()
    encoder_adjacency.setdiag(1)
    encoder_graph = dgl.from_scipy(encoder_adjacency.tocsr()).to(device)
    encoder_features = dataset.processed_Xs[0].to(device)
    encoder = AveEncoder(
        dataset.feature_size,
        [256],
        cli.latent_dim,
    ).to(device)
    adjacency_decoder = GraphTransformerDecoder_FC(
        cli.latent_dim,
        256,
        max_nodes,
        directed=True,
    ).to(device)
    node_decoder = NodeFeatureDecoder(
        cli.latent_dim,
        max_nodes,
        node_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(adjacency_decoder.parameters())
        + list(node_decoder.parameters()),
        lr=cli.learning_rate,
    )

    def motif_loss() -> torch.Tensor:
        latent, _log_std = encoder(
            encoder_graph,
            encoder_features,
            [1, max_nodes],
        )
        adjacency_logits = adjacency_decoder(latent)
        node_logits = node_decoder(latent)
        reconstructed = ReconstructedDataWrapper(
            reconstructed_adj=adjacency_logits,
            node_feat_logits=node_logits,
            edge_feat_logits=None,
            relation_keys=motif_counter.relation_keys,
            node_onehot_info=node_onehot_info,
            feature_onehot_mapping=observed_wrapper.feature_onehot_mapping,
            edge_onehot_info=edge_onehot_info,
            edge_feature_info_mapping=motif_counter.feature_info_mapping,
            use_soft_adj=True,
            prob_temperature=1.0,
            device=str(device),
        )
        predicted_counts = motif_counter.count_batch(reconstructed, batch_size=1)
        return compute_motif_loss(
            observed_counts=observed_counts,
            predicted_counts=predicted_counts,
            loss_mode=cli.loss_mode,
        )

    losses = []
    adjacency_grad_norm = None
    node_grad_norm = None
    encoder_grad_norm = None
    adjacency_gradients_finite = False
    node_gradients_finite = False
    encoder_gradients_finite = False
    report_every = max(1, cli.steps // 4)

    for step in range(cli.steps):
        optimizer.zero_grad(set_to_none=True)
        with contextlib.redirect_stdout(io.StringIO()):
            loss = motif_loss()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"Non-finite motif loss at step {step}: {loss.item()}")
        loss.backward()

        current_adjacency_grad_norm = parameter_grad_norm(adjacency_decoder)
        current_node_grad_norm = parameter_grad_norm(node_decoder)
        current_encoder_grad_norm = parameter_grad_norm(encoder)
        if step == 0:
            adjacency_grad_norm = current_adjacency_grad_norm
            node_grad_norm = current_node_grad_norm
            encoder_grad_norm = current_encoder_grad_norm
            adjacency_gradients_finite = finite_parameter_gradients(adjacency_decoder)
            node_gradients_finite = finite_parameter_gradients(node_decoder)
            encoder_gradients_finite = finite_parameter_gradients(encoder)

        losses.append(float(loss.detach().cpu().item()))
        optimizer.step()
        if step == 0 or (step + 1) % report_every == 0 or step + 1 == cli.steps:
            print(
                f"step={step + 1:04d} motif_loss={losses[-1]:.8f} "
                f"encoder_grad={current_encoder_grad_norm:.8g} "
                f"adj_grad={current_adjacency_grad_norm:.8g} "
                f"node_grad={current_node_grad_norm:.8g}"
            )

    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        final_loss = float(motif_loss().detach().cpu().item())
    initial_loss = losses[0]
    best_loss = min(min(losses), final_loss)
    decreased = best_loss < initial_loss - 1e-8
    passed = bool(
        decreased
        and adjacency_grad_norm is not None
        and adjacency_grad_norm > 0.0
        and node_grad_norm is not None
        and node_grad_norm > 0.0
        and encoder_grad_norm is not None
        and encoder_grad_norm > 0.0
        and adjacency_gradients_finite
        and node_gradients_finite
        and encoder_gradients_finite
    )

    payload = {
        "dataset": cli.dataset,
        "database_name": cli.database_name,
        "graph_index": graph_index,
        "num_nodes": max_nodes,
        "num_motif_entries": int(observed_counts.shape[1]),
        "loss_mode": cli.loss_mode,
        "steps": cli.steps,
        "learning_rate": cli.learning_rate,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "loss_decreased": decreased,
        "first_adjacency_decoder_grad_norm": adjacency_grad_norm,
        "first_node_decoder_grad_norm": node_grad_norm,
        "first_encoder_grad_norm": encoder_grad_norm,
        "adjacency_gradients_finite": adjacency_gradients_finite,
        "node_gradients_finite": node_gradients_finite,
        "encoder_gradients_finite": encoder_gradients_finite,
        "passed": passed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if cli.output is not None:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Saved diagnostic: {cli.output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
