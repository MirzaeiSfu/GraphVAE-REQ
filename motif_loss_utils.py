"""Utilities for motif-loss computation and motif diagnostics."""

import torch


def compute_motif_loss(observed_counts, predicted_counts, loss_mode="abs_log_ratio"):
    """
    Compute motif loss by normalizing inside each graph first, then averaging
    across graphs in the batch.

    This gives each graph equal weight regardless of how many motif types are
    active in that graph. Motifs with zero observed count are excluded from a
    graph's normalization term.

    loss_mode='abs_log_ratio' keeps the original absolute log-ratio penalty.
    loss_mode='squared_log_ratio' uses a squared log-ratio penalty, which is
    often smoother when fine-tuning near zero.
    """
    if observed_counts.shape != predicted_counts.shape:
        raise ValueError(
            f"Shape mismatch: observed {tuple(observed_counts.shape)} vs "
            f"predicted {tuple(predicted_counts.shape)}"
        )

    mask = observed_counts != 0
    if not mask.any():
        return torch.tensor(0.0, device=observed_counts.device)

    safe_observed = observed_counts.clamp(min=1e-8)
    safe_predicted = predicted_counts.clamp(min=1e-8)

    log_ratio = torch.log(safe_predicted / safe_observed)
    if loss_mode == "abs_log_ratio":
        per_motif_loss = torch.abs(log_ratio)
    elif loss_mode == "squared_log_ratio":
        per_motif_loss = log_ratio.pow(2)
    else:
        raise ValueError(f"Unknown motif loss mode: {loss_mode}")

    per_motif_loss = per_motif_loss * mask.to(per_motif_loss.dtype)

    active_motif_count = mask.sum(dim=1)
    valid_graph_mask = active_motif_count > 0
    if not valid_graph_mask.any():
        return torch.tensor(0.0, device=observed_counts.device)

    per_graph_loss = per_motif_loss.sum(dim=1)
    per_graph_loss = per_graph_loss[valid_graph_mask] / active_motif_count[valid_graph_mask].to(per_motif_loss.dtype)

    return per_graph_loss.mean()


def compute_hard_motif_metrics(observed_counts, hard_predicted_counts):
    """
    Compute evaluation-only motif metrics on the discretized reconstruction.

    `hard_motif_loss` reuses the absolute log-ratio penalty on hard motif
    counts so we can compare the thresholded graph against the target counts.
    `hard_motif_exact_zero` is stricter: it requires an exact count match for
    every motif entry, including motifs whose observed count is zero.
    """
    hard_motif_loss = compute_motif_loss(
        observed_counts=observed_counts,
        predicted_counts=hard_predicted_counts,
        loss_mode="abs_log_ratio",
    )

    exact_match = torch.isclose(
        hard_predicted_counts,
        observed_counts,
        atol=1e-6,
        rtol=0.0,
    )
    hard_motif_exact_zero_per_graph = exact_match.all(dim=1)
    hard_motif_exact_zero = hard_motif_exact_zero_per_graph.all()

    return hard_motif_loss, hard_motif_exact_zero, hard_motif_exact_zero_per_graph


def get_motif_temperature(epoch, total_epochs, start_temp, end_temp, anneal_start_frac):
    """
    Linearly anneal motif-count temperatures late in training so we keep the
    early optimization smooth and only sharpen the decoded logits near the end.
    """
    start_temp = max(float(start_temp), 1e-3)
    end_temp = max(float(end_temp), 1e-3)
    anneal_start_frac = min(max(float(anneal_start_frac), 0.0), 1.0)

    if total_epochs <= 1 or abs(start_temp - end_temp) < 1e-12:
        return start_temp

    progress = epoch / max(total_epochs - 1, 1)
    if progress <= anneal_start_frac:
        return start_temp

    anneal_progress = min(
        max((progress - anneal_start_frac) / max(1.0 - anneal_start_frac, 1e-8), 0.0),
        1.0,
    )
    return start_temp + (end_temp - start_temp) * anneal_progress


def get_reconstructed_adj_probs(reconstructed_adj, prob_temperature=1.0):
    """
    Convert the decoder output to adjacency probabilities once so evaluation
    can sweep multiple hard thresholds without rebuilding the full wrapper.
    """
    adj = reconstructed_adj.detach()
    if adj.dim() == 4:
        adj = adj.squeeze(-1)

    adj_min = adj.min().item()
    adj_max = adj.max().item()
    is_logit = (adj_min < -0.01) or (adj_max > 1.01)
    if is_logit:
        return torch.sigmoid(adj / max(float(prob_temperature), 1e-3))
    return adj


def summarize_hard_motif_threshold_sweep(
    observed_counts,
    adj_probs,
    hard_recon_wrapper,
    motif_counter,
    batch_size,
    thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
):
    """
    Evaluate a few hard thresholds to see whether the hard motif gap is mostly
    a cutoff issue or a deeper mismatch in the reconstructed graph.
    """
    original_all_adj = hard_recon_wrapper.all_adj
    relation_keys = list(original_all_adj.keys())
    sweep_parts = []

    for threshold in thresholds:
        thresholded_adj = (adj_probs >= threshold).to(adj_probs.dtype)
        hard_recon_wrapper.all_adj = {rk: thresholded_adj for rk in relation_keys}
        hard_counts = motif_counter.count_batch(hard_recon_wrapper, batch_size=batch_size)
        hard_loss, _, hard_exact_per_graph = compute_hard_motif_metrics(
            observed_counts=observed_counts,
            hard_predicted_counts=hard_counts,
        )
        hard_exact_count = int(hard_exact_per_graph.sum().item())
        sweep_parts.append(
            f"{threshold:.1f}:{hard_loss.item():.4f} ({hard_exact_count}/{hard_exact_per_graph.numel()})"
        )

    hard_recon_wrapper.all_adj = original_all_adj
    return "hard_threshold_sweep | " + " | ".join(sweep_parts)
