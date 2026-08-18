"""Utilities for motif-loss computation and motif diagnostics."""

import math
import torch
import torch.nn.functional as F


CALIBRATED_GAUSSIAN_MOTIF_LOSS_MODES = {
    "calibrated_gaussian",
}


def _validate_motif_count_shapes(observed_counts, predicted_counts):
    if observed_counts.shape != predicted_counts.shape:
        raise ValueError(
            f"Shape mismatch: observed {tuple(observed_counts.shape)} vs "
            f"predicted {tuple(predicted_counts.shape)}"
        )


def _apply_log_ratio_loss_mode(log_ratio, loss_mode):
    if loss_mode == "abs_log_ratio":
        return torch.abs(log_ratio)
    if loss_mode == "squared_log_ratio":
        return log_ratio.pow(2)
    raise ValueError(f"Unknown motif loss mode: {loss_mode}")


def _softclip_min(tensor, minimum):
    return minimum + F.softplus(tensor - minimum)


def compute_calibrated_gaussian_motif_loss(
    observed_counts,
    predicted_counts,
    min_log_sigma=-6.0,
    eps=1e-12,
):
    """
    Compute Kia-MM style calibrated Gaussian NLL for motif counts.

    This is the Gaussian negative log-likelihood, commonly called Gaussian
    NLL, with sigma calibrated from the current minibatch.

    Each motif-vector column is treated as its own graph statistic. For motif
    column u, sigma_u is estimated from the minibatch RMSE between the observed
    counts and the reconstructed expected counts. The loss then evaluates the
    Gaussian negative log-likelihood of the observed count under
    N(predicted_count, sigma_u^2).
    """
    _validate_motif_count_shapes(observed_counts, predicted_counts)

    if observed_counts.shape[-1] == 0:
        return torch.tensor(0.0, device=observed_counts.device)

    return compute_calibrated_gaussian_motif_statistic_loss(
        observed_statistics=observed_counts,
        predicted_statistics=predicted_counts,
        min_log_sigma=min_log_sigma,
        eps=eps,
        reduction="mean",
    )


def compute_calibrated_gaussian_motif_statistic_loss(
    observed_statistics,
    predicted_statistics,
    valid_mask=None,
    min_log_sigma=-6.0,
    eps=1e-12,
    reduction="mean",
):
    """Compute one calibrated Gaussian loss per motif statistic.

    Inputs have shape ``(B, M, ...)``.  Each motif ``M`` receives one sigma,
    estimated from its minibatch RMSE across every valid trailing entry.  A
    shared ``valid_mask`` has shape ``(M, ...)`` and excludes representation
    padding.  With no trailing dimensions (scalar total counts), no mask is
    needed.

    This is the base definition for full matrices and scalar total counts.
    ``compute_calibrated_gaussian_motif_channel_loss`` applies the same formula
    independently to row/column marginal or histogram channels. The objective
    is MSE-derived but is a calibrated Gaussian NLL rather than plain MSE.
    """
    if observed_statistics.shape != predicted_statistics.shape:
        raise ValueError(
            f"Shape mismatch: observed {tuple(observed_statistics.shape)} vs "
            f"predicted {tuple(predicted_statistics.shape)}"
        )
    if observed_statistics.ndim < 2:
        raise ValueError(
            "Expected motif statistics with shape (B, M, ...), "
            f"got {tuple(observed_statistics.shape)}."
        )
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(
            f"Unknown motif statistic loss reduction: {reduction}. "
            "Expected 'mean', 'sum', or 'none'."
        )

    batch_size, num_motifs = observed_statistics.shape[:2]
    if batch_size == 0:
        raise ValueError("Cannot calibrate motif statistics from an empty batch.")
    if num_motifs == 0:
        per_motif_loss = predicted_statistics.new_empty((0,))
        if reduction == "none":
            return per_motif_loss
        return predicted_statistics.sum() * 0.0

    expected_mask_shape = tuple(observed_statistics.shape[1:])
    if valid_mask is None:
        valid_mask = torch.ones(
            expected_mask_shape,
            dtype=torch.bool,
            device=predicted_statistics.device,
        )
    elif tuple(valid_mask.shape) != expected_mask_shape:
        raise ValueError(
            "Motif statistic mask must match (M, ...): "
            f"mask={tuple(valid_mask.shape)}, "
            f"statistics={tuple(observed_statistics.shape)}."
        )
    else:
        valid_mask = valid_mask.to(
            device=predicted_statistics.device,
            dtype=torch.bool,
        )

    valid_spatial_count = valid_mask.reshape(num_motifs, -1).sum(dim=1)
    if (valid_spatial_count == 0).any():
        invalid_indices = torch.nonzero(
            valid_spatial_count == 0,
            as_tuple=False,
        ).flatten().detach().cpu().tolist()
        raise ValueError(
            "Every motif statistic must contain at least one valid entry; "
            f"empty masks found for motif indices {invalid_indices}."
        )

    residual = observed_statistics - predicted_statistics
    masked_residual = torch.where(
        valid_mask.unsqueeze(0),
        residual,
        torch.zeros((), dtype=residual.dtype, device=residual.device),
    )
    reduction_dims = (0,) + tuple(range(2, residual.ndim))
    valid_entry_count = valid_spatial_count.to(residual.dtype) * batch_size
    squared_error_sum = masked_residual.pow(2).sum(dim=reduction_dims)
    mse_by_motif = squared_error_sum / valid_entry_count
    rmse_by_motif = mse_by_motif.clamp_min(float(eps) ** 2).sqrt()
    log_sigma = _softclip_min(torch.log(rmse_by_motif), float(min_log_sigma))

    per_motif_loss = (
        0.5
        * squared_error_sum
        / (torch.exp(2.0 * log_sigma) * valid_entry_count)
        + log_sigma
        + 0.5 * math.log(2.0 * math.pi)
    )

    if reduction == "none":
        return per_motif_loss
    if reduction == "mean":
        return per_motif_loss.mean()
    return per_motif_loss.sum()


def compute_calibrated_gaussian_motif_channel_loss(
    observed_statistics,
    predicted_statistics,
    valid_mask,
    min_log_sigma=-6.0,
    eps=1e-12,
    reduction="mean",
):
    """Calibrate direction channels separately, then average per motif.

    ``observed_statistics`` has shape ``(B, M, C, ...)`` and ``valid_mask``
    has shape ``(M, C, ...)``. Each valid channel gets its own RMSE-calibrated
    sigma. Channel losses are then averaged within a motif, so an ``N x N``
    result with row and column channels does not receive twice the outer weight
    of an ``N x 1`` or ``1 x N`` result with only one meaningful channel.
    """
    if observed_statistics.shape != predicted_statistics.shape:
        raise ValueError(
            f"Shape mismatch: observed {tuple(observed_statistics.shape)} vs "
            f"predicted {tuple(predicted_statistics.shape)}"
        )
    if observed_statistics.ndim < 3:
        raise ValueError(
            "Expected channel-valued motif statistics with shape (B, M, C, ...), "
            f"got {tuple(observed_statistics.shape)}."
        )
    if tuple(valid_mask.shape) != tuple(observed_statistics.shape[1:]):
        raise ValueError(
            "Motif channel mask must match (M, C, ...): "
            f"mask={tuple(valid_mask.shape)}, "
            f"statistics={tuple(observed_statistics.shape)}."
        )
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(
            f"Unknown motif channel loss reduction: {reduction}. "
            "Expected 'mean', 'sum', or 'none'."
        )

    batch_size, num_motifs, num_channels = observed_statistics.shape[:3]
    if num_motifs == 0:
        per_motif_loss = predicted_statistics.new_empty((0,))
        if reduction == "none":
            return per_motif_loss
        return predicted_statistics.sum() * 0.0

    valid_mask = valid_mask.to(
        device=predicted_statistics.device,
        dtype=torch.bool,
    )
    flattened_mask = valid_mask.reshape(
        num_motifs * num_channels,
        *valid_mask.shape[2:],
    )
    valid_channels = flattened_mask.reshape(
        num_motifs * num_channels,
        -1,
    ).any(dim=1)
    valid_channel_count_by_motif = valid_channels.reshape(
        num_motifs,
        num_channels,
    ).sum(dim=1)
    if (valid_channel_count_by_motif == 0).any():
        invalid_indices = torch.nonzero(
            valid_channel_count_by_motif == 0,
            as_tuple=False,
        ).flatten().detach().cpu().tolist()
        raise ValueError(
            "Every motif must contain at least one valid statistic channel; "
            f"empty channel masks found for motif indices {invalid_indices}."
        )

    flattened_observed = observed_statistics.reshape(
        batch_size,
        num_motifs * num_channels,
        *observed_statistics.shape[3:],
    )
    flattened_predicted = predicted_statistics.reshape_as(flattened_observed)
    active_channel_losses = compute_calibrated_gaussian_motif_statistic_loss(
        observed_statistics=flattened_observed[:, valid_channels],
        predicted_statistics=flattened_predicted[:, valid_channels],
        valid_mask=flattened_mask[valid_channels],
        min_log_sigma=min_log_sigma,
        eps=eps,
        reduction="none",
    )

    active_flat_indices = torch.nonzero(
        valid_channels,
        as_tuple=False,
    ).flatten()
    active_motif_indices = active_flat_indices // num_channels
    summed_loss_by_motif = active_channel_losses.new_zeros(
        (num_motifs,)
    ).index_add(0, active_motif_indices, active_channel_losses)
    per_motif_loss = summed_loss_by_motif / valid_channel_count_by_motif.to(
        active_channel_losses.dtype
    )

    if reduction == "none":
        return per_motif_loss
    if reduction == "mean":
        return per_motif_loss.mean()
    return per_motif_loss.sum()


def compute_calibrated_gaussian_kiarash_statistics_loss(
    observed_statistics,
    predicted_statistics,
    min_log_sigma=-6.0,
    eps=1e-12,
    reduction="sum",
):
    """Apply GraphVAE-MM's separate-sigma loss to its eight kernel statistics.

    The five transition matrices, two degree histograms, and triangle scalar
    are heterogeneous tensors. Each receives its own minibatch-RMSE sigma and
    Gaussian NLL, exactly like the loop in ``main.OptimizerVAE``. The legacy
    outer reduction is a sum across all eight statistics.
    """
    if len(observed_statistics) != 8 or len(predicted_statistics) != 8:
        raise ValueError(
            "Kiarash statistics must contain exactly eight tensors: "
            "P^1..P^5, in/out degree histograms, and total triangles."
        )
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(
            f"Unknown Kiarash-statistics loss reduction: {reduction}. "
            "Expected 'mean', 'sum', or 'none'."
        )

    statistic_losses = []
    for observed, predicted in zip(observed_statistics, predicted_statistics):
        if observed.shape != predicted.shape:
            raise ValueError(
                "Kiarash statistic shape mismatch: "
                f"observed {tuple(observed.shape)} vs "
                f"predicted {tuple(predicted.shape)}."
            )
        if observed.ndim < 1:
            raise ValueError("Every Kiarash statistic must include a batch dimension.")
        per_statistic_loss = compute_calibrated_gaussian_motif_statistic_loss(
            observed_statistics=observed.unsqueeze(1),
            predicted_statistics=predicted.unsqueeze(1),
            min_log_sigma=min_log_sigma,
            eps=eps,
            reduction="none",
        )
        statistic_losses.append(per_statistic_loss[0])

    stacked_losses = torch.stack(statistic_losses)
    if reduction == "none":
        return stacked_losses
    if reduction == "mean":
        return stacked_losses.mean()
    return stacked_losses.sum()


def compute_calibrated_gaussian_motif_matrix_loss(
    observed_matrices,
    predicted_matrices,
    valid_mask,
    min_log_sigma=-6.0,
    eps=1e-12,
    reduction="mean",
):
    """
    Compute Kia-MM calibrated Gaussian NLLs for matrix-valued motifs.

    Every motif matrix is treated as a separate graph statistic, just as Kia's
    GraphVAE-MM treats each transition matrix P^1, ..., P^5 separately. For
    motif ``u``, one sigma_u is calibrated from the minibatch RMSE over every
    valid spatial entry. The Gaussian NLL is then averaged over those same
    entries. The default ``mean`` reduction averages the per-motif losses so
    the objective scale stays stable when a rule set contains many motifs.
    ``sum`` remains available to reproduce GraphVAE-MM's outer reduction.

    ``valid_mask`` excludes only the artificial bottom/right padding used to
    stack naturally shaped 1x1, 1xN, Nx1, and NxN motif results.

    Parameters
    ----------
    observed_matrices, predicted_matrices : torch.Tensor
        Shape ``(B, M, N_max, N_max)``.
    valid_mask : torch.Tensor
        Boolean-compatible tensor with shape ``(M, N_max, N_max)``.
    reduction : str
        ``"mean"`` averages over motif statistics, ``"sum"`` reproduces
        Kia's outer sum, and ``"none"`` returns the ``(M,)`` vector of
        independently calibrated losses so callers can weight motif groups.
    """
    if observed_matrices.shape != predicted_matrices.shape:
        raise ValueError(
            f"Shape mismatch: observed {tuple(observed_matrices.shape)} vs "
            f"predicted {tuple(predicted_matrices.shape)}"
        )
    if observed_matrices.ndim != 4:
        raise ValueError(
            "Expected motif matrices with shape (B, M, N_max, N_max), "
            f"got {tuple(observed_matrices.shape)}."
        )
    if tuple(valid_mask.shape) != tuple(observed_matrices.shape[1:]):
        raise ValueError(
            "Motif matrix mask shape must match (M, N_max, N_max): "
            f"mask={tuple(valid_mask.shape)}, "
            f"matrices={tuple(observed_matrices.shape)}."
        )
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(
            f"Unknown motif matrix loss reduction: {reduction}. "
            "Expected 'mean', 'sum', or 'none'."
        )

    try:
        return compute_calibrated_gaussian_motif_statistic_loss(
            observed_statistics=observed_matrices,
            predicted_statistics=predicted_matrices,
            valid_mask=valid_mask,
            min_log_sigma=min_log_sigma,
            eps=eps,
            reduction=reduction,
        )
    except ValueError as exc:
        # Preserve the established matrix-specific wording for callers/tests.
        if "Every motif statistic" in str(exc):
            raise ValueError(
                str(exc).replace("motif statistic", "motif matrix")
            ) from exc
        raise


def compute_motif_loss_asymmetric(observed_counts, predicted_counts, loss_mode="abs_log_ratio"):
    """
    Legacy asymmetric motif loss.

    Only motifs with nonzero observed count contribute to each graph's loss, so
    it penalizes missing existing motifs but does not penalize newly created
    motifs whose observed count is zero.
    """
    _validate_motif_count_shapes(observed_counts, predicted_counts)

    mask = observed_counts != 0
    if not mask.any():
        return torch.tensor(0.0, device=observed_counts.device)

    safe_observed = observed_counts.clamp(min=1e-8)
    safe_predicted = predicted_counts.clamp(min=1e-8)

    log_ratio = torch.log(safe_predicted / safe_observed)
    per_motif_loss = _apply_log_ratio_loss_mode(log_ratio, loss_mode)
    per_motif_loss = per_motif_loss * mask.to(per_motif_loss.dtype)

    active_motif_count = mask.sum(dim=1)
    valid_graph_mask = active_motif_count > 0
    if not valid_graph_mask.any():
        return torch.tensor(0.0, device=observed_counts.device)

    per_graph_loss = per_motif_loss.sum(dim=1)
    per_graph_loss = per_graph_loss[valid_graph_mask] / active_motif_count[valid_graph_mask].to(per_motif_loss.dtype)

    return per_graph_loss.mean()


compute_motif_loss_old = compute_motif_loss_asymmetric


def compute_motif_loss(
    observed_counts,
    predicted_counts,
    loss_mode="abs_log_ratio",
    laplace_pseudocount=1.0,
):
    """
    Compute a symmetric motif loss by averaging a Laplace-smoothed log-ratio
    penalty across motifs inside each graph, then averaging across graphs.

    The Laplace pseudocount keeps zero counts well-defined and makes motifs
    with observed count zero contribute to the loss, so extra motifs created in
    the reconstructed graph are penalized too.
    """
    _validate_motif_count_shapes(observed_counts, predicted_counts)

    if observed_counts.shape[-1] == 0:
        return torch.tensor(0.0, device=observed_counts.device)

    if loss_mode in CALIBRATED_GAUSSIAN_MOTIF_LOSS_MODES:
        return compute_calibrated_gaussian_motif_loss(
            observed_counts=observed_counts,
            predicted_counts=predicted_counts,
        )

    laplace_pseudocount = float(laplace_pseudocount)
    if laplace_pseudocount <= 0.0:
        raise ValueError("laplace_pseudocount must be > 0.")

    safe_observed = observed_counts + laplace_pseudocount
    safe_predicted = predicted_counts + laplace_pseudocount

    log_ratio = torch.log(safe_predicted / safe_observed)
    per_motif_loss = _apply_log_ratio_loss_mode(log_ratio, loss_mode)
    per_graph_loss = per_motif_loss.mean(dim=1)
    return per_graph_loss.mean()


def compute_masked_motif_loss(
    observed_counts,
    predicted_counts,
    motif_mask,
    loss_mode="abs_log_ratio",
    laplace_pseudocount=1.0,
):
    """
    Compute motif loss on only a selected subset of motif-vector columns.

    `motif_mask` is a 1D boolean mask over the motif dimension. If it selects
    no columns, the returned loss is zero.
    """
    _validate_motif_count_shapes(observed_counts, predicted_counts)

    if motif_mask is None:
        return compute_motif_loss(
            observed_counts=observed_counts,
            predicted_counts=predicted_counts,
            loss_mode=loss_mode,
            laplace_pseudocount=laplace_pseudocount,
        )

    if motif_mask.ndim != 1:
        raise ValueError(
            f"Expected 1D motif mask, got shape {tuple(motif_mask.shape)}."
        )
    if motif_mask.numel() != observed_counts.shape[1]:
        raise ValueError(
            f"Motif mask length {motif_mask.numel()} does not match motif dimension "
            f"{observed_counts.shape[1]}."
        )

    motif_mask = motif_mask.to(device=observed_counts.device, dtype=torch.bool)
    if not motif_mask.any():
        return torch.tensor(0.0, device=observed_counts.device)

    return compute_motif_loss(
        observed_counts=observed_counts[:, motif_mask],
        predicted_counts=predicted_counts[:, motif_mask],
        loss_mode=loss_mode,
        laplace_pseudocount=laplace_pseudocount,
    )


def compute_hard_motif_metrics(observed_counts, hard_predicted_counts):
    """
    Compute evaluation-only motif metrics on the discretized reconstruction.

    `hard_motif_loss` reuses the symmetric Laplace-smoothed absolute
    log-ratio penalty on hard motif counts so we compare both missing motifs
    and extra motifs in the thresholded graph against the target counts.
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


def summarize_single_graph_motif_counts(observed_counts, hard_predicted_counts):
    """
    Format the full observed/predicted motif-count vectors for a single graph.

    This is meant for tiny-overfit debugging when the remaining hard motif
    mismatch is small enough that inspecting the exact counts is more useful
    than another scalar summary.
    """
    if observed_counts.ndim != 2 or hard_predicted_counts.ndim != 2:
        raise ValueError("Expected batched motif counts with shape (B, M).")
    if observed_counts.shape != hard_predicted_counts.shape:
        raise ValueError(
            f"Shape mismatch: observed {tuple(observed_counts.shape)} vs "
            f"predicted {tuple(hard_predicted_counts.shape)}"
        )
    if observed_counts.shape[0] != 1:
        raise ValueError(
            "summarize_single_graph_motif_counts only supports a batch of size 1."
        )

    observed = observed_counts[0].detach().cpu().tolist()
    predicted = hard_predicted_counts[0].detach().cpu().tolist()
    delta = (hard_predicted_counts[0] - observed_counts[0]).detach().cpu().tolist()

    return [
        f"hard_motif_target[0]: {observed}",
        f"hard_motif_pred[0]: {predicted}",
        f"hard_motif_delta[0]: {delta}",
    ]
