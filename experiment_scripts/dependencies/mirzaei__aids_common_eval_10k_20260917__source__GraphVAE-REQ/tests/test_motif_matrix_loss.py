import math

import pytest
import torch
import torch.nn.functional as F

from motif_counting.motif_loss_utils import (
    compute_calibrated_gaussian_motif_matrix_loss,
)


def _kia_reference_per_motif(
    observed,
    predicted,
    valid_mask,
    min_log_sigma=-6.0,
    eps=1e-12,
):
    """Direct motif-by-motif version of Kia's OptimizerVAE loop."""
    losses = []
    for motif_idx in range(observed.shape[1]):
        mask = valid_mask[motif_idx]
        residual = (
            observed[:, motif_idx][:, mask]
            - predicted[:, motif_idx][:, mask]
        )
        rmse = residual.pow(2).mean().sqrt()
        log_sigma = torch.log(rmse.clamp_min(float(eps)))
        log_sigma = min_log_sigma + F.softplus(
            log_sigma - min_log_sigma
        )
        sigma = torch.exp(log_sigma)
        nll = (
            0.5 * (residual / sigma).pow(2)
            + log_sigma
            + 0.5 * math.log(2.0 * math.pi)
        )
        losses.append(nll.mean())
    return torch.stack(losses)


def test_matrix_loss_matches_kia_loop_for_all_natural_matrix_shapes():
    batch_size = 2
    num_motifs = 4
    n_max = 3
    observed = torch.linspace(
        0.0,
        4.0,
        steps=batch_size * num_motifs * n_max * n_max,
        dtype=torch.float64,
    ).reshape(batch_size, num_motifs, n_max, n_max)
    offsets = torch.tensor(
        [0.2, -0.5, 1.1, -1.7],
        dtype=torch.float64,
    ).view(1, num_motifs, 1, 1)
    predicted = observed + offsets

    valid_mask = torch.zeros(
        num_motifs,
        n_max,
        n_max,
        dtype=torch.bool,
    )
    valid_mask[0, :1, :1] = True  # 1x1
    valid_mask[1, :1, :] = True   # 1xN
    valid_mask[2, :, :1] = True   # Nx1
    valid_mask[3, :, :] = True    # NxN

    expected = _kia_reference_per_motif(
        observed,
        predicted,
        valid_mask,
    )
    actual = compute_calibrated_gaussian_motif_matrix_loss(
        observed_matrices=observed,
        predicted_matrices=predicted,
        valid_mask=valid_mask,
        reduction="none",
    )

    torch.testing.assert_close(actual, expected)
    mean_loss = compute_calibrated_gaussian_motif_matrix_loss(
        observed_matrices=observed,
        predicted_matrices=predicted,
        valid_mask=valid_mask,
    )
    torch.testing.assert_close(mean_loss, expected.mean())
    summed_loss = compute_calibrated_gaussian_motif_matrix_loss(
        observed_matrices=observed,
        predicted_matrices=predicted,
        valid_mask=valid_mask,
        reduction="sum",
    )
    torch.testing.assert_close(summed_loss, expected.sum())


def test_matrix_loss_ignores_padding_and_backpropagates_only_valid_entries():
    observed = torch.zeros(2, 2, 3, 3, dtype=torch.float64)
    valid_mask = torch.zeros(2, 3, 3, dtype=torch.bool)
    valid_mask[0, :1, :2] = True
    valid_mask[1, :, :1] = True

    base_prediction = torch.tensor(
        [
            [
                [[0.5, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                [[0.2, 9.0, 8.0], [0.4, 7.0, 6.0], [0.8, 5.0, 4.0]],
            ],
            [
                [[1.5, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[0.3, 8.0, 7.0], [0.6, 6.0, 5.0], [1.2, 4.0, 3.0]],
            ],
        ],
        dtype=torch.float64,
    )
    changed_padding = base_prediction.clone()
    expanded_mask = valid_mask.unsqueeze(0).expand_as(changed_padding)
    changed_padding[~expanded_mask] = 1_000_000.0

    base_loss = compute_calibrated_gaussian_motif_matrix_loss(
        observed,
        base_prediction,
        valid_mask,
    )
    padded_loss = compute_calibrated_gaussian_motif_matrix_loss(
        observed,
        changed_padding,
        valid_mask,
    )
    torch.testing.assert_close(base_loss, padded_loss)

    predicted = changed_padding.clone().requires_grad_(True)
    loss = compute_calibrated_gaussian_motif_matrix_loss(
        observed,
        predicted,
        valid_mask,
    )
    loss.backward()

    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[expanded_mask].abs().sum().item() > 0.0
    assert torch.count_nonzero(predicted.grad[~expanded_mask]).item() == 0


def test_matrix_loss_is_finite_for_exact_matches():
    observed = torch.zeros(2, 2, 2, 2)
    predicted = observed.clone().requires_grad_(True)
    valid_mask = torch.ones(2, 2, 2, dtype=torch.bool)

    per_motif_loss = compute_calibrated_gaussian_motif_matrix_loss(
        observed,
        predicted,
        valid_mask,
        reduction="none",
    )

    assert per_motif_loss.shape == (2,)
    assert torch.isfinite(per_motif_loss).all()
    per_motif_loss.sum().backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_matrix_loss_validates_shapes_masks_and_reduction():
    observed = torch.zeros(2, 2, 3, 3)
    predicted = torch.zeros_like(observed)
    valid_mask = torch.ones(2, 3, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_calibrated_gaussian_motif_matrix_loss(
            observed,
            predicted[:, :1],
            valid_mask,
        )
    with pytest.raises(ValueError, match="mask shape"):
        compute_calibrated_gaussian_motif_matrix_loss(
            observed,
            predicted,
            valid_mask[:1],
        )
    with pytest.raises(ValueError, match="empty masks"):
        invalid_mask = valid_mask.clone()
        invalid_mask[1] = False
        compute_calibrated_gaussian_motif_matrix_loss(
            observed,
            predicted,
            invalid_mask,
        )
    with pytest.raises(ValueError, match="reduction"):
        compute_calibrated_gaussian_motif_matrix_loss(
            observed,
            predicted,
            valid_mask,
            reduction="invalid",
        )


def test_matrix_loss_handles_an_empty_motif_dimension():
    observed = torch.zeros(2, 0, 3, 3)
    predicted = observed.clone().requires_grad_(True)
    valid_mask = torch.zeros(0, 3, 3, dtype=torch.bool)

    per_motif_loss = compute_calibrated_gaussian_motif_matrix_loss(
        observed,
        predicted,
        valid_mask,
        reduction="none",
    )
    total_loss = compute_calibrated_gaussian_motif_matrix_loss(
        observed,
        predicted,
        valid_mask,
    )

    assert per_motif_loss.shape == (0,)
    assert total_loss.item() == 0.0
