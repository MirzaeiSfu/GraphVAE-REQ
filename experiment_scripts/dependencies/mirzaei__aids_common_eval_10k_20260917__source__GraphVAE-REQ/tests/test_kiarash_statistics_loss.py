import math

import torch
import torch.nn.functional as F

from motif_counting.motif_loss_utils import (
    compute_calibrated_gaussian_kiarash_statistics_loss,
)
from motif_counting.motif_representations import (
    compute_kiarash_statistics_from_full_matrices,
)


def _legacy_optimizer_kernel_loss(observed_statistics, predicted_statistics):
    losses = []
    for observed, predicted in zip(observed_statistics, predicted_statistics):
        log_sigma = ((predicted - observed) ** 2).mean().sqrt().log()
        log_sigma = -6.0 + F.softplus(log_sigma + 6.0)
        losses.append(
            (
                0.5 * ((predicted - observed) / log_sigma.exp()).pow(2)
                + log_sigma
                + 0.5 * math.log(2.0 * math.pi)
            ).mean()
        )
    return torch.stack(losses)


def test_kiarash_statistics_loss_matches_optimizer_vae_outer_sum():
    observed_adjacency = torch.linspace(
        0.05,
        0.95,
        steps=3 * 4 * 4,
        dtype=torch.float64,
    ).reshape(3, 1, 4, 4)
    predicted_adjacency = (
        observed_adjacency.roll(shifts=1, dims=-1) * 0.8 + 0.07
    ).requires_grad_(True)
    matrix_mask = torch.ones(1, 4, 4, dtype=torch.bool)

    observed_statistics = compute_kiarash_statistics_from_full_matrices(
        observed_adjacency,
        matrix_mask,
    )
    predicted_statistics = compute_kiarash_statistics_from_full_matrices(
        predicted_adjacency,
        matrix_mask,
    )
    expected_per_statistic = _legacy_optimizer_kernel_loss(
        observed_statistics,
        predicted_statistics,
    )
    actual_per_statistic = (
        compute_calibrated_gaussian_kiarash_statistics_loss(
            observed_statistics,
            predicted_statistics,
            reduction="none",
        )
    )
    actual_sum = compute_calibrated_gaussian_kiarash_statistics_loss(
        observed_statistics,
        predicted_statistics,
    )

    torch.testing.assert_close(
        actual_per_statistic,
        expected_per_statistic,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        actual_sum,
        expected_per_statistic.sum(),
        rtol=1e-12,
        atol=1e-12,
    )
    actual_sum.backward()
    assert predicted_adjacency.grad is not None
    assert torch.isfinite(predicted_adjacency.grad).all()
    assert predicted_adjacency.grad.abs().sum().item() > 0.0
