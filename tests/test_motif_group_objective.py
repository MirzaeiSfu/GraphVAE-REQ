import pytest
import torch

from motif_counting.motif_loss_utils import (
    compute_calibrated_gaussian_motif_channel_loss,
    compute_calibrated_gaussian_motif_statistic_loss,
)
from motif_counting.motif_objective import (
    NON_LITERAL_MOTIF_GROUP,
    SYNTACTIC_LITERAL_MOTIF_GROUP,
    build_motif_group_objectives,
    calibrate_group_histogram_specs,
    compute_grouped_motif_loss,
)
from motif_counting.motif_representations import represent_full_motif_matrices


def make_full_matrices():
    observed = torch.tensor(
        [
            [
                [[0.0, 1.0], [2.0, 0.0]],
                [[1.0, 0.0], [0.0, 3.0]],
                [[0.0, 2.0], [1.0, 0.0]],
                [[2.0, 1.0], [0.0, 1.0]],
            ],
            [
                [[0.0, 2.0], [1.0, 0.0]],
                [[2.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [3.0, 0.0]],
                [[1.0, 2.0], [1.0, 0.0]],
            ],
        ]
    )
    predicted = (observed + 0.25).requires_grad_(True)
    matrix_mask = torch.ones(4, 2, 2, dtype=torch.bool)
    literal_mask = torch.tensor([False, False, True, True])
    return observed, predicted, matrix_mask, literal_mask


def test_group_specific_full_matrix_and_marginal_losses_share_one_count():
    observed, predicted, matrix_mask, literal_mask = make_full_matrices()
    groups = build_motif_group_objectives(
        syntactic_literal_mask=literal_mask,
        non_literal_output_mode="full_matrix",
        non_literal_loss_mode="calibrated_gaussian",
        non_literal_weight=2.0,
        syntactic_literal_output_mode="row_column_marginals",
        syntactic_literal_loss_mode="calibrated_gaussian",
        syntactic_literal_weight=3.0,
    )

    result = compute_grouped_motif_loss(
        observed_full_matrices=observed,
        predicted_full_matrices=predicted,
        full_matrix_mask=matrix_mask,
        groups=groups,
    )

    expected_non_literal = compute_calibrated_gaussian_motif_statistic_loss(
        observed[:, :2],
        predicted[:, :2],
        matrix_mask[:2],
    )
    observed_literal, literal_statistic_mask, _ = represent_full_motif_matrices(
        observed[:, 2:],
        matrix_mask[2:],
        output_mode="row_column_marginals",
    )
    predicted_literal, _, _ = represent_full_motif_matrices(
        predicted[:, 2:],
        matrix_mask[2:],
        output_mode="row_column_marginals",
    )
    expected_literal = compute_calibrated_gaussian_motif_channel_loss(
        observed_literal,
        predicted_literal,
        literal_statistic_mask,
    )

    torch.testing.assert_close(
        result.group_losses[NON_LITERAL_MOTIF_GROUP],
        expected_non_literal,
    )
    torch.testing.assert_close(
        result.group_losses[SYNTACTIC_LITERAL_MOTIF_GROUP],
        expected_literal,
    )
    torch.testing.assert_close(
        result.loss,
        expected_non_literal + expected_literal,
    )
    torch.testing.assert_close(
        result.weighted_loss,
        2.0 * expected_non_literal + 3.0 * expected_literal,
    )

    result.weighted_loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.abs().sum().item() > 0.0


def test_group_specific_histogram_is_calibrated_only_from_observed_group():
    observed, predicted, matrix_mask, literal_mask = make_full_matrices()
    groups = build_motif_group_objectives(
        syntactic_literal_mask=literal_mask,
        non_literal_output_mode="total_count",
        non_literal_loss_mode="squared_log_ratio",
        non_literal_weight=1.0,
        syntactic_literal_output_mode="marginal_histogram",
        syntactic_literal_loss_mode="calibrated_gaussian",
        syntactic_literal_weight=0.5,
    )
    groups = calibrate_group_histogram_specs(
        observed_full_matrices=observed,
        full_matrix_mask=matrix_mask,
        groups=groups,
        histogram_num_bins=4,
    )

    assert groups[0].name == NON_LITERAL_MOTIF_GROUP
    assert groups[0].histogram_spec is None
    assert groups[1].name == SYNTACTIC_LITERAL_MOTIF_GROUP
    assert groups[1].histogram_spec is not None
    assert groups[1].histogram_spec["bin_edges"].shape == (2, 3)

    result = compute_grouped_motif_loss(
        observed_full_matrices=observed,
        predicted_full_matrices=predicted,
        full_matrix_mask=matrix_mask,
        groups=groups,
        histogram_num_bins=4,
    )
    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.weighted_loss)


def test_structured_group_rejects_non_gaussian_loss():
    with pytest.raises(ValueError, match="requires motif_loss_mode=calibrated_gaussian"):
        build_motif_group_objectives(
            syntactic_literal_mask=torch.tensor([False, True]),
            non_literal_output_mode="full_matrix",
            non_literal_loss_mode="squared_log_ratio",
            non_literal_weight=1.0,
            syntactic_literal_output_mode="row_column_marginals",
            syntactic_literal_loss_mode="calibrated_gaussian",
            syntactic_literal_weight=1.0,
        )
