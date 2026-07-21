import pytest
import torch

from motif_counting.motif_counter import RelationalMotifCounter
from motif_counting.motif_loss_utils import (
    compute_calibrated_gaussian_motif_channel_loss,
    compute_calibrated_gaussian_motif_statistic_loss,
)
from motif_counting.motif_representations import (
    build_marginal_histogram_spec,
    canonicalize_motif_output_mode,
    compute_degree_histograms_from_full_matrices,
    compute_marginal_histograms,
    compute_row_column_marginals,
    compute_total_motif_count,
)


def test_output_mode_aliases_have_clear_canonical_names():
    assert canonicalize_motif_output_mode("matrix") == "full_matrix"
    assert canonicalize_motif_output_mode("count") == "total_count"
    assert canonicalize_motif_output_mode("row_column_marginals") == (
        "row_column_marginals"
    )
    assert canonicalize_motif_output_mode("marginal_histogram") == (
        "marginal_histogram"
    )
    assert canonicalize_motif_output_mode("degree_histogram") == (
        "degree_histogram"
    )


def test_degree_histogram_matches_graphvae_mm_triangular_memberships():
    matrices = torch.tensor(
        [[[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]],
        requires_grad=True,
    )
    matrix_mask = torch.ones(1, 3, 3, dtype=torch.bool)

    histograms, histogram_mask = compute_degree_histograms_from_full_matrices(
        matrices,
        matrix_mask,
    )

    degrees = torch.tensor([1.0, 2.0, 1.0])
    centers = torch.arange(3, dtype=matrices.dtype)
    expected = torch.relu(
        1.0 - torch.abs(degrees.unsqueeze(1) - centers.unsqueeze(0)) * 0.1
    ).sum(dim=0)
    assert histograms.shape == (1, 1, 3)
    torch.testing.assert_close(histograms[0, 0], expected)
    assert histogram_mask.all()

    histograms.sum().backward()
    assert matrices.grad is not None
    assert torch.isfinite(matrices.grad).all()
    assert matrices.grad.abs().sum().item() > 0.0


def test_degree_histogram_rejects_non_square_natural_motif_results():
    matrices = torch.zeros(1, 1, 3, 3)
    vector_mask = torch.zeros(1, 3, 3, dtype=torch.bool)
    vector_mask[0, :, :1] = True

    with pytest.raises(ValueError, match="requires natural N_max x N_max"):
        compute_degree_histograms_from_full_matrices(matrices, vector_mask)


def test_square_result_retains_both_row_and_column_marginals():
    result = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)

    marginals, valid_mask = compute_row_column_marginals(result, n_max=3)

    assert marginals.shape == (1, 2, 3)
    torch.testing.assert_close(marginals[0, 0], torch.tensor([3.0, 7.0, 0.0]))
    torch.testing.assert_close(marginals[0, 1], torch.tensor([4.0, 6.0, 0.0]))
    torch.testing.assert_close(
        valid_mask,
        torch.tensor([[True, True, False], [True, True, False]]),
    )
    marginals.sum().backward()
    torch.testing.assert_close(result.grad, torch.full_like(result, 2.0))


def test_n_by_one_result_preserves_only_its_node_sized_row_marginal():
    result = torch.tensor([[[1.0], [2.0], [3.0]]])

    marginals, valid_mask = compute_row_column_marginals(result, n_max=3)

    torch.testing.assert_close(marginals[0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.count_nonzero(marginals[0, 1]).item() == 0
    assert valid_mask[0].all()
    assert not valid_mask[1].any()
    torch.testing.assert_close(compute_total_motif_count(result), torch.tensor([6.0]))


def test_one_by_n_result_preserves_only_its_node_sized_column_marginal():
    result = torch.tensor([[[1.0, 2.0, 3.0]]])

    marginals, valid_mask = compute_row_column_marginals(result, n_max=3)

    assert torch.count_nonzero(marginals[0, 0]).item() == 0
    torch.testing.assert_close(marginals[0, 1], torch.tensor([1.0, 2.0, 3.0]))
    assert not valid_mask[0].any()
    assert valid_mask[1].all()
    torch.testing.assert_close(compute_total_motif_count(result), torch.tensor([6.0]))


def test_one_by_one_result_is_not_duplicated_across_marginal_channels():
    result = torch.tensor([[[5.0]]])

    marginals, valid_mask = compute_row_column_marginals(result, n_max=3)

    torch.testing.assert_close(marginals[0, 0], torch.tensor([5.0, 0.0, 0.0]))
    assert torch.count_nonzero(marginals[0, 1]).item() == 0
    assert valid_mask.sum().item() == 1
    assert valid_mask[0, 0]


def test_soft_marginal_histogram_conserves_each_valid_vector_and_overflow():
    marginals = torch.tensor(
        [
            [[[0.0, 1.0, 3.0], [0.0, 0.0, 0.0]]],
            [[[1.0, 2.0, 8.0], [0.0, 0.0, 0.0]]],
        ],
        requires_grad=True,
    )
    valid_mask = torch.tensor([[[True, True, True], [False, False, False]]])
    spec = build_marginal_histogram_spec(
        marginals.detach(),
        valid_mask,
        num_bins=4,
        smoothing=0.25,
    )

    histograms, histogram_mask = compute_marginal_histograms(
        marginals,
        valid_mask,
        spec,
    )

    assert histograms.shape == (2, 1, 2, 4)
    torch.testing.assert_close(histograms[:, 0, 0].sum(dim=-1), torch.tensor([3.0, 3.0]))
    assert torch.count_nonzero(histograms[:, 0, 1]).item() == 0
    assert histogram_mask[0, 0].all()
    assert not histogram_mask[0, 1].any()

    overflow = marginals.detach().clone()
    overflow[:, :, 0] = 1_000_000.0
    overflow_histograms, _ = compute_marginal_histograms(
        overflow,
        valid_mask,
        spec,
    )
    torch.testing.assert_close(
        overflow_histograms[:, 0, 0].sum(dim=-1),
        torch.tensor([3.0, 3.0]),
    )
    assert (overflow_histograms[:, 0, 0, -1] > 2.99).all()

    histograms[..., 0].sum().backward()
    assert marginals.grad is not None
    assert torch.isfinite(marginals.grad).all()
    assert marginals.grad[:, :, 0].abs().sum().item() > 0.0
    assert torch.count_nonzero(marginals.grad[:, :, 1]).item() == 0


def test_generic_statistic_loss_masks_padding_for_marginals():
    observed = torch.zeros(2, 1, 2, 3)
    predicted = torch.tensor(
        [
            [[[0.5, 1.0, 1000.0], [1000.0, 1000.0, 1000.0]]],
            [[[1.0, 2.0, 1000.0], [1000.0, 1000.0, 1000.0]]],
        ],
        requires_grad=True,
    )
    valid_mask = torch.tensor([[[True, True, False], [False, False, False]]])

    loss = compute_calibrated_gaussian_motif_statistic_loss(
        observed,
        predicted,
        valid_mask,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert predicted.grad[:, :, 0, :2].abs().sum().item() > 0.0
    assert torch.count_nonzero(predicted.grad[:, :, 0, 2:]).item() == 0
    assert torch.count_nonzero(predicted.grad[:, :, 1]).item() == 0


def test_channel_loss_calibrates_row_and_column_directions_separately():
    observed = torch.zeros(2, 2, 2, 2)
    predicted = torch.tensor(
        [
            [[[1.0, 1.0], [3.0, 3.0]], [[2.0, 2.0], [1000.0, 1000.0]]],
            [[[1.5, 1.5], [4.0, 4.0]], [[2.5, 2.5], [1000.0, 1000.0]]],
        ],
        requires_grad=True,
    )
    valid_mask = torch.tensor(
        [
            [[True, True], [True, True]],
            [[True, True], [False, False]],
        ]
    )

    per_motif_loss = compute_calibrated_gaussian_motif_channel_loss(
        observed,
        predicted,
        valid_mask,
        reduction="none",
    )
    motif_0_row_loss = compute_calibrated_gaussian_motif_statistic_loss(
        observed[:, 0:1, 0],
        predicted[:, 0:1, 0],
        valid_mask[0:1, 0],
    )
    motif_0_column_loss = compute_calibrated_gaussian_motif_statistic_loss(
        observed[:, 0:1, 1],
        predicted[:, 0:1, 1],
        valid_mask[0:1, 1],
    )
    motif_1_row_loss = compute_calibrated_gaussian_motif_statistic_loss(
        observed[:, 1:2, 0],
        predicted[:, 1:2, 0],
        valid_mask[1:2, 0],
    )

    torch.testing.assert_close(
        per_motif_loss[0],
        (motif_0_row_loss + motif_0_column_loss) / 2.0,
    )
    torch.testing.assert_close(per_motif_loss[1], motif_1_row_loss)

    per_motif_loss.sum().backward()
    assert predicted.grad is not None
    assert predicted.grad[:, 0, 0].abs().sum().item() > 0.0
    assert predicted.grad[:, 0, 1].abs().sum().item() > 0.0
    assert predicted.grad[:, 1, 0].abs().sum().item() > 0.0
    assert torch.count_nonzero(predicted.grad[:, 1, 1]).item() == 0


def test_counter_integrates_all_five_canonical_modes():
    counter = RelationalMotifCounter.__new__(RelationalMotifCounter)
    counter.device = "cpu"
    counter.rules = [["edge(X,Y)"]]
    counter.values = [[["T"]]]
    counter.multiples = [0]
    counter.functors = [["edge"]]
    counter.states = [[2]]
    counter.variables = [[None]]
    counter.masks = [[None]]
    counter.base_indices = [[0]]
    counter.mask_indices = [[]]
    counter.sort_indices = [[(False, 0)]]
    counter.stack_indices = [[]]

    class FakePreprocessor:
        num_graphs = 2
        N_max = 2
        feature_onehot_mapping = {}

        def __init__(self):
            self.adjacency = torch.tensor(
                [[[0.0, 1.0], [2.0, 0.0]], [[0.0, 3.0], [4.0, 0.0]]]
            )

        def get_batch(self, start, end):
            batch = self.adjacency[start:end]
            size = end - start
            features = torch.zeros(size, 2, 1)
            onehot = torch.zeros(size, 2, 1)
            return features, onehot, {"edge": batch}, None

    preprocessor = FakePreprocessor()
    full_matrix, full_mask = counter.count_batch(
        preprocessor,
        output_mode="full_matrix",
    )
    marginals, marginal_mask = counter.count_batch(
        preprocessor,
        output_mode="row_column_marginals",
    )
    histograms, histogram_mask, histogram_spec = counter.count_batch(
        preprocessor,
        output_mode="marginal_histogram",
        histogram_num_bins=4,
    )
    repeated_histograms, repeated_mask, _ = counter.count_batch(
        preprocessor,
        output_mode="marginal_histogram",
        histogram_spec=histogram_spec,
    )
    degree_histograms, degree_histogram_mask = counter.count_batch(
        preprocessor,
        output_mode="degree_histogram",
    )
    total_count = counter.count_batch(preprocessor, output_mode="total_count")

    assert full_matrix.shape == (2, 1, 2, 2)
    assert full_mask.all()
    torch.testing.assert_close(marginals[:, 0, 0], preprocessor.adjacency.sum(dim=2))
    torch.testing.assert_close(marginals[:, 0, 1], preprocessor.adjacency.sum(dim=1))
    assert marginal_mask.all()
    assert histograms.shape == (2, 1, 2, 4)
    assert histogram_mask.all()
    torch.testing.assert_close(histograms, repeated_histograms)
    torch.testing.assert_close(histogram_mask, repeated_mask)
    assert degree_histograms.shape == (2, 1, 2)
    assert degree_histogram_mask.all()
    torch.testing.assert_close(
        total_count,
        preprocessor.adjacency.flatten(start_dim=1).sum(dim=1, keepdim=True),
    )
