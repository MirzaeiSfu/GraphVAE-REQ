import pytest
import torch

from motif_counting.motif_counter import RelationalMotifCounter


def make_counter() -> RelationalMotifCounter:
    counter = RelationalMotifCounter.__new__(RelationalMotifCounter)
    counter.device = "cpu"
    return counter


def test_matrix_result_retains_entries_and_matches_scalar_count():
    counter = make_counter()
    left = torch.tensor(
        [[[1.0, 2.0, 3.0]], [[2.0, 0.0, 1.0]]],
        requires_grad=True,
    )
    right = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[0.5, 1.0], [1.0, 0.0], [0.0, 2.0]],
        ],
        requires_grad=True,
    )

    raw_matrix = counter._compute_result_matrix_batched([left, right])
    padded_matrix, valid_mask = counter._pad_result_matrix_batched(
        raw_matrix,
        N_max=3,
    )
    scalar_count = counter._compute_result_batched([left, right])

    assert raw_matrix.shape == (2, 1, 2)
    assert padded_matrix.shape == (2, 3, 3)
    assert valid_mask.shape == (3, 3)
    assert valid_mask.sum().item() == 2
    torch.testing.assert_close(padded_matrix[:, :1, :2], raw_matrix)
    assert torch.count_nonzero(padded_matrix[:, 1:, :]).item() == 0

    retained_sum = (
        padded_matrix * valid_mask.unsqueeze(0).to(padded_matrix.dtype)
    ).flatten(start_dim=1).sum(dim=1)
    torch.testing.assert_close(retained_sum, scalar_count)

    retained_sum.sum().backward()
    assert left.grad is not None
    assert right.grad is not None
    assert torch.isfinite(left.grad).all()
    assert torch.isfinite(right.grad).all()
    assert left.grad.abs().sum().item() > 0
    assert right.grad.abs().sum().item() > 0


def test_count_batch_matrix_mode_stacks_graph_batches_and_returns_shared_mask():
    counter = make_counter()
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
        num_graphs = 3
        N_max = 2
        feature_onehot_mapping = {}

        def __init__(self):
            self.adjacency = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)

        def get_batch(self, start, end):
            batch = self.adjacency[start:end]
            size = end - start
            features = torch.zeros(size, 2, 1)
            onehot = torch.zeros(size, 2, 1)
            return features, onehot, {"edge": batch}, None

    preprocessor = FakePreprocessor()

    matrices, valid_mask = counter.count_batch(
        preprocessor,
        batch_size=2,
        output_mode="matrix",
    )

    assert matrices.shape == (3, 1, 2, 2)
    assert valid_mask.shape == (1, 2, 2)
    torch.testing.assert_close(matrices[:, 0], preprocessor.adjacency)
    assert valid_mask.all()

    scalar_counts = counter.count_batch(preprocessor, batch_size=2)
    expected_counts = preprocessor.adjacency.flatten(start_dim=1).sum(dim=1, keepdim=True)
    torch.testing.assert_close(scalar_counts, expected_counts)


def test_count_batch_rejects_unknown_output_mode():
    counter = make_counter()

    class EmptyPreprocessor:
        num_graphs = 1
        N_max = 1
        feature_onehot_mapping = {}

    with pytest.raises(ValueError, match="Unknown motif output mode"):
        counter.count_batch(EmptyPreprocessor(), output_mode="unknown")
