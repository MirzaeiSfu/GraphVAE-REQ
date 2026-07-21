"""Differentiable representations of a motif matrix-chain result.

Every motif rule ends in a natural ``H x W`` tensor where each spatial
dimension is either ``1`` or ``N_max``.  This module gives those results
explicit, consistently named reductions for use as graph statistics.
"""

from typing import Dict

import torch
import torch.nn.functional as F


MOTIF_OUTPUT_MODE_ALIASES = {
    "count": "total_count",
    "matrix": "full_matrix",
}
MOTIF_OUTPUT_MODES = {
    "full_matrix",
    "row_column_marginals",
    "marginal_histogram",
    "degree_histogram",
    "total_count",
}
MOTIF_OUTPUT_MODE_CHOICES = MOTIF_OUTPUT_MODES | set(MOTIF_OUTPUT_MODE_ALIASES)
MOTIF_MARGINAL_CHANNELS = ("row", "column")
MASKED_MOTIF_OUTPUT_MODES = MOTIF_OUTPUT_MODES - {"total_count"}


def canonicalize_motif_output_mode(output_mode: str) -> str:
    """Return the canonical output-mode name, accepting legacy aliases."""
    canonical_mode = MOTIF_OUTPUT_MODE_ALIASES.get(output_mode, output_mode)
    if canonical_mode not in MOTIF_OUTPUT_MODES:
        raise ValueError(
            f"Unknown motif output mode: {output_mode}. "
            f"Expected one of {sorted(MOTIF_OUTPUT_MODE_CHOICES)}."
        )
    return canonical_mode


def _validate_natural_result(result: torch.Tensor, n_max: int) -> None:
    if result.ndim != 3:
        raise ValueError(
            "Expected a batched motif result with shape (B, H, W), "
            f"got {tuple(result.shape)}."
        )
    height, width = result.shape[1:]
    if height < 1 or width < 1 or height > n_max or width > n_max:
        raise ValueError(
            "Motif result has an invalid natural shape: "
            f"result={height}x{width}, N_max={n_max}."
        )


def pad_full_motif_matrix(
    result: torch.Tensor,
    n_max: int,
):
    """Pad a natural motif result to ``N_max x N_max`` and return its mask."""
    _validate_natural_result(result, n_max)
    height, width = result.shape[1:]
    padded = F.pad(result, (0, n_max - width, 0, n_max - height))
    valid_mask = torch.zeros(
        n_max,
        n_max,
        dtype=torch.bool,
        device=result.device,
    )
    valid_mask[:height, :width] = True
    return padded, valid_mask


def compute_row_column_marginals(
    result: torch.Tensor,
    n_max: int,
):
    """Return meaningful row/column marginals without collapsing vectors.

    The output has shape ``(B, 2, N_max)`` with channels ``(row, column)``:

    * ``N x N``: retain both ``sum(columns)`` and ``sum(rows)``;
    * ``N x 1``: retain only the length-``N`` row marginal;
    * ``1 x N``: retain only the length-``N`` column marginal;
    * ``1 x 1``: retain one scalar in the row channel.

    Omitting the redundant scalar channel for vector/scalar results prevents a
    single motif total from being counted twice in the loss.
    """
    padded, matrix_mask = pad_full_motif_matrix(result, n_max)
    marginals, marginal_mask = compute_row_column_marginals_from_full_matrices(
        padded.unsqueeze(1),
        matrix_mask.unsqueeze(0),
    )
    return marginals[:, 0], marginal_mask[0]


def compute_total_motif_count(result: torch.Tensor) -> torch.Tensor:
    """Sum every spatial entry, yielding one scalar per graph."""
    if result.ndim != 3:
        raise ValueError(
            "Expected a batched motif result with shape (B, H, W), "
            f"got {tuple(result.shape)}."
        )
    return result.flatten(start_dim=1).sum(dim=1)


def _validate_full_motif_matrices(
    full_matrices: torch.Tensor,
    matrix_mask: torch.Tensor,
) -> None:
    if full_matrices.ndim != 4:
        raise ValueError(
            "Expected full motif matrices with shape (B, M, N_max, N_max), "
            f"got {tuple(full_matrices.shape)}."
        )
    if full_matrices.shape[-1] != full_matrices.shape[-2]:
        raise ValueError(
            "Full motif matrices must be square after padding, "
            f"got {tuple(full_matrices.shape[-2:])}."
        )
    if tuple(matrix_mask.shape) != tuple(full_matrices.shape[1:]):
        raise ValueError(
            "Full motif matrix mask must match (M, N_max, N_max): "
            f"mask={tuple(matrix_mask.shape)}, "
            f"matrices={tuple(full_matrices.shape)}."
        )


def compute_row_column_marginals_from_full_matrices(
    full_matrices: torch.Tensor,
    matrix_mask: torch.Tensor,
):
    """Derive shape-aware marginals from canonical padded motif matrices.

    Both directions are retained for a genuine matrix. For a natural ``N x 1``
    result only the row channel is valid; for ``1 x N`` only the column channel
    is valid; and for ``1 x 1`` only one scalar channel is valid.
    """
    _validate_full_motif_matrices(full_matrices, matrix_mask)
    matrix_mask = matrix_mask.to(device=full_matrices.device, dtype=torch.bool)
    masked_matrices = torch.where(
        matrix_mask.unsqueeze(0),
        full_matrices,
        torch.zeros((), dtype=full_matrices.dtype, device=full_matrices.device),
    )

    row_marginals = masked_matrices.sum(dim=3)
    column_marginals = masked_matrices.sum(dim=2)
    valid_rows = matrix_mask.any(dim=2)
    valid_columns = matrix_mask.any(dim=1)
    natural_heights = valid_rows.sum(dim=1)
    natural_widths = valid_columns.sum(dim=1)

    # A row marginal is node-sized for Nx1/NxN, and is also the sole scalar
    # channel for 1x1. A column marginal is node-sized for 1xN/NxN.
    row_channel_enabled = (natural_heights > 1) | (natural_widths == 1)
    column_channel_enabled = natural_widths > 1
    row_mask = valid_rows & row_channel_enabled.unsqueeze(1)
    column_mask = valid_columns & column_channel_enabled.unsqueeze(1)

    marginals = torch.stack((row_marginals, column_marginals), dim=2)
    marginal_mask = torch.stack((row_mask, column_mask), dim=1)
    marginals = torch.where(
        marginal_mask.unsqueeze(0),
        marginals,
        torch.zeros((), dtype=marginals.dtype, device=marginals.device),
    )
    return marginals, marginal_mask


def compute_degree_histograms_from_full_matrices(
    full_matrices: torch.Tensor,
    matrix_mask: torch.Tensor,
    histogram_width: float = 0.1,
):
    """Apply GraphVAE-MM's soft degree histogram to square motif matrices.

    Row sums of a selected ``N_max x N_max`` motif matrix become node degrees.
    The histogram has integer centers ``0, ..., N_max - 1`` and triangular
    memberships ``relu(1 - width * abs(degree - center))``, matching the
    repository's :class:`GlobalProperties.Histogram` degree statistic.  This
    representation is intended for a protected unit binary-relation motif,
    whose full count matrix is the corresponding adjacency matrix.
    """
    _validate_full_motif_matrices(full_matrices, matrix_mask)
    if histogram_width <= 0.0:
        raise ValueError("Degree histogram width must be greater than zero.")

    matrix_mask = matrix_mask.to(device=full_matrices.device, dtype=torch.bool)
    natural_heights = matrix_mask.any(dim=2).sum(dim=1)
    natural_widths = matrix_mask.any(dim=1).sum(dim=1)
    n_max = full_matrices.shape[-1]
    invalid_motifs = (natural_heights != n_max) | (natural_widths != n_max)
    if invalid_motifs.any():
        invalid_indices = torch.nonzero(
            invalid_motifs,
            as_tuple=False,
        ).flatten().detach().cpu().tolist()
        raise ValueError(
            "degree_histogram requires natural N_max x N_max motif matrices; "
            f"invalid motif indices {invalid_indices}."
        )

    masked_matrices = torch.where(
        matrix_mask.unsqueeze(0),
        full_matrices,
        torch.zeros((), dtype=full_matrices.dtype, device=full_matrices.device),
    )
    degrees = masked_matrices.sum(dim=3)
    bin_centers = torch.arange(
        n_max,
        device=full_matrices.device,
        dtype=full_matrices.dtype,
    )
    memberships = torch.relu(
        1.0
        - torch.abs(degrees.unsqueeze(-1) - bin_centers.view(1, 1, 1, n_max))
        * float(histogram_width)
    )
    histograms = memberships.sum(dim=2)
    histogram_mask = torch.ones(
        full_matrices.shape[1],
        n_max,
        dtype=torch.bool,
        device=full_matrices.device,
    )
    return histograms, histogram_mask


def represent_full_motif_matrices(
    full_matrices: torch.Tensor,
    matrix_mask: torch.Tensor,
    output_mode: str,
    histogram_num_bins: int = 16,
    histogram_smoothing: float = 0.25,
    histogram_spec: Dict[str, torch.Tensor] = None,
):
    """Derive a requested statistic from canonical full motif matrices.

    Returns ``(values, valid_mask, histogram_spec)`` for every mode. Scalar
    ``total_count`` values have no validity mask. Histogram specifications are
    calibrated only when none is supplied and must then be reused for decoded
    graphs.
    """
    output_mode = canonicalize_motif_output_mode(output_mode)
    _validate_full_motif_matrices(full_matrices, matrix_mask)
    matrix_mask = matrix_mask.to(device=full_matrices.device, dtype=torch.bool)

    if output_mode == "full_matrix":
        return full_matrices, matrix_mask, None

    if output_mode == "total_count":
        masked_matrices = torch.where(
            matrix_mask.unsqueeze(0),
            full_matrices,
            torch.zeros(
                (),
                dtype=full_matrices.dtype,
                device=full_matrices.device,
            ),
        )
        return masked_matrices.sum(dim=(2, 3)), None, None

    if output_mode == "degree_histogram":
        histograms, histogram_mask = compute_degree_histograms_from_full_matrices(
            full_matrices=full_matrices,
            matrix_mask=matrix_mask,
        )
        return histograms, histogram_mask, None

    marginals, marginal_mask = compute_row_column_marginals_from_full_matrices(
        full_matrices,
        matrix_mask,
    )
    if output_mode == "row_column_marginals":
        return marginals, marginal_mask, None

    if histogram_spec is None:
        histogram_spec = build_marginal_histogram_spec(
            observed_marginals=marginals,
            valid_mask=marginal_mask,
            num_bins=histogram_num_bins,
            smoothing=histogram_smoothing,
        )
    histograms, histogram_mask = compute_marginal_histograms(
        marginals=marginals,
        valid_mask=marginal_mask,
        histogram_spec=histogram_spec,
    )
    return histograms, histogram_mask, histogram_spec


def build_marginal_histogram_spec(
    observed_marginals: torch.Tensor,
    valid_mask: torch.Tensor,
    num_bins: int = 16,
    smoothing: float = 0.25,
    min_temperature: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """Calibrate fixed soft-histogram bins from observed marginal targets.

    Histograms operate on ``log1p(marginal)`` so one binning scheme remains
    useful for motif counts with a wide dynamic range.  Internal bin edges are
    spaced uniformly from zero to each motif's observed maximum.  The first
    and last bins include underflow/overflow, so reconstructed values never
    disappear merely because they exceed the observed range.
    """
    if observed_marginals.ndim != 4:
        raise ValueError(
            "Expected marginals with shape (B, M, 2, N_max), "
            f"got {tuple(observed_marginals.shape)}."
        )
    if tuple(valid_mask.shape) != tuple(observed_marginals.shape[1:]):
        raise ValueError(
            "Marginal mask shape must match (M, 2, N_max): "
            f"mask={tuple(valid_mask.shape)}, "
            f"marginals={tuple(observed_marginals.shape)}."
        )
    if num_bins < 2:
        raise ValueError("A marginal histogram requires at least two bins.")
    if smoothing <= 0.0:
        raise ValueError("Histogram smoothing must be greater than zero.")

    num_motifs = observed_marginals.shape[1]
    if num_motifs == 0:
        return {
            "bin_edges": observed_marginals.new_empty((0, num_bins - 1)),
            "temperature": observed_marginals.new_empty((0,)),
        }

    valid_mask = valid_mask.to(
        device=observed_marginals.device,
        dtype=torch.bool,
    )
    log_values = torch.log1p(observed_marginals.clamp_min(0.0))
    masked_log_values = torch.where(
        valid_mask.unsqueeze(0),
        log_values,
        torch.full(
            (),
            float("-inf"),
            device=log_values.device,
            dtype=log_values.dtype,
        ),
    )
    maximum_by_motif = (
        masked_log_values.permute(1, 0, 2, 3)
        .reshape(num_motifs, -1)
        .max(dim=1)[0]
    )
    if not torch.isfinite(maximum_by_motif).all():
        raise ValueError("Every motif must have at least one valid marginal entry.")

    # Give all-zero motifs a useful [0, log(2)] range while retaining the
    # observed maximum for nonzero motifs.
    minimum_span = torch.log(
        observed_marginals.new_tensor(2.0)
    )
    span_by_motif = maximum_by_motif.clamp_min(minimum_span)
    edge_fractions = torch.arange(
        1,
        num_bins,
        device=observed_marginals.device,
        dtype=observed_marginals.dtype,
    ) / float(num_bins)
    bin_edges = span_by_motif.unsqueeze(1) * edge_fractions.unsqueeze(0)
    temperature = (
        span_by_motif / float(num_bins) * float(smoothing)
    ).clamp_min(float(min_temperature))
    return {
        "bin_edges": bin_edges.detach(),
        "temperature": temperature.detach(),
    }


def compute_marginal_histograms(
    marginals: torch.Tensor,
    valid_mask: torch.Tensor,
    histogram_spec: Dict[str, torch.Tensor],
):
    """Convert row/column marginals into differentiable soft histograms."""
    if marginals.ndim != 4:
        raise ValueError(
            "Expected marginals with shape (B, M, 2, N_max), "
            f"got {tuple(marginals.shape)}."
        )
    if tuple(valid_mask.shape) != tuple(marginals.shape[1:]):
        raise ValueError(
            "Marginal mask shape must match (M, 2, N_max): "
            f"mask={tuple(valid_mask.shape)}, marginals={tuple(marginals.shape)}."
        )
    if not isinstance(histogram_spec, dict):
        raise ValueError("histogram_spec must be a dictionary.")
    if "bin_edges" not in histogram_spec or "temperature" not in histogram_spec:
        raise ValueError("histogram_spec requires bin_edges and temperature tensors.")

    batch_size, num_motifs = marginals.shape[:2]
    bin_edges = histogram_spec["bin_edges"].to(
        device=marginals.device,
        dtype=marginals.dtype,
    )
    temperature = histogram_spec["temperature"].to(
        device=marginals.device,
        dtype=marginals.dtype,
    )
    if bin_edges.ndim != 2 or bin_edges.shape[0] != num_motifs:
        raise ValueError(
            "Histogram bin_edges must have shape (M, num_bins - 1): "
            f"got {tuple(bin_edges.shape)} for M={num_motifs}."
        )
    if tuple(temperature.shape) != (num_motifs,):
        raise ValueError(
            "Histogram temperature must have shape (M,), "
            f"got {tuple(temperature.shape)}."
        )
    if (temperature <= 0).any():
        raise ValueError("Histogram temperatures must be greater than zero.")

    num_bins = bin_edges.shape[1] + 1
    if num_motifs == 0:
        return (
            marginals.new_zeros((batch_size, 0, 2, num_bins)),
            torch.zeros(0, 2, num_bins, dtype=torch.bool, device=marginals.device),
        )

    log_values = torch.log1p(marginals.clamp_min(0.0)).unsqueeze(-1)
    scaled_distance = (
        log_values - bin_edges.view(1, num_motifs, 1, 1, -1)
    ) / temperature.view(1, num_motifs, 1, 1, 1)
    above_edge_probability = torch.sigmoid(scaled_distance)
    memberships = torch.cat(
        (
            1.0 - above_edge_probability[..., :1],
            above_edge_probability[..., :-1] - above_edge_probability[..., 1:],
            above_edge_probability[..., -1:],
        ),
        dim=-1,
    )

    valid_mask = valid_mask.to(device=marginals.device, dtype=torch.bool)
    memberships = memberships * valid_mask.unsqueeze(0).unsqueeze(-1).to(
        memberships.dtype
    )
    histograms = memberships.sum(dim=3)
    histogram_mask = valid_mask.any(dim=2).unsqueeze(-1).expand(
        num_motifs,
        2,
        num_bins,
    )
    return histograms, histogram_mask
