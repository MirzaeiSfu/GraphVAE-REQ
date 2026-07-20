"""Group-specific motif representations and loss composition."""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import torch

from motif_counting.motif_loss_utils import (
    compute_calibrated_gaussian_motif_channel_loss,
    compute_calibrated_gaussian_motif_statistic_loss,
    compute_motif_loss,
)
from motif_counting.motif_representations import (
    canonicalize_motif_output_mode,
    represent_full_motif_matrices,
)


MOTIF_LOSS_MODES = {
    "abs_log_ratio",
    "squared_log_ratio",
    "calibrated_gaussian",
}
NON_LITERAL_MOTIF_GROUP = "non_literal"
SYNTACTIC_LITERAL_MOTIF_GROUP = "syntactic_literal"


@dataclass(frozen=True)
class MotifGroupObjective:
    """Configuration and mask for one independently represented motif group."""

    name: str
    motif_mask: torch.Tensor
    output_mode: str
    loss_mode: str
    weight: float
    histogram_spec: Optional[Dict[str, torch.Tensor]] = None

    @property
    def num_motifs(self) -> int:
        return int(self.motif_mask.sum().item())


@dataclass(frozen=True)
class GroupedMotifLoss:
    """Unweighted global motif mean, weighted objective, and group diagnostics."""

    loss: torch.Tensor
    weighted_loss: torch.Tensor
    group_losses: Dict[str, torch.Tensor]


def _validate_groups_against_motif_dimension(
    groups: List[MotifGroupObjective],
    num_motifs: int,
) -> None:
    coverage = torch.zeros(num_motifs, dtype=torch.int64)
    for group in groups:
        if group.motif_mask.ndim != 1 or group.motif_mask.numel() != num_motifs:
            raise ValueError(
                f"Motif mask for group {group.name!r} must have shape "
                f"({num_motifs},), got {tuple(group.motif_mask.shape)}."
            )
        coverage = coverage + group.motif_mask.to(dtype=torch.int64, device="cpu")
    if (coverage > 1).any():
        raise ValueError("Motif group masks overlap.")
    if num_motifs > 0 and (coverage == 0).any():
        missing_indices = torch.nonzero(
            coverage == 0,
            as_tuple=False,
        ).flatten().tolist()
        raise ValueError(
            "Motif group masks do not cover every motif; "
            f"missing indices {missing_indices}."
        )


def validate_motif_representation_loss(output_mode: str, loss_mode: str) -> None:
    output_mode = canonicalize_motif_output_mode(output_mode)
    if loss_mode not in MOTIF_LOSS_MODES:
        raise ValueError(
            f"Unknown motif loss mode: {loss_mode}. "
            f"Expected one of {sorted(MOTIF_LOSS_MODES)}."
        )
    if output_mode != "total_count" and loss_mode != "calibrated_gaussian":
        raise ValueError(
            f"motif_output_mode={output_mode} uses a masked Kia-MM calibrated "
            "Gaussian objective and therefore requires "
            "motif_loss_mode=calibrated_gaussian."
        )


def build_motif_group_objectives(
    syntactic_literal_mask: torch.Tensor,
    non_literal_output_mode: str,
    non_literal_loss_mode: str,
    non_literal_weight: float,
    syntactic_literal_output_mode: str,
    syntactic_literal_loss_mode: str,
    syntactic_literal_weight: float,
) -> List[MotifGroupObjective]:
    """Build active non-literal and literal groups from the counter mask."""
    if syntactic_literal_mask.ndim != 1:
        raise ValueError(
            "Expected a 1D syntactic-literal motif mask, "
            f"got {tuple(syntactic_literal_mask.shape)}."
        )
    syntactic_literal_mask = syntactic_literal_mask.to(dtype=torch.bool).cpu()
    group_specs = (
        (
            NON_LITERAL_MOTIF_GROUP,
            ~syntactic_literal_mask,
            non_literal_output_mode,
            non_literal_loss_mode,
            non_literal_weight,
        ),
        (
            SYNTACTIC_LITERAL_MOTIF_GROUP,
            syntactic_literal_mask,
            syntactic_literal_output_mode,
            syntactic_literal_loss_mode,
            syntactic_literal_weight,
        ),
    )

    groups = []
    for name, motif_mask, output_mode, loss_mode, weight in group_specs:
        if not motif_mask.any():
            continue
        output_mode = canonicalize_motif_output_mode(output_mode)
        validate_motif_representation_loss(output_mode, loss_mode)
        groups.append(
            MotifGroupObjective(
                name=name,
                motif_mask=motif_mask,
                output_mode=output_mode,
                loss_mode=loss_mode,
                weight=float(weight),
            )
        )
    return groups


def calibrate_group_histogram_specs(
    observed_full_matrices: torch.Tensor,
    full_matrix_mask: torch.Tensor,
    groups: List[MotifGroupObjective],
    histogram_num_bins: int = 16,
    histogram_smoothing: float = 0.25,
) -> List[MotifGroupObjective]:
    """Calibrate histogram bins once from all observed graphs in each group."""
    _validate_groups_against_motif_dimension(
        groups,
        observed_full_matrices.shape[1],
    )
    calibrated_groups = []
    for group in groups:
        if group.output_mode != "marginal_histogram":
            calibrated_groups.append(group)
            continue
        motif_mask = group.motif_mask.to(
            device=observed_full_matrices.device,
            dtype=torch.bool,
        )
        _, _, histogram_spec = represent_full_motif_matrices(
            full_matrices=observed_full_matrices[:, motif_mask],
            matrix_mask=full_matrix_mask[motif_mask],
            output_mode=group.output_mode,
            histogram_num_bins=histogram_num_bins,
            histogram_smoothing=histogram_smoothing,
        )
        calibrated_groups.append(
            replace(group, histogram_spec=histogram_spec)
        )
    return calibrated_groups


def _compute_group_loss(
    observed_full_matrices: torch.Tensor,
    predicted_full_matrices: torch.Tensor,
    full_matrix_mask: torch.Tensor,
    group: MotifGroupObjective,
    histogram_num_bins: int,
    histogram_smoothing: float,
) -> torch.Tensor:
    if group.output_mode == "marginal_histogram" and group.histogram_spec is None:
        raise RuntimeError(
            f"Histogram bins for motif group {group.name!r} were not calibrated. "
            "Call calibrate_group_histogram_specs on the observed full matrices "
            "before computing the grouped loss."
        )
    motif_mask = group.motif_mask.to(
        device=predicted_full_matrices.device,
        dtype=torch.bool,
    )
    group_matrix_mask = full_matrix_mask[motif_mask]
    observed_statistics, observed_mask, _ = represent_full_motif_matrices(
        full_matrices=observed_full_matrices[:, motif_mask],
        matrix_mask=group_matrix_mask,
        output_mode=group.output_mode,
        histogram_num_bins=histogram_num_bins,
        histogram_smoothing=histogram_smoothing,
        histogram_spec=group.histogram_spec,
    )
    predicted_statistics, predicted_mask, _ = represent_full_motif_matrices(
        full_matrices=predicted_full_matrices[:, motif_mask],
        matrix_mask=group_matrix_mask,
        output_mode=group.output_mode,
        histogram_num_bins=histogram_num_bins,
        histogram_smoothing=histogram_smoothing,
        histogram_spec=group.histogram_spec,
    )

    if observed_mask is None:
        if predicted_mask is not None:
            raise RuntimeError("Observed and predicted motif masks do not match.")
    elif predicted_mask is None or not torch.equal(observed_mask, predicted_mask):
        raise RuntimeError("Observed and predicted motif masks do not match.")

    if group.output_mode == "total_count":
        return compute_motif_loss(
            observed_counts=observed_statistics,
            predicted_counts=predicted_statistics,
            loss_mode=group.loss_mode,
        )
    if group.output_mode in {
        "row_column_marginals",
        "marginal_histogram",
    }:
        return compute_calibrated_gaussian_motif_channel_loss(
            observed_statistics=observed_statistics,
            predicted_statistics=predicted_statistics,
            valid_mask=predicted_mask,
        )
    return compute_calibrated_gaussian_motif_statistic_loss(
        observed_statistics=observed_statistics,
        predicted_statistics=predicted_statistics,
        valid_mask=predicted_mask,
    )


def compute_grouped_motif_loss(
    observed_full_matrices: torch.Tensor,
    predicted_full_matrices: torch.Tensor,
    full_matrix_mask: torch.Tensor,
    groups: List[MotifGroupObjective],
    histogram_num_bins: int = 16,
    histogram_smoothing: float = 0.25,
) -> GroupedMotifLoss:
    """Compute group losses from one shared full-matrix counting result."""
    if observed_full_matrices.shape != predicted_full_matrices.shape:
        raise ValueError(
            f"Shape mismatch: observed {tuple(observed_full_matrices.shape)} vs "
            f"predicted {tuple(predicted_full_matrices.shape)}"
        )
    _validate_groups_against_motif_dimension(
        groups,
        observed_full_matrices.shape[1],
    )
    zero = predicted_full_matrices.sum() * 0.0
    total_motifs = sum(group.num_motifs for group in groups)
    if total_motifs == 0:
        return GroupedMotifLoss(
            loss=zero,
            weighted_loss=zero,
            group_losses={},
        )

    group_losses = {}
    loss = zero
    weighted_loss = zero
    for group in groups:
        group_loss = _compute_group_loss(
            observed_full_matrices=observed_full_matrices,
            predicted_full_matrices=predicted_full_matrices,
            full_matrix_mask=full_matrix_mask,
            group=group,
            histogram_num_bins=histogram_num_bins,
            histogram_smoothing=histogram_smoothing,
        )
        group_losses[group.name] = group_loss
        motif_fraction = group.num_motifs / float(total_motifs)
        loss = loss + motif_fraction * group_loss
        weighted_loss = (
            weighted_loss
            + group.weight * motif_fraction * group_loss
        )

    return GroupedMotifLoss(
        loss=loss,
        weighted_loss=weighted_loss,
        group_losses=group_losses,
    )
