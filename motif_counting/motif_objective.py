"""Group-specific motif representations and loss composition."""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import torch

from motif_counting.motif_loss_utils import (
    compute_calibrated_gaussian_kiarash_statistics_loss,
    compute_calibrated_gaussian_motif_channel_loss,
    compute_calibrated_gaussian_motif_statistic_loss,
    compute_motif_loss,
)
from motif_counting.motif_representations import (
    canonicalize_motif_output_mode,
    compute_undirected_edge_count_from_full_matrices,
    represent_full_motif_matrices,
)


MOTIF_LOSS_MODES = {
    "abs_log_ratio",
    "squared_log_ratio",
    "calibrated_gaussian",
}
NON_LITERAL_MOTIF_GROUP = "non_literal"
SYNTACTIC_LITERAL_MOTIF_GROUP = "syntactic_literal"
UNIT_RELATION_MOTIF_GROUP = "unit_relation"
UNIT_RELATION_EDGE_COUNT_GROUP = "unit_relation_edge_count"


@dataclass(frozen=True)
class MotifGroupObjective:
    """Configuration and mask for one independently represented motif group."""

    name: str
    motif_mask: torch.Tensor
    output_mode: str
    loss_mode: str
    weight: float
    edge_count_weight: float = 0.0
    histogram_spec: Optional[Dict[str, torch.Tensor]] = None

    @property
    def num_motifs(self) -> int:
        return int(self.motif_mask.sum().item())


@dataclass(frozen=True)
class GroupedMotifLoss:
    """Unweighted group sum, weighted objective, and group diagnostics."""

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
    unit_relation_mask: Optional[torch.Tensor] = None,
    unit_relation_output_mode: Optional[str] = None,
    unit_relation_loss_mode: Optional[str] = None,
    unit_relation_weight: float = 0.0,
    unit_relation_edge_count_weight: float = 0.0,
) -> List[MotifGroupObjective]:
    """Build disjoint original, literal, and optional unit-relation groups."""
    if syntactic_literal_mask.ndim != 1:
        raise ValueError(
            "Expected a 1D syntactic-literal motif mask, "
            f"got {tuple(syntactic_literal_mask.shape)}."
        )
    syntactic_literal_mask = syntactic_literal_mask.to(dtype=torch.bool).cpu()
    if unit_relation_mask is None:
        unit_relation_mask = torch.zeros_like(syntactic_literal_mask)
    elif (
        unit_relation_mask.ndim != 1
        or unit_relation_mask.numel() != syntactic_literal_mask.numel()
    ):
        raise ValueError(
            "Unit-relation motif mask must match the syntactic-literal mask, "
            f"got {tuple(unit_relation_mask.shape)} and "
            f"{tuple(syntactic_literal_mask.shape)}."
        )
    else:
        unit_relation_mask = unit_relation_mask.to(dtype=torch.bool).cpu()

    unit_group_enabled = unit_relation_output_mode is not None
    if unit_group_enabled and unit_relation_loss_mode is None:
        raise ValueError(
            "unit_relation_loss_mode is required when the unit-relation motif "
            "group is enabled."
        )
    if unit_group_enabled and not unit_relation_mask.any():
        raise ValueError(
            "The unit-relation motif group was enabled, but no unit binary-"
            "relation motif survived runtime rule selection."
        )
    if unit_group_enabled and (unit_relation_mask & syntactic_literal_mask).any():
        raise ValueError(
            "Unit-relation and syntactic-literal motif masks must not overlap."
        )
    if not unit_group_enabled and unit_relation_edge_count_weight != 0.0:
        raise ValueError(
            "unit_relation_edge_count_weight requires the unit-relation motif "
            "group to be enabled."
        )

    separated_unit_mask = unit_relation_mask if unit_group_enabled else torch.zeros_like(
        unit_relation_mask
    )
    group_specs = [
        (
            NON_LITERAL_MOTIF_GROUP,
            ~syntactic_literal_mask & ~separated_unit_mask,
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
    ]
    if unit_group_enabled:
        group_specs.append(
            (
                UNIT_RELATION_MOTIF_GROUP,
                separated_unit_mask,
                unit_relation_output_mode,
                unit_relation_loss_mode,
                unit_relation_weight,
            )
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
                edge_count_weight=(
                    float(unit_relation_edge_count_weight)
                    if name == UNIT_RELATION_MOTIF_GROUP
                    else 0.0
                ),
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


def restrict_to_nonzero_weight_motif_groups(
    groups: List[MotifGroupObjective],
):
    """Drop zero-weight groups and project retained masks to active motifs.

    Counting motifs that cannot contribute to the weighted objective wastes
    substantial memory for parity-only or small hybrid experiments. This
    helper returns the projected objectives plus a mask in the original motif
    space, allowing the counter to materialize canonical full matrices only
    for motifs belonging to nonzero-weight groups.
    """
    if not groups:
        return [], torch.zeros(0, dtype=torch.bool)

    num_motifs = groups[0].motif_mask.numel()
    _validate_groups_against_motif_dimension(groups, num_motifs)
    active_mask = torch.zeros(num_motifs, dtype=torch.bool)
    retained_groups = []
    for group in groups:
        if group.weight == 0.0 and group.edge_count_weight == 0.0:
            continue
        active_mask |= group.motif_mask.to(dtype=torch.bool, device="cpu")
        retained_groups.append(group)

    if not retained_groups:
        return [], active_mask

    projected_groups = [
        replace(
            group,
            motif_mask=group.motif_mask.to(dtype=torch.bool, device="cpu")[
                active_mask
            ],
        )
        for group in retained_groups
    ]
    _validate_groups_against_motif_dimension(
        projected_groups,
        int(active_mask.sum().item()),
    )
    return projected_groups, active_mask


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
    if group.output_mode == "kiarash_statistics":
        observed_statistics, _, _ = represent_full_motif_matrices(
            full_matrices=observed_full_matrices[:, motif_mask],
            matrix_mask=group_matrix_mask,
            output_mode=group.output_mode,
        )
        predicted_statistics, _, _ = represent_full_motif_matrices(
            full_matrices=predicted_full_matrices[:, motif_mask],
            matrix_mask=group_matrix_mask,
            output_mode=group.output_mode,
        )
        return compute_calibrated_gaussian_kiarash_statistics_loss(
            observed_statistics=observed_statistics,
            predicted_statistics=predicted_statistics,
            reduction="sum",
        )

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
        # Each group loss is already averaged over the motifs in that group.
        # Compose groups directly so their explicit alpha values control their
        # relative influence, independent of how many motif rules each group
        # contains:
        #   L = sum_g alpha_g * L_g.
        loss = loss + group_loss
        weighted_loss = weighted_loss + group.weight * group_loss

        if (
            group.name == UNIT_RELATION_MOTIF_GROUP
            and group.edge_count_weight != 0.0
        ):
            motif_mask = group.motif_mask.to(
                device=predicted_full_matrices.device,
                dtype=torch.bool,
            )
            group_matrix_mask = full_matrix_mask[motif_mask]
            observed_edge_count = (
                compute_undirected_edge_count_from_full_matrices(
                    observed_full_matrices[:, motif_mask],
                    group_matrix_mask,
                )
            )
            predicted_edge_count = (
                compute_undirected_edge_count_from_full_matrices(
                    predicted_full_matrices[:, motif_mask],
                    group_matrix_mask,
                )
            )
            edge_count_loss = compute_calibrated_gaussian_motif_statistic_loss(
                observed_statistics=observed_edge_count,
                predicted_statistics=predicted_edge_count,
                reduction="sum",
            )
            group_losses[UNIT_RELATION_EDGE_COUNT_GROUP] = edge_count_loss
            loss = loss + edge_count_loss
            weighted_loss = (
                weighted_loss + group.edge_count_weight * edge_count_loss
            )

    return GroupedMotifLoss(
        loss=loss,
        weighted_loss=weighted_loss,
        group_losses=group_losses,
    )
