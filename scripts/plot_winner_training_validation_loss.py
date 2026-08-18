#!/usr/bin/env python3
"""Plot training objective and post-hoc validation score for a selected run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402


TRAIN_PATTERN = re.compile(
    r"Epoch:\s*(?P<epoch>\d+).*?\| loss:\s*(?P<loss>[-+0-9.eE]+)"
)
CHECKPOINT_PATTERN = re.compile(r"periodic_epoch_(?P<epoch>\d+)\.pt$")


def parse_training_loss(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = TRAIN_PATTERN.search(line)
            if match:
                rows.append(
                    (int(match.group("epoch")), float(match.group("loss")))
                )
    if not rows:
        raise ValueError(f"No epoch/loss records found in {path}")
    epochs = np.asarray([row[0] for row in rows], dtype=int)
    losses = np.asarray([row[1] for row in rows], dtype=float)
    if len(np.unique(epochs)) != len(epochs):
        raise ValueError(f"Duplicate epoch records found in {path}")
    return epochs, losses


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("rolling window must be positive")
    result = np.full(values.shape, np.nan, dtype=float)
    if len(values) < window:
        return result
    result[window - 1 :] = np.convolve(
        values, np.ones(window, dtype=float) / window, mode="valid"
    )
    return result


def selected_validation_curve(selection_path: Path, run_dir: Path):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    winner = selection["winner"]
    if Path(winner["artifact_dir"]).resolve() != run_dir.resolve():
        raise ValueError(
            f"Selection winner {winner['artifact_dir']} does not match {run_dir}"
        )
    points = []
    for candidate in selection["candidates"]:
        if Path(candidate["artifact_dir"]).resolve() != run_dir.resolve():
            continue
        match = CHECKPOINT_PATTERN.fullmatch(candidate["checkpoint"])
        if not match:
            continue
        points.append(
            {
                "epoch": int(match.group("epoch")),
                "mean": float(candidate["selection_score"]),
                "std": float(candidate["summary"]["score"]["std"]),
                "median": float(candidate["summary"]["score"]["median"]),
            }
        )
    points.sort(key=lambda row: row["epoch"])
    if not points:
        raise ValueError(f"No validation candidates found for {run_dir}")
    winner_match = CHECKPOINT_PATTERN.fullmatch(winner["checkpoint"])
    if winner_match is None:
        raise ValueError(f"Unexpected winner checkpoint: {winner['checkpoint']}")
    return points, int(winner_match.group("epoch")), winner


def write_curve_csv(
    path: Path,
    epochs: np.ndarray,
    losses: np.ndarray,
    smoothed: np.ndarray,
    validation_points: list[dict],
    selected_epoch: int,
    rolling_window: int,
) -> None:
    validation_by_epoch = {row["epoch"]: row for row in validation_points}
    fieldnames = [
        "epoch",
        "training_total_objective",
        f"training_total_objective_trailing_mean_{rolling_window}",
        "validation_normalized_table2_table3_mean",
        "validation_normalized_table2_table3_std",
        "validation_normalized_table2_table3_median",
        "selected_checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, loss, smooth in zip(epochs, losses, smoothed):
            validation = validation_by_epoch.get(int(epoch))
            writer.writerow(
                {
                    "epoch": int(epoch),
                    "training_total_objective": float(loss),
                    f"training_total_objective_trailing_mean_{rolling_window}": (
                        "" if not np.isfinite(smooth) else float(smooth)
                    ),
                    "validation_normalized_table2_table3_mean": (
                        "" if validation is None else validation["mean"]
                    ),
                    "validation_normalized_table2_table3_std": (
                        "" if validation is None else validation["std"]
                    ),
                    "validation_normalized_table2_table3_median": (
                        "" if validation is None else validation["median"]
                    ),
                    "selected_checkpoint": int(epoch) == selected_epoch,
                }
            )


def draw_plot(
    output_path: Path,
    epochs: np.ndarray,
    losses: np.ndarray,
    smoothed: np.ndarray,
    validation_points: list[dict],
    selected_epoch: int,
    rolling_window: int,
) -> None:
    # The project environment currently uses Matplotlib 3.3, whose bundled
    # seaborn style predates the ``seaborn-v0_8-*`` naming convention.
    plt.style.use("seaborn-whitegrid")
    figure, (train_axis, validation_axis) = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.16},
    )

    train_axis.plot(
        epochs,
        losses,
        color="#4C78A8",
        alpha=0.16,
        linewidth=0.55,
        label="Per-epoch training objective",
    )
    train_axis.plot(
        epochs,
        smoothed,
        color="#1F4E79",
        linewidth=2.0,
        label=f"Trailing mean ({rolling_window} epochs)",
    )
    train_axis.axvline(
        selected_epoch,
        color="#D62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Selected checkpoint ({selected_epoch:,})",
    )
    train_axis.set_ylabel("Training total objective")
    train_axis.set_title(
        "Winner matrix-motif run: training objective and validation selection score",
        fontsize=14,
        pad=12,
    )
    train_axis.legend(loc="upper right", frameon=True, fontsize=9)

    inset = inset_axes(train_axis, width="44%", height="43%", loc="center right")
    late_mask = epochs >= max(1000, int(epochs.min()))
    inset.plot(
        epochs[late_mask],
        losses[late_mask],
        color="#4C78A8",
        alpha=0.08,
        linewidth=0.4,
    )
    inset.plot(
        epochs[late_mask],
        smoothed[late_mask],
        color="#1F4E79",
        linewidth=1.5,
    )
    inset.axvline(selected_epoch, color="#D62728", linestyle="--", linewidth=1.0)
    finite_late = smoothed[late_mask & np.isfinite(smoothed)]
    if len(finite_late):
        low, high = np.quantile(finite_late, [0.01, 0.99])
        margin = max((high - low) * 0.18, 0.005)
        inset.set_ylim(low - margin, high + margin)
    inset.set_title("Late-training zoom", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.25)

    validation_epochs = np.asarray(
        [row["epoch"] for row in validation_points], dtype=int
    )
    validation_means = np.asarray(
        [row["mean"] for row in validation_points], dtype=float
    )
    validation_stds = np.asarray(
        [row["std"] for row in validation_points], dtype=float
    )
    validation_axis.errorbar(
        validation_epochs,
        validation_means,
        yerr=validation_stds,
        color="#F58518",
        marker="o",
        markersize=6,
        linewidth=2.0,
        capsize=5,
        label="Validation score: mean ± std over 10 rollouts",
    )
    selected_index = int(np.where(validation_epochs == selected_epoch)[0][0])
    selected_score = validation_means[selected_index]
    validation_axis.scatter(
        [selected_epoch],
        [selected_score],
        marker="*",
        s=220,
        color="#D62728",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label=f"Minimum validation mean = {selected_score:.3f}",
    )
    validation_axis.axvline(
        selected_epoch, color="#D62728", linestyle="--", linewidth=1.5
    )
    validation_axis.annotate(
        f"selected\n{selected_epoch:,}",
        xy=(selected_epoch, selected_score),
        xytext=(selected_epoch - 2300, selected_score - 0.25),
        arrowprops={"arrowstyle": "->", "color": "#D62728"},
        fontsize=9,
        color="#8B1A1A",
    )
    validation_axis.set_xlabel("Training epoch")
    validation_axis.set_ylabel("Validation normalized\nTable2+Table3 score")
    validation_axis.legend(loc="upper left", frameon=True, fontsize=9)
    validation_axis.set_xlim(int(epochs.min()), int(epochs.max()))
    validation_axis.set_ylim(bottom=0.0)

    figure.text(
        0.5,
        0.015,
        "Lower is better. Training objective and validation score are different quantities and use separate y-axes.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.09)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    selection_path = args.selection_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs, losses = parse_training_loss(run_dir / "train.log")
    smoothed = trailing_mean(losses, args.rolling_window)
    validation_points, selected_epoch, winner = selected_validation_curve(
        selection_path, run_dir
    )
    write_curve_csv(
        output_dir / "winner_training_validation_loss.csv",
        epochs,
        losses,
        smoothed,
        validation_points,
        selected_epoch,
        args.rolling_window,
    )
    draw_plot(
        output_dir / "winner_training_validation_loss.png",
        epochs,
        losses,
        smoothed,
        validation_points,
        selected_epoch,
        args.rolling_window,
    )
    draw_plot(
        output_dir / "winner_training_validation_loss.pdf",
        epochs,
        losses,
        smoothed,
        validation_points,
        selected_epoch,
        args.rolling_window,
    )
    print(f"Run: {winner['run']}")
    print(f"Training epochs: {len(epochs)}")
    print(f"Validation checkpoints: {len(validation_points)}")
    print(f"Selected epoch: {selected_epoch}")
    print(f"Wrote plots and CSV to {output_dir}")


if __name__ == "__main__":
    main()
