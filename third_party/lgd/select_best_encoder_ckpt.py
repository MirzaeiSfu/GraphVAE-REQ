#!/usr/bin/env python3
"""Print the encoder checkpoint with the lowest validation RECONSTRUCTION loss.

The encoder is pretrained with ``train.pretrain.recon: all`` and
``graph_factor: 0``, so the logged ``loss`` is the error of an untrained
graph-level head. It grows during training, and selecting on it always picked
epoch 0, an untrained encoder. Selection now uses ``loss_recon`` (the
reconstruction objective the encoder optimizes). The choice and the full
per-epoch table are written to ``SELECTION.json`` next to the run.
"""

import argparse
import json
import math
from pathlib import Path

METRIC = "loss_recon"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-epoch", type=int, default=1,
                        help="never select checkpoints before this epoch (epoch 0 is untrained)")
    args = parser.parse_args()

    ckpt_dir = args.run_dir / "ckpt"
    available = {int(path.stem): path for path in ckpt_dir.glob("*.ckpt")}
    if not available:
        raise SystemExit(f"No encoder checkpoints found in {ckpt_dir}")

    rows = []
    stats_path = args.run_dir / "val" / "stats.json"
    if stats_path.exists():
        for line in stats_path.read_text().splitlines():
            row = json.loads(line)
            epoch = int(row["epoch"])
            value = row.get(METRIC)
            if epoch in available and epoch >= args.min_epoch and value is not None and math.isfinite(float(value)):
                rows.append((float(value), -epoch, epoch))
    if not rows:
        raise SystemExit(f"No finite validation {METRIC} for checkpoints >= epoch {args.min_epoch} in {stats_path}")

    value, _, epoch = min(rows)  # lowest recon loss; ties -> later epoch
    (args.run_dir / "SELECTION.json").write_text(json.dumps({
        "metric": f"val/{METRIC}", "min_epoch": args.min_epoch, "selected_epoch": epoch,
        "selected_value": value, "checkpoint": str(available[epoch].resolve()),
        "candidates": {str(e): v for v, _, e in sorted(rows, key=lambda r: r[2])},
    }, indent=1) + "\n")
    print(available[epoch].resolve())


if __name__ == "__main__":
    main()
