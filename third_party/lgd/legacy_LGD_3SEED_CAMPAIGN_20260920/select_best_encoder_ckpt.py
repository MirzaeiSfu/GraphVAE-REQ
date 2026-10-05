#!/usr/bin/env python3
"""Print the checkpoint from the validation epoch with the lowest loss."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    ckpt_dir = args.run_dir / "ckpt"
    available = {int(path.stem): path for path in ckpt_dir.glob("*.ckpt")}
    if not available:
        raise SystemExit(f"No encoder checkpoints found in {ckpt_dir}")

    candidates = []
    stats_path = args.run_dir / "val" / "stats.json"
    if stats_path.exists():
        for line in stats_path.read_text().splitlines():
            row = json.loads(line)
            epoch = int(row["epoch"])
            if epoch in available:
                candidates.append((float(row["loss"]), epoch))

    epoch = min(candidates)[1] if candidates else max(available)
    print(available[epoch].resolve())


if __name__ == "__main__":
    main()
