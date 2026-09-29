"""Golden count/loss equality for the motif core (D34). Skips when the pinned caches are absent."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "golden"))
import capture_golden as golden  # noqa: E402

NAMES = sorted(golden.DATASETS)


def _inputs_present(name):
    spec = golden.DATASETS[name]
    return (Path(spec["motif_cache_dir"]) / f"{spec['database_name']}.pkl").is_file() and Path(
        spec["dataset_cache"]).is_file()


@pytest.mark.parametrize("name", NAMES)
def test_golden_counts_and_losses_unchanged(name):
    if not _inputs_present(name):
        pytest.skip(f"pinned caches for {name} not on this host")
    torch.set_num_threads(1)
    out, meta = golden.compute(name)
    stored_meta = json.loads((golden.GOLDEN_DIR / f"{name}.json").read_text())
    assert meta["motif_cache_sha256"] == stored_meta["motif_cache_sha256"]
    assert meta["dataset_cache_sha256"] == stored_meta["dataset_cache_sha256"]
    diffs = golden.compare(torch.load(golden.GOLDEN_DIR / f"{name}.pt"), out)
    assert not diffs, diffs
