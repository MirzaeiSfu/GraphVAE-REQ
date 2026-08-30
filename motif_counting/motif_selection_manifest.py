"""Persist and restore the exact motif values used by a training run.

The manifest stores the post-source, post-pruning, and post-objective-filter
value rows.  Evaluation can therefore count exactly the trained motifs without
recounting every ``_CP_smoothed`` combination merely to reproduce pruning.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


MOTIF_SELECTION_MANIFEST_FILENAME = "motif_selection_manifest.json"
MOTIF_SELECTION_MANIFEST_SCHEMA = "graphvae-motif-selection-v1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Decimal):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def build_motif_selection_manifest(
    counter: Any,
    selected_rules_values: Mapping[int, list],
    *,
    database_name: str,
    motif_cp_table_source: str,
    rule_prune: bool,
    full_combinations: int,
    pruned_combinations: int,
    active_groups: Optional[list] = None,
) -> Dict[str, Any]:
    """Build a self-contained manifest from the counter's final value rows."""
    selected_rules = []
    for rule_index in sorted(int(index) for index in selected_rules_values):
        value_indices = [
            int(index) for index in selected_rules_values[rule_index]
        ]
        selected_rules.append(
            {
                "rule_index": rule_index,
                "rule": _json_safe(counter.rules[rule_index]),
                "rule_source": _json_safe(counter.rule_sources[rule_index]),
                "value_columns": _json_safe(counter.value_columns[rule_index]),
                "values": [
                    {
                        "training_value_index": value_index,
                        "value_row": _json_safe(
                            counter.values[rule_index][value_index]
                        ),
                    }
                    for value_index in value_indices
                ],
            }
        )

    active_count = sum(len(rule["values"]) for rule in selected_rules)
    return {
        "schema_version": MOTIF_SELECTION_MANIFEST_SCHEMA,
        "database_name": str(database_name),
        "motif_cp_table_source": str(motif_cp_table_source),
        "rule_prune": bool(rule_prune),
        "full_combinations": int(full_combinations),
        "pruned_combinations": int(pruned_combinations),
        "active_combinations": int(active_count),
        "active_groups": _json_safe(active_groups or []),
        "rule_count": len(counter.rules),
        "selected_rules": selected_rules,
    }


def write_motif_selection_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_motif_selection_manifest(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MOTIF_SELECTION_MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported motif-selection manifest schema in {path}: "
            f"{manifest.get('schema_version')!r}."
        )
    return manifest


def apply_motif_selection_manifest(
    counter: Any,
    manifest: Mapping[str, Any],
    *,
    database_name: str,
    motif_cp_table_source: str,
    rule_prune: bool,
) -> Dict[int, list]:
    """Install saved active value rows into a freshly loaded motif counter.

    Rows are restored directly because smoothed pruning derives metadata values
    from the training split; matching them against unprocessed smoothed cache
    rows would both lose that metadata and repeat the expensive preprocessing.
    """
    expected = {
        "database_name": str(database_name),
        "motif_cp_table_source": str(motif_cp_table_source),
        "rule_prune": bool(rule_prune),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Motif-selection manifest disagrees on {key}: "
                f"{manifest.get(key)!r} vs {value!r}."
            )
    if int(manifest.get("rule_count", -1)) != len(counter.rules):
        raise ValueError(
            "Motif-selection manifest rule count does not match the motif cache."
        )

    restored_values = [[] for _ in counter.rules]
    selection: Dict[int, list] = {}
    for saved_rule in manifest.get("selected_rules", []):
        rule_index = int(saved_rule["rule_index"])
        if not 0 <= rule_index < len(counter.rules):
            raise ValueError(f"Manifest has invalid rule index {rule_index}.")
        if saved_rule.get("rule") != _json_safe(counter.rules[rule_index]):
            raise ValueError(
                f"Manifest rule {rule_index} does not match the motif cache."
            )
        if saved_rule.get("rule_source") != _json_safe(
            counter.rule_sources[rule_index]
        ):
            raise ValueError(
                f"Manifest rule source {rule_index} does not match the cache."
            )
        if saved_rule.get("value_columns") != _json_safe(
            counter.value_columns[rule_index]
        ):
            raise ValueError(
                f"Manifest value columns {rule_index} do not match the cache."
            )
        rows = [entry["value_row"] for entry in saved_rule.get("values", [])]
        restored_values[rule_index] = rows
        if rows:
            selection[rule_index] = list(range(len(rows)))

    restored_count = sum(len(indices) for indices in selection.values())
    if restored_count != int(manifest.get("active_combinations", -1)):
        raise ValueError(
            "Motif-selection manifest active count does not match its rows."
        )
    counter.values = restored_values
    counter.data_driven_smoothed_pruning_pending = False
    counter._build_motif_group_masks()
    return selection
