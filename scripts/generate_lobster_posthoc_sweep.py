#!/usr/bin/env python3
"""Generate the controlled 18-config Lobster post-training sweep."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/cluster_tests/lobster_graphvae_kia_bce_kl_motif_replacement.yaml"
OUTPUT = ROOT / "configs/loss_weight_sweeps/lobster_posthoc_selection"
SCHEDULE = ROOT / "CLUSTER_GPU_CONFIGS_LOBSTER_POSTHOC_SWEEP.txt"
SLOTS = (("cs-cl-09", "1"), ("cs-cl-16", "0"), ("cs-cl-19", "0"),
         ("cs-cl-19", "1"), ("cs-cl-26", "0"), ("cs-cl-26", "1"))
LITERAL_ONLY_SLOTS = (("cs-cl-13", "0"), ("cs-cl-17", "0"),
                      ("cs-cl-18", "0"), ("cs-cl-36", "0"))
WEIGHTS = (("baseline", 1.0, 1.0, 0.0),
           ("n1_e1_m0p1", 1.0, 1.0, 0.1),
           ("n3_e8_m0p1", 3.0, 8.0, 0.1),
           ("n3_e8_m1", 3.0, 8.0, 1.0))
LITERAL_ONLY_CASES = (
    (True, "n1_e1_m0p1_literals", 0.1),
    (True, "n1_e1_m10_literals", 10.0),
    (False, "n1_e1_m0p1_literals", 0.1),
    (False, "n1_e1_m10_literals", 10.0),
)


def main() -> None:
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configs, rows = [], []

    def add_config(kia_weights, weight_name, node, edge, motif_weight, rule_mode):
        config = copy.deepcopy(template)
        motif, loss = config["motif"], config["loss"]
        runtime, experiment = config["runtime"], config["experiment"]
        motif["motif_loss"] = motif_weight > 0
        motif["use_syntactic_literal_rules"] = rule_mode != "original"
        motif["syntactic_literal_rule_mode"] = rule_mode
        motif["motif_temperature_start"] = 1.0
        motif["motif_temperature_end"] = 1.0
        motif["motif_temperature_anneal_start_frac"] = 1.0
        loss["use_graphvae_mm_bce_kl_weights"] = kia_weights
        loss["alpha_node_feat"], loss["alpha_edge_feat"] = node, edge
        loss["alpha_motif_loss"] = motif_weight
        loss["alpha_syntactic_literal_motif_loss"] = motif_weight
        experiment["Vis_step"] = 4000
        runtime["keep_best_validation_mmd"] = False
        runtime["save_validation_checkpoints"] = False
        runtime["checkpoint_interval_epochs"] = 4000
        runtime["third_party_eval"] = False
        base_mode = "kia40_2000" if kia_weights else "plain1_1"
        stem = f"lobster_{base_mode}_{weight_name}"
        if not weight_name.endswith(f"_{rule_mode}"):
            stem += f"_{rule_mode}"
        runtime["run_label"] = stem
        runtime["graph_save_path"] = f"runs/loss_weight_sweeps/lobster_posthoc_selection/{stem}"
        path = OUTPUT / f"{stem}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configs.append(path)
        rows.append({"config": str(path.relative_to(ROOT)),
                     "bce_kl": "40/2000" if kia_weights else "1/1",
                     "rule_mode": rule_mode, "node": node, "edge": edge,
                     "motif": motif_weight, "temperature": "constant_1"})

    for kia_weights in (False, True):
        for weight_name, node, edge, motif_weight in WEIGHTS:
            rule_modes = ("original",) if motif_weight == 0 else ("original", "both")
            for rule_mode in rule_modes:
                add_config(kia_weights, weight_name, node, edge, motif_weight, rule_mode)

    # Keep the original 14 entries in their historical slots, then fill the
    # four unused wave-3 slots with the requested literal-only comparisons.
    for kia_weights, weight_name, motif_weight in LITERAL_ONLY_CASES:
        add_config(kia_weights, weight_name, 1.0, 1.0, motif_weight, "literals")

    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Lobster post-training-selection sweep (14 configs).",
             "# Inventory only: launch the wave files to avoid GPU conflicts.", ""]
    slots = [SLOTS[index % len(SLOTS)] for index in range(len(configs) - len(LITERAL_ONLY_CASES))]
    slots.extend(LITERAL_ONLY_SLOTS)
    for path, (host, gpu) in zip(configs, slots):
        lines.append(f"{host} {gpu} {path.relative_to(ROOT)}")
    SCHEDULE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for wave_index, start in enumerate(range(0, len(configs), len(SLOTS)), 1):
        wave_configs = configs[start:start + len(SLOTS)]
        wave_slots = slots[start:start + len(SLOTS)]
        wave_lines = [f"# Lobster post-training sweep wave {wave_index}.", ""]
        for path, (host, gpu) in zip(wave_configs, wave_slots):
            wave_lines.append(f"{host} {gpu} {path.relative_to(ROOT)}")
        wave_path = ROOT / f"CLUSTER_GPU_CONFIGS_LOBSTER_POSTHOC_WAVE{wave_index}.txt"
        wave_path.write_text("\n".join(wave_lines) + "\n", encoding="utf-8")
    print(f"Generated {len(configs)} configs, inventory, and 3 launch waves")


if __name__ == "__main__":
    main()
