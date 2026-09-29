"""Golden count/loss fixtures for the motif core (D34 P0).

Captured from the pre-core-v1 code so later commits can prove they change
nothing (commit A: exact equality) or only the documented D33 effect (commit B).

    python tests/golden/capture_golden.py            # write tests/golden/<name>.pt + .json
    python tests/golden/capture_golden.py --check    # recompute and compare, exit 1 on any diff

Uses the production paths: Datasets -> merge_datasets -> DataWrapper for
observed targets, ReconstructedDataWrapper for predictions, count_batch, and
compute_grouped_motif_loss. CPU, one thread, fixed seeds.
"""

import argparse
import copy
import hashlib
import json
import pickle
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from data import Datasets, DataWrapper, ReconstructedDataWrapper, merge_datasets  # noqa: E402
from motif_counting.motif_counter import RelationalMotifCounter  # noqa: E402
from motif_counting.motif_objective import (  # noqa: E402
    build_motif_group_objectives,
    calibrate_group_histogram_specs,
    compute_grouped_motif_loss,
    restrict_to_nonzero_weight_motif_groups,
)

GOLDEN_DIR = Path(__file__).resolve().parent
SCRATCH = "/local-scratch2/mirzaei/"
DATASETS = {
    "aids": dict(
        motif_cache_dir=SCRATCH + "aids_common_eval_10k_20260917/cache/motif",
        database_name="aids_undir_feat_multi",
        dataset_cache=SCRATCH + "Abdolreza/GraphVAE-REQ/cache_datasets/AIDS_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-0_bfs-all_components_features-tu-quantile8-max40.pkl",
    ),
    "lobster": dict(
        motif_cache_dir=SCRATCH + "Abdolreza/GraphVAE-REQ/cache_motifs",
        database_name="lobster_undir_feat_snap_85093d_cfg410000000",
        dataset_cache=SCRATCH + "Abdolreza/GraphVAE-REQ/cache_datasets/LOBSTER_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-0_bfs-legacy_first_component_features-lobster-optimal_v2.pkl",
    ),
}
G_TOTAL = 16
G_FULL = 4
SEED = 1234
TEMPERATURES = (1.0, 0.7)
LOSS_CONFIGS = [
    ("total_count", "calibrated_gaussian"),
    ("total_count", "abs_log_ratio"),
    ("full_matrix", "calibrated_gaussian"),
    ("row_column_marginals", "calibrated_gaussian"),
    ("marginal_histogram", "calibrated_gaussian"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _FirstGraphs:
    """Counter preprocessor restricted to the first n graphs of another."""

    def __init__(self, inner, n):
        self._inner, self.num_graphs = inner, min(n, inner.num_graphs)
        self.N_max, self.feature_onehot_mapping = inner.N_max, inner.feature_onehot_mapping

    def get_batch(self, start, end):
        return self._inner.get_batch(start, min(end, self.num_graphs))


def _first_value_per_rule(counter):
    return {r: [0] for r, rows in enumerate(counter.values) if rows}


def compute(name):
    spec = DATASETS[name]
    torch.manual_seed(0)
    counter = RelationalMotifCounter(spec["database_name"], SimpleNamespace(
        motif_cache_dir=spec["motif_cache_dir"], motif_cp_table_source="cp",
        use_syntactic_literal_rules=True, syntactic_literal_rule_mode="both",
        device="cpu", rule_prune=False,
    ))
    with open(spec["dataset_cache"], "rb") as stream:
        cache = pickle.load(stream)
    graphs = Datasets(
        copy.deepcopy(cache["list_adj"]), True, cache["list_x_train"], cache["list_label_train"],
        Max_num=None, set_diag_of_isol_Zer=False,
        list_node_onehot=cache["list_noh_train"], list_edge_onehot=cache["list_eoh_train"],
    )
    observed = DataWrapper(
        merge_datasets(graphs), counter.relation_keys, cache["node_onehot_info"],
        edge_onehot_info=cache["edge_onehot_info"],
        edge_feature_info_mapping=counter.feature_info_mapping, device="cpu",
    )
    out = {}
    subset = _first_value_per_rule(counter)
    out["obs_total_count"] = counter.count_batch(_FirstGraphs(observed, G_TOTAL), output_mode="total_count")
    obs_full_subset, mask_subset = counter.count_batch(
        _FirstGraphs(observed, G_FULL), selected_rules_values=subset, output_mode="full_matrix")
    out["obs_full_subset"], out["full_subset_mask"] = obs_full_subset, mask_subset

    g = torch.Generator().manual_seed(SEED)
    N = observed.N_max
    D = observed.all_feat_onehot.shape[-1]
    C = len(cache["edge_onehot_info"] or {}) or 1
    adj_logits = torch.randn(G_FULL, N, N, generator=g)
    adj_logits = 0.5 * (adj_logits + adj_logits.transpose(1, 2))
    node_logits = torch.randn(G_FULL, N, D, generator=g)
    edge_logits = torch.randn(G_FULL, C, N, N, generator=g) if cache["edge_onehot_info"] else None

    def predicted(temperature):
        return ReconstructedDataWrapper(
            reconstructed_adj=adj_logits, node_feat_logits=node_logits, edge_feat_logits=edge_logits,
            relation_keys=counter.relation_keys, node_onehot_info=cache["node_onehot_info"],
            feature_onehot_mapping=observed.feature_onehot_mapping,
            edge_onehot_info=cache["edge_onehot_info"],
            edge_feature_info_mapping=counter.feature_info_mapping,
            use_soft_adj=True, prob_temperature=temperature, device="cpu",
        )

    for t in TEMPERATURES:
        pred = predicted(t)
        out[f"pred_total_count_T{t}"] = counter.count_batch(pred, output_mode="total_count")
        out[f"pred_full_subset_T{t}"] = counter.count_batch(
            pred, selected_rules_values=subset, output_mode="full_matrix")[0]

    losses = {}
    literal_mask = counter.get_syntactic_literal_motif_mask()
    for output_mode, loss_mode in LOSS_CONFIGS:
        groups = build_motif_group_objectives(
            syntactic_literal_mask=literal_mask,
            non_literal_output_mode=output_mode, non_literal_loss_mode=loss_mode, non_literal_weight=1.0,
            syntactic_literal_output_mode=output_mode, syntactic_literal_loss_mode=loss_mode,
            syntactic_literal_weight=0.5,
        )
        groups, active = restrict_to_nonzero_weight_motif_groups(groups)
        selection = counter.select_rule_values_from_motif_mask(active)
        obs_full, obs_mask = counter.count_batch(
            _FirstGraphs(observed, G_FULL), selected_rules_values=selection, output_mode="full_matrix")
        groups = calibrate_group_histogram_specs(
            observed_full_matrices=obs_full, full_matrix_mask=obs_mask, groups=groups)
        for t in TEMPERATURES:
            pred_full, pred_mask = counter.count_batch(
                predicted(t), selected_rules_values=selection, output_mode="full_matrix")
            assert torch.equal(obs_mask, pred_mask)
            result = compute_grouped_motif_loss(
                observed_full_matrices=obs_full, predicted_full_matrices=pred_full,
                full_matrix_mask=pred_mask, groups=groups)
            key = f"{output_mode}|{loss_mode}|T{t}"
            losses[key] = {
                "loss": result.loss.detach().clone(),
                "weighted_loss": result.weighted_loss.detach().clone(),
                **{f"group:{k}": v.detach().clone() for k, v in result.group_losses.items()},
            }
    out["losses"] = losses
    meta = {
        "dataset": name,
        "database_name": spec["database_name"],
        "motif_cache_sha256": sha256(Path(spec["motif_cache_dir"]) / f"{spec['database_name']}.pkl"),
        "dataset_cache_sha256": sha256(spec["dataset_cache"]),
        "motif_counter_sha256": sha256(ROOT / "motif_counting" / "motif_counter.py"),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "num_values": int(sum(len(v) for v in counter.values)),
        "N_max": int(N),
        "g_total": G_TOTAL,
        "g_full": G_FULL,
        "seed": SEED,
        "adjacency_convention": "A+I incl. padded diagonal (pre-D33)",
    }
    return out, meta


def _flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten(v, f"{prefix}{k}/")
    else:
        yield prefix.rstrip("/"), tree


def compare(expected, actual):
    diffs = {}
    exp, act = dict(_flatten(expected)), dict(_flatten(actual))
    for key in sorted(set(exp) | set(act)):
        if key not in exp or key not in act:
            diffs[key] = "missing"
        elif exp[key].shape != act[key].shape:
            diffs[key] = f"shape {tuple(exp[key].shape)} != {tuple(act[key].shape)}"
        elif not torch.equal(exp[key], act[key]):
            diffs[key] = float((exp[key].double() - act[key].double()).abs().max())
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=sorted(DATASETS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    failed = False
    for name in args.datasets:
        out, meta = compute(name)
        pt, js = GOLDEN_DIR / f"{name}.pt", GOLDEN_DIR / f"{name}.json"
        if args.check:
            stored = torch.load(pt)
            stored_meta = json.loads(js.read_text())
            for k in ("motif_cache_sha256", "dataset_cache_sha256"):
                if stored_meta[k] != meta[k]:
                    print(f"[{name}] input changed: {k}")
                    failed = True
            diffs = compare(stored, out)
            print(f"[{name}] {'EXACT' if not diffs else f'{len(diffs)} differing tensors'}")
            for key, d in list(diffs.items())[:20]:
                print(f"    {key}: {d}")
            failed |= bool(diffs)
        else:
            torch.save(out, pt)
            summary = {k: float(v) for k, v in _flatten(out["losses"])}
            js.write_text(json.dumps({**meta, "losses": summary}, indent=2, sort_keys=True) + "\n")
            print(f"[{name}] wrote {pt.name} ({pt.stat().st_size // 1024} KB)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
