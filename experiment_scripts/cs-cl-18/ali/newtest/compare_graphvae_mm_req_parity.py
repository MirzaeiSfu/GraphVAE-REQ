#!/usr/bin/env python
"""Compare GraphVAE-MM and GraphVAE-REQ with motif/feature losses disabled.

The script runs one deterministic training minibatch for each requested
dataset/model pair and compares the tensors that should be identical when the
REQ additions are inactive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
MICRO_PYTHON = Path("/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python")

DATASET_NAMES = {
    "grid": {"MM": "grid", "REQ": "GRID"},
    "triangular_grid": {"MM": "triangular_grid", "REQ": "TRIANGULAR_GRID"},
    "lobster": {"MM": "lobster", "REQ": "LOBSTER"},
    "PROTEINS": {"MM": "PROTEINS", "REQ": "PROTEINS"},
}

MODEL_NAMES = ("GraphVAE-MM", "kipf")


def set_reference_seed() -> None:
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch._set_deterministic(True)
    except Exception:
        pass


def clear_repo_modules() -> None:
    exact = {
        "Aggregation",
        "GlobalProperties",
        "Synthatic_graph_generator",
        "classification",
        "data",
        "diffPool",
        "graph_statistics",
        "input_data",
        "mask_test_edges",
        "mmd_rnn",
        "model",
        "plotter",
        "randomGraphGen",
        "stat_rnn",
        "util",
        "visualization",
    }
    for name in list(sys.modules):
        if name in exact or name.startswith("dataset_feature_utils"):
            sys.modules.pop(name, None)


def patch_dgl_gin_dataset(cache_dir: Path) -> None:
    import dgl

    cache_dir.mkdir(parents=True, exist_ok=True)
    original = getattr(dgl.data, "_parity_original_GINDataset", dgl.data.GINDataset)
    dgl.data._parity_original_GINDataset = original

    def patched_gin_dataset(
        name,
        self_loop,
        degree_as_nlabel=False,
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        resolved_raw_dir = str(cache_dir if raw_dir is None else raw_dir)
        return original(
            name=name,
            self_loop=self_loop,
            degree_as_nlabel=degree_as_nlabel,
            raw_dir=resolved_raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    dgl.data.GINDataset = patched_gin_dataset


def tensor_to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().double().numpy()
    return np.asarray(value)


def tensor_sha(value) -> str | None:
    arr = tensor_to_numpy(value)
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def sparse_sha(value) -> str:
    csr = value.tocsr()
    h = hashlib.sha256()
    h.update(str(csr.shape).encode("utf-8"))
    h.update(np.asarray(csr.indptr).tobytes())
    h.update(np.asarray(csr.indices).tobytes())
    h.update(np.asarray(csr.data).tobytes())
    return h.hexdigest()


def max_abs_diff(left, right) -> float:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return float("inf")
    a = tensor_to_numpy(left)
    b = tensor_to_numpy(right)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def list_max_abs_diff(left, right) -> float:
    if len(left) != len(right):
        return float("inf")
    diffs = [max_abs_diff(a, b) for a, b in zip(left, right)]
    return max(diffs) if diffs else 0.0


def log_guss(mean, log_std, samples):
    return (
        0.5 * torch.pow((samples - mean) / log_std.exp(), 2)
        + log_std
        + 0.5 * np.log(2 * np.pi)
    )


def softclip(tensor, min_value):
    return min_value + F.softplus(tensor - min_value)


def optimizer_vae(
    reconstructed_adj,
    reconstructed_kernel_val,
    target_adj,
    target_kernel_val,
    log_std,
    mean,
    alpha,
    reconstructed_adj_logit,
    pos_weight,
    norm,
):
    recon_loss = norm * F.binary_cross_entropy_with_logits(
        reconstructed_adj_logit.float(),
        target_adj.float(),
        pos_weight=pos_weight,
    )

    latent_norm = mean.shape[0] * mean.shape[1]
    kl = (1 / latent_norm) * -0.5 * torch.sum(
        1 + 2 * log_std - mean.pow(2) - torch.exp(log_std).pow(2)
    )

    acc = (reconstructed_adj.round() == target_adj).sum() / float(
        reconstructed_adj.shape[0]
        * reconstructed_adj.shape[1]
        * reconstructed_adj.shape[2]
    )

    kernel_diff = 0
    each_kernel_loss = []
    log_sigma_values = []
    for i in range(len(target_kernel_val)):
        log_sigma = ((reconstructed_kernel_val[i] - target_kernel_val[i]) ** 2).mean().sqrt().log()
        log_sigma = softclip(log_sigma, -6)
        log_sigma_values.append(float(log_sigma.detach().cpu().item()))
        step_loss = log_guss(target_kernel_val[i], log_sigma, reconstructed_kernel_val[i]).mean()
        each_kernel_loss.append(float((step_loss.detach().cpu() * alpha[i]).item()))
        kernel_diff += step_loss * alpha[i]

    kernel_diff += recon_loss * alpha[-2]
    kernel_diff += kl * alpha[-1]
    each_kernel_loss.append(float((recon_loss * alpha[-2]).detach().cpu().item()))
    each_kernel_loss.append(float((kl * alpha[-1]).detach().cpu().item()))
    return kl, recon_loss, acc, kernel_diff, each_kernel_loss, log_sigma_values


def get_subgraph_features(org_adj, kernel_model, device):
    subgraphs = []
    for graph in org_adj:
        subgraphs.append(torch.tensor(graph.todense()))
    subgraphs = torch.stack(subgraphs).to(device)

    target_kernel_val = None
    if kernel_model is not None:
        target_kernel_val = kernel_model(subgraphs)
        target_kernel_val = [value.to("cpu") for value in target_kernel_val]
    return target_kernel_val, subgraphs.to("cpu")


def alpha_for(dataset_key: str, model_name: str):
    if model_name == "kipf":
        return [], 0, [1, 1]
    kernel_types = [
        "trans_matrix",
        "in_degree_dist",
        "out_degree_dist",
        "TotalNumberOfTriangles",
    ]
    if dataset_key == "lobster":
        return kernel_types, 5, [1, 1, 1, 1, 1, 1, 1, 1, 40, 2000]
    # Grid, triangular-grid, and REQ PROTEINS all use these BCE/KL weights.
    # GraphVAE-MM/main.py has no PROTEINS branch for GraphVAE-MM, so the
    # PROTEINS MM comparison uses the REQ branch's intended alpha explicitly.
    return kernel_types, 5, [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]


def load_repo(repo_kind: str):
    clear_repo_modules()
    repo_name = "GraphVAE-MM" if repo_kind == "MM" else "GraphVAE-REQ"
    repo_path = ROOT / repo_name
    sys.path.insert(0, str(repo_path))
    old_cwd = Path.cwd()
    os.chdir(repo_path)
    if repo_kind == "REQ":
        os.environ["DATA_DIR"] = str(repo_path / "data_raw")
    try:
        import data
        import model
        import GlobalProperties
        import util
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(repo_path))
        except ValueError:
            pass
    return repo_path, data, model, GlobalProperties, util


def prepare_dataset(repo_kind, data_mod, util_mod, dataset_key):
    dataset_name = DATASET_NAMES[dataset_key][repo_kind]
    loaded = data_mod.list_graph_loader(dataset_name, return_labels=True)

    if repo_kind == "REQ":
        (
            list_adj,
            list_x,
            list_label,
            list_node_feature,
            list_edge_feature,
            node_feature_info,
            edge_feature_info,
        ) = loaded
        list_adj, list_node_feature, list_edge_feature = data_mod.BFS(
            list_adj,
            list_node_feature,
            list_edge_feature,
        )
        list_node_onehot, list_edge_onehot, _, _ = util_mod.build_onehot_features(
            list_node_feature,
            list_edge_feature,
            list_adj,
            node_feature_info,
            edge_feature_info,
        )
        (
            list_adj_train,
            test_list_adj,
            list_x_train,
            _list_x_test,
            _list_label_train,
            _list_label_test,
            list_noh_train,
            _list_noh_test,
            list_eoh_train,
            _list_eoh_test,
        ) = data_mod.data_split(
            list_adj,
            list_x,
            list_label,
            list_node_onehot,
            list_edge_onehot,
            train_fraction=0.8,
            seed=123,
        )
        list_graphs = data_mod.Datasets(
            list_adj_train,
            True,
            list_x_train,
            list_label,
            Max_num=None,
            set_diag_of_isol_Zer=False,
            list_node_onehot=list_noh_train,
            list_edge_onehot=list_eoh_train,
        )
    else:
        list_adj, list_x, list_label = loaded
        list_adj = data_mod.BFS(list_adj)
        (
            list_adj_train,
            test_list_adj,
            list_x_train,
            _list_x_test,
            _list_label_train,
            _list_label_test,
        ) = data_mod.data_split(list_adj, list_x, list_label)
        list_graphs = data_mod.Datasets(
            list_adj_train,
            True,
            list_x_train,
            list_label,
            Max_num=None,
            set_diag_of_isol_Zer=False,
        )

    return {
        "dataset_name": dataset_name,
        "list_graphs": list_graphs,
        "test_list_adj": test_list_adj,
        "train_adj_hashes": [sparse_sha(adj) for adj in list_graphs.list_adjs[:10]],
        "max_num_nodes": int(list_graphs.max_num_nodes),
        "feature_size": int(list_graphs.feature_size),
        "train_size": int(len(list_graphs.list_adjs)),
        "test_size": int(len(test_list_adj)),
    }


def run_case(repo_kind: str, dataset_key: str, model_name: str, device: torch.device):
    set_reference_seed()
    patch_dgl_gin_dataset(ROOT / ".dgl_cache_parity")
    repo_path, data_mod, model_mod, global_mod, util_mod = load_repo(repo_kind)

    old_cwd = Path.cwd()
    os.chdir(repo_path)
    try:
        prepared = prepare_dataset(repo_kind, data_mod, util_mod, dataset_key)
        list_graphs = prepared["list_graphs"]
        kernel_types, step_num, alpha = alpha_for(dataset_key, model_name)

        subgraph_node_num = list_graphs.max_num_nodes
        degree_center = torch.tensor([[x] for x in range(0, subgraph_node_num, 1)])
        degree_width = torch.tensor([[0.1] for _ in range(0, subgraph_node_num, 1)])
        bin_center = torch.tensor([[x] for x in range(0, subgraph_node_num, 1)])
        bin_width = torch.tensor([[1] for _ in range(0, subgraph_node_num, 1)])
        kernel_model = global_mod.kernel(
            device=device,
            kernel_type=kernel_types,
            step_num=step_num,
            bin_width=bin_width,
            bin_center=bin_center,
            degree_bin_center=degree_center,
            degree_bin_width=degree_width,
        )

        encoder = model_mod.AveEncoder(list_graphs.feature_size, [256], 1024)
        decoder = model_mod.GraphTransformerDecoder_FC(
            1024,
            256,
            list_graphs.max_num_nodes,
            True,
        )

        if repo_kind == "REQ":
            model = model_mod.kernelGVAE(
                kernel_model,
                encoder,
                decoder,
                False,
                graphEmDim=1024,
                node_feature_decoder=None,
                edge_feature_decoder=None,
            )
        else:
            model = model_mod.kernelGVAE(
                kernel_model,
                encoder,
                decoder,
                False,
                graphEmDim=1024,
            )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), 0.0003)

        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        list_graphs.shuffle()
        list_graphs.processALL(self_for_none=True)
        adj_list = list_graphs.get_adj_list()
        graph_features, _ = get_subgraph_features(adj_list, kernel_model, device)
        list_graphs.set_features(graph_features)

        # First epoch shuffle from the training loop.
        list_graphs.shuffle()
        train_batch_size = 200
        org_adj, x_s, node_num, subgraphs_indexes, target_kernel_val = list_graphs.get__(
            0,
            train_batch_size,
            True,
            bfs=None,
        )

        decoder_batch_node_num = len(node_num) * [list_graphs.max_num_nodes]
        x_s = torch.cat(x_s).reshape(-1, x_s[0].shape[-1])
        _, subgraphs = get_subgraph_features(org_adj, kernel_model=None, device=device)
        batch_size = [len(org_adj), org_adj[0].shape[0]]

        import dgl

        for graph in org_adj:
            graph.setdiag(1)
        org_adj_dgl = dgl.batch([dgl.from_scipy(graph) for graph in org_adj]).to(device)
        pos_weight = torch.true_divide(
            sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(),
            subgraphs.sum(),
        )

        forward_out = model(
            org_adj_dgl.to(device),
            x_s.to(device),
            batch_size,
            subgraphs_indexes,
        )
        if repo_kind == "REQ":
            (
                reconstructed_adj,
                prior_samples,
                post_mean,
                post_log_std,
                generated_kernel_val,
                reconstructed_adj_logit,
                node_feat_logits,
                edge_feat_logits,
            ) = forward_out
        else:
            (
                reconstructed_adj,
                prior_samples,
                post_mean,
                post_log_std,
                generated_kernel_val,
                reconstructed_adj_logit,
            ) = forward_out
            node_feat_logits = None
            edge_feat_logits = None

        (
            kl_loss,
            reconstruction_loss,
            acc,
            kernel_cost,
            each_kernel_loss,
            log_sigma_values,
        ) = optimizer_vae(
            reconstructed_adj,
            generated_kernel_val,
            subgraphs.to(device),
            [value.to(device) for value in target_kernel_val],
            post_log_std,
            post_mean,
            alpha,
            reconstructed_adj_logit,
            pos_weight,
            2,
        )

        # REQ disabled terms: alpha_node_feat=0, alpha_edge_feat=0,
        # motif_loss=False/alpha_motif_loss=0, edge_count_loss=False.
        loss = kernel_cost
        optimizer.zero_grad()
        loss.backward()
        gradients = {
            name: None if param.grad is None else param.grad.detach().cpu().clone()
            for name, param in model.named_parameters()
        }
        optimizer.step()
        after_step_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        return {
            "repo": repo_kind,
            "dataset": dataset_key,
            "dataset_name": prepared["dataset_name"],
            "model": model_name,
            "train_size": prepared["train_size"],
            "test_size": prepared["test_size"],
            "max_num_nodes": prepared["max_num_nodes"],
            "feature_size": prepared["feature_size"],
            "train_adj_hashes": prepared["train_adj_hashes"],
            "batch_graphs": len(org_adj),
            "decoder_node_num_first": int(decoder_batch_node_num[0]),
            "node_feat_logits_is_none": node_feat_logits is None,
            "edge_feat_logits_is_none": edge_feat_logits is None,
            "losses": {
                "loss": float(loss.detach().cpu().item()),
                "kernel_cost": float(kernel_cost.detach().cpu().item()),
                "reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
                "kl_loss": float(kl_loss.detach().cpu().item()),
                "acc": float(acc.detach().cpu().item()),
                "pos_weight": float(pos_weight.detach().cpu().item()),
            },
            "each_kernel_loss": each_kernel_loss,
            "log_sigma_values": log_sigma_values,
            "tensors": {
                "x_s": x_s.detach().cpu(),
                "subgraphs": subgraphs.detach().cpu(),
                "post_mean": post_mean.detach().cpu(),
                "post_log_std": post_log_std.detach().cpu(),
                "prior_samples": prior_samples.detach().cpu(),
                "reconstructed_adj": reconstructed_adj.detach().cpu(),
                "reconstructed_adj_logit": reconstructed_adj_logit.detach().cpu(),
            },
            "kernel_tensors": [value.detach().cpu() for value in generated_kernel_val],
            "target_kernel_tensors": [value.detach().cpu() for value in target_kernel_val],
            "initial_state": initial_state,
            "gradients": gradients,
            "after_step_state": after_step_state,
            "hashes": {
                "x_s": tensor_sha(x_s),
                "subgraphs": tensor_sha(subgraphs),
                "post_mean": tensor_sha(post_mean),
                "post_log_std": tensor_sha(post_log_std),
                "prior_samples": tensor_sha(prior_samples),
                "reconstructed_adj": tensor_sha(reconstructed_adj),
                "reconstructed_adj_logit": tensor_sha(reconstructed_adj_logit),
            },
        }
    finally:
        os.chdir(old_cwd)
        clear_repo_modules()


def compare_cases(mm_result, req_result, atol: float):
    diffs = {}
    for key, left in mm_result["losses"].items():
        diffs[f"losses.{key}"] = abs(left - req_result["losses"][key])

    for key in mm_result["tensors"]:
        diffs[f"tensors.{key}"] = max_abs_diff(
            mm_result["tensors"][key],
            req_result["tensors"][key],
        )

    diffs["generated_kernel_val"] = list_max_abs_diff(
        mm_result["kernel_tensors"],
        req_result["kernel_tensors"],
    )
    diffs["target_kernel_val"] = list_max_abs_diff(
        mm_result["target_kernel_tensors"],
        req_result["target_kernel_tensors"],
    )
    diffs["each_kernel_loss"] = max_abs_diff(
        np.asarray(mm_result["each_kernel_loss"]),
        np.asarray(req_result["each_kernel_loss"]),
    )
    diffs["log_sigma_values"] = max_abs_diff(
        np.asarray(mm_result["log_sigma_values"]),
        np.asarray(req_result["log_sigma_values"]),
    )

    shared_state_keys = sorted(
        set(mm_result["initial_state"]).intersection(req_result["initial_state"])
    )
    diffs["initial_state"] = max(
        (
            max_abs_diff(
                mm_result["initial_state"][key],
                req_result["initial_state"][key],
            )
            for key in shared_state_keys
        ),
        default=0.0,
    )
    diffs["gradients"] = max(
        (
            max_abs_diff(
                mm_result["gradients"].get(key),
                req_result["gradients"].get(key),
            )
            for key in shared_state_keys
        ),
        default=0.0,
    )
    diffs["after_step_state"] = max(
        (
            max_abs_diff(
                mm_result["after_step_state"][key],
                req_result["after_step_state"][key],
            )
            for key in shared_state_keys
        ),
        default=0.0,
    )
    diffs["train_adj_hashes_equal"] = (
        0.0
        if mm_result["train_adj_hashes"] == req_result["train_adj_hashes"]
        else float("inf")
    )
    diffs["node_feat_logits_none_equal"] = (
        0.0
        if mm_result["node_feat_logits_is_none"] == req_result["node_feat_logits_is_none"]
        else float("inf")
    )
    diffs["edge_feat_logits_none_equal"] = (
        0.0
        if mm_result["edge_feat_logits_is_none"] == req_result["edge_feat_logits_is_none"]
        else float("inf")
    )

    max_diff = max(diffs.values()) if diffs else 0.0
    return {
        "passed": bool(max_diff <= atol),
        "max_abs_diff": max_diff,
        "diffs": diffs,
    }


def json_safe_result(result):
    return {
        key: value
        for key, value in result.items()
        if key
        not in {
            "tensors",
            "kernel_tensors",
            "target_kernel_tensors",
            "initial_state",
            "gradients",
            "after_step_state",
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["grid", "triangular_grid", "PROTEINS"],
        choices=sorted(DATASET_NAMES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_NAMES),
        choices=list(MODEL_NAMES),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--json-out",
        default=str(ROOT / "parity_results_micro.json"),
    )
    args = parser.parse_args()

    if Path(sys.executable).resolve() != MICRO_PYTHON.resolve():
        print(
            "WARNING: this script is intended to run with the micro env Python: "
            f"{MICRO_PYTHON}",
            file=sys.stderr,
        )

    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device(args.device)
    all_results = []

    for dataset_key in args.datasets:
        for model_name in args.models:
            print(f"[Run] dataset={dataset_key} model={model_name}")
            mm_result = run_case("MM", dataset_key, model_name, device)
            req_result = run_case("REQ", dataset_key, model_name, device)
            comparison = compare_cases(mm_result, req_result, args.atol)
            all_results.append(
                {
                    "dataset": dataset_key,
                    "model": model_name,
                    "passed": comparison["passed"],
                    "max_abs_diff": comparison["max_abs_diff"],
                    "mm": json_safe_result(mm_result),
                    "req": json_safe_result(req_result),
                    "diffs": comparison["diffs"],
                }
            )
            status = "PASS" if comparison["passed"] else "FAIL"
            print(
                f"[{status}] dataset={dataset_key} model={model_name} "
                f"max_abs_diff={comparison['max_abs_diff']}"
            )

    out_path = Path(args.json_out)
    out_path.write_text(json.dumps(all_results, indent=2, sort_keys=True))
    print(f"[Done] wrote {out_path}")
    if not all(item["passed"] for item in all_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
