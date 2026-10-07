#!/usr/bin/env python3
"""Rule-MMD: compare FactorBase rule-instance distributions of graph collections.

Every graph (reference or generated) goes through the same GraphVAE hard
cleanup used by ``scripts/evaluate_motif_count_distance.py`` (undirected,
no self-loops/isolates, largest connected component) and is then counted with
the training-time ``RelationalMotifCounter`` over the UNPRUNED ``_CP_smoothed``
state space of every rule.

Per graph G, rule r and state s:

    freq[r, s](G)  = count[r, s](G) / sum_s' count[r, s'](G)      (state block)
    density[r](G)  = log1p(sum_s count[r, s](G) / |V(G)|)          (density block)

The state block is the per-graph empirical distribution over the rule's
states, so it is independent of graph size and of collection size.  The
PRUNED view keeps only the state columns retained by motif=True training
(same denominators).  Rule-MMD is the squared MMD with a Gaussian kernel on
these vectors; the bandwidth is the median pairwise distance of the TRAIN
reference vectors and is shared by every method of a dataset.

Subcommands
-----------
select  reconstruct the training-time pruned selection -> selection.json
count   count one collection (train / test / generated bin) -> counts npz
score   compute Rule-MMD for all counted collections -> results json + md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

CAMPAIGN = Path(__file__).resolve().parent
REPO = CAMPAIGN / "source" / "GraphVAE-REQ"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

REFERENCE_CAP = 1000        # graphs per reference split (fixed random subset)
REFERENCE_SUBSET_SEED = 20260923
BANDWIDTH_SCALES = (0.5, 1.0, 2.0)


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


HUB_ROOT = "/local-scratch2/mirzaei/rule_mmd_20260923"


def load_inputs(dataset_dir: Path) -> Dict[str, Any]:
    """inputs.json with hub-absolute paths remapped to this worker's copy."""
    text = (dataset_dir / "inputs.json").read_text()
    if str(CAMPAIGN) != HUB_ROOT:
        text = text.replace(HUB_ROOT, str(CAMPAIGN))
    return json.loads(text)


def build_counter(dataset_dir: Path, device, view: str):
    """Counter for the 'unpruned' full state space or the 'pruned' training view.

    unpruned: rule_prune=False on ``motif_cache_unpruned_pkl`` when staged (QM9:
              training used a pre-pruned cache), else on ``motif_cache_pkl``.
    pruned:   ``motif_cache_pkl`` with the training run's own rule_prune value.
    """
    import torch  # noqa: F401
    from evaluate_motif_count_distance import load_yaml, make_counter_args
    from motif_counting.motif_counter import RelationalMotifCounter

    inputs = load_inputs(dataset_dir)
    _, flat = load_yaml(Path(inputs["effective_config"]))
    flat = dict(flat)
    if view == "unpruned":
        flat["rule_prune"] = False
        cache = inputs.get("motif_cache_unpruned_pkl") or inputs["motif_cache_pkl"]
    elif view == "pruned":
        cache = inputs["motif_cache_pkl"]
    else:
        raise ValueError(view)
    counter = RelationalMotifCounter(
        str(flat["database_name"]), make_counter_args(flat, Path(cache).parent, device)
    )
    return counter, flat, inputs


def state_key(counter, rule_index: int, row) -> Tuple[str, ...]:
    """Identify a state row by the values of its rule atoms only."""
    rule = counter.rules[rule_index]
    columns = counter.value_columns[rule_index]
    if all(atom in columns for atom in rule):
        return tuple(str(row[columns.index(atom)]) for atom in rule)
    return tuple(str(item) for item in row)  # non-FactorBase rows: exact match


def full_state_layout(counter) -> List[Dict[str, Any]]:
    layout = []
    for rule_index, rows in enumerate(counter.values):
        for value_index, row in enumerate(rows):
            layout.append(
                {
                    "column": len(layout),
                    "rule_index": rule_index,
                    "value_index": value_index,
                    "rule": [str(atom) for atom in counter.rules[rule_index]],
                    "state": list(state_key(counter, rule_index, row)),
                }
            )
    return layout


def load_cache(dataset_dir: Path):
    inputs = load_inputs(dataset_dir)
    with Path(inputs["dataset_cache_pkl"]).open("rb") as handle:
        return pickle.load(handle)


# ----------------------------------------------------------------------------
# select
# ----------------------------------------------------------------------------

def cmd_select(args) -> None:
    import torch
    from data import DataWrapper, merge_datasets
    from evaluate_motif_count_distance import prepare_training_motif_selection
    from motif_counting.motif_selection_manifest import (
        apply_motif_selection_manifest,
        load_motif_selection_manifest,
    )

    dataset_dir = args.dataset_dir.resolve()
    device = torch.device(args.device)
    full_counter, _, inputs = build_counter(dataset_dir, device, "unpruned")
    layout = full_state_layout(full_counter)
    key_to_column = {
        (entry["rule_index"], tuple(entry["state"])): entry["column"] for entry in layout
    }

    # ``flat`` is the training run's effective config (its own rule_prune).
    pruned_counter, flat, _ = build_counter(dataset_dir, device, "pruned")
    if getattr(args, "motif_batch_size", None):
        flat["motif_batch_size"] = int(args.motif_batch_size)
    if [list(r) for r in pruned_counter.rules] != [list(r) for r in full_counter.rules]:
        raise RuntimeError("Pruned and unpruned counters loaded different rule lists.")

    manifests = sorted((dataset_dir / "selection_manifests").glob("*.json"))
    selections: Dict[str, Dict[int, List[int]]] = {}
    summaries: Dict[str, Any] = {}
    if manifests:
        for manifest_path in manifests:
            counter, _, _ = build_counter(dataset_dir, device, "pruned")
            manifest = load_motif_selection_manifest(manifest_path)
            selected = apply_motif_selection_manifest(
                counter,
                manifest,
                database_name=str(flat["database_name"]),
                motif_cp_table_source=str(flat.get("motif_cp_table_source", "cp")),
                rule_prune=bool(flat.get("rule_prune", False)),
            )
            selections[manifest_path.stem] = (selected, counter)
            summaries[manifest_path.stem] = {"method": "saved_training_manifest",
                                             "manifest": str(manifest_path),
                                             "manifest_sha256": sha256(manifest_path)}
    else:
        cache = load_cache(dataset_dir)
        wrapper = DataWrapper(
            merge_datasets(cache["list_graphs"]),
            pruned_counter.relation_keys,
            cache.get("node_onehot_info"),
            edge_onehot_info=cache.get("edge_onehot_info"),
            edge_feature_info_mapping=pruned_counter.feature_info_mapping,
            device=str(device),
        )
        selected, summary = prepare_training_motif_selection(
            counter=pruned_counter, flat_config=flat, pruning_preprocessor=wrapper
        )
        selections["reconstructed"] = (selected, pruned_counter)
        summary["method"] = "reconstructed_from_training_split"
        summaries["reconstructed"] = summary

    column_sets = {}
    for name, (selected, counter) in selections.items():
        columns = []
        for rule_index in sorted(selected):
            for value_index in selected[rule_index]:
                key = (rule_index, state_key(counter, rule_index, counter.values[rule_index][value_index]))
                if key not in key_to_column:
                    raise RuntimeError(f"Pruned state {key} is absent from the unpruned space.")
                columns.append(key_to_column[key])
        column_sets[name] = sorted(set(columns))

    distinct = {tuple(v) for v in column_sets.values()}
    payload = {
        "database_name": flat["database_name"],
        "full_state_count": len(layout),
        "rule_count": len(full_counter.rules),
        "selections": {name: {"columns": cols, "count": len(cols), **summaries[name]}
                       for name, cols in column_sets.items()},
        "selections_identical": len(distinct) == 1,
        "pruned_columns": sorted(set().union(*[set(c) for c in column_sets.values()]))
        if len(distinct) > 1 else next(iter(distinct)),
        "layout": layout,
    }
    if len(distinct) > 1:
        payload["note"] = "Seeds used different pruned selections; pruned view uses their union."
    out = dataset_dir / "selection.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"[select] full={len(layout)} pruned={len(payload['pruned_columns'])} "
          f"identical={payload['selections_identical']} -> {out}")


# ----------------------------------------------------------------------------
# count
# ----------------------------------------------------------------------------

def reference_records(dataset_dir: Path, split: str, counter, device):
    """Exact-size records of a (subsampled) cache split, before hard cleanup."""
    from data import DataWrapper, merge_datasets
    from evaluate_motif_count_distance import graph_record_from_wrapper

    cache = load_cache(dataset_dir)
    dataset = cache["list_graphs" if split == "train" else "list_test_graphs"]
    total = len(dataset.list_adjs)
    rng = np.random.default_rng(REFERENCE_SUBSET_SEED + (0 if split == "train" else 1))
    indices = (np.arange(total) if total <= REFERENCE_CAP
               else np.sort(rng.choice(total, REFERENCE_CAP, replace=False)))
    merged = merge_datasets(dataset)
    merged = {key: [values[i] for i in indices] if isinstance(values, list) and len(values) == total else values
              for key, values in merged.items()}
    wrapper = DataWrapper(
        merged,
        counter.relation_keys,
        cache.get("node_onehot_info"),
        edge_onehot_info=cache.get("edge_onehot_info"),
        edge_feature_info_mapping=counter.feature_info_mapping,
        device="cpu",
    )
    node_counts = [int(dataset.list_adjs[i].shape[0]) for i in indices]
    records = [graph_record_from_wrapper(wrapper, position, n) for position, n in enumerate(node_counts)]
    return records, wrapper.feature_onehot_mapping, {"split_size": total, "indices": indices.tolist()}


def generated_records(path: Path, relation: str, node_dim: int, edge_onehot_info=None,
                      edge_feature_info_mapping=None):
    """Exact-size records of a staged DGL collection, packed like DataWrapper.

    Edge one-hots are split into the per-feature channel list the counter
    expects with the same helper DataWrapper uses for training graphs.
    """
    import dgl
    import torch
    from data import _split_edge_tensor_by_feature

    graphs, _ = dgl.load_graphs(str(path))
    records = []
    for graph in graphs:
        n = int(graph.num_nodes())
        attr = graph.ndata["attr"].float().cpu() if "attr" in graph.ndata else torch.ones(n, 1)
        if attr.shape[1] != node_dim:
            if node_dim >= 1 and attr.shape[1] == 1 and torch.all(attr == 1):
                # featureless collection: rules for these datasets use no node
                # attributes; reproduce the cache's constant one-hot layout.
                attr = torch.zeros(n, node_dim)
                attr[:, 0] = 1.0
            else:
                raise ValueError(f"{path}: node attr dim {attr.shape[1]} != cache dim {node_dim}")
        source, target = graph.edges()
        source, target = source.long(), target.long()
        adjacency = torch.zeros((n, n), dtype=torch.float32)
        adjacency[source, target] = 1.0
        edge = None
        if "attr" in graph.edata:
            rows = graph.edata["attr"].float().cpu()
            packed = torch.zeros((rows.shape[1], n, n), dtype=torch.float32)
            packed[:, source, target] = rows.transpose(0, 1)
            edge = [part[0] for part in _split_edge_tensor_by_feature(
                packed.unsqueeze(0), edge_onehot_info=edge_onehot_info,
                edge_feature_info_mapping=edge_feature_info_mapping)]
        records.append({
            "features": attr.argmax(dim=1, keepdim=True).float() + 1,
            "feat_onehot": attr,
            "adj": {relation: adjacency},
            "edge": edge,
        })
    return records


def cmd_count(args) -> None:
    import torch
    from evaluate_motif_count_distance import count_exact_graph_records, hard_graph_postprocess

    dataset_dir = args.dataset_dir.resolve()
    device = torch.device(args.device)
    counter, flat, inputs = build_counter(dataset_dir, device, "unpruned")
    started = time.time()
    meta: Dict[str, Any] = {"collection": args.name, "device": args.device}
    if args.name in ("train", "test"):
        records, mapping, extra = reference_records(dataset_dir, args.name, counter, device)
        meta.update(extra)
    else:
        path = Path(args.generated).resolve()
        cache = load_cache(dataset_dir)
        _, mapping, _ = reference_records_mapping(cache, counter)
        node_dim = mapping_node_dim(cache)
        records = generated_records(path, counter.relation_keys[0], node_dim, cache.get("edge_onehot_info"),
                                    counter.feature_info_mapping)
        meta.update({"source": str(path), "source_sha256": sha256(path)})

    import scipy.sparse as sp

    # Optional sharding: this job counts graphs shard, shard+n, shard+2n, ...
    shard, shards = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    positions = list(range(shard, len(records), shards))
    records = [records[i] for i in positions]
    hard = [hard_graph_postprocess(record) for record in records]
    del records
    node_counts = np.array([int(r["features"].shape[0]) for r in hard])
    # Count in chunks and keep only nonzero entries: an unpruned state space
    # can have millions of columns (AIDS: 1.58M), too large to hold densely.
    blocks = []
    for start in range(0, len(hard), args.chunk_size):
        chunk = count_exact_graph_records(counter, hard[start:start + args.chunk_size], mapping,
                                          args.batch_size, device).numpy()
        blocks.append(sp.csr_matrix(chunk))
        print(f"[count] {args.name}: {min(start + args.chunk_size, len(hard))}/{len(hard)} graphs "
              f"({time.time() - started:.0f}s)", flush=True)
    width = sum(len(rows) for rows in counter.values)
    counts = sp.vstack(blocks).tocsr() if blocks else sp.csr_matrix((0, width))
    meta.update({"graph_count": len(hard), "empty_after_cleanup": int((node_counts == 0).sum()),
                 "seconds": round(time.time() - started, 1), "state_count": counts.shape[1],
                 "shard": [shard, shards], "positions": positions})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, data=counts.data, indices=counts.indices, indptr=counts.indptr,
                        shape=np.array(counts.shape), node_counts=node_counts, meta=json.dumps(meta))
    print(f"[count] {args.name}: {counts.shape} nnz={counts.nnz} in {meta['seconds']}s -> {out}")


def collection_plan(dataset_dir: Path):
    """(name, generated path or None) for train, test and every staged item."""
    inputs = load_inputs(dataset_dir)
    plan = [("train", None), ("test", None)]
    for item in inputs["items"]:
        plan.append((f"{item['method']}_seed_{item['seed']}", item["path"]))
    return plan


def cmd_count_all(args) -> None:
    """Count every collection of a dataset, sharded by node count.

    The counter iterates over all state rows once per node-count bucket, so
    for huge state spaces the cost is (#buckets x #rows). Grouping equal sizes
    across ALL collections and assigning whole buckets to shards minimizes
    passes. Each shard writes <name>.shard<i>of<n>.npz for every collection
    (possibly empty), which ``load_counts`` merges.
    """
    import scipy.sparse as sp
    import torch
    from evaluate_motif_count_distance import count_exact_graph_records, hard_graph_postprocess

    dataset_dir = args.dataset_dir.resolve()
    device = torch.device(args.device)
    shard, shards = (int(x) for x in args.shard.split("/"))
    counter, flat, inputs = build_counter(dataset_dir, device, "unpruned")
    cache = load_cache(dataset_dir)
    _, mapping, _ = reference_records_mapping(cache, counter)
    node_dim = mapping_node_dim(cache)
    started = time.time()

    collections = {}
    for name, path in collection_plan(dataset_dir):
        meta: Dict[str, Any] = {"collection": name, "device": args.device}
        if path is None:
            records, _, extra = reference_records(dataset_dir, name, counter, device)
            meta.update(extra)
        else:
            records = generated_records(Path(path), counter.relation_keys[0], node_dim,
                                        cache.get("edge_onehot_info"), counter.feature_info_mapping)
            meta.update({"source": str(Path(path)), "source_sha256": sha256(Path(path))})
        hard = [hard_graph_postprocess(record) for record in records]
        for record in hard:
            # Uniform scalar column so cache and generated graphs stack in one
            # batch; the counter reads node attributes from feat_onehot.
            onehot = record["feat_onehot"]
            record["features"] = (onehot.argmax(dim=1, keepdim=True).float() + 1
                                  if onehot.shape[0] else onehot[:, :1])
        collections[name] = (hard, meta)
    print(f"[count-all] loaded {sum(len(h) for h, _ in collections.values())} graphs "
          f"in {len(collections)} collections ({time.time() - started:.0f}s)", flush=True)

    buckets: Dict[int, List[Tuple[str, int]]] = {}
    for name, (hard, _) in collections.items():
        for position, record in enumerate(hard):
            buckets.setdefault(int(record["features"].shape[0]), []).append((name, position))
    # Deterministic greedy assignment of whole buckets, costed in counter batches.
    cost = {size: -(-len(members) // args.batch_size) for size, members in buckets.items() if size > 0}
    load = [0] * shards
    owner = {}
    for size in sorted(cost, key=lambda s: (-cost[s], s)):
        target = min(range(shards), key=lambda i: (load[i], i))
        owner[size] = target
        load[target] += cost[size]
    mine = sorted(size for size, o in owner.items() if o == shard)
    if shard == 0:
        mine = [0] + mine if 0 in buckets else mine  # empty graphs: zero rows, no counting
    print(f"[count-all] shard {shard}/{shards}: {len(mine)} buckets, cost {load[shard]} batches "
          f"(total {sum(load)})", flush=True)

    width = sum(len(rows) for rows in counter.values)
    partial_dir = Path(args.output_dir) / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    rows: Dict[str, List[Tuple[int, Any]]] = {name: [] for name in collections}
    for done, size in enumerate(mine, 1):
        members = buckets[size]
        partial = partial_dir / f"bucket_{size}.npz"
        if partial.exists():  # resume: bucket already counted by an earlier attempt
            data = np.load(partial)
            block = sp.csr_matrix((data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"]))
        elif size == 0:
            block = sp.csr_matrix((len(members), width))
        else:
            # One counter batch per call: the counter returns a dense
            # (graphs x states) block, ~0.8 GB per 128 graphs for AIDS.
            records = [collections[name][0][position] for name, position in members]
            parts = []
            for start in range(0, len(records), args.batch_size):
                chunk = count_exact_graph_records(counter, records[start:start + args.batch_size], mapping,
                                                  args.batch_size, device)
                parts.append(sp.csr_matrix(chunk.numpy()))
                del chunk
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            block = sp.vstack(parts).tocsr()
            np.savez(partial, data=block.data, indices=block.indices, indptr=block.indptr,
                     shape=np.array(block.shape))
        for row, (name, position) in enumerate(members):
            rows[name].append((position, block[row]))
        print(f"[count-all] bucket n={size} ({len(members)} graphs) {done}/{len(mine)} "
              f"({time.time() - started:.0f}s)", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (hard, meta) in collections.items():
        entries = sorted(rows[name], key=lambda item: item[0])
        positions = [position for position, _ in entries]
        matrix = sp.vstack([vector for _, vector in entries]).tocsr() if entries else sp.csr_matrix((0, width))
        node_counts = np.array([int(hard[p]["features"].shape[0]) for p in positions], dtype=int)
        meta = dict(meta, graph_count=len(positions), state_count=width, shard=[shard, shards],
                    positions=positions, seconds=round(time.time() - started, 1),
                    empty_after_cleanup=int((node_counts == 0).sum()))
        np.savez_compressed(out_dir / f"{name}.shard{shard}of{shards}.npz", data=matrix.data,
                            indices=matrix.indices, indptr=matrix.indptr, shape=np.array(matrix.shape),
                            node_counts=node_counts, meta=json.dumps(meta))
    print(f"[count-all] shard {shard}/{shards} done in {time.time() - started:.0f}s", flush=True)


def load_counts(count_dir: Path):
    """Load <name>.npz or merge <name>.shard<i>of<n>.npz files -> {name: (csr, nodes, meta)}."""
    import scipy.sparse as sp

    parts: Dict[str, list] = {}
    for path in sorted(count_dir.glob("*.npz")):
        name = path.stem.split(".shard")[0]
        data = np.load(path, allow_pickle=False)
        if "counts" in data:  # dense legacy format
            matrix = sp.csr_matrix(data["counts"])
        else:
            matrix = sp.csr_matrix((data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"]))
        meta = json.loads(str(data["meta"]))
        parts.setdefault(name, []).append((meta, matrix, data["node_counts"]))
    loaded = {}
    for name, items in parts.items():
        shards = {tuple(m.get("shard", [0, 1])) for m, _, _ in items}
        expected = items[0][0].get("shard", [0, 1])[1]
        if len(shards) != expected:
            print(f"[score] skipping {name}: {len(shards)}/{expected} shards present")
            continue
        order = np.concatenate([m.get("positions", list(range(x.shape[0]))) for m, x, _ in items]).argsort()
        matrix = sp.vstack([x for _, x, _ in items]).tocsr()[order]
        nodes = np.concatenate([n for _, _, n in items])[order]
        meta = dict(items[0][0])
        meta.update({"graph_count": int(matrix.shape[0]),
                     "seconds": float(sum(m["seconds"] for m, _, _ in items)), "shards": expected})
        meta.pop("positions", None)
        loaded[name] = (matrix, nodes, meta)
    return loaded


def mapping_node_dim(cache) -> int:
    info = cache.get("node_onehot_info") or {}
    if info:
        return len(info)
    first = next((x for x in cache["list_graphs"].processed_node_onehot if x is not None), None)
    return int(first.shape[-1]) if first is not None else 1


def reference_records_mapping(cache, counter):
    """feature_onehot_mapping exactly as DataWrapper derives it for training."""
    from data import DataWrapper, merge_datasets

    dataset = cache["list_graphs"]
    merged = merge_datasets(dataset)
    total = len(dataset.list_adjs)
    merged = {key: values[:1] if isinstance(values, list) and len(values) == total else values
              for key, values in merged.items()}
    wrapper = DataWrapper(
        merged,
        counter.relation_keys,
        cache.get("node_onehot_info"),
        edge_onehot_info=cache.get("edge_onehot_info"),
        edge_feature_info_mapping=counter.feature_info_mapping,
        device="cpu",
    )
    return None, wrapper.feature_onehot_mapping, None


# ----------------------------------------------------------------------------
# score
# ----------------------------------------------------------------------------

def features(counts, node_counts: np.ndarray, layout, columns=None):
    """Sparse state-frequency block (csr) and dense log-density block."""
    import scipy.sparse as sp

    counts = sp.csr_matrix(counts, dtype=np.float64)
    rule_of_column = np.array([entry["rule_index"] for entry in layout])
    rules = sorted(set(rule_of_column.tolist()))
    rule_position = np.searchsorted(rules, rule_of_column)
    indicator = sp.csr_matrix((np.ones(len(rule_of_column)), (np.arange(len(rule_of_column)), rule_position)),
                              shape=(len(rule_of_column), len(rules)))
    totals = np.asarray((counts @ indicator).todense())               # graphs x rules
    freq = counts.copy()
    rows = np.repeat(np.arange(freq.shape[0]), np.diff(freq.indptr))
    denominators = totals[rows, rule_position[freq.indices]]
    freq.data = np.where(denominators > 0, freq.data / np.where(denominators > 0, denominators, 1), 0.0)
    freq.eliminate_zeros()
    density = np.log1p(totals / np.maximum(node_counts, 1)[:, None])
    if columns is not None:
        freq = freq[:, columns]
        kept_rules = sorted(set(rule_of_column[columns].tolist()))
        density = density[:, [rules.index(r) for r in kept_rules]]
    return freq.tocsr(), density


def sq_dists(a, b) -> np.ndarray:
    """Pairwise squared Euclidean distances for dense arrays or scipy sparse matrices."""
    import scipy.sparse as sp

    if sp.issparse(a) or sp.issparse(b):
        a, b = sp.csr_matrix(a), sp.csr_matrix(b)
        na = np.asarray(a.multiply(a).sum(1)).ravel()
        nb = np.asarray(b.multiply(b).sum(1)).ravel()
        cross = np.asarray((a @ b.T).todense())
    else:
        na, nb, cross = (a * a).sum(1), (b * b).sum(1), a @ b.T
    return np.maximum(na[:, None] + nb[None, :] - 2 * cross, 0.0)


def median_bandwidth(x: np.ndarray) -> float:
    d = np.sqrt(sq_dists(x, x))
    upper = d[np.triu_indices(x.shape[0], k=1)]
    # Identical graphs give distance 0 up to rounding noise (~1e-8); drop them
    # with a relative tolerance so dense and sparse arithmetic agree.
    upper = upper[upper > 1e-6 * upper.max()] if upper.size and upper.max() > 0 else upper[:0]
    return float(np.median(upper)) if upper.size else 1.0


def mmd2(x: np.ndarray, y: np.ndarray, sigma: float) -> Dict[str, float]:
    gamma = 1.0 / (2.0 * sigma * sigma)
    kxx, kyy, kxy = (np.exp(-gamma * sq_dists(x, x)), np.exp(-gamma * sq_dists(y, y)),
                     np.exp(-gamma * sq_dists(x, y)))
    m, n = x.shape[0], y.shape[0]
    biased = kxx.mean() + kyy.mean() - 2 * kxy.mean()
    unbiased = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                - 2 * kxy.mean()) if m > 1 and n > 1 else float("nan")
    return {"mmd2": float(max(biased, 0.0)), "mmd2_unbiased": float(unbiased)}


def cmd_score(args) -> None:
    dataset_dir = args.dataset_dir.resolve()
    inputs = load_inputs(dataset_dir)
    selection = json.loads((dataset_dir / "selection.json").read_text())
    layout = selection["layout"]
    count_dir = dataset_dir / "counts"
    loaded = load_counts(count_dir)
    if "train" not in loaded or "test" not in loaded:
        raise RuntimeError("train and test counts are required")

    views = {"unpruned": None, "pruned": np.array(selection["pruned_columns"], dtype=int)}
    results: Dict[str, Any] = {"dataset": inputs["dataset"], "database_name": selection["database_name"],
                               "full_state_count": selection["full_state_count"],
                               "pruned_state_count": len(selection["pruned_columns"]),
                               "collections": {k: v[2] for k, v in loaded.items()}, "views": {}}
    for view, columns in views.items():
        feats = {}
        for name, (counts, nodes, _) in loaded.items():
            keep = nodes > 0
            feats[name] = tuple(block[np.flatnonzero(keep)] for block in features(counts, nodes, layout, columns))
        view_out = {}
        for block_index, block in enumerate(("state", "density")):
            train_block = feats["train"][block_index]
            sigma = median_bandwidth(train_block)
            block_out = {"bandwidth": sigma, "scores": {}}
            for name in loaded:
                for ref in ("train", "test"):
                    if name == ref:
                        continue
                    x = feats[name][block_index]
                    y = feats[ref][block_index]
                    entry = mmd2(x, y, sigma)
                    entry["multiscale_mmd2"] = float(np.mean([mmd2(x, y, sigma * s)["mmd2"]
                                                              for s in BANDWIDTH_SCALES]))
                    block_out["scores"].setdefault(name, {})[ref] = entry
            view_out[block] = block_out
        results["views"][view] = view_out

    out = dataset_dir / "rule_mmd_results.json"
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"[score] -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "count", "count-all", "score"):
        p = sub.add_parser(name)
        p.add_argument("--dataset-dir", type=Path, required=True)
        if name != "score":
            p.add_argument("--device", default="cpu")
        if name == "select":
            p.add_argument("--motif-batch-size", type=int, default=None,
                           help="override graphs per pruning count batch (selection is batch-size invariant)")
        if name == "count-all":
            p.add_argument("--shard", required=True, help="i/n")
            p.add_argument("--output-dir", required=True)
            p.add_argument("--batch-size", type=int, default=128)
        if name == "count":
            p.add_argument("--name", required=True, help="train | test | <method>_seed_<k>")
            p.add_argument("--generated", default=None)
            p.add_argument("--output", required=True)
            p.add_argument("--batch-size", type=int, default=16)
            p.add_argument("--chunk-size", type=int, default=64, help="graphs counted per dense block")
            p.add_argument("--shard", default=None, help="i/n: count every n-th graph starting at i")
    args = parser.parse_args()
    {"select": cmd_select, "count": cmd_count, "count-all": cmd_count_all,
     "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    main()
