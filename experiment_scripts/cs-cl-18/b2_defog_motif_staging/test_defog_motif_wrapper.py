"""B2.1 adapter tests. Run from this directory:

    PYTHONPATH=$GRAPHVAE_REQ_ROOT python -m pytest -q test_defog_motif_wrapper.py

Uses the AIDS motif cache the D29 10k-epoch campaign used (sha 029edaac...)
and GraphVAE-REQ's AIDS max40 dataset cache, both read-only.
"""

import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

GRAPHVAE_REQ = Path(os.environ.get("GRAPHVAE_REQ_ROOT", "/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ"))
sys.path.insert(0, str(GRAPHVAE_REQ))

from data import _split_edge_tensor_by_feature  # noqa: E402
from motif_counting import motif_counter as counter_module  # noqa: E402
from motif_counting.motif_counter import RelationalMotifCounter  # noqa: E402

from defog_motif_wrapper import (  # noqa: E402
    DeFoGMotifPreprocessor,
    MotifAdapterError,
    assert_counter_digest,
    assert_feature_schema,
    build_edge_channel_groups,
    build_node_column_index,
    counter_digest,
)

MOTIF_CACHE_DIR = "/local-scratch2/mirzaei/aids_common_eval_10k_20260917/cache/motif"
DATASET_CACHE = GRAPHVAE_REQ / (
    "cache_datasets/AIDS_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_"
    "loaderseed-0_bfs-all_components_features-tu-quantile8-max40.pkl"
)
DEFOG_AIDS_META = (
    "/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG/data/"
    "aids_3seed_20260912/raw/aids_node_channels_v2_compact.pt"
)
N_MAX = 40
N_GRAPHS = 12
AIDS_EDGE_CHANNEL_OF = {("edge_label", v): v + 1 for v in range(3)}  # DeFoG: 0 = no edge


@pytest.fixture(scope="module")
def counter():
    args = SimpleNamespace(
        motif_cache_dir=MOTIF_CACHE_DIR, motif_cp_table_source="cp",
        use_syntactic_literal_rules=False, device="cpu", rule_prune=False,
    )
    return RelationalMotifCounter("aids_undir_feat_multi", args)


@pytest.fixture(scope="module")
def gvae_cache():
    with open(DATASET_CACHE, "rb") as stream:
        return pickle.load(stream)


@pytest.fixture(scope="module")
def defog_layout():
    meta = torch.load(DEFOG_AIDS_META)
    names = meta["node_channel_names"]
    values = [list(range(meta["node_channel_dims"][0]))] + [
        [int(v) for v in states] for states in meta["attribute_observed_states"]
    ]
    return names, values


@pytest.fixture(scope="module")
def node_cols(gvae_cache, defog_layout):
    return build_node_column_index(gvae_cache["node_onehot_info"], *defog_layout)


@pytest.fixture(scope="module")
def fom(gvae_cache):
    from data import _build_fom
    return _build_fom(gvae_cache["node_onehot_info"])


@pytest.fixture(scope="module")
def real_batch(gvae_cache):
    """First N_GRAPHS AIDS training graphs, padded as DataWrapper pads but with a
    ZERO diagonal. GraphVAE's processed adjacency is A+I including padded nodes
    (Datasets.process setdiag(1)); DeFoG has no self-loops. The adapter's
    declared convention is zero-diagonal — see REGISTER D31."""
    adjs = gvae_cache["list_adj"][:N_GRAPHS]
    nohs = gvae_cache["list_noh_train"][:N_GRAPHS]
    eohs = gvae_cache["list_eoh_train"][:N_GRAPHS]
    B, D = len(adjs), nohs[0].shape[1]
    adj = torch.zeros(B, N_MAX, N_MAX)
    noh = torch.zeros(B, N_MAX, D)
    eoh = torch.zeros(B, 3, N_MAX, N_MAX)
    mask = torch.zeros(B, N_MAX, dtype=torch.bool)
    for g, (a, x, e) in enumerate(zip(adjs, nohs, eohs)):
        n = a.shape[0]
        dense = a.toarray().astype(np.float32)
        np.fill_diagonal(dense, 0.0)
        adj[g, :n, :n] = torch.tensor(dense)
        noh[g, :n] = torch.tensor(x, dtype=torch.float32)
        eoh[g, :, :n, :n] = torch.tensor(e, dtype=torch.float32)
        mask[g, :n] = True
    return adj, noh, eoh, mask


class _Reference:
    """GraphVAE-REQ's own observed-data input path (DataWrapper semantics)."""

    def __init__(self, adj, noh, eoh, relation_keys, fom, edge_onehot_info, fim):
        self.num_graphs, self.N_max = adj.shape[0], adj.shape[1]
        self.feature_onehot_mapping = fom
        self._adj = {rk: adj for rk in relation_keys}
        self._edge = _split_edge_tensor_by_feature(eoh, edge_onehot_info, fim)
        self._noh = noh

    def get_batch(self, s, e):
        return (self._noh[s:e], self._noh[s:e], {k: v[s:e] for k, v in self._adj.items()},
                [x[s:e] for x in self._edge])


def _to_defog_onehot(adj, noh, eoh, mask, node_cols, dx):
    """Re-encode the same graphs the way DeFoG's dense batch holds them."""
    B, n = adj.shape[:2]
    X = torch.zeros(B, n, dx)
    X[..., node_cols] = noh
    E = torch.zeros(B, n, n, 4)
    real = (mask.unsqueeze(2) & mask.unsqueeze(1)).float()
    E[..., 0] = (1 - adj) * real          # padded pairs stay all-zero, as in DeFoG
    for k in range(3):
        E[..., k + 1] = eoh[:, k] * adj
    return X, E


def _adapter(counter, X, E, mask, fom, node_cols):
    return DeFoGMotifPreprocessor(
        X, E, mask,
        counter_relation_keys=counter.relation_keys,
        relation_channels={"edges": [1, 2, 3]},
        feature_onehot_mapping=fom,
        node_column_index=node_cols,
        edge_channel_groups=build_edge_channel_groups(counter.feature_info_mapping, AIDS_EDGE_CHANNEL_OF),
    )


@pytest.fixture(scope="module")
def reference_counts(counter, real_batch, fom, gvae_cache):
    adj, noh, eoh, _ = real_batch
    ref = _Reference(adj, noh, eoh, counter.relation_keys, fom,
                     gvae_cache["edge_onehot_info"], counter.feature_info_mapping)
    return counter.count_batch(ref, output_mode="total_count")


# ── the key test ───────────────────────────────────────────────────────────

def test_real_aids_counts_match_graphvae_path(counter, real_batch, fom, node_cols, reference_counts):
    adj, noh, eoh, mask = real_batch
    X, E = _to_defog_onehot(adj, noh, eoh, mask, node_cols, dx=59)
    got = counter.count_batch(_adapter(counter, X, E, mask, fom, node_cols), output_mode="total_count")
    assert got.shape == reference_counts.shape
    assert reference_counts.abs().sum() > 0, "reference counted nothing — test would be vacuous"
    torch.testing.assert_close(got, reference_counts, rtol=0, atol=1e-4)


# ── spec §2.4 ─────────────────────────────────────────────────────────────

def test_padding_garbage_does_not_change_counts(counter, real_batch, fom, node_cols, reference_counts):
    adj, noh, eoh, mask = real_batch
    X, E = _to_defog_onehot(adj, noh, eoh, mask, node_cols, dx=59)
    g = torch.Generator().manual_seed(0)
    pad_nodes = ~mask
    pad_pairs = ~(mask.unsqueeze(2) & mask.unsqueeze(1))
    X[pad_nodes] = torch.rand(int(pad_nodes.sum()), X.shape[-1], generator=g)
    E[pad_pairs] = torch.rand(int(pad_pairs.sum()), 4, generator=g)
    got = counter.count_batch(_adapter(counter, X, E, mask, fom, node_cols), output_mode="total_count")
    torch.testing.assert_close(got, reference_counts, rtol=0, atol=1e-4)


def test_edges_relation_is_zero_on_padded_pairs_even_when_E_is_all_zero(counter, fom, node_cols):
    B, n = 2, 6
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    X = torch.zeros(B, n, 59)
    E = torch.zeros(B, n, n, 4)                 # 1 - E0 would be 1 everywhere without the mask
    prep = _adapter(counter, X, E, mask, fom, node_cols)
    _, _, adj_b, _ = prep.get_batch(0, B)
    assert adj_b["edges"].abs().sum() == 0


def test_symmetrised_and_asymmetry_recorded(counter, fom, node_cols):
    B, n = 1, 5
    E = torch.softmax(torch.randn(B, n, n, 4, generator=torch.Generator().manual_seed(1)), -1)
    prep = _adapter(counter, torch.zeros(B, n, 59), E, torch.ones(B, n, dtype=torch.bool), fom, node_cols)
    A = prep.get_batch(0, 1)[2]["edges"]
    torch.testing.assert_close(A, A.transpose(1, 2))
    assert prep.max_asymmetry > 0


def test_diagonal_zeroed(counter, fom, node_cols):
    B, n = 1, 5
    E = torch.zeros(B, n, n, 4)
    E[..., 1] = 1.0
    prep = _adapter(counter, torch.zeros(B, n, 59), E, torch.ones(B, n, dtype=torch.bool), fom, node_cols)
    _, _, adj_b, edge_b = prep.get_batch(0, 1)
    assert torch.diagonal(adj_b["edges"], dim1=1, dim2=2).abs().sum() == 0
    assert torch.diagonal(edge_b[0], dim1=2, dim2=3).abs().sum() == 0


def test_relation_name_mismatch_fails_closed(counter, fom, node_cols):
    with pytest.raises(MotifAdapterError, match="relation names"):
        DeFoGMotifPreprocessor(
            torch.zeros(1, 3, 59), torch.zeros(1, 3, 3, 4), torch.ones(1, 3, dtype=torch.bool),
            counter_relation_keys=counter.relation_keys, relation_channels={"bonds": [1, 2, 3]},
            feature_onehot_mapping=fom, node_column_index=node_cols,
        )


def test_edge_value_without_channel_fails_closed(counter):
    with pytest.raises(MotifAdapterError, match="no DeFoG channel"):
        build_edge_channel_groups(counter.feature_info_mapping, {("edge_label", 0): 1, ("edge_label", 1): 2})


def test_node_column_missing_fails_closed(gvae_cache, defog_layout):
    names, values = defog_layout
    trimmed = [list(v) for v in values]
    trimmed[0] = [v for v in trimmed[0] if v != 0]
    with pytest.raises(MotifAdapterError, match="absent from DeFoG layout"):
        build_node_column_index(gvae_cache["node_onehot_info"], names, trimmed)


def test_counter_digest_assertion(counter):
    digest = counter_digest(counter)
    assert digest == counter_digest(counter_module)
    assert assert_counter_digest(counter, digest) == digest
    with pytest.raises(MotifAdapterError, match="digest"):
        assert_counter_digest(counter, "0" * 64)


def test_graphvae_self_loop_convention_is_what_differs(counter, real_batch, fom, gvae_cache, reference_counts):
    """Documents D31: with A+I (GraphVAE's processed convention) counts change;
    the adapter deliberately does not reproduce that."""
    adj, noh, eoh, _ = real_batch
    with_loops = adj.clone()
    idx = torch.arange(adj.shape[1])
    with_loops[:, idx, idx] = 1.0
    ref = _Reference(with_loops, noh, eoh, counter.relation_keys, fom,
                     gvae_cache["edge_onehot_info"], counter.feature_info_mapping)
    looped = counter.count_batch(ref, output_mode="total_count")
    assert not torch.allclose(looped, reference_counts)


def test_feature_schema_guard_catches_aids_binning_mismatch(gvae_cache):
    assert_feature_schema("tu-quantile8-max40", gvae_cache["cache_metadata"]["feature_schema"])
    with pytest.raises(MotifAdapterError, match="schema mismatch"):
        assert_feature_schema(gvae_cache["cache_metadata"]["feature_schema"], "tu-quantile8-maxall")


# ── soft inputs: the training-time case ───────────────────────────────────

def test_soft_semantics_recover_joint_and_gradients_are_finite(counter, fom, node_cols):
    B, n = 3, 8
    g = torch.Generator().manual_seed(2)
    logits_E = torch.randn(B, n, n, 4, generator=g)
    logits_E[0, 0, 1, 0] = logits_E[0, 1, 0, 0] = 60.0   # a pair with ~all mass on no-edge
    logits_E.requires_grad_(True)
    logits_X = torch.randn(B, n, 59, generator=g, requires_grad=True)
    E = torch.softmax(logits_E, -1)
    X = torch.softmax(logits_X, -1)
    mask = torch.ones(B, n, dtype=torch.bool)
    mask[2, 6:] = False
    prep = _adapter(counter, X, E, mask, fom, node_cols)

    _, _, adj_b, edge_b = prep.get_batch(0, B)
    Es = 0.5 * (E + E.transpose(1, 2))
    off = (1 - torch.eye(n)) * (mask.unsqueeze(2) & mask.unsqueeze(1)).float()
    for k in range(3):
        torch.testing.assert_close(adj_b["edges"] * edge_b[0][:, k], Es[..., k + 1] * off, atol=1e-6, rtol=1e-5)

    counts = counter.count_batch(prep, output_mode="total_count")
    assert torch.isfinite(counts).all()
    counts.sum().backward()
    assert torch.isfinite(logits_E.grad).all() and torch.isfinite(logits_X.grad).all()
    assert logits_E.grad.abs().sum() > 0
