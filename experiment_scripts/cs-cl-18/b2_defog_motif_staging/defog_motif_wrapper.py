"""Duck-typed GraphVAE-REQ motif preprocessor over DeFoG's predicted clean graph.

Staging copy for B2.1. Destination: ``DeFoG/src/motif/defog_motif_wrapper.py``.
The counter is imported from GraphVAE-REQ by path, never vendored (spec §2.3).

Semantics (chosen to equal GraphVAE-REQ's ReconstructedDataWrapper, so the two
families feed the counter the same quantities):

* relation matrix  A_r[i, j] = sum_{c in channels(r)} E[i, j, c]
  (for a single "edges" relation this is 1 - E[..., no_edge]);
* edge feature     T_f[k, i, j] = E[i, j, c_k] / max(sum_{c in group f} E[i, j, c], eps)
  i.e. P(type | edge). GraphVAE multiplies this by A inside the counter, so
  A * T recovers the joint E[..., c_k]. Passing the joint directly would count
  the edge probability twice;
* node predicate   columns of the per-channel-softmaxed pred.X, re-indexed into
  GraphVAE's one-hot layout through an explicit (feature, value) map.

Padded slots, the diagonal, and pairs touching a padded node are zeroed before
counting (spec §2.4.1, §2.4.3). E is symmetrised as (E + E^T)/2 (§2.4.2).
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

COUNTER_EPS = 1e-6


class MotifAdapterError(RuntimeError):
    pass


def counter_digest(counter_or_module) -> str:
    """SHA-256 of the source file defining the counter (spec §2.4.5)."""
    target = counter_or_module
    if not inspect.ismodule(target) and not inspect.isclass(target):
        target = type(target)
    path = inspect.getsourcefile(target)
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def assert_counter_digest(counter_or_module, expected_sha256: str) -> str:
    actual = counter_digest(counter_or_module)
    if actual != expected_sha256:
        raise MotifAdapterError(
            f"motif counter digest {actual} does not match the paired "
            f"GraphVAE-REQ run's {expected_sha256}; the two families would not "
            "share one counting implementation"
        )
    return actual


def build_node_column_index(
    graphvae_node_onehot_info: Mapping[int, Mapping],
    defog_channel_names: Sequence[str],
    defog_channel_values: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Map every GraphVAE one-hot column to one flat DeFoG pred.X column.

    ``defog_channel_values[c][k]`` is the raw feature value DeFoG encodes at
    compact index ``k`` of channel ``c``. Fails closed on any GraphVAE column
    with no DeFoG counterpart. Matching value labels does NOT prove matching
    bin semantics — pair this with ``assert_feature_schema``.
    """
    offsets, lookup = {}, {}
    offset = 0
    for name, values in zip(defog_channel_names, defog_channel_values):
        offsets[name] = offset
        lookup[name] = {int(v): k for k, v in enumerate(values)}
        offset += len(values)

    columns, missing = [], []
    for oh_col in sorted(graphvae_node_onehot_info):
        meta = graphvae_node_onehot_info[oh_col]
        name, value = meta["feature_name"], int(meta["value"])
        if name not in lookup or value not in lookup[name]:
            missing.append((oh_col, name, value))
            continue
        columns.append(offsets[name] + lookup[name][value])
    if missing:
        raise MotifAdapterError(f"GraphVAE node columns absent from DeFoG layout: {missing}")
    return torch.tensor(columns, dtype=torch.long)


def assert_feature_schema(graphvae_schema: str, defog_schema: str) -> None:
    """Same labels can mean different bins (AIDS max40 vs maxall); fail closed."""
    if graphvae_schema != defog_schema:
        raise MotifAdapterError(
            f"feature schema mismatch: GraphVAE cache '{graphvae_schema}' vs "
            f"DeFoG data '{defog_schema}'. Train DeFoG on the GraphVAE export "
            "or rebuild the motif cache on DeFoG's binning."
        )


def build_edge_channel_groups(
    counter_feature_info_mapping: Mapping[int, Mapping],
    defog_edge_channel_of: Mapping[Tuple[str, int], int],
) -> List[List[int]]:
    """For each counter edge feature (in feature_idx order), the DeFoG E
    channels in the counter's value-index order."""
    groups, missing = [], []
    for feature_idx in sorted(counter_feature_info_mapping):
        info = counter_feature_info_mapping[feature_idx]
        name = info["feature_name"]
        value_by_index = info["value_index_mapping"]
        group = []
        for value_index in sorted(value_by_index):
            value = int(value_by_index[value_index])
            channel = defog_edge_channel_of.get((name, value))
            if channel is None:
                missing.append((name, value))
            group.append(channel)
        groups.append(group)
    if missing:
        raise MotifAdapterError(f"counter edge values with no DeFoG channel: {missing}")
    return groups


class DeFoGMotifPreprocessor:
    """Exposes num_graphs, N_max, feature_onehot_mapping, get_batch()."""

    def __init__(
        self,
        pred_X: torch.Tensor,
        pred_E: torch.Tensor,
        node_mask: torch.Tensor,
        *,
        counter_relation_keys: Sequence[str],
        relation_channels: Mapping[str, Sequence[int]],
        feature_onehot_mapping: Dict,
        node_column_index: torch.Tensor,
        edge_channel_groups: Optional[List[List[int]]] = None,
        eps: float = COUNTER_EPS,
    ):
        if pred_E.dim() != 4 or pred_X.dim() != 3:
            raise MotifAdapterError(
                f"expected pred_X (B,n,dx) and pred_E (B,n,n,de); got "
                f"{tuple(pred_X.shape)} and {tuple(pred_E.shape)}"
            )
        expected, provided = set(counter_relation_keys), set(relation_channels)
        if expected != provided:
            raise MotifAdapterError(
                f"relation names differ from the motif cache (would count zero): "
                f"cache={sorted(expected)} adapter={sorted(provided)}"
            )

        B, n = pred_X.shape[:2]
        self.num_graphs = B
        self.N_max = n
        self.feature_onehot_mapping = feature_onehot_mapping
        self.eps = eps

        node = node_mask.to(pred_E.dtype)
        pair = node.unsqueeze(2) * node.unsqueeze(1)
        pair = pair * (1.0 - torch.eye(n, dtype=pair.dtype, device=pair.device))

        self.max_asymmetry = float((pred_E - pred_E.transpose(1, 2)).abs().max().detach())
        E = 0.5 * (pred_E + pred_E.transpose(1, 2))

        self._adj = {
            name: E[..., list(channels)].sum(-1) * pair
            for name, channels in relation_channels.items()
        }

        self._edge = None
        if edge_channel_groups:
            edge_views = []
            for group in edge_channel_groups:
                joint = E[..., group]
                cond = joint / joint.sum(-1, keepdim=True).clamp_min(eps)
                edge_views.append((cond * pair.unsqueeze(-1)).permute(0, 3, 1, 2))
            self._edge = edge_views

        onehot = pred_X[..., node_column_index.to(pred_X.device)] * node.unsqueeze(-1)
        self._feat_onehot = onehot
        self._feat = onehot

    def get_batch(self, start: int, end: int):
        adj_b = {name: matrix[start:end] for name, matrix in self._adj.items()}
        edge_b = [e[start:end] for e in self._edge] if self._edge is not None else None
        return self._feat[start:end], self._feat_onehot[start:end], adj_b, edge_b


def softmax_node_channels(pred_X_logits: torch.Tensor, node_channel_slices) -> torch.Tensor:
    """Per-channel softmax, never over the whole last dim (WS1 gotcha)."""
    return torch.cat(
        [torch.softmax(pred_X_logits[..., sl], dim=-1) for sl in node_channel_slices], dim=-1
    )
