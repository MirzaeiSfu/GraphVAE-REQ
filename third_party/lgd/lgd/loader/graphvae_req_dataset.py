"""Exact-split dataset support for GraphVAE-REQ generation experiments.

The prepared bundle format is deliberately small and self-contained::

    root/DATASET/{train,val,test}.pt
    root/DATASET/metadata.json

Each ``.pt`` file stores a list of :class:`torch_geometric.data.Data` graphs.
Categorical node fields live in ``data.x``.  Dense-edge preprocessing later
turns ``data.edge_attr`` into N x N targets; its first field is always binary
adjacency and every subsequent field is a native categorical edge feature.
Class zero in native edge fields means "no edge".
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset


def _torch_load(path):
    """Load PyG objects across both pre- and post-PyTorch-2.6 defaults."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0 has no weights_only argument.
        return torch.load(path, map_location="cpu")


class GraphVAEReqDataset(InMemoryDataset):
    """In-memory dataset made from an immutable GraphVAE-REQ split bundle."""

    def __init__(self, root, name):
        super().__init__(root=None)
        self.name = str(name)
        self.bundle_dir = Path(root).expanduser().resolve() / self.name
        metadata_path = self.bundle_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared dataset metadata: {metadata_path}. "
                "Run prepare_graphvae_req_data.py first."
            )
        self.metadata = json.loads(metadata_path.read_text())

        split_graphs = []
        split_indices = []
        offset = 0
        for split in ("train", "val", "test"):
            path = self.bundle_dir / f"{split}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing prepared split: {path}")
            graphs = _torch_load(path)
            if not isinstance(graphs, (list, tuple)):
                raise TypeError(f"{path} must contain a list of PyG Data objects")
            split_graphs.extend(graphs)
            split_indices.append(list(range(offset, offset + len(graphs))))
            offset += len(graphs)

        if not split_graphs:
            raise ValueError(f"Prepared dataset {self.bundle_dir} is empty")
        self.data, self.slices = self.collate(split_graphs)
        self.split_idxs = split_indices

