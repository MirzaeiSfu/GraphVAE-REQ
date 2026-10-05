import os
import pathlib
import random
import os.path as osp

import numpy as np
import torch.nn.functional as F
import torch
import torch_geometric.utils
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import InMemoryDataset

from datasets.abstract_dataset import (
    AbstractDataModule,
    AbstractDatasetInfos,
)
from datasets.dataset_utils import to_list, RemoveYTransform


class MutagDataset(InMemoryDataset):
    def __init__(
        self,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_name = "mutag"
        root = root

        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train_indices.pt", "val_indices.pt", "test_indices.pt"]

    @property
    def split_file_name(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.split == "train":
            return [
                f"train.pt",
                f"train_n.pickle",
                f"train_node_types.npy",
                f"train_bond_types.npy",
            ]
        elif self.split == "val":
            return [
                f"val.pt",
                f"val_n.pickle",
                f"val_node_types.npy",
                f"val_bond_types.npy",
            ]
        else:
            return [
                f"test.pt",
                f"test_n.pickle",
                f"test_node_types.npy",
                f"test_bond_types.npy",
            ]

    def download(self):
        """
        Load MUTAG via DGL's GINDataset (same source as GraphVAE-REQ) and cache
        the per-graph adjacency/node-label tensors plus the seeded 70/10/20
        split indices.
        """
        import dgl

        dataset = dgl.data.GINDataset(name="MUTAG", self_loop=False)
        graphs, _labels = dataset.graphs, dataset.labels

        all_adj = [graph.adjacency_matrix().to_dense().long() for graph in graphs]
        all_node_labels = [graph.ndata["label"].long() for graph in graphs]

        torch.save(all_adj, osp.join(self.raw_dir, "mutag_adj.pt"))
        torch.save(all_node_labels, osp.join(self.raw_dir, "mutag_node_labels.pt"))

        num_graphs = len(graphs)
        graph_ids = list(range(num_graphs))

        rng = random.Random(123)
        shuffled = list(graph_ids)
        rng.shuffle(shuffled)

        train_len = int(0.7 * num_graphs)
        val_len = int(0.1 * num_graphs)

        train_indices = torch.tensor(shuffled[:train_len])
        val_indices = torch.tensor(shuffled[train_len : train_len + val_len])
        test_indices = torch.tensor(shuffled[train_len + val_len :])
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {len(test_indices)}")

        print(f"Train indices: {train_indices}")
        print(f"Val indices: {val_indices}")
        print(f"Test indices: {test_indices}")

        torch.save(train_indices, self.raw_paths[0])
        torch.save(val_indices, self.raw_paths[1])
        torch.save(test_indices, self.raw_paths[2])

    def process(self):
        indices = torch.load(
            os.path.join(self.raw_dir, "{}_indices.pt".format(self.split))
        )
        all_adj = torch.load(osp.join(self.raw_dir, "mutag_adj.pt"))
        all_node_labels = torch.load(osp.join(self.raw_dir, "mutag_node_labels.pt"))
        data_list = []

        # get information
        self.num_node_type = max(nl.max().item() for nl in all_node_labels) + 1
        self.num_edge_type = 2
        print(f"Number of node types: {self.num_node_type}")
        print(f"Number of edge types: {self.num_edge_type}")

        for idx in indices:
            idx = int(idx)
            adj = all_adj[idx]
            node_label = all_node_labels[idx]

            nodes = F.one_hot(
                node_label, num_classes=int(self.num_node_type)
            ).float()
            edge_index = adj.nonzero().T
            edge_attr = torch.zeros(edge_index.shape[1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            data = torch_geometric.data.Data(
                x=nodes,
                edge_index=edge_index,
                edge_attr=edge_attr,
                n_nodes=nodes.shape[0],
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class MutagDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, cfg.dataset.datadir)
        transform = RemoveYTransform()

        datasets = {
            "train": MutagDataset(
                root=root_path,
                transform=transform,
                split="train",
            ),
            "val": MutagDataset(
                root=root_path,
                transform=transform,
                split="val",
            ),
            "test": MutagDataset(
                root=root_path,
                transform=transform,
                split="test",
            ),
        }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset


class MutagInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.is_molecular = False
        self.spectre = True
        self.use_charge = False
        self.dataset_name = datamodule.inner.dataset_name
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

    def to_one_hot(self, data):
        """
        call in the beginning of data
        get the one_hot encoding for a charge beginning from -1
        """
        data.charge = data.x.new_zeros((*data.x.shape[:-1], 0))
        data.x = F.one_hot(data.x, num_classes=self.num_node_types).float()
        data.edge_attr = F.one_hot(
            data.edge_attr, num_classes=self.num_edge_types
        ).float()

        return data
