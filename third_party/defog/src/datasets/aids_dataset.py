import os
import pathlib
import random
import shutil
import zipfile
import os.path as osp
import bisect
import math

import numpy as np
import torch.nn.functional as F
import torch
import torch_geometric.utils
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import InMemoryDataset, download_url

from datasets.abstract_dataset import (
    AbstractDataModule,
    AbstractDatasetInfos,
)
from datasets.dataset_utils import RemoveYTransform, split_flat_x, to_list

# Mirrors EnzymeDataset's preference for a local sibling GraphVAE-REQ copy of
# the TU Dortmund Kernel_dataset collection over a network download, when
# present.
_LOCAL_KERNEL_DATASET_DIR = os.path.join(
    pathlib.Path(os.path.realpath(__file__)).parents[3],
    "GraphVAE-REQ",
    "data_raw",
    "Kernel_dataset",
)


# Use the same quantile convention as GraphVAE-REQ. Each node variable keeps
# its own categorical channel, and we flatten their one-hot encodings into
# data.x for compatibility with the current DeFoG backbone.
_ATTRIBUTE_BINS = 8
# Version the cache because compacting observed quantile states changes the
# node feature width from 60 to GraphVAE-REQ's 59.
_NODE_CHANNEL_METADATA_FILE = "aids_node_channels_v2_compact.pt"
_PROCESSING_VERSION = "v2_compact"


def _quantile_thresholds(values, bins):
    """Match GraphVAE-REQ's quantile_thresholds() implementation."""
    ordered = sorted(float(value) for value in values)
    thresholds = []
    for bin_index in range(1, bins):
        position = max(
            0,
            min(
                len(ordered) - 1,
                math.ceil(bin_index * len(ordered) / bins) - 1,
            ),
        )
        threshold = ordered[position]
        if not thresholds or threshold > thresholds[-1]:
            thresholds.append(threshold)
    return thresholds


def _compact_observed_states(values):
    """Map observed quantile states to contiguous GraphVAE-REQ IDs."""
    observed_states = sorted(int(value) for value in np.unique(values))
    state_to_compact = {
        state: compact_state
        for compact_state, state in enumerate(observed_states)
    }
    compact_values = np.asarray(
        [state_to_compact[int(value)] for value in values], dtype=np.int64
    )
    return compact_values, observed_states


def _write_node_channel_metadata(raw_dir):
    """Persist AIDS' shared multichannel node metadata."""
    attributes = np.loadtxt(
        os.path.join(raw_dir, "AIDS_node_attributes.txt"), delimiter=","
    )
    attributes = np.atleast_2d(attributes)
    # AIDS labels are already zero-based (unlike ENZYMES labels).
    node_labels = np.loadtxt(
        os.path.join(raw_dir, "AIDS_node_labels.txt"), delimiter=","
    ).astype(np.int64)

    if len(node_labels) != len(attributes):
        raise ValueError(
            "AIDS node-label and node-attribute counts differ: "
            f"{len(node_labels)} labels vs {len(attributes)} attribute rows"
        )

    thresholds = [
        _quantile_thresholds(attributes[:, dimension], _ATTRIBUTE_BINS)
        for dimension in range(attributes.shape[1])
    ]
    binned_attributes = np.empty_like(attributes, dtype=np.int64)
    attribute_observed_states = []
    for dimension, dimension_thresholds in enumerate(thresholds):
        raw_states = np.asarray(
            [
                bisect.bisect_right(dimension_thresholds, float(value))
                for value in attributes[:, dimension]
            ],
            dtype=np.int64,
        )
        binned_attributes[:, dimension], observed_states = _compact_observed_states(
            raw_states
        )
        attribute_observed_states.append(observed_states)

    channel_names = ["node_label"] + [
        f"node_attr_{dimension:02d}" for dimension in range(attributes.shape[1])
    ]
    node_channel_labels = torch.empty(
        (len(node_labels), 1 + attributes.shape[1]), dtype=torch.long
    )
    node_channel_labels[:, 0] = torch.from_numpy(node_labels)
    node_channel_labels[:, 1:] = torch.from_numpy(binned_attributes)
    node_channel_dims = [int(node_labels.max()) + 1] + [
        int(np.unique(binned_attributes[:, dimension]).size)
        for dimension in range(binned_attributes.shape[1])
    ]
    torch.save(
        {
            "attribute_bins": _ATTRIBUTE_BINS,
            "attribute_quantile_thresholds": thresholds,
            "attribute_observed_states": attribute_observed_states,
            "node_channel_names": channel_names,
            "node_channel_dims": node_channel_dims,
            "node_channel_labels": node_channel_labels,
        },
        os.path.join(raw_dir, _NODE_CHANNEL_METADATA_FILE),
    )


class AidsDataset(InMemoryDataset):
    def __init__(
        self,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_name = "aids"
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
        metadata = torch.load(os.path.join(self.raw_dir, _NODE_CHANNEL_METADATA_FILE))
        self.node_channel_names = metadata["node_channel_names"]
        self.node_channel_dims = metadata["node_channel_dims"]
        self.attribute_quantile_thresholds = metadata[
            "attribute_quantile_thresholds"
        ]

    @property
    def raw_file_names(self):
        return [
            "train_indices.pt",
            "val_indices.pt",
            "test_indices.pt",
            "AIDS_A.txt",
            "AIDS_graph_indicator.txt",
            "AIDS_graph_labels.txt",
            "AIDS_node_labels.txt",
            "AIDS_node_attributes.txt",
            "AIDS_edge_labels.txt",
            _NODE_CHANNEL_METADATA_FILE,
        ]

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
        return [f"{self.split}_joint_attributes_{_PROCESSING_VERSION}.pt"]

    def download(self):
        """
        Fetch raw files. Prefers the local sibling GraphVAE-REQ checkout's copy of
        the TU Dortmund AIDS dataset over a network download when available.
        """
        raw_names = [
            "AIDS_A.txt",
            "AIDS_graph_indicator.txt",
            "AIDS_graph_labels.txt",
            "AIDS_node_labels.txt",
            "AIDS_node_attributes.txt",
            "AIDS_edge_labels.txt",
        ]
        local_dir = os.path.join(_LOCAL_KERNEL_DATASET_DIR, "AIDS")
        if os.path.isdir(local_dir):
            print(f"Using local AIDS dataset at {local_dir}")
            for name in raw_names:
                shutil.copyfile(
                    os.path.join(local_dir, name), os.path.join(self.raw_dir, name)
                )
        else:
            zip_path = download_url(
                "https://www.chrsmrrs.com/graphkerneldatasets/AIDS.zip",
                self.raw_dir,
            )
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)
            os.remove(zip_path)
            extracted_dir = osp.join(self.raw_dir, "AIDS")
            for name in os.listdir(extracted_dir):
                shutil.move(
                    osp.join(extracted_dir, name), osp.join(self.raw_dir, name)
                )
            shutil.rmtree(extracted_dir)

        _write_node_channel_metadata(self.raw_dir)

        # read
        path = os.path.join(self.root, "raw")
        data_graph_indicator = np.loadtxt(
            os.path.join(path, "AIDS_graph_indicator.txt"), delimiter=","
        ).astype(int)

        num_graphs = int(data_graph_indicator.max())
        graph_ids = list(range(1, num_graphs + 1))

        # 70/10/20 from GraphVAE-REQ
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
        data_adj = (
            torch.Tensor(
                np.loadtxt(os.path.join(self.raw_dir, "AIDS_A.txt"), delimiter=",")
            ).long()
            - 1
        )
        channel_metadata = torch.load(
            os.path.join(self.raw_dir, _NODE_CHANNEL_METADATA_FILE)
        )
        data_node_channel_labels = channel_metadata["node_channel_labels"]
        data_graph_indicator = torch.Tensor(
            np.loadtxt(
                os.path.join(self.raw_dir, "AIDS_graph_indicator.txt"),
                delimiter=",",
            )
        ).long()
        data_edge_label = torch.Tensor(
            np.loadtxt(
                os.path.join(self.raw_dir, "AIDS_edge_labels.txt"),
                delimiter=",",
            )
        ).long()
        data_list = []

        # get information
        self.node_channel_dims = channel_metadata["node_channel_dims"]
        self.node_channel_names = channel_metadata["node_channel_names"]
        self.num_node_type = sum(self.node_channel_dims)
        # no-edge (0) + single/double/triple (raw labels 0/1/2 -> classes 1/2/3)
        self.num_edge_type = int(data_edge_label.max().item()) + 2
        print(f"Number of flattened node types: {self.num_node_type}")
        print(f"Node channel dims: {self.node_channel_dims}")
        print(f"Number of edge types: {self.num_edge_type}")

        for idx in indices:
            offset = torch.where(data_graph_indicator == idx)[0].min()
            node_idx = data_graph_indicator == idx
            channel_labels = data_node_channel_labels[node_idx]
            node_channels = [
                F.one_hot(
                    channel_labels[:, channel_idx],
                    num_classes=int(channel_dim),
                ).float()
                for channel_idx, channel_dim in enumerate(self.node_channel_dims)
            ]
            nodes = torch.cat(node_channels, dim=-1)
            edge_idx = node_idx[data_adj[:, 0]]
            edge_index = data_adj[edge_idx] - offset
            edge_label = data_edge_label[edge_idx]
            edge_attr = torch.zeros(
                edge_index.shape[0], self.num_edge_type, dtype=torch.float
            )
            edge_attr[torch.arange(edge_index.shape[0]), edge_label + 1] = 1
            edge_index, edge_attr = remove_self_loops(edge_index.T, edge_attr)
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


class AidsDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, cfg.dataset.datadir)
        transform = RemoveYTransform()

        datasets = {
            "train": AidsDataset(
                root=root_path,
                transform=transform,
                split="train",
            ),
            "val": AidsDataset(
                root=root_path,
                transform=transform,
                split="val",
            ),
            "test": AidsDataset(
                root=root_path,
                transform=transform,
                split="test",
            ),
        }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset


class AidsInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.is_molecular = False
        self.spectre = True
        self.use_charge = False
        self.dataset_name = datamodule.inner.dataset_name
        self.node_channel_names = datamodule.inner.node_channel_names
        self.node_channel_dims = datamodule.inner.node_channel_dims
        self.attribute_quantile_thresholds = (
            datamodule.inner.attribute_quantile_thresholds
        )
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.node_types_per_channel = split_flat_x(
            self.node_types, self.node_channel_dims
        )
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

    def to_one_hot(self, data):
        """
        call in the beginning of data
        get the one_hot encoding for a charge beginning from -1
        """
        data.charge = data.x.new_zeros((*data.x.shape[:-1], 0))
        if data.x.dim() == 1:
            data.x = F.one_hot(data.x, num_classes=sum(self.node_channel_dims)).float()
        if data.edge_attr is not None and data.edge_attr.dim() == 1:
            data.edge_attr = F.one_hot(
                data.edge_attr, num_classes=int(data.edge_attr.max().item()) + 1
            ).float()

        return data
