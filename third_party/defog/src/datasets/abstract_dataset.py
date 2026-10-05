import copy
import os

import torch
import wandb
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset
from tqdm import tqdm

import utils as utils
from datasets.dataset_utils import (
    DistributionNodes,
    argmax_per_channel,
    build_channel_slices,
    flatten_channel_tensors,
    load_pickle,
    save_pickle,
    split_flat_x,
)


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            batch_size=cfg.train.batch_size if "debug" not in cfg.general.name else 2,
            num_workers=cfg.train.num_workers,
            pin_memory=getattr(cfg.dataset, "pin_memory", False),
        )
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None
        print(
            f'This dataset contains {len(datasets["train"])} training graphs, {len(datasets["val"])} validation graphs, {len(datasets["test"])} test graphs.'
        )

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def node_counts(self, max_nodes_possible=1000):
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None

        for data in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.train_dataloader()):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes, zinc=False):
        valencies = torch.zeros(
            3 * max_n_nodes - 2
        )  # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])
        if zinc:
            multiplier = torch.tensor([0, 1, 2, 3])  # zinc250

        for data in self.train_dataloader():
            n = data.x.shape[0]

            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    def _infer_node_channel_dims(self, node_types=None):
        """Finds number of node channels, defaults to one"""
        explicit_dims = getattr(self, "node_channel_dims", None)
        if explicit_dims is not None:
            dims = [int(dim) for dim in explicit_dims]
        elif node_types is not None:
            dims = [int(len(node_types))]
        elif self.output_dims is not None and "X" in self.output_dims:
            dims = [int(self.output_dims["X"])]
        else:
            raise ValueError(
                "Cannot infer node_channel_dims before node feature width is known."
            )

        if len(dims) == 0 or any(dim <= 0 for dim in dims):
            raise ValueError(f"Invalid node_channel_dims: {dims}")
        return dims

    def build_node_channel_slices(self):
        """Build slices mapping flattened node features to channel blocks."""
        return build_channel_slices(self._infer_node_channel_dims())

    def set_node_channel_info(self, node_channel_dims, node_types_per_channel=None):
        """Register the multichannel node schema for this dataset."""
        self.node_channel_dims = [int(dim) for dim in node_channel_dims]
        self.num_node_channels = len(self.node_channel_dims)
        self.node_channel_slices = self.build_node_channel_slices()
        total_width = sum(self.node_channel_dims)
        self.num_classes = total_width

        if node_types_per_channel is None:
            self.node_types_per_channel = None
            self.node_types = None
        else:
            if len(node_types_per_channel) != self.num_node_channels:
                raise ValueError(
                    "node_types_per_channel length does not match "
                    f"node_channel_dims: {len(node_types_per_channel)} vs "
                    f"{self.num_node_channels}"
                )
            self.node_types_per_channel = node_types_per_channel
            if self.num_node_channels == 1:
                self.node_types = node_types_per_channel[0]
            else:
                # For single channel; use node_types_per_channel for multi as above
                self.node_types = torch.cat(
                    [torch.as_tensor(values) for values in node_types_per_channel]
                )

    def split_flat_x(self, flat_x):
        """Split flattened node features into one tensor per categorical channel."""
        return split_flat_x(flat_x, self.node_channel_dims)

    def flatten_node_channels(self, channel_tensors):
        """Concatenate per-channel node tensors into one flat node feature tensor."""
        if len(channel_tensors) != self.num_node_channels:
            raise ValueError(
                f"Expected {self.num_node_channels} node channels, received "
                f"{len(channel_tensors)}."
            )
        return flatten_channel_tensors(channel_tensors)

    def argmax_per_channel(self, flat_x):
        """Recover one categorical index per node channel from flattened features."""
        return argmax_per_channel(flat_x, self.node_channel_dims)

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)
        self.set_node_channel_info(
            node_channel_dims=getattr(self, "node_channel_dims", [len(node_types)]),
            node_types_per_channel=getattr(
                self,
                "node_types_per_channel",
                [node_types],
            ),
        )

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch,
        )

        # ex_dense.E = ex_dense.E[..., :-1]  # debug

        example_data = {
            "X_t": ex_dense.X,
            "E_t": ex_dense.E,
            "y_t": example_batch["y"],
            "node_mask": node_mask,
        }
        self.input_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": example_batch["y"].size(1)
            + 1,  # this part take into account the conditioning
        }  # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims["X"] += ex_extra_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims["X"] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": 0,
        }
        self.set_node_channel_info(
            node_channel_dims=getattr(
                self,
                "node_channel_dims",
                [example_batch["x"].size(1)],
            ),
            node_types_per_channel=getattr(self, "node_types_per_channel", None),
        )

    def compute_reference_metrics(self, datamodule, sampling_metrics):

        ref_metrics_path = os.path.join(
            datamodule.train_dataloader().dataset.root, f"ref_metrics.pkl"
        )
        if hasattr(datamodule, "remove_h"):
            if datamodule.remove_h:
                ref_metrics_path = ref_metrics_path.replace(".pkl", "_no_h.pkl")
            else:
                ref_metrics_path = ref_metrics_path.replace(".pkl", "_h.pkl")

        # Only compute the reference metrics if they haven't been computed already
        if not os.path.exists(ref_metrics_path):

            print("Reference metrics not found. Computing them now.")
            # Transform the training dataset into a list of graphs in the appropriate format
            training_graphs = []
            print("Converting training dataset to format required by sampling metrics.")
            for data_batch in tqdm(datamodule.train_dataloader()):
                dense_data, node_mask = utils.to_dense(
                    data_batch.x,
                    data_batch.edge_index,
                    data_batch.edge_attr,
                    data_batch.batch,
                )
                if self.num_node_channels == 1:
                    dense_data = dense_data.mask(
                        node_mask, collapse=True
                    ).split(node_mask)
                else:
                    dense_data = dense_data.mask(node_mask).split(node_mask)
                    for graph in dense_data:
                        graph.X = [
                            channel.argmax(dim=-1)
                            for channel in self.split_flat_x(graph.X)
                        ]
                for graph in dense_data:
                    training_graphs.append([graph.X, graph.E])

            # defining dummy arguments
            dummy_kwargs = {
                "name": "ref_metrics",
                "current_epoch": 0,
                "val_counter": 0,
                "local_rank": 0,
                "ref_metrics": {"val": None, "test": None},
            }

            print("Computing validation reference metrics.")
            # do not have to worry about wandb because it was not init yet
            val_sampling_metrics = copy.deepcopy(sampling_metrics)

            val_ref_metrics = val_sampling_metrics.forward(
                training_graphs,
                test=False,
                **dummy_kwargs,
            )

            print("Computing test reference metrics.")
            test_sampling_metrics = copy.deepcopy(sampling_metrics)
            test_ref_metrics = test_sampling_metrics.forward(
                training_graphs,
                test=True,
                **dummy_kwargs,
            )

            print("Saving reference metrics.")
            # print(f"deg: {test_reference_metrics['degree']} | clus: {test_reference_metrics['clustering']} | orbit: {test_reference_metrics['orbit']}")
            # breakpoint()
            save_pickle(
                {"val": val_ref_metrics, "test": test_ref_metrics}, ref_metrics_path
            )

        print("Loading reference metrics.")
        self.ref_metrics = load_pickle(ref_metrics_path)
        print("Validation reference metrics:", self.ref_metrics["val"])
        print("Test reference metrics:", self.ref_metrics["test"])
