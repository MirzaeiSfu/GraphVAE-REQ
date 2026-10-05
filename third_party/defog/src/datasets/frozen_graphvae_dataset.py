"""Read exact GraphVAE split artifacts without recreating dataset splits."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


def _load_collection(path: Path, expected_digest: str, dataset: str, split: str):
    from ggm_eval import load_pyg_collection_with_metadata
    from ggm_eval.contract import collection_digest, validate_collection

    graphs, metadata = load_pyg_collection_with_metadata(path, normalize=False)
    validate_collection(graphs, name=f"{dataset} {split}")
    digest = collection_digest(graphs)
    if digest != expected_digest:
        raise ValueError(
            f"Frozen {dataset} {split} digest {digest}; expected {expected_digest}."
        )
    required = {"dataset": dataset, "split": split}
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Frozen {dataset} {split} metadata {key}="
                f"{metadata.get(key)!r}; expected {expected!r}."
            )
    return graphs, metadata


class FrozenSplitDataset(Dataset):
    def __init__(self, root: Path, dataset: str, split: str, digest: str):
        self.root = str(root)
        self.dataset_name = dataset.lower()
        filename = {
            "train": "real_train_graphs.pt",
            "validation": "real_validation_graphs.pt",
            "reference": "real_test_graphs.pt",
        }[split]
        graphs, self.metadata = _load_collection(
            root / filename, digest, dataset, split
        )
        self.graphs = [self._to_defog(graph) for graph in graphs]

    @staticmethod
    def _to_defog(graph) -> Data:
        edge_count = int(graph.edge_index.shape[1])
        edge_attr = torch.zeros((edge_count, 2), dtype=torch.float32)
        edge_attr[:, 1] = 1.0
        return Data(
            x=graph.x.detach().cpu().to(torch.float32).contiguous(),
            edge_index=graph.edge_index.detach().cpu().to(torch.long).contiguous(),
            edge_attr=edge_attr,
            y=torch.zeros((1, 0), dtype=torch.float32),
            n_nodes=torch.tensor([int(graph.num_nodes)], dtype=torch.long),
            num_nodes=int(graph.num_nodes),
        )

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


class FrozenGraphVAEDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = str(cfg.dataset.identity).upper()
        root = Path(str(cfg.dataset.root)).expanduser().resolve()
        datasets = {
            "train": FrozenSplitDataset(
                root, self.dataset_name, "train", str(cfg.dataset.train_sha256)
            ),
            "val": FrozenSplitDataset(
                root,
                self.dataset_name,
                "validation",
                str(cfg.dataset.validation_sha256),
            ),
            "test": FrozenSplitDataset(
                root,
                self.dataset_name,
                "reference",
                str(cfg.dataset.reference_sha256),
            ),
        }
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset


class FrozenGraphVAEInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.dataset_name = datamodule.dataset_name.lower()
        self.is_molecular = False
        self.spectre = True
        self.use_charge = False
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

    def compute_reference_metrics(self, datamodule, sampling_metrics):
        # Final metrics come exclusively from the frozen external package.
        self.ref_metrics = {"val": {}, "test": {}}


class FrozenExternalSamplingMetrics:
    def reset(self):
        return None

    def forward(self, *args, **kwargs):
        return {}


class FrozenNoOpVisualization:
    def visualize(self, *args, **kwargs):
        return None

    def visualize_chain(self, *args, **kwargs):
        return None
