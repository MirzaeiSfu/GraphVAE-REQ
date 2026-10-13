import time

import wandb
import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection

from metrics.abstract_metrics import (
    CrossEntropyMetric,
    KLDMetric,
)


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


def compute_multichannel_node_loss(
    pred_X, true_X, node_channel_slices, node_loss_fn
):
    """Compute categorical node loss independently for every node channel with node_loss_fn."""
    pred_X = pred_X.reshape(-1, pred_X.size(-1))
    true_X = true_X.reshape(-1, true_X.size(-1))
    node_mask = (true_X != 0.0).any(dim=-1)

    channel_losses = []
    for channel_slice in node_channel_slices:
        pred_channel = pred_X[node_mask, channel_slice]
        true_channel = true_X[node_mask, channel_slice]
        if pred_channel.shape[0] > 0:
            channel_losses.append(node_loss_fn(pred_channel, true_channel))

    if not channel_losses:
        return pred_X.new_tensor(0.0)
    return torch.stack(channel_losses).mean()


class TrainLossDiscrete(nn.Module):
    """Train with Cross entropy"""

    def __init__(self, lambda_train, kld=False, node_channel_slices=None):
        super().__init__()
        self.lambda_train = lambda_train
        if not kld:
            self.node_loss = CrossEntropyMetric()
            self.edge_loss = CrossEntropyMetric()
        else:
            self.node_loss = KLDMetric()
            self.edge_loss = KLDMetric()
        self.y_loss = CrossEntropyMetric()
        self.node_channel_slices = node_channel_slices or [slice(None)]

    def forward(
        self,
        masked_pred_X,
        masked_pred_E,
        pred_y,
        true_X,
        true_E,
        true_y,
        log: bool,
    ):
        """Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean."""
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(
            masked_pred_X, (-1, masked_pred_X.size(-1))
        )  # (bs * n, dx)
        masked_pred_E = torch.reshape(
            masked_pred_E, (-1, masked_pred_E.size(-1))
        )  # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.0).any(dim=-1)
        mask_E = (true_E != 0.0).any(dim=-1)

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = (
            compute_multichannel_node_loss(
                masked_pred_X,
                true_X,
                self.node_channel_slices,
                self.node_loss,
            )
            if mask_X.any()
            else masked_pred_X.new_tensor(0.0)
        )
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if pred_y.numel() > 0 else 0.0

        if log:
            to_log = {
                "train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                "train_loss/X_CE": (
                    self.node_loss.compute() if true_X.numel() > 0 else -1
                ),
                "train_loss/E_CE": (
                    self.edge_loss.compute() if true_E.numel() > 0 else -1
                ),
                "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1,
            }
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = (
            self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        )
        epoch_edge_loss = (
            self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        )
        epoch_y_loss = (
            self.y_loss.compute() if self.y_loss.total_samples > 0 else -1
        )

        to_log = {
            "train_epoch/x_CE": epoch_node_loss,
            "train_epoch/E_CE": epoch_edge_loss,
            "train_epoch/y_CE": epoch_y_loss,
        }
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log
