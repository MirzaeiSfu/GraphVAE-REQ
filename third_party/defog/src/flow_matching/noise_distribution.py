import torch

from src import utils
from datasets.dataset_utils import build_channel_slices, split_flat_x


class NoiseDistribution:

    def __init__(self, model_transition, dataset_infos):
        self.node_channel_dims = [
            int(dim)
            for dim in getattr(
                dataset_infos,
                "node_channel_dims",
                [dataset_infos.output_dims["X"]],
            )
        ]
        self.node_channel_slices = build_channel_slices(self.node_channel_dims)
        self.num_node_channels = len(self.node_channel_dims)
        self.x_num_classes = sum(self.node_channel_dims)
        self.e_num_classes = dataset_infos.output_dims["E"]
        self.y_num_classes = dataset_infos.output_dims["y"]
        self.x_added_classes = [0] * self.num_node_channels
        self.e_added_classes = 0
        self.y_added_classes = 0
        self.transition = model_transition

        channel_marginals = getattr(dataset_infos, "node_types_per_channel", None)
        if channel_marginals is None:
            if self.num_node_channels == 1 and getattr(dataset_infos, "node_types", None) is not None:
                channel_marginals = [dataset_infos.node_types]
            else:
                channel_marginals = [None] * self.num_node_channels

        def normalized_channel_marginals():
            result = []
            for dim, values in zip(self.node_channel_dims, channel_marginals):
                if values is None:
                    result.append(torch.ones(dim) / dim)
                    continue
                values = torch.as_tensor(values, dtype=torch.float)
                if values.numel() != dim:
                    raise ValueError(
                        "Node channel marginal has the wrong width: "
                        f"expected {dim}, got {values.numel()}"
                    )
                total = values.sum()
                result.append(
                    values / total if total > 0 else torch.ones(dim) / dim
                )
            return result

        node_marginals = normalized_channel_marginals()

        if model_transition == "uniform":
            x_limits = [torch.ones(dim) / dim for dim in self.node_channel_dims]
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

        elif model_transition == "absorbfirst":
            x_limits = []
            for dim in self.node_channel_dims:
                x_limit = torch.zeros(dim)
                x_limit[0] = 1
                x_limits.append(x_limit)
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[0] = 1

        elif model_transition == "argmax":
            edge_types = dataset_infos.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)

            e_max_dim = torch.argmax(e_marginals)
            x_limits = []
            for marginal in node_marginals:
                x_limit = torch.zeros_like(marginal)
                x_limit[torch.argmax(marginal)] = 1
                x_limits.append(x_limit)
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[e_max_dim] = 1

        elif model_transition == "absorbing":
            # Add one absorbing state independently to every node channel. A
            # joint absorbing state would reintroduce the Cartesian-product
            # semantics this representation is intended to remove.
            x_limits = []
            for channel_index, dim in enumerate(self.node_channel_dims):
                if dim > 1:
                    self.node_channel_dims[channel_index] += 1
                    self.x_added_classes[channel_index] = 1
                x_limit = torch.zeros(self.node_channel_dims[channel_index])
                x_limit[-1] = 1
                x_limits.append(x_limit)
            self.node_channel_slices = build_channel_slices(self.node_channel_dims)
            self.x_num_classes = sum(self.node_channel_dims)
            if self.e_num_classes > 1:
                self.e_num_classes += 1
                self.e_added_classes = 1

            e_limit = torch.zeros(self.e_num_classes)
            e_limit[-1] = 1

        elif model_transition == "marginal":
            x_limits = node_marginals

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "edge_marginal":
            x_limits = [torch.ones(dim) / dim for dim in self.node_channel_dims]

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "node_marginal":
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes
            x_limits = node_marginals

        else:
            raise ValueError(f"Unknown transition model: {model_transition}")

        y_limit = torch.ones(self.y_num_classes) / self.y_num_classes  # typically dummy
        print(
            f"Limit distribution of the classes | Nodes: {x_limits} | Edges: {e_limit}"
        )
        self.limit_dist = utils.PlaceHolder(
            X=torch.cat(x_limits), E=e_limit, y=y_limit
        )
        self.limit_dist.X_channels = x_limits

    def update_input_output_dims(self, input_dims):
        input_dims["X"] += sum(self.x_added_classes)
        input_dims["E"] += self.e_added_classes
        input_dims["y"] += self.y_added_classes

    def update_dataset_infos(self, dataset_infos):
        # The model output width must include virtual classes as well. For the
        # normal (non-absorbing) case this assignment is a no-op.
        dataset_infos.output_dims["X"] = self.x_num_classes
        if hasattr(dataset_infos, "atom_decoder"):
            dataset_infos.atom_decoder = (
                dataset_infos.atom_decoder + ["Y"] * sum(self.x_added_classes)
            )

    def get_limit_dist(self):
        return self.limit_dist

    def get_noise_dims(self):
        return {
            "X": self.x_num_classes,
            "E": len(self.limit_dist.E),
            "y": len(self.limit_dist.E),
        }

    def ignore_virtual_classes(self, X, E, y=None):
        if self.transition == "absorbing":
            X_channels = split_flat_x(X, self.node_channel_dims)
            new_X = torch.cat(
                [
                    channel[..., : -added] if added else channel
                    for channel, added in zip(X_channels, self.x_added_classes)
                ],
                dim=-1,
            )
            new_E = E[..., : -self.e_added_classes] if self.e_added_classes else E
            new_y = (
                y[..., : -self.y_added_classes]
                if y is not None and self.y_added_classes
                else y
            )
            return new_X, new_E, new_y
        else:
            return X, E, y

    def add_virtual_classes(self, X, E, y=None):
        X_channels = split_flat_x(X, [
            dim - added for dim, added in zip(self.node_channel_dims, self.x_added_classes)
        ])
        new_X = torch.cat(
            [
                torch.cat(
                    [channel, torch.zeros_like(channel[..., :1]).repeat(1, 1, added)],
                    dim=-1,
                )
                if added
                else channel
                for channel, added in zip(X_channels, self.x_added_classes)
            ],
            dim=-1,
        )

        if self.e_added_classes:
            e_virtual = torch.zeros_like(E[..., :1]).repeat(
                1, 1, 1, self.e_added_classes
            )
            new_E = torch.cat([E, e_virtual], dim=-1)
        else:
            new_E = E

        if y is not None:
            if self.y_added_classes:
                y_virtual = torch.zeros_like(y[..., :1]).repeat(
                    1, self.y_added_classes
                )
                new_y = torch.cat([y, y_virtual], dim=-1)
            else:
                new_y = y
        else:
            new_y = None

        return new_X, new_E, new_y
