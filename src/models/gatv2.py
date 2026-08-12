import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_classes,
            heads=4,
            attention_dropout=0.0,
            negative_slope=0.2,
            add_self_loops=True
    ):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")

        head_dim = hidden_dim // heads

        self.conv1 = GATv2Conv(
            input_dim,
            head_dim,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
            negative_slope=negative_slope,
            add_self_loops=add_self_loops
        )

        self.conv2 = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=attention_dropout,
            negative_slope=negative_slope,
            add_self_loops=add_self_loops
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        x = self.classifier(x)

        return x