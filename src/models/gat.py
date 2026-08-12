import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_classes,
            heads=4
    ):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")

        head_dim = hidden_dim // heads

        self.conv1 = GATConv(
            input_dim,
            head_dim,
            heads=heads,
            concat=True
        )

        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=True
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