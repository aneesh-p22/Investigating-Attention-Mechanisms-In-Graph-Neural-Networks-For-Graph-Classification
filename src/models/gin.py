import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINConv(
            mlp1,
            train_eps=True
        )

        self.conv2 = GINConv(
            mlp2,
            train_eps=True
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