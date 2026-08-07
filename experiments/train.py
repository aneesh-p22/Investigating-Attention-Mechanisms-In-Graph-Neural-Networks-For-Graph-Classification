from src.data.datasets import load_dataset
from src.data.splits import split_dataset
from src.data.loaders import create_loaders
from src.models.gcn import GCN


dataset = load_dataset("MUTAG")

train_idx, val_idx, test_idx = split_dataset(dataset)

train_loader, val_loader, test_loader = create_loaders(
    dataset,
    train_idx,
    val_idx,
    test_idx,
    batch_size=32
)

model = GCN(
    input_dim=dataset.num_features,
    hidden_dim=32,
    num_classes=dataset.num_classes
)

batch = next(iter(train_loader))

out = model(
    batch.x,
    batch.edge_index,
    batch.batch
)

print("Graphs in batch:", batch.num_graphs)
print("Node features:", batch.x.shape)
print("Output:", out.shape)