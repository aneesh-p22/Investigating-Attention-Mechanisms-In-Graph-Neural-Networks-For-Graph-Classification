from torch_geometric.datasets import TUDataset

def load_dataset(name, root="data/TUDataset"):
    dataset = TUDataset(
        root=root,
        name=name
    )

    return dataset