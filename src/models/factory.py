from src.models.gcn import GCN
from src.models.gat import GAT


def build_model(config, dataset):
    model_name = config["model"]

    if model_name == "gcn":
        return GCN(
            input_dim=dataset.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=dataset.num_classes
        )

    if model_name == "gat":
        return GAT(
            input_dim=dataset.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=dataset.num_classes,
            heads=config["heads"]
        )

    raise ValueError(f"Unknown model: {model_name}")