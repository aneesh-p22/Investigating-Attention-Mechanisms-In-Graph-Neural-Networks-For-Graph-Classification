from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.gin import GIN
from src.models.graphsage import GraphSAGE
from src.models.gatv2 import GATv2


def build_model(config, dataset):
    model_name = config["model"]

    if model_name == "gcn":
        return GCN(
            input_dim=dataset.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=dataset.num_classes,
            model_dropout=config["model_dropout"]
        )

    if model_name == "gat":
        return GAT(
            input_dim=dataset.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=dataset.num_classes,
            heads=config["heads"],
            model_dropout=config["model_dropout"],
            attention_dropout=config["attention_dropout"],
            negative_slope=config["negative_slope"],
            add_self_loops=config["add_self_loops"]
        )

    if model_name == "gin":
        return GIN(
            input_dim=dataset.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=dataset.num_classes,
            model_dropout=config["model_dropout"]
        )

    if model_name == "graphsage":
        return GraphSAGE(
            input_dim=dataset.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=dataset.num_classes,
            model_dropout=config["model_dropout"]
        )

    if model_name == "gatv2":
        return GATv2(
            input_dim=dataset.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=dataset.num_classes,
            heads=config["heads"],
            model_dropout=config["model_dropout"],
            attention_dropout=config["attention_dropout"],
            negative_slope=config["negative_slope"],
            add_self_loops=config["add_self_loops"]
        )

    raise ValueError(f"Unknown model: {model_name}")