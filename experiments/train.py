import argparse

import os

import torch
from torch import nn

from configs.gcn_mutag import CONFIG as GCN_MUTAG_CONFIG
from configs.gat_mutag import CONFIG as GAT_MUTAG_CONFIG
from configs.gin_mutag import CONFIG as GIN_MUTAG_CONFIG
from configs.graphsage_mutag import CONFIG as GRAPHSAGE_MUTAG_CONFIG
from configs.gatv2_mutag import CONFIG as GATV2_MUTAG_CONFIG

from src.data.datasets import load_dataset
from src.data.splits import split_dataset
from src.data.loaders import create_loaders

from src.models.factory import build_model

from src.training.trainer import train_model
from src.evaluation.evaluate import evaluate

from src.utils.seed import set_seed
from src.utils.results import save_results


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    required=True,
    choices=[
    "gcn_mutag",
    "gat_mutag",
    "gin_mutag",
    "graphsage_mutag",
    "gatv2_mutag"
    ]
)

args = parser.parse_args()


configs = {
    "gcn_mutag": GCN_MUTAG_CONFIG,
    "gat_mutag": GAT_MUTAG_CONFIG,
    "gin_mutag": GIN_MUTAG_CONFIG,
    "graphsage_mutag": GRAPHSAGE_MUTAG_CONFIG,
    "gatv2_mutag": GATV2_MUTAG_CONFIG
}

CONFIG = configs[args.config]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


dataset = load_dataset(CONFIG["dataset"])

test_accuracies = []
results = []


for split_seed in CONFIG["split_seeds"]:

    train_idx, val_idx, test_idx = split_dataset(
        dataset,
        train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        seed=split_seed
    )


    for training_seed in CONFIG["training_seeds"]:
        set_seed(training_seed)

        train_loader, val_loader, test_loader = create_loaders(
            dataset,
            train_idx,
            val_idx,
            test_idx,
            batch_size=CONFIG["batch_size"]
        )

        model = build_model(
            CONFIG,
            dataset
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )

        model, training_info = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            epochs=CONFIG["epochs"],
            device=device,
            verbose=False
        )

        checkpoint_dir = f"results/checkpoints/{args.config}"

        os.makedirs(
            checkpoint_dir,
            exist_ok=True
        )

        checkpoint_path = (
            f"{checkpoint_dir}/"
            f"split_{split_seed}_train_{training_seed}.pt"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        test_accuracies.append(test_accuracy)

        results.append({
            "model": CONFIG["model"],
            "dataset": CONFIG["dataset"],
            "split_seed": split_seed,
            "training_seed": training_seed,

            "train_ratio": CONFIG["train_ratio"],
            "val_ratio": CONFIG["val_ratio"],

            "hidden_dim": CONFIG["hidden_dim"],
            "batch_size": CONFIG["batch_size"],
            "learning_rate": CONFIG["learning_rate"],
            "epochs": CONFIG["epochs"],

            "model_dropout": CONFIG["model_dropout"],
            "weight_decay": CONFIG["weight_decay"],

            "heads": CONFIG.get("heads"),
            "attention_dropout": CONFIG.get("attention_dropout"),
            "negative_slope": CONFIG.get("negative_slope"),
            "add_self_loops": CONFIG.get("add_self_loops"),

            "best_epoch": training_info["best_epoch"],
            "best_val_loss": training_info["best_val_loss"],
            "best_val_accuracy": training_info["best_val_accuracy"],

            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        print(
            f"Seed {split_seed}: "
            f"test_loss={test_loss:.4f}, "
            f"test_accuracy={test_accuracy:.4f}"
        )


test_accuracies = torch.tensor(test_accuracies)

mean_accuracy = test_accuracies.mean().item()
std_accuracy = test_accuracies.std().item()

print()
print(f"Mean test accuracy: {mean_accuracy:.4f}")
print(f"Standard deviation: {std_accuracy:.4f}")


save_results(
    results,
    f"results/raw/{args.config}.csv"
)