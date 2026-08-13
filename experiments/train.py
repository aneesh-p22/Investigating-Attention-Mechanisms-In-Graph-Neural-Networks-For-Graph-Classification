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
from src.data.splits import (
    create_stratified_folds,
    get_outer_fold_indices,
    split_train_val
    )
from src.data.loaders import create_loaders

from src.models.factory import build_model

from src.training.trainer import train_model
from src.evaluation.evaluate import evaluate
from src.evaluation.metrics import summarise_cross_validation

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

results = []


for split_seed in CONFIG["split_seeds"]:

    folds = create_stratified_folds(
        dataset,
        num_folds=CONFIG["num_folds"],
        seed=split_seed
    )

    for test_fold in range(CONFIG["num_folds"]):

        outer_train_idx, test_idx = get_outer_fold_indices(
            folds,
            test_fold=test_fold
        )

        train_idx, val_idx = split_train_val(
            dataset,
            outer_train_idx,
            val_ratio=CONFIG["inner_val_ratio"],
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
                f"split_{split_seed}_"
                f"fold_{test_fold}_"
                f"train_{training_seed}.pt"
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

            results.append({
                "model": CONFIG["model"],
                "dataset": CONFIG["dataset"],

                "split_seed": split_seed,
                "test_fold": test_fold,
                "training_seed": training_seed,

                "num_folds": CONFIG["num_folds"],
                "inner_val_ratio": CONFIG["inner_val_ratio"],

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
                f"Split {split_seed}, "
                f"fold {test_fold}, "
                f"training seed {training_seed}: "
                f"test_loss={test_loss:.4f}, "
                f"test_accuracy={test_accuracy:.4f}"
            )


split_summaries, mean_accuracy, split_std = (
    summarise_cross_validation(results)
)

print()

for summary in split_summaries:
    print(
        f"Split {summary['split_seed']}: "
        f"mean_accuracy={summary['mean_accuracy']:.4f}, "
        f"fold_std={summary['fold_std']:.4f}, "
        f"average_training_seed_std="
        f"{summary['average_training_seed_std']:.4f}"
    )

print()
print(f"Overall mean accuracy: {mean_accuracy:.4f}")
print(f"Split-to-split standard deviation: {split_std:.4f}")


save_results(
    results,
    f"results/raw/{args.config}.csv"
)