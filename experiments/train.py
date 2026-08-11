import torch
from torch import nn

from src.data.datasets import load_dataset
from src.data.splits import split_dataset
from src.data.loaders import create_loaders
from src.models.gcn import GCN
from src.training.trainer import train_model
from src.evaluation.evaluate import evaluate
from src.utils.seed import set_seed
from src.utils.results import save_results
from configs.gcn_mutag import CONFIG


dataset = load_dataset(CONFIG["dataset"])

seeds = CONFIG["seeds"]

test_accuracies = []
results = []

for seed in seeds:
    set_seed(seed)

    train_idx, val_idx, test_idx = split_dataset(
        dataset,
        seed=seed
    )

    train_loader, val_loader, test_loader = create_loaders(
        dataset,
        train_idx,
        val_idx,
        test_idx,
        batch_size=CONFIG["batch_size"]
    )

    model = GCN(
        input_dim=dataset.num_features,
        hidden_dim=CONFIG["hidden_dim"],
        num_classes=dataset.num_classes
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"]
    )

    model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=CONFIG["epochs"],
        verbose=False
    )

    test_loss, test_accuracy = evaluate(
        model, test_loader,
        criterion
    )

    test_accuracies.append(test_accuracy)

    results.append({
        "seed": seed,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

    print(
        f"Seed {seed}: "
        f"test_loss={test_loss:.4f}"
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
    "results/raw/gcn_mutag.csv"
)