import torch
from torch import nn

from src.data.datasets import load_dataset
from src.data.splits import split_dataset
from src.data.loaders import create_loaders
from src.models.gcn import GCN
from src.training.trainer import train_model
from src.evaluation.evaluate import evaluate
from src.utils.seed import set_seed


dataset = load_dataset("MUTAG")

seeds = [0, 1, 2, 3, 4]

test_accuracies = []

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
        batch_size=32
    )

    model = GCN(
        input_dim=dataset.num_features,
        hidden_dim=32,
        num_classes=dataset.num_classes
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=100,
        verbose=False
    )

    test_loss, test_accuracy = evaluate(
        model, test_loader,
        criterion
    )

    test_accuracies.append(test_accuracy)

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