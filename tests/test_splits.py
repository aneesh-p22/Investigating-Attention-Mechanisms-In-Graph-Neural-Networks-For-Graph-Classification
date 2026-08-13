import torch

from src.data.datasets import load_dataset
from src.data.splits import (
    create_stratified_folds,
    get_outer_fold_indices,
    split_train_val
)


dataset = load_dataset("MUTAG")

num_folds = 10
split_seed = 0


folds = create_stratified_folds(
    dataset,
    num_folds=num_folds,
    seed=split_seed
)


# 1. Every graph should appear in exactly one fold.

all_fold_indices = torch.cat(folds)

assert len(all_fold_indices) == len(dataset)

assert len(torch.unique(all_fold_indices)) == len(dataset)


# 2. Each class should be distributed approximately equally across folds.

labels = torch.tensor([
    dataset[i].y.item()
    for i in range(len(dataset))
])

for class_label in labels.unique():
    class_counts = []

    for fold in folds:
        fold_labels = labels[fold]

        count = (
            fold_labels == class_label
        ).sum().item()

        class_counts.append(count)

    assert max(class_counts) - min(class_counts) <= 1


# 3. Check every outer CV iteration.

test_counts = torch.zeros(
    len(dataset),
    dtype=torch.long
)

for test_fold in range(num_folds):

    outer_train_idx, test_idx = get_outer_fold_indices(
        folds,
        test_fold=test_fold
    )

    train_idx, val_idx = split_train_val(
        dataset,
        outer_train_idx,
        val_ratio=0.1,
        seed=split_seed
    )

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())

    # Train, validation and test must not overlap.

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    # Together they must contain every graph.

    combined = (
        train_set
        | val_set
        | test_set
    )

    assert combined == set(range(len(dataset)))

    # Record how many times each graph becomes test data.

    test_counts[test_idx] += 1


# 4. Over 10-fold CV, every graph must be test data exactly once.

assert torch.all(test_counts == 1)


print("All cross-validation split tests passed.")