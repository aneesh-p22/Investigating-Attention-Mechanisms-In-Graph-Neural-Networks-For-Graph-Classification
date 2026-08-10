import torch


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    generator = torch.Generator().manual_seed(seed)

    labels = torch.tensor([
        dataset[i].y.item()
        for i in range(len(dataset))
    ])

    train_idx = []
    val_idx = []
    test_idx = []

    for class_label in labels.unique():
        class_idx = torch.where(labels == class_label)[0]

        permutation = torch.randperm(
            len(class_idx),
            generator=generator
        )

        class_idx = class_idx[permutation]

        train_end = int(train_ratio * len(class_idx))
        val_end = train_end + int(val_ratio * len(class_idx))

        train_idx.append(class_idx[:train_end])
        val_idx.append(class_idx[train_end:val_end])
        test_idx.append(class_idx[val_end:])

    train_idx = torch.cat(train_idx)
    val_idx = torch.cat(val_idx)
    test_idx = torch.cat(test_idx)

    return train_idx, val_idx, test_idx