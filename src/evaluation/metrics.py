import torch


def summarise_cross_validation(results):
    split_seeds = sorted(
        set(result["split_seed"] for result in results)
    )

    split_summaries = []

    for split_seed in split_seeds:
        split_results = [
            result
            for result in results
            if result["split_seed"] == split_seed
        ]

        test_folds = sorted(
            set(result["test_fold"] for result in split_results)
        )

        fold_means = []
        training_seed_stds = []

        for test_fold in test_folds:
            fold_accuracies = torch.tensor([
                result["test_accuracy"]
                for result in split_results
                if result["test_fold"] == test_fold
            ])

            fold_mean = fold_accuracies.mean().item()

            fold_std = fold_accuracies.std(
                unbiased=False
            ).item()

            fold_means.append(fold_mean)
            training_seed_stds.append(fold_std)

        fold_means = torch.tensor(fold_means)

        split_mean = fold_means.mean().item()

        fold_std = fold_means.std(
            unbiased=False
        ).item()

        average_training_seed_std = torch.tensor(
            training_seed_stds
        ).mean().item()

        split_summaries.append({
            "split_seed": split_seed,
            "mean_accuracy": split_mean,
            "fold_std": fold_std,
            "average_training_seed_std": average_training_seed_std
        })

    split_means = torch.tensor([
        summary["mean_accuracy"]
        for summary in split_summaries
    ])

    overall_mean = split_means.mean().item()

    overall_split_std = split_means.std(
        unbiased=False
    ).item()

    return split_summaries, overall_mean, overall_split_std