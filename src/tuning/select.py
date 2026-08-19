import torch
from torch import nn

from src.models.factory import build_model
from src.training.trainer import train_model
from src.tuning.grid import generate_grid
from src.utils.seed import set_seed


def select_hyperparameters(
        base_config,
        dataset,
        train_loader,
        val_loader,
        device,
        tuning_seeds
):
    candidates = generate_grid(
        base_config["search_space"]
    )

    best_candidate = None
    best_val_accuracy = -float("inf")
    best_val_loss = float("inf")

    candidate_results = []

    for candidate in candidates:
        val_accuracies = []
        val_losses = []

        for tuning_seed in tuning_seeds:
            set_seed(tuning_seed)

            candidate_config = base_config.copy()
            candidate_config.update(candidate)

            model = build_model(
                candidate_config,
                dataset
            ).to(device)

            criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=candidate_config["learning_rate"],
                weight_decay=candidate_config["weight_decay"]
            )

            model, training_info = train_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                epochs=candidate_config["epochs"],
                device=device,
                verbose=False
            )

            val_accuracies.append(
                training_info["best_val_accuracy"]
            )

            val_losses.append(
                training_info["best_val_loss"]
            )

        mean_val_accuracy = (
            sum(val_accuracies)
            / len(val_accuracies)
        )

        mean_val_loss = (
            sum(val_losses)
            / len(val_losses)
        )

        candidate_results.append({
            **candidate,
            "mean_val_accuracy": mean_val_accuracy,
            "mean_val_loss": mean_val_loss
        })

        if (
            mean_val_accuracy > best_val_accuracy
            or (
                mean_val_accuracy == best_val_accuracy
                and mean_val_loss < best_val_loss
            )
        ):
            best_candidate = candidate
            best_val_accuracy = mean_val_accuracy
            best_val_loss = mean_val_loss

    return best_candidate, candidate_results