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
        tuning_seed=0
):
    candidates = generate_grid(
        base_config["search_space"]
    )

    best_candidate = None
    best_val_accuracy = -float("inf")
    best_val_loss = float("inf")

    candidate_results = []

    for candidate in candidates:
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

        val_accuracy = training_info[
            "best_val_accuracy"
        ]

        val_loss = training_info[
            "best_val_loss"
        ]

        candidate_results.append({
            **candidate,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss
        })

        if (
            val_accuracy > best_val_accuracy
            or (
                val_accuracy == best_val_accuracy
                and val_loss < best_val_loss
            )
        ):
            best_candidate = candidate
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss

    return best_candidate, candidate_results