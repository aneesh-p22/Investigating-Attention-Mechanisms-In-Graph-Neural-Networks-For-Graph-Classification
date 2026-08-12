import copy

from src.training.train_epoch import train_epoch
from src.evaluation.evaluate import evaluate


def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs,
        device,
        verbose=True
):
    best_val_accuracy = 0
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = None

    for epoch in range(epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        if (
            val_accuracy > best_val_accuracy
            or (
                val_accuracy == best_val_accuracy
                and val_loss < best_val_loss
            )
        ):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        if verbose:
            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_accuracy={val_accuracy:.4f}"
            )

    model.load_state_dict(best_model_state)

    training_info = {
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
    "best_val_loss": best_val_loss
    }

    return model, training_info