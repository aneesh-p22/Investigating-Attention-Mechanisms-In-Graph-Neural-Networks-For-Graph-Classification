import torch


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            out = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            loss = criterion(out, batch.y)

            total_loss += loss.item()

            predictions = out.argmax(dim=1)

            correct += (predictions == batch.y).sum().item()
            total += batch.y.size(0)

        average_loss = total_loss / len(loader)
        accuracy = correct / total

        return average_loss, accuracy