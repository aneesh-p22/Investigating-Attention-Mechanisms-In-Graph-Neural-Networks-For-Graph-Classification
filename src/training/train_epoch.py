def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    return average_loss