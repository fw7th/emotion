import torch
import torch.nn as nn
import time
import tqdm
from .utils import unfreezeLayers, EarlyStopping
from .model import createModel


def trainModel(train_loader, val_loader, history):
    base = "/content/drive/MyDrive/emotrain"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = createModel()
    unfreezeLayers(model, 9)  # Pass the model instance to the function
    model.to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.1  # Reduced initial learning rate
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")

    # Training loop
    max_epochs = 70

    early_stopping = EarlyStopping(
        patience=8, delta=0.001, path=f"{base}/mobilenet_v3.pth"
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    for epoch in range(max_epochs):
        print("-" * 50)
        print(f"Epoch {epoch+1}/{max_epochs}")
        start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # tqdm for progress bar
        train_pbar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # gradient clipping
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)

            # Update progress bar every 10 batches
            if batch_idx % 10 == 0:
                train_pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "batch": f"{batch_idx}/{len(train_loader)}",
                    }
                )

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        val_pbar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.float() / len(val_loader.dataset)

        # Step the scheduler after the epoch
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        early_stopping(val_loss, model)

        # Save model at last epoch
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        # Print progress
        epoch_time = time.time() - start_time

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(
            f"Best Val Acc: {best_val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        print(f"Time: {epoch_time:.1f}s")
        print("-" * 50)
        print("\n")

    print(f"Training complete!")
    return model
