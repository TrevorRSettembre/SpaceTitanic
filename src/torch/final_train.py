import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # For mixed precision

from NeuralNet import NeuralNet, LabelSmoothingCrossEntropyLoss
from early_stopping import MultiMetricEarlyStopping

def retrain_final_model(
    train_dataset,
    input_size,
    output_size,
    device,
    task_type='binary',
    epochs=100,
    batch_size=32,
    num_workers=8,
    criterion=None,
    lr=1e-3,
    max_lr=1e-2,
    use_label_smoothing=False,
    smoothing_epsilon=0.1
):
    print(f"\n{'='*30}\nRetraining on Full Dataset\n{'='*30}")

    # Stratified split for final validation set
    y = train_dataset.y
    indices = np.arange(len(train_dataset))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, stratify=y, random_state=42
    )
    final_train_set = Subset(train_dataset, train_idx)
    final_val_set = Subset(train_dataset, val_idx)

    # DataLoaders
    train_dl = DataLoader(final_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(final_val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model
    model = NeuralNet(input_size, output_size, device, task_type=task_type).to(device)
    if use_label_smoothing:
        model.criterion = LabelSmoothingCrossEntropyLoss(epsilon=smoothing_epsilon)
    elif criterion is not None:
        model.criterion = criterion
    else:
        raise ValueError("A loss function must be provided either via 'criterion' or 'use_label_smoothing=True'.")

    # Optimizer and LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=epochs)

    # Early stopping
    early_stopping = MultiMetricEarlyStopping(
        monitor_metrics=["val_loss", "accuracy"],
        mode={"val_loss": "min", "accuracy": "max"},
        patience=10,
        delta=1e-4,
        verbose=True,
        stop_mode="any"
    )

    # Mixed precision
    scaler = GradScaler(enabled=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)

        model.train()
        running_loss = 0.0
        with tqdm(train_dl, desc=f"Training Epoch {epoch + 1}", unit="batch") as pbar:
            for batch_idx, (x_batch, y_batch) in enumerate(pbar):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss, _ = model.compute_loss(x_batch, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.detach().item()
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix(loss=loss.item(), avg_loss=avg_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            with tqdm(val_dl, desc="Validating", unit="batch") as pbar:
                for x_val, y_val in pbar:
                    x_val, y_val = x_val.to(device), y_val.to(device)

                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        loss, output = model.compute_loss(x_val, y_val)

                    val_loss += loss.item()
                    correct += (output.argmax(1) == y_val).sum().item()
                    pbar.set_postfix(val_loss=val_loss / len(val_dl))

        avg_train_loss = running_loss / len(train_dl)
        avg_val_loss = val_loss / len(val_dl)
        val_accuracy = correct / len(final_val_set)
        scheduler.step()

        print(f"Epoch {epoch + 1} Summary | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy * 100:.2f}%")

        metrics = {"val_loss": avg_val_loss, "accuracy": val_accuracy}
        early_stopping(metrics, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    early_stopping.load_best_model(model)
    print("✅ Final model trained and best weights loaded.")
    return model