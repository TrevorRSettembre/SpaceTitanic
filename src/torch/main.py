
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from torch_lr_finder import LRFinder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from CleanCSV import CleanCSV
from NeuralNet import NeuralNet
from final_train import retrain_final_model
from losses import FocalLoss
from early_stopping import MultiMetricEarlyStopping

# ------------------- Hyperparameters ------------------- #
epochs = 150
kfolds = 5
batch_size = 64
num_workers = 8
lr = None
max_lr = None
patience = 5
use_focal_loss = True

# ------------------- Load Data ------------------- #
train_path = "../../data/train.csv"
train_dataset = CleanCSV(train_path, target_column="Transported")
input_size = train_dataset.input_size
output_size = 2  # binary classification with 2 logits
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")
print(f"🚦 Dataset size: {len(train_dataset)}")
print(f"🧪 Sample input shape: {train_dataset[0][0].shape}")
print(f"🎯 Sample target: {train_dataset[0][1]}")
print("🎯 Unique labels in dataset:", torch.unique(train_dataset.y))

# ------------------- Class Weights ------------------- #
train_dataset.y = train_dataset.y.long()
classes = np.unique(train_dataset.y.numpy())
class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=train_dataset.y.numpy())
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
print(f"📊 Class Weights: {class_weights}")

# ------------------- AMP ------------------- #
scaler = torch.amp.GradScaler(enabled=True)

# ------------------- Cross-Validation ------------------- #
kf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)
fold_accuracies = []
fold_losses = []
train_loss_curve = []
val_loss_curve = []
val_accuracy_curve = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset.X, train_dataset.y)):
    print(f"Fold {fold+1} — Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    print(f"\n{'='*40}\n📁 Fold {fold+1}/{kfolds}")

    # DataLoaders
    train_dl = DataLoader(torch.utils.data.Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(torch.utils.data.Subset(train_dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model
    model = NeuralNet(input_size, output_size, device, task_type='binary').to(device)
    criterion = FocalLoss(weight=class_weights) if use_focal_loss else nn.CrossEntropyLoss(weight=class_weights)
    model.criterion = criterion

    # LR Finder (first fold only)
    if fold == 0:
        print("🔍 Running LR Finder...")
        stable_criterion = nn.CrossEntropyLoss(weight=class_weights)
        lr_finder_optimizer = optim.AdamW(model.parameters(), lr=5e-5)
        lr_finder = LRFinder(model, optimizer=lr_finder_optimizer, criterion=stable_criterion, device=device)
        lr_finder.range_test(train_dl, start_lr=5e-5, end_lr=1e-1, num_iter=100)

        losses = lr_finder.history.get('loss', [])
        if all(np.isfinite(l) and l > 0 for l in losses):
            lr_finder.plot()
            os.makedirs("lr_plots", exist_ok=True)
            filename = f"lr_finder_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(os.path.join("lr_plots", filename))
            print(f"📈 LR plot saved to: lr_plots/{filename}")
        else:
            print("⚠️ Skipping LR plot due to invalid or non-positive loss values.")

        print("📉 LR Finder raw losses (first 10):", losses[:10])
        print("📉 LRs (first 10):", lr_finder.history.get('lr', [])[:10])

        if losses and all(np.isfinite(l) for l in losses):
            lr = lr_finder.history['lr'][np.gradient(losses).argmin()]
            max_lr = lr * 10
            print(f"✅ Suggested learning rate: {lr:.2e}")
        else:
            print("⚠️ Falling back to default LR values.")
            lr, max_lr = 1e-3, 1e-2

        lr_finder.reset()

    if lr is None or max_lr is None:
        lr = 1e-3
        max_lr = 1e-2

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=epochs)

    early_stopping = MultiMetricEarlyStopping(
        monitor_metrics=["val_loss", "accuracy"],
        mode={"val_loss": "min", "accuracy": "max"},
        patience=patience,
        delta=1e-4,
        verbose=True,
        stop_mode="any"
    )

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in tqdm(train_dl, desc="Training", unit="batch"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, output = model.compute_loss(x_batch, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.detach().item()

        avg_train_loss = running_loss / len(train_dl)
        print(f"Avg Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_dl, desc="Evaluating", unit="batch"):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss, output = model.compute_loss(x_batch, y_batch)
                val_loss += loss.detach().item()
                correct += (output.argmax(1) == y_batch).sum().item()
        if epoch == 0 and fold == 0:
            print("🔍 Sample preds:", output.argmax(1)[:10].cpu().numpy())
            print("🔍 Actual:", y_batch[:10].cpu().numpy())
            print("🧠 Output logits sample:", output[:5].detach().cpu().numpy())

        val_loss /= len(val_dl)
        accuracy = correct / len(val_dl.dataset)
        print(f"✅ Val Accuracy: {accuracy * 100:.2f}%, Val Loss: {val_loss:.4f}")
        train_loss_curve.append(avg_train_loss)
        val_loss_curve.append(val_loss)
        val_accuracy_curve.append(accuracy)
        scheduler.step()
        early_stopping({"val_loss": val_loss, "accuracy": accuracy}, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered.")
            break

    early_stopping.load_best_model(model)
    fold_accuracies.append(accuracy)
    fold_losses.append(val_loss)

print(f"\n{'='*40}")
print(f"📈 Average Accuracy across {kfolds} folds: {np.mean(fold_accuracies) * 100:.2f}%")
print(f"📉 Average Loss: {np.mean(fold_losses):.4f}")

final_criterion = FocalLoss(weight=class_weights) if use_focal_loss else nn.CrossEntropyLoss(weight=class_weights)
final_model = retrain_final_model(
    train_dataset=train_dataset,
    input_size=input_size,
    output_size=output_size,
    device=device,
    criterion=final_criterion,
    epochs=25,
    batch_size=batch_size,
    num_workers=num_workers,
    lr=lr,
    max_lr=max_lr,
    use_label_smoothing=True,
    smoothing_epsilon=0.05,
)


test_path = "../../data/test.csv"
test_dataset = CleanCSV(test_path)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

final_model.eval()
predictions, ids = [], []
with torch.no_grad():
    for x_batch, passenger_ids in test_dl:
        x_batch = x_batch.to(device)
        output = final_model(x_batch)
        preds = (output.softmax(dim=1)[:, 1] > 0.5).int().cpu().numpy()
        if isinstance(passenger_ids, torch.Tensor):
            passenger_ids = passenger_ids.cpu().numpy().astype(str)
        predictions.extend(preds)
        ids.extend(passenger_ids)

bool_predictions = [p == 1 for p in predictions]
os.makedirs("../../output", exist_ok=True)
submission_df = pd.DataFrame({"PassengerId": ids, "Transported": bool_predictions})
submission_df.to_csv("../../output/submission.csv", index=False)
print("✅ Submission file saved.")


# ------------------- Plotting ------------------- #
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_curve, label="Train Loss")
plt.plot(val_loss_curve, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([acc * 100 for acc in val_accuracy_curve], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
print("📈 Training curves saved as training_metrics.png")
