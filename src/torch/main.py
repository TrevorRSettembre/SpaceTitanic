# Author: Trevor Settembre
# Project Title: SpaceShip Titanic
# Description: Main training script with KFold, Mixup, Label Smoothing, SWA, and threshold optimization

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch_lr_finder import LRFinder

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from CleanCSV import CleanCSV
from NeuralNet import NeuralNet
from losses import FocalLoss
from early_stopping import MultiMetricEarlyStopping
from NeuralNet import LabelSmoothingCrossEntropyLoss

# ------------------- Mixup ------------------- #
def mixup_data(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------------- Hyperparameters ------------------- #
epochs = 100
kfolds = 5
batch_size = 64
num_workers = 8
patience = 10
alpha_param = 0.1
use_focal_loss = False
use_label_smoothing = True
use_mixup = True
use_compile = False
smoothing_epsilon = 0.02
gNoise_scale = 0.002
ttaPasses = 3
use_swa = True
swa_start = int(epochs * 0.75)

# ------------------- Load Data ------------------- #
train_dataset = CleanCSV("../../data/train.csv", target_column="Transported")
input_size = train_dataset.input_size
output_size = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

classes = np.unique(train_dataset.y.numpy())
class_weights = torch.tensor(
    compute_class_weight('balanced', classes=classes, y=train_dataset.y.numpy()),
    dtype=torch.float32
).to(device)

kf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)
train_loss_curve, val_loss_curve, val_accuracy_curve = [], [], []
fold_accuracies, fold_losses, optimal_thresholds = [], [], []

scaler = torch.amp.GradScaler("cuda", enabled=True)

# ------------------- Training Loop ------------------- #
def custom_update_bn(loader, model, device):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.running_mean.zero_()
            module.running_var.fill_(1)

    n = 0
    for input, _ in loader:
        input = input.to(device)
        b = input.size(0)
        momentum = b / float(n + b)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.momentum = momentum
        model(input)
        n += b

for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset.X, train_dataset.y)):
    print(f"\nFold {fold+1}/{kfolds}")

    train_dl = DataLoader(torch.utils.data.Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(torch.utils.data.Subset(train_dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = NeuralNet(input_size, output_size, device).to(device)
    if use_compile:
        model = torch.compile(model)

    criterion = LabelSmoothingCrossEntropyLoss(epsilon=smoothing_epsilon) if use_label_smoothing else (
        FocalLoss(gamma=1.0, weight=class_weights) if use_focal_loss else nn.CrossEntropyLoss(weight=class_weights))
    model.criterion = criterion

    # LR Finder
    print(f"🔍 Running LR Finder for Fold {fold + 1}...")
    lr_finder = LRFinder(model, optimizer=optim.AdamW(model.parameters()), criterion=nn.CrossEntropyLoss(), device=device)
    lr_finder.range_test(train_dl, start_lr=1e-4, end_lr=5e-2, num_iter=100)
    losses = lr_finder.history['loss']
    lr = lr_finder.history['lr'][np.gradient(losses).argmin()] if all(np.isfinite(losses)) else 1e-3
    lr_finder.reset()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=lr)
    early_stopping = MultiMetricEarlyStopping(monitor_metrics=["f1"], mode="max", patience=patience,verbose=True, stop_mode="any")

    for epoch in range(epochs):
        print(f"\n📘 Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in tqdm(train_dl):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if use_mixup:
                mixed_x, y_a, y_b, lam = mixup_data(x_batch, y_batch, alpha_param)
                output = model(mixed_x)
                loss = mixup_loss(criterion, output, y_a, y_b, lam)
            else:
                output = model(x_batch)
                loss = criterion(output, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.detach().item()

        avg_train_loss = running_loss / len(train_dl)

        # Validation + Threshold Sweep
        model.eval()
        val_probs, val_targets = [], []
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                loss, output = model.compute_loss(x_batch, y_batch)
                val_loss += loss.detach().item()
                probs = torch.softmax(output, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
                correct += (output.argmax(1) == y_batch).sum().item()

        val_probs, val_targets = np.array(val_probs), np.array(val_targets)
        accuracy = correct / len(val_dl.dataset)

        best_thresh, best_f1 = 0.5, 0
        for t in np.linspace(0.3, 0.7, 41):
            score = f1_score(val_targets, (val_probs > t).astype(int))
            if score > best_f1:
                best_f1, best_thresh = score, t

        print(f"Fold {fold+1} Best F1: {best_f1:.4f} @ Threshold: {best_thresh:.2f}")
        optimal_thresholds.append(best_thresh)

        train_loss_curve.append(avg_train_loss)
        val_loss_curve.append(val_loss / len(val_dl))
        val_accuracy_curve.append(accuracy)

        print(f"✅ Val Accuracy: {accuracy * 100:.2f}%, Val Loss: {val_loss:.4f}")
        early_stopping({"f1": best_f1}, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered.")
            break
        if use_swa and epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

    if use_swa:
        custom_update_bn(train_dl, swa_model, device)
        model.load_state_dict(swa_model.module.state_dict())

    early_stopping.load_best_model(model)
    torch.save(model.state_dict(), f"models/fold_{fold}.pth")
    fold_accuracies.append(accuracy)
    fold_losses.append(val_loss)

# ------------------- Inference ------------------- #
print(f"Avg Accuracy: {np.mean(fold_accuracies):.4f}")
final_thresh = np.mean(optimal_thresholds)
print(f"Using averaged threshold: {final_thresh:.4f}")

test_dataset = CleanCSV("../../data/test.csv")
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

ensemble_outputs = []
for fold in range(kfolds):
    model = NeuralNet(input_size, output_size, device).to(device)
    model.load_state_dict(torch.load(f"models/fold_{fold}.pth"))
    model.eval()

    fold_probs = []
    with torch.no_grad():
        for x_batch, _ in test_dl:
            x_batch = x_batch.to(device)
            tta_probs = []
            for _ in range(ttaPasses):
                noisy_input = x_batch + torch.randn_like(x_batch) * gNoise_scale
                output = model(noisy_input)
                tta_probs.append(torch.softmax(output, dim=1)[:, 1].cpu().numpy())
            fold_probs.append(np.mean(tta_probs, axis=0))

    ensemble_outputs.append(np.concatenate(fold_probs))

final_probs = np.mean(ensemble_outputs, axis=0)
final_probs = np.clip(final_probs, 0.05, 0.95)

# Diagnostic stats
print(f"🔍 Prediction confidence stats → Min: {final_probs.min():.4f}, Max: {final_probs.max():.4f}, Mean: {final_probs.mean():.4f}")
print(f"⚠️  Original validation-based threshold: {final_thresh:.4f}")

# Blended threshold: average of validation threshold and 0.5
submission_thresh = (final_thresh + 0.6) / 2
print(f"🧪 Blended submission threshold: {submission_thresh:.4f}")

# Apply threshold to probabilities
final_preds = final_probs > submission_thresh

# Show result counts
print(f"🧮 #Predicted True: {(final_preds).sum()}, False: {(~final_preds).sum()}")

plt.figure(figsize=(8, 4))
plt.hist(final_probs, bins=50, color='skyblue', edgecolor='black')
plt.axvline(submission_thresh, color='red', linestyle='--', label=f"Threshold = {submission_thresh:.2f}")
plt.title("Distribution of Final Prediction Probabilities")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("final_prob_distribution.png")
print("📈 Saved final_prob_distribution.png")




submission_df = pd.DataFrame({
    "PassengerId": test_dataset.passenger_ids,
    "Transported": final_preds.astype(bool)
})

os.makedirs("../../output", exist_ok=True)
submission_df.to_csv("../../output/submission.csv", index=False)

# ------------------- Plot ------------------- #
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_curve, label="Train Loss")
plt.plot(val_loss_curve, label="Val Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([a * 100 for a in val_accuracy_curve], label="Val Accuracy")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
print("✅ Saved training curves and submission")
