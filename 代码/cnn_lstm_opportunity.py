import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# =====================================================
# 0. Device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =====================================================
# 1. Load Data
# =====================================================
X = np.load("X.npy")          # (N,64,15)
y = np.load("y.npy")          # (N,)
groups = np.load("groups.npy")  # (N,) subject ids

if len(y.shape) > 1:
    y = np.argmax(y, axis=1)

NUM_CLASSES = len(np.unique(y))

print("X shape:", X.shape)
print("y shape:", y.shape)
print("groups shape:", groups.shape)
print("subjects:", np.unique(groups))
print("num subjects:", len(np.unique(groups)))
print("NUM_CLASSES:", NUM_CLASSES)

# =====================================================
# 2. reshape 参数
# =====================================================
# 64 -> 4 x 16
n_steps = 4
n_length = X.shape[1] // n_steps
n_features = X.shape[2]

print("n_steps:", n_steps)
print("n_length:", n_length)
print("n_features:", n_features)

# =====================================================
# 3. CNN-LSTM Model
# =====================================================
class CNNLSTM(nn.Module):
    def __init__(self, n_features, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.SELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SELU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        B, T, L, F = x.shape
        x = x.view(B*T, L, F)
        x = x.permute(0, 2, 1)

        x = self.cnn(x)
        x = x.squeeze(-1)

        x = x.view(B, T, 64)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

N_SPLITS = 4   
EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 30
SEED = 42

num_groups = len(np.unique(groups))
N_SPLITS = min(N_SPLITS, num_groups)

print("Using GroupKFold n_splits =", N_SPLITS)

gkf = GroupKFold(n_splits=N_SPLITS)

metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}

total_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

# =====================================================
# 5. Cross Validation
# =====================================================
fold_no = 1

for train_idx, test_idx in gkf.split(X, y, groups):

    print("\n==============================")
    print(f"Fold {fold_no}")

    X_train_full = X[train_idx]
    y_train_full = y[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    group_train = groups[train_idx]

    # -------------------------------------------------
    # train / val split (按subject切)
    # -------------------------------------------------
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1,
        random_state=SEED
    )

    tr_idx, val_idx = next(
        gss.split(X_train_full, y_train_full, group_train)
    )

    X_train = X_train_full[tr_idx]
    y_train = y_train_full[tr_idx]

    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    # -------------------------------------------------
    # fold内标准化（避免泄漏）
    # -------------------------------------------------
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_test  = (X_test - mean) / std

    # -------------------------------------------------
    # reshape
    # -------------------------------------------------
    X_train = X_train.reshape(-1, n_steps, n_length, n_features)
    X_val   = X_val.reshape(-1, n_steps, n_length, n_features)
    X_test  = X_test.reshape(-1, n_steps, n_length, n_features)

    # -------------------------------------------------
    # dataloader
    # -------------------------------------------------
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_X_tensor = torch.tensor(
        X_test, dtype=torch.float32
    ).to(device)

    test_y_tensor = torch.tensor(
        y_test, dtype=torch.long
    ).to(device)

    # =================================================
    # Model
    # =================================================
    model = CNNLSTM(n_features, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8
    )

    best_val_loss = float("inf")
    best_state = None
    counter = 0

    # =================================================
    # Train
    # =================================================
    for epoch in range(EPOCHS):

        # -------------------------
        # train
        # -------------------------
        model.train()

        train_loss_sum = 0
        train_count = 0

        for bx, by in train_loader:

            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()

            out = model(bx)
            loss = criterion(out, by)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * bx.size(0)
            train_count += bx.size(0)

        train_loss = train_loss_sum / train_count

        # -------------------------
        # val
        # -------------------------
        model.eval()

        val_loss_sum = 0
        val_count = 0
        val_correct = 0  # 统计预测正确的数量

        with torch.no_grad():

            for vx, vy in val_loader:

                vx = vx.to(device)
                vy = vy.to(device)

                out = model(vx)
                loss = criterion(out, vy)

                val_loss_sum += loss.item() * vx.size(0)
                val_count += vx.size(0)

                # 计算准确率
                preds = torch.argmax(out, dim=1)
                val_correct += torch.sum(preds == vy).item()

        val_loss = val_loss_sum / val_count
        val_accuracy = val_correct / val_count  # 计算验证集准确率

        scheduler.step(val_loss)

        # 每10个epoch打印一次验证集准确率
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d} | "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_accuracy={val_accuracy * 100:.2f}%"
            )

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= PATIENCE:
            print("Early stopping")
            break

    # =================================================
    # Test
    # =================================================
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(test_X_tensor)
        preds = torch.argmax(logits, dim=1)

    y_true = test_y_tensor.cpu().numpy()
    y_pred = preds.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )
    rec = recall_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    total_conf_matrix += cm

    metrics["accuracy"].append(acc)
    metrics["precision"].append(prec)
    metrics["recall"].append(rec)
    metrics["f1_score"].append(f1)

    print(
        f"Fold {fold_no} Result | "
        f"ACC={acc*100:.2f}% "
        f"F1={f1:.4f}"
    )

    fold_no += 1

# =====================================================
# Final Result
# =====================================================
print("\n==============================")
print("Final GroupKFold Result")

for name, vals in metrics.items():

    mean = np.mean(vals)
    std = np.std(vals)

    if name == "accuracy":
        print(f"{name}: {mean*100:.2f}% ± {std*100:.2f}%")
    else:
        print(f"{name}: {mean:.4f} ± {std:.4f}")

# =====================================================
# Confusion Matrix
# =====================================================
plt.figure(figsize=(10, 8))

sns.heatmap(
    total_conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Total Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_total.png", dpi=300)

print("Saved confusion_matrix_total.png")

# =====================================================
# Excel
# =====================================================
excel_data = {
    "模型名称": ["CNNLSTM"],
    "Accuracy (%)": [round(np.mean(metrics["accuracy"]) * 100, 2)],
    "Precision": [round(np.mean(metrics["precision"]), 4)],
    "Recall": [round(np.mean(metrics["recall"]), 4)],
    "F1_Score": [round(np.mean(metrics["f1_score"]), 4)]
}

df = pd.DataFrame(excel_data)

excel_filename = "Model_Evaluation_Metrics.xlsx"
df.to_excel(excel_filename, index=False)

print("Saved:", excel_filename)