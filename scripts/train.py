import os
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from utils.logger import setup_logger
from utils.seed import set_seed
from utils.data import load_data, encode_labels, build_dataloader, train_val_split
from utils.metrics import compute_metrics, save_metrics_csv, plot_confusion_matrix, plot_loss_acc
from plot_per_class_metrics import plot_per_class_metrics, plot_topk_accuracy
from model import SentenceTransformerWithHead


@dataclass
class Config:
    model_name: str = "all-MiniLM-L6-v2"
    data_path: str = "data/resume.csv"
    save_dir: str = "models/resume_model_v1"
    batch_size: int = 32
    epochs: int = 50      
    lr: float = 2e-5
    seed: int = 42
    log_file: str = "train.log"
    topk: tuple = (1, 3, 5)


cfg = Config()

os.makedirs(cfg.save_dir, exist_ok=True)
os.makedirs(os.path.join(cfg.save_dir, "plots"), exist_ok=True)

logger = setup_logger(cfg.log_file, cfg.save_dir)
set_seed(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# --------------------
# Load Data
# --------------------
df = load_data(cfg.data_path)
df, le = encode_labels(df, os.path.join(cfg.save_dir, "label_encoder.pkl"))

train_df, val_df = train_val_split(df, seed=cfg.seed)

train_loader = build_dataloader(train_df, cfg.batch_size, weighted=True)
val_loader = build_dataloader(val_df, cfg.batch_size, weighted=False)

# --------------------
# Model
# --------------------
model = SentenceTransformerWithHead(cfg.model_name, len(le.classes_)).to(device)

# ✅ Class weights
class_counts = train_df["label"].value_counts().sort_index().values
weights = 1.0 / class_counts
weights = weights / weights.sum()

class_weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=cfg.lr)

total_steps = len(train_loader) * cfg.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

metrics_list = []
all_val_labels = []
all_val_probs = []

# --------------------
# Training
# --------------------
for epoch in range(cfg.epochs):
    model.train()
    total_loss = 0
    train_preds, train_labels = [], []

    for texts, labels in train_loader:
        labels = labels.to(device)

        optimizer.zero_grad()

        inputs = model.transformer.tokenize(texts)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())

    train_loss = total_loss / len(train_loader)
    train_acc, _, _ = compute_metrics(train_labels, train_preds, le.classes_)

    # --------------------
    # Validation
    # --------------------
    model.eval()
    total_loss = 0
    val_preds, val_labels_epoch = [], []
    val_probs_epoch = []

    with torch.no_grad():
        for texts, labels in val_loader:
            labels = labels.to(device)

            inputs = model.transformer.tokenize(texts)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            val_preds.extend(preds)
            val_labels_epoch.extend(labels.cpu().numpy())
            val_probs_epoch.append(probs)

    val_loss = total_loss / len(val_loader)

    val_acc, report, cm = compute_metrics(
        val_labels_epoch, val_preds, le.classes_
    )

    metrics_list.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

    # ✅ FIX top-k accumulation
    all_val_labels.extend(val_labels_epoch)
    all_val_probs.append(np.vstack(val_probs_epoch))

    # ✅ DEBUG (IMPORTANT)
    unique_preds, counts = np.unique(val_preds, return_counts=True)
    logger.info(f"Epoch {epoch+1} Prediction Dist: {dict(zip(unique_preds, counts))}")

    logger.info(
        f"Epoch {epoch+1} | "
        f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
        f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
    )

# --------------------
# Save outputs
# --------------------
metrics_df = pd.DataFrame(metrics_list)

save_metrics_csv(metrics_df, os.path.join(cfg.save_dir, "metrics.csv"))

plot_confusion_matrix(
    cm,
    le.classes_,
    os.path.join(cfg.save_dir, "plots", "confusion_matrix.png")
)

plot_loss_acc(metrics_df, os.path.join(cfg.save_dir, "plots"))

pd.DataFrame(report).transpose().to_csv(
    os.path.join(cfg.save_dir, "classification_report.csv")
)

plot_per_class_metrics(
    os.path.join(cfg.save_dir, "classification_report.csv"),
    os.path.join(cfg.save_dir, "plots")
)

# ✅ Fix stacked probs
all_val_probs = np.vstack(all_val_probs)

plot_topk_accuracy(
    np.array(all_val_labels),
    all_val_probs,
    le.classes_,
    os.path.join(cfg.save_dir, "plots"),
    ks=cfg.topk
)

# --------------------
# Save model
# --------------------
model.transformer.save(os.path.join(cfg.save_dir, "transformer"))
torch.save(model.head.state_dict(), os.path.join(cfg.save_dir, "classifier.pt"))

logger.info("Training complete and model saved successfully.")