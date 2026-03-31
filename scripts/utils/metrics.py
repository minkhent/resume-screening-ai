import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

def compute_metrics(y_true, y_pred, classes):
    acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),  # ✅ ensure all classes included
        target_names=classes,
        output_dict=True,
        zero_division=0                   # ✅ fix warning properly
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes)))  # ✅ aligned with classes
    )

    return acc, report, cm


def save_metrics_csv(metrics_df, path):
    metrics_df.to_csv(path, index=False)


def plot_confusion_matrix(cm, classes, path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues"
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_loss_acc(metrics_df, save_dir):
    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['epoch'], metrics_df['train_acc'], label='Train Acc')
    plt.plot(metrics_df['epoch'], metrics_df['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/accuracy_curve.png")
    plt.close()