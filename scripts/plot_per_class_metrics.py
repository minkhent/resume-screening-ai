import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import top_k_accuracy_score

sns.set(style="whitegrid")


def plot_per_class_metrics(classification_report_csv, save_dir):
    """
    Plots per-class precision, recall, F1-score bar charts.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(classification_report_csv, index_col=0)
    
    # Filter out aggregate rows like 'accuracy', 'macro avg', 'weighted avg'
    class_metrics = df.loc[~df.index.str.contains("avg|accuracy")]
    
    # Precision plot
    plt.figure(figsize=(12,6))
    sns.barplot(x=class_metrics.index, y=class_metrics["precision"])
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.ylabel("Precision")
    plt.title("Per-class Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_precision.png"))
    plt.close()
    
    # Recall plot
    plt.figure(figsize=(12,6))
    sns.barplot(x=class_metrics.index, y=class_metrics["recall"])
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.ylabel("Recall")
    plt.title("Per-class Recall")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_recall.png"))
    plt.close()
    
    # F1-score plot
    plt.figure(figsize=(12,6))
    sns.barplot(x=class_metrics.index, y=class_metrics["f1-score"])
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.ylabel("F1 Score")
    plt.title("Per-class F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_f1.png"))
    plt.close()


def plot_topk_accuracy(y_true, y_probs, class_names, save_dir, ks=[1,3,5]):
    """
    Computes and plots top-k accuracy for given k values.
    y_true : ground-truth labels
    y_probs : model output probabilities (numpy array shape: n_samples x n_classes)
    ks : list of k values for top-k
    """
    import numpy as np
    os.makedirs(save_dir, exist_ok=True)
    
    topk_results = {}
    for k in ks:
        topk_acc = top_k_accuracy_score(y_true, y_probs, k=k)
        topk_results[f"top-{k}"] = topk_acc
    # Plot
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(topk_results.keys()), y=list(topk_results.values()))
    plt.ylim(0,1)
    plt.ylabel("Top-k Accuracy")
    plt.title("Top-k Accuracy")
    plt.savefig(os.path.join(save_dir, "topk_accuracy.png"))
    plt.close()


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    classification_report_csv = "models/resume_model_v1/classification_report.csv"
    save_dir = "models/resume_model_v1/plots"
    
    # Per-class metrics plots
    plot_per_class_metrics(classification_report_csv, save_dir)
