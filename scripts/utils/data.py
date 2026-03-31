import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from sentence_transformers import InputExample
import torch

def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path).drop_duplicates()
    df["Resume_str"] = df["Resume_str"].fillna("")
    df["Category"] = df["Category"].fillna("Unknown")

    return df


def encode_labels(df, save_path=None):
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Category"])

    if save_path:
        import joblib
        joblib.dump(le, save_path)

    return df, le


def build_dataloader(df, batch_size, weighted=True):
    examples = [
        InputExample(texts=[text], label=int(label))
        for text, label in zip(df["Resume_str"], df["label"])
    ]

    def collate_fn(batch):
        texts = [ex.texts[0] for ex in batch]
        labels = torch.tensor([ex.label for ex in batch])
        return texts, labels

    if weighted:
        labels = df["label"].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        return DataLoader(
            examples,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn
        )

    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


def train_val_split(df, test_size=0.2, seed=42):
    return train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed
    )