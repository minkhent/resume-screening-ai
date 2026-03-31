# src/model.py

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, features):
        x = features['sentence_embedding']
        x = self.dropout(x)
        return self.linear(x)


class SentenceTransformerWithHead(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.transformer = SentenceTransformer(model_name)
        self.head = ClassificationHead(
            self.transformer.get_sentence_embedding_dimension(),
            num_classes
        )

    def forward(self, inputs):
        features = self.transformer(inputs)
        return self.head(features)