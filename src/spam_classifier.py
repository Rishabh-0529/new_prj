"""Spam email classifier using TF-IDF or word embeddings with PyTorch."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    data_path: str
    mode: str
    batch_size: int
    epochs: int
    lr: float
    max_features: int
    min_freq: int
    seed: int
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


class TfidfDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: Sequence[int]) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class EmbeddingDataset(Dataset):
    def __init__(self, token_ids: List[List[int]], labels: Sequence[int]) -> None:
        self.token_ids = token_ids
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[List[int], torch.Tensor]:
        return self.token_ids[idx], self.labels[idx]


class SpamClassifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


class EmbeddingSpamClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens, offsets)
        return self.linear(embedded).squeeze(1)


def build_vocab(texts: Iterable[str], min_freq: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counts.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def tokenize(text: str) -> List[str]:
    return [token.strip(".,!?\"'()[]{}:;").lower() for token in text.split() if token]


def encode_texts(texts: Sequence[str], vocab: Dict[str, int]) -> List[List[int]]:
    token_ids: List[List[int]] = []
    for text in texts:
        ids = [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)]
        token_ids.append(ids if ids else [vocab["<unk>"]])
    return token_ids


def collate_embedding(batch: Sequence[Tuple[List[int], torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_lists, labels = zip(*batch)
    tokens = torch.tensor([token for tokens in token_lists for token in tokens], dtype=torch.long)
    offsets = torch.tensor([0] + list(np.cumsum([len(tokens) for tokens in token_lists])[:-1]), dtype=torch.long)
    labels_tensor = torch.stack(labels)
    return tokens, offsets, labels_tensor


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    mode: str,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        if mode == "tfidf":
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
        else:
            tokens, offsets, labels = batch
            tokens = tokens.to(device)
            offsets = offsets.to(device)
            labels = labels.to(device)
            logits = model(tokens, offsets)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    mode: str,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if mode == "tfidf":
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
            else:
                tokens, offsets, labels = batch
                tokens = tokens.to(device)
                offsets = offsets.to(device)
                labels = labels.to(device)
                logits = model(tokens, offsets)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def build_tfidf_loaders(
    texts: Sequence[str],
    labels: Sequence[int],
    config: Config,
) -> Tuple[DataLoader, DataLoader, int]:
    vectorizer = TfidfVectorizer(max_features=config.max_features)
    features = vectorizer.fit_transform(texts).toarray()
    x_train, x_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=config.seed, stratify=labels
    )
    train_ds = TfidfDataset(x_train, y_train)
    val_ds = TfidfDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    return train_loader, val_loader, features.shape[1]


def build_embedding_loaders(
    texts: Sequence[str],
    labels: Sequence[int],
    config: Config,
) -> Tuple[DataLoader, DataLoader, int]:
    vocab = build_vocab(texts, config.min_freq)
    token_ids = encode_texts(texts, vocab)
    x_train, x_val, y_train, y_val = train_test_split(
        token_ids, labels, test_size=0.2, random_state=config.seed, stratify=labels
    )
    train_ds = EmbeddingDataset(x_train, y_train)
    val_ds = EmbeddingDataset(x_val, y_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_embedding,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        collate_fn=collate_embedding,
    )
    return train_loader, val_loader, len(vocab)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Spam email classifier")
    parser.add_argument("--data-path", default="data/sample_spam.csv")
    parser.add_argument("--mode", choices=["tfidf", "embedding"], default="tfidf")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    texts, labels = load_dataset(config.data_path)

    if config.mode == "tfidf":
        train_loader, val_loader, input_dim = build_tfidf_loaders(texts, labels, config)
        model: nn.Module = SpamClassifier(input_dim)
    else:
        train_loader, val_loader, vocab_size = build_embedding_loaders(texts, labels, config)
        model = EmbeddingSpamClassifier(vocab_size)

    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, config.device, config.mode)
        val_acc = evaluate(model, val_loader, config.device, config.mode)
        print(f"Epoch {epoch}: loss={train_loss:.4f} val_acc={val_acc:.2%}")


if __name__ == "__main__":
    main()
