# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import argparse
import random
from pathlib import Path

import mpmath as mp
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


class RiemannZeroDataset(Dataset):
    def __init__(
        self, num_zeros: int = 50, negatives: int = 200, imag_max: float = 50.0
    ):
        zeros = [mp.zetazero(i) for i in range(1, num_zeros + 1)]
        pos = np.array(
            [[float(z.real), float(z.imag)] for z in zeros], dtype=np.float32
        )
        pos_labels = np.ones((len(pos), 1), dtype=np.float32)

        rng = np.random.default_rng(0)
        neg_real = rng.uniform(0.0, 1.0, size=(negatives, 1))
        neg_imag = rng.uniform(0.0, imag_max, size=(negatives, 1))
        neg = np.hstack([neg_real, neg_imag]).astype(np.float32)
        neg_labels = np.zeros((len(neg), 1), dtype=np.float32)

        self.data = torch.from_numpy(np.vstack([pos, neg]))
        self.labels = torch.from_numpy(np.vstack([pos_labels, neg_labels]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ZeroPredictor(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def loss_fn(
    pred: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy(pred, target)
    distance = ((inputs[:, 0] - 0.5) ** 2) * target.squeeze(1)
    return bce + distance.mean()


def train_epoch(model, loader, optimizer) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y, x)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def eval_epoch(model, loader) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            loss = loss_fn(out, y, x)
            total += loss.item() * x.size(0)
    return total / len(loader.dataset)


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.best = float("inf")
        self.stop = False

    def __call__(self, loss: float) -> bool:
        if loss < self.best:
            self.best = loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return self.stop


def cross_validate(
    dataset: Dataset, k: int = 5, epochs: int = 20, lr: float = 1e-3
) -> None:
    indices = list(range(len(dataset)))
    fold_size = len(dataset) // k
    for fold in range(k):
        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = list(set(indices) - set(val_idx))
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)
        model = ZeroPredictor()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        stopper = EarlyStopping(patience=3)
        for epoch in range(epochs):
            _ = train_epoch(model, train_loader, opt)
            val_loss = eval_epoch(model, val_loader)
            if stopper(val_loss):
                break
        print(f"Fold {fold+1} validation loss: {stopper.best:.4f}")


def fine_tune(
    model_path: Path, dataset: Dataset, epochs: int = 5, lr: float = 1e-4
) -> None:
    model = ZeroPredictor()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        train_epoch(model, loader, opt)
    torch.save(model.state_dict(), model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model on Riemann zeros")
    parser.add_argument("--folds", type=int, default=3, help="Number of folds for CV")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    seed_everything()
    ds = RiemannZeroDataset()
    cross_validate(ds, k=args.folds, epochs=args.epochs)


if __name__ == "__main__":
    main()
