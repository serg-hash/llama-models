# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Self-evolving neural network example.

This script demonstrates a toy model that modifies its own source code to adjust
its architecture after each run if performance falls below a threshold. Each
execution represents one generation in the model's history.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters that may be rewritten
HIDDEN_SIZES = [16, 16]
THRESHOLD = 0.75
MAX_ATTEMPTS = 5

HISTORY_FILE = Path("self_history.json")


def load_attempt() -> int:
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open() as f:
            data = json.load(f)
        return int(data.get("attempt", 0))
    return 0


def save_attempt(attempt: int) -> None:
    with HISTORY_FILE.open("w") as f:
        json.dump({"attempt": attempt}, f)


def build_model(input_dim: int = 10) -> nn.Module:
    layers = []
    prev = input_dim
    for size in HIDDEN_SIZES:
        layers.append(nn.Linear(prev, size))
        layers.append(nn.ReLU())
        prev = size
    layers.append(nn.Linear(prev, 2))
    return nn.Sequential(*layers)


def generate_data(n: int = 256, input_dim: int = 10) -> TensorDataset:
    x = torch.randn(n, input_dim)
    weights = torch.randn(input_dim)
    y = (x @ weights > 0).long()
    return TensorDataset(x, y)


def train(model: nn.Module, data: TensorDataset) -> float:
    loader = DataLoader(data, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(5):
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            preds = model(batch_x).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def update_source(new_sizes: list[int]) -> None:
    path = Path(__file__)
    lines = path.read_text().splitlines()
    with path.open("w") as f:
        for line in lines:
            if line.startswith("HIDDEN_SIZES"):
                f.write(f"HIDDEN_SIZES = {new_sizes}  # auto-updated\n")
            else:
                f.write(f"{line}\n")


def main() -> None:
    attempt = load_attempt()
    if attempt >= MAX_ATTEMPTS:
        print("Max attempts reached. Stopping evolution.")
        return

    model = build_model()
    data = generate_data()
    accuracy = train(model, data)
    print(f"Attempt {attempt}: accuracy={accuracy:.2f}")

    if accuracy < THRESHOLD:
        # mutate architecture
        new_sizes = [s + random.randint(4, 8) for s in HIDDEN_SIZES] + [16]
        update_source(new_sizes)
        save_attempt(attempt + 1)
        print(f"Performance below threshold. Evolving architecture to {new_sizes}.")
    else:
        print("Desired performance achieved. Evolution stops.")


if __name__ == "__main__":
    main()
