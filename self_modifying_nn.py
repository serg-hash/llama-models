# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Simple self-modifying neural network example."""

import os
import sys
from pathlib import Path

import torch
from torch import nn, optim

HIDDEN_SIZE = 16
THRESHOLD = 0.01
MAX_ATTEMPTS = 5


def rewrite_code(new_size: int) -> None:
    """Rewrite this file updating the hidden layer size."""
    path = Path(__file__).resolve()
    lines = path.read_text().splitlines()
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            if line.startswith("HIDDEN_SIZE"):
                fh.write(f"HIDDEN_SIZE = {new_size}\n")
            else:
                fh.write(line + "\n")
    print(f"Updated {path.name} with HIDDEN_SIZE={new_size}")


def train() -> float:
    """Train a minimal network to fit y=2*x."""
    model = nn.Sequential(
        nn.Linear(1, HIDDEN_SIZE), nn.ReLU(), nn.Linear(HIDDEN_SIZE, 1)
    )
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = 2 * x
    for _ in range(1000):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    return float(loss.item())


if __name__ == "__main__":
    attempt = int(os.environ.get("ATTEMPT", "1"))
    final_loss = train()
    print(f"Attempt {attempt}: loss={final_loss:.6f}")
    if final_loss > THRESHOLD and attempt < MAX_ATTEMPTS:
        rewrite_code(HIDDEN_SIZE + 8)
        os.environ["ATTEMPT"] = str(attempt + 1)
        os.execv(sys.executable, [sys.executable, __file__])
