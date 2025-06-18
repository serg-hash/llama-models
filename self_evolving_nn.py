# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Self-evolving neural network demonstration.

This script trains a simple feed-forward network to predict the real and
imaginary parts of the first few non-trivial zeros of the Riemann zeta
function. If the loss remains above a threshold, it rewrites itself with a
larger hidden size and an incremented generation counter.

The code is purely educational and does not actually solve the Riemann
Hypothesis. It demonstrates one approach to self-modifying code for
illustrative purposes.
"""

from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from riemann_zero_explorer import compute_zero

# Hyperparameters that may be modified during self-rewrite
GENERATION = 1
HIDDEN_SIZE = 16
LEARNING_RATE = 0.01
THRESHOLD = 0.1
MAX_GENERATIONS = 5


class ZeroDataset(torch.utils.data.Dataset):
    """Dataset of Riemann zeta zeros."""

    def __init__(self, n: int):
        zeros: List[complex] = [complex(compute_zero(i)) for i in range(1, n + 1)]
        self.inputs = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(1)
        self.targets = torch.tensor(
            [[z.real, z.imag] for z in zeros], dtype=torch.float32
        )

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


def create_model(hidden_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )


def train_model(model: nn.Module, dataset: ZeroDataset) -> float:
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for _ in range(200):
        for x, y in loader:
            optim.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()
    return loss.item()


def rewrite_self(new_hidden_size: int, new_generation: int) -> None:
    """Rewrite this file with updated hyperparameters."""
    path = Path(__file__)
    source = path.read_text().splitlines()
    for idx, line in enumerate(source):
        if line.startswith("GENERATION ="):
            source[idx] = f"GENERATION = {new_generation}"
        elif line.startswith("HIDDEN_SIZE ="):
            source[idx] = f"HIDDEN_SIZE = {new_hidden_size}"
    path.write_text("\n".join(source) + "\n")


def main() -> None:
    print(f"Generation {GENERATION}, hidden size {HIDDEN_SIZE}")
    dataset = ZeroDataset(n=5)
    model = create_model(HIDDEN_SIZE)
    loss = train_model(model, dataset)
    print(f"Final loss: {loss:.4f}")

    if loss > THRESHOLD:
        if GENERATION >= MAX_GENERATIONS:
            print("Max generations reached without achieving threshold.")
            return
        print("Loss above threshold, rewriting with larger hidden size...")
        rewrite_self(HIDDEN_SIZE + 8, GENERATION + 1)
        print("Re-run the script to continue evolution.")
    else:
        print("Threshold met. Evolution complete.")


if __name__ == "__main__":
    main()
