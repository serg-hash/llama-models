"""Demonstration of a simple self-modifying neural network.

This script trains a tiny network to map the index of a Riemann zeta zero to
its imaginary component. If training loss remains high, the script rewrites
itself with a larger hidden layer in order to "evolve" and improve on the next
execution.

The goal is purely educational; it does not actually solve the Riemann
Hypothesis.
"""

import os

import torch
from torch import nn, optim
import mpmath as mp

HIDDEN_SIZE = 16
DATA_POINTS = 10
THRESHOLD = 0.5


def generate_zeros(n: int):
    """Return the imaginary parts of the first n nontrivial zeros."""
    mp.dps = 50
    return [mp.zetazero(i).imag for i in range(1, n + 1)]


class RiemannNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x).relu()
        return self.fc2(x)


def self_modify(new_size: int) -> None:
    """Rewrite this file with an updated hidden size."""
    file_path = os.path.abspath(__file__)
    with open(file_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("HIDDEN_SIZE ="):
            lines[i] = f"HIDDEN_SIZE = {new_size}\n"
    with open(file_path, "w") as f:
        f.writelines(lines)
    print(f"[Self-Evolution] Increased hidden size to {new_size}.")


def train() -> float:
    zeros = generate_zeros(DATA_POINTS)
    indices = torch.arange(1, DATA_POINTS + 1, dtype=torch.float32).unsqueeze(1)
    targets = torch.tensor(zeros, dtype=torch.float32).unsqueeze(1)

    model = RiemannNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for _ in range(200):
        optimizer.zero_grad()
        pred = model(indices)
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
    print(f"Final loss: {loss.item():.4f}")
    return loss.item()


if __name__ == "__main__":
    loss = train()
    if loss > THRESHOLD:
        print(
            "Loss above threshold. The network will attempt to improve itself on"
            " the next run."
        )
        self_modify(HIDDEN_SIZE + 8)
    else:
        print("Training successful. No modification required.")

