# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Example dynamic network with an optimized training loop."""

from __future__ import annotations

import random
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DynamicNetwork(nn.Module):
    """Simple feedforward network with dynamic architecture."""

    def __init__(
        self, input_size: int, hidden_sizes: List[int], output_size: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        last_size = input_size
        for h in hidden_sizes:
            self.layers.append(nn.Linear(last_size, h))
            last_size = h
        self.output_layer = nn.Linear(last_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

    def get_architecture(self) -> List[int]:
        return [layer.out_features for layer in self.layers]


class ArchitectureMutator:
    """Randomly mutate the hidden layer sizes."""

    def mutate(self, hidden_sizes: List[int]) -> List[int]:
        if random.random() < 0.5 and hidden_sizes:
            i = random.randint(0, len(hidden_sizes) - 1)
            hidden_sizes[i] = max(1, hidden_sizes[i] + random.choice([-1, 1]))
        else:
            hidden_sizes.append(random.randint(1, 10))
        return hidden_sizes


def train_and_evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
) -> float:
    """Train ``model`` using ``loader`` and return the final loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.to(device)
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return float(loss.item())


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(200, 3)
    y = torch.randn(200, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

    hidden = [5]
    mutator = ArchitectureMutator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(10):
        net = DynamicNetwork(3, hidden, 1)
        loss = train_and_evaluate(net, loader, device)
        print(
            f"Epoch {epoch + 1} | Loss: {loss:.4f} | Architecture: {net.get_architecture()}"
        )
        hidden = mutator.mutate(hidden)


if __name__ == "__main__":
    main()
