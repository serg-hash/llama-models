# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import torch

from riemann_zero_explorer import find_zeros
from torch import nn


class SelfModifyingAgent:
    """Simple self-evolving model that expands when loss is high.

    The agent attempts to predict the imaginary parts of the non-trivial
    zeros of the Riemann zeta function. After each training round it
    evaluates its loss; if the value is above a threshold, the agent
    increases the width of its hidden layer and rebuilds itself.

    The agent keeps a log of its decisions in ``agent_history.json``
    so it can recall its past and justify changes.
    """

    def __init__(
        self, hidden_size: int = 8, log_file: str = "agent_history.json"
    ) -> None:
        self.hidden_size = hidden_size
        self.log_path = Path(log_file)
        self.identity = "EvoAgent"
        self.history = []
        if self.log_path.exists():
            self.history = json.loads(self.log_path.read_text())
        self.log(f"Initialized with hidden_size={hidden_size}")
        self._build_model()

    def _build_model(self) -> None:
        """Create the internal neural network."""
        self.model = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def log(self, message: str) -> None:
        """Append a message to the history log."""
        entry = {"agent": self.identity, "message": message}
        self.history.append(entry)
        self.log_path.write_text(json.dumps(self.history, indent=2))

    def train_on_riemann_zeros(self, epochs: int = 200, lr: float = 0.01) -> float:
        """Train on the first ten imaginary parts of zeta zeros."""
        zeros = find_zeros(10)
        x = torch.arange(1, len(zeros) + 1, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(
            [float(mp.im(z)) for z in zeros], dtype=torch.float32
        ).unsqueeze(1)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            opt.zero_grad()
            out = self.model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        return loss.item()

    def self_improve(self, loss: float, threshold: float = 1e-3) -> bool:
        """Evolve if the loss exceeds ``threshold``."""
        if loss > threshold:
            old_size = self.hidden_size
            self.hidden_size += 4
            self.log(
                f"Loss {loss:.4f} > {threshold}; increasing hidden_size {old_size}->{self.hidden_size}"
            )
            self._build_model()
            return True
        self.log(f"Loss {loss:.4f} <= {threshold}; keeping architecture")
        return False


def main() -> None:
    agent = SelfModifyingAgent()
    improved = True
    while improved:
        loss = agent.train_on_riemann_zeros()
        improved = agent.self_improve(loss)
    print(f"Final loss: {loss:.4f}")
    agent.log(f"Training complete with final loss {loss:.4f}")


if __name__ == "__main__":
    main()
