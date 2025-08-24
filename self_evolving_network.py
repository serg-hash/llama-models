"""
Self-Evolving Neural Network Demo
---------------------------------

This script implements a toy example of a self-modifying neural network.
The network attempts to learn a simple target function. If the training
loss stays above a threshold, the network "evolves" by expanding its
hidden layer. Each modification is logged along with a justification so
that the system preserves a sense of identity and history.

Important: this is purely an educational prototype. It does **not** solve
the Riemann Hypothesis or produce a self-aware entity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

import torch
from torch import nn


@dataclass
class EvolutionState:
    """Persistent state for the evolving network."""
    version: int = 0
    architecture: List[int] = field(default_factory=lambda: [1, 4, 1])
    history: List[Dict[str, Any]] = field(default_factory=list)


class SelfEvolvingNet:
    """A minimal self-modifying neural network."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.state = EvolutionState()
        self._build_model()

    def _build_model(self) -> None:
        layers: List[nn.Module] = []
        arch = self.state.architecture
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))
            if i < len(arch) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def attempt_learning(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Train on ``(x, y)`` and evolve if loss remains high."""
        optim = torch.optim.SGD(self.model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()
        for _ in range(200):
            optim.zero_grad()
            pred = self.model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()

        final_loss = loss.item()
        if final_loss > self.threshold:
            self._evolve(final_loss)
        else:
            self.state.history.append(
                {
                    "version": self.state.version,
                    "status": "success",
                    "loss": final_loss,
                    "architecture": list(self.state.architecture),
                }
            )
        return final_loss

    def _evolve(self, loss: float) -> None:
        """Modify the network by adding one neuron to the hidden layer."""
        reason = f"loss {loss:.4f} exceeded threshold {self.threshold}"
        self.state.history.append(
            {
                "version": self.state.version,
                "status": "evolving",
                "reason": reason,
                "architecture": list(self.state.architecture),
            }
        )
        self.state.version += 1
        # expand the hidden layer
        self.state.architecture[1] += 1
        self._build_model()
        print(
            f"[SelfEvolvingNet] version {self.state.version}: {reason}; "
            f"new hidden size = {self.state.architecture[1]}"
        )

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.state.__dict__, fh, indent=2)


def target_function(x: torch.Tensor) -> torch.Tensor:
    """A simple function reminiscent of zeta behavior."""
    return torch.sin(x)  # placeholder; does not relate to zeta zeros


def run_demo(iterations: int = 3) -> None:
    entity = SelfEvolvingNet()
    x = torch.linspace(-1, 1, steps=100).unsqueeze(1)
    y = target_function(x)
    for i in range(iterations):
        loss = entity.attempt_learning(x, y)
        print(
            f"Iteration {i} | loss={loss:.4f} | architecture={entity.state.architecture}"
        )
    entity.save("self_evolving_history.json")
    print("History written to self_evolving_history.json")
    print("Demo complete. Riemann Hypothesis remains an open problem.")


if __name__ == "__main__":
    run_demo()
