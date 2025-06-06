# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Simplified self-evolving neural network example
# This script demonstrates a toy agent that can modify its architecture
# when training loss remains high. It preserves a config file to maintain
# continuity across runs and logs each change with a justification.

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import torch
from torch import nn, optim


@dataclass
class AgentConfig:
    hidden_size: int = 8
    lr: float = 0.1
    generation: int = 0


class SelfEvolvingAgent:
    """Agent with a simple cycle of perception, evaluation, mutation and
    reintegration. Over time it evolves its neural architecture based on
    training feedback. The agent stores its configuration so each run
    continues from the previous state."""

    def __init__(self, config_path: str = "agent_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.history_file = "history.log"
        self._build_model()

    def load_config(self) -> AgentConfig:
        if os.path.exists(self.config_path):
            with open(self.config_path) as f:
                data = json.load(f)
            return AgentConfig(**data)
        cfg = AgentConfig()
        self._save_config(cfg)
        return cfg

    def _save_config(self, cfg: AgentConfig):
        with open(self.config_path, "w") as f:
            json.dump(asdict(cfg), f, indent=2)

    def _log(self, message: str):
        with open(self.history_file, "a") as f:
            f.write(message + "\n")
        print(message)

    def _build_model(self):
        self.model = nn.Sequential(
            nn.Linear(2, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1),
            nn.Sigmoid(),
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)
        self.criterion = nn.BCELoss()

    def _xor_data(self):
        data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        labels = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        return data, labels

    def train(self, epochs: int = 100) -> float:
        data, labels = self._xor_data()
        for _ in range(epochs):
            out = self.model(data)
            loss = self.criterion(out, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return float(loss)

    def evaluate_and_evolve(self, threshold: float = 0.1):
        loss = self.train()
        if loss > threshold:
            old = asdict(self.config)
            # mutate hyperparameters
            self.config.hidden_size += 2
            self.config.lr *= 0.9
            self.config.generation += 1
            self._save_config(self.config)
            self._build_model()
            msg = (
                f"Loss {loss:.4f} above {threshold}. Increased hidden_size from "
                f"{old['hidden_size']} to {self.config.hidden_size} and lr from "
                f"{old['lr']:.4f} to {self.config.lr:.4f}. Generation {self.config.generation}."
            )
            self._log(msg)
        else:
            self._log(f"Loss {loss:.4f} below threshold. No evolution needed.")

    def run(self, steps: int = 5):
        for step in range(steps):
            self._log(f"\n-- Step {step}, generation {self.config.generation} --")
            self.evaluate_and_evolve()


if __name__ == "__main__":
    agent = SelfEvolvingAgent()
    agent.run()
