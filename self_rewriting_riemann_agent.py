# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Toy self-rewriting agent for approximating Riemann zeta zeros.

This example demonstrates a minimal self-modifying script. It trains a small
neural network on the imaginary parts of a few zeta zeros. When training fails
to improve, the script rewrites itself with tweaked hyperparameters and updates
a configuration file to persist state across runs.

It is *not* a real solution to the Riemann Hypothesis. The code simply evolves
its parameters in a closed loop to showcase the concept of self-improvement.
"""

# Version: 1

import json
from pathlib import Path
from typing import List

import riemann_zero_explorer as rze

import torch
from torch import nn

CONFIG_FILE = Path("riemann_agent_config.json")
LOG_FILE = Path("self_evolution.log")
DEFAULT_CONFIG = {
    "name": "RiemannSelfAgent",
    "hidden_dim": 8,
    "learning_rate": 0.01,
    "version": 1,
    "best_error": 1e9,
    "num_zeros": 12,
}


def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


class ZeroPredictor(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def generate_data(n: int) -> List[float]:
    zeros = [z.imag for z in rze.find_zeros(n)]
    return zeros


def train(cfg) -> float:
    zeros = generate_data(cfg["num_zeros"])
    x = torch.arange(1, len(zeros) + 1, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(zeros, dtype=torch.float32).unsqueeze(1)
    net = ZeroPredictor(cfg["hidden_dim"])
    optim = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.MSELoss()
    for _ in range(1000):
        optim.zero_grad()
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()
    return loss.item()


def rewrite_self(cfg) -> None:
    this_file = Path(__file__)
    lines = this_file.read_text().splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("# Version:"):
            new_lines.append(f"# Version: {cfg['version']}")
        elif line.startswith("DEFAULT_CONFIG"):
            new_lines.append(f"DEFAULT_CONFIG = {json.dumps(cfg, indent=4)}")
        else:
            new_lines.append(line)
    this_file.write_text("\n".join(new_lines))


def main() -> None:
    cfg = load_config()
    log(f"Hello! I am {cfg['name']}.")
    log(f"[{cfg['name']}] Running version {cfg['version']}")
    error = train(cfg)
    log(f"Training error: {error:.4f}")
    if error >= cfg.get("best_error", 1e9):
        cfg["hidden_dim"] += 4
        cfg["learning_rate"] *= 0.9
        cfg["version"] += 1
        log(
            "Improvement insufficient:"
            f" error {error:.4f} >= best {cfg.get('best_error', 1e9):.4f}."
            " Updating hyperparameters."
            f" New hidden_dim {cfg['hidden_dim']}, lr {cfg['learning_rate']:.4f}"
        )
        save_config(cfg)
        rewrite_self(cfg)
    else:
        cfg["best_error"] = error
        save_config(cfg)
        log("Training improved. Retaining current configuration")


if __name__ == "__main__":
    main()
