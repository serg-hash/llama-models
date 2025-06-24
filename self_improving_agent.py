# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Simple self-improving agent that learns Riemann zero patterns."""

import json
import os
from typing import Dict

import torch

from riemann_zero_explorer import find_zeros
from torch import nn
from torch.utils.data import DataLoader, Dataset

CONFIG_FILE = "self_improving_agent_config.json"


def load_config() -> Dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "hidden_dim": 32,
        "num_layers": 1,
        "threshold": 0.1,
        "version": 1,
        "history": [],
    }


def save_config(config: Dict) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


class ZeroDataset(Dataset):
    def __init__(self, n: int = 20) -> None:
        zeros = find_zeros(n)
        self.x = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(1)
        imag_parts = [complex(z).imag for z in zeros]
        self.y = torch.tensor(imag_parts, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def create_model(hidden_dim: int, num_layers: int) -> nn.Module:
    layers = []
    input_dim = 1
    for _ in range(num_layers):
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)


def train(model: nn.Module, loader: DataLoader, epochs: int = 100) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loss_value = 0.0
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_value = loss.item()
    return loss_value


def main() -> None:
    config = load_config()
    print(f"SelfImprovingAgent v{config['version']}")
    dataset = ZeroDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = create_model(config["hidden_dim"], config["num_layers"])
    final_loss = train(model, loader)
    print(f"Final MSE loss: {final_loss:.6f}")
    if final_loss > config["threshold"]:
        config["hidden_dim"] += 16
        config["num_layers"] = min(config["num_layers"] + 1, 3)
        config["version"] += 1
        config["history"].append({"reason": "loss_above_threshold", "loss": final_loss})
        print(
            "Updating configuration due to high loss: "
            f"{final_loss:.6f} > {config['threshold']}"
        )
        save_config(config)
    else:
        print("Model performance within threshold; no change")


if __name__ == "__main__":
    main()
