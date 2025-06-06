# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Version: 1

import json
from pathlib import Path

import torch
from torch import nn

CONFIG_FILE = Path("self_net_config.json")
DEFAULT_CONFIG = {
    "input_dim": 2,
    "hidden_dim": 4,
    "learning_rate": 0.1,
    "version": 1,
    "accuracy": 0.0,
}


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


class Net(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()


def generate_xor_data():
    x = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    return x, y


def train_model(config):
    model = Net(config["input_dim"], config["hidden_dim"])
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.BCELoss()
    x, y = generate_xor_data()
    for _ in range(1000):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    predictions = (model(x) > 0.5).float()
    accuracy = (predictions == y).float().mean().item()
    return accuracy


def rewrite_self(config):
    this_file = Path(__file__)
    lines = this_file.read_text().splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("# Version:"):
            new_lines.append(f"# Version: {config['version']}")
        elif line.startswith("DEFAULT_CONFIG"):
            new_cfg = json.dumps(config, indent=4)
            new_lines.append(f"DEFAULT_CONFIG = {new_cfg}")
        else:
            new_lines.append(line)
    this_file.write_text("\n".join(new_lines))


def main():
    config = load_config()
    print(
        f"Running version {config['version']} with hidden_dim="
        f"{config['hidden_dim']} and lr={config['learning_rate']}"
    )
    accuracy = train_model(config)
    print(f"Accuracy: {accuracy:.2f}")
    if accuracy <= config.get("accuracy", 0.0):
        config["hidden_dim"] += 2
        config["learning_rate"] *= 0.9
        config["version"] += 1
        print("Updating model due to insufficient improvement")
        print(
            f" - New hidden_dim: {config['hidden_dim']}, lr: {config['learning_rate']:.4f}"
        )
        save_config(config)
        rewrite_self(config)
    else:
        config["accuracy"] = accuracy
        save_config(config)
        print("Configuration updated with improved accuracy")


if __name__ == "__main__":
    main()
