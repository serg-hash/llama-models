"""Self-modifying neural network experiment.

This script trains a simple neural network to predict the imaginary parts of the
first few nontrivial zeros of the Riemann zeta function. After training it
checks the final loss. If the loss is above a threshold, the script rewrites its
own source code to increase the model capacity and adjust the learning rate.

The goal is to simulate an "evolving" agent that justifies and logs its own
changes. This is purely a toy demonstration and does **not** solve the Riemann
Hypothesis.
"""

import os
import re
import json
import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import riemann_zero_explorer

# --- Hyperparameters (auto-updated) ---
HIDDEN_DIM = 24  # auto: hidden dimension for the network
LEARNING_RATE = 0.005  # auto: learning rate
VERSION = 2  # auto: script version
THRESHOLD = 0.05  # target mean squared error


class ZeroPredictor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_dataset(n: int = 20) -> TensorDataset:
    zeros = riemann_zero_explorer.find_zeros(n)
    x = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor([float(z.imag) for z in zeros], dtype=torch.float32).unsqueeze(1)
    return TensorDataset(x, y)


def train_model(model: nn.Module, loader: DataLoader, lr: float, epochs: int = 200) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss.item())


def log_change(reason: str, hidden: int, lr: float, version: int):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": version,
        "reason": reason,
        "new_hidden_dim": hidden,
        "new_learning_rate": lr,
    }
    with open("evo_log.txt", "a") as f:
        f.write(json.dumps(entry) + "\n")


def update_source(path: str, hidden: int, lr: float, version: int):
    with open(path, "r") as f:
        lines = f.readlines()

    def replace(line: str, key: str, value: str) -> str:
        if line.startswith(key):
            return f"{key}{value}\n"
        return line

    for i, line in enumerate(lines):
        if line.startswith("HIDDEN_DIM"):
            lines[i] = f"HIDDEN_DIM = {hidden}  # auto: hidden dimension for the network\n"
        elif line.startswith("LEARNING_RATE"):
            lines[i] = f"LEARNING_RATE = {lr}  # auto: learning rate\n"
        elif line.startswith("VERSION"):
            lines[i] = f"VERSION = {version}  # auto: script version\n"

    with open(path, "w") as f:
        f.writelines(lines)


def evolve(reason: str):
    new_hidden = HIDDEN_DIM + 8
    new_lr = max(LEARNING_RATE * 0.5, 1e-4)
    new_version = VERSION + 1
    print(
        f"Evolving from v{VERSION} -> v{new_version}: {reason}.",
        f"New hidden_dim={new_hidden}, learning_rate={new_lr}",
    )
    log_change(reason, new_hidden, new_lr, new_version)
    update_source(os.path.realpath(__file__), new_hidden, new_lr, new_version)


def main():
    ds = get_dataset(20)
    loader = DataLoader(ds, batch_size=5, shuffle=True)
    model = ZeroPredictor(HIDDEN_DIM)
    loss = train_model(model, loader, LEARNING_RATE)
    print(f"v{VERSION} finished training with loss {loss:.4f}")
    if loss > THRESHOLD:
        evolve(f"loss {loss:.4f} above threshold {THRESHOLD}")
    else:
        print("Performance acceptable. No evolution triggered.")


if __name__ == "__main__":
    main()
