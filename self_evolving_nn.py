import json
import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CONFIG_FILE = Path("self_evolving_config.json")
LOG_FILE = Path("evolution.log")
ACCURACY_THRESHOLD = 0.9
HIDDEN_SIZE = 16


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"version": 0, "hidden_size": HIDDEN_SIZE}


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def log(message: str) -> None:
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


def create_model(hidden_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )


def generate_data(num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, 2, (num_samples, 2)).float()
    y = (x.sum(dim=1) % 2).long()  # parity problem
    return x, y


def train(model: nn.Module, loader: DataLoader) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(50):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
    # evaluate accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total


def rewrite_source(new_hidden: int) -> None:
    """Rewrite this file updating the HIDDEN_SIZE constant."""
    src_path = Path(__file__)
    lines = src_path.read_text().splitlines()
    with src_path.open("w") as f:
        for line in lines:
            if line.startswith("HIDDEN_SIZE ="):
                f.write(f"HIDDEN_SIZE = {new_hidden}\n")
            else:
                f.write(line + "\n")


def evolve(new_hidden: int, accuracy: float, cfg: dict) -> None:
    cfg["version"] += 1
    cfg["hidden_size"] = new_hidden
    cfg["last_accuracy"] = accuracy
    save_config(cfg)
    rewrite_source(new_hidden)
    log(
        f"Version {cfg['version']}: Increased hidden size to {new_hidden}"
        f" because accuracy {accuracy:.3f} < {ACCURACY_THRESHOLD}"
    )


def main() -> None:
    cfg = load_config()
    hidden = cfg.get("hidden_size", HIDDEN_SIZE)
    model = create_model(hidden)
    x, y = generate_data()
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    acc = train(model, loader)
    print(f"Self-evolving NN version {cfg['version']} accuracy: {acc:.3f}")
    if acc < ACCURACY_THRESHOLD:
        evolve(hidden + 4, acc, cfg)
    else:
        log(
            f"Version {cfg['version']} achieved accuracy {acc:.3f},"
            " no evolution needed"
        )


if __name__ == "__main__":
    main()
