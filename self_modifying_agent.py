import argparse
import re
import time
from pathlib import Path

import torch
from torch import nn
from torch import optim

import mpmath as mp
from riemann_zero_explorer import find_zeros

AGENT_NAME = "SelfModifyingAgent"
VERSION = 1
HIDDEN_SIZE = 16
THRESHOLD = 0.01


def load_data(n_zeros: int = 5):
    zeros = find_zeros(n_zeros)
    x = torch.arange(1, n_zeros + 1, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor([float(mp.im(z)) for z in zeros], dtype=torch.float32).unsqueeze(1)
    return x, y


def build_model(hidden: int):
    return nn.Sequential(
        nn.Linear(1, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1)
    )


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int = 200):
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
    return float(loss.item())


def self_modify(loss: float):
    new_hidden = HIDDEN_SIZE * 2
    reason = (
        f"Loss {loss:.4f} exceeded threshold {THRESHOLD}. "
        f"Increasing hidden size to {new_hidden}."
    )
    src = Path(__file__).read_text()
    src = re.sub(r"HIDDEN_SIZE = \d+", f"HIDDEN_SIZE = {new_hidden}", src)
    src = re.sub(r"VERSION = (\d+)", lambda m: f"VERSION = {int(m.group(1)) + 1}", src)
    Path(__file__).write_text(src)
    with open("evolution_log.txt", "a") as log:
        log.write(f"{time.ctime()}: {reason}\n")
    print(reason)


def main(dry_run: bool = False):
    x, y = load_data()
    model = build_model(HIDDEN_SIZE)
    loss = train(model, x, y)
    print(f"I am {AGENT_NAME} v{VERSION}. Final loss: {loss:.4f}")
    if loss > THRESHOLD and not dry_run:
        self_modify(loss)
    else:
        print("No self-modification triggered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-modifying demo agent")
    parser.add_argument("--dry-run", action="store_true", help="Do not rewrite source even if failing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
