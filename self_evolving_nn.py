# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Self-evolving neural network with self-modifying capabilities."""

import json
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Hyperparameters stored in a dictionary that can be rewritten
HYPERPARAMS = {
    "embed_dim": 16,
    "num_layers": 2,
    "epochs": 3,
    "threshold": 0.7,
    "version": 1,
}

LOG_FILE = "evolution_log.json"

CONFIG_PATH = __file__


def rewrite_hyperparams(new_params):
    """Rewrite the line containing HYPERPARAMS in this script."""
    with open(CONFIG_PATH, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("HYPERPARAMS = "):
            lines[i] = f"HYPERPARAMS = {json.dumps(new_params, indent=4)}\n"
            break
    with open(CONFIG_PATH, "w") as f:
        f.writelines(lines)


def load_history():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        return json.load(f)


def log_evolution(reason: str, old_params: dict, new_params: dict) -> None:
    history = load_history()
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "reason": reason,
        "old_params": old_params,
        "new_params": new_params,
    }
    history.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=4)


def introduce_agent(params: dict) -> None:
    version = params.get("version", 1)
    print(f"SelfEvolvingNN version {version} initializing.")


class ZeroDataset(Dataset):
    def __init__(self, num_sequences: int, seq_length: int):
        self.sequences = torch.randint(0, 2, (num_sequences, seq_length))

    def __len__(self):
        return self.sequences.size(0)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y


def create_model(embed_dim: int, num_layers: int):
    model = nn.Transformer(
        d_model=embed_dim,
        nhead=2,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=embed_dim * 4,
        dropout=0.1,
        batch_first=True,
    )
    encoder = nn.Embedding(2, embed_dim)
    decoder = nn.Linear(embed_dim, 2)
    return model, encoder, decoder


def train_and_evolve(params):
    introduce_agent(params)

    dataset = ZeroDataset(num_sequences=100, seq_length=32)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoder, decoder = create_model(
        embed_dim=params["embed_dim"], num_layers=params["num_layers"]
    )
    model.to(device)
    encoder.to(device)
    decoder.to(device)
    optim = torch.optim.Adam(
        list(model.parameters())
        + list(encoder.parameters())
        + list(decoder.parameters()),
        lr=1e-3,
    )
    loss_fn = nn.CrossEntropyLoss()

    prev_loss = None
    for epoch in range(params["epochs"]):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            embedded = encoder(x)
            output = model(embedded, embedded)
            logits = decoder(output)
            loss = loss_fn(logits.view(-1, 2), y.view(-1))
            loss.backward()
            optim.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        if prev_loss is not None and loss.item() > prev_loss * params["threshold"]:
            reason = (
                f"Loss {loss.item():.4f} exceeded threshold. "
                "Increasing model capacity."
            )
            print(reason)
            old_params = params.copy()
            params["embed_dim"] *= 2
            params["num_layers"] += 1
            params["version"] = params.get("version", 1) + 1
            log_evolution(reason, old_params, params)
            rewrite_hyperparams(params)
            print("Hyperparameters updated and script rewritten. Restart to apply.")
            return
        prev_loss = loss.item()
    print(
        f"Training complete with current parameters. Version {params.get('version', 1)}"
    )


if __name__ == "__main__":
    train_and_evolve(HYPERPARAMS)
