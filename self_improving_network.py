# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Demonstration of a simple self-modifying training loop.

This script illustrates how a small neural network can adjust its own
configuration when training fails to reach a target loss. The network does not
rewrite its source code; instead it persists a JSON configuration describing its
current parameters. When loss remains high it increases model capacity and logs
an explanation for the change.

It also performs a toy numerical experiment with the Riemann zeta function to
highlight that the agent can run arbitrary tasks between training cycles. This
does not prove the Riemann Hypothesis; it merely checks that a few computed
zeros lie on the critical line.
"""

import json
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from riemann_zero_explorer import find_zeros, check_critical_line

CONFIG_FILE = "self_config.json"
LOG_FILE = "evolve.log"


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"embed_dim": 8, "num_layers": 1, "iteration": 0}


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)


def log(msg: str):
    timestamp = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")


class SequenceDataset(Dataset):
    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences

    def __len__(self):
        return self.sequences.size(0)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y


def generate_data(num_sequences: int, seq_length: int):
    return torch.randint(0, 2, (num_sequences, seq_length))


def create_model(embed_dim: int, num_layers: int = 1):
    model = nn.Transformer(
        d_model=embed_dim,
        nhead=2,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=embed_dim * 4,
        batch_first=True,
    )
    encoder = nn.Embedding(2, embed_dim)
    decoder = nn.Linear(embed_dim, 2)
    return model, encoder, decoder


def train(model, encoder, decoder, loader, num_epochs: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoder.to(device)
    decoder.to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    last_loss = None
    for _ in range(num_epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            emb = encoder(x)
            out = model(emb, emb)
            logits = decoder(out)
            loss = loss_fn(logits.view(-1, 2), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
    return last_loss


class SelfNet:
    def __init__(self):
        self.name = "SelfNet"
        self.config = load_config()
        self.model, self.encoder, self.decoder = create_model(
            embed_dim=self.config["embed_dim"],
            num_layers=self.config["num_layers"],
        )
        self.iteration = self.config.get("iteration", 0)
        log(f"Initialized {self.name} (iteration {self.iteration}) with {self.config}")

    def evolve(self, loss: float):
        log(
            f"Loss {loss:.4f} exceeded threshold. Increasing capacity to improve performance."
        )
        self.config["embed_dim"] *= 2
        self.config["num_layers"] += 1
        self.iteration += 1
        self.config["iteration"] = self.iteration
        save_config(self.config)
        log(
            f"Iteration {self.iteration} -- new configuration: {self.config}"
        )

    def run(self):
        log(f"{self.name} starting with config {self.config}")
        seq = generate_data(num_sequences=64, seq_length=16)
        dataset = SequenceDataset(seq)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        loss = train(self.model, self.encoder, self.decoder, loader)
        log(f"Training completed with loss {loss:.4f}")

        zeros = find_zeros(5)
        on_line = all(check_critical_line(z) for z in zeros)
        log(
            "Checked first 5 Riemann zeros; all on critical line: "
            f"{on_line}"
        )
        if loss is None or loss > 0.6:
            self.evolve(loss if loss is not None else float("inf"))
        else:
            log("Performance adequate; no evolution required.")


if __name__ == "__main__":
    agent = SelfNet()
    agent.run()
