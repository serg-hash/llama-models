# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def generate_data(num_sequences: int, seq_length: int):
    """Generate random binary sequences with zeros and ones."""
    data = torch.randint(0, 2, (num_sequences, seq_length))
    return data


class ZeroDataset(Dataset):
    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences

    def __len__(self):
        return self.sequences.size(0)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y


def create_model(embed_dim: int, num_layers: int = 2):
    """Create a tiny Transformer model."""
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


def train(model, encoder, decoder, loader, num_epochs: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    for epoch in range(num_epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            embedded = encoder(x)
            output = model(embedded, embedded)
            logits = decoder(output)
            loss = loss_fn(logits.view(-1, 2), y.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    sequences = generate_data(num_sequences=100, seq_length=32)
    dataset = ZeroDataset(sequences)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model, enc, dec = create_model(embed_dim=16)
    train(model, enc, dec, loader)
