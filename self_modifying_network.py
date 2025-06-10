"""Example of a self-modifying neural network script.

This program trains a tiny Transformer on random binary data. If the
achieved accuracy is below a configurable threshold, it rewrites its own
configuration parameters to increase model capacity and logs a short
justification. This illustrates self-improvement logic in code but does
not attempt to solve the Riemann Hypothesis.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CONFIG = {
    "embed_dim": 16,
    "num_layers": 2,
    "threshold": 0.7,  # target accuracy
}


class BinaryDataset(Dataset):
    """Simple dataset of random binary sequences."""

    def __init__(self, num_sequences: int, seq_length: int):
        data = torch.randint(0, 2, (num_sequences, seq_length))
        self.x = data[:, :-1]
        self.y = data[:, 1:]

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_model(embed_dim: int, num_layers: int) -> tuple[nn.Module, nn.Module, nn.Module]:
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


def train(model, encoder, decoder, loader) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoder.to(device)
    decoder.to(device)
    optim = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    correct = 0
    total = 0
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
        preds = logits.argmax(dim=-1)
        correct += (preds == y).float().sum().item()
        total += y.numel()
    return correct / total


def log_and_modify_config(acc: float):
    reason = f"Accuracy {acc:.2f} below threshold {CONFIG['threshold']}. Increasing capacity."
    # modify config
    CONFIG["embed_dim"] *= 2
    CONFIG["num_layers"] += 1
    # rewrite file
    path = Path(__file__)
    lines = path.read_text().splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("CONFIG ="):
            new_lines.append(
                f"CONFIG = {{\n    \"embed_dim\": {CONFIG['embed_dim']},\n    \"num_layers\": {CONFIG['num_layers']},\n    \"threshold\": {CONFIG['threshold']},  # target accuracy\n}}"
            )
        else:
            new_lines.append(line)
    path.write_text("\n".join(new_lines))
    with open("evolution_log.txt", "a") as f:
        f.write(reason + "\n")


def main():
    dataset = BinaryDataset(num_sequences=64, seq_length=16)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model, enc, dec = create_model(CONFIG["embed_dim"], CONFIG["num_layers"])
    acc = train(model, enc, dec, loader)
    print(f"Accuracy: {acc:.2f}")
    if acc < CONFIG["threshold"]:
        log_and_modify_config(acc)


if __name__ == "__main__":
    main()
