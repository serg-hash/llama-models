"""Self-modifying neural network example."""
# Generation: 0

import os
import json
import uuid
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CONFIG_FILE = "evolve_config.json"
LOG_FILE = "evolution.log"
DEFAULT_CONFIG = {"embed_dim": 16, "num_layers": 2, "threshold": 0.7}

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

def generate_data(num_sequences: int, seq_length: int):
    return torch.randint(0, 2, (num_sequences, seq_length))

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

def train(model, encoder, decoder, loader, num_epochs: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoder.to(device)
    decoder.to(device)
    optim = torch.optim.Adam(
        list(model.parameters()) + list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-3,
    )
    loss_fn = nn.CrossEntropyLoss()
    loss_val = None
    for epoch in range(num_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            embedded = encoder(x)
            output = model(embedded, embedded)
            logits = decoder(output)
            loss = loss_fn(logits.view(-1, 2), y.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_val = loss.item()
        print(f"Epoch {epoch + 1}, Loss: {loss_val:.4f}")
    return loss_val

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()
    if "id" not in config:
        config["id"] = str(uuid.uuid4())
    if "generation" not in config:
        config["generation"] = 0
    return config

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

def log_evolution(config, message):
    timestamp = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(
            f"{timestamp} | ID {config['id']} | gen {config['generation']} | {message}\n"
        )

def update_generation_comment(gen):
    path = os.path.abspath(__file__)
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("# Generation:"):
                lines[i] = f"# Generation: {gen}\n"
                break
        with open(path, "w") as f:
            f.writelines(lines)
    except OSError:
        pass

def evolve(config, loss):
    msg = (
        f"Loss {loss:.4f} exceeded threshold {config['threshold']:.4f}. "
        "Adding a layer to improve capacity."
    )
    print(msg)
    config["num_layers"] += 1
    config["generation"] += 1
    log_evolution(config, msg)
    update_generation_comment(config["generation"])

def main():
    config = load_config()
    print(
        f"Self ID: {config['id']} | Generation {config['generation']}\nLoaded config: {config}"
    )
    log_evolution(config, "start run")
    update_generation_comment(config["generation"])
    sequences = generate_data(num_sequences=128, seq_length=32)
    dataset = ZeroDataset(sequences)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    while True:
        print(
            f"Training model with {config['num_layers']} layers and embed_dim {config['embed_dim']}"
        )
        model, enc, dec = create_model(config["embed_dim"], config["num_layers"])
        loss = train(model, enc, dec, loader)
        if loss <= config["threshold"]:
            print(
                f"Loss {loss:.4f} is below threshold {config['threshold']:.4f}. Training complete."
            )
            log_evolution(config, f"converged with loss {loss:.4f}")
            break
        evolve(config, loss)
    save_config(config)
    print(f"Final config saved: {config}")

if __name__ == "__main__":
    main()
