import json
import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CONFIG_FILE = 'self_evolving_config.json'
LOG_FILE = 'self_evolving_log.txt'


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    # default configuration
    config = {
        'hidden_size': 16,
        'learning_rate': 1e-3,
        'epochs': 5,
        'threshold': 0.3
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    return config


def log(message: str):
    with open(LOG_FILE, 'a') as f:
        f.write(message + '\n')
    print(message)


def build_model(input_dim: int, hidden_size: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_dim),
    )


def train(model, data_loader, epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
    return loss.item()


def evolve_config(config):
    # simple evolution: increase hidden size if loss above threshold
    config['hidden_size'] = int(config['hidden_size'] * 1.5)
    config['learning_rate'] *= 1.1
    log(f"Evolucionando configuracion: hidden_size={config['hidden_size']}, lr={config['learning_rate']}")
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    config = load_config()

    # generate simple dataset: XOR
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([0, 1, 1, 0])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = build_model(2, config['hidden_size'], 2)
    final_loss = train(model, loader, config['epochs'], config['learning_rate'])

    log(f"Perdida final: {final_loss:.4f}")
    if final_loss > config['threshold']:
        log('Rendimiento insuficiente. Iniciando auto-mejora...')
        evolve_config(config)
        log('Re-ejecutando con la nueva configuracion.')
        os.execv(sys.executable, ['python'] + sys.argv)
    else:
        log('Rendimiento aceptable. Evolucion completa.')


if __name__ == '__main__':
    main()
