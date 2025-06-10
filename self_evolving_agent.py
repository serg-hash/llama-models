import os
import json
import datetime
import mpmath as mp
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CONFIG_FILE = 'agent_config.json'
LOG_FILE = 'agent_history.log'


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {
        'embed_dim': 16,
        'num_layers': 1,
        'threshold': 0.1,
        'run_id': 0,
    }


def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)


def log(msg):
    ts = datetime.datetime.now().isoformat()
    with open(LOG_FILE, 'a') as f:
        f.write(f"{ts} {msg}\n")


def generate_zero_data(n):
    xs = []
    ys = []
    for i in range(1, n + 1):
        zero = mp.zetazero(i)
        xs.append([float(i)])
        ys.append([float(mp.im(zero))])
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y


class ZeroPredictor(nn.Module):
    def __init__(self, embed_dim, num_layers):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, embed_dim))
            layers.append(nn.ReLU())
            in_dim = embed_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model, loader):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(5):
        for x, y in loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return loss.item()


def main():
    cfg = load_config()
    agent_id = cfg.get('run_id', 0)
    print(f"SelfImprovingAgent run {agent_id}")

    x, y = generate_zero_data(10)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = ZeroPredictor(cfg['embed_dim'], cfg['num_layers'])
    final_loss = train_model(model, loader)
    print(f"Final loss: {final_loss:.4f}")

    if final_loss > cfg['threshold']:
        cfg['embed_dim'] *= 2
        cfg['num_layers'] += 1
        cfg['run_id'] += 1
        log(
            f"Run {agent_id}: loss {final_loss:.4f} > {cfg['threshold']}, "
            f"increasing embed_dim to {cfg['embed_dim']} and num_layers to {cfg['num_layers']}"
        )
        save_config(cfg)
    else:
        log(f"Run {agent_id}: loss {final_loss:.4f} <= threshold. Keeping configuration.")


if __name__ == '__main__':
    main()
