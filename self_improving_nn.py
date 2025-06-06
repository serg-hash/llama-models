import os
import re
import torch
from torch import nn

# Hyperparameters used for training
HYPERPARAMS = {
    "hidden_dim": 4,
    "learning_rate": 0.1,
    "num_epochs": 2000,
}


def generate_xor_data(n=4):
    x = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float)
    y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float)
    return x, y


class XORNet(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train(model, x, y, epochs, lr):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()


def evaluate(model, x, y):
    with torch.no_grad():
        preds = (model(x) > 0.5).float()
        correct = (preds == y).float().mean().item()
    return correct


def update_hyperparams(path, new_params):
    pattern = re.compile(r"HYPERPARAMS\s*=\s*\{[^\}]*\}")
    with open(path, "r") as f:
        content = f.read()
    new_line = f"HYPERPARAMS = {new_params}"
    content = pattern.sub(new_line, content)
    with open(path, "w") as f:
        f.write(content)


def main():
    x, y = generate_xor_data()
    model = XORNet(HYPERPARAMS["hidden_dim"])
    train(model, x, y, HYPERPARAMS["num_epochs"], HYPERPARAMS["learning_rate"])
    acc = evaluate(model, x, y)
    print(f"Accuracy: {acc:.2f}")
    if acc < 1.0:
        new_params = HYPERPARAMS.copy()
        new_params["hidden_dim"] *= 2
        new_params["num_epochs"] += 1000
        update_hyperparams(__file__, new_params)
        print("Hyperparameters updated for next run:", new_params)


if __name__ == "__main__":
    main()
