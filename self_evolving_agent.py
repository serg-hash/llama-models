import json
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import random
from pathlib import Path


class SelfEvolvingAgent:
    """A simple self-modifying neural network agent."""

    def __init__(self, name: str, config_path: Path):
        self.name = name
        self.config_path = Path(config_path)
        self.load_config()

    def load_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {
                "hidden_size": 16,
                "learning_rate": 1e-3,
                "num_layers": 1,
                "version": 1,
            }
            self.save_config()

    def save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def build_model(self):
        layers = []
        input_dim = 2
        hidden = self.config["hidden_size"]
        for _ in range(self.config["num_layers"]):
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            input_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.model = nn.Sequential(*layers)

    def generate_data(self, n=100):
        X = torch.randint(0, 2, (n, 2)).float()
        y = (X.sum(dim=1, keepdim=True) % 2)  # XOR
        return TensorDataset(X, y)

    def train_once(self):
        self.build_model()
        dataset = self.generate_data()
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        loss_fn = nn.BCEWithLogitsLoss()
        for epoch in range(10):
            for x, y in loader:
                pred = self.model(x)
                loss = loss_fn(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self.evaluate()

    def evaluate(self):
        dataset = self.generate_data(n=30)
        loader = DataLoader(dataset, batch_size=1)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                pred = torch.sigmoid(self.model(x)) > 0.5
                correct += (pred.int() == y.int()).sum().item()
                total += y.size(0)
        return correct / total

    def evolve(self, accuracy):
        if accuracy < 0.9:
            self.config["hidden_size"] *= 2
            self.config["num_layers"] += 1
            self.config["version"] += 1
            self.save_config()
            with open("evolution.log", "a") as f:
                f.write(
                    f"{self.name} evolved to version {self.config['version']} "
                    f"after accuracy {accuracy:.2f}\n"
                )
            print(
                f"{self.name}: accuracy {accuracy:.2f}, evolving to version {self.config['version']}"
            )
        else:
            print(f"{self.name}: accuracy {accuracy:.2f}, no evolution needed")


class MultiAgentSimulator:
    def __init__(self, num_agents=3, config_path="agent_config.json"):
        self.num_agents = num_agents
        self.config_path = Path(config_path)

    def run(self):
        best_acc = 0
        for idx in range(self.num_agents):
            agent = SelfEvolvingAgent(name=f"Agent{idx}", config_path=self.config_path)
            acc = agent.train_once()
            agent.evolve(acc)
            if acc > best_acc:
                best_acc = acc
        print(f"Best accuracy across agents: {best_acc:.2f}")


if __name__ == "__main__":
    simulator = MultiAgentSimulator()
    simulator.run()
