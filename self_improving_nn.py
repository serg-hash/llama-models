import json
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

class SelfImprovingNN:
    """A toy neural network that rewrites its configuration when performance is low."""

    def __init__(self):
        self.config = {
            "hidden_size": 16,
            "learning_rate": 1e-3,
            "layer_count": 1,
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._build_model()

    def _build_model(self):
        layers = []
        input_size = 2
        for _ in range(self.config["layer_count"]):
            layers.append(nn.Linear(input_size, self.config["hidden_size"]))
            layers.append(nn.ReLU())
            input_size = self.config["hidden_size"]
        layers.append(nn.Linear(input_size, 1))
        self.model = nn.Sequential(*layers).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train(self, data, targets, epochs=100):
        self.model.train()
        for epoch in range(epochs):
            outputs = self.model(data)
            loss = self.loss_fn(outputs.view(-1), targets)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if (epoch + 1) % 25 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        return loss.item()

    def evaluate(self, data, targets):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            preds = (torch.sigmoid(outputs).view(-1) > 0.5).float()
            acc = (preds == targets).float().mean().item()
        return acc

    def self_improve(self, performance):
        if performance < 0.9:
            print("Performance below threshold. Modifying architecture...")
            old_config = copy.deepcopy(self.config)
            self.config["layer_count"] += 1
            self.config["hidden_size"] *= 2
            self.config["learning_rate"] *= 0.5
            print(f"Old config: {old_config}\nNew config: {self.config}")
            self._build_model()
            self.save_state()

    def save_state(self):
        with open("self_improving_state.json", "w") as f:
            json.dump(self.config, f, indent=2)

    def load_state(self):
        if os.path.exists("self_improving_state.json"):
            with open("self_improving_state.json") as f:
                self.config = json.load(f)
            self._build_model()


def create_xor_data(device):
    inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32, device=device)
    targets = torch.tensor([0,1,1,0], dtype=torch.float32, device=device)
    return inputs, targets


def main():
    print("Self-improving neural network demo (does not solve the Riemann Hypothesis).")
    agent = SelfImprovingNN()
    agent.load_state()
    inputs, targets = create_xor_data(agent.device)
    for cycle in range(3):
        print(f"\nTraining cycle {cycle+1}")
        agent.train(inputs, targets, epochs=100)
        acc = agent.evaluate(inputs, targets)
        print(f"Accuracy: {acc:.2f}")
        agent.self_improve(acc)
    print("Final config:", agent.config)


if __name__ == "__main__":
    main()
