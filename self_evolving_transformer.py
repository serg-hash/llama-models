import inspect
import json
import os
import random
import re

import torch

from zero_transformer import generate_data, ZeroDataset, create_model


class SelfEvolvingTransformer:
    """Simple agent that mutates its embedding size when training stalls."""

    def __init__(self, config_path: str = "self_agent_config.json"):
        self.config_path = config_path
        self.load_config()
        self.model, self.encoder, self.decoder = create_model(
            embed_dim=self.config["embed_dim"]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.optim = torch.optim.Adam(
            list(self.model.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder.parameters()),
            lr=self.config["lr"],
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.performance_log = []

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {"embed_dim": 16, "lr": 1e-3}

    def save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self.config, f)

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        running_loss = 0.0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optim.zero_grad()
            embedded = self.encoder(x)
            output = self.model(embedded, embedded)
            logits = self.decoder(output)
            loss = self.loss_fn(logits.view(-1, 2), y.view(-1))
            loss.backward()
            self.optim.step()
            running_loss += loss.item()
        avg = running_loss / len(loader)
        print(f"Epoch loss: {avg:.4f}")
        return avg

    def evaluate_and_mutate(self, loss: float):
        self.performance_log.append(loss)
        if len(self.performance_log) > 1 and loss >= self.performance_log[-2]:
            print("Mutation triggered. Rewriting source...")
            self.mutate_self()
            self.save_config()

    def mutate_self(self):
        source = inspect.getsource(SelfEvolvingTransformer)

        def change_embed(match: re.Match) -> str:
            current = int(match.group(1))
            new_val = max(4, current + random.choice([-2, 2]))
            self.config["embed_dim"] = new_val
            return str(new_val)

        pattern = re.compile(r'"embed_dim":\s*(\d+)')
        mutated = pattern.sub(lambda m: f'"embed_dim": {change_embed(m)}', source, count=1)
        path = os.path.realpath(__file__)
        with open(path, "w") as f:
            f.write(mutated)

    def run(self, epochs: int = 3):
        data = generate_data(num_sequences=100, seq_length=32)
        dataset = ZeroDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        for _ in range(epochs):
            loss = self.train_epoch(loader)
            self.evaluate_and_mutate(loss)


# Basic internal test

def test_self_awareness():
    agent = SelfEvolvingTransformer()
    assert hasattr(agent, "mutate_self")
    assert callable(agent.train_epoch)


if __name__ == "__main__":
    test_self_awareness()
    agent = SelfEvolvingTransformer()
    agent.run(epochs=5)
