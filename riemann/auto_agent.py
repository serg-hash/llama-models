import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

STATE_FILE = Path(__file__).with_suffix('.state.json')


class SelfModifyingAgent:
    """A toy self-modifying neural network.

    The agent trains on the XOR problem. If the loss is too high after
    training, it increases the width of the hidden layer and retries. The
    current architecture and version are stored on disk so the agent can
    keep a sense of "identity" across runs.
    """

    def __init__(self):
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            self.version = data["version"] + 1
            self.hidden_size = data["hidden_size"]
        else:
            self.version = 0
            self.hidden_size = 2
        self._build_model()
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        self.model = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def train_once(self, epochs: int = 100) -> float:
        """Train the network once and return the final loss."""
        x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        opt = optim.SGD(self.model.parameters(), lr=0.1)
        for _ in range(epochs):
            opt.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            opt.step()
        return float(loss.item())

    def justify_and_store(self, loss: float) -> None:
        """Persist the current state and print a justification."""
        data = {"version": self.version, "hidden_size": self.hidden_size}
        STATE_FILE.write_text(json.dumps(data))
        if loss > 0.1:
            reason = (
                f"loss {loss:.3f} too high, increasing hidden size to {self.hidden_size}"
            )
        else:
            reason = f"loss {loss:.3f} acceptable, no change"
        print(f"[Agent v{self.version}] {reason}")

    def evolve(self, max_generations: int = 10) -> float:
        """Repeatedly train and modify until the loss is acceptable."""
        for _ in range(max_generations):
            loss = self.train_once()
            if loss <= 0.1:
                self.justify_and_store(loss)
                return loss
            self.justify_and_store(loss)
            self.hidden_size += 1
            self.version += 1
            self._build_model()
        return loss


if __name__ == "__main__":
    agent = SelfModifyingAgent()
    final_loss = agent.evolve()
    print(f"final loss: {final_loss:.3f}")
