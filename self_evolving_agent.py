import argparse
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class EvolutionConfig:
    input_length: int = 15
    hidden_size: int = 32
    learning_rate: float = 1e-2
    epochs: int = 5
    batch_size: int = 32
    generations: int = 5
    target_accuracy: float = 0.9


@dataclass
class EvolutionRecord:
    generation: int
    config: EvolutionConfig
    accuracy: float
    loss: float
    justification: str


@dataclass
class EvolutionState:
    lineage_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    records: List[EvolutionRecord] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "lineage_id": self.lineage_id,
            "records": [
                {
                    "generation": record.generation,
                    "config": asdict(record.config),
                    "accuracy": record.accuracy,
                    "loss": record.loss,
                    "justification": record.justification,
                }
                for record in self.records
            ],
        }


class SelfEvolvingAgent:
    def __init__(self, config: EvolutionConfig, state: EvolutionState | None = None):
        self.config = config
        self.state = state or EvolutionState()

    def _build_dataset(self, num_samples: int) -> TensorDataset:
        inputs = torch.randint(0, 2, (num_samples, self.config.input_length))
        parity = inputs.sum(dim=1) % 2
        targets = parity.long()
        features = inputs.float()
        return TensorDataset(features, targets)

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.config.input_length, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 2),
        )

    def _train_and_evaluate(self) -> Tuple[float, float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = self._build_dataset(num_samples=512)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        model = self._build_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.config.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            all_x, all_y = dataset.tensors
            logits = model(all_x.to(device))
            predictions = logits.argmax(dim=1).cpu()
            accuracy = (predictions == all_y).float().mean().item()
            final_loss = loss_fn(logits, all_y.to(device)).item()

        return accuracy, final_loss

    def _mutate_config(self, accuracy: float, loss: float) -> str:
        justification_parts = []
        if accuracy < 0.8:
            self.config.hidden_size = int(self.config.hidden_size * 1.5)
            justification_parts.append(
                "Precisión baja: aumento la capacidad con más neuronas ocultas."
            )
        if loss > 0.5:
            self.config.learning_rate *= 0.7
            justification_parts.append("Pérdida alta: reduzco el learning rate para estabilidad.")
        if accuracy >= 0.8 and loss <= 0.5:
            self.config.epochs += 1
            justification_parts.append("Buen rendimiento: entreno una época extra para afinar.")

        if not justification_parts:
            justification_parts.append("Sin cambios necesarios en esta generación.")

        return " ".join(justification_parts)

    def run(self) -> EvolutionState:
        for generation in range(1, self.config.generations + 1):
            accuracy, loss = self._train_and_evaluate()
            justification = self._mutate_config(accuracy, loss)
            self.state.records.append(
                EvolutionRecord(
                    generation=generation,
                    config=EvolutionConfig(**asdict(self.config)),
                    accuracy=accuracy,
                    loss=loss,
                    justification=justification,
                )
            )
            print(
                f"Generación {generation}: accuracy={accuracy:.3f}, loss={loss:.3f}. "
                f"{justification}"
            )
            if accuracy >= self.config.target_accuracy:
                print("Objetivo alcanzado, deteniendo evolución.")
                break
        return self.state


def save_state(state: EvolutionState, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state.as_dict(), handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simula una red neuronal autoreescribible que ajusta su configuración "
            "cuando falla. Es una demostración educativa, no una prueba matemática."
        )
    )
    parser.add_argument("--generations", type=int, default=5, help="Número de iteraciones")
    parser.add_argument("--hidden-size", type=int, default=32, help="Tamaño inicial")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Learning rate inicial")
    parser.add_argument("--epochs", type=int, default=5, help="Épocas por generación")
    parser.add_argument("--state-path", type=str, help="Ruta para guardar historial JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvolutionConfig(
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        generations=args.generations,
    )
    agent = SelfEvolvingAgent(config)
    state = agent.run()
    if args.state_path:
        save_state(state, args.state_path)
        print(f"Estado guardado en {args.state_path}")


if __name__ == "__main__":
    main()
