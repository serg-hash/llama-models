import numpy as np


class SelfImprovingNN:
    """A simple self-improving neural network."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x.dot(self.W1) + self.b1)
        return h.dot(self.W2) + self.b2

    def mutate(self, rate: float = 0.1) -> None:
        """Mutate the network weights."""
        self.W1 += rate * np.random.randn(*self.W1.shape)
        self.b1 += rate * np.random.randn(*self.b1.shape)
        self.W2 += rate * np.random.randn(*self.W2.shape)
        self.b2 += rate * np.random.randn(*self.b2.shape)

    def copy(self) -> "SelfImprovingNN":
        clone = SelfImprovingNN(self.W1.shape[0], self.W1.shape[1], self.W2.shape[1])
        clone.W1 = self.W1.copy()
        clone.b1 = self.b1.copy()
        clone.W2 = self.W2.copy()
        clone.b2 = self.b2.copy()
        return clone


class GeneticAlgorithm:
    def __init__(self, population_size: int, input_dim: int, hidden_dim: int, output_dim: int):
        self.population = [SelfImprovingNN(input_dim, hidden_dim, output_dim) for _ in range(population_size)]

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> list:
        fitness = []
        for agent in self.population:
            preds = agent.forward(x)
            mse = np.mean((preds - y) ** 2)
            fitness.append(mse)
        return fitness

    def select_best(self, fitness: list) -> int:
        return int(np.argmin(fitness))

    def evolve(self, x: np.ndarray, y: np.ndarray, generations: int = 10, mutation_rate: float = 0.1) -> None:
        for gen in range(generations):
            fitness = self.evaluate(x, y)
            best_idx = self.select_best(fitness)
            best_agent = self.population[best_idx].copy()
            best_mse = fitness[best_idx]
            print(f"Generation {gen}, Best MSE: {best_mse:.4f}")

            # Reproduce best agent with mutation
            new_population = [best_agent]
            for _ in range(len(self.population) - 1):
                child = best_agent.copy()
                child.mutate(mutation_rate)
                new_population.append(child)
            self.population = new_population


def main() -> None:
    # Simple function approximation: y = sin(2 * pi * x)
    x = np.linspace(0, 1, 100).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    ga = GeneticAlgorithm(population_size=5, input_dim=1, hidden_dim=8, output_dim=1)
    ga.evolve(x, y, generations=20, mutation_rate=0.05)


if __name__ == "__main__":
    main()
