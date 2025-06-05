import argparse
from typing import List
import mpmath as mp
import numpy as np

from riemann_zero_explorer import find_zeros

mp.dps = 50  # ensure high precision for Riemann zeros


def generate_dark_matter_signal(n: int) -> np.ndarray:
    """Generate a mock dark matter signal as random counts."""
    rng = np.random.default_rng()
    # using Poisson distribution to mimic rare event counts
    return rng.poisson(lam=5.0, size=n)


def correlate_sequences(seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
    """Return the cross-correlation between two sequences."""
    return np.correlate(seq1 - seq1.mean(), seq2 - seq2.mean(), mode="full")


def analyze(n: int):
    zeros = find_zeros(n)
    zero_imag = np.array([mp.im(z) for z in zeros], dtype=float)
    dm_signal = generate_dark_matter_signal(n)

    correlation = correlate_sequences(zero_imag, dm_signal)
    print("Riemann zeros (imag parts):", zero_imag)
    print("Dark matter signal:", dm_signal)
    print("Cross-correlation:", correlation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze mock dark matter data against Riemann zeros"
    )
    parser.add_argument("-n", type=int, default=10, help="Number of points to compute")
    args = parser.parse_args()
    analyze(args.n)
