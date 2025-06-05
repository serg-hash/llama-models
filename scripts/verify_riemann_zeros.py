# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import argparse

import matplotlib.pyplot as plt
import mpmath as mp


def compute_zeros(n):
    return [mp.zetazero(i) for i in range(1, n + 1)]


def verify_real_parts(zeros, tol=1e-9):
    return [abs(z.real - 0.5) < tol for z in zeros]


def plot_zeros(zeros, output):
    xs = [z.real for z in zeros]
    ys = [z.imag for z in zeros]
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, color="blue")
    plt.axvline(0.5, color="red", linestyle="--", label="Re=0.5")
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    plt.title(
        "First {} non-trivial zeros of the Riemann zeta function".format(len(zeros))
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)


def main():
    parser = argparse.ArgumentParser(description="Verify Riemann zeta zeros and plot")
    parser.add_argument("N", type=int, help="Number of zeros to verify")
    parser.add_argument(
        "--output", default="riemann_zeros.png", help="Output plot filename"
    )
    args = parser.parse_args()

    zeros = compute_zeros(args.N)
    results = verify_real_parts(zeros)

    for idx, (zero, ok) in enumerate(zip(zeros, results), 1):
        status = "OK" if ok else "FAIL"
        print(f"Zero {idx}: {zero} -> {status}")

    plot_zeros(zeros, args.output)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
