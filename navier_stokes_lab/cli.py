# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""CLI del laboratorio: simula Taylor–Green y reporta el error espectral."""

from __future__ import annotations

import argparse
import sys

from navier_stokes_lab.solver import simulate_taylor_green


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Laboratorio Navier–Stokes 2D (Taylor–Green espectral)"
    )
    parser.add_argument("--n-grid", type=int, default=32, help="Resolución (par)")
    parser.add_argument("--nu", type=float, default=0.05, help="Viscosidad ν")
    parser.add_argument("--dt", type=float, default=0.005, help="Paso temporal RK4")
    parser.add_argument("--steps", type=int, default=40, help="Número de pasos")
    parser.add_argument(
        "--no-dealias",
        action="store_true",
        help="Desactiva la regla 2/3 de dealiasing",
    )
    return parser


def format_report(result) -> str:
    lines = [
        "Laboratorio Navier–Stokes 2D  |  Taylor–Green periódico",
        f"n={result.n_grid}  ν={result.nu}  dt={result.dt}  pasos={len(result.times) - 1}",
        "",
        f"{'t':>8}  {'E num':>12}  {'E exact':>12}  {'|E| rel':>10}  {'L2(ω)':>10}",
        "-" * 60,
    ]
    for t, e_num, e_ex, err in zip(
        result.times, result.energy, result.energy_exact, result.vorticity_l2_error
    ):
        rel = abs(e_num - e_ex) / e_ex if e_ex else 0.0
        lines.append(f"{t:8.4f}  {e_num:12.8f}  {e_ex:12.8f}  {rel:10.2e}  {err:10.2e}")
    lines.append("-" * 60)
    lines.append(
        f"Error relativo final de energía: {result.final_energy_rel_error:.3e}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = simulate_taylor_green(
        n_grid=args.n_grid,
        nu=args.nu,
        dt=args.dt,
        n_steps=args.steps,
        dealias=not args.no_dealias,
    )
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
