# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Solución exacta del vórtice de Taylor–Green periódico en 2D.

En el toro [0, 2π]², con viscosidad ν ≥ 0:

    u =  sin(x) cos(y) exp(-2 ν t)
    v = -cos(x) sin(y) exp(-2 ν t)
    ω =  2 sin(x) sin(y) exp(-2 ν t)

El término no lineal se anula y la energía cinética media decae como
(1/4) exp(-4 ν t).
"""

from __future__ import annotations

import numpy as np


def periodic_grid(n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodos equiespaciados en [0, 2π) × [0, 2π)."""
    if n_grid < 4:
        raise ValueError("n_grid must be at least 4")
    x1d = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    return np.meshgrid(x1d, x1d, indexing="ij")


def decay(nu: float, t: float) -> float:
    return float(np.exp(-2.0 * nu * t))


def taylor_green_fields(
    n_grid: int, nu: float = 0.05, t: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (u, v) del vórtice de Taylor–Green."""
    x, y = periodic_grid(n_grid)
    amp = decay(nu, t)
    u = np.sin(x) * np.cos(y) * amp
    v = -np.cos(x) * np.sin(y) * amp
    return u, v


def taylor_green_vorticity(n_grid: int, nu: float = 0.05, t: float = 0.0) -> np.ndarray:
    """Vorticidad ω = ∂x v − ∂y u."""
    x, y = periodic_grid(n_grid)
    return 2.0 * np.sin(x) * np.sin(y) * decay(nu, t)


def kinetic_energy_density(u: np.ndarray, v: np.ndarray) -> float:
    """Energía cinética media (1/2) ⟨|u|²⟩."""
    return 0.5 * float(np.mean(u * u + v * v))


def kinetic_energy_exact(nu: float, t: float) -> float:
    """Energía cinética exacta del Taylor–Green: (1/4) exp(-4 ν t)."""
    return 0.25 * float(np.exp(-4.0 * nu * t))


def energy_time_derivative_exact(nu: float, t: float) -> float:
    """d/dt of the Taylor–Green kinetic energy: −ν exp(−4 ν t)."""
    return -float(nu) * float(np.exp(-4.0 * nu * t))


def mean_square_vorticity(omega: np.ndarray) -> float:
    """Mean of ω². For 2D periodic incompressible flow this equals ⟨|∇u|²⟩."""
    return float(np.mean(omega * omega))


def enstrophy_density(omega: np.ndarray) -> float:
    """Enstrofía media (1/2) ⟨ω²⟩."""
    return 0.5 * float(np.mean(omega * omega))
