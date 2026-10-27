# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Solver espectral 2D de Navier–Stokes incompresible (formulación en vorticidad).

    ∂t ω + u · ∇ω = ν Δω
    Δψ = −ω,   u = (∂y ψ, −∂x ψ)

En el toro [0, 2π]², con collocation Fourier, dealiasing 2/3 y RK4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from navier_stokes_lab.projector import wavenumber_mesh
from navier_stokes_lab.taylor_green import (
    kinetic_energy_density,
    kinetic_energy_exact,
    taylor_green_vorticity,
)


@dataclass(frozen=True)
class SimulateResult:
    times: np.ndarray
    energy: np.ndarray
    energy_exact: np.ndarray
    vorticity_l2_error: np.ndarray
    omega: np.ndarray
    nu: float
    dt: float
    n_grid: int

    @property
    def final_energy_rel_error(self) -> float:
        exact = self.energy_exact[-1]
        if exact == 0.0:
            return float("inf")
        return abs(self.energy[-1] - exact) / exact


class SpectralNS2D:
    """Navier–Stokes periódico 2D en variables (ω, ψ)."""

    def __init__(self, n_grid: int = 32, nu: float = 0.05, dealias: bool = True):
        if n_grid % 2 != 0:
            raise ValueError("n_grid must be even")
        self.n_grid = n_grid
        self.nu = float(nu)
        self.dealias = dealias
        kx2, ky2, ksq = wavenumber_mesh(n_grid)
        self.kx = kx2
        self.ky = ky2
        self.ksq = ksq
        ksq_safe = ksq.copy()
        ksq_safe[0, 0] = 1.0
        self._inv_ksq = np.where(ksq == 0.0, 0.0, 1.0 / ksq_safe)
        cutoff = n_grid / 3.0
        self._dealias_mask = (np.abs(kx2) < cutoff) & (np.abs(ky2) < cutoff)

    def stream_function_hat(self, omega_hat: np.ndarray) -> np.ndarray:
        return omega_hat * self._inv_ksq

    def velocity_from_vorticity(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        omega_hat = np.fft.fft2(omega)
        psi_hat = self.stream_function_hat(omega_hat)
        u = np.fft.ifft2(1j * self.ky * psi_hat).real
        v = np.fft.ifft2(-1j * self.kx * psi_hat).real
        return u, v

    def _maybe_dealias(self, field_hat: np.ndarray) -> np.ndarray:
        if not self.dealias:
            return field_hat
        return field_hat * self._dealias_mask

    def rhs(self, omega: np.ndarray) -> np.ndarray:
        """dω/dt = −u · ∇ω + ν Δω."""
        omega_hat = self._maybe_dealias(np.fft.fft2(omega))
        psi_hat = self.stream_function_hat(omega_hat)
        u = np.fft.ifft2(1j * self.ky * psi_hat).real
        v = np.fft.ifft2(-1j * self.kx * psi_hat).real
        dx_omega = np.fft.ifft2(1j * self.kx * omega_hat).real
        dy_omega = np.fft.ifft2(1j * self.ky * omega_hat).real
        advection = u * dx_omega + v * dy_omega
        advection_hat = self._maybe_dealias(np.fft.fft2(advection))
        diffusion_hat = -self.nu * self.ksq * omega_hat
        return np.fft.ifft2(-advection_hat + diffusion_hat).real

    def step_rk4(self, omega: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.rhs(omega)
        k2 = self.rhs(omega + 0.5 * dt * k1)
        k3 = self.rhs(omega + 0.5 * dt * k2)
        k4 = self.rhs(omega + dt * k3)
        return omega + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_taylor_green(
    n_grid: int = 32,
    nu: float = 0.05,
    dt: float = 0.005,
    n_steps: int = 40,
    dealias: bool = True,
) -> SimulateResult:
    """Integra Taylor–Green y compara con la solución exacta en cada paso."""
    solver = SpectralNS2D(n_grid=n_grid, nu=nu, dealias=dealias)
    omega = taylor_green_vorticity(n_grid, nu=nu, t=0.0)
    times = np.empty(n_steps + 1, dtype=float)
    energy = np.empty(n_steps + 1, dtype=float)
    energy_exact = np.empty(n_steps + 1, dtype=float)
    vorticity_l2_error = np.empty(n_steps + 1, dtype=float)

    def snapshot(step: int, field: np.ndarray) -> None:
        t = step * dt
        u, v = solver.velocity_from_vorticity(field)
        exact_omega = taylor_green_vorticity(n_grid, nu=nu, t=t)
        times[step] = t
        energy[step] = kinetic_energy_density(u, v)
        energy_exact[step] = kinetic_energy_exact(nu, t)
        vorticity_l2_error[step] = float(np.sqrt(np.mean((field - exact_omega) ** 2)))

    snapshot(0, omega)
    for step in range(1, n_steps + 1):
        omega = solver.step_rk4(omega, dt)
        snapshot(step, omega)

    return SimulateResult(
        times=times,
        energy=energy,
        energy_exact=energy_exact,
        vorticity_l2_error=vorticity_l2_error,
        omega=omega,
        nu=nu,
        dt=dt,
        n_grid=n_grid,
    )
