# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Tests del laboratorio: identidades de Leray, Taylor–Green y el solver."""

from __future__ import annotations

import unittest

import numpy as np

from navier_stokes_lab.projector import (
    leray_numerator,
    leray_numerator_divergence,
    leray_project,
    mean_square_gradient,
    spectral_divergence,
)
from navier_stokes_lab.solver import SpectralNS2D, simulate_taylor_green
from navier_stokes_lab.taylor_green import (
    energy_time_derivative_exact,
    kinetic_energy_density,
    kinetic_energy_exact,
    mean_square_vorticity,
    taylor_green_fields,
    taylor_green_vorticity,
)


class TestLerayIdentity(unittest.TestCase):
    """Las mismas identidades que `lean/NavierStokesLab/Leray.lean`."""

    def test_numerator_is_divergence_free_on_random_integers(self):
        rng = np.random.default_rng(2026)
        for kx, ky, ux, uy in rng.integers(-12, 13, size=(80, 4)):
            self.assertEqual(leray_numerator_divergence(int(kx), int(ky), int(ux), int(uy)), 0)

    def test_numerator_is_idempotent_up_to_ksq(self):
        rng = np.random.default_rng(7)
        for kx, ky, ux, uy in rng.integers(-8, 9, size=(40, 4)):
            kx, ky, ux, uy = (int(kx), int(ky), int(ux), int(uy))
            ksq = kx * kx + ky * ky
            px, py = leray_numerator(kx, ky, ux, uy)
            qx, qy = leray_numerator(kx, ky, px, py)
            self.assertEqual(qx, ksq * px)
            self.assertEqual(qy, ksq * py)

    def test_fixes_divergence_free_modes(self):
        # k · u = 0 ⇒ numerador = |k|² u
        kx, ky, ux, uy = 3, -1, 1, 3
        self.assertEqual(kx * ux + ky * uy, 0)
        ksq = kx * kx + ky * ky
        px, py = leray_numerator(kx, ky, ux, uy)
        self.assertEqual((px, py), (ksq * ux, ksq * uy))


class TestTaylorGreen(unittest.TestCase):
    def test_divergence_is_machine_zero(self):
        u, v = taylor_green_fields(32, nu=0.1, t=0.3)
        div = spectral_divergence(u, v)
        self.assertLess(np.max(np.abs(div)), 1e-12)

    def test_energy_formula(self):
        u, v = taylor_green_fields(48, nu=0.05, t=0.2)
        self.assertAlmostEqual(
            kinetic_energy_density(u, v), kinetic_energy_exact(0.05, 0.2), places=10
        )

    def test_formal_energy_identity(self):
        """Numerical counterpart of `lean/NavierStokesLab/Energy.lean`.

        For Taylor–Green, d/dt (½ ‖u‖²) + ν ‖∇u‖² = 0, and in 2D
        periodic incompressible flow ‖∇u‖² = ‖ω‖².
        """
        nu, t, n_grid = 0.05, 0.2, 48
        u, v = taylor_green_fields(n_grid, nu=nu, t=t)
        omega = taylor_green_vorticity(n_grid, nu=nu, t=t)
        dE_dt = energy_time_derivative_exact(nu, t)
        grad_sq = mean_square_gradient(u, v)
        omega_sq = mean_square_vorticity(omega)
        self.assertAlmostEqual(grad_sq, omega_sq, places=10)
        self.assertAlmostEqual(dE_dt + nu * grad_sq, 0.0, places=10)
        self.assertAlmostEqual(dE_dt, -4.0 * nu * kinetic_energy_exact(nu, t), places=12)

    def test_vorticity_matches_curl(self):
        n_grid = 32
        u, v = taylor_green_fields(n_grid, nu=0.0, t=0.0)
        solver = SpectralNS2D(n_grid=n_grid, nu=0.0, dealias=False)
        # ω = ∂x v − ∂y u reconstruida desde (u,v) vía FFT
        kx, ky = solver.kx, solver.ky
        omega_from_vel = np.fft.ifft2(
            1j * kx * np.fft.fft2(v) - 1j * ky * np.fft.fft2(u)
        ).real
        omega_exact = taylor_green_vorticity(n_grid, nu=0.0, t=0.0)
        self.assertLess(np.max(np.abs(omega_from_vel - omega_exact)), 1e-12)


class TestSpectralSolver(unittest.TestCase):
    def test_nonlinear_term_vanishes_on_taylor_green(self):
        n_grid = 32
        solver = SpectralNS2D(n_grid=n_grid, nu=0.0, dealias=True)
        omega = taylor_green_vorticity(n_grid, nu=0.0, t=0.0)
        residual = solver.rhs(omega)
        self.assertLess(np.max(np.abs(residual)), 1e-12)

    def test_velocity_roundtrip_from_vorticity(self):
        n_grid = 32
        solver = SpectralNS2D(n_grid=n_grid, nu=0.05)
        u_exact, v_exact = taylor_green_fields(n_grid, nu=0.05, t=0.0)
        omega = taylor_green_vorticity(n_grid, nu=0.05, t=0.0)
        u, v = solver.velocity_from_vorticity(omega)
        self.assertLess(np.max(np.abs(u - u_exact)), 1e-12)
        self.assertLess(np.max(np.abs(v - v_exact)), 1e-12)

    def test_leray_projection_clears_divergence(self):
        rng = np.random.default_rng(1)
        n_grid = 24
        u = rng.normal(size=(n_grid, n_grid))
        v = rng.normal(size=(n_grid, n_grid))
        pu, pv = leray_project(u, v)
        div = spectral_divergence(pu, pv)
        self.assertLess(np.max(np.abs(div)), 1e-10)

    def test_short_integration_tracks_exact_energy(self):
        result = simulate_taylor_green(n_grid=32, nu=0.05, dt=0.005, n_steps=20)
        self.assertLess(result.final_energy_rel_error, 1e-5)
        self.assertLess(result.vorticity_l2_error[-1], 1e-6)


if __name__ == "__main__":
    unittest.main()
