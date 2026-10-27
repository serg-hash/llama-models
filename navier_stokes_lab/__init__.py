# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Laboratorio 2D de Navier–Stokes incompresible (Taylor–Green + Lean 4)."""

from navier_stokes_lab.projector import leray_project, spectral_divergence
from navier_stokes_lab.solver import SpectralNS2D, SimulateResult, simulate_taylor_green
from navier_stokes_lab.taylor_green import (
    kinetic_energy_exact,
    taylor_green_fields,
    taylor_green_vorticity,
)

__all__ = [
    "SpectralNS2D",
    "SimulateResult",
    "kinetic_energy_exact",
    "leray_project",
    "simulate_taylor_green",
    "spectral_divergence",
    "taylor_green_fields",
    "taylor_green_vorticity",
]
