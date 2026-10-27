# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Proyector de Leray discreto en Fourier (contraparte computacional de Lean).

Para un número de onda k ≠ 0:

    P̂(k) u = u − k (k · u) / |k|²

y P̂(0) = 0 (media nula). En enteros, el numerador |k|² u − k (k · u)
es divergencia-libre de forma exacta: k · numerador = 0.
"""

from __future__ import annotations

import numpy as np


def integer_wavenumbers(n_grid: int) -> np.ndarray:
    """Números de onda enteros compatibles con FFT periódica en [0, 2π)."""
    return np.fft.fftfreq(n_grid, d=1.0 / n_grid)


def wavenumber_mesh(n_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx = integer_wavenumbers(n_grid)
    ky = integer_wavenumbers(n_grid)
    kx2, ky2 = np.meshgrid(kx, ky, indexing="ij")
    ksq = kx2 * kx2 + ky2 * ky2
    return kx2, ky2, ksq


def mean_square_gradient(u: np.ndarray, v: np.ndarray) -> float:
    """Mean of |∇u|² from spectral derivatives on the torus [0, 2π)²."""
    n_grid = u.shape[0]
    kx, ky, _ = wavenumber_mesh(n_grid)
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    ux = np.fft.ifft2(1j * kx * u_hat).real
    uy = np.fft.ifft2(1j * ky * u_hat).real
    vx = np.fft.ifft2(1j * kx * v_hat).real
    vy = np.fft.ifft2(1j * ky * v_hat).real
    return float(np.mean(ux * ux + uy * uy + vx * vx + vy * vy))


def leray_numerator(kx: int, ky: int, ux: int, uy: int) -> tuple[int, int]:
    """Numerador entero |k|² u − k (k · u). Ver `lean/NavierStokesLab/Leray.lean`."""
    ksq = kx * kx + ky * ky
    divergence = kx * ux + ky * uy
    return ksq * ux - kx * divergence, ksq * uy - ky * divergence


def leray_numerator_divergence(kx: int, ky: int, ux: int, uy: int) -> int:
    px, py = leray_numerator(kx, ky, ux, uy)
    return kx * px + ky * py


def _zero_nyquist(field_hat: np.ndarray) -> np.ndarray:
    """Anula el modo de Nyquist (k = ±n/2), mal definido en FFT reales de tamaño par."""
    n_grid = field_hat.shape[0]
    if n_grid % 2 == 0:
        nyquist = n_grid // 2
        field_hat = field_hat.copy()
        field_hat[nyquist, :] = 0.0
        field_hat[:, nyquist] = 0.0
    return field_hat


def leray_project(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Proyecta un campo (u, v) al subespacio espectral de divergencia nula."""
    if u.shape != v.shape or u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("u and v must be square arrays of the same shape")
    n_grid = u.shape[0]
    kx2, ky2, ksq = wavenumber_mesh(n_grid)
    u_hat = _zero_nyquist(np.fft.fft2(u))
    v_hat = _zero_nyquist(np.fft.fft2(v))
    divergence_hat = kx2 * u_hat + ky2 * v_hat
    ksq_safe = ksq.copy()
    ksq_safe[0, 0] = 1.0
    scale = np.where(ksq == 0.0, 0.0, divergence_hat / ksq_safe)
    u_hat = _zero_nyquist(u_hat - kx2 * scale)
    v_hat = _zero_nyquist(v_hat - ky2 * scale)
    u_divfree = np.fft.ifft2(u_hat).real
    v_divfree = np.fft.ifft2(v_hat).real
    return u_divfree, v_divfree


def spectral_divergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Divergencia ∂x u + ∂y v en el espacio de Fourier."""
    n_grid = u.shape[0]
    kx2, ky2, _ = wavenumber_mesh(n_grid)
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    return np.fft.ifft2(1j * kx2 * u_hat + 1j * ky2 * v_hat).real
