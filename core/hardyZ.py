# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from mpmath import mp

from .theta import dtheta_dt, theta
from .zeta import zeta

mp.dps = 50


def hardy_z(t):
    s = mp.mpf("0.5") + 1j * mp.mpf(t)
    return mp.e ** (1j * theta(t)) * zeta(s)


def dz_dt(t):
    """
    Derivada de Z(t) con regla del producto:
    Z(t) = e^{iθ(t)} ζ(1/2+it)
    Z'(t) ≈ i θ'(t) e^{iθ(t)} ζ(s) + e^{iθ(t)} * i * ζ'(s)
    Aproximamos ζ'(s) con diferencias finitas en s (paso pequeño y adaptativo).
    """
    s = mp.mpf("0.5") + 1j * mp.mpf(t)
    et = mp.e ** (1j * theta(t))
    # paso adaptativo en el eje imaginario (clamp)
    ht = mp.mpf("1e-6") * (1 + abs(t)) ** (-0.5)
    zeta_p = (zeta(s + 1j * ht) - zeta(s - 1j * ht)) / (2j * ht)
    return 1j * dtheta_dt(t) * et * zeta(s) + et * (1j * zeta_p)
