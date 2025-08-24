# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Hardy Z function utilities."""
from __future__ import annotations

import mpmath as mp

import numpy as np

mp.mp.dps = 50


def theta(t: float) -> mp.mpf:
    """Riemann--Siegel theta function."""
    t = mp.mpf(t)
    return mp.arg(mp.gamma(mp.mpf("0.25") + 0.5j * t)) - t / 2 * mp.log(mp.pi)


def hardy_z(t: np.ndarray | float) -> np.ndarray:
    """Compute Hardy's Z(t) for scalar or array-like t."""
    ts = np.atleast_1d(np.asarray(t, dtype=float))
    result = []
    for ti in ts:
        z = mp.zeta(mp.mpf("0.5") + 1j * ti) * mp.exp(1j * theta(ti))
        result.append(float(mp.re(z)))
    return np.array(result)
