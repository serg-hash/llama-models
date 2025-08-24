# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from mpmath import mp

mp.dps = 50


def theta(t):
    return mp.arg(mp.gamma(0.25 + 0.5j * t)) - t * mp.log(mp.sqrt(mp.pi))


def dtheta_dt(t):
    """
    θ'(t) = 1/2 * Re[ψ(1/4 + i t/2)] - log(√π)
    donde ψ es la digamma. Esto evita diferencias finitas inestables.
    """
    z = mp.mpf("0.25") + 0.5j * mp.mpf(t)
    return 0.5 * mp.re(mp.digamma(z)) - mp.log(mp.sqrt(mp.pi))
