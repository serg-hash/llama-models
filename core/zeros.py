# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from mpmath import mp, sign

from .hardyZ import dz_dt, hardy_z

mp.dps = 50


def refine_zero(t0, tol=1e-20, maxsteps=50):
    f = hardy_z
    df = dz_dt
    x = mp.mpf(t0)
    for _ in range(maxsteps):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < mp.mpf("1e-30"):
            break
        step = fx / dfx
        # limitar paso para evitar saltos grandes
        step = sign(step) * min(abs(step), mp.mpf("0.5"))
        xn = x - step
        if abs(xn - x) < mp.mpf(tol):
            return mp.mpf(xn)
        x = xn
    # último intento: bisección local si hay cambio de signo
    a = x - mp.mpf("0.5")
    b = x + mp.mpf("0.5")
    fa, fb = f(a), f(b)
    if sign(fa) != sign(fb):
        for _ in range(60):
            m = (a + b) / 2
            fm = f(m)
            if fm == 0 or abs(b - a) < mp.mpf(tol):
                return mp.mpf(m)
            if sign(fa) != sign(fm):
                b, fb = m, fm
            else:
                a, fa = m, fm
    return mp.mpf(x)


def find_zeros_in_window(t, dt, guesses=10):
    xs = [t + (i + 0.5) * dt / guesses for i in range(guesses)]
    zeros = []
    for x in xs:
        left = x - dt / (2 * guesses)
        right = x + dt / (2 * guesses)
        z1 = hardy_z(left)
        z2 = hardy_z(right)
        if sign(z1) == 0:
            zeros.append(mp.mpf(left))
            continue
        if sign(z1) != sign(z2):
            try:
                zr = refine_zero(x)
                if t <= zr <= t + dt:
                    zeros.append(mp.mpf(zr))
            except Exception:
                pass
    # quitar casi-duplicados por tolerancia
    zeros = sorted(zeros)
    dedup = []
    for z in zeros:
        if not dedup or abs(z - dedup[-1]) > mp.mpf("1e-9"):
            dedup.append(z)
    return dedup
