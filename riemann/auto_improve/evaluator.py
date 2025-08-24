# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Evaluation of zero-finding algorithms."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .z_function import hardy_z


class Evaluator:
    """Evaluate algorithms for locating zeros of Hardy's Z."""

    def __init__(self, zeros_ref: Iterable[float], tol: float = 1e-8) -> None:
        self.zeros_ref = np.array(list(zeros_ref), dtype=float)
        self.tol = tol

    def find_zeros(self, n: int, step: float = 0.2) -> List[float]:
        """Locate first *n* zeros with a simple sign-change search."""
        zeros: List[float] = []
        t = 0.0
        z_prev = hardy_z(t)[0]
        while len(zeros) < n:
            t += step
            z_curr = hardy_z(t)[0]
            if z_prev == 0:
                zeros.append(t - step)
            elif z_prev * z_curr < 0:
                zeros.append(float(self._bisect(t - step, t)))
            z_prev = z_curr
        return zeros

    def _bisect(self, a: float, b: float, iters: int = 50) -> float:
        fa = hardy_z(a)[0]
        fb = hardy_z(b)[0]
        for _ in range(iters):
            m = (a + b) / 2
            fm = hardy_z(m)[0]
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return (a + b) / 2

    def metrics(self, zeros: Iterable[float]) -> dict:
        zeros = np.array(list(zeros), dtype=float)
        n = min(len(self.zeros_ref), len(zeros))
        errors = np.abs(zeros[:n] - self.zeros_ref[:n])
        mae = float(errors.mean())
        captured = float(np.mean(errors <= self.tol))
        return {"mae_t": mae, "captured_rate": captured}


def load_reference(path: Path | None = None) -> List[float]:
    if path is None:
        path = Path(__file__).resolve().parent / "data" / "zeros_first_20.json"
    with path.open() as f:
        return json.load(f)
