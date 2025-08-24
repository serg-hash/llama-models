# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Select best configuration based on metrics."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple


Metrics = Dict[str, float]
Config = Dict[str, float]


def select(
    current: Tuple[Config, Metrics], candidates: Iterable[Tuple[Config, Metrics]]
) -> Tuple[Config, Metrics]:
    best_cfg, best_metrics = current
    best_score = best_metrics["mae_t"]
    for cfg, metrics in candidates:
        score = metrics["mae_t"]
        if score < best_score:
            best_cfg, best_metrics = cfg, metrics
            best_score = score
    return best_cfg, best_metrics
