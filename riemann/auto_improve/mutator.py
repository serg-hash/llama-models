# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Simple configuration mutator."""
from __future__ import annotations

import copy
import random
from typing import Dict


def mutate(config: Dict[str, float]) -> Dict[str, float]:
    """Mutate step size within safe bounds."""
    new_cfg = copy.deepcopy(config)
    step = new_cfg.get("step", 0.5)
    step += random.uniform(-0.05, 0.05)
    new_cfg["step"] = max(0.05, step)
    return new_cfg
