# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Run auto-improvement loop."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict

import yaml

from .evaluator import Evaluator, load_reference
from .mutator import mutate
from .selector import select


def run_loop(
    iterations: int = 3, nzeros: int = 20, seed: int = 0, outdir: str = "runs/latest"
) -> Dict[str, float]:
    random.seed(seed)
    zeros_ref = load_reference()
    evaluator = Evaluator(zeros_ref)
    cfg = {"step": 0.5}
    os.makedirs(outdir, exist_ok=True)
    results = []
    for i in range(iterations):
        zeros = evaluator.find_zeros(nzeros, step=cfg["step"])
        metrics = evaluator.metrics(zeros)
        results.append({"iteration": i, "config": dict(cfg), "metrics": metrics})
        candidate = mutate(cfg)
        zeros_c = evaluator.find_zeros(nzeros, step=candidate["step"])
        metrics_c = evaluator.metrics(zeros_c)
        cfg, _ = select((cfg, metrics), [(candidate, metrics_c)])
    with open(Path(outdir) / "codex.yaml", "w") as f:
        yaml.safe_dump({"seed": seed, "final_config": cfg}, f)
    with open(Path(outdir) / "report.md", "w") as f:
        f.write("# Auto Improvement Report\n\n")
        for r in results:
            f.write(
                f"Iteration {r['iteration']}: step={r['config']['step']:.3f}, "
                f"MAE={r['metrics']['mae_t']:.3e}, captured={r['metrics']['captured_rate']:.3f}\n"
            )
    return cfg


if __name__ == "__main__":
    run_loop()
