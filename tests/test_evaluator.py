# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from riemann.auto_improve.evaluator import Evaluator, load_reference


def test_baseline_evaluator():
    zeros_ref = load_reference()[:5]
    ev = Evaluator(zeros_ref, tol=1e-6)
    zeros = ev.find_zeros(len(zeros_ref))
    metrics = ev.metrics(zeros)
    assert metrics["captured_rate"] >= 0.8
