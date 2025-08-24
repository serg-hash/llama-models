# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import numpy as np
from riemann.auto_improve.z_function import hardy_z


def test_symmetry():
    t = np.linspace(0, 5, 5)
    assert np.allclose(hardy_z(t), hardy_z(-t), atol=1e-10)
