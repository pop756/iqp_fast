# Copyright 2026 Youngseok Lee
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_binary_csv(path: str | Path) -> np.ndarray:
    bits = np.loadtxt(path, delimiter=",", dtype=np.int32)
    if bits.ndim == 1:
        bits = bits.reshape(1, -1)
    if np.any(bits == -1):
        bits = ((bits + 1) // 2).astype(np.int32)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError(f"{path} contains values other than 0/1 or -1/+1")
    return bits.astype(np.int32, copy=False)

