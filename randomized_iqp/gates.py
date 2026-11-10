# Copyright 2026 Youngseok Lee
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import math
import random
from typing import Any


Gate = list[list[list[int]]]


def fully_connected_gates(num_bits: int, max_order: int) -> Gate:
    gates: Gate = []
    for order in range(1, max_order + 1):
        for combo in itertools.combinations(range(num_bits), order):
            gates.append([list(combo)])
    return gates


def vertical_gates(rows: int, cols: int, max_row_order: int | None = None) -> Gate:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if max_row_order is None:
        max_row_order = cols
    if max_row_order <= 0 or max_row_order > cols:
        raise ValueError("max_row_order must be in [1, cols]")
    num_bits = rows * cols
    gates: Gate = []
    for row in range(rows):
        row_bits = [row * cols + col for col in range(cols)]
        for size in range(1, max_row_order + 1):
            for combo in itertools.combinations(row_bits, size):
                gates.append([list(combo)])
    for col in range(cols):
        column_bits = [row * cols + col for row in range(rows)]
        for combo in itertools.combinations(column_bits, 2):
            gates.append([list(combo)])
    expected = rows * sum(math.comb(cols, order) for order in range(1, max_row_order + 1)) + cols * math.comb(rows, 2)
    if len(gates) != expected:
        raise AssertionError(f"vertical gate count mismatch: {len(gates)} != {expected}")
    if num_bits <= 0:
        raise AssertionError("unreachable")
    return gates


def sparse_er_gates(num_bits: int, edge_probability: float, seed: int = 1234, include_singletons: bool = True) -> Gate:
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")
    if edge_probability < 0.0 or edge_probability > 1.0:
        raise ValueError("edge_probability must be in [0, 1]")
    rng = random.Random(int(seed))
    gates: Gate = []
    if include_singletons:
        for bit in range(num_bits):
            gates.append([[bit]])
    for i, j in itertools.combinations(range(num_bits), 2):
        if rng.random() < edge_probability:
            gates.append([[i, j]])
    return gates


def explicit_edges_gates(num_bits: int, edges: list[list[int]], include_singletons: bool = True) -> Gate:
    gates: Gate = []
    if include_singletons:
        for bit in range(num_bits):
            gates.append([[bit]])
    for i, j in edges:
        gates.append([[int(i), int(j)]])
    return gates


def build_visible_gates(name: str, layout: dict[str, Any], visible_bits: int) -> Gate:
    layout_type = str(layout["type"])
    if layout_type == "fully_connected":
        return fully_connected_gates(visible_bits, int(layout["max_order"]))
    if layout_type == "vertical":
        rows = int(layout["rows"])
        cols = int(layout["cols"])
        if rows * cols != visible_bits:
            raise ValueError(f"{name}: rows*cols={rows * cols} does not match visible_bits={visible_bits}")
        max_row_order = layout.get("max_row_order")
        return vertical_gates(rows, cols, int(max_row_order) if max_row_order is not None else None)
    if layout_type == "sparse_er":
        return sparse_er_gates(
            visible_bits,
            float(layout["edge_probability"]),
            int(layout.get("seed", 1234)),
            bool(layout.get("include_singletons", True)),
        )
    if layout_type == "explicit_edges":
        import numpy as np
        edges_path = str(layout["edges_path"])
        edges = np.load(edges_path).tolist()
        return explicit_edges_gates(
            visible_bits,
            edges,
            bool(layout.get("include_singletons", True)),
        )
    raise ValueError(f"Unsupported layout type {layout_type!r} for {name}")


def expected_gate_count(layout: dict[str, Any], visible_bits: int) -> int:
    layout_type = str(layout["type"])
    if layout_type == "fully_connected":
        max_order = int(layout["max_order"])
        return sum(math.comb(visible_bits, order) for order in range(1, max_order + 1))
    if layout_type == "vertical":
        rows = int(layout["rows"])
        cols = int(layout["cols"])
        max_row_order = int(layout.get("max_row_order", cols))
        return rows * sum(math.comb(cols, order) for order in range(1, max_row_order + 1)) + cols * math.comb(rows, 2)
    if layout_type == "sparse_er":
        return len(sparse_er_gates(
            visible_bits,
            float(layout["edge_probability"]),
            int(layout.get("seed", 1234)),
            bool(layout.get("include_singletons", True)),
        ))
    if layout_type == "explicit_edges":
        import numpy as np
        edges = np.load(str(layout["edges_path"]))
        singletons = int(bool(layout.get("include_singletons", True))) * visible_bits
        return singletons + len(edges)
    raise ValueError(f"Unsupported layout type {layout_type!r}")


def gate_key(gate: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(wire) for wire in generator) for generator in gate)

