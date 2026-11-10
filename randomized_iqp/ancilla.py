# Copyright 2026 Youngseok Lee
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def nonempty_subsets(items: list[int]) -> list[list[int]]:
    subsets: list[list[int]] = []
    for size in range(1, len(items) + 1):
        subsets.extend([list(combo) for combo in itertools.combinations(items, size)])
    return subsets


def augment_visible_gates_with_ancilla(
    visible_gates: list[list[list[int]]],
    visible_bits: int,
    ancilla_bits: int,
    policy: str = "all_subsets",
) -> tuple[list[list[list[int]]], list[int], list[int]]:
    visible_wires = list(range(visible_bits))
    ancilla_wires = list(range(visible_bits, visible_bits + ancilla_bits))
    if ancilla_bits <= 0:
        return list(visible_gates), visible_wires, []
    if policy != "all_subsets":
        raise ValueError(f"Unsupported ancilla gate policy {policy!r}; expected 'all_subsets'")

    gates = list(visible_gates)
    for gate in visible_gates:
        for generator in gate:
            for ancilla_subset in nonempty_subsets(ancilla_wires):
                gates.append([ancilla_subset + list(generator)])
    return gates, visible_wires, ancilla_wires


def branch_bitstrings(ancilla_bits: int) -> list[str]:
    return ["".join(bits) for bits in itertools.product("01", repeat=ancilla_bits)]


def reduce_to_visible_branch(
    gates: list[list[list[int]]],
    params: torch.Tensor | np.ndarray,
    visible_bits: int,
    ancilla_wires: list[int],
    branch_bits: str,
    zero_tol: float = 1e-12,
) -> tuple[list[list[list[int]]], np.ndarray]:
    if len(branch_bits) != len(ancilla_wires):
        raise ValueError("branch_bits length must match ancilla_wires length")
    signs = {wire: (1.0 if bit == "0" else -1.0) for wire, bit in zip(ancilla_wires, branch_bits)}
    params_np = np.asarray(params, dtype=np.float64)
    merged_params: dict[tuple[int, ...], float] = {}

    for param, gate in zip(params_np, gates):
        for generator in gate:
            sign = 1.0
            visible_generator = tuple(int(wire) for wire in generator if wire < visible_bits)
            for wire in generator:
                if wire in signs:
                    sign *= signs[wire]
            if visible_generator:
                merged_params[visible_generator] = merged_params.get(visible_generator, 0.0) + float(param) * sign

    visible_gates: list[list[list[int]]] = []
    visible_params: list[float] = []
    for visible_generator, param in merged_params.items():
        if abs(param) <= zero_tol:
            continue
        visible_gates.append([list(visible_generator)])
        visible_params.append(param)
    return visible_gates, np.asarray(visible_params, dtype=np.float64)


def visible_generator_coefficients(
    gates: list[list[list[int]]],
    params: torch.Tensor | np.ndarray,
    visible_bits: int | None = None,
    zero_tol: float = 1e-12,
) -> dict[tuple[int, ...], float]:
    params_np = np.asarray(params, dtype=np.float64)
    if len(gates) != len(params_np):
        raise ValueError(f"Expected {len(gates)} params, got {len(params_np)}")
    coefficients: dict[tuple[int, ...], float] = {}
    for param, gate in zip(params_np, gates):
        for generator in gate:
            key = tuple(int(wire) for wire in generator)
            if not key:
                continue
            if visible_bits is not None and any(wire < 0 or wire >= visible_bits for wire in key):
                raise ValueError(f"Generator {key} is not contained in {visible_bits} visible wires")
            coefficients[key] = coefficients.get(key, 0.0) + float(param)
    return {key: value for key, value in coefficients.items() if abs(value) > zero_tol}


def branch_character(branch_bits: str, subset_bits: str) -> float:
    if len(branch_bits) != len(subset_bits):
        raise ValueError("branch_bits and subset_bits must have the same length")
    sign = 1.0
    for branch_bit, subset_bit in zip(branch_bits, subset_bits):
        if subset_bit == "1" and branch_bit == "1":
            sign *= -1.0
        elif subset_bit not in {"0", "1"}:
            raise ValueError(f"Invalid subset bit {subset_bit!r}")
    return sign


def embed_visible_branches_into_ancilla(
    branch_circuits: dict[str, tuple[list[list[list[int]]], torch.Tensor | np.ndarray]],
    visible_bits: int,
    ancilla_bits: int,
    ancilla_wires: list[int] | None = None,
    zero_tol: float = 1e-12,
) -> tuple[list[list[list[int]]], np.ndarray, dict[str, Any]]:
    expected_branches = branch_bitstrings(int(ancilla_bits))
    if set(branch_circuits) != set(expected_branches):
        missing = sorted(set(expected_branches) - set(branch_circuits))
        extra = sorted(set(branch_circuits) - set(expected_branches))
        raise ValueError(f"branch_circuits must contain exactly {expected_branches}; missing={missing}, extra={extra}")
    ancilla_wires = list(range(int(visible_bits), int(visible_bits) + int(ancilla_bits))) if ancilla_wires is None else [int(wire) for wire in ancilla_wires]
    if len(ancilla_wires) != int(ancilla_bits):
        raise ValueError("ancilla_wires length must match ancilla_bits")
    if any(wire < int(visible_bits) for wire in ancilla_wires):
        raise ValueError("ancilla_wires must be outside the visible wire range")

    branch_coeffs = {
        branch: visible_generator_coefficients(gates, params, visible_bits=visible_bits, zero_tol=zero_tol)
        for branch, (gates, params) in branch_circuits.items()
    }
    visible_generators = sorted({key for coefficients in branch_coeffs.values() for key in coefficients})
    subset_bits_list = expected_branches
    normalizer = 1.0 / float(2 ** int(ancilla_bits))

    embedded_gates: list[list[list[int]]] = []
    embedded_params: list[float] = []
    coefficient_rows: list[dict[str, Any]] = []
    for visible_generator in visible_generators:
        theta_by_branch = {
            branch: float(branch_coeffs[branch].get(visible_generator, 0.0))
            for branch in expected_branches
        }
        for subset_bits in subset_bits_list:
            alpha = normalizer * sum(
                theta_by_branch[branch] * branch_character(branch, subset_bits)
                for branch in expected_branches
            )
            coefficient_rows.append(
                {
                    "visible_generator": list(visible_generator),
                    "ancilla_subset_bits": subset_bits,
                    "coefficient": float(alpha),
                }
            )
            if abs(alpha) <= zero_tol:
                continue
            ancilla_subset = [wire for wire, bit in zip(ancilla_wires, subset_bits) if bit == "1"]
            embedded_gates.append([ancilla_subset + list(visible_generator)])
            embedded_params.append(float(alpha))

    metadata = {
        "visible_bits": int(visible_bits),
        "ancilla_bits": int(ancilla_bits),
        "visible_wires": list(range(int(visible_bits))),
        "ancilla_wires": ancilla_wires,
        "branches": expected_branches,
        "coefficient_rows": coefficient_rows,
        "transform": "inverse_walsh_hadamard_over_ancilla_branches",
    }
    return embedded_gates, np.asarray(embedded_params, dtype=np.float64), metadata


def export_ancilla_branches(artifact_path: str | Path, output_dir: str | Path | None = None) -> Path:
    artifact_path = Path(artifact_path)
    artifact = torch.load(artifact_path, map_location="cpu")
    visible_bits = int(artifact["visible_bits"])
    ancilla_wires = [int(wire) for wire in artifact["ancilla_wires"]]
    gates = artifact["gates"]
    params = torch.as_tensor(artifact["params"], dtype=torch.float64).numpy()
    root = Path(output_dir) if output_dir else artifact_path.parent / "ancilla_branches"
    root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "artifact": str(artifact_path),
        "visible_bits": visible_bits,
        "ancilla_wires": ancilla_wires,
        "branches": [],
    }
    for branch in branch_bitstrings(len(ancilla_wires)):
        branch_dir = root / f"branch_{branch}"
        branch_dir.mkdir(parents=True, exist_ok=True)
        visible_gates, visible_params = reduce_to_visible_branch(gates, params, visible_bits, ancilla_wires, branch)
        (branch_dir / "gates_visible_all.json").write_text(json.dumps(visible_gates), encoding="utf-8")
        np.save(branch_dir / "params_visible_all.npy", visible_params)
        branch_meta = {
            "branch": branch,
            "visible_bits": visible_bits,
            "num_gates": len(visible_gates),
            "params_file": "params_visible_all.npy",
            "gates_file": "gates_visible_all.json",
        }
        (branch_dir / "branch_meta.json").write_text(json.dumps(branch_meta, indent=2), encoding="utf-8")
        summary["branches"].append(branch_meta)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return root

