# Copyright 2026 Youngseok Lee
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


class CudaQVisibleSampler:
    def __init__(self, target: str = "nvidia", angle_scale: float = 2.0) -> None:
        import cudaq

        self.cudaq = cudaq
        self.target = target
        self.angle_scale = float(angle_scale)

    @staticmethod
    def _apply_multi_rz(kernel: Any, angle: Any, qubits: Any, wires: list[int]) -> None:
        if not wires:
            raise ValueError("MultiRZ generator must contain at least one wire")
        if len(wires) == 1:
            kernel.rz(angle, qubits[wires[0]])
            return
        target = wires[-1]
        for control in wires[:-1]:
            kernel.cx(qubits[control], qubits[target])
        kernel.rz(angle, qubits[target])
        for control in reversed(wires[:-1]):
            kernel.cx(qubits[control], qubits[target])

    def _build_kernel(self, n_qubits: int, gates: list[list[list[int]]]) -> Any:
        kernel, params = self.cudaq.make_kernel(list[float])
        qubits = kernel.qalloc(n_qubits)
        for wire in range(n_qubits):
            kernel.h(qubits[wire])
        for param_index, gate in enumerate(gates):
            for generator in gate:
                self._apply_multi_rz(kernel, self.angle_scale * params[param_index], qubits, generator)
        for wire in range(n_qubits):
            kernel.h(qubits[wire])
        kernel.mz(qubits)
        return kernel

    def sample_gates(
        self,
        n_qubits: int,
        gates: list[list[list[int]]],
        params: np.ndarray,
        num_samples: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        self.cudaq.set_target(self.target)
        kernel = self._build_kernel(int(n_qubits), gates)
        counts = self.cudaq.sample(kernel, np.asarray(params, dtype=float).tolist(), shots_count=int(num_samples))
        rows: list[list[int]] = []
        for bitstring, count in counts.items():
            rows.extend([[int(bit) for bit in bitstring]] * int(count))
        bits = torch.as_tensor(rows[:num_samples], dtype=torch.long)
        if bits.shape != (num_samples, n_qubits):
            raise ValueError(f"CUDA-Q returned shape {tuple(bits.shape)}, expected {(num_samples, n_qubits)}")
        if torch.any((bits != 0) & (bits != 1)):
            raise ValueError("CUDA-Q returned non-binary samples")
        return bits

    def sample_artifact(self, artifact_path: str | Path, num_samples: int) -> torch.Tensor:
        artifact = torch.load(artifact_path, map_location="cpu")
        n_qubits = int(artifact["n_qubits"])
        if int(artifact.get("ancilla_bits", 0)) != 0:
            raise ValueError("sample_artifact only supports direct visible sampling for ancilla_bits=0")
        params = torch.as_tensor(artifact["params"], dtype=torch.float64).numpy()
        return self.sample_gates(n_qubits, artifact["gates"], params, int(num_samples))


class BitflipVisibleSampler:
    def __init__(self) -> None:
        pass

    def sample_gates(
        self,
        n_qubits: int,
        gates: list[list[list[int]]],
        params: np.ndarray,
        num_samples: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        rng = np.random.default_rng(None if seed is None else int(seed))
        n_qubits = int(n_qubits)
        num_samples = int(num_samples)
        bits = np.zeros((num_samples, n_qubits), dtype=np.uint8)
        flat_params = np.asarray(params, dtype=float).reshape(-1)
        if len(gates) != flat_params.shape[0]:
            raise ValueError(f"Expected {len(gates)} params, got {flat_params.shape[0]}")

        for param, gate in zip(flat_params, gates):
            prob = float(np.sin(float(param)) ** 2)
            if prob <= 0.0:
                continue
            if prob >= 1.0:
                flips = np.ones(num_samples, dtype=bool)
            else:
                flips = rng.random(num_samples) < prob
            if not bool(np.any(flips)):
                continue
            for generator in gate:
                for wire in generator:
                    bits[flips, int(wire)] ^= 1

        out = torch.as_tensor(bits, dtype=torch.long)
        expected_shape = (num_samples, n_qubits)
        if out.shape != expected_shape:
            raise ValueError(f"bitflip sampler returned shape {tuple(out.shape)}, expected {expected_shape}")
        return out

    def sample_artifact(self, artifact_path: str | Path, num_samples: int, seed: int | None = None) -> torch.Tensor:
        artifact = torch.load(artifact_path, map_location="cpu")
        n_qubits = int(artifact["n_qubits"])
        if int(artifact.get("ancilla_bits", 0)) != 0:
            raise ValueError("sample_artifact only supports direct visible sampling for ancilla_bits=0")
        params = torch.as_tensor(artifact["params"], dtype=torch.float64).numpy()
        return self.sample_gates(n_qubits, artifact["gates"], params, int(num_samples), seed=seed)


class SingleQubitIqpSampler:
    def __init__(self, angle_scale: float = 2.0) -> None:
        self.angle_scale = float(angle_scale)

    def sample_gates(
        self,
        n_qubits: int,
        gates: list[list[list[int]]],
        params: np.ndarray,
        num_samples: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        rng = np.random.default_rng(None if seed is None else int(seed))
        n_qubits = int(n_qubits)
        num_samples = int(num_samples)
        flat_params = np.asarray(params, dtype=float).reshape(-1)
        if len(gates) != flat_params.shape[0]:
            raise ValueError(f"Expected {len(gates)} params, got {flat_params.shape[0]}")

        angles = np.zeros(n_qubits, dtype=np.float64)
        for param, gate in zip(flat_params, gates):
            for generator in gate:
                wires = [int(wire) for wire in generator]
                if len(wires) != 1:
                    raise ValueError(
                        "single_qubit_iqp sampler only supports one-local generators; "
                        f"got generator {generator}"
                    )
                wire = wires[0]
                if wire < 0 or wire >= n_qubits:
                    raise ValueError(f"Invalid wire {wire} for n_qubits={n_qubits}")
                angles[wire] += self.angle_scale * float(param)

        probs_one = np.sin(0.5 * angles) ** 2
        probs_one = np.clip(probs_one, 0.0, 1.0)
        bits = rng.random((num_samples, n_qubits)) < probs_one[None, :]
        return torch.as_tensor(bits, dtype=torch.long)

    def sample_artifact(self, artifact_path: str | Path, num_samples: int, seed: int | None = None) -> torch.Tensor:
        artifact = torch.load(artifact_path, map_location="cpu")
        n_qubits = int(artifact["n_qubits"])
        if int(artifact.get("ancilla_bits", 0)) != 0:
            raise ValueError("sample_artifact only supports direct visible sampling for ancilla_bits=0")
        params = torch.as_tensor(artifact["params"], dtype=torch.float64).numpy()
        return self.sample_gates(n_qubits, artifact["gates"], params, int(num_samples), seed=seed)


def _fwht(values: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh-Hadamard transform over the last flat dimension."""
    out = values.clone()
    size = int(out.numel())
    if size <= 0 or size & (size - 1):
        raise ValueError(f"FWHT input size must be a positive power of two, got {size}")
    width = 1
    while width < size:
        view = out.view(-1, width * 2)
        left = view[:, :width].clone()
        right = view[:, width:].clone()
        view[:, :width] = left + right
        view[:, width:] = left - right
        width *= 2
    return out


def _wire_mask(wires: list[int]) -> int:
    mask = 0
    for wire in wires:
        mask |= 1 << int(wire)
    return mask


class ExactIqpSampler:
    def __init__(
        self,
        device: str = "cuda",
        angle_scale: float = 2.0,
        bit_order: str = "little",
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        if bit_order not in {"little", "big"}:
            raise ValueError("bit_order must be 'little' or 'big'")
        self.device = torch.device(device)
        self.angle_scale = float(angle_scale)
        self.bit_order = bit_order

    def _probabilities(self, n_qubits: int, gates: list[list[list[int]]], params: np.ndarray) -> torch.Tensor:
        n_qubits = int(n_qubits)
        num_states = 1 << n_qubits
        theta = torch.zeros(num_states, dtype=torch.float64, device=self.device)
        half_angle_scale = 0.5 * self.angle_scale
        flat_params = np.asarray(params, dtype=float).reshape(-1)
        if len(gates) != flat_params.shape[0]:
            raise ValueError(f"Expected {len(gates)} params, got {flat_params.shape[0]}")

        for param, gate in zip(flat_params, gates):
            angle = float(param) * half_angle_scale
            if angle == 0.0:
                continue
            for generator in gate:
                theta[_wire_mask(generator)] += angle

        diagonal_phase_angle = _fwht(theta)
        phase = torch.exp(-1j * diagonal_phase_angle.to(torch.complex128))
        amplitudes = _fwht(phase) / float(num_states)
        probs = amplitudes.real.square() + amplitudes.imag.square()
        probs = torch.clamp(probs.real, min=0.0)
        norm = probs.sum()
        if not torch.isfinite(norm) or float(norm.item()) <= 0.0:
            raise ValueError("Invalid exact IQP probability distribution")
        return probs / norm

    def sample_gates(
        self,
        n_qubits: int,
        gates: list[list[list[int]]],
        params: np.ndarray,
        num_samples: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        probs = self._probabilities(int(n_qubits), gates, params)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=probs.device).manual_seed(int(seed))
        indices = torch.multinomial(probs, int(num_samples), replacement=True, generator=generator)
        shifts = torch.arange(int(n_qubits), device=indices.device, dtype=torch.long)
        bits = ((indices[:, None] >> shifts[None, :]) & 1).to(torch.long)
        if self.bit_order == "big":
            bits = torch.flip(bits, dims=[1])
        return bits.cpu()

    def sample_artifact(self, artifact_path: str | Path, num_samples: int, seed: int | None = None) -> torch.Tensor:
        artifact = torch.load(artifact_path, map_location="cpu")
        n_qubits = int(artifact["n_qubits"])
        if int(artifact.get("ancilla_bits", 0)) != 0:
            raise ValueError("sample_artifact only supports direct visible sampling for ancilla_bits=0")
        params = torch.as_tensor(artifact["params"], dtype=torch.float64).numpy()
        return self.sample_gates(n_qubits, artifact["gates"], params, int(num_samples), seed=seed)


def _make_sampler(
    backend: str,
    target: str,
    device: str,
    angle_scale: float,
    bit_order: str,
) -> CudaQVisibleSampler | ExactIqpSampler | BitflipVisibleSampler | SingleQubitIqpSampler:
    if backend == "cudaq":
        return CudaQVisibleSampler(target=target, angle_scale=angle_scale)
    if backend == "exact_iqp":
        return ExactIqpSampler(device=device, angle_scale=angle_scale, bit_order=bit_order)
    if backend == "bitflip":
        return BitflipVisibleSampler()
    if backend == "single_qubit_iqp":
        return SingleQubitIqpSampler(angle_scale=angle_scale)
    raise ValueError(
        f"Unsupported sample backend {backend!r}; expected "
        "'cudaq', 'exact_iqp', 'bitflip', or 'single_qubit_iqp'"
    )


def sample_ancilla_branches(
    branches_dir: str | Path,
    num_samples: int,
    target: str = "nvidia",
    backend: str = "cudaq",
    device: str = "cuda",
    angle_scale: float = 2.0,
    bit_order: str = "little",
    seed: int = 1234,
    branch_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    branches_dir = Path(branches_dir)
    summary = json.loads((branches_dir / "summary.json").read_text(encoding="utf-8"))
    branches = [str(branch["branch"]) for branch in summary["branches"]]
    if not branches:
        raise ValueError(f"No branches found in {branches_dir}")
    if branch_weights is None:
        base_count = int(num_samples) // len(branches)
        remainder = int(num_samples) % len(branches)
        branch_counts = {
            branch: base_count + (1 if index < remainder else 0)
            for index, branch in enumerate(branches)
        }
    else:
        weights = np.asarray([float(branch_weights.get(branch, 0.0)) for branch in branches], dtype=np.float64)
        if np.any(weights < 0) or not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
            raise ValueError(f"Invalid branch_weights for {branches_dir}: {branch_weights}")
        weights = weights / float(weights.sum())
        raw = weights * float(num_samples)
        counts = np.floor(raw).astype(np.int64)
        remainder = int(num_samples) - int(counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            for index in order[:remainder]:
                counts[int(index)] += 1
        branch_counts = {branch: int(counts[index]) for index, branch in enumerate(branches)}
    sampler = _make_sampler(
        backend=str(backend),
        target=str(target),
        device=str(device),
        angle_scale=float(angle_scale),
        bit_order=str(bit_order),
    )
    chunks: list[torch.Tensor] = []
    for branch_index, branch in enumerate(branches):
        count = branch_counts[branch]
        if count <= 0:
            continue
        branch_dir = branches_dir / f"branch_{branch}"
        meta = json.loads((branch_dir / "branch_meta.json").read_text(encoding="utf-8"))
        gates = json.loads((branch_dir / meta["gates_file"]).read_text(encoding="utf-8"))
        params = np.load(branch_dir / meta["params_file"]).astype(float)
        sample_seed = int(seed) + 1009 * int(branch_index) if backend in {"exact_iqp", "bitflip", "single_qubit_iqp"} else None
        chunks.append(sampler.sample_gates(int(meta["visible_bits"]), gates, params, count, seed=sample_seed))
    bits = torch.cat(chunks, dim=0)
    order = torch.randperm(bits.shape[0], generator=torch.Generator().manual_seed(int(seed)))
    return bits[order][: int(num_samples)], branch_counts


def save_samples(
    output_path: str | Path,
    bits: torch.Tensor,
    source: str,
    metadata: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bits": bits.cpu().to(torch.long),
            "samples": bits.cpu().to(torch.long),
            "source": source,
            "metadata": metadata,
        },
        output_path,
    )

