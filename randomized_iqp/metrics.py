# Copyright 2026 Youngseok Lee
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .data import load_binary_csv


def load_sample_bits(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        if "bits" in payload:
            return payload["bits"].to(torch.long)
        if "samples" in payload:
            return payload["samples"].to(torch.long)
    return torch.as_tensor(payload, dtype=torch.long)


def subset_bits(bits: torch.Tensor, max_samples: int | None, seed: int) -> torch.Tensor:
    bits = bits.to(torch.long)
    if max_samples is None or bits.shape[0] <= max_samples:
        return bits
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(bits.shape[0], generator=generator)[: int(max_samples)]
    return bits[indices]


def hamming_rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    distances = torch.cdist(x.to(torch.float32), y.to(torch.float32), p=1)
    return torch.exp(-distances / (2.0 * float(sigma) ** 2))


def mmd2_hamming_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float) -> float:
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("MMD requires at least two samples in each set")
    k_xx = hamming_rbf_kernel(x, x, sigma)
    k_yy = hamming_rbf_kernel(y, y, sigma)
    k_xy = hamming_rbf_kernel(x, y, sigma).mean()
    n_x = x.shape[0]
    n_y = y.shape[0]
    u_xx = (k_xx.sum() - k_xx.diagonal().sum()) / (n_x * (n_x - 1))
    u_yy = (k_yy.sum() - k_yy.diagonal().sum()) / (n_y * (n_y - 1))
    return float((u_xx + u_yy - 2.0 * k_xy).item())


def bit_distribution_metrics(real_bits: torch.Tensor, fake_bits: torch.Tensor) -> dict[str, float]:
    real = real_bits.to(torch.float32)
    fake = fake_bits.to(torch.float32)
    real_corr = real.T @ real / real.shape[0]
    fake_corr = fake.T @ fake / fake.shape[0]
    return {
        "bit_marginal_l1": float((real.mean(dim=0) - fake.mean(dim=0)).abs().mean().item()),
        "pairwise_bit_correlation_l1": float((real_corr - fake_corr).abs().mean().item()),
    }


def sample_diversity_metrics(bits: torch.Tensor) -> dict[str, float | int]:
    unique = int(torch.unique(bits.to(torch.long), dim=0).shape[0])
    total = int(bits.shape[0])
    return {
        "num_samples": total,
        "unique_samples": unique,
        "unique_rate": float(unique / max(total, 1)),
        "duplicate_rate": float(1.0 - unique / max(total, 1)),
    }


def evaluate_against_csv(
    real_path: str | Path,
    fake_bits: torch.Tensor,
    sigmas: list[float],
    max_samples: int | None,
    seed: int,
) -> dict[str, Any]:
    real_bits = torch.as_tensor(load_binary_csv(real_path), dtype=torch.long)
    n = min(real_bits.shape[0], fake_bits.shape[0])
    real_bits = subset_bits(real_bits, min(n, max_samples) if max_samples is not None else n, seed)
    fake_bits = subset_bits(fake_bits[:n], real_bits.shape[0], seed + 1)
    row: dict[str, Any] = {
        "num_real": int(real_bits.shape[0]),
        "num_fake": int(fake_bits.shape[0]),
        **bit_distribution_metrics(real_bits, fake_bits),
    }
    values = []
    for sigma in sigmas:
        value = mmd2_hamming_rbf(real_bits, fake_bits, float(sigma))
        row[f"mmd2_sigma_{sigma:g}"] = value
        values.append(value)
    row["mmd2_mean"] = float(sum(values) / len(values))
    return row

