# Copyright 2026 Youngseok Lee
# SPDX-License-Identifier: Apache-2.0
# Parts (the MMD loss estimators) are adapted from iqpopt (https://github.com/XanaduAI/iqpopt),
# Apache License 2.0, and modified; see LICENSE and NOTICE.

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from .ancilla import augment_visible_gates_with_ancilla, export_ancilla_branches
from .config import ExperimentConfig
from .data import load_binary_csv
from .gates import build_visible_gates, expected_gate_count, gate_key
from .io import save_json



def branch_bits_for_index(index: int, width: int) -> str:
    if width < 0:
        raise ValueError(f"branch width must be non-negative, got {width}")
    if width == 0:
        return ""
    if index < 0 or index >= 2 ** width:
        raise ValueError(f"branch index {index} does not fit in {width} ancilla bits")
    return format(int(index), f"0{width}b")


def label_branch_mapping(labels: np.ndarray | list[int], ancilla_bits: int) -> list[dict[str, Any]]:
    unique_labels = sorted(int(value) for value in np.unique(np.asarray(labels, dtype=np.int64)))
    capacity = 2 ** int(ancilla_bits)
    if len(unique_labels) > capacity:
        raise ValueError(
            f"{len(unique_labels)} labels require more than {ancilla_bits} ancilla bits "
            f"(capacity={capacity})"
        )
    return [
        {"branch_index": index, "branch_bits": branch_bits_for_index(index, int(ancilla_bits)), "label": label}
        for index, label in enumerate(unique_labels)
    ]


def label_conditioned_mmd_loss(
    params: Any,
    key: Any,
    iqp_circuit: Any,
    label_ground_truths: tuple[Any, ...],
    label_condition_masks: tuple[Any, ...],
    label_condition_values: tuple[Any, ...],
    sigma: float | list[float],
    n_ops: int,
    n_samples: int,
    wires: list[int] | None = None,
    init_coefs: Any = None,
    indep_estimates: bool = False,
    jit: bool = False,
    sqrt_loss: bool = False,
    return_expvals: bool = False,
    max_batch_ops: int | None = None,
    max_batch_samples: int | None = None,
) -> Any:
    import jax
    import iqp_fast.gen_qml as gen

    if not label_ground_truths:
        raise ValueError("label-conditioned loss requires at least one label branch")

    losses = []
    for ground_truth, condition_mask, condition_values in zip(
        label_ground_truths,
        label_condition_masks,
        label_condition_values,
    ):
        key, subkey = jax.random.split(key, 2)
        losses.append(
            gen.mmd_loss_iqp(
                params=params,
                iqp_circuit=iqp_circuit,
                ground_truth=ground_truth,
                sigma=sigma,
                n_ops=n_ops,
                n_samples=n_samples,
                key=subkey,
                init_coefs=init_coefs,
                wires=wires,
                indep_estimates=indep_estimates,
                jit=jit,
                sqrt_loss=sqrt_loss,
                return_expvals=return_expvals,
                max_batch_ops=max_batch_ops,
                max_batch_samples=max_batch_samples,
                condition_mask=condition_mask,
                condition_values=condition_values,
            )
        )
    return sum(losses) / len(losses)


def _data_term(ground_truth: Any, visible_ops: Any) -> Any:
    """jnp.mean(1 - 2 * ((ground_truth @ visible_ops.T) % 2), axis=0), on the GPU kernel when available."""
    from iqp_fast._fastpath import data_term

    return data_term(ground_truth, visible_ops)


def _branch_loss_kernels(params, key, iqp_circuit, ground_truth, branch_condition_masks, branch_condition_values,
                         branch_sample_counts, sigmas, n_ops, n_samples, wires, init_coefs, indep_estimates, sqrt_loss,
                         return_expvals, max_batch_ops, max_batch_samples):
    """branch_weighted_mixture_mmd_loss with the expectation values of all bandwidths from one evaluation of the
    iqp_fast kernel path, or None when that path does not apply (then the loop below runs). The jax.random calls,
    their keys and their order are those of the loop: per bandwidth the ops, then the samples."""
    import jax
    import jax.numpy as jnp

    route = getattr(iqp_circuit, "_fast_route", None)
    if route is None or max_batch_ops is not None or max_batch_samples is not None or iqp_circuit.bitflip:
        return None
    active_branches = [(mask, values, int(count)) for mask, values, count in
                       zip(branch_condition_masks, branch_condition_values, branch_sample_counts) if int(count) > 0]
    if not active_branches:
        return None
    try:
        condition_masks = np.stack([np.asarray(mask, dtype=bool) for mask, _, _ in active_branches], axis=0)
        condition_values = np.stack([np.asarray(values, dtype=np.int32) for _, values, _ in active_branches], axis=0)
    except jax.errors.TracerArrayConversionError:
        return None
    condition_counts = tuple(int(count) for _, _, count in active_branches)
    if int(sum(condition_counts)) != int(n_samples):
        return None
    plan = route(int(n_ops), int(n_samples), indep_estimates, condition_masks, condition_values, condition_counts,
                 tuple(wires))
    if plan is None:
        return None
    from iqp_fast import _fastpath

    split, segments = plan
    vis = _fastpath.tables(iqp_circuit, split)["vis"]
    if len(wires) != len(vis) or [int(x) for x in sorted(set(wires))] != [int(x) for x in vis]:
        return None
    blocks, data_ops = [], []
    for sigma_value in sigmas:
        p_mmd = (1 - jnp.exp(-1 / 2 / float(sigma_value) ** 2)) / 2
        key, subkey = jax.random.split(key, 2)
        visible_ops = jnp.array(
            jax.random.binomial(subkey, 1, p_mmd, shape=(n_ops, len(wires))),
            dtype=float,
        )
        # the loop's all_ops holds visible_ops at the sorted wires, i.e. column j at visible wire j
        key, samples_key = jax.random.split(key, 2)
        samples, _ = _fastpath._draw_visible(iqp_circuit, samples_key, int(n_samples), None, condition_masks,
                                             condition_values, condition_counts, vis)
        blocks.append((visible_ops, samples))
        data_ops.append(visible_ops)
    # per op the sums over the samples of the expectation-value samples and of their squares
    sums = _fastpath.expval_blocks(iqp_circuit, split, segments, params, init_coefs, blocks, sums=True, visible=True)
    losses = []
    for visible_ops, sm in zip(data_ops, sums):
        tr_iqp_sum = sm[:, 0]
        tr_iqp_sq_sum = sm[:, 1]
        tr_iqp = tr_iqp_sum / int(n_samples)
        correction = tr_iqp_sq_sum / (int(n_samples) * int(n_samples))
        tr_train = _data_term(ground_truth, visible_ops)
        train_count = len(ground_truth)
        res = (
            (tr_iqp * tr_iqp - correction) * int(n_samples) / (int(n_samples) - 1)
            - 2 * tr_iqp * tr_train
            + (tr_train * tr_train * train_count - 1) / (train_count - 1)
        )
        res = res if return_expvals else jnp.mean(res)
        res = jnp.sqrt(jnp.abs(res)) if sqrt_loss else res
        losses.append(res)
    if return_expvals:
        return losses
    return sum(losses) / len(losses)


def branch_weighted_mixture_mmd_loss(
    params: Any,
    key: Any,
    iqp_circuit: Any,
    ground_truth: Any,
    branch_condition_masks: tuple[Any, ...],
    branch_condition_values: tuple[Any, ...],
    branch_sample_counts: tuple[int, ...],
    sigma: float | list[float],
    n_ops: int,
    n_samples: int,
    wires: list[int] | None = None,
    init_coefs: Any = None,
    indep_estimates: bool = False,
    jit: bool = False,
    sqrt_loss: bool = False,
    return_expvals: bool = False,
    max_batch_ops: int | None = None,
    max_batch_samples: int | None = None,
) -> Any:
    import jax
    import jax.numpy as jnp

    if not branch_condition_masks:
        raise ValueError("branch-weighted loss requires at least one active branch")
    if len(branch_condition_masks) != len(branch_condition_values) or len(branch_condition_masks) != len(branch_sample_counts):
        raise ValueError("branch condition masks, values, and sample counts must have the same length")
    total_branch_samples = int(sum(int(count) for count in branch_sample_counts))
    if total_branch_samples <= 1:
        raise ValueError("branch-weighted loss needs at least two total branch samples")
    if int(n_samples) != total_branch_samples:
        n_samples = total_branch_samples
    if wires is None:
        wires = list(range(iqp_circuit.n_qubits))

    sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
    init_coefs = jnp.array(init_coefs) if init_coefs is not None else None
    fast = _branch_loss_kernels(params, key, iqp_circuit, ground_truth, branch_condition_masks,
                                branch_condition_values, branch_sample_counts, sigmas, n_ops, n_samples, wires,
                                init_coefs, indep_estimates, sqrt_loss, return_expvals, max_batch_ops, max_batch_samples)
    if fast is not None:
        return fast
    losses = []
    for sigma_value in sigmas:
        p_mmd = (1 - jnp.exp(-1 / 2 / float(sigma_value) ** 2)) / 2
        key, subkey = jax.random.split(key, 2)
        visible_ops = jnp.array(
            jax.random.binomial(subkey, 1, p_mmd, shape=(n_ops, len(wires))),
            dtype=float,
        )
        # Vectorized placement (a python loop over qubits unrolls into a huge
        # graph with pathological float32 XLA codegen); same column order.
        all_ops = jnp.zeros((n_ops, iqp_circuit.n_qubits), dtype=float)
        all_ops = all_ops.at[:, jnp.asarray(sorted(set(wires)))].set(visible_ops)

        active_branches = [
            (mask, values, int(count))
            for mask, values, count in zip(
                branch_condition_masks,
                branch_condition_values,
                branch_sample_counts,
            )
            if int(count) > 0
        ]
        if not active_branches:
            raise ValueError("branch-weighted loss requires at least one branch with positive samples")
        try:
            # numpy keeps the branch layout static, which op_expval's fast pairwise
            # path needs; traced masks fall back to jnp
            condition_masks = np.stack([np.asarray(mask, dtype=bool) for mask, _, _ in active_branches], axis=0)
            condition_values = np.stack([np.asarray(values, dtype=np.int32) for _, values, _ in active_branches], axis=0)
        except jax.errors.TracerArrayConversionError:
            condition_masks = jnp.stack([jnp.asarray(mask, dtype=bool) for mask, _, _ in active_branches], axis=0)
            condition_values = jnp.stack(
                [jnp.asarray(values, dtype=jnp.int32) for _, values, _ in active_branches],
                axis=0,
            )
        branch_counts = [count for _, _, count in active_branches]
        if int(sum(branch_counts)) != int(n_samples):
            raise ValueError("sum(condition_counts) must equal n_samples")
        # Keep per-branch sample counts as a concrete python tuple (static):
        # op_expval's sample-batch allocation needs their concrete values, and a
        # jnp array can become a tracer under jax.grad / lax.scan. op_expval_batch
        # converts to jnp internally.
        condition_counts = tuple(int(c) for c in branch_counts)

        key, samples_key = jax.random.split(key, 2)
        branch_samples = iqp_circuit.op_expval(
            params,
            all_ops,
            int(n_samples),
            samples_key,
            init_coefs,
            indep_estimates,
            return_samples=True,
            max_batch_ops=max_batch_ops,
            max_batch_samples=max_batch_samples,
            condition_mask=condition_masks,
            condition_values=condition_values,
            condition_counts=condition_counts,
            ops_wires=tuple(wires),
        )
        tr_iqp_sum = jnp.sum(branch_samples, axis=-1)
        tr_iqp_sq_sum = jnp.sum(branch_samples * branch_samples, axis=-1)
        tr_iqp = tr_iqp_sum / int(n_samples)
        correction = tr_iqp_sq_sum / (int(n_samples) * int(n_samples))
        tr_train = _data_term(ground_truth, visible_ops)
        train_count = len(ground_truth)

        if iqp_circuit.bitflip:
            res = tr_iqp * tr_iqp - 2 * tr_iqp * tr_train + (tr_train * tr_train * train_count - 1) / (train_count - 1)
        else:
            res = (
                (tr_iqp * tr_iqp - correction) * int(n_samples) / (int(n_samples) - 1)
                - 2 * tr_iqp * tr_train
                + (tr_train * tr_train * train_count - 1) / (train_count - 1)
            )

        res = res if return_expvals else jnp.mean(res)
        res = jnp.sqrt(jnp.abs(res)) if sqrt_loss else res
        losses.append(res)

    if return_expvals:
        return losses
    return sum(losses) / len(losses)


def branch_separated_mmd_loss(
    params: Any,
    key: Any,
    iqp_circuit: Any,
    ground_truth: Any,
    sigma: float | list[float],
    n_ops: int,
    n_samples: int,
    branch_condition_masks: tuple[Any, ...],
    branch_condition_values: tuple[Any, ...],
    branch_regularization: dict[str, Any],
    wires: list[int] | None = None,
    init_coefs: Any = None,
    indep_estimates: bool = False,
    jit: bool = False,
    sqrt_loss: bool = False,
    return_expvals: bool = False,
    max_batch_ops: int | None = None,
    max_batch_samples: int | None = None,
    training_step: Any | None = None,
    training_total_steps: int | None = None,
) -> Any:
    import jax
    import jax.numpy as jnp
    import iqp_fast.gen_qml as gen

    base_loss = gen.mmd_loss_iqp(
        params=params,
        iqp_circuit=iqp_circuit,
        ground_truth=ground_truth,
        sigma=sigma,
        n_ops=n_ops,
        n_samples=n_samples,
        key=key,
        init_coefs=init_coefs,
        wires=wires,
        indep_estimates=indep_estimates,
        jit=jit,
        sqrt_loss=sqrt_loss,
        return_expvals=return_expvals,
        max_batch_ops=max_batch_ops,
        max_batch_samples=max_batch_samples,
    )
    if return_expvals:
        return base_loss

    if float(branch_regularization.get("weight", 0.0)) <= 0 or not branch_condition_masks:
        return base_loss
    weight = branch_regularization_weight(
        branch_regularization,
        training_step=training_step,
        training_total_steps=training_total_steps,
    )

    if wires is None:
        wires = list(range(iqp_circuit.n_qubits))
    branch_count = len(branch_condition_masks)
    if branch_count < 2:
        return base_loss

    branch_samples_per_branch = int(branch_regularization.get("num_samples", 256))
    if branch_samples_per_branch <= 1:
        raise ValueError("qcbm.branch_regularization.num_samples must be greater than 1")
    total_branch_samples = branch_count * branch_samples_per_branch
    branch_counts = jnp.full((branch_count,), branch_samples_per_branch, dtype=jnp.int32)

    # one single-Z op per visible wire (vectorized; see branch_weighted_mixture_mmd_loss)
    all_ops = jnp.zeros((len(wires), iqp_circuit.n_qubits), dtype=float)
    all_ops = all_ops.at[:, jnp.asarray(sorted(set(int(w) for w in wires)))].set(jnp.eye(len(wires), dtype=float))

    condition_masks = jnp.stack([jnp.asarray(mask, dtype=bool) for mask in branch_condition_masks], axis=0)
    condition_values = jnp.stack(
        [jnp.asarray(values, dtype=jnp.int32) for values in branch_condition_values],
        axis=0,
    )
    key, branch_key = jax.random.split(key, 2)
    branch_max_batch_ops = branch_regularization.get("max_batch_ops", max_batch_ops)
    branch_max_batch_samples = branch_regularization.get("max_batch_samples", None)
    z_samples = iqp_circuit.op_expval(
        params,
        all_ops,
        int(total_branch_samples),
        branch_key,
        init_coefs,
        indep_estimates,
        return_samples=True,
        max_batch_ops=branch_max_batch_ops,
        max_batch_samples=branch_max_batch_samples,
        condition_mask=condition_masks,
        condition_values=condition_values,
        condition_counts=branch_counts,
        ops_wires=tuple(wires),
    )
    z_samples = jnp.reshape(
        z_samples,
        (len(wires), branch_count, branch_samples_per_branch),
    )
    z_samples = jnp.transpose(z_samples, (1, 2, 0))

    sigmas = branch_regularization.get("sigmas")
    if sigmas is None:
        sigmas = [float(s) for s in branch_regularization.get("z_sigmas", [2.8, 5.6, 11.2])]
    elif isinstance(sigmas, (int, float)):
        sigmas = [float(sigmas)]
    else:
        sigmas = [float(value) for value in sigmas]
    if not sigmas:
        raise ValueError("qcbm.branch_regularization.sigmas must not be empty")

    def biased_mmd2(x: Any, y: Any, sigma_value: float) -> Any:
        x2 = jnp.sum(x * x, axis=-1)
        y2 = jnp.sum(y * y, axis=-1)
        dxx = jnp.maximum(x2[:, None] + x2[None, :] - 2.0 * (x @ x.T), 0.0)
        dyy = jnp.maximum(y2[:, None] + y2[None, :] - 2.0 * (y @ y.T), 0.0)
        dxy = jnp.maximum(x2[:, None] + y2[None, :] - 2.0 * (x @ y.T), 0.0)
        inv = 1.0 / (2.0 * float(sigma_value) ** 2)
        return jnp.mean(jnp.exp(-dxx * inv)) + jnp.mean(jnp.exp(-dyy * inv)) - 2.0 * jnp.mean(jnp.exp(-dxy * inv))

    pair_values = []
    for i in range(branch_count):
        for j in range(i + 1, branch_count):
            per_sigma = [biased_mmd2(z_samples[i], z_samples[j], sigma_value) for sigma_value in sigmas]
            pair_values.append(sum(per_sigma) / len(per_sigma))
    pair_values = jnp.asarray(pair_values, dtype=jnp.float64)

    mode = str(branch_regularization.get("mode", "exp_topk")).lower()
    if mode == "negative_mmd":
        branch_penalty = -jnp.mean(pair_values)
    else:
        topk = int(branch_regularization.get("topk_pairs", min(4, int(pair_values.shape[0]))))
        topk = max(1, min(topk, int(pair_values.shape[0])))
        tau = float(branch_regularization.get("tau", 0.05))
        if tau <= 0:
            raise ValueError("qcbm.branch_regularization.tau must be positive")
        closest = jnp.sort(pair_values)[:topk]
        branch_penalty = jnp.mean(jnp.exp(-closest / tau))

    return base_loss + weight * branch_penalty


def branch_z_separation_penalty(
    params: Any,
    key: Any,
    iqp_circuit: Any,
    branch_condition_masks: tuple[Any, ...],
    branch_condition_values: tuple[Any, ...],
    branch_regularization: dict[str, Any],
    wires: list[int] | None = None,
    init_coefs: Any = None,
    indep_estimates: bool = False,
    max_batch_ops: int | None = None,
    max_batch_samples: int | None = None,
    training_step: Any | None = None,
    training_total_steps: int | None = None,
) -> Any:
    import jax
    import jax.numpy as jnp

    if float(branch_regularization.get("weight", 0.0)) <= 0 or not branch_condition_masks:
        return 0.0
    weight = branch_regularization_weight(
        branch_regularization,
        training_step=training_step,
        training_total_steps=training_total_steps,
    )
    if wires is None:
        wires = list(range(iqp_circuit.n_qubits))
    branch_count = len(branch_condition_masks)
    if branch_count < 2:
        return 0.0

    branch_samples_per_branch = int(branch_regularization.get("num_samples", 256))
    if branch_samples_per_branch <= 1:
        raise ValueError("qcbm.branch_regularization.num_samples must be greater than 1")
    total_branch_samples = branch_count * branch_samples_per_branch
    branch_counts = jnp.full((branch_count,), branch_samples_per_branch, dtype=jnp.int32)

    # one single-Z op per visible wire (vectorized; see branch_weighted_mixture_mmd_loss)
    all_ops = jnp.zeros((len(wires), iqp_circuit.n_qubits), dtype=float)
    all_ops = all_ops.at[:, jnp.asarray(sorted(set(int(w) for w in wires)))].set(jnp.eye(len(wires), dtype=float))

    condition_masks = jnp.stack([jnp.asarray(mask, dtype=bool) for mask in branch_condition_masks], axis=0)
    condition_values = jnp.stack(
        [jnp.asarray(values, dtype=jnp.int32) for values in branch_condition_values],
        axis=0,
    )
    branch_max_batch_ops = branch_regularization.get("max_batch_ops", max_batch_ops)
    branch_max_batch_samples = branch_regularization.get("max_batch_samples", None)
    z_samples = iqp_circuit.op_expval(
        params,
        all_ops,
        int(total_branch_samples),
        key,
        init_coefs,
        indep_estimates,
        return_samples=True,
        max_batch_ops=branch_max_batch_ops,
        max_batch_samples=branch_max_batch_samples,
        condition_mask=condition_masks,
        condition_values=condition_values,
        condition_counts=branch_counts,
        ops_wires=tuple(wires),
    )
    z_samples = jnp.reshape(z_samples, (len(wires), branch_count, branch_samples_per_branch))
    z_samples = jnp.transpose(z_samples, (1, 2, 0))

    sigmas = branch_regularization.get("sigmas")
    if sigmas is None:
        sigmas = [float(s) for s in branch_regularization.get("z_sigmas", [2.8, 5.6, 11.2])]
    elif isinstance(sigmas, (int, float)):
        sigmas = [float(sigmas)]
    else:
        sigmas = [float(value) for value in sigmas]
    if not sigmas:
        raise ValueError("qcbm.branch_regularization.sigmas must not be empty")

    def biased_mmd2(x: Any, y: Any, sigma_value: float) -> Any:
        x2 = jnp.sum(x * x, axis=-1)
        y2 = jnp.sum(y * y, axis=-1)
        dxx = jnp.maximum(x2[:, None] + x2[None, :] - 2.0 * (x @ x.T), 0.0)
        dyy = jnp.maximum(y2[:, None] + y2[None, :] - 2.0 * (y @ y.T), 0.0)
        dxy = jnp.maximum(x2[:, None] + y2[None, :] - 2.0 * (x @ y.T), 0.0)
        inv = 1.0 / (2.0 * float(sigma_value) ** 2)
        return jnp.mean(jnp.exp(-dxx * inv)) + jnp.mean(jnp.exp(-dyy * inv)) - 2.0 * jnp.mean(jnp.exp(-dxy * inv))

    pair_values = []
    for i in range(branch_count):
        for j in range(i + 1, branch_count):
            per_sigma = [biased_mmd2(z_samples[i], z_samples[j], sigma_value) for sigma_value in sigmas]
            pair_values.append(sum(per_sigma) / len(per_sigma))
    pair_values = jnp.asarray(pair_values, dtype=jnp.float64)

    mode = str(branch_regularization.get("mode", "exp_topk")).lower()
    if mode == "negative_mmd":
        return -jnp.mean(pair_values)
    topk = int(branch_regularization.get("topk_pairs", min(4, int(pair_values.shape[0]))))
    topk = max(1, min(topk, int(pair_values.shape[0])))
    tau = float(branch_regularization.get("tau", 0.05))
    if tau <= 0:
        raise ValueError("qcbm.branch_regularization.tau must be positive")
    closest = jnp.sort(pair_values)[:topk]
    return jnp.mean(jnp.exp(-closest / tau))


def branch_regularization_weight(
    branch_regularization: dict[str, Any],
    training_step: Any | None = None,
    training_total_steps: int | None = None,
) -> Any:
    import jax.numpy as jnp

    base_weight = float(branch_regularization.get("weight", 0.0))
    schedule = branch_regularization.get("schedule") or branch_regularization.get("weight_schedule")
    if not schedule or not bool(schedule.get("enabled", True)):
        return base_weight

    start_weight = float(schedule.get("start_weight", schedule.get("initial_weight", base_weight)))
    end_weight = float(schedule.get("end_weight", schedule.get("final_weight", 0.0)))
    if training_step is None:
        training_step = 0
    total_steps = schedule.get("total_steps", training_total_steps)
    if total_steps is None:
        total_steps = 1
    total_steps = max(1, int(total_steps))

    step = jnp.asarray(training_step, dtype=jnp.float64)
    denom = float(max(total_steps - 1, 1))
    progress = jnp.clip(step / denom, 0.0, 1.0)
    schedule_type = str(schedule.get("type", "power")).lower()

    if schedule_type in {"power", "polynomial", "poly"}:
        power = float(schedule.get("power", 2.0))
        if power <= 0:
            raise ValueError("qcbm.branch_regularization.schedule.power must be positive")
        factor = (1.0 - progress) ** power
    elif schedule_type in {"exp", "exponential"}:
        rate = float(schedule.get("rate", 5.0))
        if rate <= 0:
            raise ValueError("qcbm.branch_regularization.schedule.rate must be positive")
        factor = jnp.exp(-rate * progress)
        if bool(schedule.get("normalize_end", True)):
            end_factor = jnp.exp(-rate)
            factor = (factor - end_factor) / (1.0 - end_factor)
            factor = jnp.clip(factor, 0.0, 1.0)
    elif schedule_type == "linear":
        factor = 1.0 - progress
    else:
        raise ValueError(f"Unsupported branch regularization schedule type: {schedule_type}")

    return end_weight + (start_weight - end_weight) * factor


def branch_separated_mmd_components(
    params: Any,
    key: Any,
    iqp_circuit: Any,
    ground_truth: Any,
    sigma: float | list[float],
    n_ops: int,
    n_samples: int,
    branch_condition_masks: tuple[Any, ...],
    branch_condition_values: tuple[Any, ...],
    branch_regularization: dict[str, Any],
    wires: list[int] | None = None,
    init_coefs: Any = None,
    indep_estimates: bool = False,
    jit: bool = False,
    sqrt_loss: bool = False,
    max_batch_ops: int | None = None,
    max_batch_samples: int | None = None,
    training_step: Any | None = None,
    training_total_steps: int | None = None,
) -> dict[str, float]:
    import jax
    import iqp_fast.gen_qml as gen

    key_data, key_branch = jax.random.split(key, 2)
    data_mmd = gen.mmd_loss_iqp(
        params=params,
        iqp_circuit=iqp_circuit,
        ground_truth=ground_truth,
        sigma=sigma,
        n_ops=n_ops,
        n_samples=n_samples,
        key=key_data,
        init_coefs=init_coefs,
        wires=wires,
        indep_estimates=indep_estimates,
        jit=jit,
        sqrt_loss=sqrt_loss,
        return_expvals=False,
        max_batch_ops=max_batch_ops,
        max_batch_samples=max_batch_samples,
    )
    branch_term = branch_z_separation_penalty(
        params=params,
        key=key_branch,
        iqp_circuit=iqp_circuit,
        branch_condition_masks=branch_condition_masks,
        branch_condition_values=branch_condition_values,
        branch_regularization=branch_regularization,
        wires=wires,
        init_coefs=init_coefs,
        indep_estimates=indep_estimates,
        max_batch_ops=max_batch_ops,
        max_batch_samples=max_batch_samples,
        training_step=training_step,
        training_total_steps=training_total_steps,
    )
    weight = branch_regularization_weight(
        branch_regularization,
        training_step=training_step,
        training_total_steps=training_total_steps,
    )
    total = data_mmd + weight * branch_term
    return {"data_mmd": data_mmd, "branch_term": branch_term, "branch_weight": weight, "total_check": total}


class RandomizedIqpTrainer:
    def __init__(self, experiment: ExperimentConfig) -> None:
        import os
        import jax

        # Default float32; set IQPOPT_X64=1 to force float64 (see iqp_optimizer.py).
        if os.environ.get("IQPOPT_X64", "0") in ("1", "true", "True"):
            jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = __import__("jax.numpy", fromlist=[""])
        self.experiment = experiment
        self.config = experiment.raw
        self.qcbm = self.config["qcbm"]
        self.visible_bits = int(self.config["data"]["visible_bits"])
        self.seed = int(self.config.get("seed", 1234))
        self.train_bits_np = load_binary_csv(experiment.train_data)
        if self.train_bits_np.shape[-1] != self.visible_bits:
            raise ValueError(f"train data has {self.train_bits_np.shape[-1]} bits, expected {self.visible_bits}")
        self.train_bits = self.jnp.asarray(self.train_bits_np, dtype=self.jnp.int32)

    def checkpoint_dir(self, layout_name: str, ancilla_bits: int) -> Path:
        return self.experiment.outputs_dir / "checkpoints" / layout_name / f"a{ancilla_bits}"

    def train_layout(
        self,
        layout_name: str,
        max_ancilla_bits: int,
        stage_steps: int | None = None,
        start_ancilla_bits: int = 0,
        direct_max_ancilla: bool = False,
    ) -> dict[str, Any]:
        if start_ancilla_bits < 0 or start_ancilla_bits > max_ancilla_bits:
            raise ValueError(f"start_ancilla_bits={start_ancilla_bits} is outside [0, {max_ancilla_bits}]")

        training_mode = str(self.qcbm.get("training_mode", "")).strip().lower()
        label_conditioned_ancilla = training_mode == "label_conditioned_ancilla"
        cluster_weighted_ancilla = training_mode == "cluster_weighted_ancilla"
        direct_max_ancilla = bool(direct_max_ancilla or training_mode == "direct_max_ancilla")
        enabled_special_modes = [label_conditioned_ancilla, cluster_weighted_ancilla, direct_max_ancilla]
        if sum(bool(value) for value in enabled_special_modes) > 1:
            raise ValueError("Only one ancilla training mode can be active")

        artifacts: dict[str, Any] = {}
        if label_conditioned_ancilla:
            source_path: Path | None = None
            previous_gates_for_stage = None
            previous_params_for_stage = None
            if self.qcbm.get("init_from_artifact") in (None, False):
                source_path, source_artifact = self._load_label_condition_source_artifact(layout_name)
                previous_gates_for_stage = source_artifact["gates"]
                previous_params_for_stage = torch.as_tensor(source_artifact["params"], dtype=torch.float64).numpy()
            artifact = self.train_stage(
                layout_name=layout_name,
                ancilla_bits=int(max_ancilla_bits),
                previous_gates=previous_gates_for_stage,
                previous_params=previous_params_for_stage,
                stage_steps=stage_steps,
                label_condition_source=str(source_path) if source_path is not None else None,
            )
            artifacts[f"a{max_ancilla_bits}"] = {
                "artifact": str(self.checkpoint_dir(layout_name, max_ancilla_bits) / "qcbm_artifact.pt"),
                "final_loss": float(torch.as_tensor(artifact["losses"])[-1].item()) if len(artifact["losses"]) else math.nan,
                "training_mode": "label_conditioned_ancilla",
                "source_artifact": str(source_path) if source_path is not None else None,
                "init_from_artifact": artifact.get("init_from_artifact"),
            }
            return artifacts

        if cluster_weighted_ancilla:
            if max_ancilla_bits <= 0:
                raise ValueError("cluster_weighted_ancilla requires ancilla_bits > 0")
            artifact = self.train_stage(
                layout_name=layout_name,
                ancilla_bits=int(max_ancilla_bits),
                previous_gates=None,
                previous_params=None,
                stage_steps=stage_steps,
            )
            artifacts[f"a{max_ancilla_bits}"] = {
                "artifact": str(self.checkpoint_dir(layout_name, max_ancilla_bits) / "qcbm_artifact.pt"),
                "final_loss": float(torch.as_tensor(artifact["losses"])[-1].item()) if len(artifact["losses"]) else math.nan,
                "training_mode": "cluster_weighted_ancilla",
                "init_from_artifact": artifact.get("init_from_artifact"),
            }
            return artifacts

        if direct_max_ancilla:
            artifact = self.train_stage(
                layout_name=layout_name,
                ancilla_bits=int(max_ancilla_bits),
                previous_gates=None,
                previous_params=None,
                stage_steps=stage_steps,
            )
            artifacts[f"a{max_ancilla_bits}"] = {
                "artifact": str(self.checkpoint_dir(layout_name, max_ancilla_bits) / "qcbm_artifact.pt"),
                "final_loss": float(torch.as_tensor(artifact["losses"])[-1].item()) if len(artifact["losses"]) else math.nan,
                "training_mode": "direct_max_ancilla",
            }
            return artifacts

        previous_artifact: dict[str, Any] | None = None
        previous_params: np.ndarray | None = None
        previous_gates: list[list[list[int]]] | None = None
        if start_ancilla_bits > 0:
            previous_artifact = self._load_stage_artifact(layout_name, start_ancilla_bits - 1)
            previous_gates = previous_artifact["gates"]
            previous_params = torch.as_tensor(previous_artifact["params"], dtype=torch.float64).numpy()
            previous_losses = torch.as_tensor(previous_artifact.get("losses", []), dtype=torch.float64)
            artifacts[f"a{start_ancilla_bits - 1}"] = {
                "artifact": str(self.checkpoint_dir(layout_name, start_ancilla_bits - 1) / "qcbm_artifact.pt"),
                "loaded": True,
                "final_loss": float(previous_losses[-1].item()) if len(previous_losses) else math.nan,
            }

        for ancilla_bits in range(start_ancilla_bits, max_ancilla_bits + 1):
            artifact = self.train_stage(
                layout_name=layout_name,
                ancilla_bits=ancilla_bits,
                previous_gates=previous_gates,
                previous_params=previous_params,
                stage_steps=stage_steps,
            )
            artifacts[f"a{ancilla_bits}"] = {
                "artifact": str(self.checkpoint_dir(layout_name, ancilla_bits) / "qcbm_artifact.pt"),
                "final_loss": float(torch.as_tensor(artifact["losses"])[-1].item()) if len(artifact["losses"]) else math.nan,
            }
            previous_artifact = artifact
            previous_gates = previous_artifact["gates"]
            previous_params = torch.as_tensor(previous_artifact["params"], dtype=torch.float64).numpy()
        return artifacts

    def _load_stage_artifact(self, layout_name: str, ancilla_bits: int) -> dict[str, Any]:
        artifact_path = self.checkpoint_dir(layout_name, ancilla_bits) / "qcbm_artifact.pt"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Cannot resume from missing artifact: {artifact_path}")
        return torch.load(artifact_path, map_location="cpu")


    def _use_label_conditioned_training(self) -> bool:
        return str(self.qcbm.get("training_mode", "")).strip().lower() == "label_conditioned_ancilla"

    def _use_cluster_weighted_training(self) -> bool:
        return str(self.qcbm.get("training_mode", "")).strip().lower() == "cluster_weighted_ancilla"

    def _label_condition_config(self) -> dict[str, Any]:
        payload = self.qcbm.get("label_condition", {})
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError("qcbm.label_condition must be a mapping")
        return payload

    def _ensure_iqpopt_vendor(self) -> None:
        vendor_raw = self.qcbm.get("iqpopt_vendor_path")
        if vendor_raw is None:
            vendor_raw = self.config.get("vendor", {}).get("iqpopt_path")
        if vendor_raw is None:
            return
        vendor_path = self.experiment.resolve(vendor_raw)
        if not vendor_path.exists():
            raise FileNotFoundError(f"Configured iqp_fast vendor path does not exist: {vendor_path}")
        vendor_str = str(vendor_path)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)

        loaded = sys.modules.get("iqp_fast")
        if loaded is None:
            return
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is None:
            return
        loaded_path = Path(loaded_file).resolve()
        if vendor_path == loaded_path or vendor_path in loaded_path.parents:
            return
        for module_name in list(sys.modules):
            if module_name == "iqp_fast" or module_name.startswith("iqp_fast."):
                del sys.modules[module_name]

    def _resolve_label_path(self) -> Path:
        condition_cfg = self._label_condition_config()
        if condition_cfg.get("labels_path") is not None:
            label_path = self.experiment.resolve(condition_cfg["labels_path"])
            if not label_path.exists():
                raise FileNotFoundError(f"Label file does not exist: {label_path}")
            return label_path

        train_path = self.experiment.train_data
        candidates = []
        if train_path.name.startswith("x_"):
            candidates.append(train_path.with_name("y_" + train_path.name[2:]))
        candidates.extend([train_path.with_name("y_train.csv"), train_path.with_name("y.csv")])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not infer label file. Set qcbm.label_condition.labels_path "
            f"for train_data={train_path}"
        )

    def _load_label_vector(self) -> tuple[Path, np.ndarray]:
        label_path = self._resolve_label_path()
        labels = np.loadtxt(label_path, delimiter=",", dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(labels) != len(self.train_bits_np):
            raise ValueError(
                f"Label count {len(labels)} from {label_path} does not match "
                f"train sample count {len(self.train_bits_np)}"
            )
        return label_path, labels

    def _load_label_condition_source_artifact(self, layout_name: str) -> tuple[Path, dict[str, Any]]:
        condition_cfg = self._label_condition_config()
        explicit = condition_cfg.get("source_a0_artifact") or condition_cfg.get("source_artifact")
        if explicit is not None:
            artifact_path = self.experiment.resolve(explicit)
        else:
            source_outputs = condition_cfg.get("source_outputs_dir")
            if source_outputs is None:
                raise ValueError(
                    "label_conditioned_ancilla requires qcbm.label_condition.source_outputs_dir "
                    "or source_a0_artifact"
                )
            source_ancilla = int(condition_cfg.get("source_ancilla_bits", 0))
            artifact_path = (
                self.experiment.resolve(source_outputs)
                / "checkpoints"
                / layout_name
                / f"a{source_ancilla}"
                / "qcbm_artifact.pt"
            )
        if not artifact_path.exists():
            raise FileNotFoundError(f"Source visible artifact does not exist: {artifact_path}")
        artifact = torch.load(artifact_path, map_location="cpu")
        return artifact_path, artifact

    def _load_init_artifact(self, layout_name: str) -> tuple[Path, dict[str, Any]] | None:
        init_cfg = self.qcbm.get("init_from_artifact")
        if init_cfg in (None, False):
            return None
        if isinstance(init_cfg, str):
            artifact_path = self.experiment.resolve(init_cfg)
        elif isinstance(init_cfg, dict):
            explicit = init_cfg.get("artifact") or init_cfg.get("path")
            if explicit is not None:
                artifact_path = self.experiment.resolve(explicit)
            else:
                source_outputs = init_cfg.get("outputs_dir")
                if source_outputs is None:
                    raise ValueError("qcbm.init_from_artifact requires artifact/path or outputs_dir")
                source_ancilla = int(init_cfg.get("ancilla_bits", 0))
                artifact_path = (
                    self.experiment.resolve(source_outputs)
                    / "checkpoints"
                    / layout_name
                    / f"a{source_ancilla}"
                    / "qcbm_artifact.pt"
                )
        else:
            raise ValueError("qcbm.init_from_artifact must be a string path or mapping")
        if not artifact_path.exists():
            raise FileNotFoundError(f"Initial parameter artifact does not exist: {artifact_path}")
        return artifact_path, torch.load(artifact_path, map_location="cpu")

    def _build_label_condition_payload(
        self,
        ancilla_bits: int,
        n_qubits: int,
        ancilla_wires: list[int],
    ) -> dict[str, Any]:
        if ancilla_bits <= 0:
            raise ValueError("label-conditioned training requires ancilla_bits > 0")
        if len(ancilla_wires) != ancilla_bits:
            raise ValueError(f"Expected {ancilla_bits} ancilla wires, got {ancilla_wires}")

        label_path, labels = self._load_label_vector()
        mapping = label_branch_mapping(labels, ancilla_bits)
        ground_truths = []
        condition_masks = []
        condition_values = []
        rows = []
        for entry in mapping:
            label = int(entry["label"])
            selected = labels == label
            if not np.any(selected):
                raise ValueError(f"No training samples for label {label}")
            mask = np.zeros(n_qubits, dtype=bool)
            values = np.zeros(n_qubits, dtype=np.int32)
            for wire, bit in zip(ancilla_wires, str(entry["branch_bits"])):
                mask[int(wire)] = True
                values[int(wire)] = int(bit)
            ground_truths.append(self.jnp.asarray(self.train_bits_np[selected], dtype=self.jnp.int32))
            condition_masks.append(self.jnp.asarray(mask, dtype=bool))
            condition_values.append(self.jnp.asarray(values, dtype=self.jnp.int32))
            rows.append(
                {
                    "label": label,
                    "branch_index": int(entry["branch_index"]),
                    "branch_bits": str(entry["branch_bits"]),
                    "sample_count": int(np.sum(selected)),
                }
            )
        return {
            "labels_path": str(label_path),
            "mapping": rows,
            "ground_truths": tuple(ground_truths),
            "condition_masks": tuple(condition_masks),
            "condition_values": tuple(condition_values),
        }

    @staticmethod
    def _allocate_branch_sample_counts(weights: np.ndarray, total_samples: int) -> list[int]:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or len(weights) == 0:
            raise ValueError("branch weights must be a non-empty vector")
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError(f"Invalid branch weights: {weights}")
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            raise ValueError("At least one branch weight must be positive")
        weights = weights / weight_sum
        total_samples = int(total_samples)
        raw = weights * float(total_samples)
        counts = np.floor(raw).astype(np.int64)
        remainder = total_samples - int(counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            for index in order[:remainder]:
                counts[int(index)] += 1
        positive = np.flatnonzero(weights > 0)
        if len(positive) <= total_samples:
            for index in positive:
                if counts[int(index)] == 0:
                    donor_order = np.argsort(-counts)
                    for donor in donor_order:
                        if counts[int(donor)] > 1:
                            counts[int(donor)] -= 1
                            counts[int(index)] += 1
                            break
        if int(counts.sum()) != total_samples:
            raise AssertionError(f"branch count allocation mismatch: {counts.sum()} != {total_samples}")
        return [int(value) for value in counts.tolist()]

    def _apply_cluster_weight_jitter(self, weights: np.ndarray) -> np.ndarray:
        """Apply optional random perturbation to branch weights before normalization."""
        cluster_cfg = self._cluster_weight_config()
        jitter_cfg = cluster_cfg.get("branch_weight_jitter")
        if jitter_cfg is None:
            return np.asarray(weights, dtype=np.float64)

        if isinstance(jitter_cfg, dict):
            jitter = float(jitter_cfg.get("scale", jitter_cfg.get("sigma", 0.0)))
            mode = str(jitter_cfg.get("mode", "lognormal")).lower()
            seed = jitter_cfg.get("seed", None)
        else:
            jitter = float(jitter_cfg)
            mode = "lognormal"
            seed = None

        if jitter is None or jitter <= 0:
            return np.asarray(weights, dtype=np.float64)

        weights = np.asarray(weights, dtype=np.float64).copy()
        rng = np.random.default_rng(self.seed if seed is None else int(seed))

        positive = weights > 0
        if not np.any(positive):
            return weights

        if mode == "lognormal":
            factors = rng.lognormal(mean=0.0, sigma=jitter, size=weights.shape)
        elif mode == "uniform":
            factors = 1.0 + rng.uniform(-jitter, jitter, size=weights.shape)
        elif mode == "gaussian":
            factors = 1.0 + rng.normal(0.0, jitter, size=weights.shape)
        else:
            raise ValueError(f"Unsupported branch_weight_jitter mode: {mode}")

        factors = np.asarray(factors, dtype=np.float64)
        factors = np.where(factors > 0, factors, 0.0)
        weights[positive] = weights[positive] * factors[positive]
        return weights


    def _cluster_weight_config(self) -> dict[str, Any]:
        payload = self.qcbm.get("cluster_weighted", {})
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError("qcbm.cluster_weighted must be a mapping")
        return payload

    def _branch_regularization_config(self) -> dict[str, Any]:
        payload = self.qcbm.get("branch_regularization", {})
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError("qcbm.branch_regularization must be a mapping")
        return payload

    def _build_all_branch_condition_payload(
        self,
        ancilla_bits: int,
        n_qubits: int,
        ancilla_wires: list[int],
    ) -> dict[str, Any]:
        if ancilla_bits <= 0:
            raise ValueError("branch regularization requires ancilla_bits > 0")
        if len(ancilla_wires) != ancilla_bits:
            raise ValueError(f"Expected {ancilla_bits} ancilla wires, got {ancilla_wires}")

        condition_masks = []
        condition_values = []
        rows = []
        for branch_index in range(2 ** int(ancilla_bits)):
            branch = branch_bits_for_index(branch_index, int(ancilla_bits))
            mask = np.zeros(n_qubits, dtype=bool)
            values = np.zeros(n_qubits, dtype=np.int32)
            for wire, bit in zip(ancilla_wires, branch):
                mask[int(wire)] = True
                values[int(wire)] = int(bit)
            condition_masks.append(self.jnp.asarray(mask, dtype=bool))
            condition_values.append(self.jnp.asarray(values, dtype=self.jnp.int32))
            rows.append({"branch_index": int(branch_index), "branch_bits": branch})
        return {
            "mapping": rows,
            "condition_masks": tuple(condition_masks),
            "condition_values": tuple(condition_values),
        }

    def _build_cluster_weight_payload(
        self,
        ancilla_bits: int,
        n_qubits: int,
        ancilla_wires: list[int],
    ) -> dict[str, Any]:
        if ancilla_bits <= 0:
            raise ValueError("cluster-weighted training requires ancilla_bits > 0")
        if len(ancilla_wires) != ancilla_bits:
            raise ValueError(f"Expected {ancilla_bits} ancilla wires, got {ancilla_wires}")

        label_path, labels = self._load_label_vector()
        mapping = label_branch_mapping(labels, ancilla_bits)
        cluster_cfg = self._cluster_weight_config()
        explicit_weights = cluster_cfg.get("branch_weights")
        label_counts = {int(label): int(np.sum(labels == int(label))) for label in np.unique(labels)}
        if explicit_weights is None:
            weights = np.asarray([label_counts[int(entry["label"])] for entry in mapping], dtype=np.float64)
        elif isinstance(explicit_weights, dict):
            weights = np.asarray([float(explicit_weights[str(entry["label"])]) for entry in mapping], dtype=np.float64)
        else:
            weights = np.asarray(explicit_weights, dtype=np.float64)
            if weights.shape != (len(mapping),):
                raise ValueError(f"branch_weights length {weights.shape} does not match active branches {len(mapping)}")
        weights = weights / float(weights.sum())
        weights = self._apply_cluster_weight_jitter(weights)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            raise ValueError("After jittering, branch weights must remain positive")
        weights = weights / total_weight
        sample_counts = self._allocate_branch_sample_counts(weights, int(self.qcbm["n_samples"]))

        condition_masks = []
        condition_values = []
        rows = []
        for entry, weight, sample_count in zip(mapping, weights.tolist(), sample_counts):
            mask = np.zeros(n_qubits, dtype=bool)
            values = np.zeros(n_qubits, dtype=np.int32)
            for wire, bit in zip(ancilla_wires, str(entry["branch_bits"])):
                mask[int(wire)] = True
                values[int(wire)] = int(bit)
            condition_masks.append(self.jnp.asarray(mask, dtype=bool))
            condition_values.append(self.jnp.asarray(values, dtype=self.jnp.int32))
            rows.append(
                {
                    "label": int(entry["label"]),
                    "branch_index": int(entry["branch_index"]),
                    "branch_bits": str(entry["branch_bits"]),
                    "sample_count": int(label_counts[int(entry["label"])]),
                    "mixture_weight": float(weight),
                    "mixture_sample_count": int(sample_count),
                }
            )
        return {
            "labels_path": str(label_path),
            "mapping": rows,
            "condition_masks": tuple(condition_masks),
            "condition_values": tuple(condition_values),
            "sample_counts": tuple(int(value) for value in sample_counts),
            "weights": tuple(float(value) for value in weights.tolist()),
        }

    def train_stage(
        self,
        layout_name: str,
        ancilla_bits: int,
        previous_gates: list[list[list[int]]] | None = None,
        previous_params: np.ndarray | None = None,
        stage_steps: int | None = None,
        label_condition_source: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_iqpopt_vendor()
        from iqp_fast import IqpSimulator, Trainer
        import iqp_fast.gen_qml as gen
        from iqp_fast.utils import initialize_from_data

        layouts = self.config["layouts"]
        layout = layouts[layout_name]
        visible_gates = build_visible_gates(layout_name, layout, self.visible_bits)
        expected_count = expected_gate_count(layout, self.visible_bits)
        if len(visible_gates) != expected_count:
            raise ValueError(f"{layout_name}: expected {expected_count} visible gates, got {len(visible_gates)}")

        gates, visible_wires, ancilla_wires = augment_visible_gates_with_ancilla(
            visible_gates,
            visible_bits=self.visible_bits,
            ancilla_bits=int(ancilla_bits),
            policy=str(self.qcbm.get("ancilla_gate_policy", "all_subsets")),
        )
        n_qubits = self.visible_bits + int(ancilla_bits)
        circuit = IqpSimulator(
            n_qubits,
            gates,
            device=str(self.qcbm.get("device", "lightning.gpu")),
            spin_sym=bool(self.qcbm.get("spin_sym", False)),
            sparse=bool(self.qcbm.get("sparse", False)),
            bitflip=bool(self.qcbm.get("bitflip", False)),
        )
        init_artifact_path: Path | None = None
        if previous_gates is None and previous_params is None:
            init_artifact = self._load_init_artifact(layout_name)
            if init_artifact is not None:
                init_artifact_path, init_payload = init_artifact
                previous_gates = init_payload["gates"]
                previous_params = torch.as_tensor(init_payload["params"], dtype=torch.float64).numpy()
        params = self._initial_params(
            initialize_from_data=initialize_from_data,
            gates=gates,
            visible_bits=self.visible_bits,
            previous_gates=previous_gates,
            previous_params=previous_params,
            initialize_missing_visible=init_artifact_path is None,
        )
        schedule = self._lr_schedule(layout_name=layout_name, ancilla_bits=int(ancilla_bits), stage_steps=stage_steps)

        label_condition_payload: dict[str, Any] | None = None
        cluster_weight_payload: dict[str, Any] | None = None
        branch_regularization_payload: dict[str, Any] | None = None
        branch_regularization_cfg = self._branch_regularization_config()
        use_branch_regularization = (
            bool(branch_regularization_cfg.get("enabled", False))
            and float(branch_regularization_cfg.get("weight", 0.0)) > 0.0
            and int(ancilla_bits) > 0
        )
        loss_fn: Any = gen.mmd_loss_iqp
        if self._use_label_conditioned_training():
            label_condition_payload = self._build_label_condition_payload(
                ancilla_bits=int(ancilla_bits),
                n_qubits=n_qubits,
                ancilla_wires=ancilla_wires,
            )
            loss_fn = label_conditioned_mmd_loss
        elif self._use_cluster_weighted_training():
            cluster_weight_payload = self._build_cluster_weight_payload(
                ancilla_bits=int(ancilla_bits),
                n_qubits=n_qubits,
                ancilla_wires=ancilla_wires,
            )
            loss_fn = branch_weighted_mixture_mmd_loss

        if use_branch_regularization:
            if label_condition_payload is not None or cluster_weight_payload is not None:
                raise ValueError("branch_regularization currently supports standard all-data MMD training only")
            branch_regularization_payload = self._build_all_branch_condition_payload(
                ancilla_bits=int(ancilla_bits),
                n_qubits=n_qubits,
                ancilla_wires=ancilla_wires,
            )
            loss_fn = branch_separated_mmd_loss

        loss_kwargs: dict[str, Any] = {
            "params": self.jnp.asarray(params, dtype=float),
            "iqp_circuit": circuit,
            "sigma": [float(s) for s in self.qcbm["sigmas"]],
            "n_ops": int(self.qcbm["n_ops"]),
            "n_samples": int(self.qcbm["n_samples"]),
            "wires": visible_wires,
            "jit": bool(self.qcbm.get("loss_jit", True)),
            "sqrt_loss": False,
            "max_batch_ops": self.qcbm.get("max_batch_ops"),
            "max_batch_samples": self.qcbm.get("max_batch_samples"),
        }
        if label_condition_payload is None:
            loss_kwargs["ground_truth"] = self.train_bits
        else:
            loss_kwargs.update(
                {
                    "label_ground_truths": label_condition_payload["ground_truths"],
                    "label_condition_masks": label_condition_payload["condition_masks"],
                    "label_condition_values": label_condition_payload["condition_values"],
                }
            )
        if cluster_weight_payload is not None:
            loss_kwargs["ground_truth"] = self.train_bits
            loss_kwargs.update(
                {
                    "branch_condition_masks": cluster_weight_payload["condition_masks"],
                    "branch_condition_values": cluster_weight_payload["condition_values"],
                    "branch_sample_counts": cluster_weight_payload["sample_counts"],
                }
            )
        if branch_regularization_payload is not None:
            loss_kwargs["ground_truth"] = self.train_bits
            loss_kwargs.update(
                {
                    "branch_condition_masks": branch_regularization_payload["condition_masks"],
                    "branch_condition_values": branch_regularization_payload["condition_values"],
                    "branch_regularization": dict(branch_regularization_cfg),
                    "training_step": 0,
                    "training_total_steps": int(sum(int(block["steps"]) for block in schedule if int(block["steps"]) > 0)),
                }
            )
            component_kwargs = {key: value for key, value in loss_kwargs.items() if key != "params"}

            def branch_component_fn(params_value: Any, key_value: Any, step_value: int | None = None) -> dict[str, float]:
                kwargs = dict(component_kwargs)
                if step_value is not None:
                    kwargs["training_step"] = int(step_value)
                return branch_separated_mmd_components(
                    params=params_value,
                    key=key_value,
                    **kwargs,
                )

            loss_kwargs["_component_fn"] = branch_component_fn
            loss_kwargs["_component_interval"] = int(branch_regularization_cfg.get("log_interval", self.qcbm.get("turbo") or 1))
            loss_kwargs["_step_kwarg_name"] = "training_step"

        losses: list[float] = []
        schedule_rows: list[dict[str, Any]] = []
        current_params = loss_kwargs["params"]
        select_best_params = bool(self.qcbm.get("select_best_params", False))
        monitor_interval_cfg = self.qcbm.get("monitor_interval")
        monitor_interval = int(monitor_interval_cfg) if monitor_interval_cfg is not None else None
        selected_params: np.ndarray | None = None
        selected_loss = math.inf
        selected_score = math.inf
        selected_step: int | None = None
        total_seen = 0
        turbo = self.qcbm.get("turbo")
        best_selection_window = max(1, int(self.qcbm.get("best_selection_smoothing_window", 1)))

        def selection_score(loss_values: np.ndarray, loss_index: int) -> float:
            loss_index = int(loss_index)
            if best_selection_window <= 1 or len(loss_values) <= 1:
                return float(loss_values[loss_index])
            half_window = best_selection_window // 2
            start = max(0, loss_index - half_window)
            end = min(len(loss_values), loss_index + half_window + 1)
            if end - start < best_selection_window:
                if start == 0:
                    end = min(len(loss_values), best_selection_window)
                else:
                    start = max(0, end - best_selection_window)
            return float(np.mean(loss_values[start:end]))

        active_schedule = [block for block in schedule if int(block["steps"]) > 0]
        optimizer_name = str(self.qcbm.get("optimizer", "Adam"))
        compile_once_lr_schedule = bool(
            self.qcbm.get(
                "compile_once_lr_schedule",
                optimizer_name == "Adam" and len(active_schedule) > 1,
            )
        )

        def candidate_params_from_run(
            trainer_obj: Any,
            run_losses_np: np.ndarray,
            run_final_params: np.ndarray,
            run_offset: int,
        ) -> tuple[float | None, float | None, np.ndarray | None, int | None]:
            if not len(run_losses_np):
                return None, None, None, None
            final_loss_index = len(run_losses_np) - 1
            candidates: list[tuple[float, float, np.ndarray, int]] = [
                (
                    selection_score(run_losses_np, final_loss_index),
                    float(run_losses_np[final_loss_index]),
                    np.asarray(run_final_params, dtype=np.float64).copy(),
                    run_offset + final_loss_index,
                )
            ]
            if select_best_params and monitor_interval is not None:
                params_hist = getattr(trainer_obj, "params_hist", [])
                if len(params_hist):
                    params_hist_np = np.asarray(params_hist, dtype=np.float64)
                    for hist_index, hist_params in enumerate(params_hist_np):
                        if turbo is None:
                            loss_index = hist_index * monitor_interval
                        else:
                            loss_index = hist_index
                            if not np.any(np.abs(hist_params) > 1e-14):
                                continue
                        if loss_index < len(run_losses_np):
                            candidates.append(
                                (
                                    selection_score(run_losses_np, loss_index),
                                    float(run_losses_np[loss_index]),
                                    np.asarray(hist_params, dtype=np.float64).copy(),
                                    run_offset + loss_index,
                                )
                            )
            score, loss_value, params_value, step = min(candidates, key=lambda item: (item[0], item[1]))
            return float(score), float(loss_value), params_value, int(step)

        def append_schedule_rows_from_losses(run_losses_np: np.ndarray) -> None:
            offset = 0
            for block_index, block in enumerate(schedule):
                steps = int(block["steps"])
                if steps <= 0:
                    continue
                block_losses_np = run_losses_np[offset: offset + steps]
                block_losses = block_losses_np.tolist()
                block_selected_loss: float | None = None
                block_selected_score: float | None = None
                block_selected_step: int | None = None
                if len(block_losses_np):
                    block_scores = [selection_score(block_losses_np, idx) for idx in range(len(block_losses_np))]
                    best_idx = int(np.argmin(np.asarray(block_scores, dtype=np.float64)))
                    block_selected_score = float(block_scores[best_idx])
                    block_selected_loss = float(block_losses_np[best_idx])
                    block_selected_step = total_seen + offset + best_idx
                schedule_rows.append(
                    {
                        "block": block_index,
                        "kind": block["kind"],
                        "steps": steps,
                        "lr": float(block["lr"]),
                        "loss_first": float(block_losses[0]) if block_losses else None,
                        "loss_last": float(block_losses[-1]) if block_losses else None,
                        "loss_best_monitored": block_selected_loss,
                        "loss_best_smoothed": block_selected_score,
                        "best_monitored_step": block_selected_step,
                        "best_selection_smoothing_window": best_selection_window,
                    }
                )
                offset += steps

        if compile_once_lr_schedule and len(active_schedule) > 1:
            if optimizer_name != "Adam":
                raise ValueError("compile_once_lr_schedule currently supports Adam only")
            lr_values = np.concatenate(
                [
                    np.full(int(block["steps"]), float(block["lr"]), dtype=np.float64)
                    for block in active_schedule
                ]
            )
            lr_values_jnp = self.jnp.asarray(lr_values, dtype=float)

            def optax_lr_schedule(step: Any) -> Any:
                index = self.jnp.minimum(step, lr_values_jnp.shape[0] - 1)
                return lr_values_jnp[index]

            loss_kwargs["params"] = current_params
            trainer = Trainer(
                optimizer_name,
                loss_fn,
                optax_lr_schedule,
                bool(self.qcbm.get("opt_jit", False)),
            )
            trainer.train(
                n_iters=int(lr_values.shape[0]),
                loss_kwargs=loss_kwargs,
                convergence_interval=self.qcbm.get("convergence_interval"),
                monitor_interval=monitor_interval,
                turbo=turbo,
                random_state=self.seed + ancilla_bits * 1000,
            )
            run_losses_np = np.asarray(trainer.losses, dtype=np.float64)
            losses.extend(float(v) for v in run_losses_np.tolist())
            run_final_params = np.asarray(trainer.final_params, dtype=np.float64).copy()
            score, loss_value, params_value, step = candidate_params_from_run(
                trainer,
                run_losses_np,
                run_final_params,
                total_seen,
            )
            if score is not None and select_best_params and score < selected_score:
                selected_score = float(score)
                selected_loss = float(loss_value)
                selected_params = np.asarray(params_value, dtype=np.float64).copy()
                selected_step = int(step)
            current_params = self.jnp.asarray(run_final_params)
            append_schedule_rows_from_losses(run_losses_np)
            total_seen += len(run_losses_np)
        else:
            for block_index, block in enumerate(schedule):
                if block["steps"] <= 0:
                    continue
                loss_kwargs["params"] = current_params
                trainer = Trainer(
                    optimizer_name,
                    loss_fn,
                    float(block["lr"]),
                    bool(self.qcbm.get("opt_jit", False)),
                )
                trainer.train(
                    n_iters=int(block["steps"]),
                    loss_kwargs=loss_kwargs,
                    convergence_interval=self.qcbm.get("convergence_interval"),
                    monitor_interval=monitor_interval,
                    turbo=turbo,
                    random_state=self.seed + ancilla_bits * 1000 + block_index,
                )
                block_losses_np = np.asarray(trainer.losses, dtype=np.float64)
                block_losses = block_losses_np.tolist()
                losses.extend(float(v) for v in block_losses)
                block_final_params = np.asarray(trainer.final_params, dtype=np.float64).copy()

                block_selected_loss: float | None = None
                block_selected_score: float | None = None
                block_selected_step: int | None = None
                score, loss_value, params_value, step = candidate_params_from_run(
                    trainer,
                    block_losses_np,
                    block_final_params,
                    total_seen,
                )
                if score is not None:
                    block_selected_score = float(score)
                    block_selected_loss = float(loss_value)
                    block_selected_step = int(step)
                    if select_best_params and block_selected_score < selected_score:
                        selected_score = float(block_selected_score)
                        selected_loss = float(block_selected_loss)
                        selected_params = np.asarray(params_value, dtype=np.float64).copy()
                        selected_step = int(block_selected_step)

                current_params = self.jnp.asarray(block_final_params)
                schedule_rows.append(
                    {
                        "block": block_index,
                        "kind": block["kind"],
                        "steps": int(block["steps"]),
                        "lr": float(block["lr"]),
                        "loss_first": float(block_losses[0]) if block_losses else None,
                        "loss_last": float(block_losses[-1]) if block_losses else None,
                        "loss_best_monitored": block_selected_loss,
                        "loss_best_smoothed": block_selected_score,
                        "best_monitored_step": block_selected_step,
                        "best_selection_smoothing_window": best_selection_window,
                    }
                )
                total_seen += len(block_losses_np)


        training_final_params = np.asarray(current_params, dtype=np.float64).copy()
        if select_best_params and selected_params is not None:
            final_params = selected_params.copy()
            selected_source = "best_monitored"
        else:
            final_params = training_final_params
            selected_source = "final"
            if losses:
                selected_loss = float(losses[-1])
                selected_score = float(losses[-1])
                selected_step = len(losses) - 1
        output_dir = self.checkpoint_dir(layout_name, int(ancilla_bits))
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact: dict[str, Any] = {
            "format": "blobs16_randomized_iqp_v1",
            "layout_name": layout_name,
            "layout": layout,
            "n_qubits": n_qubits,
            "visible_bits": self.visible_bits,
            "ancilla_bits": int(ancilla_bits),
            "visible_wires": visible_wires,
            "ancilla_wires": ancilla_wires,
            "visible_gates": visible_gates,
            "gates": gates,
            "params": torch.as_tensor(final_params, dtype=torch.float64),
            "training_final_params": torch.as_tensor(training_final_params, dtype=torch.float64),
            "losses": torch.as_tensor(losses, dtype=torch.float64),
            "selected_loss": selected_loss if math.isfinite(selected_loss) else None,
            "selected_score": selected_score if math.isfinite(selected_score) else None,
            "selected_step": selected_step,
            "selected_source": selected_source,
            "best_selection_smoothing_window": best_selection_window,
            "schedule": schedule_rows,
            "config": self.config,
            "previous_stage_loaded": previous_gates is not None,
            "training_mode": str(self.qcbm.get("training_mode", "regressive")),
        }
        if init_artifact_path is not None:
            artifact["init_from_artifact"] = str(init_artifact_path)
        if label_condition_payload is not None:
            artifact["label_condition"] = {
                "labels_path": label_condition_payload["labels_path"],
                "mapping": label_condition_payload["mapping"],
                "source_artifact": label_condition_source,
                "objective": "mean_per_label_conditioned_mmd",
            }
        if cluster_weight_payload is not None:
            artifact["cluster_weighted"] = {
                "labels_path": cluster_weight_payload["labels_path"],
                "mapping": cluster_weight_payload["mapping"],
                "branch_weights": list(cluster_weight_payload["weights"]),
                "branch_sample_counts": list(cluster_weight_payload["sample_counts"]),
                "objective": "all_data_mmd_with_stratified_active_ancilla_branch_mixture",
            }
        if branch_regularization_payload is not None:
            artifact["branch_regularization"] = {
                "config": dict(branch_regularization_cfg),
                "mapping": branch_regularization_payload["mapping"],
                "objective": "all_data_mmd_plus_branch_z_separation",
            }
        torch.save(artifact, output_dir / "qcbm_artifact.pt")
        torch.save(
            {
                "losses": artifact["losses"],
                "schedule": schedule_rows,
                "selected_loss": artifact["selected_loss"],
                "selected_score": artifact["selected_score"],
                "selected_step": artifact["selected_step"],
                "selected_source": artifact["selected_source"],
                "best_selection_smoothing_window": artifact["best_selection_smoothing_window"],
            },
            output_dir / "qcbm_mmd_curve.pt",
        )
        np.savetxt(output_dir / "loss_curve.csv", np.asarray(losses, dtype=np.float64), delimiter=",")
        save_json(
            output_dir / "stage_meta.json",
            {
                "layout_name": layout_name,
                "ancilla_bits": int(ancilla_bits),
                "visible_gate_count": len(visible_gates),
                "total_gate_count": len(gates),
                "schedule": schedule_rows,
                "selected_loss": artifact["selected_loss"],
                "selected_score": artifact["selected_score"],
                "selected_step": artifact["selected_step"],
                "selected_source": artifact["selected_source"],
                "best_selection_smoothing_window": artifact["best_selection_smoothing_window"],
                "previous_stage_loaded": artifact["previous_stage_loaded"],
                "training_mode": artifact["training_mode"],
                "init_from_artifact": artifact.get("init_from_artifact"),
                "label_condition": artifact.get("label_condition"),
                "cluster_weighted": artifact.get("cluster_weighted"),
                "branch_regularization": artifact.get("branch_regularization"),
                "artifact": str(output_dir / "qcbm_artifact.pt"),
            },
        )
        if ancilla_bits > 0:
            export_ancilla_branches(output_dir / "qcbm_artifact.pt", output_dir / "ancilla_branches")
        return artifact

    def _initial_params(
        self,
        initialize_from_data: Any,
        gates: list[list[list[int]]],
        visible_bits: int,
        previous_gates: list[list[list[int]]] | None,
        previous_params: np.ndarray | None,
        initialize_missing_visible: bool = True,
    ) -> np.ndarray:
        visible_indices = [
            index for index, gate in enumerate(gates)
            if all(wire < visible_bits for generator in gate for wire in generator)
        ]
        visible_gates = [gates[index] for index in visible_indices]
        params = np.zeros(len(gates), dtype=np.float64)
        init_mode = str(self.qcbm.get("init_mode", "")).strip().lower()
        if init_mode == "unbiased":
            # Data-agnostic unbiased center (paper thm:rand-bp / app:bp):
            # one-body (wire-count 1) generators -> pi/4, two-body (wire-count 2)
            # generators -> 0 (every nontrivial correlator vanishes at this
            # center), then add small INDEPENDENT Gaussian noise to every angle
            # to break the symmetry between ancilla branches. Body is determined
            # by the number of wires in a gate's generator, not by index.
            for index, gate in enumerate(gates):
                wire_count = max((len(generator) for generator in gate), default=0)
                params[index] = np.pi / 4.0 if wire_count == 1 else 0.0
            noise_scale = float(self.qcbm.get("init_param_noise", 0.05))
            if noise_scale > 0:
                rng = np.random.default_rng(self.seed + len(gates) + int(visible_bits))
                params[:] = params + rng.normal(0.0, noise_scale, size=len(gates))
            return np.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
        if bool(self.qcbm.get("random_init", False)):
            rng = np.random.default_rng(self.seed + len(gates) + int(visible_bits))
            params[:] = rng.normal(
                0.0,
                float(self.qcbm.get("random_init_scale", self.qcbm.get("init_scale", 0.0001))),
                size=len(gates),
            )
        elif initialize_missing_visible:
            visible_params = np.asarray(
                initialize_from_data(
                    visible_gates,
                    self.train_bits,
                    scale=float(self.qcbm.get("init_scale", 0.0001)),
                    param_noise=float(self.qcbm.get("init_param_noise", 0.0001)),
                    zero_two_body=bool(self.qcbm.get("zero_two_body", False)),
                    noise_all=bool(self.qcbm.get("noise_all", False)),
                ),
                dtype=np.float64,
            )
            params[visible_indices] = visible_params

        copied = np.zeros(len(gates), dtype=bool)
        if previous_gates is not None and previous_params is not None:
            prev_by_key = {gate_key(gate): float(param) for gate, param in zip(previous_gates, previous_params)}
            for index, gate in enumerate(gates):
                key = gate_key(gate)
                if key in prev_by_key:
                    params[index] = prev_by_key[key]
                    copied[index] = True

        new_noise = float(self.qcbm.get("new_ancilla_param_noise", 0.0001))
        if new_noise > 0:
            rng = np.random.default_rng(self.seed + len(gates))
            visible_set = set(visible_indices)
            noise_indices = [
                index for index in range(len(gates))
                if index not in visible_set and not copied[index]
            ]
            params[noise_indices] = rng.normal(0.0, new_noise, size=len(noise_indices))
        return np.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)

    def _lr_schedule(
        self,
        layout_name: str,
        ancilla_bits: int,
        stage_steps: int | None = None,
    ) -> list[dict[str, Any]]:
        total_steps = int(stage_steps if stage_steps is not None else self.qcbm["stage_steps"])
        base_lr = float(self.qcbm["base_lr"])
        layout_training = self.config.get("layout_training", {}).get(layout_name, {})
        if ancilla_bits > 0:
            ancilla_base_lrs = layout_training.get("ancilla_base_lrs", {})
            if str(ancilla_bits) in ancilla_base_lrs:
                base_lr = float(ancilla_base_lrs[str(ancilla_bits)])
            elif ancilla_bits in ancilla_base_lrs:
                base_lr = float(ancilla_base_lrs[ancilla_bits])
            elif "ancilla_base_lr" in layout_training:
                base_lr = float(layout_training["ancilla_base_lr"])
            else:
                base_lr *= float(layout_training.get("ancilla_lr_factor", 1.0))
        elif "base_lr" in layout_training:
            base_lr = float(layout_training["base_lr"])

        if ancilla_bits <= 0 or not bool(self.config.get("warmup", {}).get("enabled_for_ancilla", True)):
            return [{"kind": "main", "steps": total_steps, "lr": base_lr}]

        warmup_cfg = self.config["warmup"]
        warmup_steps = max(0, int(round(total_steps * float(warmup_cfg.get("fraction", 0.2)))))
        warmup_steps = min(warmup_steps, total_steps)
        requested_blocks = int(warmup_cfg.get("blocks", 5))
        num_blocks = min(max(requested_blocks, 1), max(warmup_steps, 1))
        start_lr_factor = float(layout_training.get("ancilla_warmup_start_lr_factor", warmup_cfg.get("start_lr_factor", 0.1)))
        start_lr = base_lr * start_lr_factor
        warmup_lrs = np.linspace(start_lr, base_lr, num_blocks)
        block_sizes = [warmup_steps // num_blocks] * num_blocks
        for index in range(warmup_steps % num_blocks):
            block_sizes[index] += 1

        schedule = [
            {"kind": "warmup", "steps": int(steps), "lr": float(lr)}
            for steps, lr in zip(block_sizes, warmup_lrs)
            if steps > 0
        ]
        remaining = total_steps - warmup_steps
        if remaining > 0:
            schedule.append({"kind": "main", "steps": remaining, "lr": base_lr})
        return schedule

