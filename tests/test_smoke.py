# Copyright 2026 Youngseok Lee
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: the estimator matches the exact expectation on a small circuit, and the MMD loss has a gradient.

Run with `python tests/test_smoke.py` or `pytest tests`. Small circuits use the XLA paths on every backend.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np

import iqp_fast
from iqp_fast.gen_qml import mmd_loss_iqp
from iqp_fast.utils import local_gates

N = 4


def exact_expval(gates, params, op):
    """<Z_op> = mean over all bitstrings z of cos(2 sum_g theta_g sum_{gen in g} parity(op.gen) (-1)^parity(z.gen))."""
    vals = []
    for z in itertools.product([0, 1], repeat=N):
        phase = 0.0
        for theta, gate in zip(params, gates):
            for gen in gate:
                if sum(op[w] for w in gen) % 2:
                    phase += theta * (-1) ** (sum(z[w] for w in gen) % 2)
        vals.append(np.cos(2 * phase))
    return float(np.mean(vals))


def test_op_expval_matches_exact():
    gates = local_gates(N, 3)  # 1-, 2- and 3-body generators
    sim = iqp_fast.IqpSimulator(N, gates)
    params = np.random.default_rng(0).uniform(-0.5, 0.5, len(gates))
    ops = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 1], [1, 1, 1, 1]])
    expvals, stds = sim.op_expval(jnp.array(params), jnp.array(ops), 40000, jax.random.PRNGKey(1))
    for op, e, s in zip(ops, np.asarray(expvals), np.asarray(stds)):
        assert abs(e - exact_expval(gates, params, op)) < 5 * s + 1e-6, (op, e, s)


def test_mmd_loss_has_gradient():
    gates = local_gates(N, 2)
    sim = iqp_fast.IqpSimulator(N, gates)
    data = jnp.array(np.random.default_rng(2).integers(0, 2, (64, N)))
    params = jnp.full(len(gates), 0.1)
    loss, grad = jax.value_and_grad(mmd_loss_iqp)(params, sim, data, 1.0, 100, 500, jax.random.PRNGKey(3))
    assert np.isfinite(loss) and np.all(np.isfinite(grad)) and np.any(np.asarray(grad) != 0)


if __name__ == "__main__":
    test_op_expval_matches_exact()
    test_mmd_loss_has_gradient()
    print("smoke OK")
