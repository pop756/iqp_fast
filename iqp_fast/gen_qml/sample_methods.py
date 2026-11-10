# Derived from iqpopt (https://github.com/XanaduAI/iqpopt), Apache License 2.0.
# Part of iqp_fast; see LICENSE and NOTICE.
# Modifications Copyright 2026 Youngseok Lee. Changes: package renamed to iqp_fast; code unchanged.

from functools import partial
import jax
import jax.numpy as jnp
import cvxpy as cp
from .utils import gaussian_kernel


def mmd_loss_slow_samples(ground_truth: jnp.ndarray, model_samples: jnp.ndarray, sigma: float) -> float:
    """Calculates the Maximum Mean Discrepancy (MMD) loss from samples.

    Args:
        ground_truth (jnp.ndarray): Samples from the ground truth.
        model_samples (jnp.ndarray): Samples from the test model.
        sigma (float): Sigma parameter, the width of the kernel.

    Returns:
        float: The value of the MMD loss.
    """
    n = len(ground_truth)
    m = len(model_samples)

    K_pp = jnp.array([[gaussian_kernel(sigma, x, y)
                     for x in ground_truth] for y in ground_truth])
    sum_pp = jnp.sum(
        K_pp) - jnp.sum(jnp.array([gaussian_kernel(sigma, x, x) for x in ground_truth]))

    K_pq = jnp.array([[gaussian_kernel(sigma, x, y)
                     for x in model_samples] for y in ground_truth])
    sum_pq = jnp.sum(K_pq)

    K_qq = jnp.array([[gaussian_kernel(sigma, x, y)
                     for x in model_samples] for y in model_samples])
    sum_qq = jnp.sum(
        K_qq) - jnp.sum(jnp.array([gaussian_kernel(sigma, x, x) for x in model_samples]))

    return 1/n/(n-1) * sum_pp - 2/n/m * sum_pq + 1/m/(m-1) * sum_qq


def mmd_loss_samples(ground_truth: jnp.ndarray, model_samples: jnp.ndarray, sigma: float) -> float:
    """Calculates the Maximum Mean Discrepancy (MMD) loss from samples with
    jax methods to make it faster.

    Args:
        ground_truth (jnp.ndarray): Samples from the ground truth.
        model_samples (jnp.ndarray): Samples from the test model.
        sigma (float): Sigma parameter, the width of the kernel.

    Returns:
        float: The value of the MMD loss.
    """
    n = len(ground_truth)
    m = len(model_samples)
    ground_truth = jnp.array(ground_truth)
    model_samples = jnp.array(model_samples)

    # K_pp
    K_pp = jnp.zeros((ground_truth.shape[0], ground_truth.shape[0]))

    def body_fun(i, val):
        def inner_body_fun(j, inner_val):
            return inner_val.at[i, j].set(gaussian_kernel(sigma, ground_truth[i], ground_truth[j]))
        return jax.lax.fori_loop(0, ground_truth.shape[0], inner_body_fun, val)
    K_pp = jax.lax.fori_loop(0, ground_truth.shape[0], body_fun, K_pp)
    sum_pp = jnp.sum(K_pp) - n

    # K_pq
    K_pq = jnp.zeros((ground_truth.shape[0], model_samples.shape[0]))

    def body_fun(i, val):
        def inner_body_fun(j, inner_val):
            return inner_val.at[i, j].set(gaussian_kernel(sigma, ground_truth[i], model_samples[j]))
        return jax.lax.fori_loop(0, model_samples.shape[0], inner_body_fun, val)
    K_pq = jax.lax.fori_loop(0, ground_truth.shape[0], body_fun, K_pq)
    sum_pq = jnp.sum(K_pq)

    # K_qq
    K_qq = jnp.zeros((model_samples.shape[0], model_samples.shape[0]))

    def body_fun(i, val):
        def inner_body_fun(j, inner_val):
            return inner_val.at[i, j].set(gaussian_kernel(sigma, model_samples[i], model_samples[j]))
        return jax.lax.fori_loop(0, model_samples.shape[0], inner_body_fun, val)
    K_qq = jax.lax.fori_loop(0, model_samples.shape[0], body_fun, K_qq)
    sum_qq = jnp.sum(K_qq) - m

    return 1/n/(n-1) * sum_pp - 2/n/m * sum_pq + 1/m/(m-1) * sum_qq


def kgel_opt_samples(witnesses: jnp.ndarray, ground_truth: jnp.ndarray, model_samples: jnp.ndarray,
                     sigma: float) -> list:
    """Calculates the kernel generalized empiral likelihood (kgel) test of equation 6 in https://arxiv.org/pdf/2306.09780.

    Args:
        witnesses (jnp.ndarray): The witness points for the evaluation of the kernel (see the mentioned eq. 6).
        ground_truth (jnp.ndarray): Samples from the true distribution.
        model_samples (jnp.ndarray): Samples from the test model.
        sigma (float): Sigma parameter, the width of the kernel.

    Returns:
        list:
            result (float): Final value of the KL divergence of the optimization.
            pi.value (jnp.ndarray): Final values of the pi variable of the optimization.
    """
    # Construct the problem.
    pi = cp.Variable(len(ground_truth))
    uniform = 1/len(ground_truth)*jnp.ones(shape=(len(ground_truth),))

    objective = cp.Minimize(cp.sum(cp.rel_entr(pi, uniform)))

    truth_kernels = jnp.array(
        [list(map(partial(gaussian_kernel, sigma, s), witnesses)) for s in ground_truth])
    model_kernels = jnp.array(
        [list(map(partial(gaussian_kernel, sigma, s), witnesses)) for s in model_samples])
    constraints = pi @ truth_kernels - jnp.mean(model_kernels, axis=0)

    prob = cp.Problem(objective, [
                      c == 0 for c in constraints] + [cp.sum(pi) == 1] + [p >= 0 for p in pi])
    # The optimal objective is returned by prob.solve().
    result = prob.solve()

    # The optimal value for pi is stored in pi.value.
    return result, pi.value
