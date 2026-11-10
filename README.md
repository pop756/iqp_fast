# iqp_fast

A faster [iqpopt](https://github.com/XanaduAI/iqpopt): the Monte Carlo estimator of IQP circuit
expectation values and the MMD loss, rewritten with GPU kernels for AMD ROCm.

**Starting point.** iqp_fast starts from a modified iqpopt. Among other changes, it adds to upstream:
- float32 by default;
- grouped and pairwise expectation paths;
- branch/label conditioning.

**Reference.** That modified iqpopt, run in float32, is the reference below.

**What is the same.** For a given key, iqp_fast draws the same random operators and samples as the
reference and returns the same estimate, within float32 rounding. Only the arithmetic is reorganised.

**Upstream comparison.** Upstream iqpopt turns on float64 at import, which changes the random stream. Its
numbers are therefore not directly comparable.

## Performance

**Workload.** 2,500 Adam steps of the branch-weighted MMD loss on a 484-qubit circuit with 3 ancilla
qubits:
- 1,097,712 gates: all 1- and 2-body generators, plus 19,844 3-body generators, plus their ancilla copies;
- 1,000 random operators and 3,000 samples per step, two kernel bandwidths;
- float32.

**Setup.** One idle AMD Instinct MI300X, with the judge of the optimisation arena described below. The
reference was measured on 2026-11-09 and iqp_fast on 2026-11-10.

**What the clock covers.** For both columns the clock starts after the import and a short warm-up on a
tiny circuit. The one-time kernel compile at import, about 25 s, is therefore not included.

| | Reference (modified iqpopt, float32) | iqp_fast |
|---|---|---|
| Wall time: build circuit, compile, 2,500 steps | 150.89 s | 8.85 s (median of 3) |
| Speedup | 1× | **17.04×** |
| Time per step after compilation | ~50 ms | 1.52 ms |
| GPU memory peak | 8.79 GiB | 0.77 GiB |

**Circuits the code was not tuned on.** iqp_fast was also checked on three circuits it had never seen.
- Each one differs from the benchmark in size, data, number of ancillas or loss.
- Together they cover 3- and 4-body generators and the plain `mmd_loss_iqp`.

It passed the same correctness checks on each. It took 0.10× to 0.32× of the reference's time for 100
steps, including build and compile.

**Correctness.** At 9 training steps (8 drawn at random, plus the last), the loss and gradient used in
training were recomputed by the reference at the same parameters and key:
- loss: within 1e-5 relative;
- gradient: within 1e-4 of max|g|.

On five small circuits (1- to 4-body), the `op_expval` values and standard deviations were also compared,
within 1e-5 absolute.

The reference itself is float32 and not bitwise reproducible on the GPU, so these tolerances are about 20
times its own run-to-run spread. A different random stream would move the estimate by about 1e-2.

## Requirements

- Python ≥ 3.10 and JAX ≥ 0.6. Tested with jax 0.6.0 and the jax-rocm7 plugin.
- For the GPU kernels: a ROCm GPU and `hipcc`. Tested with HIP 7.0 on gfx942 (MI300X).
  - The kernels are compiled once, when the package is first imported. That takes about 25 s on an
    MI300X, and the result is cached in `$XDG_CACHE_HOME/iqpopt_kernels`.
  - They are used only for large estimates, where n_ops × gate groups × samples ≥ 5e9.
  - Without ROCm or `hipcc`, or for small circuits, iqp_fast uses XLA paths that give the same results.

## Install

```sh
pip install "git+https://github.com/pop756/iqp_fast.git"
pip install "iqp_fast[research] @ git+https://github.com/pop756/iqp_fast.git"   # plus the dependencies of randomized_iqp
```

## Usage

The API is iqpopt's, so existing code only changes its import:

```python
import jax, jax.numpy as jnp
import iqp_fast as iqpopt
from iqp_fast.gen_qml import mmd_loss_iqp
from iqp_fast.utils import local_gates

gates = local_gates(20, 3)
sim = iqpopt.IqpSimulator(20, gates)
params = jnp.full(len(gates), 0.1)
ops = jnp.eye(20, dtype=int)[:5]
expvals, stds = sim.op_expval(params, ops, 10_000, jax.random.PRNGKey(0))
```

`randomized_iqp.trainer.branch_weighted_mixture_mmd_loss` is the branch-weighted loss used in the
benchmark above.

The default precision is float32, while upstream iqpopt turns on float64 at import. Environment switches:

| Variable | Effect |
|---|---|
| `IQPOPT_X64=1` | float64 |
| `IQPOPT_FAST=0` or `IQPOPT_NO_KERNELS=1` | turns the GPU kernel path off |
| `IQPOPT_EXACT=1` | turns off the pairwise path and the kernel path; the grouped path stays on |

The smoke test checks the estimator against the exact expectation on a small circuit:

```sh
PYTHONPATH=. python tests/test_smoke.py
```

## Contents

- `iqp_fast/`: the estimator and the MMD loss (iqpopt with the new paths). The GPU kernels are in
  `iqp_fast/_kernels.hip` and are called through `jax.ffi` from `iqp_fast/_fastpath.py`.
- `randomized_iqp/`: training utilities for the mixed IQP-QCBM, including
  `branch_weighted_mixture_mmd_loss`.
  - The mixed IQP-QCBM writes an ancilla-extended IQP circuit as a weighted mixture of ancilla-free IQP
    circuits ("branches").
  - The model is described in Lee & Kim, arXiv:2607.27883.

## How it was made

The kernel path was written by Claude Opus 5.5 (Anthropic) in a 6-hour optimisation arena. It started
from a modified iqpopt and had to keep the estimator unchanged. A fixed judge scored it:
- the judge owned the training loop;
- it compared the loss and gradient with the reference at steps the code could not know in advance;
- it capped GPU memory;
- it ran circuits the code had never seen.

Before scoring, the full diff was read for rule violations, and none were found.

## License and attribution

Apache License 2.0; see `LICENSE` and `NOTICE`. iqp_fast is derived from iqpopt by Erik Recio-Armengol and
Joseph Bowles (https://github.com/XanaduAI/iqpopt). Files taken from iqpopt state their changes in their
headers.

If you use this software, please cite the iqpopt paper:

> E. Recio-Armengol, J. Bowles, *IQPopt: Fast optimization of instantaneous quantum polynomial circuits in
> JAX*, arXiv:2501.04776.

If you use `randomized_iqp`, the mixed IQP-QCBM and its branch-weighted MMD loss, please also cite:

> Y. Lee, H. Kim, *Trainability and Mode Separation of Mixed IQP-QCBMs*, arXiv:2607.27883 (2026).
> https://arxiv.org/abs/2607.27883
