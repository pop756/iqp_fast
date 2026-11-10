# Derived from iqpopt (https://github.com/XanaduAI/iqpopt), Apache License 2.0.
# Part of iqp_fast; see LICENSE and NOTICE.
# Modifications Copyright 2026 Youngseok Lee. Changes: float32 by default;
#   grouped and pairwise expectation paths; branch/label conditioning of op_expval and op_expval_batch
#   (condition_mask, condition_values, condition_counts) and ops_wires; vectorized generator construction;
#   sparse-op GPU kernel path.

from jax._src.typing import Array
from scipy.sparse import csr_matrix
import itertools
try:  # hash-based first-occurrence numbering of the generator groups (np.unique sorts; 5x slower)
    import pandas as _pd
except Exception:  # noqa: BLE001
    _pd = None
import pennylane as qml
import jax.numpy as jnp
import numpy as np
import os
import jax
# Precision: default float32. The hot path (parity matmuls, expectation products)
# is stable in float32 and runs fast on FP64-limited GPUs (NVIDIA RTX/workstation
# cards cripple float64 to ~1/64). Set IQPOPT_X64=1 to force float64.
if os.environ.get("IQPOPT_X64", "0") in ("1", "true", "True"):
    jax.config.update("jax_enable_x64", True)
# IQPOPT_EXACT=1 turns off the fast paths added on 2026-10-08 (_op_expval_pairs), so
# results reproduce earlier runs exactly; the default matches them to float32 rounding.
EXACT = os.environ.get("IQPOPT_EXACT", "0") in ("1", "true", "True")
# Sparse-op kernel path (iqp_fast/_fastpath.py) on ROCm GPUs; IQPOPT_FAST=0 turns it off.
FAST = os.environ.get("IQPOPT_FAST", "1") not in ("0", "false", "False")
from functools import cached_property
from . import _fastpath
if FAST:  # build / load the kernels at import (a generic library, not workload-shaped)
    _fastpath.available()

# Max number of wires outside ops_wires for the grouped expval path (its sign
# table has 2**h columns); beyond this op_expval_batch uses the dense path.
MAX_HIDDEN_WIRES = 8
# Largest (entries x samples) array of the CPU entry form of _pair_quad.
ENTRY_BUDGET = 2 ** 24
# op_expval takes _op_expval_pairs when n_groups >= PAIRS_MIN_GAIN * (n_visible + K^2):
# per op and sample, _phases costs n_groups and the pairwise path ~ n_visible + K^2.
PAIRS_MIN_GAIN = 4


def _wire_matrix(gens, n_qubits):
    """(n_generators, L) int: the sorted distinct wires of each generator, padded with n_qubits
    (the same wire sets as tuple(sorted({int(j) for j in gen})), built without a python loop per gen)."""
    n = len(gens)
    dt = np.int32 if n_qubits < 2 ** 31 - 1 else np.int64
    lens = np.fromiter(map(len, gens), dtype=np.int64, count=n)
    L = max(int(lens.max()) if n else 1, 1)
    flat = np.fromiter(itertools.chain.from_iterable(gens), dtype=np.int64, count=int(lens.sum())).astype(dt)
    starts = np.cumsum(lens) - lens
    M = np.full((n, L), n_qubits, dtype=dt)
    for j in range(L):  # column j: the j-th listed wire of the generators with more than j wires
        sel = np.flatnonzero(lens > j)
        M[sel, j] = flat[starts[sel] + j]
    M.sort(axis=1)
    dup = np.zeros(M.shape, dtype=bool)
    dup[:, 1:] = (M[:, 1:] == M[:, :-1]) & (M[:, 1:] < n_qubits)
    if dup.any():
        M[dup] = n_qubits
        M.sort(axis=1)
    return M


def _first_occurrence_ids(rows, base):
    """Ids of the distinct rows of an int matrix numbered by first occurrence, and the first index of each."""
    if len(rows) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if rows.shape[1] * np.log2(base) < 62:
        keys = np.zeros(len(rows), dtype=np.int64)
        for j in range(rows.shape[1]):
            keys = keys * base + rows[:, j]
        if _pd is not None:
            ids = np.asarray(_pd.factorize(keys)[0], dtype=np.int64)  # numbered by first appearance
            first = np.empty(int(ids.max()) + 1, dtype=np.int64)
            first[ids[::-1]] = np.arange(len(ids))[::-1]  # the last write is the first occurrence
            return ids, first
        _, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    else:
        _, first, inv = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank[inv.reshape(-1)], first[order]


def _gather_parity(bits_t, idx):
    """int8 parities, shape (len(idx),) + bits_t.shape[1:], of the 0/1 rows of
    bits_t selected by each row of idx. idx is padded with bits_t.shape[0],
    which points at an appended zero row."""
    bits_t = bits_t.astype(jnp.int8)
    bits_t = jnp.concatenate([bits_t, jnp.zeros((1,) + bits_t.shape[1:], jnp.int8)])
    par = bits_t[idx[:, 0]]
    for j in range(1, idx.shape[1]):
        par = par ^ bits_t[idx[:, j]]
    return par


class IqpSimulator:
    """ Class that creates an IqpSimulator object corresponding to a parameterized IQP circuit"""

    def __init__(self, n_qubits: int, gates: list, device: str = "lightning.qubit",
                 spin_sym: bool = False, init_gates: list = None,
                 sparse: bool = False, bitflip: bool = False):
        """
        Args:
            n_qubits (int): Total number of qubits of the circuit.
            gates (list[list[list[int]]]): Specification of the trainable gates. Each element of gates corresponds to a
                unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
                Generators are specified by listing the qubits on which an X operator acts.
            device (str, optional): Pennylane device used for calculating probabilities and sampling.
            spin_sym (bool, optional): If True, the circuit is equivalent to one where the initial state
                1/sqrt(2)(|00...0> + |11...1>) is used in place of |00...0>.
            init_gates (list[list[list[int]]], optional): A specification of gates of the same form as the gates argument. The
                parameters of these gates will be defined by init_coefs later on.
            sparse (bool, optional): If True, generators and ops are always stored in sparse matrix format, leading
                to better memory efficiency and potentially faster runtime.
            bitflip (bool, optional): If True, the circuit is equivalent to a classical stochastic model where the
                gates correspond to correlated bitflips.

        Raises:
            Exception: when gates and params have a different number of elements.
        """
        self.n_qubits = n_qubits
        self.gates = gates
        self.n_gates = len(gates)
        self.sparse = sparse
        self.init_gates = init_gates
        self.device = device
        self.spin_sym = spin_sym
        self.bitflip = bitflip

        self.generators_sp = None
        self._split_cache = {}

        # Wires of every generator (gates first, then init_gates), in the row
        # order of self.generators. Building these in python instead of one
        # device array per gate makes construction ~1000x faster for 1e5+ gates.
        all_gens = [gen for gate in gates for gen in gate]
        len_gen_init = 0
        if self.init_gates is not None:
            init_gens = [gen for gate in self.init_gates for gen in gate]
            len_gen_init = len(init_gens)
            all_gens += init_gens
        self._gw = _wire_matrix(all_gens, n_qubits)

        len_gen = len(self._gw)
        self.par_transform = False if max(map(len, self.gates)) == 1 and self.init_gates is None else True

        if sparse:
            rows, cols = np.nonzero(self._gw < n_qubits)
            cols = self._gw[rows, cols]
            self.generators_sp = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len_gen, n_qubits))

        if self.par_transform:
            # Transformation matrix from the number of independent parameters to the number of total generators
            self.trans_par = np.zeros((len_gen, len(gates)))
            i = 0
            for j, gens in enumerate(gates):
                for gen in gens:
                    # Matrix that linearly transforms the vector of parameters that are trained into the vector of parameters that apply to the generators
                    self.trans_par[i, j] = 1
                    i += 1
            self.trans_par = jnp.array(self.trans_par)

        if self.init_gates is not None:
            # Matrix that transforms the static parameters (initial coefficients) into a vector of size generators so it can be summed with the variational parameters
            self.trans_coef = np.zeros((len_gen, len(self.init_gates)))
            i = len_gen - len_gen_init
            for j, gens in enumerate(self.init_gates):
                for gen in gens:
                    self.trans_coef[i, j] = 1
                    i += 1
            self.trans_coef = jnp.array(self.trans_coef)

    @cached_property
    def _gen_wires(self):
        """Wires of every generator (gates first, then init_gates) as sorted tuples."""
        return [tuple(int(x) for x in row[row < self.n_qubits]) for row in self._gw]

    @cached_property
    def generators(self):
        """Dense (n_generators, n_qubits) 0/1 generator matrix, built on first use
        (only the bitflip and dense fallback paths need it). float32 is exact
        here and keeps XLA's host->device constant copy under the ROCm ~4GB limit.
        Kept as numpy: a jnp array created during a trace would be a tracer."""
        G = np.zeros((len(self._gw), self.n_qubits + 1), dtype=np.float32)
        G[np.arange(len(self._gw))[:, None], self._gw] = 1
        return np.ascontiguousarray(G[:, :self.n_qubits])

    def _ops_split(self, ops_wires):
        """Static data for _phases, cached per ops_wires.

        Ops are zero outside ops_wires ("visible" wires), so a generator's parity
        with an op only depends on its visible wires. Generators sharing the same
        visible part form one group, and their remaining h hidden wires enter only
        through the sign of the sample's hidden bits. The phase sum over all
        generators then becomes a sum over groups v of
            parity(op . v) * R[sample, v],   R = s_v(sample) * sum_m W[v, m] s_m(sample_hidden),
        with W[v, m] the summed angles of the generators in group v whose hidden
        wires are the bitmask m. Returns None if h > MAX_HIDDEN_WIRES.
        """
        key = None if ops_wires is None else tuple(sorted({int(w) for w in ops_wires}))
        if key not in self._split_cache:
            vis = np.ones(self.n_qubits, dtype=bool)
            if key is not None:
                vis[:] = False
                vis[list(key)] = True
            hidden = np.flatnonzero(~vis)
            split = None
            if len(hidden) <= MAX_HIDDEN_WIRES:
                nq = self.n_qubits
                M = self._gw
                visx = np.append(vis, False)  # the pad wire nq is neither visible nor hidden
                bit = np.zeros(nq + 1, dtype=np.int64)
                bit[hidden] = 1 << np.arange(len(hidden))
                hmask = bit[M].sum(axis=1)
                vmask = visx[M]
                nvis = vmask.sum(axis=1)
                V = np.where(vmask, M, nq)
                V.sort(axis=1)
                # generators without visible wires never contribute: their id is n_groups, which the
                # scatter drops; groups are numbered by first occurrence
                kmax = max(int(nvis.max()) if len(nvis) else 0, 1)
                V = V[:, :kmax]
                nonempty = nvis > 0
                ids, first = _first_occurrence_ids(V[nonempty], nq + 1)
                n_groups = len(first)
                gid = np.full(len(M), n_groups, dtype=np.int32)
                gid[nonempty] = ids
                vidx = V[nonempty][first].astype(np.int32).reshape(n_groups, kmax)
                pairs = None
                if kmax <= 2:
                    # tables for _op_expval_pairs, indexed by position among the visible
                    # wires; n_groups points at an appended zero angle
                    vis_w = np.flatnonzero(vis)
                    pos = np.zeros(nq + 1, dtype=np.int64)
                    pos[vis_w] = np.arange(len(vis_w))
                    size = (vidx < nq).sum(axis=1)
                    single = np.full(len(vis_w), n_groups, dtype=np.int32)
                    ptab = np.full((len(vis_w), len(vis_w)), n_groups, dtype=np.int32)
                    g1 = np.flatnonzero(size == 1)
                    single[pos[vidx[g1, 0]]] = g1
                    g2 = np.flatnonzero(size == 2)
                    pa, pb = pos[vidx[g2, 0]], pos[vidx[g2, 1]]
                    ptab[pa, pb] = g2
                    ptab[pb, pa] = g2
                    pairs = dict(vis=vis_w.astype(np.int32), single=single, ptab=ptab, pgid=g2.astype(np.int32),
                                 pi=pa.astype(np.int32), pj=pb.astype(np.int32))
                split = dict(
                    vidx=vidx, gid=gid, hmask=hmask.astype(np.int32),
                    hidden=hidden.astype(np.int32), n_groups=n_groups, pairs=pairs,
                    mbits=((np.arange(2 ** len(hidden))[:, None] >> np.arange(len(hidden))) & 1).astype(np.float32))
            self._split_cache[key] = split
        return self._split_cache[key]

    def _phases(self, split, effective_params, ops, samples):
        """phases[i, k] = 2 sum_g theta_g parity(ops_i . g) (-1)^parity(samples_k . g),
        the cos argument of the dense path, computed with the _ops_split data."""
        W = jnp.zeros((split["n_groups"], split["mbits"].shape[0]), jnp.float32).at[
            split["gid"], split["hmask"]].add(effective_params.astype(jnp.float32), mode="drop")
        hidden_signs = 1 - 2 * ((split["mbits"] @ samples[:, split["hidden"]].T.astype(jnp.float32)) % 2)
        # (n_groups, n_samples) layout: the matmuls below then avoid a transposed
        # right operand, which is ~2x faster on CPU (same on GPU). HIGHEST keeps
        # ROCm/TF32-class GPUs from rounding the angles (z error 1e-4 -> 1e-7 on
        # MNIST a=4) at no measurable cost.
        hi = jax.lax.Precision.HIGHEST
        RT = (1 - 2 * _gather_parity(samples.T, split["vidx"])).astype(jnp.float32) * jnp.matmul(W, hidden_signs, precision=hi)
        return jnp.matmul(2 * _gather_parity(ops.T, split["vidx"]).astype(jnp.float32).T, RT, precision=hi)

    @staticmethod
    def _segments(split, n_samples, condition_mask, condition_values, condition_counts):
        """Static runs (start, count, hidden bits) of the sample slots on which every
        hidden wire is fixed, or None. Hidden wires are fixed only when conditioned, and
        reading which ones are needs the condition arrays as numpy."""
        hidden = split["hidden"]
        mask, values = condition_mask, condition_values
        conditioned = mask is not None and values is not None
        if conditioned and getattr(mask, "ndim", None) == 2:
            # leave malformed batched conditions to the checks of the chunked paths
            if condition_counts is None or isinstance(condition_counts, jax.core.Tracer):
                return None
            counts = np.asarray(condition_counts).astype(np.int64)
            if (counts.ndim != 1 or len(counts) != mask.shape[0] or np.any(counts < 0)
                    or counts.sum() != n_samples or getattr(values, "shape", None) != mask.shape):
                return None
        if len(hidden) == 0:
            return ((0, int(n_samples), ()),)
        if not (conditioned and isinstance(mask, np.ndarray) and isinstance(values, np.ndarray)):
            return None
        mask = mask.astype(bool)
        if mask.ndim == 1:
            return ((0, int(n_samples), tuple(int(b) for b in values[hidden])),) if mask[hidden].all() else None
        active = np.flatnonzero(counts)
        if not all(mask[b, hidden].all() for b in active):
            return None
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        return tuple((int(starts[b]), int(counts[b]), tuple(int(x) for x in values[b, hidden])) for b in active)

    def _fast_route(self, n_ops, n_samples, indep_estimates, condition_mask, condition_values, condition_counts,
                    ops_wires):
        """(split, segments) when op_expval takes the kernel path (iqp_fast/_fastpath.py) for these arguments,
        else None."""
        if EXACT or not FAST or self.bitflip or self.sparse or self.spin_sym or indep_estimates:
            return None
        split = self._ops_split(ops_wires)
        if (split is None or not _fastpath.supported(self, split) or not _fastpath.worth_it(split, n_ops, n_samples)
                or not _fastpath.available()):
            return None
        segments = self._segments(split, n_samples, condition_mask, condition_values, condition_counts)
        return None if segments is None else (split, segments)

    def _draw_samples(self, key, n_samples, max_batch_samples, condition_mask, condition_values, condition_counts):
        """The conditioned samples of one op batch and the next key, exactly as the
        chunked paths of op_expval / op_expval_batch draw them."""
        nq = self.n_qubits
        mbs = n_samples if max_batch_samples is None else int(max_batch_samples)
        if mbs < n_samples and n_samples % mbs == 0:  # the lax.scan sample path
            keys = jax.random.split(key, n_samples // mbs + 1)
            samples = jnp.concatenate([jax.random.randint(k, (mbs, nq), 0, 2) for k in keys[:-1]])
            key = keys[-1]
        else:
            parts = []
            for i in range(int(np.ceil(n_samples / mbs))):
                key, subkey = jax.random.split(key, 2)
                parts.append(jax.random.randint(subkey, (min(mbs, n_samples - i * mbs), nq), 0, 2))
            samples = jnp.concatenate(parts)
        if condition_mask is not None and condition_values is not None:
            mask = jnp.asarray(condition_mask, dtype=bool)
            values = jnp.asarray(condition_values, dtype=samples.dtype)
            if mask.ndim == 2:  # slot k takes the branch that the count layout gives it
                slot = np.repeat(np.arange(len(condition_counts)), np.asarray(condition_counts).astype(np.int64))
                mask, values = mask[slot], values[slot]
            samples = jnp.where(mask, values, samples)
        return samples, key

    def _op_expval_pairs(self, split, segments, params, ops, n_samples, key, init_coefs, return_samples,
                         max_batch_ops, max_batch_samples, condition_mask, condition_values, condition_counts):
        """op_expval for circuits whose generators have at most two visible wires, with
        the hidden bits fixed on static segments of the sample slots (no hidden wires,
        or every hidden wire conditioned). Same samples and estimator as the chunked
        paths. With s = 1 - 2 z on the visible wires, theta the segment's group angles
        (W summed over the hidden masks with the segment's hidden signs) and
        parity(A . (i,j)) = A_i + A_j - 2 A_i A_j:
            phase(A, z) = 2 sum_i A_i L_i(z) - 4 sum_{pair groups (i,j) inside A} s_i s_j theta_ij,
            L_i(z) = s_i (theta_i + sum_j theta_ij s_j).
        The (n_ops x n_groups x n_samples) product of _phases becomes an
        (n_samples x n_v x n_v) one plus a sum over the few pair groups inside each op
        (the MMD ops of large circuits are sparse). Memory no longer grows with
        n_groups, so the samples are not chunked."""
        p = split["pairs"]
        hi = jax.lax.Precision.HIGHEST
        n_groups = split["n_groups"]
        effective_params = self.trans_par @ params if self.par_transform else params
        effective_params = effective_params + self.trans_coef @ init_coefs if self.init_gates is not None else effective_params
        W = jnp.zeros((n_groups, split["mbits"].shape[0]), jnp.float32).at[
            split["gid"], split["hmask"]].add(effective_params.astype(jnp.float32), mode="drop")
        bits = np.asarray([seg[2] for seg in segments], dtype=np.float32).reshape(len(segments), -1)
        hsign = 1 - 2 * ((bits @ split["mbits"].T) % 2)  # (n_seg, 2**h)
        theta = jnp.concatenate([jnp.matmul(W, jnp.asarray(hsign.T), precision=hi),
                                 jnp.zeros((1, len(segments)), jnp.float32)])  # (n_groups + 1, n_seg); last row 0
        th1 = theta[p["single"]]
        # scattered from the pair list: the gradient then gathers n_pairs entries instead
        # of scatter-adding n_v^2 (single-threaded on CPU)
        nv = len(p["vis"])
        th2 = [jnp.zeros((nv, nv), jnp.float32).at[p["pi"], p["pj"]].set(t).at[p["pj"], p["pi"]].set(t)
               for t in theta[p["pgid"]].T]
        expvals = []
        for batch_ops in jnp.array_split(ops, np.ceil(ops.shape[0] / (max_batch_ops or ops.shape[0]))):
            samples, key = self._draw_samples(key, n_samples, max_batch_samples, condition_mask,
                                              condition_values, condition_counts)
            s = (1 - 2 * samples[:, p["vis"]]).astype(jnp.float32)
            A = batch_ops[:, p["vis"]].astype(jnp.float32)
            L = jnp.concatenate([s[a:a + c] * (th1[:, k] + jnp.matmul(s[a:a + c], th2[k], precision=hi))
                                 for k, (a, c, _) in enumerate(segments)])
            phases = 2 * jnp.matmul(A, L.T, precision=hi) - 4 * self._pair_quad(p, n_groups, A, s, theta, segments)
            # independent row blocks: XLA's CPU runtime runs separate elementwise ops in
            # parallel, but each (n_ops x n_samples) one on a single thread
            expvals += [jnp.cos(ph) for ph in jnp.array_split(phases, max(1, min(16, phases.shape[0] // 32)))]
        expvals = jnp.concatenate(expvals)
        if return_samples:
            return expvals
        return jnp.mean(expvals, axis=-1), jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)

    @staticmethod
    def _pair_quad(p, n_groups, A, s, theta, segments, K=20):
        """quad[op, z] = sum over the pair groups (i,j) inside op of s_i s_j theta_ij of z's segment.

        Support form: each op's (at most K) wires, padded, and one K x K bilinear form
        per op and segment (batched matmuls). Entry form: the list of (op, pair group)
        entries with a static size from a few buckets, gathered and summed per op.
        GPUs run the support form (a data-dependent branch costs a host sync per call,
        and segment sums become atomics); CPUs list the entries when they are few
        (small batched matmuls are slow there) and use the support form, in op chunks,
        otherwise. Ops with more than K wires use the pair-chunked scan."""
        n_ops, nv = A.shape
        ns = s.shape[0]
        K = min(nv, K)
        hi = jax.lax.Precision.HIGHEST
        vals, idx = jax.lax.top_k(A, K)
        idx = jnp.where(vals > 0, idx, nv)  # nv: padding, an appended zero wire
        ptab = jnp.asarray(np.pad(p["ptab"], ((0, 1), (0, 1)), constant_values=n_groups))
        sp = jnp.concatenate([s, jnp.zeros((ns, 1), s.dtype)], axis=1)
        seg_of = np.repeat(np.arange(len(segments)), [c for _, c, _ in segments])
        on_cpu = jax.default_backend() == "cpu"

        def support(idx):
            G = ptab[idx[:, :, None], idx[:, None, :]]  # (ops, K, K)
            sS = jnp.transpose(sp[:, idx], (1, 0, 2))  # (ops, ns, K)
            return jnp.concatenate([
                0.5 * jnp.sum(sS[:, a:a + c] * jnp.einsum("oza,oab->ozb", sS[:, a:a + c], theta[G, k], precision=hi), axis=-1)
                for k, (a, c, _) in enumerate(segments)], axis=1)

        def support_chunked(theta, s, chunk=64):
            n_ch = -(-n_ops // chunk)
            pad = jnp.full((n_ch * chunk - n_ops, K), nv, idx.dtype)
            out = jax.lax.map(jax.checkpoint(support), jnp.concatenate([idx, pad]).reshape(n_ch, chunk, K))
            return out.reshape(n_ch * chunk, ns)[:n_ops]

        def pairs_scan(theta, s):
            chunk = 1 << max(0, min(11, (2 ** 22 // ns).bit_length() - 1))  # (chunk x ns) <= 2^22
            n_ch = -(-len(p["pgid"]) // chunk)
            pad = n_ch * chunk - len(p["pgid"])
            g, i, j = (jnp.asarray(np.pad(x, (0, pad), constant_values=c)).reshape(n_ch, chunk)
                       for x, c in ((p["pgid"], n_groups), (p["pi"], 0), (p["pj"], 0)))
            onehot = jax.nn.one_hot(seg_of, len(segments), dtype=jnp.float32).T

            def body(acc, x):
                g, i, j = x
                R = (s[:, i] * s[:, j]).T * jnp.matmul(theta[g], onehot, precision=hi)
                return acc + jnp.matmul(A[:, i] * A[:, j], R, precision=hi), None
            return jax.lax.scan(jax.checkpoint(body), jnp.zeros((n_ops, ns), jnp.float32), (g, i, j))[0]

        overflow = jnp.max(jnp.sum(A, axis=1)) > K
        if not on_cpu:
            # computed outside the cond: kernels inside a branch run ~10x slower on ROCm
            q = support(idx)
            return jax.lax.cond(overflow, pairs_scan, lambda theta, s: q, theta, s)

        ia, ib = np.triu_indices(K, 1)
        C = len(ia)
        g = ptab[idx[:, ia], idx[:, ib]]
        valid = (g < n_groups).reshape(-1)
        nnz = jnp.sum(valid)
        sizes = [z for z in (256 << j for j in range(12)) if z * ns <= ENTRY_BUDGET and z < n_ops * C]
        ia, ib = jnp.asarray(ia), jnp.asarray(ib)
        onehot = jax.nn.one_hot(seg_of, len(segments), dtype=jnp.float32).T
        sT = s.T

        def entries(size):
            def f(theta, s):
                flat = jnp.nonzero(valid, size=size, fill_value=valid.shape[0] - 1)[0]
                op, c = flat // C, flat % C
                m = (jnp.arange(size) < nnz).astype(jnp.float32)[:, None]
                th = jnp.matmul(theta[g.reshape(-1)[flat]] * m, onehot, precision=hi)  # (size, ns)
                q = sT[idx[op, ia[c]]] * sT[idx[op, ib[c]]] * th
                return jax.ops.segment_sum(q, op, num_segments=n_ops, indices_are_sorted=True)
            return jax.checkpoint(f)

        branch = jnp.where(overflow, len(sizes) + 1, jnp.searchsorted(jnp.asarray(sizes, dtype=jnp.int32), nnz))
        return jax.lax.switch(branch, [entries(z) for z in sizes] + [support_chunked, pairs_scan], theta, s)

    def iqp_circuit(self, params: jnp.ndarray, init_coefs: list = None):
        """IQP circuit in pennylane form.

        Args:
            params (jnp.ndarray): The parameters of the IQP gates.
            init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.
        """

        if self.spin_sym:
            qml.PauliRot(2*jnp.pi/4, "Y"+"X"*(self.n_qubits-1), wires=range(self.n_qubits))

        for i in range(self.n_qubits):
            qml.Hadamard(i)

        if self.init_gates is not None:
            for par, gate in zip(init_coefs, self.init_gates):
                for gen in gate:
                    qml.MultiRZ(2*par, wires=gen)

        for par, gate in zip(params, self.gates):
            for gen in gate:
                qml.MultiRZ(2*par, wires=gen)

        for i in range(self.n_qubits):
            qml.Hadamard(i)

    def sample(self, params: jnp.ndarray, init_coefs: list = None, shots: int = 1) -> jnp.ndarray:
        """Sample the IQP circuit using state vector simulation in Pennylane.
        Only possible for circuits with small numbers of qubits.

        Args:
            params (jnp.ndarray): The parameters of the IQP gates.
            init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.
            shots (int): Number of samples that are output. Defaults to 1.

        Returns:
            jnp.ndarray: The bitstring samples.
        """
        if self.bitflip:
            samples = []
            for _ in range(shots):
                sample = np.zeros(self.n_qubits)
                if self.spin_sym:
                    sample = (sample+1) % 2 if np.random.rand() > 0.5 else sample

                if self.init_gates is not None:
                    for par, gate in zip(init_coefs, self.init_gates):
                        for gen in gate:
                            if np.random.rand() < jnp.sin(par) ** 2:
                                sample[gen] = sample[gen]+1
                                sample = sample % 2

                for par, gate in zip(params, self.gates):
                    for gen in gate:
                        if np.random.rand() < jnp.sin(par) ** 2:
                            sample[gen] = sample[gen] + 1
                            sample = sample % 2
                samples.append(sample)
            return jnp.array(samples)

        else:
            dev = qml.device(self.device, wires=self.n_qubits)

            @qml.set_shots(shots)
            @qml.qnode(dev)
            def sample_circuit(params):
                self.iqp_circuit(params, init_coefs)
                return qml.sample(wires=range(self.n_qubits))
            return sample_circuit(params)

    def probs(self, params: jnp.ndarray, init_coefs: list = None) -> jnp.ndarray:
        """Returns the probabilities of all possible bitstrings using state vector simulation in
        PennyLane. Only possible for circuits with small numbers of qubits.

        Args:
            params (jnp.ndarray): The parameters of the IQP gates.
            init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.

        Returns:
            jnp.ndarray: Probabilities of all possible bitstrings.
        """

        if self.bitflip:
            raise NotImplementedError(
                "probs not implemented for bitflip circuits")
        dev = qml.device(self.device, wires=self.n_qubits)

        @qml.qnode(dev)
        def probs_circuit(params):
            self.iqp_circuit(params, init_coefs)
            return qml.probs(wires=range(self.n_qubits))
        return probs_circuit(params)

    def __op_expval_indep(self, params: jnp.ndarray, ops: jnp.ndarray,
                          n_samples: int, key: Array, return_samples, init_coefs: list = None,
                          condition_mask: jnp.ndarray = None,
                          condition_values: jnp.ndarray = None) -> list:
        """
        Batch evaluate an array of ops in the same way as self.op_expval_batch, but using independent randomness
        for each estimator. The estimators for each op are therefore uncorrelated.
        """

        def update(carry, op):
            key1, key2 = jax.random.split(carry, 2)
            expval = self.op_expval_batch(
                params, op, n_samples, key1, init_coefs, False, return_samples,
                condition_mask=condition_mask,
                condition_values=condition_values)
            return key2, expval

        if self.sparse:
            expvals = []
            stds = []
            for op in ops:
                key, val = update(key, op)
                if return_samples:
                    expvals.append(val[0])
                else:
                    expvals.append(val[0][0])
                    stds.append(val[1][0])

            if return_samples:
                return jnp.array(expvals)
            else:
                return jnp.array(expvals), jnp.array(stds)/jnp.sqrt(n_samples)

        else:
            _, op_expvals = jax.lax.scan(update, key, ops)

            if return_samples:
                return op_expvals
            else:
                return op_expvals[0], op_expvals[1]

    def op_expval_batch(self, params: jnp.ndarray, ops: jnp.ndarray, n_samples: int,
                        key: Array, init_coefs: list = None, indep_estimates: bool = False,
                        return_samples: bool = False,
                        condition_mask: jnp.ndarray = None,
                        condition_values: jnp.ndarray = None,
                        condition_counts: jnp.ndarray = None,
                        ops_wires: list = None) -> list:
        """Estimate the expectation values of a batch of Pauli-Z type operators. A set of l operators must be specified
        by an array of shape (l,n_qubits), where each row is a binary vector that specifies on which qubit a Pauli Z
        operator acts.
        The expectation values are estimated using a randomized method whose precision in controlled by n_samples,
        with larger values giving higher precision. Estimates are unbiased, however may be correlated. To request
        uncorrelated estimate, use indep_estimates=True at the cost of larger runtime.

        Args:
            params (jnp.ndarray): The parameters of the IQP circuit.
            ops (jnp.ndarray): Operator/s for those we want to know the expected value.
            n_samples (int): Number of samples used to calculate the IQP expectation value.
            key (Array): Jax key to control the randomness of the process.
            init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.
            indep_estimates (bool): Whether to use independent estimates of the ops in a batch (takes longer).
            return_samples (bool): if True, an extended array that contains the values of the estimator for each
                of the n_samples samples is returned.
            condition_counts (jnp.ndarray, optional): Per-branch sample counts for batched branch conditioning.
            ops_wires (list, optional): Wires on which ops can be nonzero (e.g. the visible wires of an MMD loss).
                Exact; lets wires outside it be folded into a small sign table. None means all wires.

        Returns:
            list: List of Vectors. The expectation values are estimated of the ops and their standard deviation.
        """
        multi_condition = False
        if condition_mask is not None and condition_values is not None:
            condition_mask = jnp.asarray(condition_mask)
            condition_values = jnp.asarray(condition_values)

            if condition_mask.ndim == 2:
                multi_condition = True
                if condition_counts is None:
                    raise ValueError("condition_counts must be provided when passing batched condition masks")
                if condition_values.ndim != 2 or condition_values.shape != condition_mask.shape:
                    raise ValueError("condition_values must have the same shape as condition_mask")
                condition_mask = condition_mask.astype(bool)
                if isinstance(condition_counts, (list, tuple, np.ndarray)):
                    condition_counts_np = None
                    try:
                        condition_counts_np = np.asarray(condition_counts)
                    except Exception:
                        condition_counts_np = None
                    if condition_counts_np is not None:
                        if condition_counts_np.ndim != 1:
                            raise ValueError("condition_counts must be one-dimensional")
                        if condition_counts_np.size != condition_mask.shape[0]:
                            raise ValueError("condition_counts and condition_mask batch sizes must match")
                        # Zero counts are permitted: a branch with no samples in a
                        # chunk contributes nothing (apply_branch's valid-mask skips
                        # it). This keeps the per-chunk shape fixed for the scan
                        # sample-batching path. Only the total must match.
                        if np.any(condition_counts_np < 0):
                            raise ValueError("condition_counts must be non-negative")
                        if int(condition_counts_np.sum()) != int(n_samples):
                            raise ValueError("sum(condition_counts) must equal n_samples")
                condition_counts = jnp.asarray(condition_counts)
                if condition_counts.ndim != 1:
                    raise ValueError("condition_counts must be one-dimensional")
                if condition_counts.shape[0] != condition_mask.shape[0]:
                    raise ValueError("condition_counts and condition_mask batch sizes must match")
                if not isinstance(condition_counts, jax.core.Tracer):
                    condition_counts_np = np.asarray(condition_counts)
                    if np.any(condition_counts_np < 0):
                        raise ValueError("condition_counts must be non-negative")
                    if int(condition_counts_np.sum()) != int(n_samples):
                        raise ValueError("sum(condition_counts) must equal n_samples")
            elif condition_mask.ndim == 1:
                condition_mask = condition_mask.astype(bool)
                condition_counts = None
            else:
                raise ValueError("condition_mask must be one- or two-dimensional")
        else:
            condition_counts = None

        if indep_estimates and not self.bitflip and not multi_condition:
            return self.__op_expval_indep(
                params, ops, n_samples, key, return_samples, init_coefs,
                condition_mask=condition_mask,
                condition_values=condition_values,
            )

        samples = jax.random.randint(key, (n_samples, self.n_qubits), 0, 2)
        if multi_condition:
            # Direct per-slot branch conditioning (fixes the sample-chunking
            # variance bug). Assign each of the n_samples output slots to a branch
            # via the cumulative condition_counts, then HARD-SET that slot's
            # conditioned qubits to the branch's pattern. This replaces the old
            # reject-and-bootstrap apply_branch path below, which kept only the
            # ~n/2^a probes whose random conditioned bits happened to match a
            # branch -- so under max_batch_samples chunking the distinct-probe pool
            # per branch shrank ~chunk-fold and the gradient-estimator variance
            # exploded (cluster/label training plateaued). With direct conditioning
            # every branch gets its full quota of FRESH independent probes
            # regardless of batching, so the estimate is chunk-invariant.
            _edges = jnp.cumsum(jnp.asarray(condition_counts))
            _slot = jnp.arange(n_samples, dtype=jnp.int32)
            _slot_branch = jnp.clip(
                jnp.searchsorted(_edges, _slot, side="right"),
                0, condition_mask.shape[0] - 1,
            )
            _smask = jnp.asarray(condition_mask, dtype=bool)[_slot_branch]
            _svals = jnp.asarray(condition_values, dtype=samples.dtype)[_slot_branch]
            samples = jnp.where(_smask, _svals, samples)
        match_samples = samples
        if condition_mask is not None and condition_values is not None and not multi_condition:
            condition_mask = jnp.asarray(condition_mask, dtype=bool)
            condition_values = jnp.asarray(condition_values, dtype=samples.dtype)
            samples = jnp.where(condition_mask[None, :], condition_values[None, :], samples)

        effective_params = self.trans_par @ params if self.par_transform else params
        effective_params = effective_params + self.trans_coef @ init_coefs if self.init_gates is not None else effective_params

        if self.bitflip:

            use_sparse = (self.sparse or isinstance(ops, csr_matrix)) and not isinstance(ops, jax.core.Tracer)

            if use_sparse:

                if isinstance(ops, csr_matrix):
                    if self.generators_sp is None:
                        self.generators_sp = csr_matrix(self.generators)

                else:
                    ops = csr_matrix(ops)

                ops_gen = ops.dot(self.generators_sp.T)
                
                if self.spin_sym:
                    ops_sum = np.squeeze(np.asarray(ops.sum(axis=-1)))
                    
                del ops

                ops_gen.data %= 2
                ops_gen = ops_gen.toarray().astype(np.float32)

            else:
                # float32 parity intermediate (see dense-path note below).
                ops_gen = ((ops @ self.generators.T) % 2).astype(jnp.float32)
                if self.spin_sym:
                    ops_sum = jnp.sum(ops, axis=-1)

            par_ops_gates = (2 * effective_params.astype(jnp.float32) * ops_gen).astype(jnp.float32)

            expvals = jnp.prod(jnp.cos(par_ops_gates), axis=-1)

            if self.spin_sym:
                # flip expvals of odd operators with prob 1/2
                odd_ops = 1 - 2 * (ops_sum % 2)
                expvals = 0.5*expvals + 0.5*odd_ops*expvals

            if multi_condition:
                expvals = jnp.expand_dims(expvals, -1)
                expvals = jnp.tile(expvals, (1, n_samples))
            else:
                if return_samples:
                    return jnp.expand_dims(expvals, -1)
                else:
                    return expvals, jnp.zeros(ops.shape[0])


        else:
            use_sparse = self.sparse or isinstance(ops, csr_matrix)
            use_sparse = use_sparse and not isinstance(ops, jax.core.Tracer)
            use_sparse = use_sparse and not isinstance(samples, jax.core.Tracer)

        if (not self.bitflip) and use_sparse:

            if isinstance(ops, csr_matrix):
                samples = csr_matrix(samples)
                if self.generators_sp is None:
                    self.generators_sp = csr_matrix(self.generators)
            else:
                ops = csr_matrix(ops)
                samples = csr_matrix(samples)

            ops_gen = ops.dot(self.generators_sp.T)
            ops_gen.data %= 2
            ops_gen = ops_gen.toarray().astype(np.float32)

            samples_gates = samples.dot(self.generators_sp.T)
            samples_gates.data = 2 * (samples_gates.data % 2)
            samples_gates = samples_gates.toarray()
            samples_gates = (1 - samples_gates).astype(np.float32)
            par_ops_gates = (2 * effective_params.astype(jnp.float32) * ops_gen).astype(jnp.float32)
            phases = par_ops_gates @ samples_gates.T

            if self.spin_sym:
                ops_sum = np.squeeze(np.asarray(ops.sum(axis=-1)))
                samples_sum = np.squeeze(np.asarray(samples.sum(axis=-1)))
                samples_len = samples.shape[0]
            del ops            
            del samples

        else:
            # ROCm memory fix: keep the large parity intermediates in float32.
            # These are exact in float32 (generators are binary 0/1, so the
            # modular matmuls and (...%2) results are 0/1, and samples_gates is
            # in {-1,+1}); float32 halves the peak VRAM of the big (n_ops x
            # n_gates) / (n_samples x n_gates) arrays for the D-Wave joint
            # fully-connected fine-tune (e.g. the 45 GB samples_gates at
            # 488 qubits / 1.88M gates). Mirrors the existing generators-float32
            # change which already accepted float32 here.
            split = self._ops_split(ops_wires)
            if split is not None:
                # Same phases as the dense matmuls below, without the
                # (n x n_qubits) @ (n_qubits x n_generators) products; see _phases.
                phases = self._phases(split, effective_params, ops, samples)
            else:
                ops_gen = ((ops @ self.generators.T) % 2).astype(jnp.float32)
                samples_gates = (1 - 2 * ((samples @ self.generators.T) % 2)).astype(jnp.float32)
                par_ops_gates = (2 * effective_params.astype(jnp.float32) * ops_gen).astype(jnp.float32)
                phases = par_ops_gates @ samples_gates.T
            if self.spin_sym:
                ops_sum = ops.sum(axis=-1)
                samples_sum = samples.sum(axis=-1)
                samples_len = samples.shape[0]

        if self.spin_sym:
            try:
                shape = (len(ops_sum), samples_len)
            except:
                shape = (samples_len, )

            ini_spin_sym = (2 - jnp.repeat(ops_sum, samples_len).reshape(shape) % 2 - 2*(samples_sum % 2)).astype(jnp.float32)

        else:
            ini_spin_sym = jnp.float32(1)
        # ini_spin_sym = jnp.where(self.spin_sym, ini_spin_sym, 1)

        # phases are float32 so the cos(...) expvals stay float32 (the cos
        # argument is bounded, so float32 is fine here).
        expvals = ini_spin_sym * jnp.cos(phases)

        if multi_condition:
            if n_samples <= 0:
                raise ValueError("n_samples must be positive")

            if expvals.ndim == 1:
                expvals = jnp.expand_dims(expvals, -1)
                expvals = jnp.tile(expvals, (1, n_samples))

            # expvals[:, j] is already the estimate for slot j's branch: the
            # conditioned qubits were hard-set per slot (above, right after the
            # random draw) before the expval computation, so no reject/resample is
            # needed. Each branch contributes condition_counts[branch] consecutive
            # FRESH probes -> chunk-invariant, low-variance estimate.
            if return_samples:
                return expvals
            return jnp.mean(expvals, axis=-1), jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(expvals.shape[1])


        if return_samples:
            return expvals
        else:
            return jnp.mean(expvals, axis=-1), jnp.std(expvals, axis=-1, ddof=1)/jnp.sqrt(n_samples)

    def op_expval(self, params: jnp.ndarray, ops: jnp.ndarray, n_samples: int, key: Array, init_coefs: list = None,
                  indep_estimates: bool = False, return_samples: bool = False,
                  max_batch_ops: int = None, max_batch_samples: int = None,
                  condition_mask: jnp.ndarray = None,
                  condition_values: jnp.ndarray = None,
                  condition_counts: jnp.ndarray = None,
                  ops_wires: list = None) -> list:
        """Estimate the expectation values of a batch of Pauli-Z type operators. A set of l operators must be specified
        by an array of shape (l,n_qubits), where each row is a binary vector that specifies on which qubit a Pauli Z
        operator acts.
        The expectation values are estimated using a randomized method whose precision in controlled by n_samples,
        with larger values giving higher precision. Estimates are unbiased, however may be correlated. To request
        uncorrelated estimate, use indep_estimates=True at the cost of larger runtime.
        For large batches of operators or large values of n_samples, memory can be controlled by setting max_batch_ops
        and/or max_batch_samples to a fixed value.

        Args:
            params (jnp.ndarray): The parameters of the trainable gates of the circuit.
            ops (jnp.ndarray): Array specifying the operator/s for which to estimate the expectation values.
            n_samples (int): Number of samples used to calculate the IQP expectation values. Higher values result in
                higher precision.
            key (Array): Jax key to control the randomness of the process.
            init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.
            indep_estimates (bool): Whether to use independent estimates of the ops in a batch.
            return_samples (bool): if True, an extended array that contains the values of the estimator for each
                of the n_samples samples is returned.
            max_batch_ops (int): Maximum number of operators in a batch. Defaults to None, which means taking all ops at once.
                Can only be used if ops is a jnp.array
            max_batch_samples (int): Maximum number of samples in a batch. Defaults to None, which means taking all n_samples at once.
            ops_wires (list, optional): Wires on which ops can be nonzero; see op_expval_batch.

        Returns:
            list: List of Vectors. The expected value of each op and its standard deviation.
        """

        # do not batch ops if ops is sparse
        if isinstance(ops, csr_matrix):
            return self.op_expval_batch(
                    params, ops, n_samples, key, init_coefs, indep_estimates, return_samples,
                    condition_mask=condition_mask,
                    condition_values=condition_values,
                    condition_counts=condition_counts, ops_wires=ops_wires)

        if not (EXACT or self.bitflip or self.sparse or self.spin_sym or indep_estimates):
            route = self._fast_route(ops.reshape(-1, ops.shape[-1]).shape[0], n_samples, indep_estimates,
                                     condition_mask, condition_values, condition_counts, ops_wires)
            if route is not None:
                return _fastpath.op_expval_fast(
                    self, route[0], route[1], params, ops.reshape(-1, ops.shape[-1]), n_samples, key,
                    None if init_coefs is None else jnp.asarray(init_coefs), return_samples, max_batch_ops,
                    max_batch_samples, condition_mask, condition_values, condition_counts)
            split = self._ops_split(ops_wires)
            nv = 0 if split is None or split["pairs"] is None else len(split["pairs"]["vis"])
            if nv and split["n_groups"] >= PAIRS_MIN_GAIN * (nv + min(nv, 20) ** 2):
                segments = self._segments(split, n_samples, condition_mask, condition_values, condition_counts)
                if segments is not None:
                    return self._op_expval_pairs(
                        split, segments, params, ops.reshape(-1, ops.shape[-1]), n_samples, key,
                        None if init_coefs is None else jnp.asarray(init_coefs), return_samples, max_batch_ops,
                        max_batch_samples, condition_mask, condition_values, condition_counts)

        if max_batch_ops is None:
            max_batch_ops = len(ops)

        if max_batch_samples is None:
            max_batch_samples = n_samples

        if self.bitflip:
            n_samples = max_batch_samples

        if len(ops.shape) == 1:
            ops = ops.reshape(1, -1)

        if self.bitflip:
            expvals = jnp.empty((0, 1))
        else:
            expvals = jnp.empty((0, n_samples))

        init_coefs = jnp.array(init_coefs) if init_coefs is not None else None

        # Detect the multi-branch (cluster-weighted) conditioning path: a 2-D
        # condition_mask with per-branch counts. In that case the sample-batch
        # loop below must hand op_expval_batch a per-batch condition_counts that
        # (a) sums to that batch's batch_n_samples and (b) preserves per-branch
        # totals across all batches; otherwise op_expval_batch raises
        # "sum(condition_counts) must equal n_samples". See plan/root-cause notes.
        # NOTE: condition_mask may be a JAX tracer (the loss is traced under
        # jax.grad / lax.scan), so use the .ndim attribute directly -- np.asarray
        # on a tracer raises TracerArrayConversionError. The per-branch sample
        # counts, by contrast, are static (built from python ints) and are
        # required concretely below for the allocation.
        multi_condition = (
            condition_mask is not None
            and condition_values is not None
            and getattr(condition_mask, "ndim", None) == 2
        )

        if multi_condition:
            # Materialize the consecutive per-branch allocation once. Concretely,
            # lay the per-branch counts end-to-end as a length-n_samples sequence
            # (branch 0 repeated counts[0] times, then branch 1, ...). Splitting
            # that sequence into consecutive chunks of size batch_n_samples and
            # counting branch occurrences per chunk gives condition_counts_i that
            # satisfy both invariants exactly.
            condition_counts_np = np.asarray(condition_counts).astype(np.int64)
            if condition_counts_np.ndim != 1:
                raise ValueError("condition_counts must be one-dimensional")
            if int(condition_counts_np.sum()) != int(n_samples):
                raise ValueError("sum(condition_counts) must equal n_samples")
            # branch_of_sample[k] = which branch the k-th sample column belongs to.
            branch_of_sample = np.repeat(
                np.arange(condition_counts_np.shape[0], dtype=np.int64),
                condition_counts_np,
            )
            n_branches = condition_counts_np.shape[0]

        # Memory-bounded sample batching. The Python sample loop below materializes
        # every chunk's large (batch_n_samples x n_gates) intermediates and, under
        # reverse-mode autodiff, UNROLLS so all chunks' activations stay live at
        # once -- which multiplied (not reduced) gradient memory and OOM'd the
        # D-Wave joint fully-connected fine-tune. When the sample axis divides
        # evenly into fixed-size chunks we instead drive the chunks with
        # jax.lax.scan wrapped in jax.checkpoint: scan keeps the chunk shapes
        # fixed, and the checkpoint makes the backward pass REMATERIALIZE one
        # chunk's samples_gates at a time instead of stacking all of them, so peak
        # VRAM is ~one chunk regardless of n_samples. The final per-sample matrix
        # is tiny (n_ops x n_samples), so stacking the chunk outputs is cheap.
        # The downstream loss only reduces over the sample axis (mean and
        # mean-of-squares), which is permutation invariant, so the interleaved
        # chunk-major column order produced here is exact.
        use_sample_scan = (
            (not self.bitflip)
            and max_batch_samples is not None
            and max_batch_samples < n_samples
            and (n_samples % max_batch_samples == 0)
        )
        if use_sample_scan:
            chunk_sz = int(max_batch_samples)
            n_sample_chunks = n_samples // chunk_sz
            if multi_condition:
                # Static per-chunk per-branch counts (each row sums to chunk_sz),
                # zeros allowed. op_expval_batch tolerates empty branches (its
                # valid-mask skips them) and, because the per-chunk counts arrive
                # as a JAX tracer under scan, its strict-positivity validation is
                # bypassed -- so the full-branch fixed shape is safe.
                counts_per_chunk = np.stack([
                    np.bincount(
                        branch_of_sample[c * chunk_sz:(c + 1) * chunk_sz],
                        minlength=n_branches,
                    ).astype(np.int64)
                    for c in range(n_sample_chunks)
                ], axis=0)
                counts_per_chunk_j = jnp.asarray(counts_per_chunk, dtype=jnp.int32)

        for batch_ops in jnp.array_split(ops, np.ceil(ops.shape[0] / max_batch_ops)):
            if use_sample_scan:
                keys = jax.random.split(key, n_sample_chunks + 1)
                chunk_keys = keys[:n_sample_chunks]
                key = keys[n_sample_chunks]

                def _chunk(carry, scan_in):
                    if multi_condition:
                        ck, counts_row = scan_in
                        be = self.op_expval_batch(
                            params, batch_ops, chunk_sz, ck, init_coefs,
                            indep_estimates, return_samples=True,
                            condition_mask=condition_mask,
                            condition_values=condition_values,
                            condition_counts=counts_row, ops_wires=ops_wires,
                        )
                    else:
                        ck = scan_in
                        be = self.op_expval_batch(
                            params, batch_ops, chunk_sz, ck, init_coefs,
                            indep_estimates, return_samples=True,
                            condition_mask=condition_mask,
                            condition_values=condition_values,
                            condition_counts=condition_counts, ops_wires=ops_wires,
                        )
                    return carry, be

                _chunk = jax.checkpoint(_chunk)
                xs = (chunk_keys, counts_per_chunk_j) if multi_condition else chunk_keys
                _, stacked = jax.lax.scan(_chunk, 0, xs)
                # stacked: (n_chunks, n_ops_batch, chunk_sz) ->
                # (n_ops_batch, n_chunks * chunk_sz).
                tmp_expvals = jnp.transpose(stacked, (1, 0, 2)).reshape(
                    stacked.shape[1], n_sample_chunks * chunk_sz)
            else:
                tmp_expvals = jnp.empty((len(batch_ops), 0))
                for i in range(np.ceil(n_samples / max_batch_samples).astype(jnp.int64)):
                    batch_n_samples = min(max_batch_samples, n_samples - i * max_batch_samples)
                    key, subkey = jax.random.split(key, 2)

                    if multi_condition:
                        # Consecutive chunk [start, stop) of the laid-out sample
                        # sequence handled by this sample-batch.
                        start = i * max_batch_samples
                        stop = start + batch_n_samples
                        chunk_branches = branch_of_sample[start:stop]
                        # Per-branch counts within this chunk (length n_branches).
                        counts_i = np.bincount(chunk_branches, minlength=n_branches).astype(np.int64)
                        # op_expval_batch rejects non-positive counts, so subset to
                        # only the branches present in this chunk. Row order of the
                        # subset is the ascending branch index, preserving the
                        # column ordering that the multi_condition fori_loop produces.
                        active = np.nonzero(counts_i > 0)[0]
                        batch_condition_mask = condition_mask[active]
                        batch_condition_values = condition_values[active]
                        batch_condition_counts = jnp.asarray(counts_i[active], dtype=jnp.int32)
                        batch_expval = self.op_expval_batch(
                            params, batch_ops, int(batch_n_samples), subkey, init_coefs,
                            indep_estimates, return_samples=True,
                            condition_mask=batch_condition_mask,
                            condition_values=batch_condition_values,
                            condition_counts=batch_condition_counts, ops_wires=ops_wires,
                        )
                    else:
                        batch_expval = self.op_expval_batch(
                            params, batch_ops, batch_n_samples, subkey, init_coefs,
                            indep_estimates, return_samples=True,
                            condition_mask=condition_mask,
                            condition_values=condition_values,
                            condition_counts=condition_counts, ops_wires=ops_wires
                        )
                    tmp_expvals = jnp.concatenate((tmp_expvals, batch_expval), axis=-1)
            expvals = jnp.concatenate((expvals, tmp_expvals), axis=0)

        if self.bitflip:
            if return_samples:
                return expvals
            else:
                return jnp.mean(expvals, axis=-1), jnp.zeros(len(ops))
        else:
            if return_samples:
                return expvals
            else:
                return jnp.mean(expvals, axis=-1), jnp.std(expvals, axis=-1, ddof=1)/jnp.sqrt(n_samples)
