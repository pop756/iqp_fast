# Copyright 2026 Youngseok Lee
# SPDX-License-Identifier: Apache-2.0

"""Sparse-op estimator path of IqpSimulator.op_expval for ROCm GPUs (HIP kernels through jax.ffi).

Same samples and the same estimator as the grouped path (_phases); only the arithmetic is reorganised.
With s = 1 - 2 z on the visible wires, u_v(z) = theta_v(segment of z) s_v(z) for a group v (the
generators sharing the visible part v, see IqpSimulator._ops_split), S the support of an op A and
x_v = |S & v|, the cos argument of the grouped path is
    phase(A, z) = 2 sum_v parity(x_v) u_v(z)
                = 2 sum_{i in S} F_i(z) + 2 sum_{v: x_v >= 2} (parity(x_v) - x_v) u_v(z),
    F_i(z) = sum_{v contains i} u_v(z).
F (samples x visible wires) is shared by all ops: one matmul per segment for the pair groups plus a
gather kernel for the >= 3-body groups. The ops of an MMD loss are sparse, so the correction (groups
with at least two wires in S) is a short list per op, evaluated by a kernel with the samples packed
as bits. The gradient is the transpose of each piece (custom_vjp).
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_kernels.hip")
_STATE = {}
_TARGETS = (("iqp_pack", "IqpPack"), ("iqp_f3", "IqpF3"), ("iqp_d3", "IqpD3"), ("iqp_phase", "IqpPhase"),
            ("iqp_phase_bwd", "IqpPhaseBwd"), ("iqp_bits", "IqpBits"), ("iqp_trdata", "IqpTrData"),
            ("iqp_order", "IqpOrder"), ("iqp_unpad", "IqpUnpad"), ("iqp_theta", "IqpTheta"),
            ("iqp_theta_bwd", "IqpThetaBwd"), ("iqp_pairmm", "IqpPairMM"), ("iqp_pairmm_bwd", "IqpPairMMBwd"))
HI = jax.lax.Precision.HIGHEST


def _hipcc():
    for c in (shutil.which("hipcc"), "/opt/rocm/bin/hipcc"):
        if c and os.path.exists(c):
            return c
    return None


def _cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "iqpopt_kernels")


def _load():
    """Build (once per cache dir) and register the kernels. Returns True if they are usable."""
    if os.environ.get("IQPOPT_NO_KERNELS", "0") in ("1", "true", "True"):
        return False
    try:
        dev = jax.devices()[0]
        if dev.platform != "gpu" or "rocm" not in str(dev.client.platform_version).lower():
            return False
        arch = str(getattr(dev, "compute_capability", "") or "native")
        hipcc = _hipcc()
        if hipcc is None:
            return False
        with open(_SRC, "rb") as f:
            src = f.read()
        tag = hashlib.sha256(src + arch.encode() + jax.__version__.encode()).hexdigest()[:16]
        so = os.path.join(_cache_dir(), f"iqpk_{tag}.so")
        if not os.path.exists(so):
            os.makedirs(_cache_dir(), exist_ok=True)
            fd, tmp = tempfile.mkstemp(suffix=".so", dir=_cache_dir())
            os.close(fd)
            cmd = [hipcc, "-O3", "-std=c++17", f"--offload-arch={arch}", "-shared", "-fPIC", "-munsafe-fp-atomics",
                   "-Wno-defaulted-function-deleted", "-I", jax.ffi.include_dir(), _SRC, "-o", tmp, "-lrocblas"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                os.unlink(tmp)
                raise RuntimeError(r.stderr[-2000:])
            os.replace(tmp, so)
        lib = ctypes.cdll.LoadLibrary(so)
        for name, sym in _TARGETS:
            jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(lib, sym)), platform="ROCM")
        _STATE["lib"] = lib
        _STATE["blas"] = lib.iqp_init_blas() == 0  # rocBLAS handle of the pair-angle products
        return True
    except Exception as e:  # noqa: BLE001 - any failure falls back to the XLA paths
        _STATE["error"] = repr(e)
        return False


def available():
    if "ok" not in _STATE:
        _STATE["ok"] = _load()
    return _STATE["ok"]


MAX_NV = 2048  # LDS of the op kernels holds two ints per visible wire


def supported(sim, split):
    """The kernels need at least one group and at most MAX_NV visible wires."""
    nv = sim.n_qubits - len(split["hidden"])
    return split["n_groups"] > 0 and 0 < nv <= MAX_NV and split["vidx"].shape[1] <= 8


_PIB = jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS


def _take(x, idx):
    """x[idx] along axis 0 for a static index array (lax.gather on a ready-made (M, 1) int32 constant: no
    index arithmetic for XLA to constant-fold)."""
    idx = np.asarray(idx, dtype=np.int32).reshape(-1, 1)
    dn = jax.lax.GatherDimensionNumbers(offset_dims=tuple(range(1, x.ndim)), collapsed_slice_dims=(0,),
                                        start_index_map=(0,))
    return jax.lax.gather(x, idx, dn, slice_sizes=(1,) + tuple(x.shape[1:]), mode=_PIB)


# Below this many (op x group x sample) terms of the dense phase product, the XLA paths of op_expval are as
# fast in total (compilation included) as the kernels; leave such small circuits to them.
MIN_WORK = 5e9


def worth_it(split, n_ops, n_samples):
    return float(n_ops) * float(split["n_groups"]) * float(n_samples) >= MIN_WORK


def _i32(x):
    x = np.asarray(x, dtype=np.int32).reshape(-1)
    return x if x.size else np.zeros(1, np.int32)


def tables(sim, split):
    """Static index tables of the visible group structure (numpy), cached in the split dict."""
    if "fast" in split:
        return split["fast"]
    nq = sim.n_qubits
    hidden = split["hidden"]
    vis_mask = np.ones(nq, dtype=bool)
    vis_mask[hidden] = False
    vis = np.flatnonzero(vis_mask)
    nv = len(vis)
    pos = np.full(nq + 1, -1, dtype=np.int64)
    pos[vis] = np.arange(nv)
    vidx = split["vidx"]
    n_groups = split["n_groups"]
    P = pos[np.minimum(vidx, nq)]  # positions, -1 = padding
    size = (P >= 0).sum(axis=1)
    g_all = np.arange(n_groups)
    m1 = size == 1  # singles
    # pairs: local pair index p, positions pi < pj; ptab[a, b] = p for a < b
    m2 = size == 2
    pair_gid = g_all[m2]
    pi, pj = P[m2, 0], P[m2, 1]
    ptab = np.full((nv, nv), -1, dtype=np.int32)
    ptab[pi, pj] = np.arange(len(pair_gid))
    ptab[pj, pi] = np.arange(len(pair_gid))
    # >= 3-body groups: CSR of positions, wire -> groups, pair -> groups
    m3 = size >= 3
    g3_gid = g_all[m3]
    P3 = P[m3]
    s3 = size[m3]
    n3 = len(g3_gid)
    out = dict(nv=nv, vis=vis.astype(np.int32), pair_gid=pair_gid.astype(np.int32),
               pi=pi.astype(np.int32), pj=pj.astype(np.int32), ptab=ptab, n2=len(pair_gid), g3_gid=g3_gid.astype(np.int32),
               n3=n3)
    if n3:
        kmax = P3.shape[1]
        valid = P3 >= 0
        # wire -> (group, other wires): entry per (group v, column j) of a valid wire, sorted by wire
        own_v, own_j = np.nonzero(valid)
        cols = np.arange(kmax)
        except1 = np.stack([np.delete(cols, j) for j in range(kmax)])  # (kmax, kmax - 1)
        order = np.argsort(P3[own_v, own_j], kind="stable")
        own_v, own_j = own_v[order], own_j[order]
        w3_v = own_v
        w3_o = P3[own_v[:, None], except1[own_j]]
        w3_o = np.where(w3_o < 0, nv, w3_o)  # pad wire nv: the zero row of bitsT
        w3_off = np.concatenate([[0], np.cumsum(np.bincount(P3[valid], minlength=nv))]).astype(np.int32)
        # pair (a < b) -> (group, other wires)
        ia, ib = np.triu_indices(kmax, 1)
        ok = valid[:, ia] & valid[:, ib]
        pv, pc = np.nonzero(ok)
        keys = P3[pv, ia[pc]] * nv + P3[pv, ib[pc]]
        except2 = np.stack([np.delete(cols, [a, b]) for a, b in zip(ia, ib)]) if kmax > 2 else np.zeros((len(ia), 0), int)
        p3_o = P3[pv[:, None], except2[pc]]
        if p3_o.shape[1] == 0:
            p3_o = np.full((len(pv), 1), -1)
        p3_o = np.where(p3_o < 0, nv, p3_o)
        order = np.argsort(keys, kind="stable")
        p3_v, p3_o = pv[order], p3_o[order]
        p3_off = np.concatenate([[0], np.cumsum(np.bincount(keys, minlength=nv * nv))]).astype(np.int32)
        worder = np.argsort(-np.diff(w3_off), kind="stable").astype(np.int32)  # longest group lists first
        # per wire: its distinct partner wires (other wires of its groups) and, per list entry, the local
        # index of each other wire in that list (n_partners = the pad wire, a zero row)
        pw, pw_off, w3_ol = [], [0], np.zeros_like(w3_o)
        for i in range(nv):
            t0, t1 = w3_off[i], w3_off[i + 1]
            o = w3_o[t0:t1]
            u = np.unique(o[o < nv])
            loc = np.searchsorted(u, np.minimum(o, nv - 1))
            w3_ol[t0:t1] = np.where(o < nv, loc, len(u))
            pw.append(u)
            pw_off.append(pw_off[-1] + len(u))
        out.update(worder=worder, ko=int(kmax - 1), ko2=int(max(kmax - 2, 1)), w3_off=w3_off, w3_v=_i32(w3_v),
                   w3_o=_i32(w3_o), p3_off=p3_off, p3_v=_i32(p3_v), p3_o=_i32(p3_o), w3_ol=_i32(w3_ol),
                   pw=_i32(np.concatenate(pw) if pw else np.zeros(0)), pw_off=np.asarray(pw_off, np.int32),
                   np_max=int(np.max(np.diff(pw_off))) if nv else 0)
    else:
        out.update(worder=np.arange(nv, dtype=np.int32), ko=1, ko2=1, w3_off=np.zeros(nv + 1, np.int32), w3_v=np.zeros(1, np.int32),
                   w3_o=np.full(1, nv, np.int32), p3_off=np.zeros(2, np.int32), p3_v=np.zeros(1, np.int32),
                   p3_o=np.full(1, nv, np.int32), w3_ol=np.zeros(1, np.int32), pw=np.zeros(1, np.int32),
                   pw_off=np.zeros(nv + 1, np.int32), np_max=0)
    # theta layout: rows [0, nv) the single group of each wire (zero row if none), [nv, nv + n2) the pair
    # groups, then the >=3-body groups; W[row, hidden mask] = summed params of its generators
    n2 = len(pair_gid)
    row = np.empty(n_groups + 1, dtype=np.int64)
    row[g_all[m1]] = P[m1, 0]
    row[pair_gid] = nv + np.arange(n2)
    row[g3_gid] = nv + n2 + np.arange(n3)
    gid = split["gid"].astype(np.int64)
    ok = gid < n_groups
    nh = split["mbits"].shape[0]
    slot = row[gid[ok]] * nh + split["hmask"][ok]
    gens = np.flatnonzero(ok)
    n_rows = nv + n2 + n3
    out.update(n_rows=n_rows, nh=nh)
    if len(slot) == 0 or np.bincount(slot, minlength=n_rows * nh).max() <= 1:  # one generator per slot: a gather
        winv = np.full(n_rows * nh, len(gid), dtype=np.int64)
        winv[slot] = gens
        gslot = np.full(len(gid), -1, dtype=np.int64)
        gslot[gens] = slot
        out.update(inj=True, winv=winv.astype(np.int32), gslot=gslot.astype(np.int32))
    else:
        out.update(inj=False, wslot=slot.astype(np.uint32), wgen=gens.astype(np.uint32))
    split["fast"] = out
    return out


def _theta(sim, split, tb, segments, params, init_coefs):
    """(th1 (nv, n_seg), th2 (n2, n_seg), th3 (n3, n_seg), Th2 (n_seg, nv, nv) or None): the angles of the
    single / pair / >=3-body groups on each segment. Cached for repeated calls with the same params
    object (the two bandwidths of an MMD loss share them)."""
    seg_bits = tuple(seg[2] for seg in segments)
    c = sim.__dict__.get("_theta_cache")
    if c is not None and c[0] is params and c[1] is init_coefs and c[2] == seg_bits and c[3] is split:
        return c[4]
    nv, n2, n3, nh = tb["nv"], tb["n2"], tb["n3"], tb["nh"]
    nseg = len(segments)
    eff = sim.trans_par @ params if sim.par_transform else params
    eff = eff + sim.trans_coef @ init_coefs if sim.init_gates is not None else eff
    eff = eff.astype(jnp.float32)
    bits = np.asarray(seg_bits, dtype=np.float32).reshape(nseg, -1)
    hsign = (1 - 2 * ((bits @ split["mbits"].T) % 2)).astype(np.float32)  # (n_seg, 2**h)
    if tb["inj"] and tb["n_rows"] > 0:  # kernels: one generator per (group, hidden mask)
        out = _theta_kernels(tb, nseg, hsign)(eff)
        out = out if n2 else out[:3] + (None,)
        sim.__dict__["_theta_cache"] = (params, init_coefs, seg_bits, split, out)
        return out
    if tb["inj"]:
        Wf = _take(jnp.concatenate([eff, jnp.zeros((1,), jnp.float32)]), tb["winv"])
    else:
        Wf = jnp.zeros((tb["n_rows"] * nh,), jnp.float32).at[tb["wslot"]].add(_take(eff, tb["wgen"]),
                                                                               mode="promise_in_bounds")
    W = Wf.reshape(tb["n_rows"], nh)
    if nh == 1:  # no hidden wires
        theta = W * jnp.asarray(hsign.T)  # (n_rows, n_seg)
    else:
        theta = jnp.sum(W[:, None, :] * jnp.asarray(hsign)[None, :, :], axis=-1)  # (n_rows, n_seg)
    th1 = theta[:nv]
    th2 = theta[nv:nv + n2] if n2 else jnp.zeros((1, nseg), jnp.float32)
    th3 = theta[nv + n2:] if n3 else jnp.zeros((1, nseg), jnp.float32)
    Th2 = None
    if n2:
        t2 = th2.T  # (n_seg, n2)
        # both triangles of the symmetric (nv, nv) pair-angle matrix of each segment, by one scatter
        idx = np.concatenate([np.stack([tb["pi"], tb["pj"]], 1), np.stack([tb["pj"], tb["pi"]], 1)]).astype(np.int32)
        dn = jax.lax.ScatterDimensionNumbers(update_window_dims=(0,), inserted_window_dims=(1, 2),
                                             scatter_dims_to_operand_dims=(1, 2))
        Th2 = jax.lax.scatter(jnp.zeros((nseg, nv, nv), jnp.float32), idx, jnp.concatenate([t2, t2], axis=1), dn,
                              indices_are_sorted=False, unique_indices=True, mode=_PIB)
    out = (th1, th2, th3, Th2)
    sim.__dict__["_theta_cache"] = (params, init_coefs, seg_bits, split, out)
    return out


def _theta_kernels(tb, nseg, hsign):
    """custom_vjp eff -> (th1, th2, th3, Th2) on the kernels (see _theta)."""
    nv, n2, n3 = tb["nv"], tb["n2"], tb["n3"]
    H = np.ascontiguousarray(hsign, dtype=np.float32)
    winv = tb["winv"].reshape(tb["n_rows"], -1)
    shapes = (jax.ShapeDtypeStruct((nv, nseg), jnp.float32), jax.ShapeDtypeStruct((max(n2, 1), nseg), jnp.float32),
              jax.ShapeDtypeStruct((max(n3, 1), nseg), jnp.float32),
              jax.ShapeDtypeStruct((nseg, nv, nv) if n2 else (1, 1, 1), jnp.float32))

    def fwd_call(eff):
        return jax.ffi.ffi_call("iqp_theta", shapes)(eff, winv, H, tb["ptab"], n2=np.int32(n2), n3=np.int32(n3))

    n_gens = len(tb["gslot"])

    def bwd(res, g):
        dth1, dth2, dth3, dTh2 = g
        deff, _ = jax.ffi.ffi_call("iqp_theta_bwd", (jax.ShapeDtypeStruct((n_gens,), jnp.float32),
                                                     jax.ShapeDtypeStruct((max(n2, 1), nseg), jnp.float32)))(
            dth1, dth2, dth3, dTh2, tb["gslot"], H, _i32(tb["pi"]), _i32(tb["pj"]), n2=np.int32(n2))
        return (deff,)

    @jax.custom_vjp
    def theta(eff):
        return fwd_call(eff)

    theta.defvjp(lambda eff: (fwd_call(eff), None), bwd)
    return theta


def _pair_products(nseg, nv):
    """custom_vjp (Th2, X) -> Th2[k] @ X[k] over the segments on rocBLAS (no XLA dot to autotune)."""
    def call(Th2, X):
        return jax.ffi.ffi_call("iqp_pairmm", jax.ShapeDtypeStruct(X.shape, jnp.float32))(Th2, X)

    def bwd(X, dM):
        return jax.ffi.ffi_call("iqp_pairmm_bwd", jax.ShapeDtypeStruct((nseg, nv, nv), jnp.float32))(dM, X), None

    @jax.custom_vjp
    def pmm(Th2, X):
        return call(Th2, X)

    pmm.defvjp(lambda Th2, X: (call(Th2, X), X), bwd)
    return pmm


def _slots(segments):
    """Slots (n_slot * 3,) int32 = (segment, word, bit mask) for every word a segment's sample range meets,
    grouped by segment, and seg_slot (n_seg + 1,): the slot range of each segment."""
    out, seg_slot = [], [0]
    for k, (a, c, _) in enumerate(segments):
        for w in range(a // 32, (a + c - 1) // 32 + 1):
            lo, hi = max(a, 32 * w) - 32 * w, min(a + c, 32 * w + 32) - 32 * w
            out.append((k, w, ((1 << hi) - 1) ^ ((1 << lo) - 1)))
        seg_slot.append(len(out))
    arr = np.asarray(out, dtype=np.int64)
    arr[:, 2] = arr[:, 2].astype(np.uint32).view(np.int32)
    return arr.astype(np.int32).reshape(-1), np.asarray(seg_slot, np.int32)


def _pack(samples, vis, nv, ns, seg_of, loc_of, nseg, c_pad):
    """bitsT (nv + 1, ceil(ns / 32)) uint32 (last row zero) and sTp (n_seg, nv, c_pad) float32: 1 - 2 z of
    the visible columns, sample z at [seg_of[z], :, loc_of[z]], zero padded."""
    nw = -(-ns // 32)
    return jax.ffi.ffi_call("iqp_pack", (jax.ShapeDtypeStruct((nv + 1, nw), jnp.uint32),
                                         jax.ShapeDtypeStruct((nseg, nv, c_pad), jnp.float32)))(
        samples.astype(jnp.int32), vis, seg_of, loc_of)


def _make_ops(tb, slots, seg_slot, nv, ns_all, ns, nseg, op_sig, op_off, vstart, f2s):
    """custom_vjp wrappers of the kernels for one layout of sample blocks: block s (its ops
    op_off[s]..op_off[s+1], samples s ns .. (s+1) ns of the ns_all-long sample axis) has the virtual segments
    s nseg .. (s+1) nseg - 1 (angle column = virtual segment % nseg)."""
    has2, has3 = int(tb["n2"] > 0), int(tb["n3"] > 0)
    th2s, th3s = (max(tb["n2"], 1), nseg), (max(tb["n3"], 1), nseg)
    ko, ko2 = np.int32(tb["ko"]), np.int32(tb["ko2"])

    # F = F2 + F3: F2 read from its padded layout F2p (nseg, nv, n_blocks c_pad), F3 the >=3-body part
    def f3_call(th3, bitsT, F2p):
        return jax.ffi.ffi_call("iqp_f3", jax.ShapeDtypeStruct((nv, ns_all), jnp.float32))(
            th3, bitsT, tb["w3_off"], tb["w3_v"], tb["w3_o"], slots, tb["worder"], F2p, vstart, ko=ko)

    def f3_bwd(bitsT, g):
        if has3:
            d = jax.ffi.ffi_call("iqp_d3", jax.ShapeDtypeStruct(th3s, jnp.float32))(
                g, bitsT, tb["w3_off"], tb["w3_v"], tb["w3_o"], slots, seg_slot, tb["worder"], tb["w3_ol"], tb["pw"],
                tb["pw_off"], ko=ko, np_max=np.int32(tb["np_max"]))
        else:
            d = jnp.zeros(th3s, jnp.float32)
        dF2p = jax.ffi.ffi_call("iqp_unpad", jax.ShapeDtypeStruct(f2s, jnp.float32))(g, vstart)
        return d, None, dF2p

    @jax.custom_vjp
    def f3(th3, bitsT, F2p):
        return f3_call(th3, bitsT, F2p)

    f3.defvjp(lambda th3, bitsT, F2p: (f3_call(th3, bitsT, F2p), bitsT), f3_bwd)

    def phase_call(F, th2, th3, A, bitsT, perm):
        n_ops = A.shape[0]
        return jax.ffi.ffi_call("iqp_phase", (jax.ShapeDtypeStruct((n_ops, ns), jnp.float32),
                                              jax.ShapeDtypeStruct((nv, -(-n_ops // 32)), jnp.uint32),
                                              jax.ShapeDtypeStruct((n_ops, 2), jnp.float32)))(
            A, bitsT, F, th2, th3, tb["ptab"], tb["p3_off"], tb["p3_v"], tb["p3_o"], slots, seg_slot, op_sig, perm,
            has2=np.int32(has2), has3=np.int32(has3), ko2=ko2)

    def order(A):  # ops by decreasing support size (their blocks are the longest)
        n_ops = A.shape[0]
        return jax.ffi.ffi_call("iqp_order", (jax.ShapeDtypeStruct((n_ops,), jnp.int32),
                                              jax.ShapeDtypeStruct((n_ops,), jnp.int32)))(A)[0]

    def bwd_call(A, bitsT, g, AT, perm, dS, from_phase):
        return jax.ffi.ffi_call(
            "iqp_phase_bwd", (jax.ShapeDtypeStruct((nv, ns_all), jnp.float32),
                              jax.ShapeDtypeStruct(th2s, jnp.float32), jax.ShapeDtypeStruct(th3s, jnp.float32)))(
            A, bitsT, g, AT, tb["ptab"], tb["p3_off"], tb["p3_v"], tb["p3_o"], slots, seg_slot, op_sig, op_off, perm,
            dS, has2=np.int32(has2), has3=np.int32(has3), ko2=ko2, from_phase=np.int32(from_phase))

    def phase_fwd(F, th2, th3, A, bitsT):
        perm = order(A)
        ph, AT, _ = phase_call(F, th2, th3, A, bitsT, perm)
        return ph, (A, bitsT, AT, perm)

    def phase_bwd(res, g):
        A, bitsT, AT, perm = res
        QT, dth2, dth3 = bwd_call(A, bitsT, g, AT, perm, jnp.zeros((1, 2), jnp.float32), 0)
        return QT, dth2, dth3, None, None  # k_q returns 2 QT, the cotangent of F

    @jax.custom_vjp
    def phase(F, th2, th3, A, bitsT):
        return phase_call(F, th2, th3, A, bitsT, order(A))[0]

    phase.defvjp(phase_fwd, phase_bwd)

    # per op: (sum_z cos(phase), sum_z cos(phase)^2); the cotangent of phase is formed in the kernels
    def sums_fwd(F, th2, th3, A, bitsT):
        perm = order(A)
        ph, AT, sm = phase_call(F, th2, th3, A, bitsT, perm)
        return sm, (A, bitsT, AT, perm, ph)

    def sums_bwd(res, dsm):
        A, bitsT, AT, perm, ph = res
        QT, dth2, dth3 = bwd_call(A, bitsT, ph, AT, perm, dsm, 1)
        return QT, dth2, dth3, None, None

    @jax.custom_vjp
    def phase_sums(F, th2, th3, A, bitsT):
        return phase_call(F, th2, th3, A, bitsT, order(A))[2]

    phase_sums.defvjp(sums_fwd, sums_bwd)
    return f3, phase, phase_sums


def _draw_visible(sim, key, n_samples, max_batch_samples, condition_mask, condition_values, condition_counts, vis):
    """The samples of IqpSimulator._draw_samples (same jax.random calls and keys) on the columns the kernels read:
    the conditioning is applied to the visible columns a condition fixes; hidden columns stay as drawn (their
    values are fixed per segment and enter only through the segment angles)."""
    nq = sim.n_qubits
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
        mask = np.asarray(condition_mask, dtype=bool)  # concrete: _segments only accepts numpy conditions
        values = np.asarray(condition_values)
        if mask.ndim == 2:
            slot = np.repeat(np.arange(len(condition_counts)), np.asarray(condition_counts).astype(np.int64))
            mask, values = mask[slot], values[slot]
        else:
            mask = np.broadcast_to(mask, (n_samples, nq))
            values = np.broadcast_to(values, (n_samples, nq))
        cols = vis[np.any(mask[:, vis], axis=0)]
        if len(cols):
            sub = jnp.where(mask[:, cols], values[:, cols].astype(np.int32), samples[:, cols])
            samples = samples.at[:, cols].set(sub.astype(samples.dtype))
    return samples, key


def expval_blocks(sim, split, segments, params, init_coefs, blocks, sums=False, visible=False):
    """cos(phase) (n_ops_s, n_samples) for each block s = (ops_s, samples_s) of ops with their own samples
    (all with the segment layout `segments`), evaluated together: the blocks' samples are laid end to end
    on one sample axis, so every kernel is launched once for all blocks. With sums=True, per block the
    (n_ops_s, 2) sums over the samples of cos(phase) and cos(phase)^2 instead. visible=True: the blocks' ops
    are given on the visible wires only (columns in the order of tables()["vis"])."""
    tb = tables(sim, split)
    nv = tb["nv"]
    nseg = len(segments)
    nblk = len(blocks)
    th1, th2, th3, Th2 = _theta(sim, split, tb, segments, params, init_coefs)
    counts = [c for _, c, _ in segments]
    ns = int(sum(counts))
    ns_all = nblk * ns
    vsegments = [(s * ns + a, c, bits) for s in range(nblk) for (a, c, bits) in segments]
    vcounts = counts * nblk
    vstart = np.concatenate([[0], np.cumsum(vcounts)]).astype(np.int64)
    c_pad = int(max(counts))
    vseg = np.repeat(np.arange(nblk * nseg), vcounts)
    # sTp layout (nseg, nv, nblk c_pad): segment k of block s at columns s c_pad + local index
    seg_of = (vseg % nseg).astype(np.int32)
    loc_of = ((vseg // nseg) * c_pad + np.arange(ns_all) - vstart[vseg]).astype(np.int32)
    slots, seg_slot = _slots(vsegments)
    n_ops = [int(o.shape[0]) for o, _ in blocks]
    op_off = np.concatenate([[0], np.cumsum(n_ops)]).astype(np.int32)
    op_sig = np.repeat(np.arange(nblk), n_ops).astype(np.int32)
    f3, phase_op, sums_op = _make_ops(tb, slots, seg_slot, nv, ns_all, ns, nseg, op_sig, op_off,
                                      vstart.astype(np.int32), (nseg, nv, nblk * c_pad))
    vis = tb["vis"]
    samples = blocks[0][1] if nblk == 1 else jnp.concatenate([s for _, s in blocks])
    bitsT, sTp = _pack(samples, vis, nv, ns_all, seg_of, loc_of, nseg, nblk * c_pad)  # sTp (nseg, nv, nblk c_pad)
    ops = blocks[0][0] if nblk == 1 else jnp.concatenate([o for o, _ in blocks])
    A = ops.astype(jnp.float32) if visible else _take(ops.T, vis).T.astype(jnp.float32)
    inner = th1.T[:, :, None]  # (nseg, nv, 1)
    if Th2 is not None:  # pair groups: one batched matmul over the (padded) segments of all blocks
        if _STATE.get("blas"):
            inner = inner + _pair_products(nseg, nv)(Th2, sTp)
        else:
            inner = inner + jnp.einsum("kij,kjc->kic", Th2, sTp, precision=HI)
    F2p = sTp * inner  # (nseg, nv, nblk c_pad)
    F = f3(th3, bitsT, F2p)  # (nv, ns_all) = F2 + the >=3-body part
    ex = sums_op(F, th2, th3, A, bitsT) if sums else jnp.cos(phase_op(F, th2, th3, A, bitsT))
    return [ex] if nblk == 1 else [ex[op_off[s]:op_off[s + 1]] for s in range(nblk)]


def op_expval_fast(sim, split, segments, params, ops, n_samples, key, init_coefs, return_samples, max_batch_ops,
                   max_batch_samples, condition_mask, condition_values, condition_counts):
    """op_expval on the sparse-op kernels; same samples (IqpSimulator._draw_samples) and estimator. The op
    batches (each with its own samples) are evaluated together."""
    vis = tables(sim, split)["vis"]
    blocks = []
    for batch_ops in jnp.array_split(ops, np.ceil(ops.shape[0] / (max_batch_ops or ops.shape[0]))):
        samples, key = _draw_visible(sim, key, n_samples, max_batch_samples, condition_mask, condition_values,
                                     condition_counts, vis)
        blocks.append((batch_ops, samples))
    ex = expval_blocks(sim, split, segments, params, init_coefs, blocks)
    expvals = ex[0] if len(ex) == 1 else jnp.concatenate(ex)
    if return_samples:
        return expvals
    return jnp.mean(expvals, axis=-1), jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)


def data_term(ground_truth, visible_ops):
    """jnp.mean(1 - 2 * ((ground_truth @ visible_ops.T) % 2), axis=0), the data part of the MMD loss.

    With integer data the parity of each data row on each op is the xor of the row's bits on the op's
    support, so the kernel computes the same values exactly (sums of +-1) from bit-packed data. The bit
    packing of a data array is shared by repeated calls with the same array object (both bandwidths)."""
    gt = ground_truth
    if not (available() and getattr(gt, "ndim", 0) == 2 and getattr(visible_ops, "ndim", 0) == 2
            and jnp.issubdtype(gt.dtype, jnp.integer) and gt.shape[1] == visible_ops.shape[1]
            and 0 < gt.shape[1] <= MAX_NV and 0 < gt.shape[0] < 2 ** 24 and visible_ops.shape[0] > 0):
        return jnp.mean(1 - 2 * ((ground_truth @ visible_ops.T) % 2), axis=0)
    N, C = gt.shape
    if isinstance(gt, jax.core.Tracer):
        dataT = jax.ffi.ffi_call("iqp_bits", jax.ShapeDtypeStruct((C, -(-N // 32)), jnp.uint32))(gt.astype(jnp.int32))
    else:  # concrete data: packed once on the host (a constant of the trace)
        c = _STATE.get("data_cache")
        if c is not None and c[0] is gt:
            dataT = c[1]
        else:
            nw = -(-N // 32)
            bits = np.zeros((nw * 32, C), dtype=np.uint32)
            bits[:N] = np.asarray(gt).astype(np.int64) & 1
            dataT = np.ascontiguousarray(
                (bits.reshape(nw, 32, C) << np.arange(32, dtype=np.uint32)[None, :, None]).sum(axis=1, dtype=np.uint32).T)
            _STATE["data_cache"] = (gt, dataT)
    return jax.ffi.ffi_call("iqp_trdata", jax.ShapeDtypeStruct((visible_ops.shape[0],), jnp.float32))(
        visible_ops.astype(jnp.float32), dataT, n_rows=np.int32(N))
