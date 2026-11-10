// Copyright 2026 Youngseok Lee
// SPDX-License-Identifier: Apache-2.0

// HIP kernels of the iqp_fast sparse-op estimator path (see iqp_fast/_fastpath.py).
//
// Notation (wire indices are positions among the visible wires, 0..nv-1):
//   bitsT[i, w]   bit b of word w is sample z = 32 w + b's bit on visible wire i
//   s_v(z)        (-1)^(xor of z's bits on the wires of v)
//   slot          (segment k, word w, mask m): the bits of word w that belong to segment k. The slots of a
//                 segment are contiguous (seg_slot[k] .. seg_slot[k+1]); every word is covered by the slots
//                 of the segments it meets, so a slot has a single angle per group.
//   th2 / th3     angles (n_groups_of_kind, n_seg) of the pair / >=3-body groups
//   w3_*          per wire i: the >=3-body groups containing i (w3_v) and their other wires (w3_o, KO wide)
//   p3_*          per pair (a < b): the >=3-body groups containing both (p3_v), their other wires (p3_o, KO2)
// Ops (rows of A, 0/1 floats) are sparse; each op block extracts the support S of its op.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#define RESTRICT __restrict__

static inline ffi::Error last_error() {
  hipError_t e = hipGetLastError();
  if (e != hipSuccess) return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(e));
  return ffi::Error::Success();
}

__device__ __forceinline__ float flip(float x, uint32_t bit) {  // x * (bit ? -1 : 1)
  return __int_as_float(__float_as_int(x) ^ (int)(bit << 31));
}

// 32 floats d[0..31] of the word at z0 = 32 w of a row of length ns (clamped reads past the end)
__device__ __forceinline__ void load32(const float* RESTRICT row, int wd, int ns, float* v) {
  const int z0 = wd * 32;
  if (z0 + 32 <= ns) {
    const float4* q = (const float4*)(row + z0);
    if ((((uintptr_t)q) & 15) == 0) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float4 x = q[j];
        v[4 * j] = x.x;
        v[4 * j + 1] = x.y;
        v[4 * j + 2] = x.z;
        v[4 * j + 3] = x.w;
      }
    } else {
#pragma unroll
      for (int b = 0; b < 32; ++b) v[b] = row[z0 + b];
    }
  } else {
#pragma unroll
    for (int b = 0; b < 32; ++b) v[b] = row[min(z0 + b, ns - 1)];
  }
}

// 32 floats row[off .. off + 31] of a row of length n; off may be negative or run past the end (clamped)
__device__ __forceinline__ void load32_at(const float* RESTRICT row, int off, int n, float* v) {
  if (off >= 0 && off + 32 <= n) {
    const float* p = row + off;
    if ((((uintptr_t)p) & 15) == 0) {
      const float4* q = (const float4*)p;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float4 x = q[j];
        v[4 * j] = x.x;
        v[4 * j + 1] = x.y;
        v[4 * j + 2] = x.z;
        v[4 * j + 3] = x.w;
      }
    } else {
#pragma unroll
      for (int b = 0; b < 32; ++b) v[b] = p[b];
    }
  } else {
#pragma unroll
    for (int b = 0; b < 32; ++b) v[b] = row[min(max(off + b, 0), n - 1)];
  }
}

__device__ __forceinline__ float sum32v(const float* v, uint32_t par, uint32_t m) {  // sum_{b in m} s_b v[b]
  float p = 0.f;
#pragma unroll
  for (int b = 0; b < 32; ++b) {
    const float x = flip(v[b], (par >> b) & 1u);
    p += ((m >> b) & 1u) ? x : 0.f;
  }
  return p;
}

__device__ __forceinline__ float sum32(const float* RESTRICT row, int wd, int ns, uint32_t par, uint32_t m) {
  float v[32];
  load32(row, wd, ns, v);
  return sum32v(v, par, m);
}

__device__ __forceinline__ float sum32_at(const float* RESTRICT row, int off, int n, uint32_t par, uint32_t m) {
  float v[32];
  load32_at(row, off, n, v);
  return sum32v(v, par, m);
}

// sum32_at over G = -sin(ph) (d1 + d2 cos(ph)) of a phase row
__device__ __forceinline__ float sum32_ph(const float* RESTRICT row, int off, int n, uint32_t par, uint32_t m,
                                          float d1, float d2) {
  float v[32];
  load32_at(row, off, n, v);
#pragma unroll
  for (int b = 0; b < 32; ++b) {
    float sn, cs;
    sincosf(v[b], &sn, &cs);
    v[b] = -sn * (d1 + d2 * cs);
  }
  return sum32v(v, par, m);
}

__device__ __forceinline__ float wave_sum(float x) {
  for (int off = 32; off > 0; off >>= 1) x += __shfl_down(x, off, 64);
  return x;
}

// Exclusive scan over the block (blockDim.x a multiple of 64, <= 1024); *total = block sum.
__device__ inline int block_scan(int x, int* buf, int* total) {
  const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6, nwv = blockDim.x >> 6;
  int incl = x;
  for (int off = 1; off < 64; off <<= 1) {
    const int y = __shfl_up(incl, off, 64);
    if (lane >= off) incl += y;
  }
  if (lane == 63) buf[wv] = incl;
  __syncthreads();
  int base = 0, tot = 0;
  for (int u = 0; u < nwv; ++u) {
    const int y = buf[u];
    if (u < wv) base += y;
    tot += y;
  }
  __syncthreads();
  *total = tot;
  return base + incl - x;
}

// ------------------------------------------------------------------ packing
// bitsT[i, w] from samples (ns, nq) int32 at the visible columns (row nv: zeros, the pad wire), and
// sTp[seg_of[z], i, loc_of[z]] = 1 - 2 z_i (float; the caller zeroes the padding). Block = one word
// (32 samples) x 64 wires, 4 threads per wire.
__global__ void __launch_bounds__(256) k_pack(const int* RESTRICT samples, const int* RESTRICT vis,
                                              const int* RESTRICT seg_of, const int* RESTRICT loc_of,
                                              uint32_t* RESTRICT bitsT, float* RESTRICT sTp, int ns, int nq, int nw,
                                              int nv, int c_pad) {
  const int w = blockIdx.x;
  const int il = threadIdx.x & 63;
  const int i = blockIdx.y * 64 + il;  // i == nv writes the pad row
  const int b0 = threadIdx.x >> 6;  // 0..3
  __shared__ uint32_t part[4][64];
  uint32_t word = 0;
  if (i < nv) {
    const int col = vis[i];
    for (int b = b0; b < 32; b += 4) {
      const int z = w * 32 + b;
      if (z < ns) {
        const uint32_t x = (uint32_t)samples[(size_t)z * nq + col] & 1u;
        word |= x << b;
        sTp[((size_t)seg_of[z] * nv + i) * c_pad + loc_of[z]] = x ? -1.f : 1.f;
      }
    }
  }
  part[b0][il] = word;
  __syncthreads();
  if (b0 == 0 && i < nv) bitsT[(size_t)i * nw + w] = part[0][il] | part[1][il] | part[2][il] | part[3][il];
  if (b0 == 0 && i == nv) bitsT[(size_t)nv * nw + w] = 0u;
}

// ------------------------------------------------------------------ >=3-body linear part
// F3[i, z] = sum_{v >= 3-body, v contains i} th3[v, seg(z)] s_v(z).  Block = wire i; lane = 16 slots x 4
// subs (the subs split the group list, summed by shuffles); the group list is staged in LDS.
#define F3_WAVES 8
#define F3_LDS 6144
template <int KO>
__global__ void __launch_bounds__(64 * F3_WAVES) k_f3(const float* RESTRICT th3, const uint32_t* RESTRICT bitsT,
                                                    const int* RESTRICT w3_off, const int* RESTRICT w3_v,
                                                    const int* RESTRICT w3_o, const int* RESTRICT slots,
                                                    const int* RESTRICT worder, const float* RESTRICT F2p,
                                                    const int* RESTRICT vstart, float* RESTRICT out, int ns, int nw,
                                                    int nseg, int nslot, int nv, int c_pad, int cw) {
  // out = F2 + F3: F2[i, z] = F2p[k, i, s c_pad + (z - vstart[vk])] for z in virtual segment vk = s nseg + k
  // (F2p: (nseg, nv, cw) with cw = n_blocks c_pad)
  __shared__ int lds[F3_LDS];
  const int i = worder[blockIdx.x];  // wires with the longest group lists first
  const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
  const int sub = lane >> 4;
  const int t0 = w3_off[i], t1 = w3_off[i + 1];
  const int per = KO + nseg;
  const int chunk = F3_LDS / per;
  for (int sbase = blockIdx.y * F3_WAVES * 16; sbase < nslot; sbase += gridDim.y * F3_WAVES * 16) {
    const int sl = sbase + wv * 16 + (lane & 15);
    const bool act = sl < nslot;
    int k = 0, wd = 0;
    uint32_t m = 0;
    if (act) {
      k = slots[3 * sl] % nseg;  // virtual segment -> angle column
      wd = slots[3 * sl + 1];
      m = (uint32_t)slots[3 * sl + 2];
    }
    const uint32_t bwi = act ? bitsT[(size_t)i * nw + wd] : 0u;
    float acc[32];
#pragma unroll
    for (int b = 0; b < 32; ++b) acc[b] = 0.f;
    for (int c0 = t0; c0 < t1; c0 += chunk) {
      const int c1 = min(t1, c0 + chunk);
      __syncthreads();
      for (int x = threadIdx.x; x < (c1 - c0) * per; x += blockDim.x) {
        const int t = c0 + x / per, f = x % per;
        lds[x] = f < KO ? w3_o[(size_t)t * KO + f] : __float_as_int(th3[(size_t)w3_v[t] * nseg + (f - KO)]);
      }
      __syncthreads();
      if (act) {
        const int cnt = c1 - c0;
        int t = sub;
        for (; t + 28 < cnt; t += 32) {  // 8 groups per iteration: all loads issued first
          uint32_t par[8];
          float th[8];
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            const int* e = lds + (t + 4 * u) * per;
            uint32_t pw = bwi;
#pragma unroll
            for (int o = 0; o < KO; ++o) pw ^= bitsT[(size_t)e[o] * nw + wd];
            par[u] = pw;
            th[u] = __int_as_float(e[KO + k]);
          }
#pragma unroll
          for (int u = 0; u < 8; ++u)
#pragma unroll
            for (int b = 0; b < 32; ++b) acc[b] += flip(th[u], (par[u] >> b) & 1u);
        }
        for (; t < cnt; t += 4) {
          const int* e = lds + t * per;
          uint32_t par = bwi;
#pragma unroll
          for (int o = 0; o < KO; ++o) par ^= bitsT[(size_t)e[o] * nw + wd];
          const float th = __int_as_float(e[KO + k]);
#pragma unroll
          for (int b = 0; b < 32; ++b) acc[b] += flip(th, (par >> b) & 1u);
        }
      }
    }
#pragma unroll
    for (int b = 0; b < 32; ++b) {
      acc[b] += __shfl_xor(acc[b], 16, 64);
      acc[b] += __shfl_xor(acc[b], 32, 64);
    }
    if (act && sub == 0) {
      const int vk = slots[3 * sl];
      const float* f2 = F2p + ((size_t)k * nv + i) * cw + (size_t)(vk / nseg) * c_pad + ((ptrdiff_t)wd * 32 - vstart[vk]);
      float* o = out + (size_t)i * ns + (size_t)wd * 32;
#pragma unroll
      for (int b = 0; b < 32; ++b)
        if ((m >> b) & 1u) o[b] = acc[b] + f2[b];
    }
  }
}

// dF2p[k, i, s c_pad + loc] = dF[i, vstart[vk] + loc] for loc < count(vk), else 0 (vk = s nseg + k): the
// cotangent of the padded F2 of k_f3.
__global__ void __launch_bounds__(256) k_unpad(const float* RESTRICT dF, const int* RESTRICT vstart,
                                               float* RESTRICT out, int ns, int nv, int nseg, int c_pad, int cw) {
  const int row = blockIdx.x;  // k nv + i
  const int k = row / nv, i = row - k * nv;
  for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < cw; c += gridDim.y * blockDim.x) {
    const int s = c / c_pad, loc = c - s * c_pad, vk = s * nseg + k;
    const int z0 = vstart[vk];
    out[(size_t)row * cw + c] = (loc < vstart[vk + 1] - z0) ? dF[(size_t)i * ns + z0 + loc] : 0.f;
  }
}

// dth3[v, k] += sum_{z in k} s_v(z) dF[i, z] for each wire i of v.  Block = wire i, threads = (group of
// the wire's list, segment) tasks; dF[i, :] and bitsT[i, :] staged in LDS (when they fit).
template <int KO>
__global__ void __launch_bounds__(512) k_d3(const float* RESTRICT dF, const uint32_t* RESTRICT bitsT,
                                             const int* RESTRICT w3_off, const int* RESTRICT w3_v,
                                             const int* RESTRICT w3_o, const int* RESTRICT slots,
                                             const int* RESTRICT seg_slot, const int* RESTRICT worder,
                                             float* RESTRICT dth3, int ns, int nw, int nseg, int nvseg, int stage) {
  extern __shared__ float drow_s[];
  const int i = worder[blockIdx.x];
  const float* drow = dF + (size_t)i * ns;
  const uint32_t* brow = bitsT + (size_t)i * nw;
  if (stage) {
    uint32_t* bs = (uint32_t*)(drow_s + ns);
    for (int z = threadIdx.x; z < ns; z += blockDim.x) drow_s[z] = drow[z];
    for (int x = threadIdx.x; x < nw; x += blockDim.x) bs[x] = brow[x];
    __syncthreads();
    drow = drow_s;
    brow = bs;
  }
  const int t0 = w3_off[i], t1 = w3_off[i + 1];
  const int nt = t1 - t0;
  const int vk0 = blockIdx.y * nseg, nvk = min(nseg, nvseg - vk0);  // this block's sample block
  for (int task = threadIdx.x; task < nt * nvk; task += blockDim.x) {
    const int k = vk0 + task / nt, t = t0 + task % nt;  // (virtual) segment-major: a wave shares the segment
    int ow[KO];
#pragma unroll
    for (int o = 0; o < KO; ++o) ow[o] = w3_o[(size_t)t * KO + o];
    float part = 0.f;
    const int s1 = seg_slot[k + 1];
    int sl = seg_slot[k];
    for (; sl + 3 < s1; sl += 4) {
      uint32_t par[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int wd = slots[3 * (sl + u) + 1];
        uint32_t pw = brow[wd];
#pragma unroll
        for (int o = 0; o < KO; ++o) pw ^= bitsT[(size_t)ow[o] * nw + wd];
        par[u] = pw;
      }
#pragma unroll
      for (int u = 0; u < 4; ++u)
        part += sum32(drow, slots[3 * (sl + u) + 1], ns, par[u], (uint32_t)slots[3 * (sl + u) + 2]);
    }
    for (; sl < s1; ++sl) {
      const int wd = slots[3 * sl + 1];
      uint32_t par = brow[wd];
#pragma unroll
      for (int o = 0; o < KO; ++o) par ^= bitsT[(size_t)ow[o] * nw + wd];
      part += sum32(drow, wd, ns, par, (uint32_t)slots[3 * sl + 2]);
    }
    atomicAdd(&dth3[(size_t)w3_v[t] * nseg + k % nseg], part);
  }
}

// k_d3 with the bit rows of the wire's partner wires and its dF row staged in LDS (block = wire i, sample
// block y; thread = group of the wire's list). Per slot, the signed sum over the 32 samples is
// sum_b d_b - 2 sum_{b: odd parity} d_b, the second sum from 8 nibble tables of subset sums of the slot's d.
#define D3S_NB 256
__device__ __forceinline__ int nv_pad(int) { return 0; }
#define D3S_LCH 16  // slots per table chunk
template <int KO>
__global__ void __launch_bounds__(D3S_NB) k_d3s(const float* RESTRICT dF, const uint32_t* RESTRICT bitsT,
                                               const int* RESTRICT w3_off, const int* RESTRICT w3_v,
                                               const int* RESTRICT w3_ol, const int* RESTRICT pw_off,
                                               const int* RESTRICT pw, const int* RESTRICT slots,
                                               const int* RESTRICT seg_slot, const int* RESTRICT worder,
                                               float* RESTRICT dth3, int ns, int nw, int nseg, int nvseg) {
  extern __shared__ uint32_t sm[];
  const int i = worder[blockIdx.x];
  const int vk0 = blockIdx.y * nseg, nvk = min(nseg, nvseg - vk0);
  const int sl_lo = seg_slot[vk0], sl_hi = seg_slot[vk0 + nvk];
  const int t0 = w3_off[i], t1 = w3_off[i + 1];
  if (sl_lo >= sl_hi || t0 >= t1) return;
  const int w_lo = slots[3 * sl_lo + 1], nwb = slots[3 * (sl_hi - 1) + 1] - w_lo + 1;
  const int str = nwb | 1;  // odd row stride: partner rows at the same word fall in different banks
  const int p0 = pw_off[i], npart = pw_off[i + 1] - p0;
  const int nsl = sl_hi - sl_lo;
  int* SW = (int*)sm;  // slot words (relative to w_lo) and masks of this sample block
  uint32_t* SM = (uint32_t*)(SW + nsl);
  uint32_t* own = SM + nsl;
  uint32_t* P = own + str;
  float* D = (float*)(P + (size_t)(npart + 1) * str);
  float* LUT = D + (size_t)nwb * 32;
  for (int x = threadIdx.x; x < nsl; x += D3S_NB) {
    SW[x] = slots[3 * (sl_lo + x) + 1] - w_lo;
    SM[x] = (uint32_t)slots[3 * (sl_lo + x) + 2];
  }
  // staging: 8 independent loads in flight per thread
  for (int w = threadIdx.x; w < nwb; w += D3S_NB) own[w] = bitsT[(size_t)i * nw + w_lo + w];
  for (int p = 0; p <= npart; ++p) {
    const uint32_t* src = bitsT + (size_t)(p < npart ? pw[p0 + p] : nv_pad(nw)) * nw + w_lo;
    for (int w = threadIdx.x; w < nwb; w += D3S_NB) P[(size_t)p * str + w] = p < npart ? src[w] : 0u;
  }
  {
    const float* drow = dF + (size_t)i * ns + (size_t)w_lo * 32;
    const int nz = min(nwb * 32, ns - w_lo * 32);
    int x = threadIdx.x;
    for (; x + 7 * D3S_NB < nz; x += 8 * D3S_NB) {
      float v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) v[u] = drow[x + u * D3S_NB];
#pragma unroll
      for (int u = 0; u < 8; ++u) D[x + u * D3S_NB] = v[u];
    }
    for (; x < nwb * 32; x += D3S_NB) D[x] = x < nz ? drow[x] : 0.f;
  }
  __syncthreads();
  // threads = (group, slot half): with nt <= D3S_NB / 2 groups two threads share a group's slots
  const int nt = t1 - t0;
  const int nsp = nt <= D3S_NB / 2 ? 2 : 1, tpp = D3S_NB / nsp;
  const int hsp = threadIdx.x / tpp, tl = threadIdx.x - hsp * tpp;
  for (int k = vk0; k < vk0 + nvk; ++k) {
    const int s_lo = seg_slot[k], s_hi = seg_slot[k + 1];
    for (int tb = t0; tb < t1; tb += tpp) {
      const int t = tb + tl;
      int ol[KO];
#pragma unroll
      for (int o = 0; o < KO; ++o) ol[o] = t < t1 ? w3_ol[(size_t)t * KO + o] : npart;
      float acc = 0.f;
      for (int c0 = s_lo; c0 < s_hi; c0 += D3S_LCH) {
        const int nc = min(D3S_LCH, s_hi - c0);
        __syncthreads();
        for (int x = threadIdx.x; x < nc * 128; x += D3S_NB) {  // tables: (slot, nibble q, subset x)
          const int j = x >> 7, q = (x >> 4) & 7, sub = x & 15;
          const int sl = c0 + j - sl_lo;
          const uint32_t m = SM[sl] >> (4 * q);
          const float* d = D + (size_t)SW[sl] * 32 + 4 * q;
          float v = 0.f;
#pragma unroll
          for (int b = 0; b < 4; ++b)
            if (((sub & m) >> b) & 1) v += d[b];
          LUT[x] = v;
        }
        __syncthreads();
        if (t < t1) {
          for (int j = hsp; j < nc; j += nsp) {
            const int w = SW[c0 + j - sl_lo];
            uint32_t par = own[w];
#pragma unroll
            for (int o = 0; o < KO; ++o) par ^= P[(size_t)ol[o] * str + w];
            const float* L = LUT + j * 128;
            float tot = 0.f, odd = 0.f;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              tot += L[q * 16 + 15];
              odd += L[q * 16 + ((par >> (4 * q)) & 15u)];
            }
            acc += tot - 2.f * odd;
          }
        }
      }
      if (t < t1) atomicAdd(&dth3[(size_t)w3_v[t] * nseg + k % nseg], acc);
    }
  }
}

// ------------------------------------------------------------------ per-op support and entry lists
// Ordered support S of op row o (LDS) and its membership bitmask inS, by the first wave. If AT is given,
// bit o of AT[i, o / 32] is set for i in S (the transposed op bitmap for k_q).
__device__ inline int load_support(const float* RESTRICT A, int o, int nv, int* S, uint32_t* inS, int* wsh,
                                   uint32_t* RESTRICT AT, int nwo) {
  for (int x = threadIdx.x; x < nv / 32 + 1; x += blockDim.x) inS[x] = 0u;  // bit nv (the pad wire) stays 0
  __syncthreads();
  if (threadIdx.x < 64) {
    const int lane = threadIdx.x;
    int cnt = 0;
    const float* row = A + (size_t)o * nv;
    for (int base = 0; base < nv; base += 256) {
      bool on[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int i = base + 64 * u + lane;
        on[u] = (i < nv) && ((((int)row[i]) & 1) != 0);
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const unsigned long long msk = __ballot(on[u]);
        const int pos = cnt + __popcll(msk & ((1ull << lane) - 1ull));
        if (on[u]) {
          const int i = base + 64 * u + lane;
          S[pos] = i;
          atomicOr(&inS[i >> 5], 1u << (i & 31));
          if (AT) atomicOr(&AT[(size_t)i * nwo + (o >> 5)], 1u << (o & 31));
        }
        cnt += __popcll(msk);
      }
    }
    if (lane == 0) *wsh = cnt;
  }
  __syncthreads();
  return *wsh;
}

__device__ __forceinline__ bool in_s(const uint32_t* inS, int i) { return (inS[i >> 5] >> (i & 31)) & 1u; }

// Candidate pairs c in [c0, c0 + blockDim) of S x S (c = p * w + q, p < q), one per thread:
//   L2[3e..] = (a, b, pair id) for the pair groups inside S (at most blockDim per chunk);
//   >=3-body groups v containing the pair with x = |v & S| >= 2, counted once (canonical: the pair is
//   the two smallest wires of v & S): entries (v, 2 (parity(x) - x), a, b, other wires) of width
//   4 + KO2 in L3, numbered by a block scan and written by windows of cap3 entries (fill_l3).
struct Cand {
  int a, b, e0, e1, o3, cnt3;
};

template <int KO2>
__device__ inline bool canon(const int* RESTRICT p3_o, int e, const uint32_t* inS, int b, int* x) {
  int c = 2;
  bool ok = true;
#pragma unroll
  for (int u = 0; u < KO2; ++u) {
    const int wj = p3_o[(size_t)e * KO2 + u];
    if (in_s(inS, wj)) {
      ++c;
      if (wj < b) ok = false;
    }
  }
  *x = c;
  return ok;
}

template <int KO2>
__device__ inline Cand scan_candidates(int c0, int w, const int* S, const uint32_t* inS, int nv, int has2, int has3,
                                       const int* RESTRICT ptab, const int* RESTRICT p3_off,
                                       const int* RESTRICT p3_o, int* L2, int* scanbuf, int* n2, int* n3) {
  const int c = c0 + threadIdx.x;
  const int ww = w > 0 ? w : 1;
  const int p = c / ww, q = c - p * ww;
  const bool valid = c < w * w && p < q;
  Cand cd = {0, 0, 0, 0, 0, 0};
  int pid = -1;
  if (valid) {
    cd.a = S[p];
    cd.b = S[q];
    const size_t pk = (size_t)cd.a * nv + cd.b;
    if (has2) pid = ptab[pk];
    if (has3) {
      cd.e0 = p3_off[pk];
      cd.e1 = p3_off[pk + 1];
      for (int e = cd.e0; e < cd.e1; ++e) {
        int x;
        if (canon<KO2>(p3_o, e, inS, cd.b, &x)) ++cd.cnt3;
      }
    }
  }
  int tot2, tot3;
  const int o2 = block_scan(pid >= 0 ? 1 : 0, scanbuf, &tot2);
  cd.o3 = block_scan(cd.cnt3, scanbuf, &tot3);
  if (pid >= 0) {  // support positions (bit words staged per position) and the pair id
    L2[3 * o2] = p;
    L2[3 * o2 + 1] = q;
    L2[3 * o2 + 2] = pid;
  }
  *n2 = tot2;
  *n3 = tot3;
  __syncthreads();
  return cd;
}

template <int KO2>
__device__ inline void fill_l3(const Cand& cd, int base, int cap3, const uint32_t* inS, const int* RESTRICT p3_v,
                               const int* RESTRICT p3_o, int* L3) {
  if (cd.cnt3 && cd.o3 < base + cap3 && cd.o3 + cd.cnt3 > base) {
    int u = cd.o3;
    for (int e = cd.e0; e < cd.e1; ++e) {
      int x;
      if (canon<KO2>(p3_o, e, inS, cd.b, &x)) {
        if (u >= base && u < base + cap3) {
          int* r = L3 + (size_t)(u - base) * (4 + KO2);
          r[0] = p3_v[e];
          r[1] = 2 * ((x & 1) - x);
          r[2] = cd.a;
          r[3] = cd.b;
#pragma unroll
          for (int j = 0; j < KO2; ++j) r[4 + j] = p3_o[(size_t)e * KO2 + j];
        }
        ++u;
      }
    }
  }
  __syncthreads();
}

template <int KO2>
__device__ __forceinline__ uint32_t entry_par3(const int* r, const uint32_t* RESTRICT bitsT, int nw, int wd) {
  uint32_t par = bitsT[(size_t)r[2] * nw + wd] ^ bitsT[(size_t)r[3] * nw + wd];
#pragma unroll
  for (int j = 0; j < KO2; ++j) par ^= bitsT[(size_t)r[4 + j] * nw + wd];
  return par;
}

#define PH_NB 128    // threads of k_phase
#define PB_NB 256    // threads of k_phase_bwd
#define L3_INTS 512  // ints of the L3 window
#define TH_STAGE 16  // k_phase stages the pair angles in LDS for nseg <= TH_STAGE
#define WS 24        // support positions whose sample bit words the op kernels stage in LDS
#define SB_SLOTS 128 // ... for at most this many slots of the op's sample block

__host__ __device__ inline size_t op_lds_ints(int nv, int nb) {
  return (size_t)nv + nv / 32 + 1 + 3 * nb + L3_INTS + 16 + 4 + (size_t)(WS + 2) * SB_SLOTS;
}

// phase[o, z] = 2 sum_{i in S} F[i, z] - 4 sum_{pair groups in S} th2 s_a s_b
//             + sum_{>=3-body v, |v & S| >= 2} 2 (parity - x) th3 s_v.   Block = op, threads = slots.
template <int KO2>
__global__ void __launch_bounds__(PH_NB) k_phase(const float* RESTRICT A, const uint32_t* RESTRICT bitsT,
                                                 const float* RESTRICT F, const float* RESTRICT th2,
                                                 const float* RESTRICT th3, const int* RESTRICT ptab,
                                                 const int* RESTRICT p3_off, const int* RESTRICT p3_v,
                                                 const int* RESTRICT p3_o, const int* RESTRICT slots,
                                                 const int* RESTRICT seg_slot, const int* RESTRICT op_sig,
                                                 const int* RESTRICT perm, float* RESTRICT out, uint32_t* RESTRICT AT,
                                                 float* RESTRICT sums, int nv, int ns, int ns_o, int nw, int nseg,
                                                 int nwo, int has2, int has3) {
  extern __shared__ int smem[];
  int* S = smem;
  uint32_t* inS = (uint32_t*)(S + nv);
  int* L2 = (int*)(inS + nv / 32 + 1);
  int* L3 = L2 + 3 * PH_NB;
  int* scanbuf = L3 + L3_INTS;
  int* wsh = scanbuf + 16;
  uint32_t* SB = (uint32_t*)(wsh + 4);  // SB[p PH_NB + thread]: bit word of support position p at the thread's slot
  float* thl = (float*)(SB + WS * SB_SLOTS);  // PH_NB * nseg staged pair angles (nseg <= TH_STAGE)
  const bool stage = nseg <= TH_STAGE;
  const int cap3 = L3_INTS / (4 + KO2);
  const int o = perm[blockIdx.x];  // ops in decreasing support size: the long blocks start first
  const int w = load_support(A, o, nv, S, inS, wsh, AT, nwo);
  const int sg = op_sig[o];  // the op's sample block: its slots and the offset of its output row
  const int sl0 = seg_slot[sg * nseg], sl1 = seg_slot[(sg + 1) * nseg];
  for (int sbase = sl0; sbase < sl1; sbase += PH_NB) {  // one slot per thread unless the block has more
    const int sl = sbase + threadIdx.x;
    const bool act = sl < sl1;
    int k = 0, wd = 0;
    uint32_t m = 0;
    if (act) {
      k = slots[3 * sl] % nseg;
      wd = slots[3 * sl + 1];
      m = (uint32_t)slots[3 * sl + 2];
    }
    float acc[32];
#pragma unroll
    for (int b = 0; b < 32; ++b) acc[b] = 0.f;
    if (act) {
      const int ws = min(w, WS);
      int p = 0;
      for (; p + 3 < ws; p += 4) {
        uint32_t x[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) x[u] = bitsT[(size_t)S[p + u] * nw + wd];
#pragma unroll
        for (int u = 0; u < 4; ++u) SB[(p + u) * PH_NB + threadIdx.x] = x[u];
      }
      for (; p < ws; ++p) SB[p * PH_NB + threadIdx.x] = bitsT[(size_t)S[p] * nw + wd];
    }
    if (act) {
      int p = 0;
      for (; p + 1 < w; p += 2) {
        float v0[32], v1[32];
        load32(F + (size_t)S[p] * ns, wd, ns, v0);
        load32(F + (size_t)S[p + 1] * ns, wd, ns, v1);
#pragma unroll
        for (int b = 0; b < 32; ++b) acc[b] += v0[b] + v1[b];
      }
      if (p < w) {
        float v0[32];
        load32(F + (size_t)S[p] * ns, wd, ns, v0);
#pragma unroll
        for (int b = 0; b < 32; ++b) acc[b] += v0[b];
      }
#pragma unroll
      for (int b = 0; b < 32; ++b) acc[b] *= 2.f;
    }
    for (int c0 = 0; c0 < w * w; c0 += PH_NB) {
      int n2, n3;
      const Cand cd = scan_candidates<KO2>(c0, w, S, inS, nv, has2, has3, ptab, p3_off, p3_o, L2, scanbuf, &n2, &n3);
      if (stage) {
        int x = threadIdx.x;
        for (; x + 3 * PH_NB < n2 * nseg; x += 4 * PH_NB) {
          float v[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int y = x + u * PH_NB;
            v[u] = th2[(size_t)L2[3 * (y / nseg) + 2] * nseg + y % nseg];
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) thl[x + u * PH_NB] = -4.f * v[u];
        }
        for (; x < n2 * nseg; x += PH_NB) thl[x] = -4.f * th2[(size_t)L2[3 * (x / nseg) + 2] * nseg + x % nseg];
        __syncthreads();
      }
      for (int base = 0; base == 0 || base < n3; base += cap3) {
        fill_l3<KO2>(cd, base, cap3, inS, p3_v, p3_o, L3);
        const int m3 = min(cap3, n3 - base);
        if (act) {
          if (base == 0) {
            int e = 0;
            for (; e + 3 < n2; e += 4) {
              uint32_t par[4];
              float th[4];
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                const int pa = L2[3 * (e + u)], pb = L2[3 * (e + u) + 1];
                const uint32_t ba = pa < WS ? SB[pa * PH_NB + threadIdx.x] : bitsT[(size_t)S[pa] * nw + wd];
                const uint32_t bb = pb < WS ? SB[pb * PH_NB + threadIdx.x] : bitsT[(size_t)S[pb] * nw + wd];
                par[u] = ba ^ bb;
                th[u] = stage ? thl[(e + u) * nseg + k] : -4.f * th2[(size_t)L2[3 * (e + u) + 2] * nseg + k];
              }
#pragma unroll
              for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int b = 0; b < 32; ++b) acc[b] += flip(th[u], (par[u] >> b) & 1u);
            }
            for (; e < n2; ++e) {
              const int pa = L2[3 * e], pb = L2[3 * e + 1];
              const uint32_t ba = pa < WS ? SB[pa * PH_NB + threadIdx.x] : bitsT[(size_t)S[pa] * nw + wd];
              const uint32_t bb = pb < WS ? SB[pb * PH_NB + threadIdx.x] : bitsT[(size_t)S[pb] * nw + wd];
              const uint32_t par = ba ^ bb;
              const float th = stage ? thl[e * nseg + k] : -4.f * th2[(size_t)L2[3 * e + 2] * nseg + k];
#pragma unroll
              for (int b = 0; b < 32; ++b) acc[b] += flip(th, (par >> b) & 1u);
            }
          }
          int e = 0;
          for (; e + 1 < m3; e += 2) {  // two entries' loads in flight
            const int* r0 = L3 + (size_t)e * (4 + KO2);
            const int* r1 = r0 + (4 + KO2);
            const uint32_t par0 = entry_par3<KO2>(r0, bitsT, nw, wd);
            const uint32_t par1 = entry_par3<KO2>(r1, bitsT, nw, wd);
            const float t0 = (float)r0[1] * th3[(size_t)r0[0] * nseg + k];
            const float t1 = (float)r1[1] * th3[(size_t)r1[0] * nseg + k];
#pragma unroll
            for (int b = 0; b < 32; ++b) acc[b] += flip(t0, (par0 >> b) & 1u) + flip(t1, (par1 >> b) & 1u);
          }
          for (; e < m3; ++e) {
            const int* r = L3 + (size_t)e * (4 + KO2);
            const uint32_t par = entry_par3<KO2>(r, bitsT, nw, wd);
            const float th = (float)r[1] * th3[(size_t)r[0] * nseg + k];
#pragma unroll
            for (int b = 0; b < 32; ++b) acc[b] += flip(th, (par >> b) & 1u);
          }
        }
        __syncthreads();
      }
    }
    float c1 = 0.f, c2 = 0.f;  // this slot's sums of cos(phase) and cos(phase)^2
    if (act) {
      float* op = out + (size_t)o * ns_o + ((ptrdiff_t)wd * 32 - (ptrdiff_t)sg * ns_o);
#pragma unroll
      for (int b = 0; b < 32; ++b)
        if ((m >> b) & 1u) {
          op[b] = acc[b];
          const float c = cosf(acc[b]);
          c1 += c;
          c2 += c * c;
        }
    }
    c1 = wave_sum(c1);
    c2 = wave_sum(c2);
    __syncthreads();
    if ((threadIdx.x & 63) == 0) {
      scanbuf[2 * (threadIdx.x >> 6)] = __float_as_int(c1);
      scanbuf[2 * (threadIdx.x >> 6) + 1] = __float_as_int(c2);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float t1 = 0.f, t2 = 0.f;
      for (int u = 0; u < PH_NB / 64; ++u) {
        t1 += __int_as_float(scanbuf[2 * u]);
        t2 += __int_as_float(scanbuf[2 * u + 1]);
      }
      if (sbase == sl0) {
        sums[2 * o] = t1;
        sums[2 * o + 1] = t2;
      } else {
        sums[2 * o] += t1;
        sums[2 * o + 1] += t2;
      }
    }
    __syncthreads();
  }
}

// Transpose of the correction part of k_phase: dth2[p, k] += -4 sum_o sum_{z in k} s_a s_b G[o, z],
// dth3[v, k] += coef sum_o sum_{z in k} s_v G[o, z].  Block = op; G[o, :] staged in LDS (when it fits);
// threads = (segment, entry) tasks (segment-major: a wave reads the same G words), one atomic each.
template <int KO2>
__global__ void __launch_bounds__(PB_NB) k_phase_bwd(const float* RESTRICT A, const uint32_t* RESTRICT bitsT,
                                                     const float* RESTRICT G, const int* RESTRICT ptab,
                                                     const int* RESTRICT p3_off, const int* RESTRICT p3_v,
                                                     const int* RESTRICT p3_o, const int* RESTRICT slots,
                                                     const int* RESTRICT seg_slot, const int* RESTRICT op_sig,
                                                     const int* RESTRICT perm, const float* RESTRICT dS,
                                                     float* RESTRICT dth2, float* RESTRICT dth3, int nv, int ns, int nw,
                                                     int nseg, int has2, int has3, int stage, int from_phase) {
  extern __shared__ int smem[];
  int* S = smem;
  uint32_t* inS = (uint32_t*)(S + nv);
  int* L2 = (int*)(inS + nv / 32 + 1);
  int* L3 = L2 + 3 * PB_NB;
  int* scanbuf = L3 + L3_INTS;
  int* wsh = scanbuf + 16;
  uint32_t* SB = (uint32_t*)(wsh + 4);  // SB[p nsl + j]: bit word of support position p at slot sl0 + j
  int* SWd = (int*)(SB + WS * SB_SLOTS);         // the op's slot words and masks (sb_ok)
  uint32_t* SMk = (uint32_t*)(SWd + SB_SLOTS);
  float* grow_s = (float*)(SMk + SB_SLOTS);
  const int cap3 = L3_INTS / (4 + KO2);
  const int o = perm[blockIdx.x];
  const int w = load_support(A, o, nv, S, inS, wsh, nullptr, 0);
  if (w < 2) return;
  const int sg = op_sig[o];
  const int sl0 = seg_slot[sg * nseg], nsl = seg_slot[(sg + 1) * nseg] - sl0;
  const bool sb_ok = nsl <= SB_SLOTS;  // the op's slots fit the staging buffer
  if (sb_ok) {
    for (int j = threadIdx.x; j < nsl; j += PB_NB) {
      SWd[j] = slots[3 * (sl0 + j) + 1];
      SMk[j] = (uint32_t)slots[3 * (sl0 + j) + 2];
    }
    const int ws = min(w, WS);
    int x = threadIdx.x;
    for (; x + 3 * PB_NB < ws * nsl; x += 4 * PB_NB) {
      uint32_t v[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int y = x + u * PB_NB, pp = y / nsl;
        v[u] = bitsT[(size_t)S[pp] * nw + slots[3 * (sl0 + y - pp * nsl) + 1]];
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) SB[x + u * PB_NB] = v[u];
    }
    for (; x < ws * nsl; x += PB_NB) {
      const int pp = x / nsl;
      SB[x] = bitsT[(size_t)S[pp] * nw + slots[3 * (sl0 + x - pp * nsl) + 1]];
    }
  }
  const int zoff = sg * ns;  // G rows are local to the op's sample block (length ns)
  const float* grow = G + (size_t)o * ns;
  const bool conv = from_phase && !stage;  // G from the phase row, converted at each read
  const float cd1 = from_phase ? dS[2 * o] : 0.f, cd2 = from_phase ? 2.f * dS[2 * o + 1] : 0.f;
#define SUM32G(row, off, n, par, m) (conv ? sum32_ph(row, off, n, par, m, cd1, cd2) : sum32_at(row, off, n, par, m))
  if (stage && from_phase) {  // G = -sin(phase) (dS1 + 2 dS2 cos(phase)), from the phase row
    const float d1 = dS[2 * o], d2 = 2.f * dS[2 * o + 1];
    int z = threadIdx.x;
    for (; z + 3 * PB_NB < ns; z += 4 * PB_NB) {
      float v[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) v[u] = grow[z + u * PB_NB];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        float sn, cs;
        sincosf(v[u], &sn, &cs);
        grow_s[z + u * PB_NB] = -sn * (d1 + d2 * cs);
      }
    }
    for (; z < ns; z += PB_NB) {
      float sn, cs;
      sincosf(grow[z], &sn, &cs);
      grow_s[z] = -sn * (d1 + d2 * cs);
    }
    __syncthreads();
    grow = grow_s;
  } else if (stage) {
    int z = threadIdx.x;
    for (; z + 3 * PB_NB < ns; z += 4 * PB_NB) {
      const float g0 = grow[z], g1 = grow[z + PB_NB], g2 = grow[z + 2 * PB_NB], g3 = grow[z + 3 * PB_NB];
      grow_s[z] = g0;
      grow_s[z + PB_NB] = g1;
      grow_s[z + 2 * PB_NB] = g2;
      grow_s[z + 3 * PB_NB] = g3;
    }
    for (; z < ns; z += PB_NB) grow_s[z] = grow[z];
    __syncthreads();
    grow = grow_s;
  }
  for (int c0 = 0; c0 < w * w; c0 += PB_NB) {
    int n2, n3;
    const Cand cd = scan_candidates<KO2>(c0, w, S, inS, nv, has2, has3, ptab, p3_off, p3_o, L2, scanbuf, &n2, &n3);
    for (int base = 0; base == 0 || base < n3; base += cap3) {
      fill_l3<KO2>(cd, base, cap3, inS, p3_v, p3_o, L3);
      const int e2 = base == 0 ? n2 : 0;
      const int ne = e2 + max(0, min(cap3, n3 - base));
      for (int task = threadIdx.x; task < ne * nseg; task += blockDim.x) {
        const int k = task / ne, e = task % ne;
        float part = 0.f;
        const int s1 = seg_slot[sg * nseg + k + 1];
        if (e < e2) {
          const int pa = L2[3 * e], pb = L2[3 * e + 1];
          const int a = S[pa], b = S[pb];
          int sl = seg_slot[sg * nseg + k];
          if (sb_ok && pa < WS && pb < WS) {  // everything from LDS
            for (; sl < s1; ++sl) {
              const int j = sl - sl0;
              part += SUM32G(grow, SWd[j] * 32 - zoff, ns, SB[pa * nsl + j] ^ SB[pb * nsl + j], SMk[j]);
            }
          }
          for (; sl + 3 < s1; sl += 4) {
            uint32_t par[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
              const int wd = slots[3 * (sl + u) + 1];
              par[u] = bitsT[(size_t)a * nw + wd] ^ bitsT[(size_t)b * nw + wd];
            }
#pragma unroll
            for (int u = 0; u < 4; ++u)
              part += SUM32G(grow, slots[3 * (sl + u) + 1] * 32 - zoff, ns, par[u], (uint32_t)slots[3 * (sl + u) + 2]);
          }
          for (; sl < s1; ++sl) {
            const int wd = slots[3 * sl + 1];
            const uint32_t par = bitsT[(size_t)a * nw + wd] ^ bitsT[(size_t)b * nw + wd];
            part += SUM32G(grow, wd * 32 - zoff, ns, par, (uint32_t)slots[3 * sl + 2]);
          }
          atomicAdd(&dth2[(size_t)L2[3 * e + 2] * nseg + k], -4.f * part);
        } else {
          const int* r = L3 + (size_t)(e - e2) * (4 + KO2);
          for (int sl = seg_slot[sg * nseg + k]; sl < s1; ++sl) {
            const int wd = slots[3 * sl + 1];
            part += SUM32G(grow, wd * 32 - zoff, ns, entry_par3<KO2>(r, bitsT, nw, wd), (uint32_t)slots[3 * sl + 2]);
          }
          atomicAdd(&dth3[(size_t)r[0] * nseg + k], (float)r[1] * part);
        }
      }
      __syncthreads();
    }
  }
}
#undef SUM32G

// QT[i, z] = 2 sum_{o: i in S_o, o in the sample block of z} G[o, z - block offset] (the cotangent of F),
// the ops of wire i read from the transposed bitmap AT (written by k_phase). Block = (wire i, Q_ZB samples):
// one wave builds the ordered op list, then each thread sums 4 consecutive samples over it (float4 loads,
// coalesced). G rows have length ns_g; sample block s = ops op_off[s]..op_off[s+1], samples s ns_g ...
#define Q_CAP 4096
#define Q_NB 256
#define Q_ZB (4 * Q_NB)
__device__ __forceinline__ float g_of(float ph, float d1, float d2) {  // -sin(ph) (d1 + d2 cos(ph))
  float sn, cs;
  sincosf(ph, &sn, &cs);
  return -sn * (d1 + d2 * cs);
}

__global__ void __launch_bounds__(Q_NB) k_q(const uint32_t* RESTRICT AT, const float* RESTRICT G,
                                            const int* RESTRICT op_off, const float* RESTRICT dS, float* RESTRICT QT,
                                            int ns, int ns_g, int nwo, int nblk, int from_phase) {
  __shared__ int L[Q_CAP];
  __shared__ int cnt_s;
  const int i = blockIdx.x;
  const int lane = threadIdx.x & 63;
  const uint32_t* row = AT + (size_t)i * nwo;
  const int z0 = blockIdx.y * Q_ZB + 4 * threadIdx.x;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int j0 = 0; j0 < nwo; j0 += Q_CAP / 32) {  // chunks of at most Q_CAP ops
    const int j1 = min(nwo, j0 + Q_CAP / 32);
    __syncthreads();
    if (threadIdx.x < 64) {
      int cnt = 0;
      for (int j = j0; j < j1; j += 64) {
        const uint32_t x = (j + lane < j1) ? row[j + lane] : 0u;
        const int c = __popc(x);
        int incl = c;
        for (int off = 1; off < 64; off <<= 1) {
          const int y = __shfl_up(incl, off, 64);
          if (lane >= off) incl += y;
        }
        int pos = cnt + incl - c;
        uint32_t y = x;
        while (y) {
          const int b = __ffs(y) - 1;
          L[pos++] = (j + lane) * 32 + b;
          y &= y - 1;
        }
        cnt += __shfl(incl, 63, 64);
      }
      if (lane == 0) cnt_s = cnt;
    }
    __syncthreads();
    const int cnt = cnt_s;
    if (z0 < ns) {
      const int s = min(z0 / ns_g, nblk - 1);
      const int zl = z0 - s * ns_g;
      const int o0 = op_off[s], o1 = op_off[s + 1];
      int e = 0;
      while (e < cnt && L[e] < o0) ++e;
      int e1 = e;
      while (e1 < cnt && L[e1] < o1) ++e1;
      if (from_phase && zl + 4 <= ns_g && z0 + 4 <= ns && ((zl & 3) == 0) && ((ns_g & 3) == 0)) {
        for (; e + 1 < e1; e += 2) {
          float4 v[2];
          float d1[2], d2[2];
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            v[u] = *(const float4*)(G + (size_t)L[e + u] * ns_g + zl);
            d1[u] = dS[2 * L[e + u]];
            d2[u] = 2.f * dS[2 * L[e + u] + 1];
          }
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            acc[0] += g_of(v[u].x, d1[u], d2[u]);
            acc[1] += g_of(v[u].y, d1[u], d2[u]);
            acc[2] += g_of(v[u].z, d1[u], d2[u]);
            acc[3] += g_of(v[u].w, d1[u], d2[u]);
          }
        }
        for (; e < e1; ++e) {
          const float4 v = *(const float4*)(G + (size_t)L[e] * ns_g + zl);
          const float d1 = dS[2 * L[e]], d2 = 2.f * dS[2 * L[e] + 1];
          acc[0] += g_of(v.x, d1, d2);
          acc[1] += g_of(v.y, d1, d2);
          acc[2] += g_of(v.z, d1, d2);
          acc[3] += g_of(v.w, d1, d2);
        }
      } else if (from_phase) {
        for (int u = 0; u < 4; ++u) {
          const int z = z0 + u;
          if (z >= ns) break;
          const int su = min(z / ns_g, nblk - 1), zu = z - su * ns_g;
          const int p0 = op_off[su], p1 = op_off[su + 1];
          for (int f = 0; f < cnt; ++f)
            if (L[f] >= p0 && L[f] < p1)
              acc[u] += g_of(G[(size_t)L[f] * ns_g + zu], dS[2 * L[f]], 2.f * dS[2 * L[f] + 1]);
        }
      } else if (zl + 4 <= ns_g && z0 + 4 <= ns && ((zl & 3) == 0) && ((ns_g & 3) == 0)) {
        for (; e + 3 < e1; e += 4) {
          float4 v[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) v[u] = *(const float4*)(G + (size_t)L[e + u] * ns_g + zl);
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            acc[0] += v[u].x;
            acc[1] += v[u].y;
            acc[2] += v[u].z;
            acc[3] += v[u].w;
          }
        }
        for (; e < e1; ++e) {
          const float4 v = *(const float4*)(G + (size_t)L[e] * ns_g + zl);
          acc[0] += v.x;
          acc[1] += v.y;
          acc[2] += v.z;
          acc[3] += v.w;
        }
      } else {  // a group of 4 samples meeting two sample blocks or the end: per sample
        for (int u = 0; u < 4; ++u) {
          const int z = z0 + u;
          if (z >= ns) break;
          const int su = min(z / ns_g, nblk - 1), zu = z - su * ns_g;
          const int p0 = op_off[su], p1 = op_off[su + 1];
          for (int f = 0; f < cnt; ++f)
            if (L[f] >= p0 && L[f] < p1) acc[u] += G[(size_t)L[f] * ns_g + zu];
        }
      }
    }
  }
  if (z0 < ns) {
    float* q = QT + (size_t)i * ns + z0;
    if (z0 + 4 <= ns && ((((uintptr_t)q) & 15) == 0)) {
      *(float4*)q = make_float4(2.f * acc[0], 2.f * acc[1], 2.f * acc[2], 2.f * acc[3]);
    } else {
      for (int u = 0; u < 4 && z0 + u < ns; ++u) q[u] = 2.f * acc[u];
    }
  }
}

// Op order for the op kernels: decreasing support size (counting sort; the order within a size is
// arbitrary, results do not depend on it). k_sizes: one wave per op. k_order: one block.
__global__ void __launch_bounds__(64) k_sizes(const float* RESTRICT A, int* RESTRICT sz, int nv) {
  const int o = blockIdx.x, lane = threadIdx.x;
  const float* row = A + (size_t)o * nv;
  int c = 0;
  for (int base = 0; base < nv; base += 256) {
    bool on[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int i = base + 64 * u + lane;
      on[u] = (i < nv) && ((((int)row[i]) & 1) != 0);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) c += __popcll(__ballot(on[u]));
  }
  if (lane == 0) sz[o] = c;
}

__global__ void __launch_bounds__(1024) k_order(const int* RESTRICT sz, int* RESTRICT perm, int n_ops, int nb) {
  extern __shared__ int hist[];  // nb buckets (support sizes 0 .. nb - 1)
  for (int b = threadIdx.x; b < nb; b += blockDim.x) hist[b] = 0;
  __syncthreads();
  for (int o = threadIdx.x; o < n_ops; o += blockDim.x) atomicAdd(&hist[min(sz[o], nb - 1)], 1);
  __syncthreads();
  if (threadIdx.x == 0) {  // bucket starts, largest size first
    int acc = 0;
    for (int b = nb - 1; b >= 0; --b) {
      const int c = hist[b];
      hist[b] = acc;
      acc += c;
    }
  }
  __syncthreads();
  for (int o = threadIdx.x; o < n_ops; o += blockDim.x) perm[atomicAdd(&hist[min(sz[o], nb - 1)], 1)] = o;
}

// ------------------------------------------------------------------ group angles
// theta[row, k] = sum_m eff[winv[row, m]] H[k, m] (winv = n_gens: no generator), split into the single /
// pair / >=3-body rows th1 / th2 / th3; Th2[k, i, j] = th2[ptab[i, j], k] (0 where no pair group).
__global__ void __launch_bounds__(256) k_theta(const float* RESTRICT eff, const int* RESTRICT winv,
                                               const float* RESTRICT H, float* RESTRICT th1, float* RESTRICT th2,
                                               float* RESTRICT th3, int n_gens, int nv, int n2, int n3, int nh,
                                               int nseg) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = nv + n2 + n3;
  if (x >= n_rows * nseg) return;
  const int row = x / nseg, k = x - row * nseg;
  float acc = 0.f;
#pragma unroll 8
  for (int m = 0; m < nh; ++m) {
    const int g = winv[(size_t)row * nh + m];
    const float v = eff[min(g, n_gens - 1)];
    acc += (g < n_gens ? v : 0.f) * H[k * nh + m];
  }
  if (row < nv) th1[(size_t)row * nseg + k] = acc;
  else if (row < nv + n2) th2[(size_t)(row - nv) * nseg + k] = acc;
  else th3[(size_t)(row - nv - n2) * nseg + k] = acc;
}

__global__ void __launch_bounds__(256) k_th2(const float* RESTRICT th2, const int* RESTRICT ptab, float* RESTRICT Th2,
                                             int nv, int nseg) {
  const size_t x = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t nn = (size_t)nv * nv;
  if (x >= nn * nseg) return;
  const int k = (int)(x / nn);
  const int pid = ptab[x - (size_t)k * nn];
  Th2[x] = pid >= 0 ? th2[(size_t)pid * nseg + k] : 0.f;
}

// d2[p, k] = dth2[p, k] + dTh2[k, pi, pj] + dTh2[k, pj, pi]
__global__ void __launch_bounds__(256) k_dth2(const float* RESTRICT dth2, const float* RESTRICT dTh2,
                                              const int* RESTRICT pi, const int* RESTRICT pj, float* RESTRICT d2,
                                              int n2, int nv, int nseg, int has_th2) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= n2 * nseg) return;
  const int p = x / nseg, k = x - p * nseg;
  float v = dth2[x];
  if (has_th2) {
    const size_t base = (size_t)k * nv * nv;
    v += dTh2[base + (size_t)pi[p] * nv + pj[p]] + dTh2[base + (size_t)pj[p] * nv + pi[p]];
  }
  d2[x] = v;
}

// deff[g] = sum_k H[k, m] dtheta[row, k] for the slot (row, m) of generator g (0 if it has none)
__global__ void __launch_bounds__(256) k_dgen(const float* RESTRICT dth1, const float* RESTRICT d2,
                                              const float* RESTRICT dth3, const int* RESTRICT gslot,
                                              const float* RESTRICT H, float* RESTRICT deff, int n_gens, int nv, int n2,
                                              int nh, int nseg) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_gens) return;
  const int slot = gslot[g];
  float acc = 0.f;
  if (slot >= 0) {
    const int row = slot / nh, m = slot - row * nh;
    const float* d = row < nv ? dth1 + (size_t)row * nseg
                     : row < nv + n2 ? d2 + (size_t)(row - nv) * nseg : dth3 + (size_t)(row - nv - n2) * nseg;
#pragma unroll 8
    for (int k = 0; k < nseg; ++k) acc += H[k * nh + m] * d[k];
  }
  deff[g] = acc;
}

// ------------------------------------------------------------------ data term
// dataT[i, w]: bit b of word w = low bit of x[32 w + b, i] for x (N, C) int32.  Block = word x 64 columns.
__global__ void __launch_bounds__(256) k_bits(const int* RESTRICT x, uint32_t* RESTRICT dataT, int N, int C, int NW) {
  const int w = blockIdx.x;
  const int il = threadIdx.x & 63;
  const int i = blockIdx.y * 64 + il;
  const int b0 = threadIdx.x >> 6;
  __shared__ uint32_t part[4][64];
  uint32_t word = 0;
  if (i < C)
    for (int b = b0; b < 32; b += 4) {
      const int r = w * 32 + b;
      if (r < N) word |= ((uint32_t)x[(size_t)r * C + i] & 1u) << b;
    }
  part[b0][il] = word;
  __syncthreads();
  if (b0 == 0 && i < C) dataT[(size_t)i * NW + w] = part[0][il] | part[1][il] | part[2][il] | part[3][il];
}

// out[o] = mean over the N data rows of (-1)^(parity of the row on the support of op o)
//        = (N - 2 #odd) / N, the exact value of jnp.mean(1 - 2 ((data @ ops.T) % 2), axis=0).  Block = op.
__global__ void __launch_bounds__(256) k_trdata(const float* RESTRICT A, const uint32_t* RESTRICT dataT,
                                                float* RESTRICT out, int nv, int N, int NW) {
  extern __shared__ int smem[];
  int* S = smem;
  uint32_t* inS = (uint32_t*)(S + nv);
  int* wsh = (int*)(inS + nv / 32 + 1);
  int* red = wsh + 4;
  const int o = blockIdx.x;
  const int w = load_support(A, o, nv, S, inS, wsh, nullptr, 0);
  int cnt = 0;
  for (int j = threadIdx.x; j < NW; j += blockDim.x) {
    uint32_t x = 0;
    for (int p = 0; p < w; ++p) x ^= dataT[(size_t)S[p] * NW + j];
    cnt += __popc(x);
  }
  for (int off = 32; off > 0; off >>= 1) cnt += __shfl_down(cnt, off, 64);
  if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = cnt;
  __syncthreads();
  if (threadIdx.x == 0) {
    int tot = 0;
    for (int u = 0; u < (int)(blockDim.x >> 6); ++u) tot += red[u];
    out[o] = (float)(N - 2 * tot) * (1.0f / (float)N);  // as XLA lowers the mean
  }
}

// ---------------------------------------------------------------- FFI handlers
static ffi::Error PackImpl(hipStream_t st, ffi::Buffer<ffi::S32> samples, ffi::Buffer<ffi::S32> vis,
                           ffi::Buffer<ffi::S32> seg_of, ffi::Buffer<ffi::S32> loc_of,
                           ffi::ResultBuffer<ffi::U32> bitsT, ffi::ResultBuffer<ffi::F32> sTp) {
  const int ns = (int)samples.dimensions()[0], nq = (int)samples.dimensions()[1];
  const int nv = (int)vis.dimensions()[0];  // bitsT has nv + 1 rows
  const int nw = (int)bitsT->dimensions()[1];
  const int c_pad = (int)sTp->dimensions()[2];
  (void)hipMemsetAsync(sTp->typed_data(), 0, sTp->size_bytes(), st);
  dim3 grid(nw, (nv + 64) / 64);
  hipLaunchKernelGGL(k_pack, grid, dim3(256), 0, st, samples.typed_data(), vis.typed_data(), seg_of.typed_data(),
                     loc_of.typed_data(), bitsT->typed_data(), sTp->typed_data(), ns, nq, nw, nv, c_pad);
  return last_error();
}

#define KO_SWITCH(KO, ...)                                                                    \
  switch (KO) {                                                                               \
    case 1: { constexpr int K_ = 1; __VA_ARGS__; } break;                                     \
    case 2: { constexpr int K_ = 2; __VA_ARGS__; } break;                                     \
    case 3: { constexpr int K_ = 3; __VA_ARGS__; } break;                                     \
    case 4: { constexpr int K_ = 4; __VA_ARGS__; } break;                                     \
    case 5: { constexpr int K_ = 5; __VA_ARGS__; } break;                                     \
    case 6: { constexpr int K_ = 6; __VA_ARGS__; } break;                                     \
    case 7: { constexpr int K_ = 7; __VA_ARGS__; } break;                                     \
    default: return ffi::Error(ffi::ErrorCode::kInvalidArgument, "group size out of range"); \
  }

static ffi::Error F3Impl(hipStream_t st, ffi::Buffer<ffi::F32> th3, ffi::Buffer<ffi::U32> bitsT,
                         ffi::Buffer<ffi::S32> w3_off, ffi::Buffer<ffi::S32> w3_v, ffi::Buffer<ffi::S32> w3_o,
                         ffi::Buffer<ffi::S32> slots, ffi::Buffer<ffi::S32> worder, ffi::Buffer<ffi::F32> F2p,
                         ffi::Buffer<ffi::S32> vstart, ffi::ResultBuffer<ffi::F32> out, int32_t ko) {
  const int nv = (int)bitsT.dimensions()[0] - 1, nw = (int)bitsT.dimensions()[1];
  const int ns = (int)out->dimensions()[1];
  const int nseg = (int)th3.dimensions()[1];
  const int nslot = (int)slots.dimensions()[0] / 3;
  const int cw = (int)F2p.dimensions()[2];
  const int nblk = ((int)vstart.dimensions()[0] - 1) / nseg;
  const int c_pad = cw / nblk;
  if (ko + nseg > F3_LDS) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "too many segments");
  KO_SWITCH(ko, hipLaunchKernelGGL(k_f3<K_>, dim3(nv, (nslot + 16 * F3_WAVES - 1) / (16 * F3_WAVES)), dim3(64 * F3_WAVES), 0, st, th3.typed_data(),
                                   bitsT.typed_data(), w3_off.typed_data(), w3_v.typed_data(), w3_o.typed_data(),
                                   slots.typed_data(), worder.typed_data(), F2p.typed_data(), vstart.typed_data(),
                                   out->typed_data(), ns, nw, nseg, nslot, nv, c_pad, cw));
  return last_error();
}

static ffi::Error UnpadImpl(hipStream_t st, ffi::Buffer<ffi::F32> dF, ffi::Buffer<ffi::S32> vstart,
                            ffi::ResultBuffer<ffi::F32> out) {
  const int nv = (int)dF.dimensions()[0], ns = (int)dF.dimensions()[1];
  const int nseg = (int)out->dimensions()[0], cw = (int)out->dimensions()[2];
  const int nblk = ((int)vstart.dimensions()[0] - 1) / nseg;
  hipLaunchKernelGGL(k_unpad, dim3(nseg * nv, (cw + 255) / 256), dim3(256), 0, st, dF.typed_data(), vstart.typed_data(),
                     out->typed_data(), ns, nv, nseg, cw / nblk, cw);
  return last_error();
}

static ffi::Error D3Impl(hipStream_t st, ffi::Buffer<ffi::F32> dF, ffi::Buffer<ffi::U32> bitsT,
                         ffi::Buffer<ffi::S32> w3_off, ffi::Buffer<ffi::S32> w3_v, ffi::Buffer<ffi::S32> w3_o,
                         ffi::Buffer<ffi::S32> slots, ffi::Buffer<ffi::S32> seg_slot, ffi::Buffer<ffi::S32> worder,
                         ffi::Buffer<ffi::S32> w3_ol, ffi::Buffer<ffi::S32> pw, ffi::Buffer<ffi::S32> pw_off,
                         ffi::ResultBuffer<ffi::F32> out, int32_t ko, int32_t np_max) {
  const int nv = (int)bitsT.dimensions()[0] - 1, nw = (int)bitsT.dimensions()[1];
  const int ns = (int)dF.dimensions()[1];
  const int nseg = (int)out->dimensions()[1];
  const int nvseg = (int)seg_slot.dimensions()[0] - 1;
  const int nblk = (nvseg + nseg - 1) / nseg;
  (void)hipMemsetAsync(out->typed_data(), 0, out->size_bytes(), st);
  // staged variant when the partner bit rows, the dF words and the tables of a sample block fit in LDS
  const int nwb_max = (ns / nblk + 31) / 32 + 2;
  const int nsl_max = (int)slots.dimensions()[0] / 3;  // bound on the slots of one sample block
  const size_t lds_s = sizeof(uint32_t) * ((size_t)(np_max + 2) * (nwb_max | 1) + 2 * (size_t)nsl_max)
                     + sizeof(float) * ((size_t)nwb_max * 32 + 128 * D3S_LCH);
  if (np_max > 0 && lds_s <= 40 * 1024) {
    KO_SWITCH(ko, hipLaunchKernelGGL(k_d3s<K_>, dim3(nv, nblk), dim3(D3S_NB), lds_s, st, dF.typed_data(),
                                     bitsT.typed_data(), w3_off.typed_data(), w3_v.typed_data(), w3_ol.typed_data(),
                                     pw_off.typed_data(), pw.typed_data(), slots.typed_data(), seg_slot.typed_data(),
                                     worder.typed_data(), out->typed_data(), ns, nw, nseg, nvseg));
    return last_error();
  }
  const size_t lds = sizeof(float) * (size_t)ns + sizeof(uint32_t) * (size_t)nw;
  const int stage = lds <= 48 * 1024;
  KO_SWITCH(ko, hipLaunchKernelGGL(k_d3<K_>, dim3(nv, nblk), dim3(512), stage ? lds : 0, st, dF.typed_data(),
                                   bitsT.typed_data(), w3_off.typed_data(), w3_v.typed_data(), w3_o.typed_data(),
                                   slots.typed_data(), seg_slot.typed_data(), worder.typed_data(), out->typed_data(), ns,
                                   nw, nseg, nvseg, stage));
  return last_error();
}

static ffi::Error PhaseImpl(hipStream_t st, ffi::Buffer<ffi::F32> A, ffi::Buffer<ffi::U32> bitsT,
                            ffi::Buffer<ffi::F32> F, ffi::Buffer<ffi::F32> th2, ffi::Buffer<ffi::F32> th3,
                            ffi::Buffer<ffi::S32> ptab, ffi::Buffer<ffi::S32> p3_off, ffi::Buffer<ffi::S32> p3_v,
                            ffi::Buffer<ffi::S32> p3_o, ffi::Buffer<ffi::S32> slots, ffi::Buffer<ffi::S32> seg_slot,
                            ffi::Buffer<ffi::S32> op_sig, ffi::Buffer<ffi::S32> perm, ffi::ResultBuffer<ffi::F32> out,
                            ffi::ResultBuffer<ffi::U32> AT, ffi::ResultBuffer<ffi::F32> sums, int32_t has2, int32_t has3,
                            int32_t ko2) {
  const int n_ops = (int)A.dimensions()[0], nv = (int)A.dimensions()[1];
  const int nw = (int)bitsT.dimensions()[1];
  const int ns = (int)F.dimensions()[1];
  const int ns_o = (int)out->dimensions()[1];
  const int nseg = (int)th2.dimensions()[1];
  const int nwo = (int)AT->dimensions()[1];
  (void)hipMemsetAsync(AT->typed_data(), 0, AT->size_bytes(), st);
  KO_SWITCH(ko2, hipLaunchKernelGGL(k_phase<K_>, dim3(n_ops), dim3(PH_NB), sizeof(int) * (op_lds_ints(nv, PH_NB) + (nseg <= TH_STAGE ? PH_NB * nseg : 0)), st, A.typed_data(),
                                    bitsT.typed_data(), F.typed_data(), th2.typed_data(), th3.typed_data(),
                                    ptab.typed_data(), p3_off.typed_data(), p3_v.typed_data(), p3_o.typed_data(),
                                    slots.typed_data(), seg_slot.typed_data(), op_sig.typed_data(), perm.typed_data(),
                                    out->typed_data(), AT->typed_data(), sums->typed_data(), nv, ns, ns_o, nw, nseg, nwo,
                                    has2, has3));
  return last_error();
}

static ffi::Error PhaseBwdImpl(hipStream_t st, ffi::Buffer<ffi::F32> A, ffi::Buffer<ffi::U32> bitsT,
                               ffi::Buffer<ffi::F32> G, ffi::Buffer<ffi::U32> AT, ffi::Buffer<ffi::S32> ptab,
                               ffi::Buffer<ffi::S32> p3_off, ffi::Buffer<ffi::S32> p3_v, ffi::Buffer<ffi::S32> p3_o,
                               ffi::Buffer<ffi::S32> slots, ffi::Buffer<ffi::S32> seg_slot, ffi::Buffer<ffi::S32> op_sig,
                               ffi::Buffer<ffi::S32> op_off, ffi::Buffer<ffi::S32> perm, ffi::Buffer<ffi::F32> dS,
                               ffi::ResultBuffer<ffi::F32> QT, ffi::ResultBuffer<ffi::F32> dth2,
                               ffi::ResultBuffer<ffi::F32> dth3, int32_t has2, int32_t has3, int32_t ko2,
                               int32_t from_phase) {
  const int n_ops = (int)A.dimensions()[0], nv = (int)A.dimensions()[1];
  const int nw = (int)bitsT.dimensions()[1];
  const int ns = (int)G.dimensions()[1];  // local (per sample block) row length
  const int ns_q = (int)QT->dimensions()[1];
  const int nblk = (int)op_off.dimensions()[0] - 1;
  const int nseg = (int)dth2->dimensions()[1];
  const int nwo = (int)AT.dimensions()[1];
  (void)hipMemsetAsync(dth2->typed_data(), 0, dth2->size_bytes(), st);
  (void)hipMemsetAsync(dth3->typed_data(), 0, dth3->size_bytes(), st);
  hipLaunchKernelGGL(k_q, dim3(nv, (ns_q + Q_ZB - 1) / Q_ZB), dim3(Q_NB), 0, st, AT.typed_data(), G.typed_data(),
                     op_off.typed_data(), dS.typed_data(), QT->typed_data(), ns_q, ns, nwo, nblk, from_phase);
  if (has2 || has3) {
    size_t lds = sizeof(int) * op_lds_ints(nv, PB_NB);
    const int stage = lds + sizeof(float) * (size_t)ns <= 32 * 1024;
    if (stage) lds += sizeof(float) * (size_t)ns;
    KO_SWITCH(ko2, hipLaunchKernelGGL(k_phase_bwd<K_>, dim3(n_ops), dim3(PB_NB), lds, st, A.typed_data(),
                                      bitsT.typed_data(), G.typed_data(), ptab.typed_data(), p3_off.typed_data(),
                                      p3_v.typed_data(), p3_o.typed_data(), slots.typed_data(), seg_slot.typed_data(),
                                      op_sig.typed_data(), perm.typed_data(), dS.typed_data(), dth2->typed_data(),
                                      dth3->typed_data(), nv, ns, nw, nseg, has2, has3, stage, from_phase));
  }
  return last_error();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpPack, PackImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::U32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpF3, F3Impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("ko"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpD3, D3Impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("ko")
                                  .Attr<int32_t>("np_max"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpPhase, PhaseImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::U32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("has2")
                                  .Attr<int32_t>("has3")
                                  .Attr<int32_t>("ko2"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpPhaseBwd, PhaseBwdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("has2")
                                  .Attr<int32_t>("has3")
                                  .Attr<int32_t>("ko2")
                                  .Attr<int32_t>("from_phase"));

static ffi::Error BitsImpl(hipStream_t st, ffi::Buffer<ffi::S32> x, ffi::ResultBuffer<ffi::U32> dataT) {
  const int N = (int)x.dimensions()[0], C = (int)x.dimensions()[1];
  const int NW = (int)dataT->dimensions()[1];
  dim3 grid(NW, (C + 63) / 64);
  hipLaunchKernelGGL(k_bits, grid, dim3(256), 0, st, x.typed_data(), dataT->typed_data(), N, C, NW);
  return last_error();
}

static ffi::Error TrDataImpl(hipStream_t st, ffi::Buffer<ffi::F32> A, ffi::Buffer<ffi::U32> dataT,
                             ffi::ResultBuffer<ffi::F32> out, int32_t n_rows) {
  const int n_ops = (int)A.dimensions()[0], nv = (int)A.dimensions()[1];
  const int NW = (int)dataT.dimensions()[1];
  const size_t lds = sizeof(int) * ((size_t)nv + nv / 32 + 1 + 4 + 16);
  hipLaunchKernelGGL(k_trdata, dim3(n_ops), dim3(256), lds, st, A.typed_data(), dataT.typed_data(), out->typed_data(),
                     nv, n_rows, NW);
  return last_error();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpBits, BitsImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::U32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpTrData, TrDataImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("n_rows"));

static ffi::Error OrderImpl(hipStream_t st, ffi::Buffer<ffi::F32> A, ffi::ResultBuffer<ffi::S32> perm,
                            ffi::ResultBuffer<ffi::S32> sz) {
  const int n_ops = (int)A.dimensions()[0], nv = (int)A.dimensions()[1];
  hipLaunchKernelGGL(k_sizes, dim3(n_ops), dim3(64), 0, st, A.typed_data(), sz->typed_data(), nv);
  const int nb = nv + 1;
  hipLaunchKernelGGL(k_order, dim3(1), dim3(1024), sizeof(int) * nb, st, sz->typed_data(), perm->typed_data(), n_ops, nb);
  return last_error();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpOrder, OrderImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpUnpad, UnpadImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>());

static ffi::Error ThetaImpl(hipStream_t st, ffi::Buffer<ffi::F32> eff, ffi::Buffer<ffi::S32> winv, ffi::Buffer<ffi::F32> H,
                            ffi::Buffer<ffi::S32> ptab, ffi::ResultBuffer<ffi::F32> th1, ffi::ResultBuffer<ffi::F32> th2,
                            ffi::ResultBuffer<ffi::F32> th3, ffi::ResultBuffer<ffi::F32> Th2, int32_t n2, int32_t n3) {
  const int n_gens = (int)eff.dimensions()[0];
  const int nv = (int)th1->dimensions()[0], nseg = (int)th1->dimensions()[1];
  const int nh = (int)H.dimensions()[1];
  const int nx = (nv + n2 + n3) * nseg;
  hipLaunchKernelGGL(k_theta, dim3((nx + 255) / 256), dim3(256), 0, st, eff.typed_data(), winv.typed_data(), H.typed_data(),
                     th1->typed_data(), th2->typed_data(), th3->typed_data(), n_gens, nv, n2, n3, nh, nseg);
  if (n2 > 0 && Th2->dimensions()[1] == nv) {
    const size_t n = (size_t)nseg * nv * nv;
    hipLaunchKernelGGL(k_th2, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, st, th2->typed_data(), ptab.typed_data(),
                       Th2->typed_data(), nv, nseg);
  }
  return last_error();
}

static ffi::Error ThetaBwdImpl(hipStream_t st, ffi::Buffer<ffi::F32> dth1, ffi::Buffer<ffi::F32> dth2,
                               ffi::Buffer<ffi::F32> dth3, ffi::Buffer<ffi::F32> dTh2, ffi::Buffer<ffi::S32> gslot,
                               ffi::Buffer<ffi::F32> H, ffi::Buffer<ffi::S32> pi, ffi::Buffer<ffi::S32> pj,
                               ffi::ResultBuffer<ffi::F32> deff, ffi::ResultBuffer<ffi::F32> d2, int32_t n2) {
  const int n_gens = (int)deff->dimensions()[0];
  const int nv = (int)dth1.dimensions()[0], nseg = (int)dth1.dimensions()[1];
  const int nh = (int)H.dimensions()[1];
  const int has_th2 = dTh2.dimensions()[1] == nv;
  if (n2 > 0)
    hipLaunchKernelGGL(k_dth2, dim3((n2 * nseg + 255) / 256), dim3(256), 0, st, dth2.typed_data(), dTh2.typed_data(),
                       pi.typed_data(), pj.typed_data(), d2->typed_data(), n2, nv, nseg, has_th2);
  hipLaunchKernelGGL(k_dgen, dim3((n_gens + 255) / 256), dim3(256), 0, st, dth1.typed_data(), d2->typed_data(),
                     dth3.typed_data(), gslot.typed_data(), H.typed_data(), deff->typed_data(), n_gens, nv, n2, nh, nseg);
  return last_error();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpTheta, ThetaImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("n2")
                                  .Attr<int32_t>("n3"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpThetaBwd, ThetaBwdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("n2"));

// ---------------------------------------------------------------- pair-angle products (rocBLAS)
// M[k] = Th2[k] @ X[k] and its transpose dTh2[k] = dM[k] @ X[k]^T for the segments k (row-major), as strided
// batched fp32 GEMMs on a handle created at load time (iqp_init_blas).
static rocblas_handle g_rb = nullptr;

extern "C" int iqp_init_blas() {
  if (g_rb) return 0;
  if (rocblas_create_handle(&g_rb) != rocblas_status_success) return 1;
  rocblas_set_math_mode(g_rb, rocblas_default_math);  // full fp32
  rocblas_set_device_memory_size(g_rb, 0);  // no 128 MiB workspace pool (sgemm needs none): device memory as before
  return 0;
}

static ffi::Error PairMMImpl(hipStream_t st, ffi::Buffer<ffi::F32> Th2, ffi::Buffer<ffi::F32> X,
                             ffi::ResultBuffer<ffi::F32> M) {
  const int nseg = (int)Th2.dimensions()[0], nv = (int)Th2.dimensions()[1];
  const int C = (int)X.dimensions()[2];
  if (!g_rb) return ffi::Error(ffi::ErrorCode::kInternal, "rocblas handle");
  rocblas_set_stream(g_rb, st);
  const float one = 1.f, zero = 0.f;
  // column-major: M^T (C x nv) = X^T (C x nv) @ Th2^T (nv x nv)
  const rocblas_status s = rocblas_sgemm_strided_batched(
      g_rb, rocblas_operation_none, rocblas_operation_none, C, nv, nv, &one, X.typed_data(), C, (rocblas_stride)nv * C,
      Th2.typed_data(), nv, (rocblas_stride)nv * nv, &zero, M->typed_data(), C, (rocblas_stride)nv * C, nseg);
  if (s != rocblas_status_success) return ffi::Error(ffi::ErrorCode::kInternal, "rocblas_sgemm_strided_batched");
  return last_error();
}

static ffi::Error PairMMBwdImpl(hipStream_t st, ffi::Buffer<ffi::F32> dM, ffi::Buffer<ffi::F32> X,
                                ffi::ResultBuffer<ffi::F32> dTh2) {
  const int nseg = (int)dTh2->dimensions()[0], nv = (int)dTh2->dimensions()[1];
  const int C = (int)X.dimensions()[2];
  if (!g_rb) return ffi::Error(ffi::ErrorCode::kInternal, "rocblas handle");
  rocblas_set_stream(g_rb, st);
  const float one = 1.f, zero = 0.f;
  // column-major: dTh2^T (nv x nv) = (X^T)^T (nv x C) @ dM^T (C x nv), i.e. row-major dTh2 = dM X^T
  const rocblas_status s = rocblas_sgemm_strided_batched(
      g_rb, rocblas_operation_transpose, rocblas_operation_none, nv, nv, C, &one, X.typed_data(), C,
      (rocblas_stride)nv * C, dM.typed_data(), C, (rocblas_stride)nv * C, &zero, dTh2->typed_data(), nv,
      (rocblas_stride)nv * nv, nseg);
  if (s != rocblas_status_success) return ffi::Error(ffi::ErrorCode::kInternal, "rocblas_sgemm_strided_batched");
  return last_error();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpPairMM, PairMMImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(IqpPairMMBwd, PairMMBwdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>());
