// SPDX-License-Identifier: Apache-2.0
// trellis-coded linear (GGML_OP_TRELLIS, see ggml_trellis_mul_mat) on HIP.
//
// Plan 1: scalar decode, no matrix cores. The structure mirrors ggml-metal/kernels/trellis.metal
// rather than libtrellis, so nothing here depends on NVIDIA's m16n8k16 fragment layout:
//
//   input (gather + pre) -> input side -> product (decode on the fly) -> output side -> output (post + gather)
//
// Weights stay compressed in device memory; no dense weight is ever materialised.
// The decode is bit-exact against ggml_trellis_decode_rows (ggml/src/ggml-trellis.cpp).
// See nariq/FORMAT.md for the layout and PLAN-1-scalar-hip.md for the port plan.
#pragma once

#include "common.cuh"
#include "ggml.h"

#if defined(GGML_USE_HIP)
#include "mma.cuh"
#endif

#include <cmath>
#include <cstdlib>

namespace trellis_hip {

// tokens per chunk; scratch buffers are reused for every chunk
constexpr int BATCH = 512;
// largest blockwise Hadamard done in one workgroup's LDS (4096 floats = 16 KiB)
constexpr int BLOCK_LDS_MAX = 4096;

// ---- the op --------------------------------------------------------------------------------------------

struct op_t {
    const ggml_tensor * w, * x, * s16, * pre, * post, * pin, * pout;
    int rows_kind, rows_block, rows_shards, cols_kind, cols_block, cols_shards, nparts;
    const int32_t * parts;
    int K, N, bits;
    int64_t T;

    explicit op_t(const ggml_tensor * dst) {
        w    = dst->src[0];
        x    = dst->src[1];
        s16  = dst->src[2];
        pre  = dst->src[3];
        post = dst->src[4];
        pin  = dst->src[5];
        pout = dst->src[6];
        const int32_t * d = (const int32_t *) dst->op_params;
        rows_kind   = d[0];
        rows_block  = d[1];
        rows_shards = d[2];
        cols_kind   = d[3];
        cols_block  = d[4];
        cols_shards = d[5];
        nparts      = d[6];
        parts       = d + 7;
        K    = (int) w->ne[0];
        N    = (int) w->ne[1];
        T    = x->ne[1] * x->ne[2] * x->ne[3];
        bits = (int) w->type - (int) GGML_TYPE_TRL1 + 1;
    }
};

// ---- decode --------------------------------------------------------------------------------------------

// The 16-bit trellis state of ring position p: the window of the tile's circular 256*bits stream
// ending at bit (p+1)*bits. Stream bit q is bit 31 - q%32 of word q/32, so the window is taken
// MSB-first from the word pair (hi, lo).
template <int BITS>
static __device__ __forceinline__ uint32_t state_at(const uint32_t * tile, int word, int off) {
    constexpr int NW = 8 * BITS;
    const uint32_t hi = tile[word], lo = tile[(word + 1) % NW];
    return (off <= 16 ? (hi >> (16 - off)) : (hi << (off - 16)) | (lo >> (48 - off))) & 0xFFFFu;
}

// the INT8 grid: x = state * mult; v = sum of the four 6-bit digits of x, minus 126.
// Each digit is 0..63, so the byte fields of (x >> 2) are 0..63 and their sum is <= 252: the
// multiply by 0x01010101 accumulates them into the top byte without carries. Exact, |v| <= 126.
//
// This must be an UNSIGNED byte sum. ggml_cuda_dp4a() is the signed form and would read digits
// >= 32 as negative (libtrellis uses the unsigned __dp4a overload for the same reason).
static __device__ __forceinline__ int grid_int(uint32_t st) {
    const uint32_t x = st * 0x83DCD12Du;
    const uint32_t digits = (x >> 2) & 0x3F3F3F3Fu;
    return (int) __builtin_amdgcn_sad_u8(digits, 0u, (uint32_t) -126);
}

static __device__ __forceinline__ float grid_value(uint32_t st) {
    return (float) grid_int(st);
}

// ring position of element (r, c) within a 16x16 tile; the inverse of the reference map
// p = 8a + j -> row = a/4 + 8*((j>>2)&1), col = 2*(a%4) + (j&1) + 8*((j>>1)&1)
static __device__ __forceinline__ int ring_pos(int r, int c) {
    return 8 * (4 * (r % 8) + (c % 8) / 2) + 4 * (r / 8) + 2 * (c / 8) + c % 2;
}

// ---- side transforms -----------------------------------------------------------------------------------

struct paley_table {
    int8_t h[68 * 68];
};

static constexpr int paley_chi(int d, int q) {
    d = (d % q + q) % q;
    if (d == 0) {
        return 0;
    }
    for (int i = 1; i < q; ++i) {
        if ((i * i) % q == d) {
            return 1;
        }
    }
    return -1;
}

static constexpr paley_table make_paley(int m) {
    paley_table p{};
    const int q = m - 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            const int s = i == 0 ? (j > 0 ? 1 : 0) : (j == 0 ? -1 : paley_chi(j - i, q));
            p.h[i * m + j] = (int8_t) (s + (i == j ? 1 : 0));
        }
    }
    return p;
}

__constant__ paley_table k_h12 = make_paley(12);
__constant__ paley_table k_h20 = make_paley(20);
__constant__ paley_table k_h68 = make_paley(68);

static __device__ __forceinline__ int paley_entry(int m, int r, int c) {
    return m == 12 ? k_h12.h[r * 12 + c] : (m == 20 ? k_h20.h[r * 20 + c] : k_h68.h[r * 68 + c]);
}

struct hf_parts {
    int n;
    int start[GGML_TRELLIS_MAX_PARTS];
    int shard[GGML_TRELLIS_MAX_PARTS];
    int cum[GGML_TRELLIS_MAX_PARTS + 1];
};

static inline hf_parts make_hf_parts(int n, int shards, const int32_t * parts, int nparts) {
    hf_parts p = {};
    p.n = (parts != nullptr && nparts > 0) ? nparts : 1;
    int start = 0, cum = 0;
    for (int i = 0; i < p.n; ++i) {
        const int sz = (parts != nullptr && nparts > 0) ? parts[i] : n;
        p.start[i] = start;
        p.shard[i] = sz / shards;
        p.cum[i]   = cum;
        start += sz;
        cum   += sz / shards;
    }
    p.cum[p.n] = cum;
    return p;
}

static __device__ __forceinline__ int64_t stored_part(const hf_parts & p, int shard, int l) {
    int start = p.start[0], sh = p.shard[0], cum = p.cum[0];
#pragma unroll
    for (int i = 1; i < GGML_TRELLIS_MAX_PARTS; ++i) {
        if (i < p.n && l >= p.cum[i]) {
            start = p.start[i];
            sh    = p.shard[i];
            cum   = p.cum[i];
        }
    }
    return start + (int64_t) shard * sh + (l - cum);
}

// Wave-shuffle + register Hadamard for p <= 32*E elements (lane l, reg e -> l + 32*e).
template <int E>
static __device__ __forceinline__ void hadamard_warp_stages(float (&v)[E], int p, int lane) {
    const int hmax = p < 32 * E ? p : 32 * E;
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        if (h < hmax) {
            const bool hi = (lane & h) != 0;
#pragma unroll
            for (int e = 0; e < E; ++e) {
                const float o = ggml_cuda_shfl_xor(v[e], h);
                v[e] = hi ? (o - v[e]) : (v[e] + o);
            }
        }
    }
#pragma unroll
    for (int g = 1; g < E; g <<= 1) {
        if (32 * g < hmax) {
#pragma unroll
            for (int e = 0; e < E; ++e) {
                if (!(e & g)) {
                    const float a = v[e], b = v[e + g];
                    v[e]     = a + b;
                    v[e + g] = a - b;
                }
            }
        }
    }
}

// Wave-shuffle + register Hadamard for 128 elements (32 lanes x 4 elements: k = lane + 32*e).
static __device__ __forceinline__ void hadamard_128_warp(float (&v)[4], int lane) {
    hadamard_warp_stages<4>(v, 128, lane);
}

// 2-barrier padded-LDS transpose + wave-shuffle Hadamard for B in {256, 512, 1024, 2048, 4096}.
// Leaves unnormalized H_B in tile[k / 32][k % 32] past the second barrier.
template <int B, int THREADS>
static __device__ __forceinline__ void rht_row_stages(
        const float * __restrict__ src, const int32_t * __restrict__ pin, const float * __restrict__ pre,
        int b0, float (&tile)[B / 32][33]) {
    constexpr int VALUES = B / THREADS;
    constexpr int COL = B < 1024 ? B / 32 : 32;
    const int tid = threadIdx.x, lane = tid & 31;
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int k = tid + i * THREADS;
        float v = src[pin ? pin[b0 + k] : (b0 + k)];
        if (pre) {
            v *= pre[b0 + k];
        }
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float o = ggml_cuda_shfl_xor(v, h);
            v = (lane & h) ? (o - v) : (v + o);
        }
        tile[k >> 5][lane] = v;
    }
    __syncthreads();
    float v[VALUES];
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int t = tid + i * THREADS;
        v[i] = tile[(t / (32 * COL)) * COL + (t % COL)][(t / COL) & 31];
#pragma unroll
        for (int h = 1; h < COL; h <<= 1) {
            const float o = ggml_cuda_shfl_xor(v[i], h);
            v[i] = (lane & h) ? (o - v[i]) : (v[i] + o);
        }
    }
#pragma unroll
    for (int step = 1024 / THREADS; step < VALUES; step <<= 1) {
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            if (!(i & step)) {
                const float a = v[i], b = v[i + step];
                v[i]        = a + b;
                v[i + step] = a - b;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int t = tid + i * THREADS;
        tile[(t / (32 * COL)) * COL + (t % COL)][(t / COL) & 31] = v[i];
    }
    __syncthreads();
}

// Fast blockwise input Hadamard for B = 128 (0 LDS, 0 barriers) and B in {256..4096} (2 barriers).
static __global__ void k_input_rht_128(
        int K, float scale, int first, int ne1, int ne2, size_t nb1, size_t nb2, size_t nb3,
        const char * __restrict__ x, const float * __restrict__ pre, const int32_t * __restrict__ pin,
        float * __restrict__ dst) {
    const int lane = threadIdx.x & 31;
    const int b0 = blockIdx.x * 128;
    const int64_t c = first + blockIdx.y;
    const float * row = (const float *) (x + (c % ne1) * nb1 + ((c / ne1) % ne2) * nb2 + (c / ((int64_t) ne1 * ne2)) * nb3);
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int k = b0 + lane + 32 * e;
        v[e] = row[pin ? pin[k] : k] * pre[k];
    }
    hadamard_128_warp(v, lane);
    const int64_t dst_base = (int64_t) blockIdx.y * K + b0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        dst[dst_base + lane + 32 * e] = v[e] * scale;
    }
}

template <int B, int THREADS>
static __global__ void __launch_bounds__(THREADS) k_input_rht(
        int K, float scale, int first, int ne1, int ne2, size_t nb1, size_t nb2, size_t nb3,
        const char * __restrict__ x, const float * __restrict__ pre, const int32_t * __restrict__ pin,
        float * __restrict__ dst) {
    __shared__ float tile[B / 32][33];
    const int tid = threadIdx.x, lane = tid & 31;
    const int b0 = blockIdx.x * B;
    const int64_t c = first + blockIdx.y;
    const float * row = (const float *) (x + (c % ne1) * nb1 + ((c / ne1) % ne2) * nb2 + (c / ((int64_t) ne1 * ne2)) * nb3);
    rht_row_stages<B, THREADS>(row, pin, pre, b0, tile);
    const int64_t dst_base = (int64_t) blockIdx.y * K + b0;
#pragma unroll
    for (int i = 0; i < B / THREADS; ++i) {
        const int k = tid + i * THREADS;
        dst[dst_base + k] = tile[k >> 5][lane] * scale;
    }
}

// Fast blockwise output Hadamard for B = 128 and B in {256..4096}; fuses post even when src == dst.
static __global__ void k_output_rht_128(
        int N, float scale, int first, const float * __restrict__ src,
        const float * __restrict__ post, float * __restrict__ dst) {
    const int lane = threadIdx.x & 31;
    const int b0 = blockIdx.x * 128;
    const int64_t src_base = (int64_t) blockIdx.y * N + b0;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        v[e] = src[src_base + lane + 32 * e];
    }
    hadamard_128_warp(v, lane);
    const int64_t dst_base = (int64_t) (first + blockIdx.y) * N + b0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = lane + 32 * e;
        dst[dst_base + i] = post ? v[e] * scale * post[b0 + i] : v[e] * scale;
    }
}

template <int B, int THREADS>
static __global__ void __launch_bounds__(THREADS) k_output_rht(
        int N, float scale, int first, const float * __restrict__ src,
        const float * __restrict__ post, float * __restrict__ dst) {
    __shared__ float tile[B / 32][33];
    const int tid = threadIdx.x, lane = tid & 31;
    const int b0 = blockIdx.x * B;
    const float * srow = src + (int64_t) blockIdx.y * N;
    rht_row_stages<B, THREADS>(srow, nullptr, nullptr, b0, tile);
    const int64_t dst_base = (int64_t) (first + blockIdx.y) * N + b0;
#pragma unroll
    for (int i = 0; i < B / THREADS; ++i) {
        const int k = tid + i * THREADS;
        const float z = tile[k >> 5][lane] * scale;
        dst[dst_base + k] = post ? z * post[b0 + k] : z;
    }
}

// Warp-only blockwise Hadamard for small blocks (16, 32, 64), 0 LDS.
template <int E>
static __global__ void __launch_bounds__(256) k_block_warp(
        int n, int rows, int block, float scale, int first_dst,
        const float * __restrict__ src, const float * __restrict__ post, float * __restrict__ dst) {
    constexpr int S = 32 * E;
    const int lane = threadIdx.x & 31;
    const int chunks = (n + S - 1) / S;
    const int64_t item = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (item >= (int64_t) rows * chunks) {
        return;
    }
    const int row = (int) (item / chunks);
    const int k0 = (int) (item % chunks) * S;
    const float * srow = src + (int64_t) row * n;
    float v[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int k = k0 + lane + 32 * e;
        v[e] = k < n ? srow[k] : 0.0f;
    }
    hadamard_warp_stages<E>(v, block, lane);
    float * drow = dst + (int64_t) (first_dst + row) * n;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int k = k0 + lane + 32 * e;
        if (k < n) {
            drow[k] = post ? v[e] * scale * post[k] : v[e] * scale;
        }
    }
}

// Fused MLP middle: output Hadamard + post on gate & up, SwiGLU, pre + input Hadamard for down.
static __global__ void k_swiglu_rht_128(
        const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ xd,
        int n, float scale,
        const float * __restrict__ post_g, const float * __restrict__ post_u, const float * __restrict__ pre_d) {
    const int lane = threadIdx.x & 31;
    const int b0 = blockIdx.x * 128;
    const int64_t row_base = (int64_t) blockIdx.y * n + b0;
    float vg[4], vu[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = lane + 32 * e;
        vg[e] = g[row_base + i];
        vu[e] = u[row_base + i];
    }
    hadamard_128_warp(vg, lane);
    hadamard_128_warp(vu, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = lane + 32 * e;
        const float gv = vg[e] * scale * post_g[b0 + i];
        const float uv = vu[e] * scale * post_u[b0 + i];
        const float silu = gv / (1.0f + expf(-gv));
        vg[e] = (silu * uv) * pre_d[b0 + i];
    }
    hadamard_128_warp(vg, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        xd[row_base + lane + 32 * e] = vg[e] * scale;
    }
}

template <int B, int THREADS>
static __global__ void __launch_bounds__(THREADS) k_swiglu_rht(
        const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ xd,
        int n, float scale,
        const float * __restrict__ post_g, const float * __restrict__ post_u, const float * __restrict__ pre_d) {
    __shared__ float tg[B / 32][33];
    __shared__ float tu[B / 32][33];
    constexpr int VALUES = B / THREADS;
    constexpr int COL = B < 1024 ? B / 32 : 32;
    const int tid = threadIdx.x, lane = tid & 31;
    const int b0 = blockIdx.x * B;
    const int64_t row_base = (int64_t) blockIdx.y * n + b0;

#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int k = tid + i * THREADS;
        float vg = g[row_base + k];
        float vu = u[row_base + k];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float og = ggml_cuda_shfl_xor(vg, h);
            const float ou = ggml_cuda_shfl_xor(vu, h);
            vg = (lane & h) ? (og - vg) : (vg + og);
            vu = (lane & h) ? (ou - vu) : (vu + ou);
        }
        tg[k >> 5][lane] = vg;
        tu[k >> 5][lane] = vu;
    }
    __syncthreads();
    float vg[VALUES], vu[VALUES];
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int t = tid + i * THREADS;
        const int r0 = (t / (32 * COL)) * COL + (t % COL);
        const int c0 = (t / COL) & 31;
        vg[i] = tg[r0][c0];
        vu[i] = tu[r0][c0];
#pragma unroll
        for (int h = 1; h < COL; h <<= 1) {
            const float og = ggml_cuda_shfl_xor(vg[i], h);
            const float ou = ggml_cuda_shfl_xor(vu[i], h);
            vg[i] = (lane & h) ? (og - vg[i]) : (vg[i] + og);
            vu[i] = (lane & h) ? (ou - vu[i]) : (vu[i] + ou);
        }
    }
#pragma unroll
    for (int step = 1024 / THREADS; step < VALUES; step <<= 1) {
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            if (!(i & step)) {
                const float ag = vg[i], bg = vg[i + step];
                vg[i] = ag + bg; vg[i + step] = ag - bg;
                const float au = vu[i], bu = vu[i + step];
                vu[i] = au + bu; vu[i + step] = au - bu;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int t = tid + i * THREADS;
        const int r0 = (t / (32 * COL)) * COL + (t % COL);
        const int c0 = (t / COL) & 31;
        tg[r0][c0] = vg[i];
        tu[r0][c0] = vu[i];
    }
    __syncthreads();

    // SwiGLU + pre_d + first stage of down input Hadamard (overwriting tg)
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int k = tid + i * THREADS;
        const float gv = tg[k >> 5][lane] * scale * post_g[b0 + k];
        const float uv = tu[k >> 5][lane] * scale * post_u[b0 + k];
        float v = (gv / (1.0f + expf(-gv)) * uv) * pre_d[b0 + k];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float o = ggml_cuda_shfl_xor(v, h);
            v = (lane & h) ? (o - v) : (v + o);
        }
        vg[i] = v;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int k = tid + i * THREADS;
        tg[k >> 5][lane] = vg[i];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int t = tid + i * THREADS;
        vg[i] = tg[(t / (32 * COL)) * COL + (t % COL)][(t / COL) & 31];
#pragma unroll
        for (int h = 1; h < COL; h <<= 1) {
            const float o = ggml_cuda_shfl_xor(vg[i], h);
            vg[i] = (lane & h) ? (o - vg[i]) : (vg[i] + o);
        }
    }
#pragma unroll
    for (int step = 1024 / THREADS; step < VALUES; step <<= 1) {
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            if (!(i & step)) {
                const float a = vg[i], b = vg[i + step];
                vg[i] = a + b; vg[i + step] = a - b;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int t = tid + i * THREADS;
        tg[(t / (32 * COL)) * COL + (t % COL)][(t / COL) & 31] = vg[i];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int k = tid + i * THREADS;
        xd[row_base + k] = tg[k >> 5][lane] * scale;
    }
}

// Fallback blockwise Hadamard in LDS for tiny blocks (2, 4, 8) or blocks > 4096 (8192, 16384).
static __global__ void k_block(int n, int len, float scale, float * v) {
    extern __shared__ float sm[];
    const int64_t base = (int64_t) blockIdx.y * n + (int64_t) blockIdx.x * len;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        sm[i] = v[base + i];
    }
    __syncthreads();
    for (int h = 1; h < len; h <<= 1) {
        const int sh = __ffs(h) - 1;
        for (int j = threadIdx.x; j < (len >> 1); j += blockDim.x) {
            const int i = ((j >> sh) << (sh + 1)) | (j & (h - 1));
            const float x = sm[i], y = sm[i + h];
            sm[i]     = x + y;
            sm[i + h] = x - y;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        v[base + i] = sm[i] * scale;
    }
}

// Single-CTA LDS forward hadfull (len <= 12288): fuses x read + pin + pre, runs, and Paley cross in 1 launch.
static __global__ void k_hf_lds_fwd(
        int K, int len, int run, int m, float c_run, float c_m, hf_parts parts,
        int first, int ne1, int ne2, size_t nb1, size_t nb2, size_t nb3,
        const char * __restrict__ x, const float * __restrict__ pre, const int32_t * __restrict__ pin,
        float * __restrict__ dst) {
    extern __shared__ float sm[];
    const int shard = blockIdx.x;
    const int64_t c = first + blockIdx.y;
    const float * row = (const float *) (x + (c % ne1) * nb1 + ((c / ne1) % ne2) * nb2 + (c / ((int64_t) ne1 * ne2)) * nb3);
    for (int l = threadIdx.x; l < len; l += blockDim.x) {
        const int64_t i = stored_part(parts, shard, l);
        sm[l] = row[pin ? pin[i] : i] * pre[i];
    }
    __syncthreads();
    const int half = len >> 1;
    for (int h = 1; h < run; h <<= 1) {
        const int sh = __ffs(h) - 1;
        const float sc = (h == (run >> 1)) ? c_run : 1.0f;
        for (int idx = threadIdx.x; idx < half; idx += blockDim.x) {
            const int lo = ((idx >> sh) << (sh + 1)) | (idx & (h - 1));
            const float a = sm[lo], b = sm[lo + h];
            sm[lo]     = (a + b) * sc;
            sm[lo + h] = (a - b) * sc;
        }
        __syncthreads();
    }
    float * drow = dst + (int64_t) blockIdx.y * K;
    if (m == 1) {
        for (int l = threadIdx.x; l < len; l += blockDim.x) {
            drow[stored_part(parts, shard, l)] = sm[l];
        }
    } else {
        for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
            const int r = idx / run, j = idx % run;
            float sum = 0.0f;
            for (int b = 0; b < m; ++b) {
                const float v = sm[b * run + j];
                sum += paley_entry(m, r, b) > 0 ? v : -v;
            }
            drow[stored_part(parts, shard, idx)] = sum * c_m;
        }
    }
}

// Single-CTA LDS inverse hadfull (m==1 && len<=12288, or m>1 && len<=6144): cross + runs + post in 1 launch.
static __global__ void k_hf_lds_inv(
        int N, int len, int run, int m, float c_run, float c_m, hf_parts parts,
        int first_dst, const float * __restrict__ src, const float * __restrict__ post, float * __restrict__ dst) {
    extern __shared__ float sm[];
    const int shard = blockIdx.x;
    const float * srow = src + (int64_t) blockIdx.y * N;
    for (int l = threadIdx.x; l < len; l += blockDim.x) {
        sm[l] = srow[stored_part(parts, shard, l)];
    }
    __syncthreads();
    float * buf = sm;
    if (m > 1) {
        buf = sm + len;
        for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
            const int r = idx / run, j = idx % run;
            float sum = 0.0f;
            for (int b = 0; b < m; ++b) {
                const float v = sm[b * run + j];
                sum += paley_entry(m, b, r) > 0 ? v : -v;
            }
            buf[idx] = sum * c_m;
        }
        __syncthreads();
    }
    const int half = len >> 1;
    for (int h = 1; h < run; h <<= 1) {
        const int sh = __ffs(h) - 1;
        const float sc = (h == (run >> 1)) ? c_run : 1.0f;
        for (int idx = threadIdx.x; idx < half; idx += blockDim.x) {
            const int lo = ((idx >> sh) << (sh + 1)) | (idx & (h - 1));
            const float a = buf[lo], b = buf[lo + h];
            buf[lo]     = (a + b) * sc;
            buf[lo + h] = (a - b) * sc;
        }
        __syncthreads();
    }
    float * drow = dst + (int64_t) (first_dst + blockIdx.y) * N;
    for (int l = threadIdx.x; l < len; l += blockDim.x) {
        const int64_t i = stored_part(parts, shard, l);
        drow[i] = post ? buf[l] * post[i] : buf[l];
    }
}

// 2-kernel fallback for large hadfull slices (len > 12288 fwd or > 6144 inv, up to 24576): in-place safe.
template <int E>
static __global__ void __launch_bounds__(256) k_hf_runs_warp(
        int n, int rows, int shards, int len, int run, float c_run, hf_parts parts, int first_dst,
        const float * __restrict__ src, const float * __restrict__ post, float * __restrict__ dst) {
    constexpr int S = 32 * E;
    const int lane = threadIdx.x & 31;
    const int cps = (len + S - 1) / S;
    const int64_t item = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (item >= (int64_t) rows * shards * cps) {
        return;
    }
    const int row = (int) (item / ((int64_t) shards * cps));
    const int shard = (int) ((item / cps) % shards);
    const int k0 = (int) (item % cps) * S;
    const float * srow = src + (int64_t) row * n;
    float v[E];
    int64_t at[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int l = k0 + lane + 32 * e;
        at[e] = l < len ? stored_part(parts, shard, l) : -1;
        v[e] = at[e] >= 0 ? srow[at[e]] : 0.0f;
    }
    hadamard_warp_stages<E>(v, run, lane);
    float * drow = dst + (int64_t) (first_dst + row) * n;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        if (at[e] >= 0) {
            drow[at[e]] = post ? v[e] * c_run * post[at[e]] : v[e] * c_run;
        }
    }
}

static __global__ void k_hf_runs_cta(
        int n, int run, float c_run, hf_parts parts, int first_dst,
        const float * __restrict__ src, const float * __restrict__ post, float * __restrict__ dst) {
    extern __shared__ float sm[];
    const int k0 = blockIdx.x * run;
    const int shard = blockIdx.y;
    const float * srow = src + (int64_t) blockIdx.z * n;
    for (int l = threadIdx.x; l < run; l += blockDim.x) {
        sm[l] = srow[stored_part(parts, shard, k0 + l)];
    }
    __syncthreads();
    const int half = run >> 1;
    for (int h = 1; h < run; h <<= 1) {
        const int sh = __ffs(h) - 1;
        const float sc = (h == half) ? c_run : 1.0f;
        for (int idx = threadIdx.x; idx < half; idx += blockDim.x) {
            const int lo = ((idx >> sh) << (sh + 1)) | (idx & (h - 1));
            const float a = sm[lo], b = sm[lo + h];
            sm[lo]     = (a + b) * sc;
            sm[lo + h] = (a - b) * sc;
        }
        __syncthreads();
    }
    float * drow = dst + (int64_t) (first_dst + blockIdx.z) * n;
    for (int l = threadIdx.x; l < run; l += blockDim.x) {
        const int64_t i = stored_part(parts, shard, k0 + l);
        drow[i] = post ? sm[l] * post[i] : sm[l];
    }
}

static __global__ void __launch_bounds__(1024) k_hf_cross(
        int n, int m, int run, int g, float c_m, int transpose, hf_parts parts,
        const float * __restrict__ src, float * __restrict__ dst) {
    __shared__ float t[68 * 33];
    const int ncol = run < g ? run : g;
    const int j0 = blockIdx.x * g;
    const int shard = blockIdx.y;
    const float * srow = src + (int64_t) blockIdx.z * n;
    for (int idx = threadIdx.x; idx < m * ncol; idx += blockDim.x) {
        const int b = idx / ncol, jj = idx % ncol;
        t[b * 33 + jj] = srow[stored_part(parts, shard, b * run + j0 + jj)];
    }
    __syncthreads();
    float * drow = dst + (int64_t) blockIdx.z * n;
    for (int idx = threadIdx.x; idx < m * ncol; idx += blockDim.x) {
        const int a = idx / ncol, jj = idx % ncol;
        float acc = 0.0f;
        for (int b = 0; b < m; ++b) {
            const int h = transpose ? paley_entry(m, b, a) : paley_entry(m, a, b);
            const float v = t[b * 33 + jj];
            acc += h > 0 ? v : -v;
        }
        drow[stored_part(parts, shard, a * run + j0 + jj)] = acc * c_m;
    }
}

// ---- input / output ------------------------------------------------------------------------------------

// dst[t*K + k] = x_row(first + t)[pin ? pin[k] : k] * pre[k]
static __global__ void k_input(int K, int first, int ne1, int ne2, size_t nb1, size_t nb2, size_t nb3,
                               const char * __restrict__ x, const float * __restrict__ pre,
                               const int32_t * __restrict__ pin, float * __restrict__ dst) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) {
        return;
    }
    const int64_t c = first + blockIdx.y;
    const char * row = x + (c % ne1) * nb1 + ((c / ne1) % ne2) * nb2 + (c / ((int64_t) ne1 * ne2)) * nb3;
    dst[(int64_t) blockIdx.y * K + k] = ((const float *) row)[pin ? pin[k] : k] * pre[k];
}

// dst[(first + t)*N + n] = y[t*N + j] * (post ? post[j] : 1), j = pout ? pout[n] : n
static __global__ void k_output(int N, int first, const float * __restrict__ y, const float * __restrict__ post,
                                const int32_t * __restrict__ pout, float * __restrict__ dst) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) {
        return;
    }
    const int j = pout ? pout[n] : n;
    const float v = y[(int64_t) blockIdx.y * N + j];
    dst[(int64_t) (first + blockIdx.y) * N + n] = post ? v * post[j] : v;
}

// Batched output gather for input groups (post already applied by output side).
struct gather_seg {
    const float * src;
    const int32_t * idx;
    float * dst;
    int n;
};
struct gather_set {
    gather_seg seg[TRELLIS_GROUP_MAX];
};

static __global__ void k_gather_multi(gather_set set, int first) {
    const gather_seg g = set.seg[blockIdx.z];
    const int64_t t = blockIdx.y;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < g.n; j += gridDim.x * blockDim.x) {
        g.dst[(first + t) * g.n + j] = g.src[t * g.n + g.idx[j]];
    }
}

// Batched input prepare for input groups sharing x and blockwise rht (pin == nullptr).
struct group_in_args {
    const float * pre[TRELLIS_GROUP_MAX];
    float * dst[TRELLIS_GROUP_MAX];
    int n;
};

static __global__ void k_input_group_rht_128(
        int K, float scale, int first, int ne1, int ne2, size_t nb1, size_t nb2, size_t nb3,
        const char * __restrict__ x, group_in_args g) {
    const int lane = threadIdx.x & 31;
    const int b0 = blockIdx.x * 128;
    const int64_t c = first + blockIdx.y;
    const float * row = (const float *) (x + (c % ne1) * nb1 + ((c / ne1) % ne2) * nb2 + (c / ((int64_t) ne1 * ne2)) * nb3);
    float xr[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        xr[e] = row[b0 + lane + 32 * e];
    }
    const int64_t dst_base = (int64_t) blockIdx.y * K + b0;
    for (int m = 0; m < g.n; ++m) {
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int k = b0 + lane + 32 * e;
            v[e] = xr[e] * g.pre[m][k];
        }
        hadamard_128_warp(v, lane);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            g.dst[m][dst_base + lane + 32 * e] = v[e] * scale;
        }
    }
}

template <int B, int THREADS>
static __global__ void __launch_bounds__(THREADS) k_input_group_rht(
        int K, float scale, int first, int ne1, int ne2, size_t nb1, size_t nb2, size_t nb3,
        const char * __restrict__ x, group_in_args g) {
    __shared__ float tile[B / 32][33];
    constexpr int VALUES = B / THREADS;
    constexpr int COL = B < 1024 ? B / 32 : 32;
    const int tid = threadIdx.x, lane = tid & 31;
    const int b0 = blockIdx.x * B;
    const int64_t c = first + blockIdx.y;
    const float * row = (const float *) (x + (c % ne1) * nb1 + ((c / ne1) % ne2) * nb2 + (c / ((int64_t) ne1 * ne2)) * nb3);
    float xr[VALUES];
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        xr[i] = row[b0 + tid + i * THREADS];
    }
    const int64_t dst_base = (int64_t) blockIdx.y * K + b0;
    for (int m = 0; m < g.n; ++m) {
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            const int k = tid + i * THREADS;
            float v = xr[i] * g.pre[m][b0 + k];
#pragma unroll
            for (int h = 1; h < 32; h <<= 1) {
                const float o = ggml_cuda_shfl_xor(v, h);
                v = (lane & h) ? (o - v) : (v + o);
            }
            tile[k >> 5][lane] = v;
        }
        __syncthreads();
        float v[VALUES];
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            const int t = tid + i * THREADS;
            v[i] = tile[(t / (32 * COL)) * COL + (t % COL)][(t / COL) & 31];
#pragma unroll
            for (int h = 1; h < COL; h <<= 1) {
                const float o = ggml_cuda_shfl_xor(v[i], h);
                v[i] = (lane & h) ? (o - v[i]) : (v[i] + o);
            }
        }
#pragma unroll
        for (int step = 1024 / THREADS; step < VALUES; step <<= 1) {
#pragma unroll
            for (int i = 0; i < VALUES; ++i) {
                if (!(i & step)) {
                    const float a = v[i], b = v[i + step];
                    v[i] = a + b; v[i + step] = a - b;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            const int t = tid + i * THREADS;
            tile[(t / (32 * COL)) * COL + (t % COL)][(t / COL) & 31] = v[i];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            const int k = tid + i * THREADS;
            g.dst[m][dst_base + k] = tile[k >> 5][lane] * scale;
        }
        if (m + 1 < g.n) {
            __syncthreads();
        }
    }
}

// Batched output rht for input groups sharing the same rows_block in [128, 4096].
struct group_out_args {
    const float * src[TRELLIS_GROUP_MAX];
    const float * post[TRELLIS_GROUP_MAX];
    float * dst[TRELLIS_GROUP_MAX];
    int N[TRELLIS_GROUP_MAX];
    int cta0[TRELLIS_GROUP_MAX + 1];
    int n;
    int first_dst[TRELLIS_GROUP_MAX];
};

static __global__ void k_output_group_rht_128(group_out_args g, float scale) {
    int m = 0;
#pragma unroll
    for (int i = 1; i < TRELLIS_GROUP_MAX; ++i) {
        if (i < g.n && (int) blockIdx.x >= g.cta0[i]) { m = i; }
    }
    const int lane = threadIdx.x & 31;
    const int b0 = ((int) blockIdx.x - g.cta0[m]) * 128;
    const int64_t src_base = (int64_t) blockIdx.y * g.N[m] + b0;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        v[e] = g.src[m][src_base + lane + 32 * e];
    }
    hadamard_128_warp(v, lane);
    const int64_t dst_base = (int64_t) (g.first_dst[m] + blockIdx.y) * g.N[m] + b0;
    const float * post = g.post[m];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = lane + 32 * e;
        g.dst[m][dst_base + i] = post ? v[e] * scale * post[b0 + i] : v[e] * scale;
    }
}

template <int B, int THREADS>
static __global__ void __launch_bounds__(THREADS) k_output_group_rht(group_out_args g, float scale) {
    __shared__ float tile[B / 32][33];
    int m = 0;
#pragma unroll
    for (int i = 1; i < TRELLIS_GROUP_MAX; ++i) {
        if (i < g.n && (int) blockIdx.x >= g.cta0[i]) { m = i; }
    }
    const int tid = threadIdx.x, lane = tid & 31;
    const int b0 = ((int) blockIdx.x - g.cta0[m]) * B;
    const float * srow = g.src[m] + (int64_t) blockIdx.y * g.N[m];
    rht_row_stages<B, THREADS>(srow, nullptr, nullptr, b0, tile);
    const int64_t dst_base = (int64_t) (g.first_dst[m] + blockIdx.y) * g.N[m] + b0;
    const float * post = g.post[m];
#pragma unroll
    for (int i = 0; i < B / THREADS; ++i) {
        const int k = tid + i * THREADS;
        const float z = tile[k >> 5][lane] * scale;
        g.dst[m][dst_base + k] = post ? z * post[b0 + k] : z;
    }
}

// ---- the product ---------------------------------------------------------------------------------------

// The 8 windows of positions 8a..8a+7 are consecutive, so a few 32-bit anchors cover them all:
// one anchor holds SPAN = 16/BITS + 1 consecutive windows. This is libtrellis's Plan<K> scheme
// (ggml-cuda/libtrellis/src/common.cuh); it needs ANCH word pairs instead of 8.
template <int BITS>
struct anchor_plan {
    static constexpr int NW   = 8 * BITS;
    static constexpr int TOTAL = 256 * BITS;
    static constexpr int SPAN = 16 / BITS + 1;
    static constexpr int ANCH = (8 + SPAN - 1) / SPAN;
    int w0[ANCH], w1[ANCH], sh[ANCH];

    static __device__ __forceinline__ anchor_plan make(int a) {
        anchor_plan p;
#pragma unroll
        for (int x = 0; x < ANCH; ++x) {
            const int e = ((8 * a + 7 - x * SPAN + 1) * BITS) % TOTAL;  // window end, exclusive
            p.w0[x] = ((e - 32 + TOTAL) >> 5) % NW;
            p.w1[x] = (p.w0[x] + 1) % NW;
            p.sh[x] = 32 - (e & 31);
        }
        return p;
    }
};

template <int BITS, int I>
static __device__ __forceinline__ uint32_t anchored_state(const uint32_t (&r)[anchor_plan<BITS>::ANCH]) {
    constexpr int SPAN   = anchor_plan<BITS>::SPAN;
    constexpr int A      = (7 - I) / SPAN;
    constexpr int ANCHOR = 7 - A * SPAN;
    constexpr int C      = (ANCHOR - I) * BITS;
    if constexpr (C == 0) {
        return r[A] & 0xFFFFu;
    } else if constexpr (C == 16) {
        return r[A] >> 16;
    } else {
        return __builtin_amdgcn_ubfe(r[A], C, 16);
    }
}

template <int BITS>
static __device__ __forceinline__ void decode8_plan(const uint32_t * tile, const anchor_plan<BITS> & p, float * v) {
    uint32_t r[anchor_plan<BITS>::ANCH];
#pragma unroll
    for (int x = 0; x < anchor_plan<BITS>::ANCH; ++x) {
        // libtrellis's operand order: the *next* word is the low half, the anchor word the high half
        r[x] = __funnelshift_rc(tile[p.w1[x]], tile[p.w0[x]], p.sh[x]);
    }
    v[0] = grid_value(anchored_state<BITS, 0>(r));
    v[1] = grid_value(anchored_state<BITS, 1>(r));
    v[2] = grid_value(anchored_state<BITS, 2>(r));
    v[3] = grid_value(anchored_state<BITS, 3>(r));
    v[4] = grid_value(anchored_state<BITS, 4>(r));
    v[5] = grid_value(anchored_state<BITS, 5>(r));
    v[6] = grid_value(anchored_state<BITS, 6>(r));
    v[7] = grid_value(anchored_state<BITS, 7>(r));
}

// Decode GEMV v2: 8 waves per WG (128 output rows/WG), each wave owns 1 output tile (16 rows).
// Cooperatively stages CK k-tiles of (x * s16) into LDS once for all 8 waves; each wave loads 4
// consecutive k-tiles with BITS fully-coalesced 128-byte wave loads into wave-local LDS.
// Supports COLS in {1, 2, 4, 8}, Split-K (when splits > 1), and fused had128/post epilogue (when COLS == 1 && splits == 1).
constexpr float H128_SCALE = 0.08838834764831845f; // 128^-0.5

template <int BITS, int COLS>
static __device__ __forceinline__ void mv_v2_body(
        const uint32_t * __restrict__ codes, const half * __restrict__ s16,
        const float * __restrict__ x, float * __restrict__ y,
        const float * __restrict__ post, int fuse_mode,
        int K, int N, int batch, int wg_tile_base, int col0, int split, int splits) {
    constexpr int NW = 8 * BITS;
    constexpr int ANCH = anchor_plan<BITS>::ANCH;
    constexpr int CK = COLS == 1 ? 16 : (COLS <= 4 ? 8 : 4);

    __shared__ float sx[COLS][CK * 16];
    __shared__ uint32_t sm_code[8][32 * BITS];

    const int tid  = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;
    const int r = lane >> 2;
    const int c = (lane & 3) << 1;
    const int kt = K >> 4;

    const int kt_per_split = (((kt + splits - 1) / splits) + 3) & ~3;
    const int kb_start = split * kt_per_split;
    const int kb_end   = min(kb_start + kt_per_split, kt);

    const int tile_row = wg_tile_base + wave;
    const bool live = (tile_row << 4) < N;
    const uint32_t * tiles = codes + (int64_t) tile_row * kt * NW;
    const anchor_plan<BITS> plan = anchor_plan<BITS>::make(lane);

    float sum0[COLS] = {};
    float sum1[COLS] = {};

    for (int kb0 = kb_start; kb0 < kb_end; kb0 += CK) {
        const int jn = min(CK, kb_end - kb0);
        // Cooperatively stage jn k-tiles of x * s16 into LDS
        constexpr int TOTAL_X = COLS * CK * 16;
#pragma unroll
        for (int idx = tid; idx < TOTAL_X; idx += 256) {
            const int col = idx / (CK * 16);
            const int rem = idx % (CK * 16);
            const int jt  = rem >> 4;
            if (jt < jn && col0 + col < batch) {
                const float s = __half2float(s16[kb0 + jt]);
                sx[col][rem] = x[(int64_t) (col0 + col) * K + ((kb0 + jt) << 4) + (rem & 15)] * s;
            } else {
                sx[col][rem] = 0.0f;
            }
        }
        __syncthreads();

        if (live) {
            const int n4 = jn >> 2;
            const uint32_t * tile_ptr = tiles + (int64_t) kb0 * NW;

            auto accum_one_tile = [&](const uint32_t * sm_t, int jt) {
                uint32_t rbuf[ANCH];
#pragma unroll
                for (int a = 0; a < ANCH; ++a) {
                    rbuf[a] = __funnelshift_rc(sm_t[plan.w1[a]], sm_t[plan.w0[a]], plan.sh[a]);
                }
                const float w0 = grid_value(anchored_state<BITS, 0>(rbuf));
                const float w1 = grid_value(anchored_state<BITS, 1>(rbuf));
                const float w2 = grid_value(anchored_state<BITS, 2>(rbuf));
                const float w3 = grid_value(anchored_state<BITS, 3>(rbuf));
                const float w4 = grid_value(anchored_state<BITS, 4>(rbuf));
                const float w5 = grid_value(anchored_state<BITS, 5>(rbuf));
                const float w6 = grid_value(anchored_state<BITS, 6>(rbuf));
                const float w7 = grid_value(anchored_state<BITS, 7>(rbuf));
#pragma unroll
                for (int col = 0; col < COLS; ++col) {
                    const float2 x0 = *reinterpret_cast<const float2 *>(&sx[col][(jt << 4) + c]);
                    const float2 x1 = *reinterpret_cast<const float2 *>(&sx[col][(jt << 4) + c + 8]);
                    sum0[col] = fmaf(w0, x0.x, fmaf(w1, x0.y, fmaf(w2, x1.x, fmaf(w3, x1.y, sum0[col]))));
                    sum1[col] = fmaf(w4, x0.x, fmaf(w5, x0.y, fmaf(w6, x1.x, fmaf(w7, x1.y, sum1[col]))));
                }
            };

            if (n4 > 0) {
                uint32_t cur_buf[BITS];
#pragma unroll
                for (int u = 0; u < BITS; ++u) {
                    cur_buf[u] = tile_ptr[u * 32 + lane];
                }
                for (int g = 0; g < n4; ++g) {
                    uint32_t next_buf[BITS];
                    if (g + 1 < n4) {
                        const uint32_t * next_ptr = tile_ptr + (g + 1) * (4 * NW);
#pragma unroll
                        for (int u = 0; u < BITS; ++u) {
                            next_buf[u] = next_ptr[u * 32 + lane];
                        }
                    }
#pragma unroll
                    for (int u = 0; u < BITS; ++u) {
                        sm_code[wave][u * 32 + lane] = cur_buf[u];
                    }
                    asm volatile("" ::: "memory");
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        accum_one_tile(&sm_code[wave][t * NW], g * 4 + t);
                    }
                    if (g + 1 < n4) {
#pragma unroll
                        for (int u = 0; u < BITS; ++u) {
                            cur_buf[u] = next_buf[u];
                        }
                    }
                }
            }

            // Tail tiles (jn % 4)
            for (int jt = n4 * 4; jt < jn; ++jt) {
                const uint32_t * tp = tile_ptr + jt * NW;
#pragma unroll
                for (int u = lane; u < NW; u += 32) {
                    sm_code[wave][u] = tp[u];
                }
                asm volatile("" ::: "memory");
                accum_one_tile(sm_code[wave], jt);
            }
        }
        __syncthreads();
    }

    // Reduce 4 column groups within each wave
#pragma unroll
    for (int col = 0; col < COLS; ++col) {
        sum0[col] += ggml_cuda_shfl_xor(sum0[col], 1);
        sum0[col] += ggml_cuda_shfl_xor(sum0[col], 2);
        sum1[col] += ggml_cuda_shfl_xor(sum1[col], 1);
        sum1[col] += ggml_cuda_shfl_xor(sum1[col], 2);
    }

    if constexpr (COLS == 1) {
        if (splits == 1 && fuse_mode == 2) {
            // Fused output rht128 + post across all 8 waves (128 rows)
            if (live && (lane & 3) == 0) {
                sx[0][wave * 16 + r]     = sum0[0];
                sx[0][wave * 16 + r + 8] = sum1[0];
            }
            __syncthreads();
            if (wave == 0) {
                const int row_base = wg_tile_base << 4;
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    v[e] = sx[0][lane + 32 * e];
                }
                hadamard_128_warp(v, lane);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = lane + 32 * e;
                    y[row_base + i] = v[e] * H128_SCALE * post[row_base + i];
                }
            }
            return;
        }
        if (live && (lane & 3) == 0) {
            const int row0 = (tile_row << 4) + r;
            const int row1 = row0 + 8;
            float * y_out = splits > 1 ? (y + (int64_t) split * N) : y;
            if (splits == 1 && fuse_mode == 1) {
                if (row0 < N) { y_out[row0] = sum0[0] * post[row0]; }
                if (row1 < N) { y_out[row1] = sum1[0] * post[row1]; }
            } else {
                if (row0 < N) { y_out[row0] = sum0[0]; }
                if (row1 < N) { y_out[row1] = sum1[0]; }
            }
        }
    } else {
        if (live && (lane & 3) == 0) {
            const int row0 = (tile_row << 4) + r;
            const int row1 = row0 + 8;
#pragma unroll
            for (int col = 0; col < COLS; ++col) {
                if (col0 + col < batch) {
                    float * y_row = y + (int64_t) (col0 + col) * N;
                    if (row0 < N) { y_row[row0] = sum0[col]; }
                    if (row1 < N) { y_row[row1] = sum1[col]; }
                }
            }
        }
    }
}

template <int BITS, int COLS>
static __global__ void __launch_bounds__(256) k_product_mv_v2(
        const uint32_t * __restrict__ codes, const half * __restrict__ s16,
        const float * __restrict__ x, float * __restrict__ y,
        const float * __restrict__ post, int fuse_mode,
        int K, int N, int batch, int splits) {
    if constexpr (COLS == 1) {
        mv_v2_body<BITS, 1>(codes, s16, x, y, post, fuse_mode, K, N, 1, 8 * blockIdx.x, 0, blockIdx.y, splits);
    } else {
        mv_v2_body<BITS, COLS>(codes, s16, x, y, nullptr, 0, K, N, batch, 8 * blockIdx.x, COLS * blockIdx.y, 0, 1);
    }
}

// Deterministic Split-K reduction for batch == 1; optionally fuses post (fuse_mode==1) or had128+post (fuse_mode==2).
static __global__ void __launch_bounds__(128) k_splitk_reduce(
        const float * __restrict__ partials, const float * __restrict__ post, float * __restrict__ dst,
        int N, int splits, int fuse_mode) {
    const int row = blockIdx.x * 128 + threadIdx.x;
    float acc = 0.0f;
    if (row < N) {
        for (int s = 0; s < splits; ++s) {
            acc += partials[(int64_t) s * N + row];
        }
    }
    if (fuse_mode == 2) {
        __shared__ float sm[128];
        sm[threadIdx.x] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            const int lane = threadIdx.x;
            const int row_base = blockIdx.x * 128;
            float v[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[e] = sm[lane + 32 * e];
            }
            hadamard_128_warp(v, lane);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = lane + 32 * e;
                dst[row_base + i] = v[e] * H128_SCALE * post[row_base + i];
            }
        }
        return;
    }
    if (row < N) {
        dst[row] = fuse_mode == 1 ? (acc * post[row]) : acc;
    }
}

struct mv1_group_member {
    const uint32_t * codes;
    const half * s16;
    const float * x;
    float * y;
    const float * post;
    int fuse_mode;
    int N;
    int bits;
};

struct mv1_group_args {
    mv1_group_member m[TRELLIS_GROUP_MAX];
    int cta0[TRELLIS_GROUP_MAX + 1]; // cumulative WG count: (N + 127) / 128
    int n;
    int K;
    int splits;
};

template <int BITS>
static __global__ void __launch_bounds__(256) k_product_mv1_group(mv1_group_args g) {
    int mi = 0;
#pragma unroll
    for (int i = 1; i < TRELLIS_GROUP_MAX; ++i) {
        if (i < g.n && (int) blockIdx.x >= g.cta0[i]) { mi = i; }
    }
    const mv1_group_member & m = g.m[mi];
    const int wg_idx = (int) blockIdx.x - g.cta0[mi];
    mv_v2_body<BITS, 1>(m.codes, m.s16, m.x, m.y, m.post, m.fuse_mode, g.K, m.N, 1, 8 * wg_idx, 0, blockIdx.y, g.splits);
}

static __global__ void __launch_bounds__(256) k_product_mv1_mixed_group(mv1_group_args g) {
    int mi = 0;
#pragma unroll
    for (int i = 1; i < TRELLIS_GROUP_MAX; ++i) {
        if (i < g.n && (int) blockIdx.x >= g.cta0[i]) { mi = i; }
    }
    const mv1_group_member & m = g.m[mi];
    const int wg_idx = (int) blockIdx.x - g.cta0[mi];
    switch (m.bits) {
        case 1: mv_v2_body<1, 1>(m.codes, m.s16, m.x, m.y, m.post, m.fuse_mode, g.K, m.N, 1, 8 * wg_idx, 0, blockIdx.y, g.splits); break;
        case 2: mv_v2_body<2, 1>(m.codes, m.s16, m.x, m.y, m.post, m.fuse_mode, g.K, m.N, 1, 8 * wg_idx, 0, blockIdx.y, g.splits); break;
        case 3: mv_v2_body<3, 1>(m.codes, m.s16, m.x, m.y, m.post, m.fuse_mode, g.K, m.N, 1, 8 * wg_idx, 0, blockIdx.y, g.splits); break;
        case 4: mv_v2_body<4, 1>(m.codes, m.s16, m.x, m.y, m.post, m.fuse_mode, g.K, m.N, 1, 8 * wg_idx, 0, blockIdx.y, g.splits); break;
        case 5: mv_v2_body<5, 1>(m.codes, m.s16, m.x, m.y, m.post, m.fuse_mode, g.K, m.N, 1, 8 * wg_idx, 0, blockIdx.y, g.splits); break;
        default: mv_v2_body<6, 1>(m.codes, m.s16, m.x, m.y, m.post, m.fuse_mode, g.K, m.N, 1, 8 * wg_idx, 0, blockIdx.y, g.splits); break;
    }
}

// Batched Split-K reduction for input groups.
struct splitk_group_args {
    const float * partials[TRELLIS_GROUP_MAX];
    const float * post[TRELLIS_GROUP_MAX];
    float * dst[TRELLIS_GROUP_MAX];
    int N[TRELLIS_GROUP_MAX];
    int fuse_mode[TRELLIS_GROUP_MAX];
    int cta0[TRELLIS_GROUP_MAX + 1];
    int n;
    int splits;
};

static __global__ void __launch_bounds__(128) k_splitk_reduce_group(splitk_group_args g) {
    int mi = 0;
#pragma unroll
    for (int i = 1; i < TRELLIS_GROUP_MAX; ++i) {
        if (i < g.n && (int) blockIdx.x >= g.cta0[i]) { mi = i; }
    }
    const int blk = (int) blockIdx.x - g.cta0[mi];
    const int N = g.N[mi];
    const int row = blk * 128 + threadIdx.x;
    float acc = 0.0f;
    if (row < N) {
        const float * p = g.partials[mi];
        for (int s = 0; s < g.splits; ++s) {
            acc += p[(int64_t) s * N + row];
        }
    }
    if (g.fuse_mode[mi] == 2) {
        __shared__ float sm[128];
        sm[threadIdx.x] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            const int lane = threadIdx.x;
            const int row_base = blk * 128;
            float v[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[e] = sm[lane + 32 * e];
            }
            hadamard_128_warp(v, lane);
            const float * post = g.post[mi];
            float * dst = g.dst[mi];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = lane + 32 * e;
                dst[row_base + i] = v[e] * H128_SCALE * post[row_base + i];
            }
        }
        return;
    }
    if (row < N) {
        g.dst[mi][row] = g.fuse_mode[mi] == 1 ? (acc * g.post[mi][row]) : acc;
    }
}

// ---- WMMA prefill (Plan 2, route 2A: decode into LDS, then matrix cores) --------------------------------
//
// A workgroup is 4 waves and covers 64 output rows x 16 tokens per k-block of 16. Each wave owns one
// 16x16 weight tile and decodes it with the same decode8 as the scalar path, but into LDS as exact
// integer grid values (all of [-126, 126] are exact in f16). The block scale s16[k/16] is folded into
// the activations instead, once per block, so the B operand stays a plain f16 matrix of tokens.
// Result mapping (see PLAN-2 Q0): mma(D, A, B) puts the B row index in get_i and the A row index in
// get_j, i.e. D[get_i][get_j] = sum_k A[get_j][k] * B[get_i][k].
//
// Launch gated at runtime on RDNA4 (validated there only); other targets keep the scalar GEMV.

// The gfx12 fragment layouts in mma.cuh are expressed in terms of threadIdx.x, i.e. they assume one
// wave per workgroup. These helpers take the lane index explicitly so four waves can share a workgroup.
// Layout: A/B element (i, j) sits at i = lane % 16, j = 4*(lane/16) + l (half2 units); C element (i, j)
// at i = lane % 16, j = 8*(lane/16) + l (float units).
template <int STRIDE_H2, ggml_cuda_mma::data_layout dl>
static __device__ __forceinline__ void mm_load_ab_s(ggml_cuda_mma::tile<16, 8, half2, dl> & t,
                                                    const half2 * __restrict__ p, int lane) {
    const int i = lane % 16, j0 = 4 * (lane / 16);
    *reinterpret_cast<int4 *>(t.x) = *reinterpret_cast<const int4 *>(p + i * STRIDE_H2 + j0);
}

template <ggml_cuda_mma::data_layout dl>
static __device__ __forceinline__ void mm_load_ab(ggml_cuda_mma::tile<16, 8, half2, dl> & t,
                                                 const half2 * __restrict__ p, int lane) {
    mm_load_ab_s<8, dl>(t, p, lane);
}

// XOR-swizzled load for 16x8 half2 LDS tiles: j0 ^ (((i >> 2) & 1) << 2) removes 4-way bank conflicts.
template <ggml_cuda_mma::data_layout dl>
static __device__ __forceinline__ void mm_load_ab_swz(ggml_cuda_mma::tile<16, 8, half2, dl> & t,
                                                      const half2 * __restrict__ p, int lane) {
    const int i = lane & 15;
#if defined(AMD_WMMA_256B_AVAILABLE)
    const int swz = ((i >> 2) & 1) << 2;
    *reinterpret_cast<int4 *>(t.x + 0) = *reinterpret_cast<const int4 *>(p + i * 8 + (0 ^ swz));
    *reinterpret_cast<int4 *>(t.x + 4) = *reinterpret_cast<const int4 *>(p + i * 8 + (4 ^ swz));
#else
    const int j0 = (4 * (lane >> 4)) ^ (((i >> 2) & 1) << 2);
    *reinterpret_cast<int4 *>(t.x) = *reinterpret_cast<const int4 *>(p + i * 8 + j0);
#endif
}

static __device__ __forceinline__ int wmma_c_row(int lane, int l) {
#if defined(AMD_WMMA_256B_AVAILABLE)
    return 2 * l + (lane >> 4);
#else
    return 8 * (lane >> 4) + l;
#endif
}

// Prologue for FP16 hi/lo WMMA: precomputes (x * s16) split into (hi, lo) half2 pairs once per token.
// Layout: x_hilo[(col * kt + kt_idx) * 16 + (0..7: hi, 8..15: lo)] (64 bytes per token per k-tile).
static __global__ void __launch_bounds__(256) k_prep_act_hilo(
        const float * __restrict__ x, const half * __restrict__ s16,
        half2 * __restrict__ x_hilo, int K, int batch) {
    const int kt = K >> 4;
    const int64_t total_vec = (int64_t) batch * kt * 4;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vec) {
        return;
    }
    const int q   = (int) (idx & 3);          // 0..3 (4 float4s per k-tile)
    const int64_t tk = idx >> 2;
    const int kti = (int) (tk % kt);
    const int col = (int) (tk / kt);
    const float s = __half2float(s16[kti]);
    const float4 fv = *reinterpret_cast<const float4 *>(x + (int64_t) col * K + (kti << 4) + (q << 2));
    const float f0 = fv.x * s, f1 = fv.y * s, f2 = fv.z * s, f3 = fv.w * s;
    const half2 hi01 = __floats2half2_rn(f0, f1);
    const half2 hi23 = __floats2half2_rn(f2, f3);
    const float2 hf01 = __half22float2(hi01);
    const float2 hf23 = __half22float2(hi23);
    const half2 lo01 = __floats2half2_rn(f0 - hf01.x, f1 - hf01.y);
    const half2 lo23 = __floats2half2_rn(f2 - hf23.x, f3 - hf23.y);
    half2 * dst = x_hilo + tk * 16;
    dst[q * 2 + 0]     = hi01;
    dst[q * 2 + 1]     = hi23;
    dst[8 + q * 2 + 0] = lo01;
    dst[8 + q * 2 + 1] = lo23;
}

// Coalesced LDS-transposed epilogue for 128x128 WMMA kernels.
// Processes 32 tokens at a time through sm_out[32][128 + 1] (16.125 KiB), writing coalesced stores along N
// and optionally fusing post (fuse_mode==1) or had128+post (fuse_mode==2).
static __device__ __forceinline__ void mm_128x128_epilogue(
        float (&acc)[4][2][8], float (*sm_out)[129], float * __restrict__ y,
        const float * __restrict__ post, int fuse_mode,
        int N, int batch, int tile_base, int col0, int wave, int lane, int wm, int wn) {
    const int row_base = 16 * tile_base;
    const int m0 = 4 * wm;
    // 4 passes of 32 tokens: pass p covers N-waves wn_lo = 2*(p/2)..+1, tn = p%2 (each pass = 2 N-waves x 16 tokens)
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int target_tn = pass & 1;
        const int wn_base   = (pass >> 1) * 2; // 0 or 2
        if (wn == wn_base || wn == wn_base + 1) {
            const int tok32 = (wn - wn_base) * 16 + (lane & 15);
#pragma unroll
            for (int tm = 0; tm < 4; ++tm) {
                const int r_off = (m0 + tm) * 16;
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    sm_out[tok32][r_off + wmma_c_row(lane, l)] = acc[tm][target_tn][l];
                }
            }
        }
        __syncthreads();

        // 8 waves handle 32 tokens (4 tokens per wave), 32 lanes x 4 elements = 128 rows
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int tok32 = wave * 4 + t;
            const int wn_t  = wn_base + (tok32 >> 4);
            const int ccol  = col0 + (2 * wn_t + target_tn) * 16 + (tok32 & 15);
            if (ccol < batch) {
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    v[e] = sm_out[tok32][lane + 32 * e];
                }
                if (fuse_mode == 2) {
                    hadamard_128_warp(v, lane);
                    float * yrow = y + (int64_t) ccol * N + row_base;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int i = lane + 32 * e;
                        yrow[i] = v[e] * H128_SCALE * post[row_base + i];
                    }
                } else {
                    float * yrow = y + (int64_t) ccol * N + row_base;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int i = lane + 32 * e;
                        if (row_base + i < N) {
                            yrow[i] = fuse_mode == 1 ? (v[e] * post[row_base + i]) : v[e];
                        }
                    }
                }
            }
        }
        if (pass < 3) {
            __syncthreads();
        }
    }
}

// 128 rows x 128 tokens workgroup (2x4 wave grid, 8 waves = 256 threads) using precomputed (hi, lo) half2 activations.
// XOR-swizzled 16x8 half2 tiles eliminate LDS bank conflicts; total LDS = 24 KiB (2 WGs/CU).
// Prefetches next k-group activations in VGPRs to overlap global loads with WMMA compute.
template <int BITS, int KG>
static __global__ void __launch_bounds__(256, 2) k_product_mm_128x128(
        const uint32_t * __restrict__ codes, const half2 * __restrict__ x_hilo,
        float * __restrict__ y, const float * __restrict__ post, int fuse_mode,
        int K, int N, int batch) {
    constexpr int NW = 8 * BITS;
    constexpr ggml_cuda_mma::data_layout layout = ggml_cuda_mma::get_input_data_layout();

    union shared_mem_t {
        struct {
            __align__(16) half2 sw[8][KG][16 * 8];
            __align__(16) half2 sx[KG][8][2][16 * 8];
        } mm;
        float ep[32][129];
    };
    __shared__ shared_mem_t sm;

    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int wm   = wave >> 2, wn = wave & 3;
    const int tile_base = 8 * blockIdx.x;
    const int col0 = 128 * blockIdx.y;
    const int kt   = K >> 4;

    const int decode_tile = tile_base + wave;
    const bool decode_live = 16 * decode_tile < N;
    const uint32_t * decode_tiles = codes + (int64_t) decode_tile * kt * NW;

    const int m0 = 4 * wm;
    const int c2 = lane & 3;
    const int r  = lane >> 2;
    const int swz_r = ((r >> 2) & 1) << 2;
    const anchor_plan<BITS> plan = anchor_plan<BITS>::make(lane);

    // 128 tokens x 2 halves (4 half2 each = int4) = 256 threads -> 1 int4 load for hi + 1 int4 load for lo per k-tile
    const int tok = threadIdx.x >> 1;
    const int half_k = (threadIdx.x & 1) * 4;
    const int sub_s = tok >> 4;
    const int tr_s  = tok & 15;
    const int j0_swz = half_k ^ (((tr_s >> 2) & 1) << 2);
    const int col_s = col0 + tok;
    const bool tok_live = col_s < batch;

    int4 cur_hi[KG], cur_lo[KG];
    {
        const int jn0 = min(KG, kt);
#pragma unroll
        for (int j = 0; j < KG; ++j) {
            cur_hi[j] = make_int4(0, 0, 0, 0);
            cur_lo[j] = make_int4(0, 0, 0, 0);
            if (j < jn0 && tok_live) {
                const half2 * src = x_hilo + ((int64_t) col_s * kt + j) * 16;
                cur_hi[j] = *reinterpret_cast<const int4 *>(src + half_k);
                cur_lo[j] = *reinterpret_cast<const int4 *>(src + 8 + half_k);
            }
        }
    }

    float acc[4][2][8] = {};

    for (int kb0 = 0; kb0 < kt; kb0 += KG) {
        const int jn = min(KG, kt - kb0);
        if (decode_live) {
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                if (j >= jn) { break; }
                half2 * w2 = sm.mm.sw[wave][j];
                float v[8];
                decode8_plan<BITS>(decode_tiles + (int64_t) (kb0 + j) * NW, plan, v);
                w2[r * 8 + (c2 ^ swz_r)]             = __floats2half2_rn(v[0], v[1]);
                w2[r * 8 + ((c2 + 4) ^ swz_r)]       = __floats2half2_rn(v[2], v[3]);
                w2[(r + 8) * 8 + (c2 ^ swz_r)]       = __floats2half2_rn(v[4], v[5]);
                w2[(r + 8) * 8 + ((c2 + 4) ^ swz_r)] = __floats2half2_rn(v[6], v[7]);
            }
        }
#pragma unroll
        for (int j = 0; j < KG; ++j) {
            if (j >= jn) { break; }
            *reinterpret_cast<int4 *>(sm.mm.sx[j][sub_s][0] + tr_s * 8 + j0_swz) = cur_hi[j];
            *reinterpret_cast<int4 *>(sm.mm.sx[j][sub_s][1] + tr_s * 8 + j0_swz) = cur_lo[j];
        }
        __syncthreads();

        // Prefetch next k-group activations into VGPRs before WMMA
        int4 next_hi[KG], next_lo[KG];
        const int next_kb0 = kb0 + KG;
        if (next_kb0 < kt) {
            const int next_jn = min(KG, kt - next_kb0);
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                next_hi[j] = make_int4(0, 0, 0, 0);
                next_lo[j] = make_int4(0, 0, 0, 0);
                if (j < next_jn && tok_live) {
                    const half2 * src = x_hilo + ((int64_t) col_s * kt + (next_kb0 + j)) * 16;
                    next_hi[j] = *reinterpret_cast<const int4 *>(src + half_k);
                    next_lo[j] = *reinterpret_cast<const int4 *>(src + 8 + half_k);
                }
            }
        }

        ggml_cuda_mma::tile<16, 8, half2, layout> a[4][KG];
#pragma unroll
        for (int j = 0; j < KG; ++j) {
            if (j >= jn) { break; }
#pragma unroll
            for (int tm = 0; tm < 4; ++tm) {
                mm_load_ab_swz<layout>(a[tm][j], sm.mm.sw[m0 + tm][j], lane);
            }
        }
#pragma unroll
        for (int tn = 0; tn < 2; ++tn) {
            const int sub = 2 * wn + tn;
            ggml_cuda_mma::tile<16, 16, float, ggml_cuda_mma::DATA_LAYOUT_I_MAJOR> d[4];
#pragma unroll
            for (int tm = 0; tm < 4; ++tm) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    d[tm].x[l] = 0.0f;
                }
            }
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                if (j >= jn) { break; }
                ggml_cuda_mma::tile<16, 8, half2, layout> b_hi, b_lo;
                mm_load_ab_swz<layout>(b_hi, sm.mm.sx[j][sub][0], lane);
                mm_load_ab_swz<layout>(b_lo, sm.mm.sx[j][sub][1], lane);
#pragma unroll
                for (int tm = 0; tm < 4; ++tm) {
                    ggml_cuda_mma::mma(d[tm], a[tm][j], b_hi);
                }
#pragma unroll
                for (int tm = 0; tm < 4; ++tm) {
                    ggml_cuda_mma::mma(d[tm], a[tm][j], b_lo);
                }
            }
#pragma unroll
            for (int tm = 0; tm < 4; ++tm) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    acc[tm][tn][l] += d[tm].x[l];
                }
            }
        }
        if (next_kb0 < kt) {
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                cur_hi[j] = next_hi[j];
                cur_lo[j] = next_lo[j];
            }
        }
        __syncthreads();
    }

    mm_128x128_epilogue(acc, sm.ep, y, post, fuse_mode, N, batch, tile_base, col0, wave, lane, wm, wn);
}

// 64 rows x 128 tokens workgroup (4x2 wave grid, 8 waves = 256 threads) for small N (< 128).
template <int BITS, int KG>
static __global__ void __launch_bounds__(256, 2) k_product_mm_64x128(
        const uint32_t * __restrict__ codes, const half * __restrict__ s16,
        const float * __restrict__ x, float * __restrict__ y, int K, int N, int batch) {
    constexpr int NW = 8 * BITS;
    constexpr ggml_cuda_mma::data_layout layout = ggml_cuda_mma::get_input_data_layout();

    __shared__ __align__(16) half2 sw[4][KG][16 * 8];
    __shared__ __align__(16) half2 sx[KG][8][2][16 * 8];

    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm   = wave / 2, wn = wave % 2;
    const int tile = 4 * blockIdx.x + wm;
    const int row0 = 16 * tile;
    const int col0 = 128 * blockIdx.y;
    const int kt   = K / 16;
    const bool live = row0 < N;

    const int c2 = lane % 4;
    const int r  = lane / 4;
    const int swz_r = ((r >> 2) & 1) << 2;
    const anchor_plan<BITS> plan = anchor_plan<BITS>::make(lane);

    const int t0  = threadIdx.x / 4;        // 0..63
    const int kp  = (threadIdx.x % 4) * 4;  // 0, 4, 8, 12 (float index)
    const int kp2 = (threadIdx.x % 4) * 2;  // 0, 2, 4, 6 (half2 index)

    float acc[4][8] = {};

    const uint32_t * tiles = codes + (int64_t) tile * kt * NW;
    for (int kb0 = 0; kb0 < kt; kb0 += KG) {
        const int jn = min(KG, kt - kb0);
        if (live) {
#pragma unroll
            for (int j = wn; j < KG; j += 2) {
                if (j >= jn) { break; }
                half2 * w2 = sw[wm][j];
                float v[8];
                decode8_plan<BITS>(tiles + (int64_t) (kb0 + j) * NW, plan, v);
                w2[r * 8 + (c2 ^ swz_r)]             = __floats2half2_rn(v[0], v[1]);
                w2[r * 8 + ((c2 + 4) ^ swz_r)]       = __floats2half2_rn(v[2], v[3]);
                w2[(r + 8) * 8 + (c2 ^ swz_r)]       = __floats2half2_rn(v[4], v[5]);
                w2[(r + 8) * 8 + ((c2 + 4) ^ swz_r)] = __floats2half2_rn(v[6], v[7]);
            }
        }
#pragma unroll
        for (int j = 0; j < KG; ++j) {
            if (j >= jn) { break; }
            const float s = __half2float(s16[kb0 + j]);
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int tok = p * 64 + t0;
                const int sub = tok >> 4;
                const int tr  = tok & 15;
                const int kp2_swz = kp2 ^ (((tr >> 2) & 1) << 2);
                const int col = col0 + tok;
                const float4 fv = col < batch
                    ? *reinterpret_cast<const float4 *>(x + (int64_t) col * K + 16 * (kb0 + j) + kp)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float f0 = fv.x * s, f1 = fv.y * s, f2 = fv.z * s, f3 = fv.w * s;
                const half2 hi01 = __floats2half2_rn(f0, f1);
                const half2 hi23 = __floats2half2_rn(f2, f3);
                const float2 hf01 = __half22float2(hi01);
                const float2 hf23 = __half22float2(hi23);
                half2 * hi = sx[j][sub][0];
                half2 * lo = sx[j][sub][1];
                hi[tr * 8 + kp2_swz + 0] = hi01;
                hi[tr * 8 + kp2_swz + 1] = hi23;
                lo[tr * 8 + kp2_swz + 0] = __floats2half2_rn(f0 - hf01.x, f1 - hf01.y);
                lo[tr * 8 + kp2_swz + 1] = __floats2half2_rn(f2 - hf23.x, f3 - hf23.y);
            }
        }
        __syncthreads();

        if (live) {
            ggml_cuda_mma::tile<16, 8, half2, layout> a[KG];
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                if (j >= jn) { break; }
                mm_load_ab_swz<layout>(a[j], sw[wm][j], lane);
            }
#pragma unroll
            for (int tn = 0; tn < 4; ++tn) {
                const int sub = 4 * wn + tn;
                ggml_cuda_mma::tile<16, 16, float, ggml_cuda_mma::DATA_LAYOUT_I_MAJOR> d;
#pragma unroll
                for (int l = 0; l < d.ne; ++l) {
                    d.x[l] = 0.0f;
                }
#pragma unroll
                for (int j = 0; j < KG; ++j) {
                    if (j >= jn) { break; }
                    ggml_cuda_mma::tile<16, 8, half2, layout> b_hi, b_lo;
                    mm_load_ab_swz<layout>(b_hi, sx[j][sub][0], lane);
                    mm_load_ab_swz<layout>(b_lo, sx[j][sub][1], lane);
                    ggml_cuda_mma::mma(d, a[j], b_hi);
                    ggml_cuda_mma::mma(d, a[j], b_lo);
                }
#pragma unroll
                for (int l = 0; l < d.ne; ++l) {
                    acc[tn][l] += d.x[l];
                }
            }
        }
        __syncthreads();
    }

    if (!live) {
        return;
    }

#pragma unroll
    for (int tn = 0; tn < 4; ++tn) {
        const int ccol = col0 + (4 * wn + tn) * 16 + (lane & 15);
        if (ccol < batch) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                y[(int64_t) ccol * N + row0 + wmma_c_row(lane, l)] = acc[tn][l];
            }
        }
    }
}

// 128 rows x 32 tokens workgroup (4x2 wave grid, 8 waves = 256 threads) for small batches (9..31).
// Total LDS = 12 KiB (sw: 8 KiB, sx: 4 KiB) with XOR swizzle for 0 bank conflicts.
template <int BITS, int KG>
static __global__ void __launch_bounds__(256, 2) k_product_mm_128x32(
        const uint32_t * __restrict__ codes, const half * __restrict__ s16,
        const float * __restrict__ x, float * __restrict__ y, int K, int N, int batch) {
    constexpr int NW = 8 * BITS;
    constexpr ggml_cuda_mma::data_layout layout = ggml_cuda_mma::get_input_data_layout();

    __shared__ __align__(16) half2 sw[8][KG][16 * 8];
    __shared__ __align__(16) half2 sx[KG][2][2][16 * 8];

    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int wm   = wave >> 1, wn = wave & 1; // 4 M-waves (2 M-tiles each) x 2 N-waves (1 N-subtile each)
    const int tile_base = 8 * blockIdx.x;
    const int col0 = 32 * blockIdx.y;
    const int kt   = K / 16;

    const int decode_tile = tile_base + wave;
    const bool decode_live = 16 * decode_tile < N;
    const uint32_t * decode_tiles = codes + (int64_t) decode_tile * kt * NW;

    const int m0 = 2 * wm;
    const int c2 = lane & 3;
    const int r  = lane >> 2;
    const int swz_r = ((r >> 2) & 1) << 2;
    const anchor_plan<BITS> plan = anchor_plan<BITS>::make(lane);

    // 32 tokens x 4 float4s = 128 items -> first 128 threads stage activations
    const int tok = threadIdx.x >> 2;          // 0..31 (for tid < 128)
    const int kp  = (threadIdx.x & 3) * 4;
    const int kp2 = (threadIdx.x & 3) * 2;
    const int sub_s = tok >> 4;
    const int tr_s  = tok & 15;
    const int kp2_swz = kp2 ^ (((tr_s >> 2) & 1) << 2);

    float acc[2][8] = {};

    for (int kb0 = 0; kb0 < kt; kb0 += KG) {
        const int jn = min(KG, kt - kb0);
        if (decode_live) {
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                if (j >= jn) { break; }
                half2 * w2 = sw[wave][j];
                float v[8];
                decode8_plan<BITS>(decode_tiles + (int64_t) (kb0 + j) * NW, plan, v);
                w2[r * 8 + (c2 ^ swz_r)]               = __floats2half2_rn(v[0], v[1]);
                w2[r * 8 + ((c2 + 4) ^ swz_r)]         = __floats2half2_rn(v[2], v[3]);
                w2[(r + 8) * 8 + (c2 ^ swz_r)]         = __floats2half2_rn(v[4], v[5]);
                w2[(r + 8) * 8 + ((c2 + 4) ^ swz_r)]   = __floats2half2_rn(v[6], v[7]);
            }
        }
        if (threadIdx.x < 128) {
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                if (j >= jn) { break; }
                const float s = __half2float(s16[kb0 + j]);
                const int col = col0 + tok;
                const float4 fv = col < batch
                    ? *reinterpret_cast<const float4 *>(x + (int64_t) col * K + 16 * (kb0 + j) + kp)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float f0 = fv.x * s, f1 = fv.y * s, f2 = fv.z * s, f3 = fv.w * s;
                const half2 hi01 = __floats2half2_rn(f0, f1);
                const half2 hi23 = __floats2half2_rn(f2, f3);
                const float2 hf01 = __half22float2(hi01);
                const float2 hf23 = __half22float2(hi23);
                half2 * hi = sx[j][sub_s][0];
                half2 * lo = sx[j][sub_s][1];
                hi[tr_s * 8 + kp2_swz + 0] = hi01;
                hi[tr_s * 8 + kp2_swz + 1] = hi23;
                lo[tr_s * 8 + kp2_swz + 0] = __floats2half2_rn(f0 - hf01.x, f1 - hf01.y);
                lo[tr_s * 8 + kp2_swz + 1] = __floats2half2_rn(f2 - hf23.x, f3 - hf23.y);
            }
        }
        __syncthreads();

        ggml_cuda_mma::tile<16, 16, float, ggml_cuda_mma::DATA_LAYOUT_I_MAJOR> d[2];
#pragma unroll
        for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                d[tm].x[l] = 0.0f;
            }
        }
#pragma unroll
        for (int j = 0; j < KG; ++j) {
            if (j >= jn) { break; }
            ggml_cuda_mma::tile<16, 8, half2, layout> b_hi, b_lo;
            mm_load_ab_swz<layout>(b_hi, sx[j][wn][0], lane);
            mm_load_ab_swz<layout>(b_lo, sx[j][wn][1], lane);
#pragma unroll
            for (int tm = 0; tm < 2; ++tm) {
                ggml_cuda_mma::tile<16, 8, half2, layout> a;
                mm_load_ab_swz<layout>(a, sw[m0 + tm][j], lane);
                ggml_cuda_mma::mma(d[tm], a, b_hi);
                ggml_cuda_mma::mma(d[tm], a, b_lo);
            }
        }
#pragma unroll
        for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                acc[tm][l] += d[tm].x[l];
            }
        }
        __syncthreads();
    }

    const int ccol = col0 + wn * 16 + (lane & 15);
    if (ccol < batch) {
#pragma unroll
        for (int tm = 0; tm < 2; ++tm) {
            const int row_base = 16 * (tile_base + m0 + tm);
            if (row_base < N) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    y[(int64_t) ccol * N + row_base + wmma_c_row(lane, l)] = acc[tm][l];
                }
            }
        }
    }
}

// ---- host orchestration --------------------------------------------------------------------------------

static void launch_input_rht(int K, int blk, int first, int batch, const ggml_tensor * x,
                             const float * pre, const int32_t * pin, float * dst, cudaStream_t stream) {
    const float sc = (float) std::pow((double) blk, -0.5);
    const dim3 grid(K / blk, batch, 1);
    const int ne1 = (int) x->ne[1], ne2 = (int) x->ne[2];
    const size_t nb1 = x->nb[1], nb2 = x->nb[2], nb3 = x->nb[3];
    const char * xd = (const char *) x->data;
    switch (blk) {
        case 128:  k_input_rht_128<<<grid, 32, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, pre, pin, dst); break;
        case 256:  k_input_rht<256, 256><<<grid, 256, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, pre, pin, dst); break;
        case 512:  k_input_rht<512, 256><<<grid, 256, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, pre, pin, dst); break;
        case 1024: k_input_rht<1024, 256><<<grid, 256, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, pre, pin, dst); break;
        case 2048: k_input_rht<2048, 1024><<<grid, 1024, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, pre, pin, dst); break;
        case 4096: k_input_rht<4096, 1024><<<grid, 1024, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, pre, pin, dst); break;
        default:   GGML_ABORT("trellis hip: bad rht block %d", blk);
    }
}

static void launch_output_rht(int N, int blk, int first, int batch, const float * src,
                              const float * post, float * dst, cudaStream_t stream) {
    const float sc = (float) std::pow((double) blk, -0.5);
    const dim3 grid(N / blk, batch, 1);
    switch (blk) {
        case 128:  k_output_rht_128<<<grid, 32, 0, stream>>>(N, sc, first, src, post, dst); break;
        case 256:  k_output_rht<256, 256><<<grid, 256, 0, stream>>>(N, sc, first, src, post, dst); break;
        case 512:  k_output_rht<512, 256><<<grid, 256, 0, stream>>>(N, sc, first, src, post, dst); break;
        case 1024: k_output_rht<1024, 256><<<grid, 256, 0, stream>>>(N, sc, first, src, post, dst); break;
        case 2048: k_output_rht<2048, 1024><<<grid, 1024, 0, stream>>>(N, sc, first, src, post, dst); break;
        case 4096: k_output_rht<4096, 1024><<<grid, 1024, 0, stream>>>(N, sc, first, src, post, dst); break;
        default:   GGML_ABORT("trellis hip: bad rht block %d", blk);
    }
}

static void launch_swiglu_rht(int N, int blk, int batch, const float * g, const float * u, float * xd,
                              const float * post_g, const float * post_u, const float * pre_d, cudaStream_t stream) {
    const float sc = (float) std::pow((double) blk, -0.5);
    const dim3 grid(N / blk, batch, 1);
    switch (blk) {
        case 128:  k_swiglu_rht_128<<<grid, 32, 0, stream>>>(g, u, xd, N, sc, post_g, post_u, pre_d); break;
        case 256:  k_swiglu_rht<256, 256><<<grid, 256, 0, stream>>>(g, u, xd, N, sc, post_g, post_u, pre_d); break;
        case 512:  k_swiglu_rht<512, 256><<<grid, 256, 0, stream>>>(g, u, xd, N, sc, post_g, post_u, pre_d); break;
        case 1024: k_swiglu_rht<1024, 256><<<grid, 256, 0, stream>>>(g, u, xd, N, sc, post_g, post_u, pre_d); break;
        case 2048: k_swiglu_rht<2048, 1024><<<grid, 1024, 0, stream>>>(g, u, xd, N, sc, post_g, post_u, pre_d); break;
        case 4096: k_swiglu_rht<4096, 1024><<<grid, 1024, 0, stream>>>(g, u, xd, N, sc, post_g, post_u, pre_d); break;
        default:   GGML_ABORT("trellis hip: bad swiglu rht block %d", blk);
    }
}

static void launch_input_group_rht(int K, int blk, int first, int batch, const ggml_tensor * x,
                                   const group_in_args & g, cudaStream_t stream) {
    const float sc = (float) std::pow((double) blk, -0.5);
    const dim3 grid(K / blk, batch, 1);
    const int ne1 = (int) x->ne[1], ne2 = (int) x->ne[2];
    const size_t nb1 = x->nb[1], nb2 = x->nb[2], nb3 = x->nb[3];
    const char * xd = (const char *) x->data;
    switch (blk) {
        case 128:  k_input_group_rht_128<<<grid, 32, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, g); break;
        case 256:  k_input_group_rht<256, 256><<<grid, 256, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, g); break;
        case 512:  k_input_group_rht<512, 256><<<grid, 256, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, g); break;
        case 1024: k_input_group_rht<1024, 256><<<grid, 256, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, g); break;
        case 2048: k_input_group_rht<2048, 1024><<<grid, 1024, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, g); break;
        case 4096: k_input_group_rht<4096, 1024><<<grid, 1024, 0, stream>>>(K, sc, first, ne1, ne2, nb1, nb2, nb3, xd, g); break;
        default:   GGML_ABORT("trellis hip: bad group rht block %d", blk);
    }
}

static void launch_output_group_rht(int blk, int batch, const group_out_args & g, cudaStream_t stream) {
    const float sc = (float) std::pow((double) blk, -0.5);
    const dim3 grid(g.cta0[g.n], batch, 1);
    switch (blk) {
        case 128:  k_output_group_rht_128<<<grid, 32, 0, stream>>>(g, sc); break;
        case 256:  k_output_group_rht<256, 256><<<grid, 256, 0, stream>>>(g, sc); break;
        case 512:  k_output_group_rht<512, 256><<<grid, 256, 0, stream>>>(g, sc); break;
        case 1024: k_output_group_rht<1024, 256><<<grid, 256, 0, stream>>>(g, sc); break;
        case 2048: k_output_group_rht<2048, 1024><<<grid, 1024, 0, stream>>>(g, sc); break;
        case 4096: k_output_group_rht<4096, 1024><<<grid, 1024, 0, stream>>>(g, sc); break;
        default:   GGML_ABORT("trellis hip: bad group rht block %d", blk);
    }
}

static void launch_hf_runs(int n, int shards, int len, int run, float c_run, const hf_parts & p,
                           int batch, int first_dst, const float * src, const float * post, float * dst,
                           cudaStream_t stream) {
    if (run <= 128) {
        auto go = [&](auto E_tag) {
            constexpr int E = decltype(E_tag)::value;
            const int64_t items = (int64_t) batch * shards * ((len + 32 * E - 1) / (32 * E));
            k_hf_runs_warp<E><<<dim3((unsigned) ((items + 7) / 8), 1, 1), 256, 0, stream>>>(
                n, batch, shards, len, run, c_run, p, first_dst, src, post, dst);
        };
        if (run <= 32) { go(std::integral_constant<int, 1>{}); }
        else if (run == 64) { go(std::integral_constant<int, 2>{}); }
        else { go(std::integral_constant<int, 4>{}); }
    } else {
        const int threads = std::min(run / 2, 512);
        k_hf_runs_cta<<<dim3(len / run, shards, batch), std::max(threads, 32), (size_t) run * sizeof(float), stream>>>(
            n, run, c_run, p, first_dst, src, post, dst);
    }
}

// Prepare input for one op: fuses gather + pre + input side where possible.
static void prepare_input(const op_t & o, int first, int batch, float * in, cudaStream_t stream) {
    const int K = o.K;
    const float * pre = (const float *) o.pre->data;
    const int32_t * pin = o.pin ? (const int32_t *) o.pin->data : nullptr;

    if (o.cols_kind == 1 && o.cols_block >= 128 && o.cols_block <= BLOCK_LDS_MAX) {
        launch_input_rht(K, o.cols_block, first, batch, o.x, pre, pin, in, stream);
        return;
    }
    if (o.cols_kind == 2) {
        const int len = K / o.cols_shards;
        const int m = ggml_trellis_hadfull_m(len);
        const int run = len / m;
        const float c_run = (float) std::pow((double) run, -0.5);
        const float c_m = (float) std::pow((double) m, -0.5);
        const hf_parts p = make_hf_parts(K, o.cols_shards, nullptr, 0);
        if (len <= 12288) {
            const int threads = len >= 4096 ? 512 : 256;
            k_hf_lds_fwd<<<dim3(o.cols_shards, batch, 1), threads, (size_t) len * sizeof(float), stream>>>(
                K, len, run, m, c_run, c_m, p, first, (int) o.x->ne[1], (int) o.x->ne[2],
                o.x->nb[1], o.x->nb[2], o.x->nb[3], (const char *) o.x->data, pre, pin, in);
            return;
        }
        k_input<<<dim3((K + 255) / 256, batch, 1), 256, 0, stream>>>(
            K, first, (int) o.x->ne[1], (int) o.x->ne[2], o.x->nb[1], o.x->nb[2], o.x->nb[3],
            (const char *) o.x->data, pre, pin, in);
        launch_hf_runs(K, o.cols_shards, len, run, c_run, p, batch, 0, in, nullptr, in, stream);
        if (m > 1) {
            const int g = m >= 68 ? 8 : (m >= 20 ? 16 : 32);
            const int ncol = std::min(run, g);
            const int threads = std::min(((m * ncol + 31) / 32) * 32, 1024);
            k_hf_cross<<<dim3((run + g - 1) / g, o.cols_shards, batch), threads, 0, stream>>>(
                K, m, run, g, c_m, 0, p, in, in);
        }
        return;
    }

    // kind == 0, or kind == 1 with block < 128 or block > 4096
    k_input<<<dim3((K + 255) / 256, batch, 1), 256, 0, stream>>>(
        K, first, (int) o.x->ne[1], (int) o.x->ne[2], o.x->nb[1], o.x->nb[2], o.x->nb[3],
        (const char *) o.x->data, pre, pin, in);
    if (o.cols_kind == 1 && o.cols_block > 1) {
        const float sc = (float) std::pow((double) o.cols_block, -0.5);
        if (o.cols_block == 16 || o.cols_block == 32) {
            const int64_t items = (int64_t) batch * ((K + 31) / 32);
            k_block_warp<1><<<dim3((unsigned) ((items + 7) / 8), 1, 1), 256, 0, stream>>>(
                K, batch, o.cols_block, sc, 0, in, nullptr, in);
        } else if (o.cols_block == 64) {
            const int64_t items = (int64_t) batch * ((K + 63) / 64);
            k_block_warp<2><<<dim3((unsigned) ((items + 7) / 8), 1, 1), 256, 0, stream>>>(
                K, batch, o.cols_block, sc, 0, in, nullptr, in);
        } else {
            const int threads = std::max(std::min(o.cols_block / 2, 512), 1);
            k_block<<<dim3(K / o.cols_block, batch, 1), threads, (size_t) o.cols_block * sizeof(float), stream>>>(
                K, o.cols_block, sc, in);
        }
    }
}

// Finish output for one op: fuses output side + post even when pout != nullptr (then gathers if gather=true).
static void finish_output(const op_t & o, int first, int batch, float * out, float * dst, bool gather,
                          cudaStream_t stream) {
    const int N = o.N;
    const float * post = (const float *) o.post->data;
    const int32_t * pout = o.pout ? (const int32_t *) o.pout->data : nullptr;
    float * side_dst = pout ? out : dst;
    const int side_first = pout ? 0 : first;

    if (o.rows_kind == 1 && o.rows_block >= 128 && o.rows_block <= BLOCK_LDS_MAX) {
        launch_output_rht(N, o.rows_block, side_first, batch, out, post, side_dst, stream);
        if (pout && gather) {
            k_output<<<dim3((N + 255) / 256, batch, 1), 256, 0, stream>>>(N, first, out, nullptr, pout, dst);
        }
        return;
    }
    if (o.rows_kind == 2) {
        const int len = N / o.rows_shards;
        const int m = ggml_trellis_hadfull_m(len);
        const int run = len / m;
        const float c_run = (float) std::pow((double) run, -0.5);
        const float c_m = (float) std::pow((double) m, -0.5);
        const hf_parts p = make_hf_parts(N, o.rows_shards, o.parts, o.nparts);
        if ((m == 1 && len <= 12288) || (m > 1 && len <= 6144)) {
            const int threads = len >= 4096 ? 512 : 256;
            const size_t smem = (size_t) (m > 1 ? 2 * len : len) * sizeof(float);
            k_hf_lds_inv<<<dim3(o.rows_shards, batch, 1), threads, smem, stream>>>(
                N, len, run, m, c_run, c_m, p, side_first, out, post, side_dst);
        } else {
            if (m > 1) {
                const int g = m >= 68 ? 8 : (m >= 20 ? 16 : 32);
                const int ncol = std::min(run, g);
                const int threads = std::min(((m * ncol + 31) / 32) * 32, 1024);
                k_hf_cross<<<dim3((run + g - 1) / g, o.rows_shards, batch), threads, 0, stream>>>(
                    N, m, run, g, c_m, 1, p, out, out);
            }
            launch_hf_runs(N, o.rows_shards, len, run, c_run, p, batch, side_first, out, post, side_dst, stream);
        }
        if (pout && gather) {
            k_output<<<dim3((N + 255) / 256, batch, 1), 256, 0, stream>>>(N, first, out, nullptr, pout, dst);
        }
        return;
    }
    if (o.rows_kind == 1 && o.rows_block > 1) {
        const float sc = (float) std::pow((double) o.rows_block, -0.5);
        if (o.rows_block == 16 || o.rows_block == 32) {
            const int64_t items = (int64_t) batch * ((N + 31) / 32);
            k_block_warp<1><<<dim3((unsigned) ((items + 7) / 8), 1, 1), 256, 0, stream>>>(
                N, batch, o.rows_block, sc, side_first, out, post, side_dst);
            if (pout && gather) {
                k_output<<<dim3((N + 255) / 256, batch, 1), 256, 0, stream>>>(N, first, out, nullptr, pout, dst);
            }
            return;
        }
        if (o.rows_block == 64) {
            const int64_t items = (int64_t) batch * ((N + 63) / 64);
            k_block_warp<2><<<dim3((unsigned) ((items + 7) / 8), 1, 1), 256, 0, stream>>>(
                N, batch, o.rows_block, sc, side_first, out, post, side_dst);
            if (pout && gather) {
                k_output<<<dim3((N + 255) / 256, batch, 1), 256, 0, stream>>>(N, first, out, nullptr, pout, dst);
            }
            return;
        }
        const int threads = std::max(std::min(o.rows_block / 2, 512), 1);
        k_block<<<dim3(N / o.rows_block, batch, 1), threads, (size_t) o.rows_block * sizeof(float), stream>>>(
            N, o.rows_block, sc, out);
    }
    // kind == 0, or kind == 1 fallback (block < 16 or > 4096)
    if (pout && !gather) {
        k_output<<<dim3((N + 255) / 256, batch, 1), 256, 0, stream>>>(N, 0, out, post, nullptr, out);
    } else {
        k_output<<<dim3((N + 255) / 256, batch, 1), 256, 0, stream>>>(N, first, out, post, pout, dst);
    }
}

static inline int mv1_fuse_mode(const op_t & o) {
    if (o.pout != nullptr) {
        return 0;
    }
    if (o.rows_kind == 0) {
        return 1;
    }
    if (o.rows_kind == 1 && o.rows_block == 128) {
        return 2;
    }
    return 0;
}

static inline int choose_mv1_splits(int total_wgs, int K) {
    const int kt = K >> 4;
    if (total_wgs >= 128 || kt < 32) {
        return 1;
    }
    int splits = (128 + total_wgs - 1) / total_wgs;
    splits = std::min(splits, kt / 16);
    splits = std::min(splits, 16);
    return std::max(splits, 1);
}

static void launch_mv1_group(const mv1_group_args & g, cudaStream_t stream) {
    bool mixed = false;
    const int bits0 = g.m[0].bits;
    for (int i = 1; i < g.n; ++i) {
        if (g.m[i].bits != bits0) { mixed = true; }
    }
    const dim3 grid(g.cta0[g.n], g.splits, 1);
    if (mixed) {
        k_product_mv1_mixed_group<<<grid, 256, 0, stream>>>(g);
        return;
    }
    switch (bits0) {
        case 1: k_product_mv1_group<1><<<grid, 256, 0, stream>>>(g); break;
        case 2: k_product_mv1_group<2><<<grid, 256, 0, stream>>>(g); break;
        case 3: k_product_mv1_group<3><<<grid, 256, 0, stream>>>(g); break;
        case 4: k_product_mv1_group<4><<<grid, 256, 0, stream>>>(g); break;
        case 5: k_product_mv1_group<5><<<grid, 256, 0, stream>>>(g); break;
        case 6: k_product_mv1_group<6><<<grid, 256, 0, stream>>>(g); break;
        default: GGML_ABORT("trellis hip: bad bit width %d", bits0);
    }
}

// Prefill/batch path on matrix cores (Plan 2 Q1). On by default for RDNA4; set GGML_TRELLIS_HIP_MM=0
// to force the scalar GEMV (the correctness oracle and the fallback for odd shapes / small batches).
static bool hip_mm_enabled() {
    static const bool on = [] {
        const char * s = getenv("GGML_TRELLIS_HIP_MM");
        return s == nullptr || atoi(s) != 0;
    }();
    return on;
}

static inline bool hip_wmma_supported(int cc) {
    return GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) ||
           GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc);
}

template <int BITS>
static void launch_mm_128x128(ggml_cuda_pool & pool, const uint32_t * codes, const half * s16,
                              const float * in, float * out, const float * post, int fuse_mode,
                              int K, int N, int batch, cudaStream_t stream) {
    const dim3 grid((N / 16 + 7) / 8, (batch + 127) / 128, 1);
    const int kt = K >> 4;
    ggml_cuda_pool_alloc<float> act_buf(pool, (size_t) batch * K);
    half2 * x_hilo = reinterpret_cast<half2 *>(act_buf.ptr);
    const int64_t total_vec = (int64_t) batch * kt * 4;
    k_prep_act_hilo<<<(int) ((total_vec + 255) / 256), 256, 0, stream>>>(in, s16, x_hilo, K, batch);
    k_product_mm_128x128<BITS, 2><<<grid, 256, 0, stream>>>(codes, x_hilo, out, post, fuse_mode, K, N, batch);
}

#define TRELLIS_HIP_MV_LAUNCH(BITS, COLS)                                                                  \
    k_product_mv_v2<BITS, COLS><<<dim3((N + 127) / 128, (batch + (COLS) - 1) / (COLS), 1), 256, 0, stream>>>( \
        codes, s16, in, out, nullptr, 0, K, N, batch, 1)

#define TRELLIS_HIP_CASE(BITS)                                                                             \
    case BITS:                                                                                             \
        if (batch == 1) {                                                                                  \
            k_product_mv_v2<BITS, 1><<<dim3((N + 127) / 128, splits, 1), 256, 0, stream>>>(                \
                codes, s16, in, out, post, fuse_mode, K, N, 1, splits);                                    \
        } else if (wmma && batch >= 9 && N % 16 == 0 && hip_mm_enabled()) {                                \
            if (batch >= 32 && N >= 128) {                                                                 \
                launch_mm_128x128<BITS>(pool, codes, s16, in, out, post, fuse_mode, K, N, batch, stream);  \
            } else if (batch >= 32) {                                                                      \
                const dim3 grid((N / 16 + 3) / 4, (batch + 127) / 128, 1);                                 \
                k_product_mm_64x128<BITS, 2><<<grid, 256, 0, stream>>>(codes, s16, in, out, K, N, batch);  \
            } else {                                                                                       \
                const dim3 grid((N / 16 + 7) / 8, (batch + 31) / 32, 1);                                   \
                k_product_mm_128x32<BITS, 2><<<grid, 256, 0, stream>>>(codes, s16, in, out, K, N, batch);  \
            }                                                                                              \
        } else if (batch <= 2) {                                                                           \
            TRELLIS_HIP_MV_LAUNCH(BITS, 2);                                                                \
        } else if (batch <= 4) {                                                                           \
            TRELLIS_HIP_MV_LAUNCH(BITS, 4);                                                                \
        } else {                                                                                           \
            TRELLIS_HIP_MV_LAUNCH(BITS, 8);                                                                \
        }                                                                                                  \
        break

static void product(ggml_cuda_pool & pool, const op_t & o, const float * in, float * out, int batch,
                    bool wmma, cudaStream_t stream,
                    const float * post = nullptr, int fuse_mode = 0, int splits = 1) {
    const int K = o.K, N = o.N;
    const uint32_t * codes = (const uint32_t *) o.w->data;
    const half * s16 = (const half *) o.s16->data;
    switch (o.bits) {
        TRELLIS_HIP_CASE(1);
        TRELLIS_HIP_CASE(2);
        TRELLIS_HIP_CASE(3);
        TRELLIS_HIP_CASE(4);
        TRELLIS_HIP_CASE(5);
        TRELLIS_HIP_CASE(6);
        default: GGML_ABORT("trellis hip: bad bit width %d", o.bits);
    }
}

#undef TRELLIS_HIP_CASE
#undef TRELLIS_HIP_MV_LAUNCH

} // namespace trellis_hip

// ---- backend entry points ------------------------------------------------------------------------------

bool ggml_cuda_trellis_supported(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (!w || !x || !ggml_is_trellis_type(w->type) || !ggml_is_matrix(w) || !ggml_is_contiguous(w)) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || x->ne[0] != w->ne[0]) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    const int32_t * d = (const int32_t *) op->op_params;
    if (d[6] < 0 || d[6] > GGML_TRELLIS_MAX_PARTS ||
        !ggml_trellis_desc_supported(w->ne[0], w->ne[1], d, 7 + d[6])) {
        return false;
    }
    const trellis_hip::op_t o(op);
    if (o.K > INT32_MAX / 2 || o.N > INT32_MAX / 2 || o.N % 4 || o.T > INT32_MAX) {
        return false;
    }
    // the side kernels index rows with int and run one grid row per token chunk
    auto side_ok = [](int64_t n, int kind, int block, int shards) {
        if (n > INT32_MAX / 2) {
            return false;
        }
        if (kind == 1) {
            return block >= 2 && block <= 16384;
        }
        if (kind == 2) {
            return shards >= 1 && n / shards <= 24576;
        }
        return true;
    };
    if (!side_ok(o.N, o.rows_kind, o.rows_block, o.rows_shards) ||
        !side_ok(o.K, o.cols_kind, o.cols_block, o.cols_shards)) {
        return false;
    }
    // the side tensors are required and must be contiguous with the shapes the format fixes
    const ggml_type types[] = {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_I32};
    const int64_t sizes[]   = {o.K / 16, o.K, o.N, o.K, o.N};
    for (int i = 0; i < 5; ++i) {
        const ggml_tensor * t = op->src[i + 2];
        if (!t) {
            if (i < 3) {
                return false;
            }
            continue;
        }
        if (t->type != types[i] || !ggml_is_contiguous(t) || ggml_nelements(t) != sizes[i]) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_op_trellis(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace trellis_hip;
    const op_t o(dst);
    const int K = o.K, N = o.N;
    cudaStream_t stream = ctx.stream();

    if (o.T == 1) {
        const int fmode = mv1_fuse_mode(o);
        const int splits = choose_mv1_splits((N + 127) / 128, K);
        const size_t n_in  = (size_t) K;
        const size_t n_out = splits > 1 ? (size_t) splits * N : (fmode > 0 ? 0 : (size_t) N);
        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n_in + n_out);
        float * const in  = scratch.ptr;
        float * const out = in + n_in;

        prepare_input(o, 0, 1, in, stream);
        if (splits > 1) {
            product(ctx.pool(), o, in, out, 1, false, stream, nullptr, 0, splits);
            float * red_dst = fmode > 0 ? (float *) dst->data : out;
            k_splitk_reduce<<<(N + 127) / 128, 128, 0, stream>>>(
                out, (const float *) o.post->data, red_dst, N, splits, fmode);
            if (fmode == 0) {
                finish_output(o, 0, 1, out, (float *) dst->data, true, stream);
            }
        } else if (fmode > 0) {
            product(ctx.pool(), o, in, (float *) dst->data, 1, false, stream, (const float *) o.post->data, fmode, 1);
        } else {
            product(ctx.pool(), o, in, out, 1, false, stream);
            finish_output(o, 0, 1, out, (float *) dst->data, true, stream);
        }
        return;
    }

    const int64_t capacity = std::min<int64_t>(o.T, BATCH);
    const size_t n_in  = (size_t) capacity * K;
    const size_t n_out = (size_t) capacity * N;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n_in + n_out);
    float * const in  = scratch.ptr;
    float * const out = in + n_in;

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool wmma = hip_wmma_supported(cc);

    for (int64_t first = 0; first < o.T; first += BATCH) {
        const int batch = (int) std::min<int64_t>(o.T - first, BATCH);
        const bool can_fuse_mm = wmma && batch >= 32 && N >= 128 && N % 16 == 0 && hip_mm_enabled();
        const int fmode = can_fuse_mm ? mv1_fuse_mode(o) : 0;
        prepare_input(o, (int) first, batch, in, stream);
        if (fmode > 0) {
            product(ctx.pool(), o, in, (float *) dst->data + first * N, batch, wmma, stream,
                    (const float *) o.post->data, fmode, 1);
        } else {
            product(ctx.pool(), o, in, out, batch, wmma, stream);
            finish_output(o, (int) first, batch, out, (float *) dst->data, true, stream);
        }
    }
}

static bool trellis_hip_groupable(const ggml_tensor * t) {
    if (t->op != GGML_OP_TRELLIS || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !ggml_cuda_trellis_supported(t)) {
        return false;
    }
    const trellis_hip::op_t o(t);
    return o.pin == nullptr;
}

int ggml_cuda_trellis_group_size(const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * a = cgraph->nodes[i];
    if (!trellis_hip_groupable(a)) {
        return 1;
    }
    int n = 1;
    while (n < TRELLIS_GROUP_MAX && i + n < cgraph->n_nodes) {
        const ggml_tensor * b = cgraph->nodes[i + n];
        if (!trellis_hip_groupable(b) || b->src[1] != a->src[1] || b->src[0]->ne[0] != a->src[0]->ne[0]) {
            break;
        }
        ++n;
    }
    return n;
}

void ggml_cuda_op_trellis_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, int n) {
    using namespace trellis_hip;
    GGML_ASSERT(n >= 1 && n <= TRELLIS_GROUP_MAX);
    cudaStream_t stream = ctx.stream();

    op_t ops[TRELLIS_GROUP_MAX] = {
        op_t(nodes[0]),
        op_t(nodes[n > 1 ? 1 : 0]),
        op_t(nodes[n > 2 ? 2 : 0]),
        op_t(nodes[n > 3 ? 3 : 0]),
        op_t(nodes[n > 4 ? 4 : 0]),
        op_t(nodes[n > 5 ? 5 : 0]),
        op_t(nodes[n > 6 ? 6 : 0]),
        op_t(nodes[n > 7 ? 7 : 0]),
    };

    const int K = ops[0].K;
    const int64_t T = ops[0].T;
    const int64_t capacity = std::min<int64_t>(T, BATCH);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool wmma = hip_wmma_supported(cc);

    // Deduplicate input buffers when (cols_kind, cols_block, cols_shards, pre) match
    int in_slot[TRELLIS_GROUP_MAX];
    int n_in_slots = 0;
    for (int i = 0; i < n; ++i) {
        int found = -1;
        for (int j = 0; j < i; ++j) {
            if (ops[i].cols_kind == ops[j].cols_kind &&
                ops[i].cols_block == ops[j].cols_block &&
                ops[i].cols_shards == ops[j].cols_shards &&
                ops[i].pre->data == ops[j].pre->data) {
                found = in_slot[j];
                break;
            }
        }
        in_slot[i] = found >= 0 ? found : n_in_slots++;
    }

    int total_wgs = 0;
    int fmodes[TRELLIS_GROUP_MAX] = {};
    for (int i = 0; i < n; ++i) {
        total_wgs += (ops[i].N + 127) / 128;
        fmodes[i] = T == 1 ? mv1_fuse_mode(ops[i]) : 0;
    }
    const int splits = T == 1 ? choose_mv1_splits(total_wgs, K) : 1;

    size_t total_out = 0;
    for (int i = 0; i < n; ++i) {
        if (T == 1 && splits == 1 && fmodes[i] > 0) {
            continue;
        }
        total_out += (size_t) (T == 1 ? splits : capacity) * ops[i].N;
    }
    const size_t total_in = (size_t) n_in_slots * capacity * K;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), total_in + total_out);

    float * in_bufs[TRELLIS_GROUP_MAX];
    for (int s = 0; s < n_in_slots; ++s) {
        in_bufs[s] = scratch.ptr + (size_t) s * capacity * K;
    }
    float * out_bufs[TRELLIS_GROUP_MAX];
    float * op_ptr = scratch.ptr + total_in;
    for (int i = 0; i < n; ++i) {
        if (T == 1 && splits == 1 && fmodes[i] > 0) {
            out_bufs[i] = (float *) nodes[i]->data;
        } else {
            out_bufs[i] = op_ptr;
            op_ptr += (size_t) (T == 1 ? splits : capacity) * ops[i].N;
        }
    }

    for (int64_t first = 0; first < T; first += BATCH) {
        const int batch = (int) std::min<int64_t>(T - first, BATCH);

        // 1. Prepare inputs (batch members sharing cols_kind == 1 and same cols_block in [128, 4096])
        bool slot_done[TRELLIS_GROUP_MAX] = {};
        for (int i = 0; i < n; ++i) {
            const int s = in_slot[i];
            if (slot_done[s]) { continue; }
            if (ops[i].cols_kind == 1 && ops[i].cols_block >= 128 && ops[i].cols_block <= BLOCK_LDS_MAX) {
                const int blk = ops[i].cols_block;
                group_in_args gi = {};
                for (int j = i; j < n; ++j) {
                    const int sj = in_slot[j];
                    if (!slot_done[sj] && ops[j].cols_kind == 1 && ops[j].cols_block == blk) {
                        gi.pre[gi.n] = (const float *) ops[j].pre->data;
                        gi.dst[gi.n] = in_bufs[sj];
                        gi.n++;
                        slot_done[sj] = true;
                    }
                }
                if (gi.n > 1) {
                    launch_input_group_rht(K, blk, (int) first, batch, ops[i].x, gi, stream);
                } else {
                    launch_input_rht(K, blk, (int) first, batch, ops[i].x, gi.pre[0], nullptr, gi.dst[0], stream);
                }
            } else {
                prepare_input(ops[i], (int) first, batch, in_bufs[s], stream);
                slot_done[s] = true;
            }
        }

        // 2. Product (grouped single launch for batch == 1, with Split-K and fused epilogue)
        bool out_done[TRELLIS_GROUP_MAX] = {};
        if (batch == 1) {
            mv1_group_args g = {};
            g.n = n;
            g.K = K;
            g.splits = splits;
            for (int i = 0; i < n; ++i) {
                g.m[i].codes     = (const uint32_t *) ops[i].w->data;
                g.m[i].s16       = (const half *) ops[i].s16->data;
                g.m[i].x         = in_bufs[in_slot[i]];
                g.m[i].y         = out_bufs[i];
                g.m[i].post      = (splits == 1 && fmodes[i] > 0) ? (const float *) ops[i].post->data : nullptr;
                g.m[i].fuse_mode = splits == 1 ? fmodes[i] : 0;
                g.m[i].N         = ops[i].N;
                g.m[i].bits      = ops[i].bits;
                g.cta0[i + 1]    = g.cta0[i] + (ops[i].N + 127) / 128;
                if (splits == 1 && fmodes[i] > 0) {
                    out_done[i] = true;
                }
            }
            launch_mv1_group(g, stream);
            if (splits > 1) {
                splitk_group_args sg = {};
                sg.n = n;
                sg.splits = splits;
                for (int i = 0; i < n; ++i) {
                    sg.partials[i]  = out_bufs[i];
                    sg.post[i]      = (const float *) ops[i].post->data;
                    sg.dst[i]       = fmodes[i] > 0 ? (float *) nodes[i]->data : out_bufs[i];
                    sg.N[i]         = ops[i].N;
                    sg.fuse_mode[i] = fmodes[i];
                    sg.cta0[i + 1]  = sg.cta0[i] + (ops[i].N + 127) / 128;
                    if (fmodes[i] > 0) {
                        out_done[i] = true;
                    }
                }
                k_splitk_reduce_group<<<sg.cta0[n], 128, 0, stream>>>(sg);
            }
        } else {
            for (int i = 0; i < n; ++i) {
                const bool can_fuse_mm = wmma && batch >= 32 && ops[i].N >= 128 && ops[i].N % 16 == 0 && hip_mm_enabled();
                const int fm = can_fuse_mm ? mv1_fuse_mode(ops[i]) : 0;
                if (fm > 0) {
                    product(ctx.pool(), ops[i], in_bufs[in_slot[i]], (float *) nodes[i]->data + first * ops[i].N,
                            batch, wmma, stream, (const float *) ops[i].post->data, fm, 1);
                    out_done[i] = true;
                } else {
                    product(ctx.pool(), ops[i], in_bufs[in_slot[i]], out_bufs[i], batch, wmma, stream);
                }
            }
        }

        // 3. Output sides (batch members sharing rows_kind == 1 and same rows_block in [128, 4096]) + batched pout gather
        gather_set gset = {};
        int nseg = 0, nmax = 0;
        for (int i = 0; i < n; ++i) {
            if (out_done[i]) { continue; }
            if (ops[i].rows_kind == 1 && ops[i].rows_block >= 128 && ops[i].rows_block <= BLOCK_LDS_MAX) {
                const int blk = ops[i].rows_block;
                group_out_args go = {};
                for (int j = i; j < n; ++j) {
                    if (!out_done[j] && ops[j].rows_kind == 1 && ops[j].rows_block == blk) {
                        const bool has_pout = ops[j].pout != nullptr;
                        go.src[go.n]       = out_bufs[j];
                        go.post[go.n]      = (const float *) ops[j].post->data;
                        go.dst[go.n]       = has_pout ? out_bufs[j] : (float *) nodes[j]->data;
                        go.N[go.n]         = ops[j].N;
                        go.first_dst[go.n] = has_pout ? 0 : (int) first;
                        go.cta0[go.n + 1]  = go.cta0[go.n] + ops[j].N / blk;
                        go.n++;
                        out_done[j] = true;
                        if (has_pout) {
                            gset.seg[nseg++] = {out_bufs[j], (const int32_t *) ops[j].pout->data, (float *) nodes[j]->data, ops[j].N};
                            nmax = std::max(nmax, ops[j].N);
                        }
                    }
                }
                if (go.n > 1) {
                    launch_output_group_rht(blk, batch, go, stream);
                } else {
                    launch_output_rht(go.N[0], blk, go.first_dst[0], batch, go.src[0], go.post[0], go.dst[0], stream);
                }
            } else {
                finish_output(ops[i], (int) first, batch, out_bufs[i], (float *) nodes[i]->data, false, stream);
                out_done[i] = true;
                if (ops[i].pout != nullptr) {
                    gset.seg[nseg++] = {out_bufs[i], (const int32_t *) ops[i].pout->data, (float *) nodes[i]->data, ops[i].N};
                    nmax = std::max(nmax, ops[i].N);
                }
            }
        }
        if (nseg > 0) {
            const int blocks = std::min((nmax + 255) / 256, 64);
            k_gather_multi<<<dim3(blocks, batch, nseg), 256, 0, stream>>>(gset, (int) first);
        }
    }
}

// Fuse MLP gate/up output Hadamard + SwiGLU + down input Hadamard into one kernel.
int ggml_cuda_trellis_mlp_chain(const ggml_cgraph * cgraph, int i) {
    using namespace trellis_hip;
    if (i + 3 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * a = cgraph->nodes[i];
    const ggml_tensor * b = cgraph->nodes[i + 1];
    const ggml_tensor * glu = cgraph->nodes[i + 2];
    const ggml_tensor * down = cgraph->nodes[i + 3];
    if (a->op != GGML_OP_TRELLIS || b->op != GGML_OP_TRELLIS || glu->op != GGML_OP_GLU || down->op != GGML_OP_TRELLIS) {
        return 0;
    }
    if (!ggml_cuda_trellis_supported(a) || !ggml_cuda_trellis_supported(b) || !ggml_cuda_trellis_supported(down)) {
        return 0;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 || glu->src[1] == nullptr ||
        glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu)) {
        return 0;
    }
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (!((gate == a && up == b) || (gate == b && up == a)) || down->src[1] != glu) {
        return 0;
    }
    const op_t og(gate), ou(up), od(down);
    const int blk = og.rows_block;
    if (og.x != ou.x || og.K != ou.K ||
        og.rows_kind != 1 || ou.rows_kind != 1 || ou.rows_block != blk || og.pout != nullptr || ou.pout != nullptr ||
        od.cols_kind != 1 || od.cols_block != blk || od.pin != nullptr ||
        blk < 128 || blk > BLOCK_LDS_MAX || (blk & (blk - 1)) != 0 ||
        og.N != ou.N || od.K != og.N || og.N % blk != 0 || od.T != og.T) {
        return 0;
    }
    const ggml_op ops[4] = {GGML_OP_TRELLIS, GGML_OP_TRELLIS, GGML_OP_GLU, GGML_OP_TRELLIS};
    const int outputs[1] = {i + 3};
    if (!ggml_can_fuse_subgraph(cgraph, i, 4, ops, outputs, 1)) {
        return 0;
    }
    return 4;
}

void ggml_cuda_op_trellis_mlp_chain(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes) {
    using namespace trellis_hip;
    const ggml_tensor * glu = nodes[2];
    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up   = glu->src[1];
    ggml_tensor * down = nodes[3];
    const op_t og(gate), ou(up), od(down);
    cudaStream_t stream = ctx.stream();

    const int64_t capacity = std::min<int64_t>(og.T, BATCH);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool wmma = hip_wmma_supported(cc);

    const bool same_in = og.cols_kind == ou.cols_kind && og.cols_block == ou.cols_block &&
                         og.cols_shards == ou.cols_shards && og.pin == ou.pin &&
                         og.pre->data == ou.pre->data;
    const bool group_in_rht = !same_in && og.pin == nullptr && ou.pin == nullptr &&
                              og.cols_kind == 1 && ou.cols_kind == 1 && og.cols_block == ou.cols_block &&
                              og.cols_block >= 128 && og.cols_block <= BLOCK_LDS_MAX;

    const int splits_gu = og.T == 1 ? choose_mv1_splits((og.N + 127) / 128 + (ou.N + 127) / 128, og.K) : 1;
    const int splits_d  = og.T == 1 ? choose_mv1_splits((od.N + 127) / 128, od.K) : 1;
    const int fmode_d   = og.T == 1 ? mv1_fuse_mode(od) : 0;

    const size_t n_in   = (size_t) (same_in ? 1 : 2) * capacity * og.K;
    const size_t n_xd   = (size_t) capacity * og.N;
    const size_t n_pgu  = splits_gu > 1 ? (size_t) 2 * splits_gu * og.N : 0;
    const size_t n_out  = og.T == 1
        ? (splits_d > 1 ? (size_t) splits_d * od.N : (fmode_d > 0 ? 0 : (size_t) od.N))
        : (size_t) capacity * od.N;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n_in + n_xd + n_pgu + n_out);
    float * in_g  = scratch.ptr;
    float * in_u  = same_in ? in_g : in_g + (size_t) capacity * og.K;
    float * xd    = scratch.ptr + n_in;
    float * p_g   = xd + n_xd;
    float * p_u   = p_g + (size_t) splits_gu * og.N;
    float * out_d = p_g + n_pgu;

    const int blk = og.rows_block;

    for (int64_t first = 0; first < og.T; first += BATCH) {
        const int batch = (int) std::min<int64_t>(og.T - first, BATCH);

        if (same_in) {
            prepare_input(og, (int) first, batch, in_g, stream);
        } else if (group_in_rht) {
            group_in_args gi = {};
            gi.pre[0] = (const float *) og.pre->data; gi.dst[0] = in_g;
            gi.pre[1] = (const float *) ou.pre->data; gi.dst[1] = in_u;
            gi.n = 2;
            launch_input_group_rht(og.K, og.cols_block, (int) first, batch, og.x, gi, stream);
        } else {
            prepare_input(og, (int) first, batch, in_g, stream);
            prepare_input(ou, (int) first, batch, in_u, stream);
        }

        if (batch == 1) {
            mv1_group_args g = {};
            g.n = 2;
            g.K = og.K;
            g.splits = splits_gu;
            g.m[0] = {(const uint32_t *) og.w->data, (const half *) og.s16->data, in_g, splits_gu > 1 ? p_g : (float *) gate->data, nullptr, 0, og.N, og.bits};
            g.m[1] = {(const uint32_t *) ou.w->data, (const half *) ou.s16->data, in_u, splits_gu > 1 ? p_u : (float *) up->data,   nullptr, 0, ou.N, ou.bits};
            g.cta0[1] = (og.N + 127) / 128;
            g.cta0[2] = g.cta0[1] + (ou.N + 127) / 128;
            launch_mv1_group(g, stream);
            if (splits_gu > 1) {
                splitk_group_args sg = {};
                sg.n = 2;
                sg.splits = splits_gu;
                sg.partials[0] = p_g; sg.dst[0] = (float *) gate->data; sg.N[0] = og.N; sg.cta0[1] = (og.N + 127) / 128;
                sg.partials[1] = p_u; sg.dst[1] = (float *) up->data;   sg.N[1] = ou.N; sg.cta0[2] = sg.cta0[1] + (ou.N + 127) / 128;
                k_splitk_reduce_group<<<sg.cta0[2], 128, 0, stream>>>(sg);
            }
        } else {
            product(ctx.pool(), og, in_g, (float *) gate->data, batch, wmma, stream);
            product(ctx.pool(), ou, in_u, (float *) up->data,   batch, wmma, stream);
        }

        launch_swiglu_rht(og.N, blk, batch, (const float *) gate->data, (const float *) up->data, xd,
                          (const float *) og.post->data, (const float *) ou.post->data, (const float *) od.pre->data, stream);

        if (batch == 1) {
            if (splits_d > 1) {
                product(ctx.pool(), od, xd, out_d, 1, false, stream, nullptr, 0, splits_d);
                float * red_dst = fmode_d > 0 ? (float *) down->data : out_d;
                k_splitk_reduce<<<(od.N + 127) / 128, 128, 0, stream>>>(
                    out_d, (const float *) od.post->data, red_dst, od.N, splits_d, fmode_d);
                if (fmode_d == 0) {
                    finish_output(od, 0, 1, out_d, (float *) down->data, true, stream);
                }
            } else if (fmode_d > 0) {
                product(ctx.pool(), od, xd, (float *) down->data, 1, false, stream, (const float *) od.post->data, fmode_d, 1);
            } else {
                product(ctx.pool(), od, xd, out_d, 1, false, stream);
                finish_output(od, 0, 1, out_d, (float *) down->data, true, stream);
            }
        } else {
            const bool can_fuse_d = wmma && batch >= 32 && od.N >= 128 && od.N % 16 == 0 && hip_mm_enabled();
            const int fmd = can_fuse_d ? mv1_fuse_mode(od) : 0;
            if (fmd > 0) {
                product(ctx.pool(), od, xd, (float *) down->data + first * od.N, batch, wmma, stream,
                        (const float *) od.post->data, fmd, 1);
            } else {
                product(ctx.pool(), od, xd, out_d, batch, wmma, stream);
                finish_output(od, (int) first, batch, out_d, (float *) down->data, true, stream);
            }
        }
    }
}
