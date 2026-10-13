// P7-20: hand-written CUDA B1 (state adjoint / `dhu`) in the K2 shape of P7-19
// (register-resident state + `mma.sync`/`ldmatrix`), now with a **TMA + warp-specialized**
// operand path — the exact "next step" flagged by §22.5.
//
// Differences vs `b1_reg_probe.cu` (P7-19):
//   * operands q/k/w/dO/dV are staged into smem by **`cp.async.bulk.tensor` (TMA)** issues
//     from a dedicated *producer* warp (warp index NW); the NW compute warps wait on an
//     `mbarrier` per stage (`mbarrier::complete_tx::bytes`) instead of `__pipeline_wait_prior`.
//   * the smem operand layout uses **TMA SWIZZLE_128B** (`Swizzle<3,4,3>` on the byte
//     offset), so `ldmatrix` is conflict-free *without* the +8 padding: q/k/w rows are
//     split into two 64-column (128 B) swizzle atoms; dO/dV is a single 64-column atom.
//     The ldmatrix helpers apply the same XOR to compute the swizzled row address.
//   * no `__syncthreads()` in the steady-state loop: stage reuse is handed back to the
//     producer via a per-stage `empty` mbarrier (arrive count = NW).
//
// Recurrence unchanged (mirrors check_bwd_dhu_kernel.py / fla/ops/common/chunk_delta_h.py):
//   for c = NT-1 .. 0:
//       dh[c] = bf16(S) ; du = k @ S + dv ; dv2[c] = bf16(du)
//       S = S*e + scale*(q^T @ do) - (w^T @ du)
//   dh0 = S
//
// Build (inside kernel_lab):
//   nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared b1_tma_probe.cu -o /tmp/b1_tma.so -lcuda
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>

#define DEV __device__ __forceinline__
using bf16 = __nv_bfloat16;

constexpr int KD = 128;   // K = D
constexpr int VD = 128;   // V = D
constexpr int BV = 64;    // V columns per CTA (== one 128 B swizzle atom)
constexpr int NW = 4;     // compute warps per CTA
constexpr int BW = BV / NW;   // V columns per warp (== 16)

constexpr int BWS = BW + 8;   // per-warp sdh/sdu row stride (kept padded; written by warp)

DEV float b2f(bf16 x) { return __bfloat162float(x); }
DEV bf16  f2b(float x) { return __float2bfloat16(x); }
DEV uint32_t saddr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---- mbarrier / TMA ---------------------------------------------------------
DEV void mbar_init(uint64_t* b, uint32_t cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(saddr(b)), "r"(cnt));
}
DEV void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(saddr(b)) : "memory");
}
DEV void mbar_arrive_tx(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 ::"r"(saddr(b)), "r"(bytes) : "memory");
}
// B1T_EPROXY (P7-36): order the generic-proxy smem reads against the async-proxy
// (TMA) writes to the same stage buffer.  `__threadfence_block` (membar.cta) does NOT
// order across the two proxies; the correct primitive is fence.proxy.async.
DEV void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
DEV void mbar_wait(uint64_t* b, int ph) {
    asm volatile(
        "{\n\t.reg .pred P;\n"
        "LAB_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@!P bra LAB_%=;\n}\n" ::"r"(saddr(b)), "r"(ph) : "memory");
}
DEV void tma5d(const CUtensorMap* m, void* sm, int c0, int c1, int c2, int c3, int c4,
               uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2,%3,%4,%5,%6}], [%7];\n"
        ::"r"(saddr(sm)), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
          "r"(saddr(bar))
        : "memory");
}
#ifdef B1T_TMADH
// P7-32: store the per-chunk `dh` [KD,BV] tile via TMA (smem->global) instead of the
// per-thread scalar `st.global`.  §34.1 localized the scalar store's cost to `lg_throttle`
// (the LSU/store-queue backs up) + a ~3x L2 write amplification, *not* DRAM bandwidth;
// moving it to the TMA engine takes it off the LSU issue path entirely.  The tile is
// per-warp [KD, BW] (SWIZZLE_NONE, 32 B inner dim), so no cross-warp barrier is needed.
DEV void tma5d_store(const CUtensorMap* m, const void* sm, int c0, int c1, int c2, int c3,
                     int c4) {
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "
        "[%0, {%1,%2,%3,%4,%5}], [%6];\n"
        ::"l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(saddr(sm))
        : "memory");
}
#endif
#if defined(B1T_TMADH) || defined(B1T_TMADV)
DEV void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int N>
DEV void tma_store_wait_read() {
    asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
}
#endif
#ifdef B1T_TMADV
// P7-33: same idea as B1T_TMADH but for the per-chunk `dv2` output [BT,BW].  `dv2` is
// token-major [B,T,H,VD], so the per-warp slice is [BT tokens, BW cols]; a 4D tensor map
// (dims {VD,T,H,B}, box {BW,BT,1,1}, SWIZZLE_NONE) stores it off the LSU path, and TMA
// clips the OOB token rows of a non-divisible T automatically (the scalar path needs an
// explicit `tr<T`).
DEV void tma4d_store(const CUtensorMap* m, const void* sm, int c0, int c1, int c2, int c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
        "[%0, {%1,%2,%3,%4}], [%5];\n"
        ::"l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(saddr(sm))
        : "memory");
}
#endif
#ifdef B1T_ETMA
// P7-21: stage the per-chunk gate factor E^tot = 2^{g_cum[last]} ([128] fp32) via TMA
// instead of reading it from global *inside* the mma loop.  The reg kernel already
// prefetched it (`b1_reg_probe.cu:150`); §23.1 dropped it from the TMA stage, which
// moved the whole chunk's `S *= e` behind a cold global load on the critical path.
DEV void tma3d(const CUtensorMap* m, void* sm, int c0, int c1, int c2, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2,%3,%4}], [%5];\n"
        ::"r"(saddr(sm)), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(saddr(bar))
        : "memory");
}
#endif

// ---- swizzle (Swizzle<3,4,3>, 128 B) on a [rows][64] bf16 tile ------------------
DEV uint32_t sw_off(int row, int col) {   // col in [0,64)
    int b = (row * 64 + col) * 2;
#ifdef B1T_NOSW
    return (uint32_t)b;
#else
    b = b ^ (((b >> 7) & 7) << 4);
    return (uint32_t)b;
#endif
}
// scalar read of a swizzled [rows][64] bf16 tile at logical (row,col)
DEV bf16 sw_read(const bf16 (*reg)[64], int row, int col) {
    const char* p = (const char*)&reg[0][0] + sw_off(row, col);
    return *(const bf16*)p;
}
template <int BT>
DEV void ldsm_sw_x4(uint32_t o[4], const bf16 (*reg)[BT][64], int rowbase, int colbase, int l) {
    int mq = l >> 3, r = l & 7;
    int row = rowbase + ((mq & 1) ? 8 : 0) + r;
    int col = colbase + ((mq & 2) ? 8 : 0);
    const bf16* base = &reg[col >> 6][0][0];
    uint32_t a = saddr(base) + sw_off(row, col & 63);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(a));
}
template <int BT>
DEV void ldsm_sw_x4_t(uint32_t o[4], const bf16 (*reg)[BT][64], int rowbase, int colbase, int l) {
    int mq = l >> 3, r = l & 7;
    int row = rowbase + ((mq >= 2) ? 8 : 0) + r;
    int col = colbase + ((mq & 1) ? 8 : 0);
    const bf16* base = &reg[col >> 6][0][0];
    uint32_t a = saddr(base) + sw_off(row, col & 63);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(a));
}
template <int BT>
DEV void ldsm_sw_x2_t(uint32_t o[2], const bf16 (*reg)[BT][64], int rowbase, int colbase, int l) {
    int row = rowbase + (l & 15);
    uint32_t a = saddr(&reg[0][0][0]) + sw_off(row, colbase);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(o[0]), "=r"(o[1]) : "r"(a));
}
// single-region [rows][64] bf16 tile (dO/dV in this probe)
DEV void ldsm_sw1_x2_t(uint32_t o[2], const bf16 (*reg)[64], int rowbase, int colbase, int l) {
    int row = rowbase + (l & 15);
    uint32_t a = saddr(&reg[0][0]) + sw_off(row, colbase);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(o[0]), "=r"(o[1]) : "r"(a));
}
// padded x2.trans, for the per-warp sdh/sdu mirrors (written by the warp, not TMA)
DEV void ldsm_x2_t_pad(uint32_t o[2], const bf16* base, int rowbase, int colbase, int stride,
                       int l) {
    int r = l & 15;
    uint32_t a = saddr(base) + (uint32_t)((rowbase + r) * stride + colbase) * 2u;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(o[0]), "=r"(o[1]) : "r"(a));
}
DEV void mma16816(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// --------------------------------------------------------------------------- smem
template <int BT>
struct alignas(1024) Stage {
    alignas(1024) bf16 q[2][BT][64];
    alignas(1024) bf16 k[2][BT][64];
    alignas(1024) bf16 w[2][BT][64];
    alignas(1024) bf16 dO[BT][64];
    alignas(1024) bf16 dV[BT][64];
#ifdef B1T_ETMA
    alignas(1024) float e[KD];
#endif
};
template <int BT, int NSTAGE>
struct alignas(1024) Shared {
    Stage<BT> st[NSTAGE];
    bf16 sdh[NW][KD][BWS];
    bf16 sdu[NW][BT][BWS];
#ifdef B1T_TMADH
    // double-buffered per-warp `dh` store tile, unpadded [KD,BW] (TMA needs exact stride)
    alignas(1024) bf16 dho[2][NW][KD][BW];
#endif
#ifdef B1T_TMADV
    // double-buffered per-warp `dv2` store tile, unpadded [BT,BW] (token-major output)
    alignas(1024) bf16 duo[2][NW][BT][BW];
#endif
    alignas(8) uint64_t full[NSTAGE];
    alignas(8) uint64_t empty[NSTAGE];
#ifdef B1T_ETMA_SEP
    alignas(8) uint64_t efull[NSTAGE];
#endif
};

// ------------------------------------------------------------------------ kernel
template <int BT, int NSTAGE>
__global__ void __launch_bounds__((NW + 1) * 32)
b1_tma_kernel(const __grid_constant__ CUtensorMap qmap,
              const __grid_constant__ CUtensorMap kmap,
              const __grid_constant__ CUtensorMap wmap,
              const __grid_constant__ CUtensorMap dOmap,
              const __grid_constant__ CUtensorMap dVmap,
#ifdef B1T_ETMA
              const __grid_constant__ CUtensorMap emap,
#endif
#ifdef B1T_TMADH
              const __grid_constant__ CUtensorMap dhmap,
#endif
#ifdef B1T_TMADV
              const __grid_constant__ CUtensorMap dv2map,
#endif
              const float* __restrict__ etot,
              bf16* __restrict__ dh_g, float* __restrict__ dh0_g, bf16* __restrict__ dv2_g,
              int T, int NT, int H, float scale) {
    extern __shared__ __align__(1024) char smem_raw[];
    Shared<BT, NSTAGE>* sh = reinterpret_cast<Shared<BT, NSTAGE>*>(smem_raw);

    const int v0 = blockIdx.x * BV;
    const int nh = blockIdx.y;
    const int n = nh / H, h = nh % H;
    const int l = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = l >> 2, t = l & 3;

    // ---- init barriers ----
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < NSTAGE; ++s) { mbar_init(&sh->full[s], 1); mbar_init(&sh->empty[s], NW);
#ifdef B1T_ETMA_SEP
            mbar_init(&sh->efull[s], 1);
#endif
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    // =============================================================== producer warp
    if (warp == NW) {
        if (l == 0) {
            constexpr uint32_t BYTES = (uint32_t)(8 * BT * 128);   // 3 qkv halves*2 + dO+dV
            for (int i = 0; i < NT; ++i) {
#if defined(B1T_SYNC)
                __syncthreads();
#endif
#if defined(B1T_FENCE)
                __threadfence_block();
#endif
                const int c = NT - 1 - i;
                const int s = i % NSTAGE;
                const int t0 = c * BT;
                if (i >= NSTAGE) mbar_wait(&sh->empty[s], ((i / NSTAGE) + 1) & 1);
#if defined(B1T_EPROXY) || defined(B1T_EPROXY_PRO)
                fence_proxy_async();
#endif
#if defined(B1T_ETMA) && defined(B1T_ETMA_SEP)
                mbar_arrive_tx(&sh->full[s], BYTES);
                mbar_arrive_tx(&sh->efull[s], (uint32_t)(KD * 4));
#elif defined(B1T_ETMA)
                mbar_arrive_tx(&sh->full[s], BYTES + (uint32_t)(KD * 4));
#else
                mbar_arrive_tx(&sh->full[s], BYTES);
#endif
                Stage<BT>* st = &sh->st[s];
                uint64_t* bar = &sh->full[s];
                // box {64,1,1,BT,1}: dims {64,2,H,T,B} over [B,T,H,D]
                tma5d(&qmap, st->q[0], 0, 0, h, t0, n, bar);
                tma5d(&qmap, st->q[1], 0, 1, h, t0, n, bar);
                tma5d(&kmap, st->k[0], 0, 0, h, t0, n, bar);
                tma5d(&kmap, st->k[1], 0, 1, h, t0, n, bar);
                tma5d(&wmap, st->w[0], 0, 0, h, t0, n, bar);
                tma5d(&wmap, st->w[1], 0, 1, h, t0, n, bar);
                int vhalf = v0 >> 6;   // BV == 64
                tma5d(&dOmap, st->dO[0], 0, vhalf, h, t0, n, bar);
                tma5d(&dVmap, st->dV[0], 0, vhalf, h, t0, n, bar);
#ifdef B1T_ETMA
                // E^tot tile [B,NT,H,KD] fp32: dims {KD, NT*H, B}, box {KD,1,1}
#ifdef B1T_ETMA_SEP
                tma3d(&emap, st->e, 0, c * H + h, n, &sh->efull[s]);
#else
                tma3d(&emap, st->e, 0, c * H + h, n, bar);
#endif
#endif
            }
        }
    } else {
        // =========================================================== compute warps
        bf16 (*sdh)[BWS] = sh->sdh[warp];
        bf16 (*sdu)[BWS] = sh->sdu[warp];

        constexpr int MTK = KD / 16;      // 8
        constexpr int MTKt = BT / 16;
        constexpr int KT1 = KD / 16;      // 8
        constexpr int KT2 = BT / 16;

        float S[MTK][2][4];
        #pragma unroll
        for (int m = 0; m < MTK; ++m)
            #pragma unroll
            for (int nn = 0; nn < 2; ++nn)
                #pragma unroll
                for (int x = 0; x < 4; ++x) S[m][nn][x] = 0.f;

        const __nv_bfloat162 scale2 = __floats2bfloat162_rn(scale, scale);

        for (int i = 0; i < NT; ++i) {
#if defined(B1T_SYNC)
            __syncthreads();
#endif
#if defined(B1T_FENCE)
            __threadfence_block();
#endif
            const int c = NT - 1 - i;
            const int s = i % NSTAGE;
            mbar_wait(&sh->full[s], (i / NSTAGE) & 1);
#ifdef B1T_ETMA_SEP
            mbar_wait(&sh->efull[s], (i / NSTAGE) & 1);
#endif
            const Stage<BT>* st = &sh->st[s];
            const int vbase = v0 + warp * BW;
#if defined(B1T_ETMA) && defined(B1T_ECOPY)
            // P7-36: snapshot E^tot into registers immediately after the full-wait, so a
            // later WAR overwrite of st->e cannot be mistaken for a late read.
            float e_copy[KD];
            #pragma unroll
            for (int z = 0; z < KD; ++z) e_copy[z] = st->e[z];
#endif

            // ---- publish S: per-warp bf16 mirror (+ `dh` store: scalar or TMA) ----
            {
#if defined(B1T_TMADH) || defined(B1T_TMADV)
                // P7-33: free the double buffer before either store tile is (re)written.
                if (i >= 2 && l == 0) tma_store_wait_read<1>();
#endif
#if defined(B1T_TMADH)
                bf16 (*dho)[BW] = sh->dho[i & 1][warp];
                __syncwarp();
#elif !defined(B1T_NODH)
                const int64_t base = ((int64_t)(n * NT + c) * H + h) * KD * VD;
#endif
                #pragma unroll
                for (int m = 0; m < MTK; ++m)
                    #pragma unroll
                    for (int nn = 0; nn < 2; ++nn) {
                        int r0 = m * 16 + g, r1 = r0 + 8;
                        int cc = nn * 8 + t * 2;
                        bf16 b0 = f2b(S[m][nn][0]), b1 = f2b(S[m][nn][1]);
                        bf16 b2 = f2b(S[m][nn][2]), b3 = f2b(S[m][nn][3]);
                        sdh[r0][cc] = b0; sdh[r0][cc + 1] = b1;
                        sdh[r1][cc] = b2; sdh[r1][cc + 1] = b3;
#if defined(B1T_TMADH)
                        // P7-32: stage into smem; lane 0 issues the TMA store below.
                        dho[r0][cc] = b0; dho[r0][cc + 1] = b1;
                        dho[r1][cc] = b2; dho[r1][cc + 1] = b3;
#elif !defined(B1T_NODH)
                        // P7-31: skip the per-chunk `dh` global store to measure the
                        // B1->B2 round-trip a K2-shaped fusion would keep in registers.
                        int gc = vbase + cc;
                        dh_g[base + (int64_t)r0 * VD + gc] = b0;
                        dh_g[base + (int64_t)r0 * VD + gc + 1] = b1;
                        dh_g[base + (int64_t)r1 * VD + gc] = b2;
                        dh_g[base + (int64_t)r1 * VD + gc + 1] = b3;
#endif
                    }
            }
            __syncwarp();
#if defined(B1T_TMADH)
            if (l == 0) {
                tma5d_store(&dhmap, &sh->dho[i & 1][warp][0][0], vbase, 0, h, c, n);
#ifndef B1T_TMADV
                // P7-33: when the `dv2` store is also TMA, both are committed together
                // at the end of the chunk (one bulk group per chunk keeps the wait<1>).
                tma_store_commit();
#endif
            }
#endif
            // ---- op1: cdu[BT,BW] = k[BT,KD] @ S[KD,BW] ----
            float cdu[MTKt][2][4];
            #pragma unroll
            for (int m = 0; m < MTKt; ++m)
                #pragma unroll
                for (int nn = 0; nn < 2; ++nn)
                    #pragma unroll
                    for (int x = 0; x < 4; ++x) cdu[m][nn][x] = 0.f;
            #pragma unroll
            for (int m = 0; m < MTKt; ++m)
                #pragma unroll
                for (int kk = 0; kk < KT1; ++kk) {
                    uint32_t a[4];
                    ldsm_sw_x4<BT>(a, st->k, m * 16, kk * 16, l);
                    #pragma unroll
                    for (int nn = 0; nn < 2; ++nn) {
                        uint32_t b[2];
                        ldsm_x2_t_pad(b, &sdh[0][0], kk * 16, nn * 8, BWS, l);
                        mma16816(cdu[m][nn], a, b, cdu[m][nn]);
                    }
                }
            __syncwarp();

            // ---- du = cdu + dv ; write dv2 and per-warp bf16 du ----
            {
                #pragma unroll
                for (int m = 0; m < MTKt; ++m)
                    #pragma unroll
                    for (int nn = 0; nn < 2; ++nn) {
                        int r0 = m * 16 + g, r1 = r0 + 8;
                        int cc = nn * 8 + t * 2;
                        int tr0 = c * BT + r0, tr1 = c * BT + r1;
                        float d0 = cdu[m][nn][0], d1 = cdu[m][nn][1];
                        float d2 = cdu[m][nn][2], d3 = cdu[m][nn][3];
                        if (tr0 < T) { d0 += b2f(sw_read(st->dV, r0, warp * BW + cc)); d1 += b2f(sw_read(st->dV, r0, warp * BW + cc + 1)); }
                        if (tr1 < T) { d2 += b2f(sw_read(st->dV, r1, warp * BW + cc)); d3 += b2f(sw_read(st->dV, r1, warp * BW + cc + 1)); }
                        bf16 b0 = f2b(d0), b1 = f2b(d1), b2 = f2b(d2), b3 = f2b(d3);
                        sdu[r0][cc] = b0; sdu[r0][cc + 1] = b1;
                        sdu[r1][cc] = b2; sdu[r1][cc + 1] = b3;
#if defined(B1T_TMADV)
                        // P7-33: stage into the per-warp [BT,BW] tile; lane 0 issues the
                        // TMA store after the loop.  Rows with tr>=T are written too but
                        // TMA clips them against the tensor bound (dim1 == T).
                        sh->duo[i & 1][warp][r0][cc] = b0; sh->duo[i & 1][warp][r0][cc + 1] = b1;
                        sh->duo[i & 1][warp][r1][cc] = b2; sh->duo[i & 1][warp][r1][cc + 1] = b3;
#elif !defined(B1T_NODV2)
                        if (tr0 < T) {
                            int64_t o0 = ((int64_t)(n * T + tr0) * H + h) * VD + vbase + cc;
                            dv2_g[o0] = b0; dv2_g[o0 + 1] = b1;
                        }
                        if (tr1 < T) {
                            int64_t o1 = ((int64_t)(n * T + tr1) * H + h) * VD + vbase + cc;
                            dv2_g[o1] = b2; dv2_g[o1 + 1] = b3;
                        }
#endif
                    }
            }
            __syncwarp();
#ifdef B1T_TMADV
            if (l == 0) {
                tma4d_store(&dv2map, &sh->duo[i & 1][warp][0][0], vbase, c * BT, h, n);
                tma_store_commit();
            }
#endif

            // ---- op2/op3: S = S*e + scale*(q^T @ do) - (w^T @ du) ----
#ifdef B1T_ESYNC
            __syncwarp();
#endif
            #pragma unroll
            for (int m = 0; m < MTK; ++m) {
#if defined(B1T_ETMA) && !defined(B1T_ETMA_DROPREAD)
#ifdef B1T_EVOL
                const volatile float* ep = st->e;
                float e0 = ep[m * 16 + g];
                float e1 = ep[m * 16 + g + 8];
#elif defined(B1T_ECOPY)
                float e0 = e_copy[m * 16 + g];
                float e1 = e_copy[m * 16 + g + 8];
#else
                float e0 = st->e[m * 16 + g];
                float e1 = st->e[m * 16 + g + 8];
#endif
#else
                float e0 = etot[((int64_t)(n * NT + c) * H + h) * KD + m * 16 + g];
                float e1 = etot[((int64_t)(n * NT + c) * H + h) * KD + m * 16 + g + 8];
#endif
                #pragma unroll
                for (int kk = 0; kk < KT2; ++kk) {
                    uint32_t aq[4], aw[4];
                    ldsm_sw_x4_t<BT>(aq, st->q, kk * 16, m * 16, l);
                    ldsm_sw_x4_t<BT>(aw, st->w, kk * 16, m * 16, l);
                    #pragma unroll
                    for (int x = 0; x < 4; ++x) aw[x] = aw[x] ^ 0x80008000u;  // -w^T
                    #pragma unroll
                    for (int nn = 0; nn < 2; ++nn) {
                        if (kk == 0) {
                            S[m][nn][0] *= e0; S[m][nn][1] *= e0;
                            S[m][nn][2] *= e1; S[m][nn][3] *= e1;
                        }
                        uint32_t bd[2], bu[2];
                        ldsm_sw1_x2_t(bd, st->dO, kk * 16, warp * BW + nn * 8, l);
                        union U { uint32_t u; __nv_bfloat162 b2; } x0, x1;
                        x0.u = bd[0]; x0.b2 = __hmul2(x0.b2, scale2); bd[0] = x0.u;
                        x1.u = bd[1]; x1.b2 = __hmul2(x1.b2, scale2); bd[1] = x1.u;
                        mma16816(S[m][nn], aq, bd, S[m][nn]);
                        ldsm_x2_t_pad(bu, &sdu[0][0], kk * 16, nn * 8, BWS, l);
                        mma16816(S[m][nn], aw, bu, S[m][nn]);
                    }
                }
            }
            __syncwarp();
#if defined(B1T_EPROXY) || defined(B1T_EPROXY_CON)
            fence_proxy_async();
#endif
            if (l == 0) mbar_arrive(&sh->empty[s]);   // one arrival per consumer warp
        }

        // final: dh0 = S
        {
            const int64_t base = ((int64_t)nh) * KD * VD;
            #pragma unroll
            for (int m = 0; m < MTK; ++m)
                #pragma unroll
                for (int nn = 0; nn < 2; ++nn) {
                    int r0 = m * 16 + g, r1 = r0 + 8;
                    int gc = v0 + warp * BW + nn * 8 + t * 2;
                    dh0_g[base + (int64_t)r0 * VD + gc] = S[m][nn][0];
                    dh0_g[base + (int64_t)r0 * VD + gc + 1] = S[m][nn][1];
                    dh0_g[base + (int64_t)r1 * VD + gc] = S[m][nn][2];
                    dh0_g[base + (int64_t)r1 * VD + gc + 1] = S[m][nn][3];
                }
        }
    }
}

// ------------------------------------------------------------------- tensor maps
// Operands are laid out [B, T, H, D] (D = 128, contiguous).  A map exposes
// dims (fastest->slowest) {64, 2, H, T, B} with a box {64,1,1,BT,1} and 128 B
// swizzle: box (dim0,dim1) = the two 64-column (128 B) swizzle atoms of a row.
static void make_map(CUtensorMap* m, void* ptr, int H, int T, int B, int BT) {
    cuuint64_t gdim[5] = {64, 2, (cuuint64_t)H, (cuuint64_t)T, (cuuint64_t)B};
    cuuint64_t gstr[4] = {128, 256, (cuuint64_t)H * 256, (cuuint64_t)T * H * 256};
    cuuint32_t box[5] = {64, 1, 1, (cuuint32_t)BT, 1};
    cuuint32_t estr[5] = {1, 1, 1, 1, 1};
    if (cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, ptr, gdim, gstr, box,
                               estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
#ifdef B1T_NOSW
                               CU_TENSOR_MAP_SWIZZLE_NONE,
#else
                               CU_TENSOR_MAP_SWIZZLE_128B,
#endif
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        fprintf(stderr, "cuTensorMapEncodeTiled failed\n");
}

#ifdef B1T_TMADH
// dh is [B, NT, H, KD, VD] bf16 (contiguous).  dims {VD, KD, H, NT, B}, box {BW,KD,1,1,1},
// SWIZZLE_NONE (inner dim is only BW=16 bf16 = 32 B); one store per compute warp.
static void make_map_dh(CUtensorMap* m, void* ptr, int H, int NT, int B) {
    cuuint64_t gdim[5] = {(cuuint64_t)VD, (cuuint64_t)KD, (cuuint64_t)H, (cuuint64_t)NT,
                          (cuuint64_t)B};
    cuuint64_t gstr[4] = {(cuuint64_t)VD * 2, (cuuint64_t)KD * VD * 2,
                          (cuuint64_t)H * KD * VD * 2, (cuuint64_t)NT * H * KD * VD * 2};
    cuuint32_t box[5] = {(cuuint32_t)BW, (cuuint32_t)KD, 1, 1, 1};
    cuuint32_t estr[5] = {1, 1, 1, 1, 1};
    if (cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, ptr, gdim, gstr, box,
                               estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        fprintf(stderr, "cuTensorMapEncodeTiled(dh) failed\n");
}
#endif

#ifdef B1T_TMADV
// dv2 is [B, T, H, VD] bf16 (token-major, contiguous).  dims {VD, T, H, B},
// box {BW, BT, 1, 1}, SWIZZLE_NONE (inner dim BW*2 = 32 B); one store per compute warp,
// OOB token rows clipped by the T bound.
static void make_map_dv2(CUtensorMap* m, void* ptr, int H, int T, int B, int BT) {
    cuuint64_t gdim[4] = {(cuuint64_t)VD, (cuuint64_t)T, (cuuint64_t)H, (cuuint64_t)B};
    // gstr[i] is the byte stride of dim i+1: {H: VD*2, T: H*VD*2, B: T*H*VD*2}.
    cuuint64_t gstr[3] = {(cuuint64_t)H * VD * 2, (cuuint64_t)VD * 2,
                          (cuuint64_t)T * H * VD * 2};
    cuuint32_t box[4] = {(cuuint32_t)BW, (cuuint32_t)BT, 1, 1};
    cuuint32_t estr[4] = {1, 1, 1, 1};
    if (cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, ptr, gdim, gstr, box,
                               estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        fprintf(stderr, "cuTensorMapEncodeTiled(dv2) failed\n");
}
#endif

#ifdef B1T_ETMA
// E^tot is [B, NT, H, KD] fp32 (contiguous).  dims {KD, NT*H, B}, box {KD,1,1}, no swizzle.
static void make_map_e(CUtensorMap* m, void* ptr, int H, int NT, int B) {
    cuuint64_t gdim[3] = {(cuuint64_t)KD, (cuuint64_t)NT * H, (cuuint64_t)B};
    cuuint64_t gstr[2] = {(cuuint64_t)KD * 4, (cuuint64_t)NT * H * KD * 4};
    cuuint32_t box[3] = {(cuuint32_t)KD, 1, 1};
    cuuint32_t estr[3] = {1, 1, 1};
    if (cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, ptr, gdim, gstr, box,
                               estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        fprintf(stderr, "cuTensorMapEncodeTiled(e) failed\n");
}
#endif

// ------------------------------------------------------------------------- launch
template <int BT, int NSTAGE>
static int launch_impl(void* q, void* k, void* w, void* do_, void* dv, void* etot, void* dh,
                       void* dh0, void* dv2, int B, int T, int H, int NT, float scale,
                       cudaStream_t s) {
    CUtensorMap qm, km, wm, dom, dvm;
    make_map(&qm, q, H, T, B, BT);
    make_map(&km, k, H, T, B, BT);
    make_map(&wm, w, H, T, B, BT);
    make_map(&dom, do_, H, T, B, BT);
    make_map(&dvm, dv, H, T, B, BT);
#ifdef B1T_ETMA
    CUtensorMap em;
    make_map_e(&em, etot, H, NT, B);
#endif
#ifdef B1T_TMADH
    CUtensorMap dhm;
    make_map_dh(&dhm, dh, H, NT, B);
#endif
#ifdef B1T_TMADV
    CUtensorMap dv2m;
    make_map_dv2(&dv2m, dv2, H, T, B, BT);
#endif
    dim3 grid(VD / BV, B * H);
    int smem = (int)sizeof(Shared<BT, NSTAGE>);
    static int configured = 0;
    if (!configured) {
        if (cudaFuncSetAttribute(b1_tma_kernel<BT, NSTAGE>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return -3;
        configured = 1;
    }
    b1_tma_kernel<BT, NSTAGE><<<grid, (NW + 1) * 32, smem, s>>>(
        qm, km, wm, dom, dvm,
#ifdef B1T_ETMA
        em,
#endif
#ifdef B1T_TMADH
        dhm,
#endif
#ifdef B1T_TMADV
        dv2m,
#endif
        (const float*)etot, (bf16*)dh, (float*)dh0, (bf16*)dv2, T, NT, H,
        scale);
    return (int)cudaGetLastError();
}

extern "C" int b1t_launch(int BT, int NSTAGE, void* q, void* k, void* w, void* do_, void* dv,
                          void* etot, void* dh, void* dh0, void* dv2, int B, int T, int H, int V,
                          int NT, float scale, void* stream) {
    if (V % BV != 0) return -2;
    cudaStream_t s = (cudaStream_t)stream;
    if (BT == 64) {
        if (NSTAGE == 1) return launch_impl<64, 1>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, NT, scale, s);
        if (NSTAGE == 2) return launch_impl<64, 2>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, NT, scale, s);
    } else if (BT == 32) {
        if (NSTAGE == 1) return launch_impl<32, 1>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, NT, scale, s);
        if (NSTAGE == 2) return launch_impl<32, 2>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, NT, scale, s);
        if (NSTAGE == 3) return launch_impl<32, 3>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, NT, scale, s);
    } else if (BT == 16) {
        if (NSTAGE == 1) return launch_impl<16, 1>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, NT, scale, s);
        if (NSTAGE == 2) return launch_impl<16, 2>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, NT, scale, s);
        if (NSTAGE == 3) return launch_impl<16, 3>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, NT, scale, s);
    }
    return -1;
}

extern "C" int b1t_smem_bytes(int BT, int NSTAGE) {
    if (BT == 64) { if (NSTAGE == 1) return sizeof(Shared<64,1>); if (NSTAGE == 2) return sizeof(Shared<64,2>); }
    if (BT == 32) { if (NSTAGE == 1) return sizeof(Shared<32,1>); if (NSTAGE == 2) return sizeof(Shared<32,2>); if (NSTAGE == 3) return sizeof(Shared<32,3>); }
    if (BT == 16) { if (NSTAGE == 1) return sizeof(Shared<16,1>); if (NSTAGE == 2) return sizeof(Shared<16,2>); if (NSTAGE == 3) return sizeof(Shared<16,3>); }
    return -1;
}
