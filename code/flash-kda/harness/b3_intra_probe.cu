// P7-62: B3 `intra` in the **K2 shape** -- the last remaining structural item named in
// ROADMAP / docs/07 §60.5 ("另一条「B3 intra 折进 K2 形状」仍待").
//
// Why: at the deployed CHUNK=16 granularity B3's grid is `(NK*NC, NT, B*H)` with
// `BC=BT=16 -> NC=1`, so every program skips Parts A/C (off-diagonal, the only `tl.dot`s)
// and runs **only** Parts B/D -- the diagonal sub-block loops (docs/07 §27.1).  §27 showed
// the diagonal is algebraically two `tl.dot`s (FLA's SAFE_GATE branch already does this:
// chunk_intra.py:520-526/620-637), and P7-24 MMA-ized it in Triton (kernel 1.5-2.0x).  §30
// fused B2+B3 in Triton.  The **CUDA** production B2 (`b2_mma_probe.cu`, K2 shape) never got
// the intra fold -- this probe makes it a K2-shape kernel.
//
// K2 shape = grid `(NT, B*H)`, one block per chunk, 8 warps, warp `w` owns the output k
// columns [w*16, w*16+16) -- exactly the B2 core's warp/accumulator sharding.  Per chunk we
// stage `q/k/g` [BT,KD] + `beta` [BT] + the two diagonal `[BT,BT]` attention-adjoint tiles
// (`dAqk` from `dAv`, `dAkk` = B2's sandwiched `dA`) into smem, then compute (NC=1 diag):
//
//   Mq = tril(dAqk), Mk = tril(dAkk), Sq = triu(dAqk^T), Sk = triu(dAkk^T)   [BT,BT]
//   KG = k*2^{-g}, QG = q*2^{g}, KGd = k*beta*2^{g}                          [BT,KD]
//   dq2 = 2^{g} ⊙ (Mq @ KG)                       (Part B, query branch)
//   dk2 = 2^{g} ⊙ (Mk @ KG)                       (Part B, key branch)
//   db += sum_k dk2*k ;  dk2 *= beta               (write-gate adjoint)
//   dg  = q*dq2                                    (decay chain: \bar g += \bar q^dec ⊙ q)
//   dkt = 2^{-g} ⊙ (Sq @ QG + Sk @ KGd)            (Part D)
//   dg += (dk2 - dkt)*k ;  dk2 += dkt
//
// The four [BT,BT]@[BT,KD] matmuls are `mma.sync.m16n8k8.tf32` (matches `tl.dot`'s default
// tf32); the gating is elementwise in the accumulator fragment layout.  `db` is a reduction
// over the warp's 16 owned k columns, then across the 8 warps (like the P7-56 `db` segment).
//
// This is a **standalone** K2-shape intra (its own inputs/outputs, no dependence on the B2
// core's other segments) so it is bit-checkable against the Triton MMA B3 (`check_bwd_b3_mma.py
// ::build_intra_mma`) on the *same* `dAqk`/`dAkk`.  Folding it into the production B2 launch
// (shared smem/signature, adding into the gate's final dq/dk/dg) is the follow-up step.
//
// Build (inside kernel_lab):
//   nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared b3_intra_probe.cu -o /tmp/b3_intra.so
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define DEV __device__ __forceinline__

constexpr int KD = 128;   // K = D
constexpr int BT = 16;    // chunk rows
constexpr int NW = 8;     // warps
constexpr int BW = 16;    // output columns per warp = KD / NW

DEV uint32_t to_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}
DEV void mma8(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

struct alignas(128) Shared {
    float sq [BT][KD];    // q
    float sk [BT][KD];    // k
    float sg [BT][KD];    // g (log2 cumulative)
    float sbeta[BT];      // beta
    float sAqk[BT][BT];   // dAqk [query_local][key_local]
    float sAkk[BT][BT];   // dAkk [query_local][key_local]
    float dbrow[NW][BT];  // per-warp db partial (per row)
};

__global__ void __launch_bounds__(NW * 32)
b3intra_kernel(const float* __restrict__ q, const float* __restrict__ k,
               const float* __restrict__ g, const float* __restrict__ beta,
               const float* __restrict__ dAqk, const float* __restrict__ dAkk,
               float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ db,
               float* __restrict__ dg, int B, int T, int H, int NT) {
    extern __shared__ char smem_raw[];
    Shared* sh = reinterpret_cast<Shared*>(smem_raw);
    const int i_t = blockIdx.x, i_bh = blockIdx.y;
    const int n = i_bh / H, hh = i_bh % H;
    const int l = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int grp = l >> 2, t = l & 3;   // mma groupID / threadID-in-group
    const int t0 = i_t * BT;

    // ---- stage q/k/g/beta/dAqk/dAkk (rows t<T zeroed) ----
    for (int idx = threadIdx.x; idx < BT * KD; idx += NW * 32) {
        const int row = idx / KD, col = idx % KD, tt = t0 + row;
        float qv = 0.f, kv = 0.f, gv = 0.f;
        if (tt < T) {
            const int64_t off = ((int64_t)(n * T + tt) * H + hh) * KD + col;
            qv = q[off]; kv = k[off]; gv = g[off];
        }
        sh->sq[row][col] = qv; sh->sk[row][col] = kv; sh->sg[row][col] = gv;
    }
    if (threadIdx.x < BT) {
        const int r = threadIdx.x;
        sh->sbeta[r] = (t0 + r < T) ? beta[((int64_t)(n * T + t0 + r) * H + hh)] : 0.f;
    }
    for (int idx = threadIdx.x; idx < BT * BT; idx += NW * 32) {
        const int a = idx / BT, j = idx % BT;
        const bool v = (t0 + a < T) && (t0 + j < T);
        const int64_t off = ((int64_t)(n * T + t0 + a) * H + hh) * BT + j;
        sh->sAqk[a][j] = v ? dAqk[off] : 0.f;
        sh->sAkk[a][j] = v ? dAkk[off] : 0.f;
    }
    __syncthreads();

    // ---- four [BT,BT]@[BT,KD-slice] matmuls + gating (warp owns 16 k columns) ----
    float adq[2][4], adk[2][4], adt[2][4];   // dq2, dk2, dkt fragments (2 x n8)
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int i = 0; i < 4; ++i) { adq[nt][i] = 0.f; adk[nt][i] = 0.f; adt[nt][i] = 0.f; }

    #pragma unroll
    for (int kk = 0; kk < BT / 8; ++kk) {       // contraction = chunk rows j (k8 steps)
        const int kc = kk * 8;
        const int j0 = kc + t, j1 = kc + t + 4;
        // A fragments: Mq[grp/grp+8][j] and Mk (row-major, direct)
        uint32_t mq[4] = {to_tf32(sh->sAqk[grp][j0]), to_tf32(sh->sAqk[grp + 8][j0]),
                          to_tf32(sh->sAqk[grp][j1]), to_tf32(sh->sAqk[grp + 8][j1])};
        uint32_t mk[4] = {to_tf32(sh->sAkk[grp][j0]), to_tf32(sh->sAkk[grp + 8][j0]),
                          to_tf32(sh->sAkk[grp][j1]), to_tf32(sh->sAkk[grp + 8][j1])};
        // Sq[grp/grp+8][j] = dAqk[j][grp/grp+8] (transposed); Sk likewise.
        uint32_t sq_[4] = {to_tf32(sh->sAqk[j0][grp]),  to_tf32(sh->sAqk[j0][grp + 8]),
                           to_tf32(sh->sAqk[j1][grp]),  to_tf32(sh->sAqk[j1][grp + 8])};
        uint32_t sk_[4] = {to_tf32(sh->sAkk[j0][grp]),  to_tf32(sh->sAkk[j0][grp + 8]),
                           to_tf32(sh->sAkk[j1][grp]),  to_tf32(sh->sAkk[j1][grp + 8])};
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = warp * BW + nt * 8 + grp;   // B-operand N column
            const float kg0 = sh->sk[j0][col] * exp2f(-sh->sg[j0][col]);
            const float kg1 = sh->sk[j1][col] * exp2f(-sh->sg[j1][col]);
            const float qg0 = sh->sq[j0][col] * exp2f(sh->sg[j0][col]);
            const float qg1 = sh->sq[j1][col] * exp2f(sh->sg[j1][col]);
            const float kd0 = sh->sk[j0][col] * sh->sbeta[j0] * exp2f(sh->sg[j0][col]);
            const float kd1 = sh->sk[j1][col] * sh->sbeta[j1] * exp2f(sh->sg[j1][col]);
            uint32_t b_kg[2]  = {to_tf32(kg0), to_tf32(kg1)};
            uint32_t b_qg[2]  = {to_tf32(qg0), to_tf32(qg1)};
            uint32_t b_kgd[2] = {to_tf32(kd0), to_tf32(kd1)};
            mma8(adq[nt], mq, b_kg, adq[nt]);
            mma8(adk[nt], mk, b_kg, adk[nt]);
            mma8(adt[nt], sq_, b_qg, adt[nt]);
            mma8(adt[nt], sk_, b_kgd, adt[nt]);
        }
    }

    // ---- gating + outputs.  Accumulator fragment: rows grp/grp+8, cols nt*8+2t+(i&1). ----
    float dkpre[2][4];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int rowrel = (i < 2) ? grp : grp + 8;
            const int col = warp * BW + nt * 8 + 2 * t + (i & 1);
            const float gg = sh->sg[rowrel][col];
            adq[nt][i] *= exp2f(gg);                 // dq2 = 2^g ⊙ (Mq@KG)
            adk[nt][i] *= exp2f(gg);                 // dk2 = 2^g ⊙ (Mk@KG)
            adt[nt][i] *= exp2f(-gg);                // dkt = 2^-g ⊙ (Sq@QG + Sk@KGd)
            dkpre[nt][i] = adk[nt][i];               // pre-beta (for db)
            adk[nt][i] *= sh->sbeta[rowrel];          // dk2 *= beta
        }
    }
    float dgv[2][4];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int rowrel = (i < 2) ? grp : grp + 8;
            const int col = warp * BW + nt * 8 + 2 * t + (i & 1);
            const float kv = sh->sk[rowrel][col];
            dgv[nt][i] = sh->sq[rowrel][col] * adq[nt][i] + (adk[nt][i] - adt[nt][i]) * kv;
            adk[nt][i] += adt[nt][i];                 // dk2 += dkt
        }
    }

    // ---- store warp-owned columns ----
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int cl = nt * 8 + 2 * t;
        const int c0 = warp * BW + cl, c1 = c0 + 1;
        const int tr0 = t0 + grp, tr1 = t0 + grp + 8;
        if (tr0 < T) {
            const int64_t o0 = ((int64_t)(n * T + tr0) * H + hh) * KD;
            dq[o0 + c0] = adq[nt][0]; dq[o0 + c1] = adq[nt][1];
            dk[o0 + c0] = adk[nt][0]; dk[o0 + c1] = adk[nt][1];
            dg[o0 + c0] = dgv[nt][0]; dg[o0 + c1] = dgv[nt][1];
        }
        if (tr1 < T) {
            const int64_t o1 = ((int64_t)(n * T + tr1) * H + hh) * KD;
            dq[o1 + c0] = adq[nt][2]; dq[o1 + c1] = adq[nt][3];
            dk[o1 + c0] = adk[nt][2]; dk[o1 + c1] = adk[nt][3];
            dg[o1 + c0] = dgv[nt][2]; dg[o1 + c1] = dgv[nt][3];
        }
    }

    // ---- db = sum_k dk_prebeta * k : per-warp column sum, then cross-warp reduction ----
    {
        float pr0 = 0.f, pr1 = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int rowrel = (i < 2) ? grp : grp + 8;
                const int col = warp * BW + nt * 8 + 2 * t + (i & 1);
                const float p = dkpre[nt][i] * sh->sk[rowrel][col];
                if (i < 2) pr0 += p; else pr1 += p;
            }
        }
        // reduce over the 4 threadIDs within the group (cols 2t/2t+1 handled per thread)
        pr0 += __shfl_xor_sync(0xffffffffu, pr0, 1);
        pr0 += __shfl_xor_sync(0xffffffffu, pr0, 2);
        pr1 += __shfl_xor_sync(0xffffffffu, pr1, 1);
        pr1 += __shfl_xor_sync(0xffffffffu, pr1, 2);
        if (t == 0) { sh->dbrow[warp][grp] = pr0; sh->dbrow[warp][grp + 8] = pr1; }
    }
    __syncthreads();
    if (threadIdx.x < BT) {
        const int r = threadIdx.x;
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; ++w) s += sh->dbrow[w][r];
        if (t0 + r < T) db[((int64_t)(n * T + t0 + r) * H + hh)] = s;
    }
}

static int b3intra_impl(const void* q, const void* k, const void* g, const void* beta,
                        const void* dAqk, const void* dAkk,
                        void* dq, void* dk, void* db, void* dg,
                        int B, int T, int H, int NT, void* stream) {
    dim3 grid(NT, B * H);
    int smem = (int)sizeof(Shared);
    static int configured = 0;
    if (!configured) {
        if (cudaFuncSetAttribute(b3intra_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem) != cudaSuccess)
            return -3;
        configured = 1;
    }
    b3intra_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)q, (const float*)k, (const float*)g, (const float*)beta,
        (const float*)dAqk, (const float*)dAkk,
        (float*)dq, (float*)dk, (float*)db, (float*)dg, B, T, H, NT);
    return (int)cudaGetLastError();
}

extern "C" int b3intra_launch(const void* q, const void* k, const void* g, const void* beta,
                              const void* dAqk, const void* dAkk,
                              void* dq, void* dk, void* db, void* dg,
                              int B, int T, int H, int NT, void* stream) {
    return b3intra_impl(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, B, T, H, NT, stream);
}

extern "C" int b3intra_smem_bytes() { return (int)sizeof(Shared); }
