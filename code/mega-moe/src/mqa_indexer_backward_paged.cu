// P148-1 实验：分页 KV 的 MQA logits（lightning indexer）反向 —— CUDA 后端。
//
// 背景：DeepGEMM 的 MQA logits 有**连续版**（`fp8_fp4_mqa_logits`）与**分页版**
// （`fp8_fp4_paged_mqa_logits`）两条前向；第 157 篇（P147-1）补了连续版反向，本篇补**分页版**。
// 分页版前向（`deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh`，测试参考
// `tests/test_attention.py:225-253` 的 `ref_paged_mqa_logits`）：
//     - KV 存在物理页池 `kv_cache[num_blocks, block_kv, 1, dim]`；
//     - 每个 query 行 r=(seq i, next_n 槽 j) 有自己的 context 长度 cl[r]；
//     - 逻辑位置 s（0..cl[r]) 经页表映射到物理槽 phys = block_table[i, s/block_kv]*block_kv + s%block_kv；
//     - khat = kv_phys * scale；z_{r,h,s}=<q_{r,h}, khat>；I_{r,s}=sum_h w_{r,h} relu(z)。
//
// 分页反向与连续反向唯一的**结构性差异**：梯度必须沿**页表的逆映射散播**回物理 KV——
// `dL/dkv_phys[phys] += ...`，且同一次前向里一个物理槽会被**多行**引用（next_n 前缀重叠、
// 甚至跨序列共享页），所以必须 **scatter-accumulate**（原子加），不能「写」（否则静默丢贡献）。
//
// 布局：grid = P（每 query 行一 block），block = (32 lane, H)。每个 h 是 warp；
// 沿 lane 对 d 跨步点积 + `__shfl_xor` 归约；dq 累寄存器（每 block 独占一行 r）；
// dw 在 lane0 累加；dkv_phys 沿页表逆映射走 global `atomicAdd`。
//
// 全部架构无关（fp32 + shuffle + 原子加），H100(sm90) 真机可跑。
// 编译/运行见 harness/exp_mqa_logits_paged_backward.sh。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// grid = P (num_q_rows), block = (32, H)
// q  [P, H, D], kv [NB*BS, D] (dequantized physical), kvs [NB*BS], w [P, H], dI [P, max_len]
// cl [P], block_table [S, max_blocks]
__global__ void indexer_paged_bwd_kernel(const float* __restrict__ q,
                                         const float* __restrict__ kv,
                                         const float* __restrict__ kvs,
                                         const float* __restrict__ w,
                                         const float* __restrict__ dI,
                                         const int*   __restrict__ cl,
                                         const int*   __restrict__ block_table,
                                         float* __restrict__ dq,
                                         float* __restrict__ dw,
                                         float* __restrict__ dkv,
                                         int P, int S, int N, int H, int D, int BS,
                                         int max_blocks, int max_len) {
    const int r = blockIdx.x;
    const int i = r / N;                 // sequence index
    const int h = threadIdx.y;
    const int lane = threadIdx.x;

    const int c = cl[r];                 // this row's context length
    const int qbase = (r * H + h) * D;
    const float w_rh = w[r * H + h];
    float dw_acc = 0.0f;

    const int* bt = block_table + (size_t)i * max_blocks;

    for (int s = 0; s < c; ++s) {
        // ---- page-table inverse mapping: logical s -> physical slot ----
        const int phys = bt[s / BS] * BS + (s % BS);
        const float scale = kvs[phys];
        const float* krow = kv + (size_t)phys * D;
        float z = 0.0f;
        for (int d = lane; d < D; d += 32)
            z += q[qbase + d] * (krow[d] * scale);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            z += __shfl_xor_sync(0xffffffffu, z, off);

        const float di = dI[(size_t)r * max_len + s];
        const float rm = (z > 0.0f) ? 1.0f : 0.0f;
        dw_acc += di * (z * rm);
        const float coeff = di * w_rh * rm;      // dL/dz_{r,s,h}

        if (coeff != 0.0f) {
            for (int d = lane; d < D; d += 32) {
                const float qd = q[qbase + d];
                dq[qbase + d] += coeff * krow[d] * scale;
                // scatter-accumulate into the shared physical slot (aliasing-safe)
                atomicAdd(&dkv[(size_t)phys * D + d], coeff * qd * scale);
            }
        }
    }
    if (lane == 0) dw[r * H + h] = dw_acc;
}

// ---------------------------------------------------------------------------
// host 参考（fp64）：逐元素，含页表逆映射 + 别名累加，用于正确性对拍
// ---------------------------------------------------------------------------
static void ref_backward(const std::vector<double>& q, const std::vector<double>& kv,
                         const std::vector<double>& kvs, const std::vector<double>& w,
                         const std::vector<double>& dI, const std::vector<int>& cl,
                         const std::vector<int>& block_table, int P, int S, int N, int H, int D,
                         int BS, int max_blocks, int max_len,
                         std::vector<double>& dq, std::vector<double>& dw, std::vector<double>& dkv) {
    const int NB = (int)kv.size() / D;
    dq.assign((size_t)P * H * D, 0.0);
    dw.assign((size_t)P * H, 0.0);
    dkv.assign((size_t)NB * D, 0.0);
    for (int r = 0; r < P; ++r) {
        const int i = r / N;
        for (int h = 0; h < H; ++h) {
            const int qbase = (r * H + h) * D;
            for (int s = 0; s < cl[r]; ++s) {
                const int phys = block_table[(size_t)i * max_blocks + s / BS] * BS + s % BS;
                double z = 0.0;
                for (int d = 0; d < D; ++d)
                    z += q[qbase + d] * kv[(size_t)phys * D + d] * kvs[phys];
                const double rm = z > 0 ? 1.0 : 0.0;
                const double di = dI[(size_t)r * max_len + s];
                dw[r * H + h] += di * z * rm;
                const double dz = di * w[r * H + h] * rm;
                for (int d = 0; d < D; ++d) {
                    dq[qbase + d] += dz * kv[(size_t)phys * D + d] * kvs[phys];
                    dkv[(size_t)phys * D + d] += dz * q[qbase + d] * kvs[phys];
                }
            }
        }
    }
}

static double relerr(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += std::fabs((double)a[i] - b[i]); den += std::fabs((double)a[i]) + std::fabs(b[i]); }
    return num / (den + 1e-30);
}

static float* dvec(const std::vector<float>& v) {
    float* p; CK(cudaMalloc(&p, v.size() * 4)); CK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return p;
}

int main() {
    // ---------------- [A] 正确性（含页共享/前缀重叠的别名） ----------------
    printf("== [A] 正确性：分页反向 CUDA vs host fp64 参考（含别名累加）==\n");
    int S = 6, N = 3, H = 8, D = 64, BS = 16, max_len = 256;
    int P = S * N;
    int blocks_per_seq = (max_len + BS - 1) / BS;
    int NB = S * blocks_per_seq;          // number of physical blocks
    int NSLOT = NB * BS;                  // number of physical KV slots (= pages * block_kv)
    std::mt19937 rng(2024);
    std::normal_distribution<float> nd(0.f, 0.3f);
    std::uniform_real_distribution<float> ud(0.5f, 1.2f);

    std::vector<float> q((size_t)P * H * D), kv((size_t)NSLOT * D), kvs(NSLOT), w((size_t)P * H), dI((size_t)P * max_len);
    for (auto& x : q) x = nd(rng);
    for (auto& x : kv) x = nd(rng);
    for (auto& x : kvs) x = ud(rng);
    for (auto& x : w) x = nd(rng);
    for (auto& x : dI) x = nd(rng);

    std::vector<int> cl(P), block_table((size_t)S * blocks_per_seq);
    // distinct physical blocks per sequence, but every row (next_n slot) reads the same prefix pages
    for (int i = 0; i < S; ++i)
        for (int b = 0; b < blocks_per_seq; ++b)
            block_table[(size_t)i * blocks_per_seq + b] = i * blocks_per_seq + b;
    for (int r = 0; r < P; ++r) cl[r] = max_len / 3 + (rng() % (max_len / 2));  // ragged context lens

    std::vector<double> qd(q.begin(), q.end()), kvd(kv.begin(), kv.end()), kvsd(kvs.begin(), kvs.end()),
                        wd(w.begin(), w.end()), did(dI.begin(), dI.end());
    std::vector<double> rdq, rdw, rdkv;
    ref_backward(qd, kvd, kvsd, wd, did, cl, block_table, P, S, N, H, D, BS, blocks_per_seq, max_len, rdq, rdw, rdkv);

    float *dq, *dw, *dkv;
    CK(cudaMalloc(&dq, (size_t)P * H * D * 4));
    CK(cudaMalloc(&dw, (size_t)P * H * 4));
    CK(cudaMalloc(&dkv, (size_t)NSLOT * D * 4));
    CK(cudaMemset(dq, 0, (size_t)P * H * D * 4));
    float *pq = dvec(q), *pkv = dvec(kv), *pkvs = dvec(kvs), *pw = dvec(w), *pdI = dvec(dI);
    int *pcl, *pbt;
    CK(cudaMalloc(&pcl, P * 4));       CK(cudaMemcpy(pcl, cl.data(), P * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&pbt, block_table.size() * 4)); CK(cudaMemcpy(pbt, block_table.data(), block_table.size() * 4, cudaMemcpyHostToDevice));

    dim3 block(32, H), grid(P);
    CK(cudaMemset(dw, 0, (size_t)P * H * 4));
    CK(cudaMemset(dkv, 0, (size_t)NSLOT * D * 4));
    indexer_paged_bwd_kernel<<<grid, block>>>(pq, pkv, pkvs, pw, pdI, pcl, pbt, dq, dw, dkv,
                                              P, S, N, H, D, BS, blocks_per_seq, max_len);
    CK(cudaDeviceSynchronize());

    std::vector<float> hdq((size_t)P * H * D), hdw((size_t)P * H), hdkv((size_t)NSLOT * D);
    CK(cudaMemcpy(hdq.data(), dq, hdq.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdw.data(), dw, hdw.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdkv.data(), dkv, hdkv.size() * 4, cudaMemcpyDeviceToHost));
    printf("  dq        relerr = %.3e\n", relerr(hdq, rdq));
    printf("  dw        relerr = %.3e\n", relerr(hdw, rdw));
    printf("  dkv_phys  relerr = %.3e\n", relerr(hdkv, rdkv));

    // 对照：不做别名累加（写而非原子加，last-writer-wins）会丢多少
    std::vector<double> rdkv_naive((size_t)NSLOT * D, 0.0);
    {
        std::vector<float> zero((size_t)NSLOT * D, 0.0f); (void)zero;
        // recompute the naive (overwrite) backward analytically
        for (int r = 0; r < P; ++r) {
            const int i = r / N;
            for (int h = 0; h < H; ++h) {
                const int qbase = (r * H + h) * D;
                for (int s = 0; s < cl[r]; ++s) {
                    const int phys = block_table[(size_t)i * blocks_per_seq + s / BS] * BS + s % BS;
                    double z = 0.0;
                    for (int d = 0; d < D; ++d) z += qd[qbase + d] * kvd[(size_t)phys * D + d] * kvsd[phys];
                    const double rm = z > 0 ? 1.0 : 0.0;
                    const double dz = did[(size_t)r * max_len + s] * wd[r * H + h] * rm;
                    for (int d = 0; d < D; ++d)
                        rdkv_naive[(size_t)phys * D + d] = dz * qd[qbase + d] * kvsd[phys];  // overwrite (bug)
                }
            }
        }
    }
    double naive_err = relerr(hdkv, rdkv_naive);
    printf("  (对照) 若用「写」而非 scatter-accumulate，dkv_phys 与正确值 relerr = %.3e\n", naive_err);

    bool ok = relerr(hdq, rdq) < 1e-5 && relerr(hdw, rdw) < 1e-5 && relerr(hdkv, rdkv) < 1e-5;
    printf("  => [A] %s\n\n", ok ? "ALL PASS" : "FAIL");

    // ---------------- [B] 吞吐 ----------------
    printf("== [B] 吞吐（分页 gather + scatter 原子归约，受争用下界）==\n");
    struct Case { int S, N, H, D, BS, avg_len; const char* name; };
    Case cases[] = {
        {64,  1, 32, 128, 64, 8192,  "S=64,N=1,H=32,BS=64,avg=8192"},
        {64,  4, 32, 128, 64, 8192,  "S=64,N=4,H=32,BS=64,avg=8192"},
        {256, 1, 32, 128, 64, 8192,  "S=256,N=1,H=32,BS=64,avg=8192"},
    };
    for (auto& c : cases) {
        int Pc = c.S * c.N;
        int ml = c.avg_len + 64;
        int bps = (ml + c.BS - 1) / c.BS;
        long NBC = (long)c.S * bps;
        float* Q; CK(cudaMalloc(&Q, (size_t)Pc * c.H * c.D * 4));
        float* K; CK(cudaMalloc(&K, (size_t)NBC * c.BS * c.D * 4));
        float* Sf; CK(cudaMalloc(&Sf, (size_t)NBC * c.BS * 4));
        float* W; CK(cudaMalloc(&W, (size_t)Pc * c.H * 4));
        float* Dg; CK(cudaMalloc(&Dg, (size_t)Pc * ml * 4));
        float* DQ; CK(cudaMalloc(&DQ, (size_t)Pc * c.H * c.D * 4));
        float* DW; CK(cudaMalloc(&DW, (size_t)Pc * c.H * 4));
        float* DKV; CK(cudaMalloc(&DKV, (size_t)NBC * c.BS * c.D * 4));
        int* CL; CK(cudaMalloc(&CL, Pc * 4));
        int* BT; CK(cudaMalloc(&BT, (size_t)c.S * bps * 4));
        std::vector<int> hcl(Pc, c.avg_len), hbt((size_t)c.S * bps);
        for (int i = 0; i < c.S; ++i) for (int b = 0; b < bps; ++b) hbt[(size_t)i * bps + b] = i * bps + b;
        CK(cudaMemcpy(CL, hcl.data(), Pc * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(BT, hbt.data(), hbt.size() * 4, cudaMemcpyHostToDevice));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        auto run = [&]() {
            CK(cudaMemset(DQ, 0, (size_t)Pc * c.H * c.D * 4));
            indexer_paged_bwd_kernel<<<dim3(Pc), dim3(32, c.H)>>>(Q, K, Sf, W, Dg, CL, BT, DQ, DW, DKV,
                                                                  Pc, c.S, c.N, c.H, c.D, c.BS, bps, ml);
            CK(cudaGetLastError());
        };
        for (int i = 0; i < 3; ++i) run();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        int reps = 10;
        for (int i = 0; i < reps; ++i) run();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); ms /= reps;
        double macs = 4.0 * (double)Pc * c.avg_len * c.H * c.D;
        printf("  %-30s %8.3f ms/call  ~%.1f GFLOP/s\n", c.name, ms, macs / ms / 1e9);
        cudaFree(Q); cudaFree(K); cudaFree(Sf); cudaFree(W); cudaFree(Dg);
        cudaFree(DQ); cudaFree(DW); cudaFree(DKV); cudaFree(CL); cudaFree(BT);
    }
    printf("  注：dkv_phys 的 `atomicAdd` 沿页表逆映射散播，量 O(sum_r cl_r · H · D)；next_n>1 时\n");
    printf("      同一物理槽被多行引用（前缀重叠），别名累加是正确性硬要求而非优化。\n");
    printf("\n== 总判定：%s ==\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
