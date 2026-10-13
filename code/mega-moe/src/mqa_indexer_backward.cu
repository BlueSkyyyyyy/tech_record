// P147-1 实验：MQA logits（lightning indexer）打分算子的反向 —— CUDA 后端。
//
// 背景：DeepGEMM 只实现 MQA logits 的**前向**打分（`fp8_fp4_mqa_logits` 等，第 19/60 篇已拆），
// 但 DeepSeek-V3.2 的 lightning indexer 是要**训练**的（arXiv:2512.02556 §2.1，式 (3)/(4)：
// L = sum_t KL(p_t || softmax(I_t))），所以反向梯度是真实需求、却一直缺席。
//
// 本文件实现「给定上游 dL/dI[T,KV]，求 dL/dq、dL/dw、dL/d(kv_data)、dL/d(kv_scale)」这一层
// （打分算子 backward，softmax/KL 那层只产出 dI、不需要在这里）。前向语义（DeepGEMM README
// `:100-108`、`tests/test_attention.py:78-101`）：
//     khat_s = kv_data_s * kv_scale_s                    // 每 KV token 一个 fp32 scale
//     z_{t,s,h} = <q_{t,h}, khat_s>
//     I_{t,s}   = sum_h w_{t,h} * relu(z_{t,s,h})
// 反向（链式）：
//     dI_{t,s} = dL/dI_{t,s}（输入，无效位置 0）
//     dL/dw_{t,h}       = sum_s dI_{t,s} * relu(z_{t,s,h})
//     dL/dz_{t,s,h}     = dI_{t,s} * w_{t,h} * 1[z>0]
//     dL/dq_{t,h,d}     = sum_s dL/dz_{t,s,h} * khat_{s,d}
//     dL/dkhat_{s,d}    = sum_{t,h} dL/dz_{t,s,h} * q_{t,h,d}
//     dL/dkv_data_{s,d} = dL/dkhat_{s,d} * kv_scale_s
//     dL/dkv_scale_s    = sum_d dL/dkhat_{s,d} * kv_data_{s,d}
//
// 布局：grid = T（每 query token 一 block），block = (32 lane, H)。每个 h 是一个 warp，
// 沿 lane 对 d 做跨步点积 + `__shfl_xor` 归约得 z_{s,h}；dq 累在寄存器（每 block 独占一行 t）、
// dw 在 lane0 累加；dkv_data/dkv_scale 跨 t 汇聚，走 global `atomicAdd`（scatter 量 O(T·KV·H·D)，
// 这是本算子固有的归约代价——indexer 头数 H 很小正因如此）。
//
// 全部架构无关（fp32 浮点 + shuffle + 原子加），H100(sm90) 真机可跑。
// 编译/运行见 harness/exp_mqa_logits_indexer_backward.sh。
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

// grid = T, block = (32, H)
__global__ void indexer_bwd_kernel(const float* __restrict__ q,
                                   const float* __restrict__ kv,
                                   const float* __restrict__ kvs,
                                   const float* __restrict__ w,
                                   const float* __restrict__ dI,
                                   const int*   __restrict__ ks,
                                   const int*   __restrict__ ke,
                                   float* __restrict__ dq,
                                   float* __restrict__ dw,
                                   float* __restrict__ dkv,
                                   float* __restrict__ dkv_s,
                                   int T, int H, int D, int KV) {
    const int t = blockIdx.x;
    const int h = threadIdx.y;
    const int lane = threadIdx.x;

    const int s0 = ks[t], s1 = ke[t];
    const int qbase = (t * H + h) * D;
    const float w_th = w[t * H + h];
    float dw_acc = 0.0f;

    for (int s = s0; s < s1; ++s) {
        const float scale = kvs[s];
        const float* krow = kv + (size_t)s * D;
        float z = 0.0f;
        for (int d = lane; d < D; d += 32)
            z += q[qbase + d] * (krow[d] * scale);
        // warp 内归约
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            z += __shfl_xor_sync(0xffffffffu, z, off);

        const float di = dI[(size_t)t * KV + s];
        const float rm = (z > 0.0f) ? 1.0f : 0.0f;
        dw_acc += di * (z * rm);
        const float coeff = di * w_th * rm;      // dL/dz_{t,s,h}

        if (coeff != 0.0f) {
            for (int d = lane; d < D; d += 32) {
                const float qd = q[qbase + d];
                const float khat = krow[d] * scale;
                dq[qbase + d] += coeff * khat;
                atomicAdd(&dkv[(size_t)s * D + d], coeff * qd * scale);
                atomicAdd(&dkv_s[s], coeff * qd * krow[d]);
            }
        }
    }
    if (lane == 0) dw[t * H + h] = dw_acc;
}

// ---------------------------------------------------------------------------
// host 参考（fp64）：小规模逐元素，用于正确性对拍
// ---------------------------------------------------------------------------
static void ref_backward(const std::vector<double>& q, const std::vector<double>& kv,
                         const std::vector<double>& kvs, const std::vector<double>& w,
                         const std::vector<double>& dI, const std::vector<int>& ks,
                         const std::vector<int>& ke, int T, int H, int D, int KV,
                         std::vector<double>& dq, std::vector<double>& dw,
                         std::vector<double>& dkv, std::vector<double>& dkv_s) {
    dq.assign((size_t)T * H * D, 0.0);
    dw.assign((size_t)T * H, 0.0);
    dkv.assign((size_t)KV * D, 0.0);
    dkv_s.assign((size_t)KV, 0.0);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            int qbase = (t * H + h) * D;
            for (int s = ks[t]; s < ke[t]; ++s) {
                double z = 0.0;
                for (int d = 0; d < D; ++d) z += q[qbase + d] * kv[(size_t)s * D + d] * kvs[s];
                double rm = z > 0 ? 1.0 : 0.0;
                double di = dI[(size_t)t * KV + s];
                dw[t * H + h] += di * z * rm;
                double dz = di * w[t * H + h] * rm;
                for (int d = 0; d < D; ++d) {
                    dq[qbase + d] += dz * kv[(size_t)s * D + d] * kvs[s];
                    dkv[(size_t)s * D + d] += dz * q[qbase + d] * kvs[s];
                    dkv_s[s] += dz * q[qbase + d] * kv[(size_t)s * D + d];
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
    // ---------------- [A] 正确性 ----------------
    printf("== [A] 正确性：CUDA 反向 vs host fp64 参考 ==\n");
    int T = 48, H = 8, D = 128, KV = 256;
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.f, 0.3f);
    std::uniform_real_distribution<float> ud(0.5f, 1.2f);
    std::vector<float> q(T * H * D), kv(KV * D), kvs(KV), w(T * H), dI(T * KV);
    for (auto& x : q) x = nd(rng);
    for (auto& x : kv) x = nd(rng);
    for (auto& x : kvs) x = ud(rng);
    for (auto& x : w) x = nd(rng);
    for (auto& x : dI) x = nd(rng);
    std::vector<int> ks(T, 0), ke(T);
    for (int t = 0; t < T; ++t) ke[t] = KV / 2 + (rng() % (KV / 2 + 1));  // ragged 有效区间

    std::vector<double> qd(q.begin(), q.end()), kvd(kv.begin(), kv.end()), kvsd(kvs.begin(), kvs.end()),
                        wd(w.begin(), w.end()), did(dI.begin(), dI.end());
    std::vector<double> rdq, rdw, rdkv, rdkvs;
    ref_backward(qd, kvd, kvsd, wd, did, ks, ke, T, H, D, KV, rdq, rdw, rdkv, rdkvs);

    float *dq, *dw, *dkv, *dkvs;
    CK(cudaMalloc(&dq, (size_t)T * H * D * 4));
    CK(cudaMalloc(&dw, (size_t)T * H * 4));
    CK(cudaMalloc(&dkv, (size_t)KV * D * 4));
    CK(cudaMalloc(&dkvs, (size_t)KV * 4));
    CK(cudaMemset(dq, 0, (size_t)T * H * D * 4));
    float *pq = dvec(q), *pkv = dvec(kv), *pkvs = dvec(kvs), *pw = dvec(w), *pdI = dvec(dI);
    int *pks, *pke;
    CK(cudaMalloc(&pks, T * 4));
    CK(cudaMalloc(&pke, T * 4));
    CK(cudaMemcpy(pks, ks.data(), T * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(pke, ke.data(), T * 4, cudaMemcpyHostToDevice));

    dim3 block(32, H), grid(T);
    CK(cudaMemset(dw, 0, (size_t)T * H * 4));
    CK(cudaMemset(dkv, 0, (size_t)KV * D * 4));
    CK(cudaMemset(dkvs, 0, (size_t)KV * 4));
    indexer_bwd_kernel<<<grid, block>>>(pq, pkv, pkvs, pw, pdI, pks, pke, dq, dw, dkv, dkvs, T, H, D, KV);
    CK(cudaDeviceSynchronize());

    std::vector<float> hdq(T * H * D), hdw(T * H), hdkv(KV * D), hdkvs(KV);
    CK(cudaMemcpy(hdq.data(), dq, hdq.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdw.data(), dw, hdw.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdkv.data(), dkv, hdkv.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdkvs.data(), dkvs, hdkvs.size() * 4, cudaMemcpyDeviceToHost));
    printf("  dq        relerr = %.3e\n", relerr(hdq, rdq));
    printf("  dw        relerr = %.3e\n", relerr(hdw, rdw));
    printf("  dkv_data  relerr = %.3e\n", relerr(hdkv, rdkv));
    printf("  dkv_scale relerr = %.3e\n", relerr(hdkvs, rdkvs));
    bool ok = relerr(hdq, rdq) < 1e-5 && relerr(hdw, rdw) < 1e-5 &&
              relerr(hdkv, rdkv) < 1e-5 && relerr(hdkvs, rdkvs) < 1e-5;
    printf("  => [A] %s\n\n", ok ? "ALL PASS" : "FAIL");

    // ---------------- [B] 吞吐 ----------------
    printf("== [B] 吞吐（含 scatter 原子归约，受争用下界）==\n");
    struct Case { int T, H, D, KV; const char* name; };
    Case cases[] = {
        {1024, 8, 128, 512,  "T=1024,H=8,KV=512,D=128"},
        {1024, 8, 128, 2048, "T=1024,H=8,KV=2048,D=128"},
        {4096, 8, 128, 512,  "T=4096,H=8,KV=512,D=128"},
    };
    for (auto& c : cases) {
        size_t nq = (size_t)c.T * c.H * c.D, nkv = (size_t)c.KV * c.D;
        float* Q; CK(cudaMalloc(&Q, nq * 4));
        float* K; CK(cudaMalloc(&K, nkv * 4));
        float* S; CK(cudaMalloc(&S, c.KV * 4));
        float* W; CK(cudaMalloc(&W, (size_t)c.T * c.H * 4));
        float* Dg; CK(cudaMalloc(&Dg, (size_t)c.T * c.KV * 4));
        float* DQ; CK(cudaMalloc(&DQ, nq * 4));
        float* DW; CK(cudaMalloc(&DW, (size_t)c.T * c.H * 4));
        float* DKV; CK(cudaMalloc(&DKV, nkv * 4));
        float* DKVS; CK(cudaMalloc(&DKVS, c.KV * 4));
        int* KS; CK(cudaMalloc(&KS, c.T * 4)); int* KE; CK(cudaMalloc(&KE, c.T * 4));
        std::vector<int> hks(c.T, 0), hke(c.T, c.KV);
        CK(cudaMemcpy(KS, hks.data(), c.T * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(KE, hke.data(), c.T * 4, cudaMemcpyHostToDevice));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        auto run = [&]() {
            CK(cudaMemset(DQ, 0, nq * 4));
            indexer_bwd_kernel<<<dim3(c.T), dim3(32, c.H)>>>(Q, K, S, W, Dg, KS, KE, DQ, DW, DKV, DKVS, c.T, c.H, c.D, c.KV);
        };
        for (int i = 0; i < 3; ++i) run();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        int reps = 20;
        for (int i = 0; i < reps; ++i) run();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); ms /= reps;
        double macs = 4.0 * c.T * c.KV * c.H * c.D;   // 前向点积 2 + 反向点积 ~2
        printf("  %-26s %7.3f ms/call  ~%.1f GFLOP/s\n", c.name, ms, macs / ms / 1e9);
        cudaFree(Q); cudaFree(K); cudaFree(S); cudaFree(W); cudaFree(Dg);
        cudaFree(DQ); cudaFree(DW); cudaFree(DKV); cudaFree(DKVS); cudaFree(KS); cudaFree(KE);
    }
    printf("  注：dkv 的 `atomicAdd` scatter 量 O(T·KV·H·D)，是 indexer 反向的固有归约代价；\n");
    printf("      DeepSeek-V3.2 稀疏阶段每 query token 选 2048 KV（arXiv:2512.02556 §2.1）、\n");
    printf("      且 indexer 头数 H 很小，正是为了压住这个量。\n");
    printf("\n== 总判定：%s ==\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
