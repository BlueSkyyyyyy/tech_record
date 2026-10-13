// P149-1 实验：稀疏（块选择）MQA logits（lightning indexer）前向的**反向** —— CUDA 后端。
//
// 背景：DeepGEMM 的 MQA logits 有三条前向——连续 `fp8_fp4_mqa_logits`、分页
// `fp8_fp4_paged_mqa_logits`、**稀疏 `fp8_fp4_{paged_}sparse_mqa_logits`**（在选中的 KV 块内
// 打分、输出压缩成 `[T, num_max_sparse_blocks*SB]`）。第 157 篇（P147-1）补了连续版反向、
// 第 158 篇（P148-1）补了分页版反向，本篇补**稀疏版**。
//
// 稀疏前向语义（`tests/test_attention.py:442-475/644-670` 的 `make_sparse_kv_block_indices`
// 与压缩展开）：
//     - 每个 query 行 r 的有效窗口是 [start_r, len_r)（logical KV token）；
//     - `sparse_indices[r, l]` 是**选中的块号**（以 `sparse_block_kv=SB` 为单位、绝对），l<num_blocks[r]；
//     - 压缩输出第 c=l*SB+kvj 列对应 logical token
//           token = sparse_indices[r,l]*SB + (start_r % SB) + kvj；
//     - 只在 start_r <= token < len_r 时有效（块边界/未对齐窗口/尾块都可能越界）。
//
// 反向与连续/分页**同构**，唯一结构差异是「支持集」不再是整个窗口，而是**被选中块覆盖的稀疏子集**：
//     dL/dkv[token] += sum_{r,h : token in support(r)} dz_{r,token,h} * q_{r,h,d} * scale_token
//     dz_{r,s,h} = dI_sparse[r, c(s)] * w_{r,h} * 1[z>0]，c(s) 是 s 在压缩输出里的列（页表逆映射的稀疏版）。
// 因为是稀疏选择，**未被选中的 token 梯度恒 0**——「块选择」天然是 stop-gradient（块由另一个打分器
// 选，不可微）。且同一物理 token 会被多行引用（next_n 前缀重叠 / 跨序列共享块）→ 必须
// **scatter-accumulate**（原子加），与分页版同一条正确性硬要求。
//
// 布局：grid = P（每 query 行一 block），block = (32 lane, H)。每 h 是 warp；沿 lane 对 d 跨步点积
// + `__shfl_xor` 归约；dq 累寄存器（每 block 独占一行 r）；dw / dscale 在 lane0 累加；
// dkv[token] 沿稀疏支持集走 global `atomicAdd`。
//
// 全部架构无关（fp32 + shuffle + 原子加），H100(sm90) 真机可跑。
// 编译/运行见 harness/exp_mqa_logits_sparse_backward.sh。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// grid = P rows, block = (32, H)
// q [P,H,D], kv [KV,D] (dequantized flat), kvs [KV], w [P,H]
// dIsp [P, C] (compressed upstream grad, C = NBmax*SB), sparse_idx [P,NBmax],
// start [P], cl [P], nsb [P]  -- support derived on the fly.
// dq [P,H,D], dw [P,H], dkv [KV,D], dks [KV]
__global__ void indexer_sparse_bwd_kernel(const float* __restrict__ q,
                                          const float* __restrict__ kv,
                                          const float* __restrict__ kvs,
                                          const float* __restrict__ w,
                                          const float* __restrict__ dIsp,
                                          const int*   __restrict__ sparse_idx,
                                          const int*   __restrict__ start,
                                          const int*   __restrict__ cl,
                                          const int*   __restrict__ nsb,
                                          float* __restrict__ dq,
                                          float* __restrict__ dw,
                                          float* __restrict__ dkv,
                                          float* __restrict__ dks,
                                          int P, int H, int D, int KV, int SB,
                                          int NBmax, int C) {
    const int r = blockIdx.x;
    const int h = threadIdx.y;
    const int lane = threadIdx.x;

    const int st = start[r];
    const int en = cl[r];                 // window is [st, en)
    const int nblocks = nsb[r];
    const int boff = st % SB;             // per-row block offset (in-block shift)
    const int qbase = (r * H + h) * D;
    const float w_rh = w[r * H + h];
    float dw_acc = 0.0f;

    const int* sidx = sparse_idx + (size_t)r * NBmax;

    for (int l = 0; l < nblocks; ++l) {
        const int blk_first = sidx[l] * SB + boff;   // logical token of kvj=0 in this block
        for (int kvj = 0; kvj < SB; ++kvj) {
            const int token = blk_first + kvj;
            if (token < st || token >= en) continue;  // support mask (block/tail/unaligned)
            const int c = l * SB + kvj;               // compressed column
            const float di = dIsp[(size_t)r * C + c];
            if (di == 0.0f) continue;
            const float scale = kvs[token];
            const float* krow = kv + (size_t)token * D;
            float z = 0.0f;
            for (int d = lane; d < D; d += 32)
                z += q[qbase + d] * (krow[d] * scale);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                z += __shfl_xor_sync(0xffffffffu, z, off);

            const float rm = (z > 0.0f) ? 1.0f : 0.0f;
            dw_acc += di * (z * rm);
            const float coeff = di * w_rh * rm;       // dL/dz_{r,token,h}

            for (int d = lane; d < D; d += 32) {
                const float qd = q[qbase + d];
                dq[qbase + d] += coeff * krow[d] * scale;
                atomicAdd(&dkv[(size_t)token * D + d], coeff * qd * scale);
            }
        }
    }
    if (lane == 0) dw[r * H + h] = dw_acc;
}

// dkv_scale[token] += dz * (q · kv_phys)：因一个 token 被多个 (r,h) 累加、且目标按 token 而非行，
// 单独一趟逐 token 原子加（与主 kernel 共用同一支持集构造）。
__global__ void indexer_sparse_dscale_kernel(const float* __restrict__ q,
                                             const float* __restrict__ kv,
                                             const float* __restrict__ kvs,
                                             const float* __restrict__ w,
                                             const float* __restrict__ dIsp,
                                             const int*   __restrict__ sparse_idx,
                                             const int*   __restrict__ start,
                                             const int*   __restrict__ cl,
                                             const int*   __restrict__ nsb,
                                             float* __restrict__ dks,
                                             int P, int H, int D, int KV, int SB, int NBmax, int C) {
    const int r = blockIdx.x;
    const int h = threadIdx.y;
    const int lane = threadIdx.x;
    const int st = start[r], en = cl[r], nblocks = nsb[r], boff = st % SB;
    const int qbase = (r * H + h) * D;
    const float w_rh = w[r * H + h];
    const int* sidx = sparse_idx + (size_t)r * NBmax;
    for (int l = 0; l < nblocks; ++l) {
        const int blk_first = sidx[l] * SB + boff;
        for (int kvj = 0; kvj < SB; ++kvj) {
            const int token = blk_first + kvj;
            if (token < st || token >= en) continue;
            const int c = l * SB + kvj;
            const float di = dIsp[(size_t)r * C + c];
            if (di == 0.0f) continue;
            const float scale = kvs[token];
            const float* krow = kv + (size_t)token * D;
            float z = 0.0f;
            for (int d = lane; d < D; d += 32) z += q[qbase + d] * (krow[d] * scale);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) z += __shfl_xor_sync(0xffffffffu, z, off);
            const float rm = (z > 0.0f) ? 1.0f : 0.0f;
            const float coeff = di * w_rh * rm;
            float dot = 0.0f;
            for (int d = lane; d < D; d += 32) dot += q[qbase + d] * krow[d];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
            if (lane == 0) atomicAdd(&dks[token], coeff * dot);
        }
    }
}

// ---------------------------------------------------------------------------
// host 参考（fp64）：逐元素，含稀疏支持集构造 + scatter 累加
// ---------------------------------------------------------------------------
struct SparseCase {
    int P, H, D, KV, SB, NBmax, C;
    std::vector<int> sparse_idx;  // [P*NBmax]
    std::vector<int> start, cl, nsb;
};

static void ref_backward_sparse(const std::vector<double>& q, const std::vector<double>& kv,
                                const std::vector<double>& kvs, const std::vector<double>& w,
                                const std::vector<double>& dIsp, const SparseCase& k,
                                std::vector<double>& dq, std::vector<double>& dw,
                                std::vector<double>& dkv, std::vector<double>& dks) {
    dq.assign((size_t)k.P * k.H * k.D, 0.0);
    dw.assign((size_t)k.P * k.H, 0.0);
    dkv.assign((size_t)k.KV * k.D, 0.0);
    dks.assign((size_t)k.KV, 0.0);
    for (int r = 0; r < k.P; ++r) {
        const int st = k.start[r], en = k.cl[r], nb = k.nsb[r], boff = st % k.SB;
        const int* sidx = &k.sparse_idx[(size_t)r * k.NBmax];
        for (int h = 0; h < k.H; ++h) {
            const int qbase = (r * k.H + h) * k.D;
            for (int l = 0; l < nb; ++l) {
                const int blk_first = sidx[l] * k.SB + boff;
                for (int kvj = 0; kvj < k.SB; ++kvj) {
                    const int token = blk_first + kvj;
                    if (token < st || token >= en) continue;
                    const int c = l * k.SB + kvj;
                    const double di = dIsp[(size_t)r * k.C + c];
                    if (di == 0.0) continue;
                    const double scale = kvs[token];
                    double z = 0.0;
                    for (int d = 0; d < k.D; ++d) z += q[qbase + d] * kv[(size_t)token * k.D + d] * scale;
                    const double rm = z > 0 ? 1.0 : 0.0;
                    dw[r * k.H + h] += di * z * rm;
                    const double dz = di * w[r * k.H + h] * rm;
                    double dot = 0.0;
                    for (int d = 0; d < k.D; ++d) {
                        const double qd = q[qbase + d];
                        dq[qbase + d] += dz * kv[(size_t)token * k.D + d] * scale;
                        dkv[(size_t)token * k.D + d] += dz * qd * scale;
                        dot += qd * kv[(size_t)token * k.D + d];
                    }
                    dks[token] += dz * dot;
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
static int* dvec_int(const std::vector<int>& v) {
    int* p; CK(cudaMalloc(&p, v.size() * 4)); CK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return p;
}

// build a sparse case: dense enough support, with deliberate cross-row aliasing
static SparseCase build_case(int S, int N, int SB, int NBmax, int nb_per_row, int KV,
                             std::mt19937& rng, bool alias) {
    SparseCase k;
    k.P = S * N; k.SB = SB; k.NBmax = NBmax; k.C = NBmax * SB; k.KV = KV;
    k.sparse_idx.assign((size_t)k.P * NBmax, 0);
    k.start.assign(k.P, 0); k.cl.assign(k.P, 0); k.nsb.assign(k.P, 0);
    // rows all start at 0 to keep support within [0,KV)
    const int total_blocks = KV / SB;
    for (int r = 0; r < k.P; ++r) {
        int nblocks = std::min(nb_per_row, total_blocks);
        // choose blocks; if alias, force the first min(3,total) blocks to be shared across rows
        std::vector<int> blocks;
        if (alias) {
            for (int b = 0; b < std::min(3, nblocks); ++b) blocks.push_back(b);
        }
        while ((int)blocks.size() < nblocks) {
            int b = (int)(rng() % total_blocks);
            bool dup = false;
            for (int x : blocks) if (x == b) { dup = true; break; }
            if (!dup) blocks.push_back(b);
        }
        std::sort(blocks.begin(), blocks.end());
        for (int l = 0; l < nblocks; ++l) k.sparse_idx[(size_t)r * NBmax + l] = blocks[l];
        for (int l = nblocks; l < NBmax; ++l) k.sparse_idx[(size_t)r * NBmax + l] = blocks.empty() ? 0 : blocks.back();
        k.nsb[r] = nblocks;
        k.cl[r] = KV;                       // full window (all selected tokens valid)
    }
    return k;
}

int main() {
    // ---------------- [A] 正确性（含跨行块别名 + 稀疏支持集） ------------------
    printf("== [A] 正确性：稀疏 indexer 反向 CUDA vs host fp64（稀疏支持集 + 别名累加）==\n");
    std::mt19937 rng(2024);
    std::normal_distribution<float> nd(0.f, 0.3f);
    std::uniform_real_distribution<float> ud(0.5f, 1.2f);
    int S = 4, N = 3, H = 8, D = 64, SB = 16, NBmax = 8, KV = 16 * SB;   // 16 blocks
    SparseCase k = build_case(S, N, SB, NBmax, 6, KV, rng, /*alias=*/true);
    k.H = H; k.D = D;
    int P = k.P, C = k.C;
    std::vector<float> q((size_t)P * H * D), kv((size_t)KV * D), kvs(KV), w((size_t)P * H), dIsp((size_t)P * C);
    for (auto& x : q) x = nd(rng);
    for (auto& x : kv) x = nd(rng);
    for (auto& x : kvs) x = ud(rng);
    for (auto& x : w) x = nd(rng);
    for (auto& x : dIsp) x = nd(rng);

    std::vector<double> qd(q.begin(), q.end()), kvd(kv.begin(), kv.end()), kvsd(kvs.begin(), kvs.end()),
                        wd(w.begin(), w.end()), did(dIsp.begin(), dIsp.end());
    std::vector<double> rdq, rdw, rdkv, rdks;
    ref_backward_sparse(qd, kvd, kvsd, wd, did, k, rdq, rdw, rdkv, rdks);

    float *dq, *dw, *dkv, *dks;
    CK(cudaMalloc(&dq, (size_t)P * H * D * 4));
    CK(cudaMalloc(&dw, (size_t)P * H * 4));
    CK(cudaMalloc(&dkv, (size_t)KV * D * 4));
    CK(cudaMalloc(&dks, (size_t)KV * 4));
    CK(cudaMemset(dq, 0, (size_t)P * H * D * 4));
    CK(cudaMemset(dw, 0, (size_t)P * H * 4));
    CK(cudaMemset(dkv, 0, (size_t)KV * D * 4));
    CK(cudaMemset(dks, 0, (size_t)KV * 4));
    float *pq = dvec(q), *pkv = dvec(kv), *pkvs = dvec(kvs), *pw = dvec(w), *pdI = dvec(dIsp);
    int *psi = dvec_int(k.sparse_idx), *pst = dvec_int(k.start), *pcl = dvec_int(k.cl), *pnsb = dvec_int(k.nsb);

    dim3 block(32, H), grid(P);
    indexer_sparse_bwd_kernel<<<grid, block>>>(pq, pkv, pkvs, pw, pdI, psi, pst, pcl, pnsb,
                                               dq, dw, dkv, dks, P, H, D, KV, SB, NBmax, C);
    indexer_sparse_dscale_kernel<<<grid, block>>>(pq, pkv, pkvs, pw, pdI, psi, pst, pcl, pnsb,
                                                  dks, P, H, D, KV, SB, NBmax, C);
    CK(cudaDeviceSynchronize());

    std::vector<float> hdq((size_t)P * H * D), hdw((size_t)P * H), hdkv((size_t)KV * D), hdks(KV);
    CK(cudaMemcpy(hdq.data(), dq, hdq.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdw.data(), dw, hdw.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdkv.data(), dkv, hdkv.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdks.data(), dks, hdks.size() * 4, cudaMemcpyDeviceToHost));
    printf("  dq        relerr = %.3e\n", relerr(hdq, rdq));
    printf("  dw        relerr = %.3e\n", relerr(hdw, rdw));
    printf("  dkv       relerr = %.3e\n", relerr(hdkv, rdkv));
    printf("  dkv_scale relerr = %.3e\n", relerr(hdks, rdks));

    // 对照：用「写」而非 scatter-accumulate（last-writer-wins）会丢多少
    std::vector<double> rdkv_naive((size_t)KV * D, 0.0);
    for (int r = 0; r < P; ++r) {
        const int st = k.start[r], en = k.cl[r], nb = k.nsb[r], boff = st % SB;
        const int* sidx = &k.sparse_idx[(size_t)r * NBmax];
        for (int h = 0; h < H; ++h) {
            const int qbase = (r * H + h) * D;
            for (int l = 0; l < nb; ++l) {
                const int blk_first = sidx[l] * SB + boff;
                for (int kvj = 0; kvj < SB; ++kvj) {
                    const int token = blk_first + kvj;
                    if (token < st || token >= en) continue;
                    const int c = l * SB + kvj;
                    const double di = did[(size_t)r * C + c];
                    if (di == 0.0) continue;
                    const double scale = kvsd[token];
                    double z = 0.0;
                    for (int d = 0; d < D; ++d) z += qd[qbase + d] * kvd[(size_t)token * D + d] * scale;
                    const double rm = z > 0 ? 1.0 : 0.0;
                    const double dz = di * wd[r * H + h] * rm;
                    for (int d = 0; d < D; ++d)
                        rdkv_naive[(size_t)token * D + d] = dz * qd[qbase + d] * scale;  // overwrite (bug)
                }
            }
        }
    }
    double naive_err = relerr(hdkv, rdkv_naive);
    printf("  (对照) 若用「写」而非 scatter-accumulate，dkv 与正确值 relerr = %.3e\n", naive_err);

    bool ok = relerr(hdq, rdq) < 1e-5 && relerr(hdw, rdw) < 1e-5 &&
              relerr(hdkv, rdkv) < 1e-5 && relerr(hdks, rdks) < 1e-5;
    printf("  => [A] %s\n\n", ok ? "ALL PASS" : "FAIL");

    // ---------------- [B] 吞吐 ----------------
    printf("== [B] 吞吐（稀疏支持集 + scatter 原子归约，受争用下界）==\n");
    struct Case { int S, N, H, D, SB, avg_sel_tokens, NBmax; const char* name; };
    Case cases[] = {
        {64,  1, 32, 128, 64, 2048, 64,  "S=64,N=1,H=32,D=128,sel=2048,blk=64"},
        {64,  4, 32, 128, 64, 2048, 64,  "S=64,N=4,H=32,D=128,sel=2048,blk=64"},
        {256, 1, 32, 128, 64, 2048, 64,  "S=256,N=1,H=32,D=128,sel=2048,blk=64"},
    };
    for (auto& cs : cases) {
        int Pc = cs.S * cs.N;
        int KVc = 64 * 1024;                    // a large KV pool
        int total_blocks = KVc / cs.SB;
        int nb = std::min(cs.avg_sel_tokens / cs.SB, total_blocks);
        int Cc = cs.NBmax * cs.SB;
        std::vector<int> sidx((size_t)Pc * cs.NBmax), st(Pc, 0), ccl(Pc, KVc), nsv(Pc, nb);
        for (int r = 0; r < Pc; ++r)
            for (int l = 0; l < nb; ++l) sidx[(size_t)r * cs.NBmax + l] = l;   // first nb blocks
        float *Q, *K, *Sf, *W, *Dg, *DQ, *DW, *DKV, *DKS;
        CK(cudaMalloc(&Q, (size_t)Pc * cs.H * cs.D * 4));
        CK(cudaMalloc(&K, (size_t)KVc * cs.D * 4));
        CK(cudaMalloc(&Sf, (size_t)KVc * 4));
        CK(cudaMalloc(&W, (size_t)Pc * cs.H * 4));
        CK(cudaMalloc(&Dg, (size_t)Pc * Cc * 4));
        CK(cudaMalloc(&DQ, (size_t)Pc * cs.H * cs.D * 4));
        CK(cudaMalloc(&DW, (size_t)Pc * cs.H * 4));
        CK(cudaMalloc(&DKV, (size_t)KVc * cs.D * 4));
        CK(cudaMalloc(&DKS, (size_t)KVc * 4));
        int *PSI, *PST, *PCL, *PNSV;
        CK(cudaMalloc(&PSI, sidx.size() * 4)); CK(cudaMemcpy(PSI, sidx.data(), sidx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&PST, Pc * 4)); CK(cudaMemcpy(PST, st.data(), Pc * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&PCL, Pc * 4)); CK(cudaMemcpy(PCL, ccl.data(), Pc * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&PNSV, Pc * 4)); CK(cudaMemcpy(PNSV, nsv.data(), Pc * 4, cudaMemcpyHostToDevice));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        auto run = [&]() {
            CK(cudaMemset(DQ, 0, (size_t)Pc * cs.H * cs.D * 4));
            indexer_sparse_bwd_kernel<<<dim3(Pc), dim3(32, cs.H)>>>(Q, K, Sf, W, Dg, PSI, PST, PCL, PNSV,
                                                                    DQ, DW, DKV, DKS, Pc, cs.H, cs.D, KVc,
                                                                    cs.SB, cs.NBmax, Cc);
            CK(cudaGetLastError());
        };
        for (int i = 0; i < 3; ++i) run();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        int reps = 10;
        for (int i = 0; i < reps; ++i) run();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); ms /= reps;
        double macs = 4.0 * (double)Pc * (double)nb * cs.SB * cs.H * cs.D;
        printf("  %-40s %8.3f ms/call  ~%.2f GFLOP/s\n", cs.name, ms, macs / ms / 1e9);
        cudaFree(Q); cudaFree(K); cudaFree(Sf); cudaFree(W); cudaFree(Dg);
        cudaFree(DQ); cudaFree(DW); cudaFree(DKV); cudaFree(DKS);
        cudaFree(PSI); cudaFree(PST); cudaFree(PCL); cudaFree(PNSV);
    }
    printf("  注：稀疏支持集把 dkv 的 scatter 量从 O(sum_r cl_r) 降到 O(sum_r selected_tokens_r)，\n");
    printf("      但每 token 仍被 H 个 (r,h) 累加 -> 原子争用；生产实现按物理块做局部归约。\n");
    printf("\n== 总判定：%s ==\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
