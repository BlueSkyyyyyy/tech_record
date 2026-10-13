// P150-1 实验：**分页稀疏（块选择）MQA logits（lightning indexer）前向的反向** —— CUDA 后端。
//
// 背景：DeepGEMM 的 MQA logits 前向有四条——连续 `fp8_fp4_mqa_logits`、分页
// `fp8_fp4_paged_mqa_logits`、稀疏 `fp8_fp4_sparse_mqa_logits`、**分页稀疏
// `fp8_fp4_paged_sparse_mqa_logits`**（生产 long-context 训练真正跑的那条：KV 住在物理页池里、
// 只用 indexer 选中的一小撮块打分、输出压缩成 `[T, num_max_sparse_blocks*SB]`）。第 157 篇
// （P147-1）补了连续版反向、第 158 篇（P148-1）补了分页版、第 159 篇（P149-1）补了稀疏版。
// 本篇补**第四条 / 组合形态**：分页 × 稀疏。
//
// 分页稀疏前向语义（`tests/test_attention.py:484-672` 的 `enumerate_sparse_mqa_logits` /
// `make_sparse_kv_block_indices` 与 `:655-668` 的压缩展开，唯一正确性规范；`attention.hpp:337-441`
// 的 host 入口）：
//     - 每 query 行 r 有一个 context 长度 cl[r]（varlen 展开成每行一份，`context_starts=0`）；
//     - `sparse_indices[r, l]` 是**选中的稀疏块号**（以 `sparse_block_kv=SB` 为单位、logical），l<nsb[r]；
//     - 压缩输出第 c=l*SB+kvj 列对应 **logical** KV token
//           t = sparse_indices[r,l]*SB + (start_r % SB) + kvj；   （paged 时 start_r=0）
//     - 只在 start_r <= t < cl_r 时有效（尾块/未对齐窗口越界 → support）；
//     - **logical token t 的物理槽**：phys = block_table[r, t // PAGE_KV] * PAGE_KV + t % PAGE_KV
//       （与 metadata kernel `scheduler/sm100_sparse_mqa_logits_metadata.cuh:379-383` 的
//        `physical_kv_block_idx = block_table[...] * (PAGE_KV/SPARSE_BLOCK_KV) + logical_kv_block_idx % ...`
//        逐字等价，都乘回 `SB` 变 token 偏移）。
//
// 反向与连续/分页/稀疏**同构**，唯一结构差异是要**依次**做两级逆映射：
//     support（选中块 ∩ 有效窗口）  →  logical token  →  （页表逆映射）  →  physical slot，
// 再 scatter-accumulate：
//     dL/dkv_phys[phys] += sum_{r,h : phys(s)=phys, s in support(r)} dz_{r,s,h} * q_{r,h,d} * scale_phys
//     dL/dq_{r,h,d}     = sum_{s in support(r)} dz_{r,s,h} * kv_phys(s),d * scale_phys(s)
//     dz_{r,s,h}        = dI_sparse[r, c(s)] * w_{r,h} * 1[z>0]
// 两层别名都必须 scatter-accumulate：① next_n 前缀重叠（同一请求的多个 q token 命中同一批块）；
// ② 跨请求共享物理页（prefix sharing / beam search，或页池随机分配后的碰撞）。用「写」会丢贡献。
//
// 布局：grid = P（每 query 行一 block），block = (32 lane, H)。每 h 是 warp；沿 lane 对 d 跨步点积
// + `__shfl_xor` 归约；dq 累寄存器（每 block 独占一行 r）；dw 在 lane0 累加；dkv/dscale 沿**物理槽**走
// global `atomicAdd`。
//
// 全部架构无关（fp32 + shuffle + 原子加），H100(sm90) 真机可跑。
// 编译/运行见 harness/exp_mqa_logits_paged_sparse_backward.sh。
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
// q [P,H,D], kv [NPOOL*PAGE, D] (dequantized physical pool), kvs [NPOOL*PAGE], w [P,H]
// dIsp [P, C] (compressed upstream grad, C = NBmax*SB), sparse_idx [P,NBmax],
// start [P], cl [P], nsb [P], bt [P, MAXP] (page table, per-row)
// dq [P,H,D], dw [P,H], dkv [NPOOL*PAGE,D], dks [NPOOL*PAGE]
__global__ void indexer_paged_sparse_bwd_kernel(const float* __restrict__ q,
                                                const float* __restrict__ kv,
                                                const float* __restrict__ kvs,
                                                const float* __restrict__ w,
                                                const float* __restrict__ dIsp,
                                                const int*   __restrict__ sparse_idx,
                                                const int*   __restrict__ start,
                                                const int*   __restrict__ cl,
                                                const int*   __restrict__ nsb,
                                                const int*   __restrict__ bt,
                                                float* __restrict__ dq,
                                                float* __restrict__ dw,
                                                float* __restrict__ dkv,
                                                int P, int H, int D, int SB,
                                                int PAGE, int MAXP, int NBmax, int C) {
    const int r = blockIdx.x;
    const int h = threadIdx.y;
    const int lane = threadIdx.x;

    const int st = start[r];
    const int en = cl[r];                 // logical window is [st, en)
    const int nblocks = nsb[r];
    const int boff = st % SB;             // per-row block offset (in-block shift); 0 for paged
    const int qbase = (r * H + h) * D;
    const float w_rh = w[r * H + h];
    float dw_acc = 0.0f;

    const int* sidx = sparse_idx + (size_t)r * NBmax;
    const int* btr = bt + (size_t)r * MAXP;

    for (int l = 0; l < nblocks; ++l) {
        const int blk_first = sidx[l] * SB + boff;   // logical token of kvj=0 in this block
        for (int kvj = 0; kvj < SB; ++kvj) {
            const int token = blk_first + kvj;       // logical KV token
            if (token < st || token >= en) continue; // support mask (block/tail/unaligned)
            const int c = l * SB + kvj;              // compressed column
            const float di = dIsp[(size_t)r * C + c];
            if (di == 0.0f) continue;
            // two-level inverse map: logical token -> physical slot via page table
            const int phys = btr[token / PAGE] * PAGE + (token % PAGE);
            const float scale = kvs[phys];
            const float* krow = kv + (size_t)phys * D;
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
                atomicAdd(&dkv[(size_t)phys * D + d], coeff * qd * scale);
            }
        }
    }
    if (lane == 0) dw[r * H + h] = dw_acc;
}

// dscale_phys[phys] += dz * (q · kv_phys)：目标按**物理槽**而非行，单独一趟逐槽原子加
// （与主 kernel 共用同一 support + 页表逆映射）。
__global__ void indexer_paged_sparse_dscale_kernel(const float* __restrict__ q,
                                                   const float* __restrict__ kv,
                                                   const float* __restrict__ kvs,
                                                   const float* __restrict__ w,
                                                   const float* __restrict__ dIsp,
                                                   const int*   __restrict__ sparse_idx,
                                                   const int*   __restrict__ start,
                                                   const int*   __restrict__ cl,
                                                   const int*   __restrict__ nsb,
                                                   const int*   __restrict__ bt,
                                                   float* __restrict__ dks,
                                                   int P, int H, int D, int SB,
                                                   int PAGE, int MAXP, int NBmax, int C) {
    const int r = blockIdx.x;
    const int h = threadIdx.y;
    const int lane = threadIdx.x;
    const int st = start[r], en = cl[r], nblocks = nsb[r], boff = st % SB;
    const int qbase = (r * H + h) * D;
    const float w_rh = w[r * H + h];
    const int* sidx = sparse_idx + (size_t)r * NBmax;
    const int* btr = bt + (size_t)r * MAXP;
    for (int l = 0; l < nblocks; ++l) {
        const int blk_first = sidx[l] * SB + boff;
        for (int kvj = 0; kvj < SB; ++kvj) {
            const int token = blk_first + kvj;
            if (token < st || token >= en) continue;
            const int c = l * SB + kvj;
            const float di = dIsp[(size_t)r * C + c];
            if (di == 0.0f) continue;
            const int phys = btr[token / PAGE] * PAGE + (token % PAGE);
            const float scale = kvs[phys];
            const float* krow = kv + (size_t)phys * D;
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
            if (lane == 0) atomicAdd(&dks[phys], coeff * dot);
        }
    }
}

// ---------------------------------------------------------------------------
// host 参考（fp64）：逐元素，含 support 构造 + 页表逆映射 + scatter 累加
// ---------------------------------------------------------------------------
struct Case {
    int P, H, D, SB, PAGE, MAXP, NBmax, C, NTOK;
    std::vector<int> sparse_idx;  // [P*NBmax]
    std::vector<int> start, cl, nsb;
    std::vector<int> bt;          // [P*MAXP]
};

static void ref_bwd_paged_sparse(const std::vector<double>& q, const std::vector<double>& kv,
                                 const std::vector<double>& kvs, const std::vector<double>& w,
                                 const std::vector<double>& dIsp, const Case& k,
                                 std::vector<double>& dq, std::vector<double>& dw,
                                 std::vector<double>& dkv, std::vector<double>& dks) {
    dq.assign((size_t)k.P * k.H * k.D, 0.0);
    dw.assign((size_t)k.P * k.H, 0.0);
    dkv.assign((size_t)k.NTOK * k.D, 0.0);
    dks.assign((size_t)k.NTOK, 0.0);
    for (int r = 0; r < k.P; ++r) {
        const int st = k.start[r], en = k.cl[r], nb = k.nsb[r], boff = st % k.SB;
        const int* sidx = &k.sparse_idx[(size_t)r * k.NBmax];
        const int* btr = &k.bt[(size_t)r * k.MAXP];
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
                    const int phys = btr[token / k.PAGE] * k.PAGE + (token % k.PAGE);
                    const double scale = kvs[phys];
                    double z = 0.0;
                    for (int d = 0; d < k.D; ++d) z += q[qbase + d] * kv[(size_t)phys * k.D + d] * scale;
                    const double rm = z > 0 ? 1.0 : 0.0;
                    dw[r * k.H + h] += di * z * rm;
                    const double dz = di * w[r * k.H + h] * rm;
                    double dot = 0.0;
                    for (int d = 0; d < k.D; ++d) {
                        const double qd = q[qbase + d];
                        dq[qbase + d] += dz * kv[(size_t)phys * k.D + d] * scale;
                        dkv[(size_t)phys * k.D + d] += dz * qd * scale;
                        dot += qd * kv[(size_t)phys * k.D + d];
                    }
                    dks[phys] += dz * dot;
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

// build a paged-sparse case: page pool + per-row block table + sparse block selection, with
// deliberate two-level aliasing (next_n rows reuse blocks; some rows share pages).
static Case build_case(int S, int N, int SB, int PAGE, int NBmax, int nb_per_row, int nreq_tokens,
                       std::mt19937& rng, bool alias) {
    Case k;
    k.P = S * N; k.SB = SB; k.PAGE = PAGE; k.NBmax = NBmax; k.C = NBmax * SB;
    const int nblocks_total = (nreq_tokens + SB - 1) / SB;        // logical sparse blocks per request
    k.MAXP = (nreq_tokens + PAGE - 1) / PAGE;                     // pages per request
    k.NTOK = 0;
    k.sparse_idx.assign((size_t)k.P * NBmax, 0);
    k.start.assign(k.P, 0); k.cl.assign(k.P, 0); k.nsb.assign(k.P, 0);
    k.bt.assign((size_t)k.P * k.MAXP, 0);

    // page pool: S requests * MAXP pages, optionally letting request 1 share request 0's pages
    int num_pages = (alias ? (S - 1) : S) * k.MAXP;
    std::vector<int> pool(num_pages);
    for (int i = 0; i < num_pages; ++i) pool[i] = i;
    std::shuffle(pool.begin(), pool.end(), rng);
    std::vector<std::vector<int>> req_pages(S);
    for (int i = 0; i < S; ++i) {
        int base = (alias and i >= 1) ? 0 : i;                    // request>=1 aliases request 0
        for (int p = 0; p < k.MAXP; ++p) req_pages[i].push_back(pool[(size_t)base * k.MAXP + p]);
    }

    for (int s = 0; s < S; ++s) {
        // per-sequence selected logical sparse blocks (shared across its N next_n rows)
        std::vector<int> blocks;
        int nb = std::min(nb_per_row, nblocks_total);
        while ((int)blocks.size() < nb) {
            int b = (int)(rng() % nblocks_total);
            bool dup = false;
            for (int x : blocks) if (x == b) { dup = true; break; }
            if (!dup) blocks.push_back(b);
        }
        std::sort(blocks.begin(), blocks.end());
        for (int j = 0; j < N; ++j) {
            int r = s * N + j;
            for (int p = 0; p < k.MAXP; ++p) k.bt[(size_t)r * k.MAXP + p] = req_pages[s][p];
            for (int l = 0; l < nb; ++l) k.sparse_idx[(size_t)r * NBmax + l] = blocks[l];
            for (int l = nb; l < NBmax; ++l) k.sparse_idx[(size_t)r * NBmax + l] = blocks.empty() ? 0 : blocks.back();
            k.nsb[r] = nb;
            k.cl[r] = nreq_tokens;                                 // full window: every selected token valid
        }
    }
    k.NTOK = num_pages * PAGE;
    return k;
}

int main() {
    // ---------------- [A] 正确性（两级逆映射 + 两层别名） ------------------
    printf("== [A] 正确性：分页稀疏 indexer 反向 CUDA vs host fp64（support + 页表逆映射 + 别名累加）==\n");
    std::mt19937 rng(2024);
    std::normal_distribution<float> nd(0.f, 0.3f);
    std::uniform_real_distribution<float> ud(0.5f, 1.2f);
    int S = 5, N = 3, H = 8, D = 64, SB = 16, PAGE = 64, NBmax = 8, nreq_tokens = 12 * SB;  // 12 blocks
    Case k = build_case(S, N, SB, PAGE, NBmax, 6, nreq_tokens, rng, /*alias=*/true);
    k.H = H; k.D = D;
    int P = k.P, C = k.C, NTOK = k.NTOK;
    std::vector<float> q((size_t)P * H * D), kv((size_t)NTOK * D), kvs(NTOK), w((size_t)P * H), dIsp((size_t)P * C);
    for (auto& x : q) x = nd(rng);
    for (auto& x : kv) x = nd(rng);
    for (auto& x : kvs) x = ud(rng);
    for (auto& x : w) x = nd(rng);
    for (auto& x : dIsp) x = nd(rng);

    std::vector<double> qd(q.begin(), q.end()), kvd(kv.begin(), kv.end()), kvsd(kvs.begin(), kvs.end()),
                        wd(w.begin(), w.end()), did(dIsp.begin(), dIsp.end());
    std::vector<double> rdq, rdw, rdkv, rdks;
    ref_bwd_paged_sparse(qd, kvd, kvsd, wd, did, k, rdq, rdw, rdkv, rdks);

    float *dq, *dw, *dkv, *dks;
    CK(cudaMalloc(&dq, (size_t)P * H * D * 4));
    CK(cudaMalloc(&dw, (size_t)P * H * 4));
    CK(cudaMalloc(&dkv, (size_t)NTOK * D * 4));
    CK(cudaMalloc(&dks, (size_t)NTOK * 4));
    CK(cudaMemset(dq, 0, (size_t)P * H * D * 4));
    CK(cudaMemset(dw, 0, (size_t)P * H * 4));
    CK(cudaMemset(dkv, 0, (size_t)NTOK * D * 4));
    CK(cudaMemset(dks, 0, (size_t)NTOK * 4));
    float *pq = dvec(q), *pkv = dvec(kv), *pkvs = dvec(kvs), *pw = dvec(w), *pdI = dvec(dIsp);
    int *psi = dvec_int(k.sparse_idx), *pst = dvec_int(k.start), *pcl = dvec_int(k.cl),
        *pnsb = dvec_int(k.nsb), *pbt = dvec_int(k.bt);

    dim3 block(32, H), grid(P);
    indexer_paged_sparse_bwd_kernel<<<grid, block>>>(pq, pkv, pkvs, pw, pdI, psi, pst, pcl, pnsb, pbt,
                                                     dq, dw, dkv, P, H, D, SB, PAGE, k.MAXP, NBmax, C);
    indexer_paged_sparse_dscale_kernel<<<grid, block>>>(pq, pkv, pkvs, pw, pdI, psi, pst, pcl, pnsb, pbt,
                                                        dks, P, H, D, SB, PAGE, k.MAXP, NBmax, C);
    CK(cudaDeviceSynchronize());

    std::vector<float> hdq((size_t)P * H * D), hdw((size_t)P * H), hdkv((size_t)NTOK * D), hdks(NTOK);
    CK(cudaMemcpy(hdq.data(), dq, hdq.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdw.data(), dw, hdw.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdkv.data(), dkv, hdkv.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdks.data(), dks, hdks.size() * 4, cudaMemcpyDeviceToHost));
    printf("  dq        relerr = %.3e\n", relerr(hdq, rdq));
    printf("  dw        relerr = %.3e\n", relerr(hdw, rdw));
    printf("  dkv       relerr = %.3e\n", relerr(hdkv, rdkv));
    printf("  dkv_scale relerr = %.3e\n", relerr(hdks, rdks));

    // 对照：用「写」而非 scatter-accumulate（last-writer-wins）会丢多少（两层别名都暴露）
    std::vector<double> rdkv_naive((size_t)NTOK * D, 0.0);
    for (int r = 0; r < P; ++r) {
        const int st = k.start[r], en = k.cl[r], nb = k.nsb[r], boff = st % SB;
        const int* sidx = &k.sparse_idx[(size_t)r * NBmax];
        const int* btr = &k.bt[(size_t)r * k.MAXP];
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
                    const int phys = btr[token / PAGE] * PAGE + (token % PAGE);
                    const double scale = kvsd[phys];
                    double z = 0.0;
                    for (int d = 0; d < D; ++d) z += qd[qbase + d] * kvd[(size_t)phys * D + d] * scale;
                    const double rm = z > 0 ? 1.0 : 0.0;
                    const double dz = di * wd[r * H + h] * rm;
                    for (int d = 0; d < D; ++d)
                        rdkv_naive[(size_t)phys * D + d] = dz * qd[qbase + d] * scale;  // overwrite (bug)
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
    printf("== [B] 吞吐（support + 页表逆映射 + scatter 原子归约，受争用下界）==\n");
    struct C2 { int S, N, H, D, SB, PAGE, avg_sel_tokens, NBmax; const char* name; };
    C2 cases[] = {
        {64,  1, 32, 128, 64, 64, 2048, 64, "S=64,N=1,H=32,D=128,sel=2048,blk=64"},
        {64,  4, 32, 128, 64, 64, 2048, 64, "S=64,N=4,H=32,D=128,sel=2048,blk=64"},
        {256, 1, 32, 128, 64, 64, 2048, 64, "S=256,N=1,H=32,D=128,sel=2048,blk=64"},
    };
    for (auto& cs : cases) {
        int Pc = cs.S * cs.N;
        int req_tokens = 64 * 1024;
        int nblocks_total = req_tokens / cs.SB;
        int nb = std::min(cs.avg_sel_tokens / cs.SB, nblocks_total);
        int Cc = cs.NBmax * cs.SB;
        int maxp = req_tokens / cs.PAGE;
        int npool = cs.S * maxp;
        int ntok = npool * cs.PAGE;
        std::vector<int> sidx((size_t)Pc * cs.NBmax), st(Pc, 0), ccl(Pc, req_tokens), nsv(Pc, nb), btv((size_t)Pc * maxp);
        for (int r = 0; r < Pc; ++r) {
            for (int l = 0; l < nb; ++l) sidx[(size_t)r * cs.NBmax + l] = l;   // first nb blocks
            int s = r / cs.N;
            for (int p = 0; p < maxp; ++p) btv[(size_t)r * maxp + p] = s * maxp + p;
        }
        float *Q, *K, *Sf, *W, *Dg, *DQ, *DW, *DKV, *DKS;
        CK(cudaMalloc(&Q, (size_t)Pc * cs.H * cs.D * 4));
        CK(cudaMalloc(&K, (size_t)ntok * cs.D * 4));
        CK(cudaMalloc(&Sf, (size_t)ntok * 4));
        CK(cudaMalloc(&W, (size_t)Pc * cs.H * 4));
        CK(cudaMalloc(&Dg, (size_t)Pc * Cc * 4));
        CK(cudaMalloc(&DQ, (size_t)Pc * cs.H * cs.D * 4));
        CK(cudaMalloc(&DW, (size_t)Pc * cs.H * 4));
        CK(cudaMalloc(&DKV, (size_t)ntok * cs.D * 4));
        CK(cudaMalloc(&DKS, (size_t)ntok * 4));
        int *PSI, *PST, *PCL, *PNSV, *PBT;
        CK(cudaMalloc(&PSI, sidx.size() * 4)); CK(cudaMemcpy(PSI, sidx.data(), sidx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&PST, Pc * 4)); CK(cudaMemcpy(PST, st.data(), Pc * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&PCL, Pc * 4)); CK(cudaMemcpy(PCL, ccl.data(), Pc * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&PNSV, Pc * 4)); CK(cudaMemcpy(PNSV, nsv.data(), Pc * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&PBT, btv.size() * 4)); CK(cudaMemcpy(PBT, btv.data(), btv.size() * 4, cudaMemcpyHostToDevice));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        auto run = [&]() {
            CK(cudaMemset(DQ, 0, (size_t)Pc * cs.H * cs.D * 4));
            indexer_paged_sparse_bwd_kernel<<<dim3(Pc), dim3(32, cs.H)>>>(Q, K, Sf, W, Dg, PSI, PST, PCL, PNSV, PBT,
                                                                          DQ, DW, DKV, Pc, cs.H, cs.D, cs.SB,
                                                                          cs.PAGE, maxp, cs.NBmax, Cc);
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
        cudaFree(PSI); cudaFree(PST); cudaFree(PCL); cudaFree(PNSV); cudaFree(PBT);
    }
    printf("  注：分页稀疏把 dkv 的 scatter 量降到 O(sum_r selected_tokens_r)，但目标按**物理槽**聚合，\n");
    printf("      跨请求共享页让同一物理槽被更多 (r,h) 累加 -> 原子争用比纯稀疏更高。\n");
    printf("\n== 总判定：%s ==\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
