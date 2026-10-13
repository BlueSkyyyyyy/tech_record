// P125-1 实验：Mega mHC（HyperConnection）**反向**在目标卡（SM100）上的**单 kernel 接线**里
//   最关键的一环——把 G0（投影 GEMM 的 dgrad，输出 u0[BLOCK_M, 24]）的 TMEM 累加器留在片上、
//   直接喂给 K1（系数反向 / RMSNorm 反向 / Sinkhorn VJP），并且 K1 使用**与前向 Post worker
//   完全相同的 lane-ownership 网格**。这里把该网格与归约路径在 H100(sm90) 上**逐位对拍**，
//   并给出 TMEM 列预算与分段 worker 的真实吞吐。
//
// 背景（docs/130/131/132）：mHC 主 kernel `impls/sm100_mega_mhc.cuh` 是 SM100 专属、且只有前向；
//   前向的「残差多流混合 + 平方和统计」由一个 **per-token Post worker** 完成，它的 lane 网格是：
//     - 一个 CTA 6 个 warpgroup（768 线程）；其中 kNumPostWGs=2 个 Post warpgroup（WG1-2，每 WG 4 warp）
//     - 每个 lane 拥有 **2 个 token 行**、每个 UMMA 原子拥有 **1 个 hidden 浮点对**
//     - kNumRowsPerWarp = BLOCK_M / kNumWarpsPerWG = 16
//       kNumRowGroupsPerWarp = kNumRowsPerWarp / kNumTokenRowsPerLane = 8
//       kNumLanesPerRowGroup = 32 / kNumRowGroupsPerWarp = 4
//   两个 Post worker 各处理一半的 UMMA 原子（`umma_k_idx = worker_idx; += kNumPostWGs`），
//   各自对**全部 BLOCK_M 行**产出 `hc_norm_sqr_sums[worker][row]` 部分和，最后由 WG3 把两个
//   worker 的部分和相加（`sm100_mega_mhc.cuh:545-554`、`:639-640`）。
//
// 本文件在 H100 上**直接 include 真实头文件**（`layout/mega_mhc.cuh`、`common/math.cuh`、
//   `common/utils.cuh`）：
//   A) 逐字段核对前向 Post worker 的几何常量（不自造）。
//   B) device kernel `k_post_stats`：256 线程（2×4 warp）复刻前向 lane 网格，用真实
//      `math::warp_reduce_sum<4>` 做 row-group 归约 + 两个 worker 部分和相加，逐行 q 与
//      host double 全行和 / 全 32-lane 错误归约对拍。
//   C) TMEM 列预算：用真实 `utils::get_num_aligned_tmem_cols`。
//   D) 分段 Post 网格 worker vs 朴素 1-thread-per-row 的真实设备吞吐（sm90 实测）。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。编译/运行见 harness/exp_mhc_backward_segment.sh。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <deep_gemm/layout/mega_mhc.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>

using namespace deep_gemm;
namespace mm = deep_gemm::layout::mega_mhc;

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(_e), __LINE__); exit(1);} } while (0)

// 前向 impl 里定义的几何（先核对，再镜像）。
constexpr uint32_t UMMA_K = 32 / sizeof(float);            // = 8（impls/sm100_mega_mhc.cuh:30）
constexpr uint32_t kNumTokenRowsPerLane = 2;               // impls:31
constexpr uint32_t kNumPostWGs = 2;                        // impls:56

// 由真实布局常量派生的派生量（与 impls:108-110 逐式一致）。
constexpr uint32_t kNumRowsPerWarp = mm::BLOCK_M / mm::kNumWarpsPerWG;              // 16
constexpr uint32_t kNumRowGroupsPerWarp = kNumRowsPerWarp / kNumTokenRowsPerLane;   // 8
constexpr uint32_t kNumLanesPerRowGroup = 32 / kNumRowGroupsPerWarp;                // 4
constexpr uint32_t kNumPostWarps = kNumPostWGs * mm::kNumWarpsPerWG;                // 8
constexpr uint32_t kNumPostThreads = kNumPostWarps * 32;                            // 256

// ---------------------------------------------------------------------------
// A) host 侧几何核对
// ---------------------------------------------------------------------------
static void check_geometry() {
    printf("== A. 前向 Post worker 几何（源自真实 layout/mega_mhc.cuh 与 impls 常量）==\n");
    printf("  kNumRoutes=%u kNumHCOutputs=%u BLOCK_M=%u BLOCK_N=%u BLOCK_K=%u kNumWarpsPerWG=%u\n",
           mm::kNumRoutes, mm::kNumHCOutputs, mm::BLOCK_M, mm::BLOCK_N, mm::BLOCK_K, mm::kNumWarpsPerWG);
    printf("  UMMA_K=%u kNumTokenRowsPerLane=%u kNumPostWGs=%u\n",
           UMMA_K, kNumTokenRowsPerLane, kNumPostWGs);
    printf("  => kNumRowsPerWarp=%u kNumRowGroupsPerWarp=%u kNumLanesPerRowGroup=%u kNumPostThreads=%u\n",
           kNumRowsPerWarp, kNumRowGroupsPerWarp, kNumLanesPerRowGroup, kNumPostThreads);

    // lane -> 拥有的行（first_row_idx = warp%4 * kNumRowsPerWarp + lane / kNumLanesPerRowGroup）
    int row_count[mm::BLOCK_M]; for (auto& c : row_count) c = 0;
    int worker_row_count[kNumPostWGs][mm::BLOCK_M];
    for (auto& a : worker_row_count) for (auto& c : a) c = 0;
    for (uint32_t w = 0; w < kNumPostWarps; ++w) {
        const uint32_t worker = w / mm::kNumWarpsPerWG;
        for (uint32_t l = 0; l < 32; ++l) {
            const uint32_t first = (w % mm::kNumWarpsPerWG) * kNumRowsPerWarp + l / kNumLanesPerRowGroup;
            const uint32_t rows[2] = {first, first + kNumRowGroupsPerWarp};
            for (uint32_t r : rows) { row_count[r]++; worker_row_count[worker][r]++; }
        }
    }
    int bad_total = 0, bad_worker = 0;
    for (uint32_t r = 0; r < mm::BLOCK_M; ++r) {
        // 每个 worker 覆盖每行一次（kNumPostWGs 个 worker 各一次）；每行总共由 kNumPostWGs
        // 个 worker、各 (kNumLanesPerRowGroup) 个 lane 负责。
        if (row_count[r] != (int)(kNumPostWGs * kNumLanesPerRowGroup)) bad_total++;
        for (uint32_t wk = 0; wk < kNumPostWGs; ++wk)
            if (worker_row_count[wk][r] != (int)kNumLanesPerRowGroup) bad_worker++;
    }
    printf("  每行由 %u 个 worker × %u lane = %u 个 lane 负责；行覆盖错误=%d worker 覆盖错误=%d  [%s]\n",
           kNumPostWGs, kNumLanesPerRowGroup, kNumPostWGs * kNumLanesPerRowGroup, bad_total, bad_worker,
           (bad_total == 0 && bad_worker == 0) ? "OK" : "FAIL");

    // lane -> hidden 浮点对（每个原子内 col_pair_idx = lane & 3，hidden_pair = atom*4 + col_pair_idx）
    // 验证一个原子内 4 个 lane 恰好覆盖 UMMA_K/2=4 个浮点对（8 floats）。
    bool cover_ok = true;
    for (uint32_t l = 0; l < 32; l += kNumLanesPerRowGroup) {
        int seen[UMMA_K / 2] = {0};
        for (uint32_t k = 0; k < kNumLanesPerRowGroup; ++k) {
            const uint32_t cp = (l + k) & (kNumLanesPerRowGroup - 1);
            const uint32_t hp = /*atom*/ 0 * (UMMA_K / 2) + cp * (UMMA_K / kNumLanesPerRowGroup / 2);
            if (hp >= UMMA_K / 2) { cover_ok = false; break; }
            seen[hp]++;
        }
        for (uint32_t p = 0; p < UMMA_K / 2; ++p) if (seen[p] != 1) cover_ok = false;
    }
    printf("  一个 UMMA 原子内 4 lane 恰好覆盖 %u 个浮点对（含 col_pair_idx 双射）: %s\n",
           UMMA_K / 2, cover_ok ? "OK" : "FAIL");
}

// ---------------------------------------------------------------------------
// C) TMEM 列预算（真实 utils::get_num_aligned_tmem_cols）
// ---------------------------------------------------------------------------
static void check_tmem() {
    // 前向（impls:64-67）
    constexpr uint32_t kNumTmemColumnsPerAStage = mm::kNumRoutes * UMMA_K;              // 32
    constexpr uint32_t kAccumTmemStartColumn = mm::kNumAStages * kNumTmemColumnsPerAStage; // 4*32=128
    constexpr uint32_t kFwdUsed = kAccumTmemStartColumn + mm::kNumAccumStages * mm::BLOCK_N; // 128+4? no: 2*24
    constexpr uint32_t kFwdCols = utils::get_num_aligned_tmem_cols<kFwdUsed>();
    // 反向：G0 的累加器 u0[BLOCK_M, BLOCK_N] 留在 TMEM
    constexpr uint32_t kG0Cols = utils::get_num_aligned_tmem_cols<mm::BLOCK_N>();
    // K1 的 du0 是 [BLOCK_M, BLOCK_N]（可留在寄存器/TMEM 边角）；总预算取两者之和的保守上界
    constexpr uint32_t kAllocGran = 32;
    printf("== C. TMEM 列预算（真实 get_num_aligned_tmem_cols）==\n");
    printf("  A 每 stage 列 = kNumRoutes*UMMA_K = %u；accum 起点 = kNumAStages*A = %u\n",
           kNumTmemColumnsPerAStage, kAccumTmemStartColumn);
    printf("  前向 used = accum_start + kNumAccumStages*BLOCK_N = %u -> 对齐 %u (%u%%%u=0)\n",
           kFwdUsed, kFwdCols, kFwdCols, kAllocGran);
    printf("  反向 G0 累加器 used = BLOCK_N = %u -> 对齐 %u (整块 512 列预算的 %u/512)\n",
           mm::BLOCK_N, kG0Cols, kG0Cols);
    const bool ok = (kFwdCols == 256) && (kG0Cols == 32) && (kFwdCols + kG0Cols <= 512);
    printf("  判定: %s\n", ok ? "ALL OK" : "FAIL");
}

// ---------------------------------------------------------------------------
// B/D) device：复刻前向 Post worker lane 网格，做逐行平方和统计。
//   输入 A：row-major [T, D]，D = kNumRoutes * H（route-major 拼接）。
//   两个 Post worker 各处理一半的 UMMA 原子（atom % kNumPostWGs == worker）。
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kNumPostThreads, 1)
k_post_stats(const float* __restrict__ A, float* __restrict__ q_out,
             float* __restrict__ worker_partial, int T, int H) {
    const int D = mm::kNumRoutes * H;
    const int m_block = blockIdx.x;
    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t worker = warp_idx / mm::kNumWarpsPerWG;
    const uint32_t col_pair_idx = lane_idx & (kNumLanesPerRowGroup - 1);
    const int first_row_idx = (warp_idx % mm::kNumWarpsPerWG) * kNumRowsPerWarp +
                              lane_idx / kNumLanesPerRowGroup;
    const int rows[2] = {first_row_idx, first_row_idx + (int)kNumRowGroupsPerWarp};

    __shared__ float s_partial[kNumPostWGs][mm::BLOCK_M];

    float2 acc[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};   // per-lane float2（两个行）
    const int num_atoms = H / UMMA_K;
    for (int atom = (int)worker; atom < num_atoms; atom += kNumPostWGs) {
        const int hp = atom * (UMMA_K / 2) + (int)col_pair_idx;        // 浮点对下标（pair*2 = float）
        const int base = hp * 2;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const long row = (long)m_block * mm::BLOCK_M + rows[r];
            if (row >= T) continue;
            #pragma unroll
            for (int route = 0; route < (int)mm::kNumRoutes; ++route) {
                const float* p = A + row * D + route * H + base;
                acc[r].x = fmaf(p[0], p[0], acc[r].x);
                acc[r].y = fmaf(p[1], p[1], acc[r].y);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        // 真实归约：row-group 内 4 lane 求和（kNumLanesPerRowGroup=4）
        const float red = math::warp_reduce_sum<kNumLanesPerRowGroup>(acc[r].x + acc[r].y);
        if (col_pair_idx == 0) s_partial[worker][rows[r]] = red;
    }
    __syncthreads();
    // WG3 阶段：两个 worker 的部分和相加（impls:639-640）。block 有 256 线程，用前 64 个写 64 行。
    if (threadIdx.x < mm::BLOCK_M) {
        const int local = threadIdx.x;
        const long row = (long)m_block * mm::BLOCK_M + local;
        if (row < T) {
            q_out[row] = s_partial[0][local] + s_partial[1][local];
            if (worker_partial != nullptr) {
                worker_partial[row * 2 + 0] = s_partial[0][local];   // 记两个 worker 的分量
                worker_partial[row * 2 + 1] = s_partial[1][local];
            }
        }
    }
}

// D) 朴素：1 thread = 1 row，串行扫全 D。作为吞吐对照。
__global__ void k_coarse(const float* __restrict__ A, float* __restrict__ q_out, int T, int H) {
    const int D = mm::kNumRoutes * H;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= T) return;
    const float* p = A + (long)row * D;
    float acc = 0.f;
    for (int d = 0; d < D; ++d) acc = fmaf(p[d], p[d], acc);
    q_out[row] = acc;
}

// 一个「错误」变体：用 32-lane 全 warp 归约（而不是 row-group 内 4 lane），暴露网格的重要性。
__global__ void __launch_bounds__(kNumPostThreads, 1)
k_post_stats_wrong(const float* __restrict__ A, float* __restrict__ q_out, int T, int H) {
    const int D = mm::kNumRoutes * H;
    const int m_block = blockIdx.x;
    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t worker = warp_idx / mm::kNumWarpsPerWG;
    const uint32_t col_pair_idx = lane_idx & (kNumLanesPerRowGroup - 1);
    const int first_row_idx = (warp_idx % mm::kNumWarpsPerWG) * kNumRowsPerWarp +
                              lane_idx / kNumLanesPerRowGroup;
    const int rows[2] = {first_row_idx, first_row_idx + (int)kNumRowGroupsPerWarp};
    float acc[2] = {0.f, 0.f};
    const int num_atoms = H / UMMA_K;
    for (int atom = (int)worker; atom < num_atoms; atom += kNumPostWGs) {
        const int hp = atom * (UMMA_K / 2) + (int)col_pair_idx;
        const int base = hp * 2;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const long row = (long)m_block * mm::BLOCK_M + rows[r];
            if (row >= T) continue;
            #pragma unroll
            for (int route = 0; route < (int)mm::kNumRoutes; ++route) {
                const float* p = A + row * D + route * H + base;
                acc[r] = fmaf(p[0], p[0], acc[r]);
                acc[r] = fmaf(p[1], p[1], acc[r]);
            }
        }
    }
    __shared__ float s[kNumPostWGs][mm::BLOCK_M];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float red = math::warp_reduce_sum<32>(acc[r]);   // 错：跨 row-group 混在一起
        if (lane_idx == 0) s[worker][rows[r]] = red;
    }
    __syncthreads();
    if (threadIdx.x < mm::BLOCK_M) {
        const int local = threadIdx.x;
        const long row = (long)m_block * mm::BLOCK_M + local;
        if (row < T) q_out[row] = s[0][local] + s[1][local];
    }
}

static double relerr(const std::vector<double>& a, const std::vector<double>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += (a[i] - b[i]) * (a[i] - b[i]); d += b[i] * b[i]; }
    return std::sqrt(n / (d + 1e-300));
}

static void run_correctness(int T, int H, int seed) {
    const int D = mm::kNumRoutes * H;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ur(-1.f, 1.f);
    std::vector<float> hA((size_t)T * D);
    for (auto& v : hA) v = ur(rng);
    std::vector<double> hq((size_t)T, 0.0);
    for (int t = 0; t < T; ++t) { double s = 0; for (int d = 0; d < D; ++d) { double a = hA[(size_t)t * D + d]; s += a * a; } hq[t] = s; }

    float *dA, *dq, *dwp, *dqw;
    CK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CK(cudaMalloc(&dq, (size_t)T * sizeof(float)));
    CK(cudaMalloc(&dqw, (size_t)T * sizeof(float)));
    CK(cudaMalloc(&dwp, (size_t)T * 2 * sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    k_post_stats<<<T / mm::BLOCK_M, kNumPostThreads>>>(dA, dq, dwp, T, H);
    k_post_stats_wrong<<<T / mm::BLOCK_M, kNumPostThreads>>>(dA, dqw, T, H);
    CK(cudaDeviceSynchronize());

    std::vector<float> hq_dev(T), hq_wrong(T), hwp((size_t)T * 2);
    CK(cudaMemcpy(hq_dev.data(), dq, (size_t)T * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hq_wrong.data(), dqw, (size_t)T * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hwp.data(), dwp, (size_t)T * 2 * sizeof(float), cudaMemcpyDeviceToHost));

    auto vec_d = [](const std::vector<float>& v) { std::vector<double> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = v[i]; return o; };
    const double e_grid = relerr(vec_d(hq_dev), hq);
    const double e_wrong = relerr(vec_d(hq_wrong), hq);
    // 两个 worker 的分量和 == 全量（fp32 内）
    double worst_ws = 0;
    for (int t = 0; t < T; ++t) worst_ws = std::max<double>(worst_ws, std::abs((double)hwp[t * 2] + hwp[t * 2 + 1] - hq_dev[t]) / (std::abs(hq_dev[t]) + 1e-30));
    printf("  T=%4d H=%4d D=%5d | 正确网格 q vs double relerr=%.3e | 错(全32-lane) relerr=%.3e | 两 worker 分量和一致性=%.2e  [%s]\n",
           T, H, D, e_grid, e_wrong, worst_ws,
           (e_grid < 1e-5 && e_wrong > 1e-3) ? "OK" : (e_grid < 1e-5 ? "OK(weak)" : "FAIL"));
    cudaFree(dA); cudaFree(dq); cudaFree(dqw); cudaFree(dwp);
}

static double bench(const float* dA, float* dq, int T, int H, int which, int reps) {
    // which: 0 = 分段 Post 网格, 1 = 朴素 1-thread-per-row
    for (int i = 0; i < 3; ++i) {
        if (which == 0) k_post_stats<<<T / mm::BLOCK_M, kNumPostThreads>>>(dA, dq, nullptr, T, H);
        else { int th = 256, bl = (T + th - 1) / th; k_coarse<<<bl, th>>>(dA, dq, T, H); }
    }
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) {
        if (which == 0) k_post_stats<<<T / mm::BLOCK_M, kNumPostThreads>>>(dA, dq, nullptr, T, H);
        else { int th = 256, bl = (T + th - 1) / th; k_coarse<<<bl, th>>>(dA, dq, T, H); }
    }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= reps;
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms;
}

static void run_timing(int T, int H, int reps) {
    const int D = mm::kNumRoutes * H;
    std::vector<float> hA((size_t)T * D, 0.5f);
    float *dA, *dq;
    CK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CK(cudaMalloc(&dq, (size_t)T * sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    const double bytes = (double)hA.size() * 4.0;   // 只读 A（写 q 可忽略）
    const double t_grid = bench(dA, dq, T, H, 0, reps);
    const double t_coarse = bench(dA, dq, T, H, 1, reps);
    printf("  T=%6d H=%4d D=%5d  A=%.1f MiB | 分段网格 %8.3f us (%7.1f GB/s) | 朴素 %8.3f us (%7.1f GB/s) | 加速 %.2fx\n",
           T, H, D, bytes / 1048576.0, t_grid * 1e3, bytes / (t_grid * 1e-3) / 1e9,
           t_coarse * 1e3, bytes / (t_coarse * 1e-3) / 1e9, t_coarse / t_grid);
    cudaFree(dA); cudaFree(dq);
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  sm_%d%d  SMs=%d\n\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    check_geometry();
    printf("\n");
    check_tmem();

    printf("\n== B. 前向 Post 网格的逐行平方和统计：device 分段归约 vs host double ==\n");
    run_correctness(256, 64, 3);
    run_correctness(256, 128, 5);
    run_correctness(512, 256, 7);
    run_correctness(64, 512, 11);
    printf("  （正确网格 = row-group 内 4-lane 归约 + 两 worker 相加；错变体 = 全 32-lane 归约。）\n");

    printf("\n== D. 分段 Post 网格 vs 朴素 1-thread-per-row 的真实吞吐（sm90）==\n");
    run_timing(8192, 512, 200);
    run_timing(16384, 512, 100);
    run_timing(16384, 1024, 100);
    run_timing(32768, 1024, 50);
    printf("\nALL DONE\n");
    return 0;
}
