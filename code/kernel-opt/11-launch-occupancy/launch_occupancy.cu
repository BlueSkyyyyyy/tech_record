// 11 launch 配置与 occupancy：__launch_bounds__ / 循环展开 / ILP，落地到第 09 篇的 GEMM。
//
// 三个实验：
//   1) ILP：固定低 occupancy（每 SM 只驻留 1 个 block），改变每线程独立 FMA 链的条数。
//      ILP=1 时每条 FMA 依赖上一条，延迟盖不住 → 慢；ILP 足够大 → 打满计算管道。
//   2) __launch_bounds__(threads, MINB)：对同一个 GEMM 配置强行要求每 SM 驻留更多 block，
//      编译器被逼压低寄存器 → spill 到 local memory → 反而更慢。
//   3) thread tile 扫描：每线程算 TM×TN 个输出。tile 越大，smem 载入复用越好、寄存器越多、
//      occupancy 越低。看「复用/ILP」与「occupancy」谁说了算。
//
// 运行：scripts/run.sh 11-launch-occupancy/launch_occupancy.cu [M] [N] [K]
#include "../common/cuda_utils.cuh"

#include <cmath>
#include <cstring>

// ===========================================================================
// 实验 1：ILP —— 每线程 ILP 条互相独立的 FMA 链
//
// 每线程固定做 FMA_TOTAL 次乘加，拆成 rounds = FMA_TOTAL/ILP 轮、每轮 ILP 条独立链。
// 用 __launch_bounds__(256, 1) 明确「每 SM 只放 1 个 block」，把 occupancy 钉死在
// 256/2048 = 12.5%，这样看到的差异只来自 ILP。
// ===========================================================================
constexpr int FMA_TOTAL = 8192;

template <int ILP>
__global__ void __launch_bounds__(256, 1) k_fma_ilp(float* __restrict__ out) {
  float a[ILP];
#pragma unroll
  for (int j = 0; j < ILP; ++j) a[j] = threadIdx.x * 1e-9f + j + 1.0f;
  const float m = 1.0000001f, c = 1e-7f;
  constexpr int ROUNDS = FMA_TOTAL / ILP;
  for (int r = 0; r < ROUNDS; ++r) {
#pragma unroll
    for (int j = 0; j < ILP; ++j) a[j] = fmaf(a[j], m, c);  // 链内依赖，链间独立
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < ILP; ++j) s += a[j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// ===========================================================================
// 通用 GEMM 计算体（第 09 篇 reg_vec 的模板化）
//
//   C[M,N] = A[M,K] * B[K,N]（fp32 行主序）
//   block 输出 BM×BN；每线程输出 TM×TN 个。
//   线程布局：threadIdx.x ∈ [0,BN/TN) 管列，threadIdx.y ∈ [0,BM/TM) 管行；
//   交错映射 row = ty + TY*i, col = tx + TX*j 以避免 smem bank conflict。
//   全局→共享用 float4 向量化，内层 k 循环完全展开。
// ===========================================================================
template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void gemm_body(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C, int M, int N, int K) {
  constexpr int TX = BN / TN, TY = BM / TM, T = TX * TY;
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int t = ty * TX + tx;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // 全局→共享：float4 向量化载入。当线程数恰好等于分块 float4 个数时（09 的配置），
    // 每个线程只需搬 1 个 float4，写成单发形式以获得与第 09 篇一致的代码生成。
    if constexpr (T == BM * BK / 4 && T == BK * BN / 4) {
      const int rA = t / (BK / 4), qA = t % (BK / 4);
      const int grA = block_row + rA, gcA = k0 + qA * 4;
      float4 vA = make_float4(0.f, 0.f, 0.f, 0.f);
      if (grA < M && gcA + 3 < K)
        vA = *reinterpret_cast<const float4*>(&A[(size_t)grA * K + gcA]);
      *reinterpret_cast<float4*>(&As[rA][qA * 4]) = vA;

      const int rB = t / (BN / 4), qB = t % (BN / 4);
      const int grB = k0 + rB, gcB = block_col + qB * 4;
      float4 vB = make_float4(0.f, 0.f, 0.f, 0.f);
      if (grB < K && gcB + 3 < N)
        vB = *reinterpret_cast<const float4*>(&B[(size_t)grB * N + gcB]);
      *reinterpret_cast<float4*>(&Bs[rB][qB * 4]) = vB;
    } else {
      for (int q = t; q < BM * BK / 4; q += T) {
        const int r = q / (BK / 4), c4 = q % (BK / 4);
        const int gr = block_row + r, gc = k0 + c4 * 4;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gr < M && gc + 3 < K) v = *reinterpret_cast<const float4*>(&A[(size_t)gr * K + gc]);
        *reinterpret_cast<float4*>(&As[r][c4 * 4]) = v;
      }
      for (int q = t; q < BK * BN / 4; q += T) {
        const int r = q / (BN / 4), c4 = q % (BN / 4);
        const int gr = k0 + r, gc = block_col + c4 * 4;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gr < K && gc + 3 < N) v = *reinterpret_cast<const float4*>(&B[(size_t)gr * N + gc]);
        *reinterpret_cast<float4*>(&Bs[r][c4 * 4]) = v;
      }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a[TM], b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = As[ty + TY * i][k];
#pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx + TX * j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = block_row + ty + TY * i;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int c = block_col + tx + TX * j;
      if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
    }
  }
}

// 不带 launch_bounds：编译器自由用寄存器
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_plain(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  gemm_body<BM, BN, BK, TM, TN>(A, B, C, M, N, K);
}

// 带 __launch_bounds__(T, MINB)：要求每 SM 至少驻留 MINB 个 block
template <int BM, int BN, int BK, int TM, int TN, int MINB>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), MINB)
    gemm_lb(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
            int M, int N, int K) {
  gemm_body<BM, BN, BK, TM, TN>(A, B, C, M, N, K);
}

// ---------------------------------------------------------------------------
// 打印编译期属性 + 运行时 occupancy
// ---------------------------------------------------------------------------
template <class Kernel>
void report_launch(const char* tag, Kernel k, int threads, const DeviceInfo& d) {
  cudaFuncAttributes a{};
  CUDA_CHECK(cudaFuncGetAttributes(&a, k));
  int blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k, threads, 0));
  std::printf("  %-16s regs=%-3d spill(local)=%-5zu smem=%-6zu | %d blk/SM  occ=%5.1f%%\n", tag,
              a.numRegs, a.localSizeBytes, a.sharedSizeBytes, blocks,
              100.0 * blocks * threads / d.max_threads_per_sm);
}

int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 2048;
  int N = (argc > 2) ? std::atoi(argv[2]) : 2048;
  int K = (argc > 3) ? std::atoi(argv[3]) : 2048;

  DeviceInfo d = device_info(0);
  print_device_info(d);

  // ======================= 实验 1：ILP × occupancy =======================
  // 每线程固定 FMA_TOTAL 次乘加，改变两点：①每线程几条独立链（ILP）②每 SM 驻留多少 block。
  // 结论预告：ILP 与 occupancy 是「隐藏延迟」的两条互补路径——任何一条够大就能打满管道。
  std::printf("\n=========== 实验 1：ILP × occupancy（fp32 FMA 链）=======\n");
  {
    const int threads = 256;
    const int max_blk = d.max_threads_per_sm / threads;  // 8
    float* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, (size_t)d.sms * max_blk * threads * sizeof(float)));

    auto run = [&](const char* tag, int blocks, auto kern) {
      const double flops = 2.0 * blocks * threads * FMA_TOTAL;
      const float ms = (float)bench_ms([&] { kern<<<blocks, threads>>>(out); }, 5, 50);
      const double tf = to_tflops(flops, ms);
      std::printf("  %-6s %-22s %8.2f TFLOPS  (%5.1f%% of fp32 peak)\n", tag,
                  blocks == d.sms ? "1 blk/SM (12.5%)" : "8 blk/SM (100%)", tf,
                  100.0 * tf / 66.9);
    };
    const int g1 = d.sms, g8 = d.sms * max_blk;
    run("ILP=1", g1, k_fma_ilp<1>);
    run("ILP=2", g1, k_fma_ilp<2>);
    run("ILP=4", g1, k_fma_ilp<4>);
    run("ILP=8", g1, k_fma_ilp<8>);
    run("ILP=1", g8, k_fma_ilp<1>);
    run("ILP=2", g8, k_fma_ilp<2>);
    run("ILP=4", g8, k_fma_ilp<4>);
    run("ILP=8", g8, k_fma_ilp<8>);
    CUDA_CHECK(cudaFree(out));
  }

  // ======================= GEMM 数据集 =======================
  const double flops = 2.0 * M * N * K;
  std::printf("\n=========== GEMM M=N=K=%d（%.2f GFLOP）===========\n", M, flops / 1e9);

  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, (size_t)M * K * 4));
  CUDA_CHECK(cudaMalloc(&B, (size_t)K * N * 4));
  CUDA_CHECK(cudaMalloc(&C, (size_t)M * N * 4));
  float* hA = (float*)std::malloc((size_t)M * K * 4);
  float* hB = (float*)std::malloc((size_t)K * N * 4);
  float* hC = (float*)std::malloc((size_t)M * N * 4);
  for (size_t i = 0; i < (size_t)M * K; ++i) hA[i] = 0.001f * (float)(i % 13);
  for (size_t i = 0; i < (size_t)K * N; ++i) hB[i] = 0.001f * (float)(i % 7);
  CUDA_CHECK(cudaMemcpy(A, hA, (size_t)M * K * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, hB, (size_t)K * N * 4, cudaMemcpyHostToDevice));

  auto cpu_entry = [&](int r, int c) {
    double s = 0;
    for (int k = 0; k < K; ++k) s += (double)hA[(size_t)r * K + k] * hB[(size_t)k * N + c];
    return s;
  };
  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaMemcpy(hC, C, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    double err = 0;
    for (int i = 0; i < 8; ++i) {
      int r = (i * 257) % M, c = (i * 131) % N;
      err = std::max(err, std::fabs((double)hC[(size_t)r * N + c] - cpu_entry(r, c)));
    }
    std::printf("  [%-18s] max_abs_err = %.3e %s\n", tag, err, err < 1e-2 ? "OK" : "FAIL");
  };

  auto report_perf = [&](const char* tag, double ms) {
    const double tflops = to_tflops(flops, ms);
    std::printf("%-22s %8.4f ms  %8.2f TFLOPS  (%5.1f%% of fp32 peak)\n", tag, ms, tflops,
                100.0 * tflops / 66.9);
  };

  // ============ 实验 2：__launch_bounds__ 扫描（固定 128×128 / 8×8）============
  std::printf("\n---- 实验 2：对同一 GEMM 配置强行提高驻留 block 数 ----\n");
  {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    constexpr int T = (BM / TM) * (BN / TN);  // 256
    const dim3 block(BM / TM, BN / TN);
    const dim3 grid(div_up(N, BN), div_up(M, BM));

    report_launch("plain", gemm_plain<BM, BN, BK, TM, TN>, T, d);
    report_launch("MINB=1", gemm_lb<BM, BN, BK, TM, TN, 1>, T, d);
    report_launch("MINB=2", gemm_lb<BM, BN, BK, TM, TN, 2>, T, d);
    report_launch("MINB=3", gemm_lb<BM, BN, BK, TM, TN, 3>, T, d);
    report_launch("MINB=4", gemm_lb<BM, BN, BK, TM, TN, 4>, T, d);

    auto bench_one = [&](const char* tag, auto kern) {
      kern<<<grid, block>>>(A, B, C, M, N, K);
      CUDA_CHECK_LAST();
      check(tag);
      const double ms = bench_ms([&] { kern<<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
      report_perf(tag, ms);
    };
    bench_one("plain", gemm_plain<BM, BN, BK, TM, TN>);
    bench_one("lb<1>", gemm_lb<BM, BN, BK, TM, TN, 1>);
    bench_one("lb<2>", gemm_lb<BM, BN, BK, TM, TN, 2>);
    bench_one("lb<3>", gemm_lb<BM, BN, BK, TM, TN, 3>);
    bench_one("lb<4>", gemm_lb<BM, BN, BK, TM, TN, 4>);
  }

  // ============ 实验 3：thread tile 扫描（复用 vs occupancy）============
  // 每线程多算几个输出 = smem 载入被多复用几次 + 攒出更多 ILP，代价是寄存器多、occupancy 低。
  std::printf("\n---- 实验 3：thread tile 扫描（同 block 输出规模，不同每线程复用度）----\n");
  {
    // (BM,BN,BK,TM,TN) 四组：从「小 tile 高 occupancy」到「大 tile 低 occupancy」
    report_launch("64x64 t2x2", gemm_plain<64, 64, 8, 2, 2>, 32 * 32, d);
    report_launch("64x64 t4x4", gemm_plain<64, 64, 8, 4, 4>, 16 * 16, d);
    report_launch("64x64 t8x8", gemm_plain<64, 64, 8, 8, 8>, 8 * 8, d);
    report_launch("128x128 t8x8", gemm_plain<128, 128, 8, 8, 8>, 16 * 16, d);

    auto bench_tile = [&](const char* tag, int bm, int bn, auto kern, dim3 block) {
      const dim3 grid(div_up(N, bn), div_up(M, bm));
      kern<<<grid, block>>>(A, B, C, M, N, K);
      CUDA_CHECK_LAST();
      check(tag);
      const double ms = bench_ms([&] { kern<<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
      report_perf(tag, ms);
    };
    bench_tile("64x64 t2x2", 64, 64, gemm_plain<64, 64, 8, 2, 2>, dim3(32, 32));
    bench_tile("64x64 t4x4", 64, 64, gemm_plain<64, 64, 8, 4, 4>, dim3(16, 16));
    bench_tile("64x64 t8x8", 64, 64, gemm_plain<64, 64, 8, 8, 8>, dim3(8, 8));
    bench_tile("128x128 t8x8", 128, 128, gemm_plain<128, 128, 8, 8, 8>, dim3(16, 16));
  }

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  std::free(hA);
  std::free(hB);
  std::free(hC);
  return 0;
}
