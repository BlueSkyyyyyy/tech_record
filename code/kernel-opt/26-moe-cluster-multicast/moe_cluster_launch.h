// 26 MoE cluster multicast 的 host 侧：真实 MoE 分布、tensor map、扫 cluster 尺寸并计时。
// 被 moe_cluster.cu 在 kernel 定义之后 #include。
#pragma once

#include <cstdint>
#include <vector>

struct MoeDist {
  int G, K, N;
  std::vector<int> actual, aligned, row_off;
  int M_total = 0;
  int sum_actual = 0;
  std::vector<int> gl;
};

static MoeDist make_dist(int G, int expected, int align) {
  MoeDist d;
  d.G = G;
  d.actual.resize(G);
  d.aligned.resize(G);
  d.row_off.resize(G);
  int off = 0;
  for (int g = 0; g < G; ++g) {
    int a = (int)(expected * (0.7 + 0.6 * ((double)rand() / RAND_MAX)));
    if (a < 1) a = 1;
    int al = (a + align - 1) / align * align;
    d.actual[g] = a;
    d.aligned[g] = al;
    d.row_off[g] = off;
    off += al;
    d.sum_actual += a;
  }
  d.M_total = off;
  d.gl.assign(off, -1);
  for (int g = 0; g < G; ++g)
    for (int r = 0; r < d.actual[g]; ++r) d.gl[d.row_off[g] + r] = g;
  return d;
}

__global__ void fill_fp8_kernel(__nv_fp8_e4m3* p, size_t n, unsigned seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned x = (unsigned)(i ^ (i >> 32)) ^ (seed * 0x9E3779B9u);
    x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
    float f = 2.f * ((float)((x >> 8) & 0xFFFF) / 65535.f - 0.5f);
    p[i] = __nv_fp8_e4m3(f);
  }
}

template <int BM, int BN, int BK, int STAGES, bool MASKED, int CN>
static void launch_moe(const CUtensorMap& tmA, const CUtensorMap& tmB, float* D, const int* gl,
                       int N, int K, int max_m, int mlim, float scale, int dbase, int grid_x,
                       int grid_y, int grid_z) {
  auto fn = moe_kernel<BM, BN, BK, STAGES, MASKED, CN>;
  const int nt = (BM / 64) * 128 + 32;
  const int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  const size_t shm = (size_t)BARR + (size_t)STAGES * (BM + BN) * BK;
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
  dim3 grid(grid_x, grid_y, grid_z);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(nt);
  cfg.dynamicSmemBytes = shm;
  cfg.stream = 0;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CN;  // A 广播沿 x（N 方向）
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaError_t err = cudaLaunchKernelEx(&cfg, fn, tmA, tmB, D, gl, N, K, max_m, mlim, scale, dbase);
  cuda_check(err, "cudaLaunchKernelEx", __FILE__, __LINE__);
}

int moe_main(int argc, char** argv) {
  const char* which = (argc > 1) ? argv[1] : "all";
  int tokens = (argc > 2) ? std::atoi(argv[2]) : 16384;
  const int K = 7168, N = 3072, G = 384, topk = 6, ALIGN = 128;
  const int max_m = 128;

  DeviceInfo d0 = device_info(0);
  print_device_info(d0);

  srand(1234);
  const int expected = tokens * topk / G;
  MoeDist md = make_dist(G, expected, ALIGN);
  md.K = K;
  md.N = N;

  std::printf("\n=== MoE grouped GEMM + TMA cluster multicast (fp8 e4m3, per-tensor) ===\n");
  std::printf("config: tokens=%d topk=%d G=%d  K=%d N=%d\n", tokens, topk, G, K, N);
  std::printf("prefill(contiguous): expected_m/expert=%d  M_total=%d (aligned)  sum_actual=%d\n",
              expected, md.M_total, md.sum_actual);
  std::printf("  padding waste = %.1f%%\n", 100.0 * (md.M_total - md.sum_actual) / md.M_total);
  const double avg_mtiles = (double)md.M_total / G / 128.0;
  std::printf("  avg m-tiles/expert = %.2f  -> B L2 read amp ~%.2fx; A read amp = N/BN\n",
              avg_mtiles, avg_mtiles);

  const size_t aN = (size_t)md.M_total * K, bN = (size_t)G * N * K, cN = (size_t)md.M_total * N;
  fp8* A; fp8* B; float* D; int* gl;
  CUDA_CHECK(cudaMalloc(&A, aN));
  CUDA_CHECK(cudaMalloc(&B, bN));
  CUDA_CHECK(cudaMalloc(&D, cN * 4));
  CUDA_CHECK(cudaMalloc(&gl, (size_t)md.M_total * sizeof(int)));

  fill_fp8_kernel<<<1024, 256>>>(A, aN, 11u);
  fill_fp8_kernel<<<1024, 256>>>(B, bN, 22u);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(gl, md.gl.data(), (size_t)md.M_total * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(D, 0, cN * 4));
  const float scale = 0.8f * 0.9f;

  auto check_contig = [&](const char* tag) -> bool {
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<fp8> ha(K), hb(K);
    double err = 0, ref = 0;
    for (int s = 0; s < 12; ++s) {
      int g = (s * 47 + 3) % G;
      int row = md.row_off[g] + (s * 7) % md.actual[g];
      int col = (s * 997 + 7) % N;
      CUDA_CHECK(cudaMemcpy(ha.data(), A + (size_t)row * K, K, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hb.data(), B + ((size_t)g * N + col) * K, K, cudaMemcpyDeviceToHost));
      double e = 0;
      for (int k = 0; k < K; ++k) e += (double)(float)ha[k] * (double)(float)hb[k];
      e *= scale;
      float got;
      CUDA_CHECK(cudaMemcpy(&got, D + (size_t)row * N + col, 4, cudaMemcpyDeviceToHost));
      err = std::max(err, std::fabs((double)got - e));
      ref = std::max(ref, std::fabs(e));
    }
    bool ok = err / std::max(ref, 1.0) < 3e-2;
    std::printf("  [%-18s] max_abs_err=%.3e (ref~%.1f) %s\n", tag, err, ref, ok ? "OK" : "FAIL");
    return ok;
  };

  auto report = [&](const char* tag, double ms, double flops_used, const char* unit) {
    double tf = to_tflops(flops_used, ms);
    double bw = to_gbps((double)bN, ms);
    std::printf("%-20s %8.4f ms  %8.2f TFLOPS/%-7s (%5.1f%% peak)  B-read %6.1f GB/s\n",
                tag, ms, tf, unit, 100.0 * tf / FP8_PEAK, bw);
  };

  const double flops_aligned = 2.0 * md.M_total * N * K;
  auto want = [&](const char* n) { return std::strcmp(which, "all") == 0 || std::strcmp(which, n) == 0; };

  // ---- A) per-expert loop 基线 ----
  if (want("loop")) {
    std::printf("\n-- A) per-expert loop (G=%d 次 launch) --\n", G);
    std::vector<CUtensorMap> tmA(G), tmB(G);
    for (int g = 0; g < G; ++g) {
      tmA[g] = make_tmap_2d(A + (size_t)md.row_off[g] * K, K, md.aligned[g], 128, 128);
      tmB[g] = make_tmap_2d(B + (size_t)g * N * K, K, N, 128, 128);
    }
    auto run = [&] {
      for (int g = 0; g < G; ++g)
        launch_moe<128, 128, 128, 3, false, 1>(tmA[g], tmB[g], D, nullptr, N, K, 0,
                                               md.row_off[g] + md.actual[g], scale, md.row_off[g],
                                               N / 128, md.aligned[g] / 128, 1);
    };
    run();
    check_contig("loop");
    double t = bench_ms(run, 3, 15);
    report("loop(G launches)", t, flops_aligned, "aligned");
  }

  // ---- B) grouped + cluster multicast ----
  {
    CUtensorMap tmA = make_tmap_2d(A, K, md.M_total, 128, 128);
    std::printf("\n-- B) grouped contiguous + A-multicast (cluster.x = CN) --\n");
    std::printf("%-20s %8s  %8s %s\n", "config", "ms", "TFLOPS", "(vs CN=1)");
#define RUN_C(NAME, BN, ST, CN)                                                              \
    if (want(NAME)) {                                                                        \
      CUtensorMap tmB = make_tmap_2d(B, K, (uint64_t)G * N, 128, BN);                        \
      auto run = [&] {                                                                       \
        launch_moe<128, BN, 128, ST, false, CN>(tmA, tmB, D, gl, N, K, 0, md.M_total, scale, \
                                                0, N / BN, md.M_total / 128, 1);              \
      };                                                                                     \
      run();                                                                                 \
      check_contig(NAME);                                                                    \
      double t = bench_ms(run, 3, 20);                                                       \
      report(NAME, t, flops_aligned, "aligned");                                             \
    }
    RUN_C("grp128x128s3_c1", 128, 3, 1);
    RUN_C("grp128x128s3_c2", 128, 3, 2);
    RUN_C("grp128x128s3_c4", 128, 3, 4);
    RUN_C("grp128x256s3_c1", 256, 3, 1);
    RUN_C("grp128x256s3_c2", 256, 3, 2);
    RUN_C("grp128x256s3_c4", 256, 3, 4);
    RUN_C("grp128x128s4_c1", 128, 4, 1);
    RUN_C("grp128x128s4_c2", 128, 4, 2);
    RUN_C("grp128x128s4_c4", 128, 4, 4);
#undef RUN_C
  }

  // ---- C) masked decode + cluster ----
  if (want("masked")) {
    const int dec_expected = 32;
    srand(77);
    MoeDist mdm = make_dist(G, dec_expected, 1);
    mdm.M_total = G * max_m;
    std::printf("\n-- C) grouped masked (decode): G=%d max_m=%d expected_m=%d --\n", G, max_m,
                dec_expected);
    fp8* Am; float* Dm;
    CUDA_CHECK(cudaMalloc(&Am, (size_t)G * max_m * K));
    CUDA_CHECK(cudaMalloc(&Dm, (size_t)G * max_m * N * 4));
    CUDA_CHECK(cudaMemset(Am, 0, (size_t)G * max_m * K));
    fill_fp8_kernel<<<1024, 256>>>(Am, (size_t)G * max_m * K, 33u);
    CUDA_CHECK_LAST();
    int* mm;
    CUDA_CHECK(cudaMalloc(&mm, G * sizeof(int)));
    std::vector<int> host_mm(G);
    for (int g = 0; g < G; ++g) host_mm[g] = std::min(mdm.actual[g], max_m);
    CUDA_CHECK(cudaMemcpy(mm, host_mm.data(), G * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(Dm, 0, (size_t)G * max_m * N * 4));

    CUtensorMap tmA = make_tmap_2d(Am, K, (uint64_t)G * max_m, 128, 128);
    auto check_mask = [&] {
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<fp8> ha(K), hb(K);
      double err = 0, ref = 0;
      for (int s = 0; s < 12; ++s) {
        int g = (s * 47 + 5) % G;
        int row = (s * 7) % std::min(mdm.actual[g], max_m);
        int col = (s * 997 + 11) % N;
        CUDA_CHECK(cudaMemcpy(ha.data(), Am + ((size_t)g * max_m + row) * K, K, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb.data(), B + ((size_t)g * N + col) * K, K, cudaMemcpyDeviceToHost));
        double e = 0;
        for (int k = 0; k < K; ++k) e += (double)(float)ha[k] * (double)(float)hb[k];
        e *= scale;
        float got;
        CUDA_CHECK(cudaMemcpy(&got, Dm + ((size_t)g * max_m + row) * N + col, 4, cudaMemcpyDeviceToHost));
        err = std::max(err, std::fabs((double)got - e));
        ref = std::max(ref, std::fabs(e));
      }
      bool ok = err / std::max(ref, 1.0) < 3e-2;
      std::printf("  [%-18s] max_abs_err=%.3e (ref~%.1f) %s\n", "masked", err, ref, ok ? "OK" : "FAIL");
      return ok;
    };
#define RUN_M(NAME, BN, ST, CN)                                                                \
    if (want(NAME) || want("all")) {                                                           \
      CUtensorMap tmB = make_tmap_2d(B, K, (uint64_t)G * N, 128, BN);                          \
      auto run = [&] {                                                                         \
        launch_moe<128, BN, 128, ST, true, CN>(tmA, tmB, Dm, mm, N, K, max_m, 0, scale, 0,     \
                                               N / BN, max_m / 128, G);                        \
      };                                                                                       \
      run();                                                                                   \
      check_mask();                                                                            \
      double t = bench_ms(run, 3, 20);                                                         \
      report(NAME, t, 2.0 * mdm.sum_actual * N * K, "useful");                                 \
    }
    RUN_M("masked128x256s3_c1", 256, 3, 1);
    RUN_M("masked128x256s3_c2", 256, 3, 2);
    RUN_M("masked128x256s3_c4", 256, 3, 4);
#undef RUN_M
    CUDA_CHECK(cudaFree(Am));
    CUDA_CHECK(cudaFree(Dm));
    CUDA_CHECK(cudaFree(mm));
  }

  // ---- D) 更大 M-tile（BM=256）消 B 的 L2 放大：单独用 ALIGN=256 的分布 ----
  if (want("bigm") || want("all")) {
    srand(2468);
    MoeDist md2 = make_dist(G, expected, 256);  // 组对齐到 256，BM=256 tile 不跨 expert
    md2.K = K; md2.N = N;
    const size_t a2N = (size_t)md2.M_total * K, c2N = (size_t)md2.M_total * N;
    fp8* A2; float* D2; int* gl2;
    CUDA_CHECK(cudaMalloc(&A2, a2N));
    CUDA_CHECK(cudaMalloc(&D2, c2N * 4));
    CUDA_CHECK(cudaMalloc(&gl2, (size_t)md2.M_total * sizeof(int)));
    fill_fp8_kernel<<<1024, 256>>>(A2, a2N, 44u);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(gl2, md2.gl.data(), (size_t)md2.M_total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(D2, 0, c2N * 4));
    const double flops2_useful = 2.0 * md2.sum_actual * N * K;
    std::printf("\n-- D) 大 M-tile（ALIGN=256）: M_total=%d sum_actual=%d waste=%.1f%% --\n",
                md2.M_total, md2.sum_actual, 100.0 * (md2.M_total - md2.sum_actual) / md2.M_total);
    auto check_d = [&](const char* tag) {
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<fp8> ha(K), hb(K);
      double err = 0, ref = 0;
      for (int s = 0; s < 12; ++s) {
        int g = (s * 47 + 3) % G;
        int row = md2.row_off[g] + (s * 7) % md2.actual[g];
        int col = (s * 997 + 7) % N;
        CUDA_CHECK(cudaMemcpy(ha.data(), A2 + (size_t)row * K, K, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb.data(), B + ((size_t)g * N + col) * K, K, cudaMemcpyDeviceToHost));
        double e = 0;
        for (int k = 0; k < K; ++k) e += (double)(float)ha[k] * (double)(float)hb[k];
        e *= scale;
        float got;
        CUDA_CHECK(cudaMemcpy(&got, D2 + (size_t)row * N + col, 4, cudaMemcpyDeviceToHost));
        err = std::max(err, std::fabs((double)got - e));
        ref = std::max(ref, std::fabs(e));
      }
      std::printf("  [%-18s] max_abs_err=%.3e (ref~%.1f) %s\n", tag, err, ref,
                  err / std::max(ref, 1.0) < 3e-2 ? "OK" : "FAIL");
    };
#define RUN_D(NAME, BM, BN, ST, CN)                                                            \
    {                                                                                          \
      CUtensorMap tmA = make_tmap_2d(A2, K, md2.M_total, 128, BM);                             \
      CUtensorMap tmB = make_tmap_2d(B, K, (uint64_t)G * N, 128, BN);                          \
      auto run = [&] {                                                                         \
        launch_moe<BM, BN, 128, ST, false, CN>(tmA, tmB, D2, gl2, N, K, 0, md2.M_total, scale, \
                                               0, N / BN, md2.M_total / BM, 1);                \
      };                                                                                       \
      run();                                                                                   \
      check_d(NAME);                                                                           \
      double t = bench_ms(run, 3, 20);                                                         \
      report(NAME, t, flops2_useful, "useful");                                                \
    }
    RUN_D("d128x256s3_c1", 128, 256, 3, 1);
    RUN_D("d256x128s3_c1", 256, 128, 3, 1);
    RUN_D("d256x128s3_c2", 256, 128, 3, 2);
    RUN_D("d256x128s4_c1", 256, 128, 4, 1);
    RUN_D("d256x256s2_c1", 256, 256, 2, 1);
    RUN_D("d256x256s3_c1", 256, 256, 3, 1);
    RUN_D("d256x256s3_c2", 256, 256, 3, 2);
#undef RUN_D
    CUDA_CHECK(cudaFree(A2));
    CUDA_CHECK(cudaFree(D2));
    CUDA_CHECK(cudaFree(gl2));
  }

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(D));
  CUDA_CHECK(cudaFree(gl));
  return 0;
}
