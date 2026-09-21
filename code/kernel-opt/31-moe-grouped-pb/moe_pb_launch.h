// 31 MoE grouped per-block FP8 GEMM 的 host 侧：真实 MoE 分布 + per-block scale + 对拍计时。
// 被 moe_pb.cu 在 kernel 定义之后 #include。
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

// per-block scale：在 [0.5, 1.5] 之间随机（逐行 / 逐块不同，暴露相邻列 scale 不同的问题）
__global__ void fill_scale_kernel(float* p, size_t n, unsigned seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned x = (unsigned)(i ^ (i >> 32)) ^ (seed * 0x9E3779B9u);
    x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
    p[i] = 0.5f + ((x >> 8) & 0xFFFF) / 65535.f;
  }
}

template <int BM, int BN, int BK, int STAGES, bool MASKED, bool PERBLOCK, int MINB = 1>
static void launch_moe_pb(const CUtensorMap& tmA, const CUtensorMap& tmB, float* D, const int* gl,
                          int N, int K, int max_m, int mlim, float scale, const float* sa,
                          const float* sb, int KBLK, int dbase, int grid_x, int grid_y, int grid_z) {
  auto fn = moe_pb_kernel<BM, BN, BK, STAGES, MASKED, PERBLOCK, MINB>;
  const int nt = (BM / 64) * 128 + 32;
  const int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  const size_t shm = (size_t)BARR + (size_t)STAGES * (BM + BN) * BK;
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
  dim3 grid(grid_x, grid_y, grid_z);
  fn<<<grid, nt, shm>>>(tmA, tmB, D, gl, N, K, max_m, mlim, scale, sa, sb, KBLK, dbase);
  CUDA_CHECK_LAST();
}

int moe_pb_main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* which = (argc > 1) ? argv[1] : "all";
  int tokens = (argc > 2) ? std::atoi(argv[2]) : 16384;
  const int K = 7168, N = 3072, G = 384, topk = 6;
  const int max_m = 128;
  const int KBLK = K / 128;  // 56 个 128-k 块

  DeviceInfo d0 = device_info(0);
  print_device_info(d0);

  srand(1234);
  const int expected = tokens * topk / G;
  MoeDist md = make_dist(G, expected, 128);
  md.K = K; md.N = N;

  std::printf("\n=== MoE grouped GEMM (fp8 e4m3, per-block 128x128 scale) ===\n");
  std::printf("config: tokens=%d topk=%d G=%d  K=%d N=%d  KBLK=%d\n", tokens, topk, G, K, N, KBLK);
  std::printf("prefill(contiguous): expected_m/expert=%d  M_total=%d (aligned)  sum_actual=%d\n",
              expected, md.M_total, md.sum_actual);
  std::printf("  padding waste = %.1f%%\n", 100.0 * (md.M_total - md.sum_actual) / md.M_total);

  const size_t aN = (size_t)md.M_total * K, bN = (size_t)G * N * K, cN = (size_t)md.M_total * N;
  fp8 *A, *B;
  float* D;
  int* gl;
  CUDA_CHECK(cudaMalloc(&A, aN));
  CUDA_CHECK(cudaMalloc(&B, bN));
  CUDA_CHECK(cudaMalloc(&D, cN * 4));
  CUDA_CHECK(cudaMalloc(&gl, (size_t)md.M_total * sizeof(int)));
  fill_fp8_kernel<<<1024, 256>>>(A, aN, 11u);
  fill_fp8_kernel<<<1024, 256>>>(B, bN, 22u);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(gl, md.gl.data(), (size_t)md.M_total * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(D, 0, cN * 4));

  // per-block scale
  const size_t saN = (size_t)md.M_total * KBLK;          // contiguous 激活
  const size_t sbN = (size_t)G * (N / 128) * KBLK;       // 权重
  float *sa, *sb;
  CUDA_CHECK(cudaMalloc(&sa, saN * 4));
  CUDA_CHECK(cudaMalloc(&sb, sbN * 4));
  fill_scale_kernel<<<1024, 256>>>(sa, saN, 55u);
  fill_scale_kernel<<<1024, 256>>>(sb, sbN, 66u);
  CUDA_CHECK_LAST();
  const float scale = 0.8f * 0.9f;

  // 参考：逐 128-k 块折算。对 5 个不同列取样，报最大相对误差（避免单点随机抵消）。
  auto check = [&](const char* tag, float* Dp, fp8* Ap, int a_row_stride, int row, int arow_abs,
                   int g, int col0, bool pb = true, const float* sa_ptr = nullptr) {
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<fp8> ha(K), hb(K);
    std::vector<float> hsa(KBLK), hsb(KBLK);
    CUDA_CHECK(cudaMemcpy(ha.data(), Ap + (size_t)arow_abs * K, K, cudaMemcpyDeviceToHost));
    if (pb) {
      if (!sa_ptr) sa_ptr = sa;
      CUDA_CHECK(cudaMemcpy(hsa.data(), sa_ptr + (size_t)arow_abs * KBLK, KBLK * 4,
                            cudaMemcpyDeviceToHost));
    }
    double max_err = 0, max_ref = 0, max_rel = 0;
    for (int s = 0; s < 5; ++s) {
      int col = (col0 + s * 613) % N;
      CUDA_CHECK(cudaMemcpy(hb.data(), B + ((size_t)g * N + col) * K, K, cudaMemcpyDeviceToHost));
      double e = 0;
      if (pb) {
        CUDA_CHECK(cudaMemcpy(hsb.data(), sb + ((size_t)g * (N / 128) + col / 128) * KBLK, KBLK * 4,
                              cudaMemcpyDeviceToHost));
        for (int kb = 0; kb < KBLK; ++kb) {
          double blk = 0;
          for (int k = kb * 128; k < (kb + 1) * 128; ++k)
            blk += (double)(float)ha[k] * (double)(float)hb[k];
          e += blk * (double)hsa[kb] * (double)hsb[kb];
        }
      } else {
        for (int k = 0; k < K; ++k) e += (double)(float)ha[k] * (double)(float)hb[k];
      }
      e *= scale;
      float got;
      CUDA_CHECK(cudaMemcpy(&got, Dp + (size_t)row * N + col, 4, cudaMemcpyDeviceToHost));
      double rel = std::fabs((double)got - e) / std::max(std::fabs(e), 1.0);
      if (std::fabs(e) > 3.0) {
        if (std::fabs(e) > max_ref) { max_ref = std::fabs(e); max_err = std::fabs((double)got - e); }
        max_rel = std::max(max_rel, rel);
      }
      if ((getenv("MOE_DEBUG") || rel >= 3e-2) && s == 0)
        std::printf("      got=%.6f ref=%.6f row=%d col=%d g=%d arow=%d\n", got, e, row, col, g,
                    arow_abs);
    }
    bool ok = max_rel < 3e-2;
    std::printf("  [%-18s] max_rel_err=%.3e (ref~%.1f) %s\n", tag, max_rel, max_ref,
                ok ? "OK" : "FAIL");
    (void)a_row_stride;
    (void)max_err;
    return ok;
  };

  auto report = [&](const char* tag, double ms, double flops) {
    double tf = to_tflops(flops, ms);
    std::printf("%-20s %8.4f ms  %8.2f TFLOPS (%5.1f%% of fp8 peak)  B-read %6.1f GB/s\n", tag, ms,
                tf, 100.0 * tf / FP8_PEAK, to_gbps((double)bN, ms));
  };

  const double flops_aligned = 2.0 * md.M_total * N * K;
  auto want = [&](const char* n) { return std::strcmp(which, "all") == 0 || std::strcmp(which, n) == 0; };

  // ---- B) grouped contiguous：per-block 扫参 ----
  {
    std::printf("\n-- B) grouped contiguous · per-block 128x128（单 launch，B 行坐标=group*N+n）--\n");
#define RUN_PB(NAME, BM, BN, ST, MINB)                                                        \
    if (want(NAME)) {                                                                         \
      CUtensorMap tmA = make_tmap_2d(A, K, md.M_total, 128, BM);                              \
      CUtensorMap tmB = make_tmap_2d(B, K, (uint64_t)G * N, 128, BN);                         \
      CUDA_CHECK(cudaMemset(D, 0, cN * 4));                                                   \
      auto run = [&] {                                                                        \
        launch_moe_pb<BM, BN, 128, ST, false, true, MINB>(tmA, tmB, D, gl, N, K, 0, md.M_total, \
                                                          scale, sa, sb, KBLK, 0, N / BN,     \
                                                          md.M_total / BM, 1);                \
      };                                                                                      \
      run();                                                                                  \
      check(NAME, D, A, K, md.row_off[3] + 5, md.row_off[3] + 5, md.gl[md.row_off[3] + 5],    \
            (3 * 997 + 11) % N);                                                              \
      double t = bench_ms(run, 3, 20);                                                        \
      report(NAME, t, flops_aligned);                                                         \
    }
    RUN_PB("pb128x128s2", 128, 128, 2, 1);
    RUN_PB("pb128x128s3", 128, 128, 3, 1);
    RUN_PB("pb128x128s4", 128, 128, 4, 1);
    RUN_PB("pb64x128s3", 64, 128, 3, 1);
    RUN_PB("pb64x128s4", 64, 128, 4, 1);
    RUN_PB("pb64x128s3x2", 64, 128, 3, 2);
    RUN_PB("pb64x128s4x2", 64, 128, 4, 2);
#undef RUN_PB
  }

  // ---- B2) 同一二进制对照：PERBLOCK=false（不折算，纯 per-tensor）----
  {
    std::printf("\n-- B2) 同二进制 per-tensor 对照（PERBLOCK=false，不折算）--\n");
#define RUN_PT(NAME, BM, BN, ST)                                                              \
    if (want(NAME) || want("pt")) {                                                           \
      CUtensorMap tmA = make_tmap_2d(A, K, md.M_total, 128, BM);                              \
      CUtensorMap tmB = make_tmap_2d(B, K, (uint64_t)G * N, 128, BN);                         \
      CUDA_CHECK(cudaMemset(D, 0, cN * 4));                                                   \
      auto run = [&] {                                                                        \
        launch_moe_pb<BM, BN, 128, ST, false, false>(tmA, tmB, D, gl, N, K, 0, md.M_total,    \
                                                     scale, nullptr, nullptr, KBLK, 0, N / BN, \
                                                     md.M_total / BM, 1);                     \
      };                                                                                      \
      run();                                                                                  \
      check(NAME, D, A, K, md.row_off[3] + 5, md.row_off[3] + 5, md.gl[md.row_off[3] + 5],    \
            (3 * 997 + 11) % N, false);                                                              \
      double t = bench_ms(run, 3, 20);                                                        \
      report(NAME, t, flops_aligned);                                                         \
    }
    RUN_PT("pt128x128s2", 128, 128, 2);
    RUN_PT("pt128x128s3", 128, 128, 3);
    RUN_PT("pt128x128s4", 128, 128, 4);
    RUN_PT("pt64x128s3", 64, 128, 3);
#undef RUN_PT
  }

  // ---- C) masked grouped（decode，纯权重带宽）：per-block ----
  if (std::strncmp(which, "masked", 6) == 0 || want("all")) {
    const int dec_expected = 32;
    srand(77);
    MoeDist mdm = make_dist(G, dec_expected, 1);
    std::printf("\n-- C) grouped masked (decode): G=%d max_m=%d expected_m=%d --\n", G, max_m,
                dec_expected);
    fp8* Am; float* Dm;
    CUDA_CHECK(cudaMalloc(&Am, (size_t)G * max_m * K));
    CUDA_CHECK(cudaMalloc(&Dm, (size_t)G * max_m * N * 4));
    CUDA_CHECK(cudaMemset(Am, 0, (size_t)G * max_m * K));
    fill_fp8_kernel<<<1024, 256>>>(Am, (size_t)G * max_m * K, 33u);
    CUDA_CHECK_LAST();
    const size_t samN = (size_t)G * max_m * KBLK;
    float* sam;
    CUDA_CHECK(cudaMalloc(&sam, samN * 4));
    fill_scale_kernel<<<1024, 256>>>(sam, samN, 88u);
    CUDA_CHECK_LAST();
    int* mm;
    CUDA_CHECK(cudaMalloc(&mm, G * sizeof(int)));
    std::vector<int> host_mm(G);
    for (int g = 0; g < G; ++g) host_mm[g] = std::min(mdm.actual[g], max_m);
    CUDA_CHECK(cudaMemcpy(mm, host_mm.data(), G * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(Dm, 0, (size_t)G * max_m * N * 4));

#define RUN_MASK(NAME, BM, BN, ST)                                                            \
    if (want(NAME) || want("all")) {                                                          \
      CUtensorMap tmA = make_tmap_2d(Am, K, (uint64_t)G * max_m, 128, BM);                    \
      CUtensorMap tmB = make_tmap_2d(B, K, (uint64_t)G * N, 128, BN);                         \
      CUDA_CHECK(cudaMemset(Dm, 0, (size_t)G * max_m * N * 4));                               \
      auto run = [&] {                                                                        \
        launch_moe_pb<BM, BN, 128, ST, true, true>(tmA, tmB, Dm, mm, N, K, max_m, 0, scale,   \
                                                   sam, sb, KBLK, 0, N / BN, max_m / BM, G);  \
      };                                                                                      \
      run();                                                                                  \
      check(NAME, Dm, Am, K, 3 * max_m + 5, 3 * max_m + 5, 3, (5 * 997 + 11) % N, true, sam);            \
      double t = bench_ms(run, 3, 20);                                                        \
      report(NAME, t, 2.0 * mdm.sum_actual * N * K);                                          \
    }
    RUN_MASK("maskedPB128x128s3", 128, 128, 3);
    RUN_MASK("maskedPB64x128s3", 64, 128, 3);
#undef RUN_MASK
    CUDA_CHECK(cudaFree(Am));
    CUDA_CHECK(cudaFree(Dm));
    CUDA_CHECK(cudaFree(sam));
    CUDA_CHECK(cudaFree(mm));
  }

  CUDA_CHECK(cudaFree(sa));
  CUDA_CHECK(cudaFree(sb));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(D));
  CUDA_CHECK(cudaFree(gl));
  return 0;
}
