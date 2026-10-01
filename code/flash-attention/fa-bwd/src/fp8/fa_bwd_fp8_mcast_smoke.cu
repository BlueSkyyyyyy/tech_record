// =============================================================================
// fa_bwd_fp8_mcast_smoke.cu —— O118（第 212 轮）：fp8 主 kernel 候选② 的 de-risk
// =============================================================================
// 动机：ROADMAP「下一步候选 ②」——fp8 默认 main 的 L2 里，`red`（dK/dV 跨 CTA 归约）已被
//   F3b/F4b/F6/F7/O90–O95/O114/O117 全部收口为「本卡工作划分的硬件下界」（无软件解）。
//   唯一还没试过的**搬运量杠杆**是 **TMA cluster multicast**：同 cluster 内的若干 CTA
//   若读同一块 K/V，则由 leader 发一条 `...multicast::cluster` 广播到全 cluster，把 K/V
//   的 L2 读扇区按 cluster 大小摊薄（`read` 2.38× 差距里可动的那一半）。
//
// 本冒烟在**与主 kernel 完全相同的 K/V 4D-TMA 几何**（UINT8 / dims={D,S,Hkv,B} /
//   SWIZZLE_128B / box={128,32} ⇒ KS_SZ=4096B，即 O41 的 `kvtma` K/V 搬运）上验证：
//   ① 正确性：multicast 送进每个 CTA smem 的 SW128 tile 与「每 CTA 各自发 TMA」逐字节
//      相同，且与 host 端 `sw128_off_fp8` 手工 swizzle 的参考逐字节相同；
//   ② 搬运量：ncu `lts__t_sectors_op_read`（L2 读扇区）随 cluster 大小 1/2/4 的下降；
//   ③ 性能：event 计时（consume 工作量固定，只看 K/V 读被摊薄后的墙）。
//
// 协议与 kernel-opt 26/36 篇一致：leader（rank0）发 `...multicast::cluster` + mask=(1<<CN)-1；
//   每个 CTA（含 leader）各自 `mbarrier.arrive.expect_tx` 并等**自己本地**的 mbarrier；
//   smem 目标地址是 leader 的本地 shared 地址，硬件按同一偏移复制进 mask 内各 CTA。
//   init 后必须 `fence.mbarrier_init.release.cluster` + `barrier.cluster` 才能被远端 arrive。
//
// 编译：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//        scripts/run.sh src/fp8/fa_bwd_fp8_mcast_smoke.cu [CN]
// =============================================================================

#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, \
              __LINE__);                                                        \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// ---- 与主 kernel 一致的几何（O41 kvtma K/V 搬运）----
constexpr int D = 128;        // head_dim（fp8 一行 = 128B = SW128 atom 整行）
constexpr int BN = 32;        // 每个 K/V tile 的行数（= boxR）
constexpr int KS_SZ = (BN / 8) * 1024;  // 4096B（[32 行][128 列] SW128）
constexpr int THREADS = 128;
constexpr int STAGES = 2;

// SW128 K-major（与 src/fp8/fa_bwd_fp8_kernels.cuh 的 sw128_off_fp8 逐字一致，K=128）
__host__ __device__ __forceinline__ int sw128_off_fp8(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 7, kk = k & 127;
  const int cc = (kk >> 4) ^ rr;
  return (rg * (K >> 7) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 15);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)),
               "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(smem_u32(bar)), "r"(phase));
  }
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
}
// 与 tma_load_4d 同款（主 kernel 的单 CTA 版）。
__device__ __forceinline__ void tma_load_4d(const CUtensorMap* map, void* dst, int k0, int r0,
                                            int hd, int b, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(k0), "r"(r0), "r"(hd), "r"(b), "r"(smem_u32(bar))
      : "memory");
}
// multicast 版：leader 发一条，广播到 mask 内各 CTA 的同一 smem 偏移 + 各自本地 mbarrier。
__device__ __forceinline__ void tma_load_4d_mcast(const CUtensorMap* map, void* dst, int k0, int r0,
                                                  int hd, int b, uint64_t* bar, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      ".multicast::cluster [%0], [%1, {%2, %3, %4, %5}], [%6], %7;\n" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(k0), "r"(r0), "r"(hd), "r"(b), "r"(smem_u32(bar)), "h"(mask)
      : "memory");
}

// =============================================================================
// kernel：cluster 沿 x（同一 cluster 内所有 CTA 读同一 (h) 的同一批 K tile）
//   MCAST=false：每个 CTA 各自发 TMA（现状）
//   MCAST=true ：仅 rank0 发 multicast，全 cluster 共享一次 L2 读
//   grid = Hkv*CN（clusterDim.x=CN），NT = S/BN 个 tile/CTA。
//   out[cta][nt][KS_SZ] 落盘每 CTA 每 tile 的 smem 内容，供 host 逐字节对拍。
// =============================================================================
template <int CN, bool MCAST>
__global__ void __launch_bounds__(THREADS) mcast_kernel(const __grid_constant__ CUtensorMap kmap,
                                                        unsigned char* __restrict__ out, int NT,
                                                        int Hkv, int copy) {
  extern __shared__ __align__(1024) char smem[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem);
  // Ks 必须 128B（实际给 1024B）对齐：SW128 TMA 的 smem 目标。
  unsigned char* Ks = reinterpret_cast<unsigned char*>(smem) + 1024;  // 2 stage * KS_SZ

  const int tid = threadIdx.x;
  const int rank = (CN > 1) ? (int)cluster_rank() : 0;
  const int cid = (CN > 1) ? (blockIdx.x / CN) : blockIdx.x;  // cluster id
  // 复制因子：多个 cluster 读同一 (h) 的 K（模仿主 kernel「所有 m 块读同一份 K/V」，
  // 把 L2 压到代表性水平）。
  const int h = cid % Hkv;

  if (tid == 0) {
    mbar_init(bar + 0, 1);
    mbar_init(bar + 1, 1);
    fence_mbar_init();
  }
  __syncthreads();
  if (CN > 1) cluster_sync();  // 远端 barrier 必须已 init

  for (int nt = 0; nt < NT; ++nt) {
    const int st = nt & 1;
    const uint32_t phase = (uint32_t)((nt >> 1) & 1);
    if (tid == 0) {
      mbar_arrive_expect(bar + st, KS_SZ);
      if constexpr (MCAST) {
        if (rank == 0)
          tma_load_4d_mcast(&kmap, Ks + st * KS_SZ, 0, nt * BN, h, 0, bar + st,
                            (uint16_t)((1u << CN) - 1));
      } else {
        tma_load_4d(&kmap, Ks + st * KS_SZ, 0, nt * BN, h, 0, bar + st);
      }
    }
    mbar_wait(bar + st, phase);
    const unsigned char* src = Ks + st * KS_SZ;
    if (copy) {
      // 对拍模式：把本 tile 的 smem 原样落盘（逐字节验证 multicast 与单发一致）
      unsigned char* dst = out + (((size_t)blockIdx.x * NT + nt)) * KS_SZ;
      for (int i = tid * 2; i < KS_SZ; i += THREADS * 2) {
        dst[i] = src[i];
        dst[i + 1] = src[i + 1];
      }
    } else {
      // 轻量消费：只累加校验和（写 1 字节/CTA/tile），避免全局写-分配读污染 L2 读指标。
      unsigned acc = 0;
      for (int i = tid; i < KS_SZ; i += THREADS) acc += src[i];
      if (tid == 0) out[((size_t)blockIdx.x * NT + nt)] = (unsigned char)(acc & 0xff);
    }
    __syncthreads();  // 所有线程读完 stage st 才允许下一轮（+2）覆写
  }
}

// =============================================================================
// host
// =============================================================================
static CUtensorMap make_kmap(const void* ptr, long long Hkv, long long S) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)Hkv, 1};
  uint64_t strides[3] = {(uint64_t)(Hkv * D), (uint64_t)D, (uint64_t)(S * Hkv * D)};
  uint32_t box[4] = {128, BN, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, (void*)ptr, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = "?";
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", s);
    std::exit(1);
  }
  return map;
}

static int g_mccopy = 1;
static int g_hkv = 16;
template <int CN, bool MCAST>
static float run_case(dim3 grid, const CUtensorMap& kmap, unsigned char* dout, int NT, int reps,
                      bool do_timing) {
  const int smem = 1024 + STAGES * KS_SZ;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = (CN > 1) ? CN : 1;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  auto launch = [&]() {
    cudaError_t e = cudaLaunchKernelEx(&cfg, mcast_kernel<CN, MCAST>, kmap, dout, NT, g_hkv, g_mccopy);
    if (e != cudaSuccess) {
      fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(e));
      std::exit(1);
    }
  };
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  if (!do_timing) return 0.f;
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) launch();
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms / reps;
}

int main(int argc, char** argv) {
  const int S = 4096, Hkv = 16;
  const int NT = S / BN;
  int reps = 200;
  std::vector<int> cns;
  if (argc > 1) {
    cns.push_back(atoi(argv[1]));
  } else {
    cns = {1, 2, 4};
  }
  if (argc > 2) reps = atoi(argv[2]);
  if (argc > 3) g_mccopy = atoi(argv[3]);
  int R = 1;  // 复制因子：多个 cluster 读同一 (h) 的 K，把 L2 压到代表性水平
  if (argc > 4) R = atoi(argv[4]);
  g_hkv = Hkv;
  (void)cuInit(0);

  const size_t kn = (size_t)S * Hkv * D;
  std::vector<unsigned char> hK(kn);
  std::mt19937 rng(1234);
  for (size_t i = 0; i < kn; ++i) hK[i] = (unsigned char)(rng() & 0xff);

  unsigned char* dK = nullptr;
  CUDA_CHECK(cudaMalloc(&dK, kn));
  CUDA_CHECK(cudaMemcpy(dK, hK.data(), kn, cudaMemcpyHostToDevice));
  CUtensorMap kmap = make_kmap(dK, Hkv, S);

  // host 参考：对 (h, nt) 的 SW128 tile 手工 swizzle
  auto ref_tile = [&](int h, int nt, unsigned char* dst) {
    for (int r = 0; r < BN; ++r)
      for (int k = 0; k < D; ++k)
        dst[sw128_off_fp8(r, k, D)] = hK[(((size_t)(nt * BN + r)) * Hkv + h) * D + k];
  };

  // copy 模式落盘每 CTA 每 tile 的整块 smem；轻量模式只写 1 字节/CTA/tile。
  const size_t out_bytes = g_mccopy ? (size_t)Hkv * R * 4 * NT * KS_SZ
                                    : (size_t)Hkv * R * 4 * NT;
  unsigned char* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&dout, out_bytes));
  std::vector<unsigned char> hout(out_bytes);

  printf("=== fa_bwd_fp8_mcast_smoke (O118) ===\n");
  printf("S=%d Hkv=%d D=%d BN=%d NT=%d KS_SZ=%d reps=%d\n", S, Hkv, D, BN, NT, KS_SZ, reps);

  for (int CN : cns) {
    const dim3 grid(Hkv * R * CN);
    const size_t bytes_this = (size_t)Hkv * R * CN * NT * KS_SZ;

    // --- baseline：每 CTA 各自发 TMA ---
    CUDA_CHECK(cudaMemset(dout, 0, out_bytes));
    float ms_base = 0.f;
    if (CN == 1)
      ms_base = run_case<1, false>(grid, kmap, dout, NT, reps, true);
    else if (CN == 2)
      ms_base = run_case<2, false>(grid, kmap, dout, NT, reps, true);
    else if (CN == 4)
      ms_base = run_case<4, false>(grid, kmap, dout, NT, reps, true);
    size_t bad_base = 0;
    if (g_mccopy) {
      CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytes_this, cudaMemcpyDeviceToHost));
      // 对拍 baseline vs host 参考
      std::vector<unsigned char> ref(KS_SZ);
      for (int c = 0; c < Hkv * R * CN; ++c) {
        const int h = (c / CN) % Hkv;
        for (int nt = 0; nt < NT; ++nt) {
          ref_tile(h, nt, ref.data());
          const unsigned char* got = hout.data() + ((size_t)c * NT + nt) * KS_SZ;
          for (int i = 0; i < KS_SZ; ++i)
            if (got[i] != ref[i]) ++bad_base;
        }
      }
    }

    // --- multicast ---
    CUDA_CHECK(cudaMemset(dout, 0, out_bytes));
    float ms_mc = 0.f;
    if (CN == 1)
      ms_mc = run_case<1, true>(grid, kmap, dout, NT, reps, true);
    else if (CN == 2)
      ms_mc = run_case<2, true>(grid, kmap, dout, NT, reps, true);
    else if (CN == 4)
      ms_mc = run_case<4, true>(grid, kmap, dout, NT, reps, true);
    size_t bad_mc = 0;
    if (g_mccopy) {
      CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytes_this, cudaMemcpyDeviceToHost));
      std::vector<unsigned char> ref(KS_SZ);
      for (int c = 0; c < Hkv * R * CN; ++c) {
        const int h = (c / CN) % Hkv;
        for (int nt = 0; nt < NT; ++nt) {
          ref_tile(h, nt, ref.data());
          const unsigned char* got = hout.data() + ((size_t)c * NT + nt) * KS_SZ;
          for (int i = 0; i < KS_SZ; ++i)
            if (got[i] != ref[i]) ++bad_mc;
        }
      }
    }

    printf("CN=%d  baseline: %.4f ms  mismatch=%zu  |  mcast: %.4f ms  mismatch=%zu  | "
           "speedup=%.4fx  read_bytes(base/mcast)=%.1f/%.1f MB\n",
           CN, ms_base, bad_base, ms_mc, bad_mc, ms_base / (ms_mc > 0 ? ms_mc : 1.f),
           (double)Hkv * R * CN * NT * KS_SZ / 1e6, (double)Hkv * R * NT * KS_SZ / 1e6);
    fflush(stdout);
  }

  CUDA_CHECK(cudaFree(dK));
  CUDA_CHECK(cudaFree(dout));
  return 0;
}
