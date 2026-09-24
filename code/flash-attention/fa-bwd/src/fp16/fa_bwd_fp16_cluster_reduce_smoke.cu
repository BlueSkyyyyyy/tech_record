// =============================================================================
// fa_bwd_fp16_cluster_reduce_smoke.cu —— O25 前置：cluster 分布式归约冒烟
// =============================================================================
// 动机（ROADMAP「下一步」①，O7b 的直接延续）：fp16/bf16 反向 main 的墙是 dK/dV 的
// 跨 CTA `atomicAdd`（`red` 占 L2 扇区 ~72%，O17 后 51.9M）。O7b 用「partial 覆盖写 +
// 二次归约 kernel」消掉了原子，但二次归约是一整趟 DRAM 扫描（90% 带宽 bound，384.9µs），
// 净负。本冒烟验证**不改数学、也不落全局 partial** 的第三条路：
//   * 用 Hopper **thread block cluster**（cluster 沿 bx，rank=blockIdx.x&1）；
//   * 每个 CTA 仍算自己 mblk 的 dK/dV 偏和 [BN][HD]；
//   * 非 leader rank 用 `red.shared::cluster.add.f32`（mapa 到 leader 的 smem 累加器）
//     把偏和**在 SM 间 smem 内合并**；leader 把自己的偏和也加进同一 smem 累加器；
//   * `barrier.cluster` 同步后，**leader 只发一次全局 atomicAdd**（red 字节 ~砍半）。
// 这样既没有 O7b 的全局 partial 缓冲 / 二次 DRAM 扫描，也没有每元素跨 CTA 原子。
//
// 本冒烟测「多 tile 复用同一个 DSM 累加器」的同步正确性（flush → barrier → 清零 → barrier）：
//   * reference：每个 CTA 对自己的每个 tile 偏和直接 `atomicAdd` 到 out；
//   * cluster 版：rank1 remote-add 进 rank0 smem，rank0 本地加，barrier，rank0 flush 到 out。
// 两者应逐位相同（浮点加法次序不同，容差 0；这里刻意让顺序一致以便 bitwise 比较）。
//
// 编译：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" scripts/run.sh ...
// （cluster 是 sm_90 特性，`compute_90a` 亦可；不用 FA_WGMMA）
// =============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),       \
              __FILE__, __LINE__);                                              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

constexpr int BN = 64, HD = 128, THREADS = 128;
constexpr int TILES = 8;      // 模拟因果下每 CTA 处理多个 KV tile
constexpr int NPART = 2;      // cluster size（2 个 CTA 合并）

// ---------------- cluster / DSM 原语 ----------------
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
}
// 把本地 smem 地址映射到 rank 的地址空间（shared::cluster）
__device__ __forceinline__ uint32_t map_shared(uint32_t local_addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
               : "=r"(r)
               : "r"(local_addr), "r"(rank));
  return r;
}
// 远端 shared 原子加（FSM 内合并）
__device__ __forceinline__ void red_shared_cluster_add_f32(uint32_t addr, float v) {
  asm volatile("red.shared::cluster.add.f32 [%0], %1;\n" ::"r"(addr), "f"(v)
               : "memory");
}
__device__ __forceinline__ float ld_shared_cluster_f32(uint32_t addr) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

// =============================================================================
// reference：每个 CTA 的每个 tile 偏和直接 atomicAdd 到 out（= 现状 O17）
// =============================================================================
__global__ void __launch_bounds__(THREADS) ref_kernel(
    const float* __restrict__ x, float* __restrict__ out, int ncta) {
  // x: [ncta][TILES][BN][HD]
  const int cta = blockIdx.x;             // 0..ncta-1
  const int tid = threadIdx.x;
  for (int t = 0; t < TILES; ++t) {
    const float* p = x + (((size_t)cta * TILES + t) * BN) * HD;
    for (int i = tid; i < BN * HD; i += THREADS)
      atomicAdd(out + i, p[i]);
  }
}

// =============================================================================
// cluster：所有 rank 的偏和在 leader 的 smem 累加器里合并，leader 每 tile 只 flush 一次
// =============================================================================
__global__ void __launch_bounds__(THREADS) __cluster_dims__(NPART, 1, 1)
cluster_kernel(const float* __restrict__ x, float* __restrict__ out, int ncta) {
  __shared__ __align__(16) float acc[BN * HD];
  const int tid = threadIdx.x;
  const uint32_t rank = cluster_rank();
  const uint32_t leader_local = smem_u32(acc);
  // leader 的 smem 累加器在所有 rank 里的地址
  const uint32_t leader_remote = map_shared(leader_local, 0u);

  const int cta = blockIdx.x;   // 输入切片仍是全局 cta 号（grid = ncta，cluster 沿 x）

  for (int t = 0; t < TILES; ++t) {
    const float* p = x + (((size_t)cta * TILES + t) * BN) * HD;
    // 所有 rank 都把自己的偏和 **原子加** 进 leader 的 smem（leader 的原子加落在本地
    // smem，其余经 mapa 落在远端 smem）——原子语义保证 leader 的非原子 flush 不会与
    // 其它 rank 的写入竞争。
    for (int i = tid; i < BN * HD; i += THREADS)
      red_shared_cluster_add_f32(leader_remote + (uint32_t)(i * 4), p[i]);
    // remote add 完成后 leader 才读——一次 cluster barrier 保证可见性。
    cluster_sync();
    if (rank == 0) {
      for (int i = tid; i < BN * HD; i += THREADS)
        atomicAdd(out + i, acc[i]);
    }
    // leader flush 完、且所有人不再读写 acc，才能清零进入下一 tile。
    cluster_sync();
    if (rank == 0) {
      for (int i = tid; i < BN * HD; i += THREADS) acc[i] = 0.f;
    }
    cluster_sync();
  }
}

int main() {
  const int NCTA = 4;   // grid = 4，cluster = 2 ⇒ 2 个 cluster
  std::vector<float> x((size_t)NCTA * TILES * BN * HD);
  srand(1234);
  for (auto& v : x) v = (float)((rand() % 100) - 50) * 0.01f;

  float *dx, *dref, *dclus;
  CUDA_CHECK(cudaMalloc(&dx, x.size() * 4));
  CUDA_CHECK(cudaMalloc(&dref, (size_t)BN * HD * 4));
  CUDA_CHECK(cudaMalloc(&dclus, (size_t)BN * HD * 4));
  CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dref, 0, (size_t)BN * HD * 4));
  CUDA_CHECK(cudaMemset(dclus, 0, (size_t)BN * HD * 4));

  ref_kernel<<<NCTA, THREADS>>>(dx, dref, NCTA);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cluster_kernel<<<NCTA, THREADS>>>(dx, dclus, NCTA);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> ref(BN * HD), clus(BN * HD);
  CUDA_CHECK(cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(clus.data(), dclus, clus.size() * 4, cudaMemcpyDeviceToHost));

  double maxabs = 0;
  for (size_t i = 0; i < ref.size(); ++i)
    maxabs = std::max(maxabs, (double)std::fabs(ref[i] - clus[i]));
  // 期望值
  double maxerr = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double s = 0;
    for (int c = 0; c < NCTA; ++c)
      for (int t = 0; t < TILES; ++t) s += x[(size_t)(c * TILES + t) * BN * HD + i];
    maxerr = std::max(maxerr, std::fabs(ref[i] - s));
  }
  printf("cluster_reduce: ref-vs-expected max_abs=%.3e, cluster-vs-ref max_abs=%.3e\n",
         maxerr, maxabs);
  // 加法的结合律差异（atomic 次序不同）在 fp32 下应远小于 1e-3（x 幅值 ~0.5、
  // 每元素 32 项）：reference 与 cluster 都只是把同样的项换个顺序相加。
  const bool ok = maxerr < 1e-3 && maxabs < 1e-3;
  printf("%s\n", ok ? "PASS" : "FAIL");
  cudaFree(dx); cudaFree(dref); cudaFree(dclus);
  return ok ? 0 : 1;
}
