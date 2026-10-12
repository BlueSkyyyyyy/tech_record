// cross_layer_pipeline.cu — MegaKernel 专题 P23-1 小实验：
//   **跨层软件流水（cross-layer software pipelining）+ 环形缓冲预取**：
//   把 MPK §5.3（快照行 1326–1364）「跨 task 软件流水 / paged shared memory / task 描述预取」
//   这条此前只有资料分析（P22-1 §15.6 明确标注「真正的跨层流水需要环形缓冲/预取，本节未做」）
//   的机制在**目标卡 H100 / sm90** 上实测出来。
//
// 与 P13-1 的区别：
//   P13-1 在**单个 task 内部**把「载入 chunk -> 计算 chunk」做成 K 级流水（tile 级）；本实验是
//   **跨层（跨 task/算子边界）**：一层 = 一组 task，层 l+1 的 task 依赖层 l 的激活值，
//   在**严格跨层依赖**下，能否边算层 l、边预取层 l+1 的私有权重 tile？这正是 kernel-per-operator
//   做不到的（两个算子是两个 kernel，SMEM 生命周期不重叠，MPK §5.3 行 1312–1325 的 paged SMEM
//   就是为了解除「每个 kernel 独占自己 block 的 SMEM」这一假设）。
//
// 依赖模型（模拟一层算子里「私有权重 tile（可预取，不依赖前层）+ 跨 worker 的激活值（依赖前层）」）：
//   层 l、worker j 的 task：
//     - 私有 bulk：载入权重 tile W[l][j]（T*E 个 float，cp.async 到 SMEM）——**与 DAG 无关、可预取**
//     - 跨层依赖：读激活 act[l][(j+1)%G]（由 worker (j+1)%G 在层 l-1 写出）——**必须等**
//     - 计算：对 SMEM tile 做 E 条链 × R 轮 FMA，块内归约 -> s；写 act[l+1][j] = s + act[l][(j+1)%G]
//   worker=block，G=#SM，W=G（每层每 worker 一个 task）。层 l 的依赖落在**相邻 worker**上，故
//   既有跨层（纵向）也有跨 worker（横向）依赖，逼出真实的跨 SM 协调，而非纯本地链。
//
// 四种策略（都算同一张 L 层 × G 宽 的依赖图，数值应逐位一致）：
//   eager  ：L 次 kernel launch，每层一个 kernel（kernel-per-operator 基线；层间由 stream 串行化，
//            SMEM 随 kernel 结束释放，**无法跨层预取**）
//   barrier：单 persistent kernel + 每层 grid.sync()（集中式会合；层 l 完全结束才进层 l+1）
//   decen  ：单 persistent kernel + 逐槽事件计数器（P15/P22 去中心化；细粒度依赖，但**不做预取**）
//   ring   ：decen + **K 页环形 SMEM 跨 task 预取**（MPK §5.3 跨 task 软件流水；算层 l 时预取层 l+1..l+K-1
//            的权重 tile 到其它 SMEM 页）
//
// 编译：nvcc -O3 -arch=sm_90 cross_layer_pipeline.cu -o cross_layer_pipeline
// 用法：MK_DEV=3 ./cross_layer_pipeline [iters] [L] [R] [G] [T]
//   默认 iters=30 L=64 R=64 G=132 T=256（E=16 固定 -> tile=16KB，权重总量 ~138MB ≫ L2）

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_pipeline.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
      printf("CUDA ERROR %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,         \
             cudaGetErrorString(_e));                                           \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#ifndef KMAX
#define KMAX 8
#endif

#ifndef ELEM
#define ELEM 16
#endif
constexpr int E = ELEM;  // 每线程 float 数（E 条独立 FMA 链）；-DELEM=64 可得更重的访存 tile

// 每线程载入 E 个 float（16 条 16B cp.async）
__device__ __forceinline__ void async_load_tile(float* dst, const float* src) {
  const int off = threadIdx.x * E;
#pragma unroll
  for (int q = 0; q < E / 4; ++q) {
    __pipeline_memcpy_async(dst + off + 4 * q, src + off + 4 * q, 16);
  }
}

// 块内归约（warp shfl + 共享内存），结果给 lane0/warp0
__device__ __forceinline__ float block_reduce(float v) {
  __shared__ float red[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if (lane == 0) red[wid] = v;
  __syncthreads();
  const int nw = blockDim.x >> 5;
  if (wid == 0) {
    v = (lane < nw) ? red[lane] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  }
  return v;
}

// 计算阶段：对 SMEM tile 做 E 条链 × R 轮 FMA，块内归约
__device__ __forceinline__ float compute_tile(const float* smem, int R) {
  float acc = 0.f;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    float x = smem[threadIdx.x * E + e];
#pragma unroll 4
    for (int r = 0; r < R; ++r) x = fmaf(x, 1.0000001f, 1e-7f);
    acc += x;
  }
  return block_reduce(acc);
}

// ---------------- eager：每层一个 kernel（kernel-per-operator） ----------------
__global__ void layer_kernel(const float* __restrict__ w, float* __restrict__ act,
                             long long tile, int G, int l, int R) {
  extern __shared__ float smem[];
  const int j = blockIdx.x, tid = threadIdx.x;
  const float* wt = w + ((long long)l * G + j) * tile;
  async_load_tile(smem, wt);
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();
  float pred = 0.f;
  if (tid == 0) pred = act[(long long)l * G + (j + 1) % G];
  float s = compute_tile(smem, R);  // 仅 tid0 有效
  if (tid == 0) act[(long long)(l + 1) * G + j] = s + pred;
}

// ---------------- barrier：单 kernel + 每层 grid.sync ----------------
__global__ void barrier_kernel(const float* __restrict__ w, float* __restrict__ act,
                               long long tile, int G, int L, int R) {
  extern __shared__ float smem[];
  cg::grid_group grid = cg::this_grid();
  const int j = blockIdx.x, tid = threadIdx.x;
  for (int l = 0; l < L; ++l) {
    grid.sync();  // 层 l-1 全部完成、act[l] 可见
    const float* wt = w + ((long long)l * G + j) * tile;
    async_load_tile(smem, wt);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    float pred = 0.f;
    if (tid == 0) pred = act[(long long)l * G + (j + 1) % G];
    float s = compute_tile(smem, R);
    if (tid == 0) act[(long long)(l + 1) * G + j] = s + pred;
  }
}

// ---------------- decen：单 kernel + 逐槽事件计数器（无预取） ----------------
__global__ void decen_kernel(const float* __restrict__ w, float* __restrict__ act,
                             int* __restrict__ cnt, long long tile, int G, int L,
                             int R) {
  extern __shared__ float smem[];
  const int j = blockIdx.x, tid = threadIdx.x;
  for (int l = 0; l < L; ++l) {
    const int slot = l * G + (j + 1) % G;  // 前驱槽：worker (j+1)%G 在层 l-1 写
    const float* wt = w + ((long long)l * G + j) * tile;
    if (tid == 0) {
      while (atomicAdd(&cnt[slot], 0) < 1) __nanosleep(64);
    }
    async_load_tile(smem, wt);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();  // 保证 tid0 的轮询完成与本块拷贝就绪
    float pred = 0.f;
    if (tid == 0) {
      __threadfence();  // 确认前驱数据可见后再读
      pred = act[slot];
    }
    float s = compute_tile(smem, R);
    if (tid == 0) {
      act[(long long)(l + 1) * G + j] = s + pred;
      __threadfence();
      atomicAdd(&cnt[(long long)(l + 1) * G + j], 1);
    }
  }
}

// ---------------- ring：decen + K 页环形 SMEM 跨 task 预取 ----------------
template <int K>
__global__ void ring_kernel(const float* __restrict__ w, float* __restrict__ act,
                            int* __restrict__ cnt, long long tile, int G, int L,
                            int R) {
  extern __shared__ float smem[];  // K 页
  const int j = blockIdx.x, tid = threadIdx.x;
  const int inflight = K - 1;
  int issued = 0;
  // 序言：预取层 0..K-2 的权重 tile
  for (; issued < inflight && issued < L; ++issued) {
    async_load_tile(smem + (size_t)(issued % K) * tile,
                    w + ((long long)issued * G + j) * tile);
    __pipeline_commit();
  }
  for (int l = 0; l < L; ++l) {
    // 预取层 l+K-1 的权重 tile（写回环形页 (l-1)%K，其计算已在上一轮 __syncthreads 后结束）
    const int nl = l + inflight;
    if (nl < L) {
      async_load_tile(smem + (size_t)(nl % K) * tile,
                      w + ((long long)nl * G + j) * tile);
      __pipeline_commit();
      ++issued;
    }
    const int slot = l * G + (j + 1) % G;
    if (tid == 0) {
      while (atomicAdd(&cnt[slot], 0) < 1) __nanosleep(64);
    }
    // 等层 l 的 tile 就绪：只允许更新的 (issued-l-1) 段仍在飞
    const int prior = issued - l - 1;
    __pipeline_wait_prior(prior);
    __syncthreads();
    float pred = 0.f;
    if (tid == 0) {
      __threadfence();
      pred = act[slot];
    }
    float s = compute_tile(smem + (size_t)(l % K) * tile, R);
    if (tid == 0) {
      act[(long long)(l + 1) * G + j] = s + pred;
      __threadfence();
      atomicAdd(&cnt[(long long)(l + 1) * G + j], 1);
    }
    __syncthreads();  // 保护环形页在下一轮被重写
  }
}

// ---------------- host ----------------
static void seed_act(float* d_act, int* d_cnt, int G) {
  CUDA_CHECK(cudaMemset(d_act, 0, sizeof(float) * (size_t)G));  // act[0][*] 清零
  CUDA_CHECK(cudaMemset(d_cnt, 0, sizeof(int) * (size_t)G));
  // 预激活层 0 的入口事件（MPK prepare 预置启动事件）
  int* h = (int*)malloc(sizeof(int) * G);
  for (int i = 0; i < G; ++i) h[i] = 1;
  CUDA_CHECK(cudaMemcpy(d_cnt, h, sizeof(int) * G, cudaMemcpyHostToDevice));
  free(h);
}

// 计时封装：tile/G/T/R 固定，L 可调，iters 次取均值
struct Ctx {
  float* d_w; float* d_act; int* d_cnt; long long tile;
  int G, T, R, iters;
  size_t shmem1;
  int smem_optin;
  cudaEvent_t a, b;
};

static bool ring_ok(const Ctx& c, int K) {
  return (long)c.tile * (long)sizeof(float) * K <= c.smem_optin;
}

static bool launch_ring(const Ctx& c, int L, int K, size_t sh) {
  void* args[] = {(void*)&c.d_w, (void*)&c.d_act, (void*)&c.d_cnt,
                  (void*)&c.tile, (void*)&c.G, (void*)&L, (void*)&c.R};
  void* f = nullptr;
  switch (K) {
    case 1: f = (void*)ring_kernel<1>; break;
    case 2: f = (void*)ring_kernel<2>; break;
    case 3: f = (void*)ring_kernel<3>; break;
    case 4: f = (void*)ring_kernel<4>; break;
    case 6: f = (void*)ring_kernel<6>; break;
    case 8: f = (void*)ring_kernel<8>; break;
    default: printf("bad K=%d\n", K); std::exit(1);
  }
  CUDA_CHECK(cudaLaunchCooperativeKernel(f, dim3(c.G), dim3(c.T), args, sh, 0));
  return true;
}

static float time_eager(const Ctx& c, int L) {
  seed_act(c.d_act, c.d_cnt, c.G);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(c.a));
  for (int it = 0; it < c.iters; ++it)
    for (int l = 0; l < L; ++l)
      layer_kernel<<<c.G, c.T, c.shmem1>>>(c.d_w, c.d_act, c.tile, c.G, l, c.R);
  CUDA_CHECK(cudaEventRecord(c.b));
  CUDA_CHECK(cudaEventSynchronize(c.b));
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, c.a, c.b));
  return ms / c.iters;
}

static float time_barrier(const Ctx& c, int L) {
  seed_act(c.d_act, c.d_cnt, c.G);
  CUDA_CHECK(cudaDeviceSynchronize());
  void* args[] = {(void*)&c.d_w, (void*)&c.d_act, (void*)&c.tile,
                  (void*)&c.G, (void*)&L, (void*)&c.R};
  CUDA_CHECK(cudaEventRecord(c.a));
  for (int it = 0; it < c.iters; ++it)
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)barrier_kernel, dim3(c.G),
                                           dim3(c.T), args, c.shmem1, 0));
  CUDA_CHECK(cudaEventRecord(c.b));
  CUDA_CHECK(cudaEventSynchronize(c.b));
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, c.a, c.b));
  return ms / c.iters;
}

static float time_decen(const Ctx& c, int L) {
  seed_act(c.d_act, c.d_cnt, c.G);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(c.a));
  for (int it = 0; it < c.iters; ++it)
    decen_kernel<<<c.G, c.T, c.shmem1>>>(c.d_w, c.d_act, c.d_cnt, c.tile, c.G, L, c.R);
  CUDA_CHECK(cudaEventRecord(c.b));
  CUDA_CHECK(cudaEventSynchronize(c.b));
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, c.a, c.b));
  return ms / c.iters;
}

static float time_ring(const Ctx& c, int L, int K) {
  if (!ring_ok(c, K)) return -1.f;
  seed_act(c.d_act, c.d_cnt, c.G);
  CUDA_CHECK(cudaDeviceSynchronize());
  size_t sh = (size_t)c.tile * sizeof(float) * K;
  launch_ring(c, L, K, sh);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(c.a));
  for (int it = 0; it < c.iters; ++it) launch_ring(c, L, K, sh);
  CUDA_CHECK(cudaEventRecord(c.b));
  CUDA_CHECK(cudaEventSynchronize(c.b));
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, c.a, c.b));
  return ms / c.iters;
}

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? atoi(argv[1]) : 30;
  const int L0 = (argc > 2) ? atoi(argv[2]) : 64;
  const int R = (argc > 3) ? atoi(argv[3]) : 64;
  const int G = (argc > 4) ? atoi(argv[4]) : 132;
  const int T = (argc > 5) ? atoi(argv[5]) : 256;
  const int LMAX = 256;  // 数组按最大 L 分配，Section C 可扫 L

  int dev = 0;
  const char* devsel = getenv("MK_DEV");
  if (devsel) dev = atoi(devsel);
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int clk_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));

  const long long tile = (long long)T * E;
  const size_t wbytes = (size_t)LMAX * G * tile * sizeof(float);
  printf("== device ==\n%s, sm_%d%d, SMs=%d, clock=%.0f MHz (dev %d)\n", prop.name,
         prop.major, prop.minor, prop.multiProcessorCount, clk_khz / 1000.0, dev);
  printf("L=%d layers, G=%d workers, T=%d, E=%d, R=%d, tile=%lld floats (%lld KB), "
         "weight(alloc LMAX=%d)=%.0f MB, iters=%d\n\n",
         L0, G, T, E, R, tile, tile * 4 / 1024, LMAX, wbytes / 1048576.0, iters);

  float* d_w = nullptr;
  float* d_act = nullptr;
  int* d_cnt = nullptr;
  CUDA_CHECK(cudaMalloc(&d_w, wbytes));
  CUDA_CHECK(cudaMalloc(&d_act, sizeof(float) * (size_t)(LMAX + 1) * G));
  CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(int) * (size_t)(LMAX + 1) * G));
  {
    float* h = (float*)malloc(wbytes);
    for (size_t i = 0; i < wbytes / sizeof(float); ++i)
      h[i] = (float)((i % 1024) + 1) * 1e-3f;
    CUDA_CHECK(cudaMemcpy(d_w, h, wbytes, cudaMemcpyHostToDevice));
    free(h);
  }

  const size_t shmem1 = (size_t)tile * sizeof(float);
  const size_t shmemK = (size_t)tile * sizeof(float) * KMAX;

  CUDA_CHECK(cudaFuncSetAttribute(layer_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem1));
  CUDA_CHECK(cudaFuncSetAttribute(barrier_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem1));
  CUDA_CHECK(cudaFuncSetAttribute(decen_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem1));
  // ring 各实例的动态 SMEM 上限（K>3 时 > 48KB 默认值，必须显式设置）；超出单 block
  // optin 上限的 K 置 0 表示不可用（→ 后面的扫描会跳过该深度）
  int smem_optin = prop.sharedMemPerBlockOptin;
  auto set_ring = [&](int K, void* f) {
    long need = (long)tile * sizeof(float) * K;
    if (need > smem_optin) return;
    CUDA_CHECK(cudaFuncSetAttribute(f,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need));
  };
  set_ring(1, (void*)ring_kernel<1>);
  set_ring(2, (void*)ring_kernel<2>);
  set_ring(3, (void*)ring_kernel<3>);
  set_ring(4, (void*)ring_kernel<4>);
  set_ring(6, (void*)ring_kernel<6>);
  set_ring(8, (void*)ring_kernel<8>);
  (void)shmemK;
  int maxablk = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxablk, decen_kernel, T, shmem1));

  int coop_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&coop_blocks,
      barrier_kernel, T, shmem1));
  printf("1-page occupancy=%d block/SM; cooperative capacity=%d block * %d SM = %d, "
         "grid=%d\n\n", maxablk, coop_blocks, prop.multiProcessorCount,
         coop_blocks * prop.multiProcessorCount, G);
  if (coop_blocks * prop.multiProcessorCount < G)
    printf(">> WARNING: cooperative grid %d > capacity %d; reduce G/T.\n\n", G,
           coop_blocks * prop.multiProcessorCount);

  Ctx c{d_w, d_act, d_cnt, tile, G, T, R, iters, shmem1, smem_optin, nullptr, nullptr};
  CUDA_CHECK(cudaEventCreate(&c.a));
  CUDA_CHECK(cudaEventCreate(&c.b));

  // ---------- correctness ----------
  printf("== correctness (all strategies must be bitwise equal on act[L][*]) ==\n");
  float *h_e = (float*)malloc(sizeof(float) * G);
  float *h_b = (float*)malloc(sizeof(float) * G);
  float *h_d = (float*)malloc(sizeof(float) * G);
  float *h_r = (float*)malloc(sizeof(float) * G);
  auto read_final = [&](int L, float* h) {
    CUDA_CHECK(cudaMemcpy(h, d_act + (long long)L * G, sizeof(float) * G,
                          cudaMemcpyDeviceToHost));
  };
  const int LC = L0;
  seed_act(d_act, d_cnt, G);
  for (int l = 0; l < LC; ++l) layer_kernel<<<G, T, shmem1>>>(d_w, d_act, tile, G, l, R);
  CUDA_CHECK(cudaDeviceSynchronize()); read_final(LC, h_e);
  { seed_act(d_act, d_cnt, G);
    void* args[] = {(void*)&d_w, (void*)&d_act, (void*)&tile, (void*)&G, (void*)&LC, (void*)&R};
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)barrier_kernel, dim3(G), dim3(T), args, shmem1, 0));
    CUDA_CHECK(cudaDeviceSynchronize()); read_final(LC, h_b); }
  { seed_act(d_act, d_cnt, G);
    decen_kernel<<<G, T, shmem1>>>(d_w, d_act, d_cnt, tile, G, LC, R);
    CUDA_CHECK(cudaDeviceSynchronize()); read_final(LC, h_d); }
  { seed_act(d_act, d_cnt, G);
    int kc = 4; while (kc > 1 && !ring_ok(c, kc)) --kc;
    launch_ring(c, LC, kc, (size_t)tile * sizeof(float) * kc);
    CUDA_CHECK(cudaDeviceSynchronize()); read_final(LC, h_r); }
  auto cmp = [&](const float* x, const float* y) {
    double mx = 0; for (int i = 0; i < G; ++i) {
      double d = x[i] - y[i]; d = d < 0 ? -d : d; double s = x[i] < 0 ? -x[i] : x[i];
      double rel = d / (s > 1e-30 ? s : 1e-30); if (rel > mx) mx = rel;
    } return mx;
  };
  printf("act[L][0]=%.6f  eager vs barrier max rel=%.2e, vs decen=%.2e, vs ring=%.2e\n\n",
         h_e[0], cmp(h_e, h_b), cmp(h_e, h_d), cmp(h_e, h_r));

  int KRING = 4; while (KRING > 1 && !ring_ok(c, KRING)) --KRING;

  // ---------- Section A: 扫 R ----------
  printf("== Section A: L=%d, G=%d, sweep compute intensity R (ring K=%d) ==\n",
         L0, G, KRING);
  printf("%5s | %10s | %10s | %10s | %10s | %8s | %8s\n", "R", "eager ms",
         "barrier", "decen", "ringK", "d/b", "r/b");
  printf("------+------------+------------+------------+------------+----------+---------\n");
  for (int r : {0, 8, 16, 32, 64, 128, 256}) {
    Ctx cc = c; cc.R = r;
    float te = time_eager(cc, L0);
    float tb = time_barrier(cc, L0);
    float td = time_decen(cc, L0);
    float tr = time_ring(cc, L0, KRING);
    printf("%5d | %10.4f | %10.4f | %10.4f | %10.4f | %7.3fx | %7.3fx\n", r,
           te, tb, td, tr, td / tb, tr / tb);
  }

  // ---------- Section B: 扫流水深度 K（取访存占比较高的 R=16 以暴露深度效应） ----------
  const int RB = 16;
  Ctx cb = c; cb.R = RB;
  printf("\n== Section B: L=%d, R=%d (访存占比较高), G=%d, sweep ring depth K ==\n",
         L0, RB, G);
  printf("%3s | %10s | %10s | %10s | %8s | %10s\n", "K", "barrier ms", "decen ms",
         "ring ms", "r/d", "us/layer");
  printf("----+------------+------------+------------+----------+-----------\n");
  float tb0 = time_barrier(cb, L0);
  float td0 = time_decen(cb, L0);
  for (int k : {1, 2, 3, 4, 6, 8}) {
    float tr = time_ring(cb, L0, k);
    if (tr < 0) { printf("%3d | (SMEM 超单 block optin 上限，跳过)\n", k); continue; }
    printf("%3d | %10.4f | %10.4f | %10.4f | %7.3fx | %9.3f\n", k, tb0, td0, tr,
           tr / td0, tr * 1e3 / L0);
  }

  // ---------- Section C: 扫层数 L ----------
  printf("\n== Section C: R=%d, G=%d, sweep layer count L (ring K=4) ==\n", R, G);
  printf("%5s | %10s | %10s | %10s | %8s\n", "L", "eager ms", "barrier",
         "ringK", "r/b");
  printf("------+------------+------------+------------+----------\n");
  for (int ell : {8, 16, 32, 64, 128, 256}) {
    float te = time_eager(c, ell);
    float tb = time_barrier(c, ell);
    float tr = time_ring(c, ell, KRING);
    printf("%5d | %10.4f | %10.4f | %10.4f | %7.3fx\n", ell, te, tb, tr, tr / tb);
  }

  printf("\n>> 读法：\n");
  printf(">> eager  = kernel-per-operator：L 次 launch，层间由 stream 串行，SMEM 随 kernel 释放（不可预取）；\n");
  printf(">> barrier= 单 kernel + 每层 grid.sync（集中式会合，层间串行化其上还有全图屏障）；\n");
  printf(">> decen  = 单 kernel + 逐槽事件计数器（细粒度依赖，但不预取权重）；\n");
  printf(">> ringK  = decen + K 页环形 SMEM 跨 task 预取（MPK §5.3 跨 task 软件流水）。\n");
  printf(">> 预期：ring 相对 barrier/decen 把「下一层权重 tile 的 HBM 延迟」藏进「本层计算」，\n");
  printf(">>   收益上界 ≈ (Tload+Tcomp)/max(Tload,Tcomp)，随 R（计算占比）变化。\n");
  printf(">> 注意：R=0（纯访存、无可藏的计算）时流水反而更慢；重载 tile（-DELEM=32）才显出\n");
  printf(">>   流水深度 K 的收益（Section B 单调到 K≈4 饱和），轻载（E=16）时事件等待已够藏访存。\n");

  CUDA_CHECK(cudaEventDestroy(c.a));
  CUDA_CHECK(cudaEventDestroy(c.b));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_act));
  CUDA_CHECK(cudaFree(d_cnt));
  free(h_e); free(h_b); free(h_d); free(h_r);
  return 0;
}
