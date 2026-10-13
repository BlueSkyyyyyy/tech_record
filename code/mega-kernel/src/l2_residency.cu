// l2_residency.cu — MegaKernel 专题 P40-1 小实验：L2 常驻（persisting L2 / access policy window）
// 作为「损失三」的第三档中间量落地位置（寄存器 → L2 → HBM）。
//
// 动机（目标卡 H100 / sm90 实测）：
//   P24-1（docs/04 §17）用 ncu 直接量了损失三的 HBM 字节，其 §17.7 明留一条假设：
//     「若张量能整块驻留 L2（≤50 MB），eager 的中间量往返可能命中 L2 而不落 DRAM，
//       损失三会小于本节数字——megakernel 的损失三在『工作集超过 L2』时才充分显现。」
//   本实验就把这条假设测出来，并回答：**L2 常驻能不能当「中间量的第三档落点」用？**
//
//   构造一条三算子序列（kernel-per-operator 的 eager 形态）：
//     K1: 读 x(大) → 写 y(中间量, 尺寸 S 可变)
//     polluter: 流式读+写一个大缓冲（把 L2 冲干净，模拟真实 serving 里的并发流量）
//     K2: 读 y → 写 z
//   中间量 y 落三档，用 ncu 数 K2 的 DRAM 读字节：
//     (1) base    : 不插 polluter，K2 紧跟 K1 —— y 有机会靠「最近性」留在 L2。
//     (2) pollute : 插 polluter 把 y 逐出 L2 —— y 被迫落 HBM（损失三显形）。
//     (3) persist : 插 polluter，但给 y 加 `cudaAccessPolicyWindow`（hitProp=Persisting /
//                   missProp=Streaming）+ `cudaLimitPersistingL2CacheSize` —— y 被**保护**在 L2。
//
//   预期：K2 的 dram__bytes_read 在 base/persist ≈ 0（y 全命中 L2），在 pollute ≈ S
//   （y 从 HBM 读回）。扫 S（4→64 MB，跨越 H100 的 50 MB L2）给出边界。
//
//   含义：这是损失三的「片上保留」机制——不融合算子（不改数值、不改 DAG），只把中间量
//   声明为 persisting，就能在 S ≤ L2 时把 eager 的中间量往返从 HBM 拉回 L2。
//
// 编译：nvcc -O3 -arch=sm_90 src/l2_residency.cu -o src/l2_residency
// 用法：
//   ./l2_residency bench [iters]                 # 扫 S × {base,pollute,persist}，K2 计时表
//   ./l2_residency one <mode> <S_MB>             # 单次序列，供 ncu 采样

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
      printf("CUDA ERROR %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,         \
             cudaGetErrorString(_e));                                           \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define BYTES_PER_MB (1u << 20)
#define FLOAT4 (16)

// ---- 便宜的逐元素算子（带宽主导；float4 向量化） ----
__device__ __forceinline__ float4 op_f(float4 v) {
  v.x = fmaf(v.x, 1.5f, 0.25f);
  v.y = fmaf(v.y, 1.5f, 0.25f);
  v.z = fmaf(v.z, 1.5f, 0.25f);
  v.w = fmaf(v.w, 1.5f, 0.25f);
  return v;
}
__device__ __forceinline__ float4 op_g(float4 v) {
  v.x = fmaf(v.x, 0.75f, 0.5f);
  v.y = fmaf(v.y, 0.75f, 0.5f);
  v.z = fmaf(v.z, 0.75f, 0.5f);
  v.w = fmaf(v.w, 0.75f, 0.5f);
  return v;
}

// 初始化 kernel（命名不以 l2_ 开头，便于 ncu 的 -k regex:l2_k2 只抓 K2）
__global__ void init_fill(float4* a, long n4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float b = (float)(i % 977) * 0.001f - 0.5f;
    a[i] = make_float4(b, b + 0.1f, b + 0.2f, b + 0.3f);
  }
}

// K1: 读 x(nx4) → 写 y(ny4)。y[i] 只覆盖 i<ny4（ny4<=nx4）。
__global__ void l2_k1(const float4* __restrict__ x, float4* __restrict__ y,
                      long nx4, long ny4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < nx4; i += stride) {
    float4 v = x[i];
    if (i < ny4) y[i] = op_f(v);
  }
}

// polluter: 流式读+写一个大缓冲，把 L2 冲干净（真实 serving 里的并发流量）。
__global__ void l2_pollute(float4* __restrict__ p, long n4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) p[i] = op_g(p[i]);
}

// K2: 读 y(ny4) → 写 z(ny4)。量它的 DRAM 读字节 = 中间量 y 到底命中 L2 没有。
__global__ void l2_k2(const float4* __restrict__ y, float4* __restrict__ z,
                      long ny4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < ny4; i += stride) z[i] = op_g(y[i]);
}

static int g_threads = 256;
static int g_grid = 0;  // 按占用率顶格

static int pick_grid() {
  int dev; CUDA_CHECK(cudaGetDevice(&dev));
  int nsm; CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  int perSM; CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &perSM, (const void*)l2_k2, g_threads, 0));
  if (perSM < 1) perSM = 1;
  return nsm * perSM;
}

static void set_persist_window(cudaStream_t st, void* base, size_t bytes) {
  cudaStreamAttrValue attr;
  memset(&attr, 0, sizeof(attr));
  attr.accessPolicyWindow.base_ptr = base;
  attr.accessPolicyWindow.num_bytes = bytes;
  attr.accessPolicyWindow.hitRatio = 1.0f;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  CUDA_CHECK(cudaStreamSetAttribute(st, cudaStreamAttributeAccessPolicyWindow, &attr));
}

static void clear_persist_window(cudaStream_t st) {
  cudaStreamAttrValue attr;
  memset(&attr, 0, sizeof(attr));
  attr.accessPolicyWindow.num_bytes = 0;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  CUDA_CHECK(cudaStreamSetAttribute(st, cudaStreamAttributeAccessPolicyWindow, &attr));
  CUDA_CHECK(cudaCtxResetPersistingL2Cache());
}

// persist 模式下在 stream 上挂 y 的持久窗口。
//   persist : carveout = min(S, maxPersist)（「按中间量大小配」的合理用法）
//   pfull   : carveout = maxPersist（把整块 persisting 都划走，测过度预留的代价）
static void apply_window(const char* mode, cudaStream_t st, float4* y, long ny4,
                         size_t maxPersist) {
  size_t want = (size_t)ny4 * FLOAT4;
  if (strcmp(mode, "persist") == 0 || strcmp(mode, "pfull") == 0) {
    size_t set = (strcmp(mode, "pfull") == 0)
                     ? maxPersist
                     : (want < maxPersist ? want : maxPersist);
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set));
    set_persist_window(st, (void*)y, want);
  } else {
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
    clear_persist_window(st);
  }
}

// 序列前半：K1（写中间量 y）+ 可选 polluter（把 y 逐出 L2）。
static void run_pre(const char* mode, cudaStream_t st, const float4* x,
                    float4* y, float4* p, long nx4, long ny4, long np4) {
  l2_k1<<<g_grid, g_threads, 0, st>>>(x, y, nx4, ny4);
  if (strcmp(mode, "base") != 0) l2_pollute<<<g_grid, g_threads, 0, st>>>(p, np4);
}

// 跑一遍完整序列（供 ncu 一次性采样）。
static void run_sequence(const char* mode, cudaStream_t st,
                         const float4* x, float4* y, float4* z, float4* p,
                         long nx4, long ny4, long np4, size_t maxPersist) {
  apply_window(mode, st, y, ny4, maxPersist);
  run_pre(mode, st, x, y, p, nx4, ny4, np4);
  l2_k2<<<g_grid, g_threads, 0, st>>>(y, z, ny4);
}
// (run_pre 会把 polluter 插在 base 之外的每种模式里)

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "bench";

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  g_grid = pick_grid();

  int dev; CUDA_CHECK(cudaGetDevice(&dev));
  int maxPersistI = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&maxPersistI, cudaDevAttrMaxPersistingL2CacheSize, dev));
  size_t maxPersist = (size_t)maxPersistI;
  int maxWindow = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&maxWindow, cudaDevAttrMaxAccessPolicyWindowSize, dev));

  printf("== device ==\n%s, sm_%d%d, SMs=%d, L2=%d MB, maxPersistingL2=%zu MB, maxWindow=%d MB\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         prop.l2CacheSize / (int)BYTES_PER_MB, maxPersist / BYTES_PER_MB,
         maxWindow / (int)BYTES_PER_MB);
  printf("grid=%d (threads=%d, occupancy-capped)\n", g_grid, g_threads);

  const size_t XMB = 256, ZMB = 256, PMB = 512;  // 大流式数组：x 读、z 写、polluter 读写
  size_t xb = XMB * BYTES_PER_MB, zb = ZMB * BYTES_PER_MB, pb = PMB * BYTES_PER_MB;

  float *xa, *ya, *za, *pa;
  CUDA_CHECK(cudaMalloc(&xa, xb));
  CUDA_CHECK(cudaMalloc(&ya, 512 * BYTES_PER_MB));  // 上限：最大的 S
  CUDA_CHECK(cudaMalloc(&za, zb));
  CUDA_CHECK(cudaMalloc(&pa, pb));

  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreate(&st));
  long nx4 = (long)(xb / FLOAT4);
  long np4 = (long)(pb / FLOAT4);
  init_fill<<<g_grid, g_threads, 0, st>>>((float4*)xa, nx4);
  init_fill<<<g_grid, g_threads, 0, st>>>((float4*)pa, np4);
  init_fill<<<g_grid, g_threads, 0, st>>>((float4*)ya, (long)(512 * BYTES_PER_MB / FLOAT4));
  CUDA_CHECK(cudaDeviceSynchronize());

  if (strcmp(mode, "one") == 0) {
    int SMB = (argc > 3) ? atoi(argv[3]) : 16;
    const char* m = (argc > 2) ? argv[2] : "base";
    long ny4 = (long)((size_t)SMB * BYTES_PER_MB / FLOAT4);
    run_sequence(m, st, (const float4*)xa, (float4*)ya, (float4*)za, (float4*)pa,
                 nx4, ny4, np4, maxPersist);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("one mode=%s S=%d MB done\n", m, SMB);
  } else {
    int iters = (argc > 2) ? atoi(argv[2]) : 30;
    const int Ss[] = {4, 8, 16, 32, 48, 64};
    const char* modes[] = {"base", "pollute", "persist", "pfull"};
    printf("iters=%d, X=%zu MB, Z=%zu MB, polluter=%zu MB, 中间量 S 扫描 4..64 MB（L2=%d MB）\n\n",
           iters, XMB, ZMB, PMB, prop.l2CacheSize / (int)BYTES_PER_MB);
    printf("      | K2ms pollute | base/pollute | persist/pollute | pfull/pollute\n");
    printf("------+--------------+--------------+-----------------+---------------\n");

    for (int si = 0; si < 6; ++si) {
      int SMB = Ss[si];
      long ny4 = (long)((size_t)SMB * BYTES_PER_MB / FLOAT4);
      double t[4] = {0, 0, 0, 0};
      for (int mi = 0; mi < 4; ++mi) {
        const char* m = modes[mi];
        apply_window(m, st, (float4*)ya, ny4, maxPersist);
        for (int w = 0; w < 3; ++w)
          run_pre(m, st, (const float4*)xa, (float4*)ya, (float4*)pa, nx4, ny4, np4);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
        float total = 0;
        for (int r = 0; r < iters; ++r) {
          run_pre(m, st, (const float4*)xa, (float4*)ya, (float4*)pa, nx4, ny4, np4);
          CUDA_CHECK(cudaEventRecord(a, st));
          l2_k2<<<g_grid, g_threads, 0, st>>>((const float4*)ya, (float4*)za, ny4);
          CUDA_CHECK(cudaEventRecord(b, st));
          CUDA_CHECK(cudaEventSynchronize(b));
          float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
          total += ms;
        }
        t[mi] = total / iters;
        CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
      }
      printf("%4d MB | %12.4f | %12.3f | %15.3f | %13.3f\n",
             SMB, t[1], t[0] / t[1], t[2] / t[1], t[3] / t[1]);
    }

    // 正确性：base 与 persist 的 z 应逐位一致
    {
      long ny4 = (long)(16 * BYTES_PER_MB / FLOAT4);
      run_sequence("base", st, (const float4*)xa, (float4*)ya, (float4*)za, (float4*)pa, nx4, ny4, np4, maxPersist);
      CUDA_CHECK(cudaDeviceSynchronize());
      float* h1 = (float*)malloc((size_t)ny4 * FLOAT4);
      CUDA_CHECK(cudaMemcpy(h1, za, (size_t)ny4 * FLOAT4, cudaMemcpyDeviceToHost));
      run_sequence("persist", st, (const float4*)xa, (float4*)ya, (float4*)za, (float4*)pa, nx4, ny4, np4, maxPersist);
      CUDA_CHECK(cudaDeviceSynchronize());
      float* h2 = (float*)malloc((size_t)ny4 * FLOAT4);
      CUDA_CHECK(cudaMemcpy(h2, za, (size_t)ny4 * FLOAT4, cudaMemcpyDeviceToHost));
      double maxd = 0;
      for (long i = 0; i < ny4 * 4; ++i) {
        double d = fabs((double)h1[i] - (double)h2[i]);
        if (d > maxd) maxd = d;
      }
      printf("\ncorrectness (S=16MB): max|z_base - z_persist| = %.3e (expect 0)\n", maxd);
      free(h1); free(h2);
    }

    printf("\n>> 读法（K2-only 耗时，基准 = pollute；y 命中 L2 越快）：\n");
    printf(">>  base/pollute < 1：不插 polluter 时 y 靠「最近性」留在 L2；polluter 把它逐出。\n");
    printf(">>  persist：carveout=min(S,maxPersist)，给 y 挂 accessPolicyWindow(persisting)。\n");
    printf(">>  pfull  ：carveout=maxPersist（整块 persisting 都划走）——测过度预留的代价。\n");
    printf(">>  真实 DRAM 读字节 / L2 命中率见 `l2_residency one <mode> <S_MB>` 的 ncu 采样。\n");
  }

  CUDA_CHECK(cudaFree(xa)); CUDA_CHECK(cudaFree(ya));
  CUDA_CHECK(cudaFree(za)); CUDA_CHECK(cudaFree(pa));
  CUDA_CHECK(cudaStreamDestroy(st));
  return 0;
}
