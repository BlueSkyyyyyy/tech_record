// l2_persist_tuning.cu — MegaKernel 专题 P41-1 小实验：L2 persisting 的**精细化调参**与
// **多窗口/多流共存**。
//
// 动机：
//   P40-1（docs/04 §29）实测了 L2 常驻作为「损失三的第三档中间量落点」，结论是：
//     (1) 中间量是否留在 L2 由「写读之间流过什么」决定（polluter 一插即落 HBM）；
//     (2) `cudaAccessPolicyWindow` 的 persisting 只把 K2 的 DRAM 读降 ~23–30%（hitRatio=1.0）；
//     (3) **persisting 是「普通 L2 ↔ 常驻 L2」的零和交易**——把 31 MB carveout 全划走（pfull）
//         即便中间量很小也稳定慢 ~1.5×。
//   §29.6 明留一条尾巴：「只测单窗口、单 stream、单一 persisting 段，**未测多窗口、多 stream 各自
//   窗口、`hitRatio<1` 的概率化分区**」。本实验把这条补上。
//
// 关键机制（CUDA persisting L2）：
//   - `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, C)` 画出一块 device 级 **set-aside**（carveout），
//     H100 上限 31 MB（L2 = 50 MB）。
//   - `cudaStreamSetAttribute(..., accessPolicyWindow, w)` 给**一个 stream**挂**一个**窗口
//     `[base, base+num_bytes)`；窗口内每次命中按 `hitRatio=f` 的概率标 `hitProp=Persisting`、
//     其余标 `missProp`。所以 **hitRatio 是「概率化分区」旋钮**：只有 f×访问量去挤 carveout。
//   - 一个 stream 只能有一个窗口 → 「多窗口」必须落在**多条 stream**上。megakernel / MPK 让整图
//     很多中间量同时活着，正对应「多个并发 persisting 段共享同一 carveout」。
//
// 三个问题：
//   Q1（§A/B）概率化分区的收益：固定 carveout，把 hitRatio 从 1.0 降到 f，K2 的 DRAM 读是否≈ f×S？
//      时间上，降低 hitRatio 少占 carveout，能否比 hitRatio=1.0 更快（尤其 pfull 那种拉满 carveout 的
//      情形）？
//   Q2（§B）carveout × hitRatio 联合：pfull（carveout 拉满）的 ~1.5× 代价，能否用一个**更小的
//      carveout + hitRatio<1** 收回？
//   Q3（§C）多窗口共存：两个并发中间量 y1/y2 各挂窗口共享一个 carveout；hitRatio 如何在两个段之间
//      分配？「只保护一个」vs「两个都保护」的 DRAM 读是什么关系？
//
// 编译：nvcc -O3 -arch=sm_90 src/l2_persist_tuning.cu -o /tmp/l2_persist_tuning
// 用法：
//   ./l2_persist_tuning bench [iters]                      # §A/§B/§C 计时表（连续跑，L2 自然演化）
//   ./l2_persist_tuning one <mode> <S_MB> <f> <carveMB>    # 单窗口单次序列，供 ncu 采样
//   ./l2_persist_tuning one2 <S_MB> <fA> <carveMB> [fB]     # 双窗口双 stream 单次序列，供 ncu 采样
//     mode ∈ {pollute, base, win}（win 用 <f>/<carveMB>）

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

__global__ void init_fill(float4* a, long n4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float b = (float)(i % 977) * 0.001f - 0.5f;
    a[i] = make_float4(b, b + 0.1f, b + 0.2f, b + 0.3f);
  }
}

// K1: 读 x(nx4) → 写 y(ny4)。用于写中间量。
__global__ void l2_k1(const float4* __restrict__ x, float4* __restrict__ y,
                      long nx4, long ny4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < nx4; i += stride) {
    float4 v = x[i];
    if (i < ny4) y[i] = op_f(v);
  }
}

// 双窗口场景：K1a 写 y1，K1b 写 y2（各自尺寸 ny4）。
__global__ void l2_k1a(const float4* __restrict__ x, float4* __restrict__ y,
                       long ny4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < ny4; i += stride) y[i] = op_f(x[i]);
}
__global__ void l2_k1b(const float4* __restrict__ x, float4* __restrict__ y,
                       long ny4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < ny4; i += stride) y[i] = op_f(x[i + 1024]);  // 错开一点，避免同址
}

// polluter: 流式读+写大缓冲，把 L2 冲干净。
__global__ void l2_pollute(float4* __restrict__ p, long n4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) p[i] = op_g(p[i]);
}

// K2: 读 y(ny4) → 写 z(ny4)。量它的 DRAM 读字节 = 中间量 y 命中 L2 的程度。
__global__ void l2_k2(const float4* __restrict__ y, float4* __restrict__ z,
                      long ny4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < ny4; i += stride) z[i] = op_g(y[i]);
}
// 双窗口场景的两个 K2（命名区分，供 ncu 分别采样）。
__global__ void l2_k2a(const float4* __restrict__ y, float4* __restrict__ z,
                       long ny4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < ny4; i += stride) z[i] = op_g(y[i]);
}
__global__ void l2_k2b(const float4* __restrict__ y, float4* __restrict__ z,
                       long ny4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < ny4; i += stride) z[i] = op_g(y[i]);
}

static int g_threads = 256;
static int g_grid = 0;

static int pick_grid() {
  int dev; CUDA_CHECK(cudaGetDevice(&dev));
  int nsm; CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  int perSM; CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &perSM, (const void*)l2_k2, g_threads, 0));
  if (perSM < 1) perSM = 1;
  return nsm * perSM;
}

// 给 stream st 挂 y 的 persisting 窗口：hitRatio=f，carveout 已由调用者设过。
static void set_window(cudaStream_t st, void* base, size_t bytes, float f) {
  cudaStreamAttrValue attr;
  memset(&attr, 0, sizeof(attr));
  attr.accessPolicyWindow.base_ptr = base;
  attr.accessPolicyWindow.num_bytes = bytes;
  attr.accessPolicyWindow.hitRatio = f;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  CUDA_CHECK(cudaStreamSetAttribute(st, cudaStreamAttributeAccessPolicyWindow, &attr));
}

static void clear_window(cudaStream_t st) {
  cudaStreamAttrValue attr;
  memset(&attr, 0, sizeof(attr));
  attr.accessPolicyWindow.num_bytes = 0;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  CUDA_CHECK(cudaStreamSetAttribute(st, cudaStreamAttributeAccessPolicyWindow, &attr));
  CUDA_CHECK(cudaCtxResetPersistingL2Cache());
}

// 单窗口序列单步：可选 window（f>=0 表示挂），K1 → polluter(可选) → K2。
static void run_single(cudaStream_t st, const float4* x, float4* y, float4* z,
                       float4* p, long nx4, long ny4, long np4,
                       bool do_pollute, bool do_window, float f) {
  if (do_window) set_window(st, (void*)y, (size_t)ny4 * FLOAT4, f);
  else clear_window(st);
  l2_k1<<<g_grid, g_threads, 0, st>>>(x, y, nx4, ny4);
  if (do_pollute) l2_pollute<<<g_grid, g_threads, 0, st>>>(p, np4);
  l2_k2<<<g_grid, g_threads, 0, st>>>(y, z, ny4);
}

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
         prop.l2CacheSize / (int)BYTES_PER_MB, maxPersist / (int)BYTES_PER_MB,
         maxWindow / (int)BYTES_PER_MB);
  printf("grid=%d (threads=%d, occupancy-capped)\n", g_grid, g_threads);

  const size_t XMB = 256, ZMB = 256, PMB = 512;
  size_t xb = XMB * BYTES_PER_MB, zb = ZMB * BYTES_PER_MB, pb = PMB * BYTES_PER_MB;

  float *xa, *ya, *za, *pa, *yb, *z2;
  CUDA_CHECK(cudaMalloc(&xa, xb));
  CUDA_CHECK(cudaMalloc(&ya, 128 * BYTES_PER_MB));  // y1 / 单窗口 y（最大 64MB 够）
  CUDA_CHECK(cudaMalloc(&yb, 128 * BYTES_PER_MB));  // y2（双窗口）
  CUDA_CHECK(cudaMalloc(&za, zb));
  CUDA_CHECK(cudaMalloc(&z2, zb));                   // 双窗口场景 K2b 的独立输出（避免与 K2a 共享 z）
  CUDA_CHECK(cudaMalloc(&pa, pb));

  cudaStream_t st, stA, stB;
  CUDA_CHECK(cudaStreamCreate(&st));
  CUDA_CHECK(cudaStreamCreate(&stA));
  CUDA_CHECK(cudaStreamCreate(&stB));
  long nx4 = (long)(xb / FLOAT4);
  long np4 = (long)(pb / FLOAT4);
  init_fill<<<g_grid, g_threads, 0, st>>>((float4*)xa, nx4);
  init_fill<<<g_grid, g_threads, 0, st>>>((float4*)pa, np4);
  init_fill<<<g_grid, g_threads, 0, st>>>((float4*)ya, (long)(128 * BYTES_PER_MB / FLOAT4));
  init_fill<<<g_grid, g_threads, 0, st>>>((float4*)yb, (long)(128 * BYTES_PER_MB / FLOAT4));
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---------------- one：单窗口，供 ncu ----------------
  if (strcmp(mode, "one") == 0) {
    int SMB = (argc > 3) ? atoi(argv[3]) : 16;
    const char* m = (argc > 2) ? argv[2] : "win";
    float f = (argc > 4) ? (float)atof(argv[4]) : 1.0f;
    int carveMB = (argc > 5) ? atoi(argv[5]) : SMB;
    size_t carve = (size_t)carveMB * BYTES_PER_MB;
    if (carve > maxPersist) carve = maxPersist;
    long ny4 = (long)((size_t)SMB * BYTES_PER_MB / FLOAT4);
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, carve));
    bool doP = (strcmp(m, "base") != 0);
    bool doW = (strcmp(m, "win") == 0);
    run_single(st, (const float4*)xa, (float4*)ya, (float4*)za, (float4*)pa, nx4, ny4, np4, doP, doW, f);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("one mode=%s S=%dMB f=%.2f carve=%zuMB done\n", m, SMB, f, carve / BYTES_PER_MB);
  }
  // ---------------- oneB：只用 stB 的对照（诊断 stA vs stB） ----------------
  else if (strcmp(mode, "oneB") == 0) {
    int SMB = (argc > 2) ? atoi(argv[2]) : 8;
    float f = (argc > 3) ? (float)atof(argv[3]) : 1.0f;
    int carveMB = (argc > 4) ? atoi(argv[4]) : 16;
    size_t carve = (size_t)carveMB * BYTES_PER_MB;
    if (carve > maxPersist) carve = maxPersist;
    long ny4 = (long)((size_t)SMB * BYTES_PER_MB / FLOAT4);
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, carve));
    run_single(stB, (const float4*)xa, (float4*)yb, (float4*)za, (float4*)pa, nx4, ny4, np4, true, true, f);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("oneB S=%dMB f=%.2f carve=%zuMB done\n", SMB, f, carve / BYTES_PER_MB);
  }
  // ---------------- one2：双窗口双 stream，供 ncu ----------------
  else if (strcmp(mode, "one2") == 0) {
    int SMB = (argc > 2) ? atoi(argv[2]) : 8;
    float f = (argc > 3) ? (float)atof(argv[3]) : 1.0f;
    int carveMB = (argc > 4) ? atoi(argv[4]) : (int)(maxPersist / BYTES_PER_MB);
    float fb = (argc > 5) ? (float)atof(argv[5]) : f;  // stB 的 hitRatio（单独可调）
    size_t carve = (size_t)carveMB * BYTES_PER_MB;
    if (carve > maxPersist) carve = maxPersist;
    long ny4 = (long)((size_t)SMB * BYTES_PER_MB / FLOAT4);
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, carve));
    // 两个窗口：stA→y1（f），stB→y2（fb）；<0 表示该 stream 不挂窗口。
    // order=b：先设 stB 再设 stA（验证「先设/后设哪个窗口生效」）。
    // 注意：one2 是 fresh process，stream 本就干净，**不要**给它挂 num_bytes=0 的空窗口
    //（空窗口本身会被驱动登记为一次 access-policy 设置，可能干扰另一条 stream 的窗口——诊断发现）。
    const char* order = (argc > 6) ? argv[6] : "a";
    if (order[0] == 'b') {
      if (fb >= 0.0f) set_window(stB, (void*)yb, (size_t)ny4 * FLOAT4, fb);
      if (f >= 0.0f) set_window(stA, (void*)ya, (size_t)ny4 * FLOAT4, f);
    } else {
      if (f >= 0.0f) set_window(stA, (void*)ya, (size_t)ny4 * FLOAT4, f);
      if (fb >= 0.0f) set_window(stB, (void*)yb, (size_t)ny4 * FLOAT4, fb);
    }
    cudaEvent_t e1a, e1b, eP, eK2a;
    CUDA_CHECK(cudaEventCreate(&e1a)); CUDA_CHECK(cudaEventCreate(&e1b));
    CUDA_CHECK(cudaEventCreate(&eP)); CUDA_CHECK(cudaEventCreate(&eK2a));
    l2_k1a<<<g_grid, g_threads, 0, stA>>>((const float4*)xa, (float4*)ya, ny4);
    CUDA_CHECK(cudaEventRecord(e1a, stA));
    l2_k1b<<<g_grid, g_threads, 0, stB>>>((const float4*)xa, (float4*)yb, ny4);
    CUDA_CHECK(cudaEventRecord(e1b, stB));
    CUDA_CHECK(cudaStreamWaitEvent(st, e1a));
    CUDA_CHECK(cudaStreamWaitEvent(st, e1b));
    l2_pollute<<<g_grid, g_threads, 0, st>>>((float4*)pa, np4);
    CUDA_CHECK(cudaEventRecord(eP, st));
    CUDA_CHECK(cudaStreamWaitEvent(stA, eP));
    l2_k2a<<<g_grid, g_threads, 0, stA>>>((const float4*)ya, (float4*)za, ny4);
    CUDA_CHECK(cudaEventRecord(eK2a, stA));
    CUDA_CHECK(cudaStreamWaitEvent(stB, eP));
    CUDA_CHECK(cudaStreamWaitEvent(stB, eK2a));
    l2_k2b<<<g_grid, g_threads, 0, stB>>>((const float4*)yb, (float4*)z2, ny4);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("one2 S=%dMB(each) f=%.2f fb=%.2f carve=%zuMB done\n", SMB, f, fb, carve / BYTES_PER_MB);
  }
  // ---------------- bench ----------------
  else {
    int iters = (argc > 2) ? atoi(argv[2]) : 30;
    printf("iters=%d, X=%zu MB, Z=%zu MB, polluter=%zu MB（L2=%d MB）\n\n",
           iters, XMB, ZMB, PMB, prop.l2CacheSize / (int)BYTES_PER_MB);

    // 计时：前导（K1 + 可选 polluter，窗口已挂好） + 只夹 K2 的 event。
    auto time_k2b = [&](int SMB, bool doP, bool doW, float f, int carveMB) -> double {
      size_t carve = (size_t)carveMB * BYTES_PER_MB;
      if (carve > maxPersist) carve = maxPersist;
      CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, carve));
      long ny4 = (long)((size_t)SMB * BYTES_PER_MB / FLOAT4);
      auto pre = [&]() {
        if (doW) set_window(st, (void*)ya, (size_t)ny4 * FLOAT4, f);
        else clear_window(st);
        l2_k1<<<g_grid, g_threads, 0, st>>>((const float4*)xa, (float4*)ya, nx4, ny4);
        if (doP) l2_pollute<<<g_grid, g_threads, 0, st>>>((float4*)pa, np4);
      };
      for (int w = 0; w < 3; ++w) pre();
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaEvent_t a, b;
      CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
      float total = 0;
      for (int r = 0; r < iters; ++r) {
        pre();
        CUDA_CHECK(cudaEventRecord(a, st));
        l2_k2<<<g_grid, g_threads, 0, st>>>((const float4*)ya, (float4*)za, ny4);
        CUDA_CHECK(cudaEventRecord(b, st));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        total += ms;
      }
      CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
      return total / iters;
    };

    // ---- §A：S=16MB，carveout=16MB，hitRatio 扫描 ----
    {
      int SMB = 16, carveMB = 16;
      double t_base = time_k2b(SMB, false, false, 0, 0);   // 无 polluter（免费 L2 参照）
      double t_pol  = time_k2b(SMB, true,  false, 0, 0);   // polluter，无窗口（基准）
      printf("== A. 概率化分区：S=%d MB, carveout=%d MB, polluter=on ==\n", SMB, carveMB);
      printf("   t_pollute=%.4f ms, t_base(no polluter)=%.4f ms (base/pollute=%.3f)\n",
             t_pol, t_base, t_base / t_pol);
      printf("   hitRatio |  K2 ms  | vs pollute\n");
      printf("   ---------+---------+-----------\n");
      const float fs[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
      for (float f : fs) {
        double t = time_k2b(SMB, true, true, f, carveMB);
        printf("   %8.2f | %7.4f | %9.3f\n", f, t, t / t_pol);
      }
      printf("\n");
    }

    // ---- §B：carveout × hitRatio 联合（含 carveout 拉满 pfull 情形）----
    {
      int SMB = 16;
      double t_pol = 0;  // 复算基准
      t_pol = time_k2b(SMB, true, false, 0, 0);
      printf("== B. carveout × hitRatio 联合：S=%d MB, polluter=on (t_pollute=%.4f ms) ==\n", SMB, t_pol);
      printf("   carveout |   f=0.00  |   f=0.25  |   f=0.50  |   f=0.75  |   f=1.00\n");
      printf("   ---------+-----------+-----------+-----------+-----------+----------\n");
      const int carves[] = {4, 8, 16, 31};
      const float fs[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
      for (int c : carves) {
        printf("   %5d MB |", c);
        for (float f : fs) {
          double t = time_k2b(SMB, true, true, f, c);
          printf(" %9.3f |", t / t_pol);
        }
        printf("\n");
      }
      printf("\n");
    }

    // ---- §C：多窗口/多流共存：y1、y2 各 8 MB，共享一个 carveout ----
    {
      int SMB = 8;  // 每个 8 MB，共 16 MB
      long ny4 = (long)((size_t)SMB * BYTES_PER_MB / FLOAT4);

      struct Pair { double a, b; };

      // 双窗口一次序列：K1a/K1b 写 → polluter → K2a/K2b 串行读（各自窗口由 stream 决定）。
      auto time_twin = [&](int carveMB, float fa, float fb) -> Pair {
        size_t carve = (size_t)carveMB * BYTES_PER_MB;
        if (carve > maxPersist) carve = maxPersist;
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, carve));
        clear_window(stA); clear_window(stB);
        if (fa >= 0.0f) set_window(stA, (void*)ya, (size_t)ny4 * FLOAT4, fa);
        if (fb >= 0.0f) set_window(stB, (void*)yb, (size_t)ny4 * FLOAT4, fb);
        cudaEvent_t e1a, e1b, eP, a, b, a2, b2;
        CUDA_CHECK(cudaEventCreate(&e1a)); CUDA_CHECK(cudaEventCreate(&e1b));
        CUDA_CHECK(cudaEventCreate(&eP));
        CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
        CUDA_CHECK(cudaEventCreate(&a2)); CUDA_CHECK(cudaEventCreate(&b2));
        auto pre = [&]() {
          l2_k1a<<<g_grid, g_threads, 0, stA>>>((const float4*)xa, (float4*)ya, ny4);
          CUDA_CHECK(cudaEventRecord(e1a, stA));
          l2_k1b<<<g_grid, g_threads, 0, stB>>>((const float4*)xa, (float4*)yb, ny4);
          CUDA_CHECK(cudaEventRecord(e1b, stB));
          CUDA_CHECK(cudaStreamWaitEvent(st, e1a));
          CUDA_CHECK(cudaStreamWaitEvent(st, e1b));
          l2_pollute<<<g_grid, g_threads, 0, st>>>((float4*)pa, np4);
          CUDA_CHECK(cudaEventRecord(eP, st));
        };
        for (int w = 0; w < 3; ++w) pre();
        CUDA_CHECK(cudaDeviceSynchronize());
        double totA = 0, totB = 0;
        for (int r = 0; r < iters; ++r) {
          pre();
          CUDA_CHECK(cudaStreamWaitEvent(stA, eP));
          CUDA_CHECK(cudaEventRecord(a, stA));
          l2_k2a<<<g_grid, g_threads, 0, stA>>>((const float4*)ya, (float4*)za, ny4);
          CUDA_CHECK(cudaEventRecord(b, stA));
          CUDA_CHECK(cudaStreamWaitEvent(stB, eP));
          CUDA_CHECK(cudaStreamWaitEvent(stB, b));
          CUDA_CHECK(cudaEventRecord(a2, stB));
          l2_k2b<<<g_grid, g_threads, 0, stB>>>((const float4*)yb, (float4*)z2, ny4);
          CUDA_CHECK(cudaEventRecord(b2, stB));
          CUDA_CHECK(cudaEventSynchronize(b2));
          float msA = 0, msB = 0;
          CUDA_CHECK(cudaEventElapsedTime(&msA, a, b));
          CUDA_CHECK(cudaEventElapsedTime(&msB, a2, b2));
          totA += msA; totB += msB;
        }
        CUDA_CHECK(cudaEventDestroy(e1a)); CUDA_CHECK(cudaEventDestroy(e1b));
        CUDA_CHECK(cudaEventDestroy(eP));
        CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
        CUDA_CHECK(cudaEventDestroy(a2)); CUDA_CHECK(cudaEventDestroy(b2));
        return Pair{totA / iters, totB / iters};
      };

      Pair p_none = time_twin(0, -1.0f, -1.0f);   // 无窗口，都落 HBM（基准）
      double base = p_none.a + p_none.b;
      printf("== C. 多窗口/多 stream 共存：y1=%d MB(sA), y2=%d MB(sB)（总需求 %d MB）==\n",
             SMB, SMB, 2 * SMB);
      printf("   场景                        | K2a ms | K2b ms | vs 无窗口(和)\n");
      printf("   ----------------------------+--------+--------+--------------\n");
      auto row = [&](const char* name, Pair p) {
        printf("   %-26s | %6.4f | %6.4f | %12.3f\n", name, p.a, p.b, (p.a + p.b) / base);
      };
      row("no window", p_none);
      row("carve=16 y1 only f=1", time_twin(16, 1.0f, -1.0f));
      row("carve=16 both f=1.0", time_twin(16, 1.0f, 1.0f));
      row("carve=16 both f=0.5", time_twin(16, 0.5f, 0.5f));
      row("carve=8  both f=1.0", time_twin(8, 1.0f, 1.0f));
      row("carve=8  both f=0.5", time_twin(8, 0.5f, 0.5f));
      row("carve=31 both f=1.0", time_twin(31, 1.0f, 1.0f));
      printf("   (注：多窗口拆分 DRAM 读/命中率见 one2 的 ncu 采样)\n\n");
    }

    // ---- 正确性：有无窗口数值一致 ----
    {
      long ny4 = (long)(16 * BYTES_PER_MB / FLOAT4);
      CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
      run_single(st, (const float4*)xa, (float4*)ya, (float4*)za, (float4*)pa, nx4, ny4, np4, true, false, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      float* h1 = (float*)malloc((size_t)ny4 * FLOAT4);
      CUDA_CHECK(cudaMemcpy(h1, za, (size_t)ny4 * FLOAT4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, maxPersist));
      run_single(st, (const float4*)xa, (float4*)ya, (float4*)za, (float4*)pa, nx4, ny4, np4, true, true, 1.0f);
      CUDA_CHECK(cudaDeviceSynchronize());
      float* h2 = (float*)malloc((size_t)ny4 * FLOAT4);
      CUDA_CHECK(cudaMemcpy(h2, za, (size_t)ny4 * FLOAT4, cudaMemcpyDeviceToHost));
      double maxd = 0;
      for (long i = 0; i < ny4 * 4; ++i) {
        double d = fabs((double)h1[i] - (double)h2[i]);
        if (d > maxd) maxd = d;
      }
      printf("correctness (S=16MB): max|z_nowin - z_win| = %.3e (expect 0)\n", maxd);
      free(h1); free(h2);
    }

    printf("\n>> 读法：\n");
    printf(">>  §A：固定 carveout，hitRatio f 越小 → 去挤 carveout 的访问越少 → 保护量应≈f×S；\n");
    printf(">>      若 f<1 时 K2 仍快，说明「概率化分区」能少占 carveout 而近似保护。\n");
    printf(">>  §B：pfull（carveout=31MB）的 ~1.5× 代价，能否用「更小 carveout + hitRatio<1」收回。\n");
    printf(">>  §C：两个并发中间量共享一个 carveout —— 拆分 DRAM 读用 `one2` 的 ncu 采样。\n");
  }

  CUDA_CHECK(cudaFree(xa)); CUDA_CHECK(cudaFree(ya)); CUDA_CHECK(cudaFree(yb));
  CUDA_CHECK(cudaFree(za)); CUDA_CHECK(cudaFree(z2)); CUDA_CHECK(cudaFree(pa));
  CUDA_CHECK(cudaStreamDestroy(st)); CUDA_CHECK(cudaStreamDestroy(stA)); CUDA_CHECK(cudaStreamDestroy(stB));
  return 0;
}
