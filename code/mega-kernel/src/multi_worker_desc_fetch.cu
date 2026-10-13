// multi_worker_desc_fetch.cu — MegaKernel 专题 P51-1 小实验：
//   **多 worker（多 block）取 task 的描述符带宽标度 + `cp.async` 的「TMA/mbarrier」形态值多少？**
//
// 动机 / 接续 P50-1（docs/04 §39）：
//   §39 把 MPK worker「取 task」的四条取数路径（scalar / vec4 / cp.async 单缓冲 / cp.async 双缓冲预取）
//   在**单 block** 上量了一遍，核心结论：`cp.async` 的成本与描述符大小无关，真实 352 B 下比寄存器读快
//   2.3×，加预取再收 7–40%。但 §39.7 明确留了两条尾巴：
//     (1) 「**单 block** ⇒ 测的是『取一批的延迟』而非『访存带宽』；绝对 GB/s（2–6 GB/s）是单 SM/单 block
//          口径，远低于 HBM。要多 worker 的带宽标度需另做多 block 实验。」
//     (2) 「**`mbarrier` 形态未测**：`cp.async` 用的是 `commit_group`/`wait_group` 软件计数，未测
//          `cp.async.mbarrier.arrive` + 硬件 mbarrier（TMA 风格）——那是 §19/§30 的 TMA 线的延续。」
//
//   本节把这两条尾巴一起补掉：
//     - 把「消费者」从 1 个 block 扩到 **G 个 worker block**（真·多 worker），量 **aggregate 带宽**随 G
//       的标度、看它在多少 G 处饱和（对上 MPK §5.1「128 worker / 16 scheduler，SM 数匹配」，快照行
//       1168；以及论文 §6.1 的 worker 数）。
//     - 增加两条 **TMA bulk + 硬件 mbarrier** 路径（mode4 单缓冲 / mode5 双缓冲）：单线程发一条
//       `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` 把**整批**描述符一次搬进 SMEM，
//       由硬件 mbarrier 的 `complete_tx` 记字节、`try_wait.parity` 等完成——与 §39 的 16 B `cp.async`
//       （`commit_group`/`wait_group` 软件计数）对照。
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：
//   - worker 取 task：所有线程协作 16 B `cp.async` 批量搬 `TaskDesc` 进 SMEM 单缓冲
//     `cp_async_fence(); cp_async_wait<0>(); __syncthreads();`（`persistent_kernel.cuh:1063-1078`）；
//     `load_smem` = `cp.async.ca.shared.global.L2::128B`（`tasks/common/copy_sm80.cuh:44-57`）。
//   - 硬件 mbarrier + TMA bulk 的形态在快照里用在 **task 单体内部**（TensorMap 路径）：
//     `cp.async.bulk.tensor...mbarrier::complete_tx::bytes`（`tasks/hopper/tma_2d.cuh:119`）、
//     mbarrier 初始化/`arrive.expect_tx`/`try_wait.parity`（`tasks/hopper/barrier.cuh:42/59/75`）。
//     本节把它**搬到「取 task 描述符」这一层**做对照（描述符队列本身是连续字节，正好是 1D bulk 的用武之地）。
//   - `TaskDesc` 353 B 级：论文 §6.1 行 1365 明写「each description occupies 352 bytes」；
//     论文 §5.3「Pre-fetching task descriptions」（行 1364-1368）。
//
// 核心问题：
//   Q1（§A）六种取数路径 checksum 是否一致（多 worker）？
//   Q2（§B）**多 worker 标度**：aggregate GB/s 随 G 的标度、饱和点；per-worker 有效带宽。
//   Q3（§C）**TMA bulk + mbarrier** 相对 16 B `cp.async`：批量大小 K 的影响。
//   Q4（§D）`DW`（描述符大小，含真实 352 B）下的标度。
//
// 机制：
//   G 个常驻消费者 block（worker 微缩）。全局 `NDESC` 条自描述描述符（每条 `DW` 字，总计 ≫ L2），
//   按 **batch**（每批 K 条、连续）round-robin 分给 G 个 block（block b 处理 bb=b, b+G, b+2G, …），
//   每个 batch 内折叠成 checksum。唯一变量 = 取数路径 MODE：
//     0 scalar     : 每 lane 逐 4 B `ld.relaxed.gpu.u32` 读（§38 warp 形态）
//     1 vec4       : 每 lane 128-bit `uint4` 读
//     2 cpasync    : 全线程协作 16 B `cp.async` 进 SMEM 单缓冲 → commit/wait_all/syncthreads → 校验
//     3 cpasync_pf : 双缓冲软件流水（`cp.async.wait_group 1`）= §39 的 pre-fetch
//     4 bulk       : 单线程发一条 `cp.async.bulk`（整批一次）+ 硬件 mbarrier 等完成（TMA 风格）
//     5 bulk_pf    : TMA bulk 双缓冲（两个 mbarrier 交替）
//
// 安全：无生产者/消费者竞态（数组预写），无等待死循环（bulk 的 mbarrier 期望字节恒能被 complete_tx
//   满足；若 asm 写错会 bad≠0 或挂起，脚本带 timeout）。
//
// 编译：
//   nvcc -O3 -arch=sm_90a src/multi_worker_desc_fetch.cu -o /tmp/multi_worker_desc_fetch
// 用法：
//   ./multi_worker_desc_fetch run <G> <NDESC> <DW> <K> <mode> <NLANES> <reps>
//   ./multi_worker_desc_fetch bench

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
      printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__host__ __device__ __forceinline__ uint32_t mix(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// 与 MPK load_smem 逐字一致的 16 B cp.async（copy_sm80.cuh:44-57）
__device__ __forceinline__ void load_smem(void* smem_ptr, const void* gmem_ptr) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16, 16;\n" ::"r"(s),
               "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_wait() {
  if constexpr (N == 0)
    asm volatile("cp.async.wait_all;\n" ::);
  else
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ uint32_t ld_relaxed_u32(const uint32_t* a) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.u32 %0,[%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}

// ---- 硬件 mbarrier + TMA bulk 1D（barrier.cuh / tma_2d.cuh 的形，搬到取数与层） ----
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "MBW_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra.uni MBD_%=;\n"
      "bra.uni MBW_%=;\n"
      "MBD_%=:\n"
      "}\n" ::"r"(a),
      "r"(phase));
}
__device__ __forceinline__ void bulk_1d(void* smem_ptr, const void* gmem_ptr,
                                        uint32_t bytes, uint64_t* bar) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  uint32_t m = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], "
      "%2, [%3];\n" ::"r"(s),
      "l"(gmem_ptr),
      "r"(bytes),
      "r"(m)
      : "memory");
}

__device__ __forceinline__ uint32_t fold_words(const uint32_t* p, int DW) {
  uint32_t x = 0;
  for (int i = 0; i < DW; ++i) x ^= p[i];
  return x;
}
__device__ __forceinline__ uint32_t fold_vec4(const uint32_t* p, int DW) {
  uint32_t x = 0;
  const uint4* q = reinterpret_cast<const uint4*>(p);
  int nv = DW / 4;
  for (int i = 0; i < nv; ++i) {
    uint4 v = q[i];
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  for (int i = nv * 4; i < DW; ++i) x ^= p[i];
  return x;
}

// =====================================================================================
// 取数 kernel：唯一变量 = 取数路径 MODE；多 worker = gridDim.x 个 block
//   batch 轮转：block b 处理 batch bb = b, b+G, b+2G, ...（每 batch K 条连续描述符）
// =====================================================================================
template <int MODE>
__global__ void fetch_kernel(int NDESC, int DW, int K, int reps,
                             const uint32_t* __restrict__ descs,
                             unsigned long long* __restrict__ sink) {
  extern __shared__ __align__(16) uint32_t smem[];
  const int lane = threadIdx.x;
  const int nlanes = blockDim.x;
  const int G = gridDim.x;
  const int nbatch = NDESC / K;
  const int bytes_per_batch = K * DW * 4;
  unsigned long long acc = 0;

  const bool is_bulk = (MODE == 4 || MODE == 5);
  __shared__ uint64_t mbar[2];
  if (is_bulk && lane == 0) {
    mbar_init(&mbar[0], 1);
    if (MODE == 5) mbar_init(&mbar[1], 1);
  }
  if (is_bulk) __syncthreads();

  for (int rep = 0; rep < reps; ++rep) {
    if (MODE == 0 || MODE == 1) {
      for (int bb = blockIdx.x; bb < nbatch; bb += G) {
        const uint32_t* base = descs + (size_t)bb * K * DW;
        for (int j = lane; j < K; j += nlanes) {
          const uint32_t* p = base + (size_t)j * DW;
          acc += (MODE == 0) ? fold_words(p, DW) : fold_vec4(p, DW);
        }
      }
    } else if (MODE == 2) {
      const int chunks = K * DW / 4;
      for (int bb = blockIdx.x; bb < nbatch; bb += G) {
        const uint32_t* base = descs + (size_t)bb * K * DW;
        for (int i = lane; i < chunks; i += nlanes)
          load_smem(smem + (size_t)i * 4, base + (size_t)i * 4);
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        for (int j = lane; j < K; j += nlanes)
          acc += fold_words(smem + (size_t)j * DW, DW);
        __syncthreads();
      }
    } else if (MODE == 3) {
      const int chunks = K * DW / 4;
      uint32_t* buf[2] = {smem, smem + (size_t)K * DW};
      // 本 block 的 batch 序列：bb_t = blockIdx.x + t*G
      auto issue = [&](int t, int slot) {
        int bb = blockIdx.x + t * G;
        const uint32_t* base = descs + (size_t)bb * K * DW;
        for (int i = lane; i < chunks; i += nlanes)
          load_smem(buf[slot] + (size_t)i * 4, base + (size_t)i * 4);
        cp_commit();
      };
      const int nt = (nbatch - blockIdx.x + G - 1) / G;  // 本 block 的 batch 数
      if (nt > 0) issue(0, 0);
      for (int t = 0; t < nt; ++t) {
        // 最后一批不再预取 ⇒ 在飞组数为 0，必须 wait<0>；否则 wait<1> 会立即返回、读到未完成的数据
        if (t + 1 < nt) {
          issue(t + 1, (t + 1) & 1);
          cp_wait<1>();  // 至多 1 组（t+1）在飞 ⇒ 第 t 组必已完成
        } else {
          cp_wait<0>();
        }
        __syncthreads();
        for (int j = lane; j < K; j += nlanes)
          acc += fold_words(buf[t & 1] + (size_t)j * DW, DW);
        __syncthreads();
      }
    } else if (MODE == 4) {
      // TMA bulk 单缓冲 + 硬件 mbarrier
      uint32_t phase = 0;
      for (int bb = blockIdx.x; bb < nbatch; bb += G) {
        const uint32_t* base = descs + (size_t)bb * K * DW;
        if (lane == 0) {
          mbar_expect_tx(&mbar[0], (uint32_t)bytes_per_batch);
          bulk_1d(smem, base, (uint32_t)bytes_per_batch, &mbar[0]);
        }
        mbar_wait(&mbar[0], phase);
        phase ^= 1;
        __syncthreads();
        for (int j = lane; j < K; j += nlanes)
          acc += fold_words(smem + (size_t)j * DW, DW);
        __syncthreads();
      }
    } else {
      // MODE==5：TMA bulk 双缓冲（两个 mbarrier 交替）
      uint32_t* buf[2] = {smem, smem + (size_t)K * DW};
      uint32_t phase[2] = {0, 0};
      const int nt = (nbatch - blockIdx.x + G - 1) / G;
      auto issue = [&](int t, int slot) {
        int bb = blockIdx.x + t * G;
        const uint32_t* base = descs + (size_t)bb * K * DW;
        if (lane == 0) {
          mbar_expect_tx(&mbar[slot], (uint32_t)bytes_per_batch);
          bulk_1d(buf[slot], base, (uint32_t)bytes_per_batch, &mbar[slot]);
        }
      };
      if (nt > 0) issue(0, 0);
      for (int t = 0; t < nt; ++t) {
        if (t + 1 < nt) issue(t + 1, (t + 1) & 1);
        int slot = t & 1;
        mbar_wait(&mbar[slot], phase[slot]);
        phase[slot] ^= 1;
        __syncthreads();
        for (int j = lane; j < K; j += nlanes)
          acc += fold_words(buf[slot] + (size_t)j * DW, DW);
        __syncthreads();
      }
    }
  }
  if (acc) atomicAdd(sink, acc);
}

// =====================================================================================
// host
// =====================================================================================
__global__ void fill_kernel(uint32_t* descs, long long NDESC, int DW) {
  long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= NDESC) return;
  uint32_t* p = descs + id * DW;
  p[0] = (uint32_t)id;
  if (DW > 1) p[1] = (uint32_t)(id % 4);
  if (DW > 2) p[2] = (uint32_t)(id / 4);
  if (DW > 3) p[3] = 0u;
  for (int i = 4; i < DW; ++i) p[i] = mix((uint32_t)id, (uint32_t)i);
}

static unsigned long long host_expected(long long NDESC, int DW) {
  unsigned long long s = 0;
  for (long long id = 0; id < NDESC; ++id) {
    uint32_t x = (uint32_t)id;
    if (DW > 1) x ^= (uint32_t)(id % 4);
    if (DW > 2) x ^= (uint32_t)(id / 4);
    if (DW > 3) x ^= 0u;
    for (int i = 4; i < DW; ++i) x ^= mix((uint32_t)id, (uint32_t)i);
    s += x;
  }
  return s;
}

static const char* mode_name(int m) {
  switch (m) {
    case 0: return "scalar";
    case 1: return "vec4";
    case 2: return "cpasync";
    case 3: return "cpasync_pf";
    case 4: return "bulk";
    default: return "bulk_pf";
  }
}

static int run_fetch(int G, long long NDESC, int DW, int K, int mode, int nlanes,
                     int reps) {
  NDESC = (NDESC / K) * K;
  uint32_t* descs;
  unsigned long long* sink;
  CK(cudaMalloc(&descs, (size_t)NDESC * DW * 4));
  CK(cudaMalloc(&sink, 8));
  fill_kernel<<<(int)((NDESC + 255) / 256), 256>>>(descs, NDESC, DW);
  CK(cudaDeviceSynchronize());

  int smem_words = (mode == 3 || mode == 5) ? 2 * K * DW : K * DW;
  size_t smem_bytes = (size_t)smem_words * 4;

  auto setattr = [&](auto fn) {
    if (smem_bytes > 48 * 1024)
      CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              (int)smem_bytes));
  };
  setattr(fetch_kernel<0>); setattr(fetch_kernel<1>); setattr(fetch_kernel<2>);
  setattr(fetch_kernel<3>); setattr(fetch_kernel<4>); setattr(fetch_kernel<5>);

  unsigned long long exp = host_expected(NDESC, DW);
  unsigned long long got = 0;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float ms_accum = 0.f;
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(sink, 0, 8));
    CK(cudaEventRecord(e0));
    switch (mode) {
      case 0: fetch_kernel<0><<<G, nlanes, smem_bytes>>>(NDESC, DW, K, 1, descs, sink); break;
      case 1: fetch_kernel<1><<<G, nlanes, smem_bytes>>>(NDESC, DW, K, 1, descs, sink); break;
      case 2: fetch_kernel<2><<<G, nlanes, smem_bytes>>>(NDESC, DW, K, 1, descs, sink); break;
      case 3: fetch_kernel<3><<<G, nlanes, smem_bytes>>>(NDESC, DW, K, 1, descs, sink); break;
      case 4: fetch_kernel<4><<<G, nlanes, smem_bytes>>>(NDESC, DW, K, 1, descs, sink); break;
      default: fetch_kernel<5><<<G, nlanes, smem_bytes>>>(NDESC, DW, K, 1, descs, sink); break;
    }
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    CK(cudaEventSynchronize(e1));
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1)); ms_accum += ms;
    CK(cudaMemcpy(&got, sink, 8, cudaMemcpyDeviceToHost));
  }
  double items = (double)NDESC * (double)reps;
  double ns_per_item = (ms_accum / 1e3) / items * 1e9;
  double gbps = (double)NDESC * (double)DW * 4.0 * (double)reps / (ms_accum / 1e3) / 1e9;
  int bad = (got == exp) ? 0 : 1;
  printf("  fetch %-11s G=%-4d K=%-4d DW=%-2d(%3dB) lanes=%-4d  bad=%d  ~%8.2f ns/item  %7.1f GB/s  (%.3f ms/rep)\n",
         mode_name(mode), G, K, DW, DW * 4, nlanes, bad, ns_per_item, gbps, ms_accum / reps);

  cudaFree(descs); cudaFree(sink);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return bad;
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  printf("device %d: %s  SMs=%d  cc=%d.%d  L2=%d MB\n", dev, prop.name,
         prop.multiProcessorCount, prop.major, prop.minor, (int)(prop.l2CacheSize >> 20));
  CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
  const int NSM = prop.multiProcessorCount;

  const char* cmd = argc > 1 ? argv[1] : "bench";
  if (strcmp(cmd, "run") == 0) {
    int G = argc > 2 ? atoi(argv[2]) : 132;
    long long NDESC = argc > 3 ? atoll(argv[3]) : 2000000;
    int DW = argc > 4 ? atoi(argv[4]) : 32;
    int K = argc > 5 ? atoi(argv[5]) : 32;
    int mode = argc > 6 ? atoi(argv[6]) : 0;
    int nlanes = argc > 7 ? atoi(argv[7]) : 128;
    int reps = argc > 8 ? atoi(argv[8]) : 5;
    run_fetch(G, NDESC, DW, K, mode, nlanes, reps);
    return 0;
  }

  // ---- bench：固定总字节 ≈ 256 MB（≫ 50 MB L2），NDESC 随 DW 反比 ----
  const long long BYTES = 256LL << 20;
  auto ndesc = [&](int DW) { return BYTES / (DW * 4); };

  printf("\n===== §A 正确性：六路径 checksum 一致（G=NSM, DW=32, K=32, lanes=128） =====\n");
  for (int m = 0; m < 6; ++m) run_fetch(NSM, ndesc(32), 32, 32, m, 128, 3);

  printf("\n===== §B 多 worker 标度（DW=32=128B, K=32, lanes=128, 256MB；扫 G） =====\n");
  for (int G : {1, 2, 4, 8, 16, 32, 64, NSM, 2 * NSM}) {
    for (int m : {1, 2, 3, 4, 5}) run_fetch(G, ndesc(32), 32, 32, m, 128, 5);
    printf("\n");
  }

  printf("\n===== §C TMA bulk+mbarrier vs 16B cp.async：批量 K 扫描（G=NSM, DW=32=128B, lanes=128） =====\n");
  for (int K : {1, 2, 4, 8, 16, 32, 64}) {
    for (int m : {2, 3, 4, 5}) run_fetch(NSM, ndesc(32), 32, K, m, 128, 5);
    printf("\n");
  }

  printf("\n===== §D 描述符大小 DW 扫描（G=NSM, K=32, lanes=128；DW=88=352B 对齐 MPK §6.1） =====\n");
  for (int DW : {8, 16, 32, 88}) {
    for (int m : {1, 2, 3, 4, 5}) run_fetch(NSM, ndesc(DW), DW, 32, m, 128, 5);
    printf("\n");
  }

  printf("\n===== §E worker 线程数（warp 数）标度（G=NSM, DW=32, K=32） =====\n");
  for (int L : {32, 128, 256, 512}) {
    for (int m : {2, 3, 4, 5}) run_fetch(NSM, ndesc(32), 32, 32, m, L, 5);
    printf("\n");
  }
  return 0;
}
