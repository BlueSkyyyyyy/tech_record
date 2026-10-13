// cpasync_desc_fetch.cu — MegaKernel 专题 P50-1 小实验：
//   **worker 侧「批量取 task」到底值多少钱？`cp.async` 批量加载 TaskDesc 进 SMEM 本身
//     相对寄存器/向量读的收益是多少？**
//
// 动机 / 接续：
//   P49-1（docs/04 §38）把 MPSC 队列升级为「批量出队 + 一个 warp 的并行校验 + 128 B 真实
//   `TaskDesc`」，把单线程消费者每项 ~2.3 µs 的往返压到 warp 的 ~300 ns/item。但 §38.7 明确留了
//   一条尾巴：
//     「描述符仍是合成布局……`cp.async` 只是等效语义（本节用普通 `ld.relaxed` 读，**未实测 `cp.async`
//       批量加载本身的收益**——那是另一条线）。未测……`warp` 下 `cp.async`/`mbarrier` 的批量加载形态。」
//
//   真实锚点（mirage 快照 `f9eb70c`，同一快照）：
//     - worker 拉取 task 时**不是逐条**：`threadIdx.x==0` 先读一个 task 区间，再用**所有线程协作**、
//       每次 16 B 的 `cp.async` 把 `TaskDesc` 批量搬进 shared memory：
//         `for (i = threadIdx.x; i < num_loaded_tasks * TASK_SIZE; i += blockDim.x)`
//             `kernel::load_smem(smem + i*16, all_tasks[...] + offset*16);`
//         `cp_async_fence(); cp_async_wait<0>(); __syncthreads();`
//       （`persistent_kernel.cuh:1063-1078`；`TASK_SIZE = sizeof(TaskDesc)/16`）
//     - `load_smem` 的真身是 `cp.async.ca.shared.global.L2::128B [smem], [gmem], 16, 16;`
//       （`tasks/common/copy_sm80.cuh:44-57`，注释 *"load BYTES (and prefetch 128 bytes) from global
//       to shared memory async"*）——即 16 B 直入 SMEM、带 `L2::128B` 预取提示、**绕过寄存器**。
//     - `TaskDesc`（`runtime_header.h:296-336`，`alignas(16)`）~128 B。
//
// 核心问题：
//   Q1（§A）四种取数路径（scalar / vec4 / cp.async 单缓冲 / cp.async 双缓冲预取）**数值是否一致**？
//   Q2（§B）**描述符大小（8/16/32 字 = 32/64/128 B）**下，四种路径每项成本如何标度？
//   Q3（§C）**批量大小 K** 的影响：`cp.async` 的「一次 wait」把 K 条一起等，K 多大才摊薄？
//   Q4（§D）**预取（双缓冲）**相对单缓冲能藏掉多少 `cp.async` 延迟？
//   Q5（§E）**线程数（warp 数）** 的标度，以及每项有效带宽（GB/s）。
//
// 机制：
//   单常驻 kernel、单 block（持久化消费者微缩）。全局有一大块**已写好**的描述符数组 `descs`
//   （`NDESC` 条，每条 `DW` 字，自描述：`w0=id`、`w1=id%4`、`w2=id/4`、`w3=0`、`w[i>=4]=mix(id,i)`），
//   数组整体 ≫ L2，故每轮都是 DRAM 读。消费者反复按「批」把描述符取进来做校验（XOR 折叠成 checksum，
//   防止被优化掉、并模拟「按 task id 路由 + 读 payload」）。唯一变量是取数路径：
//     0 `scalar` : 每 lane 逐 4B `ld.relaxed.gpu.u32` 读（§38 warp 形态）
//     1 `vec4`   : 每 lane 用 128-bit `uint4` 读（寄存器向量化）
//     2 `cpasync`: 所有线程协作 16B `cp.async` 进 SMEM 单缓冲 → commit/wait_all/syncthreads → 从 SMEM 校验
//     3 `cpasync_pf`: 双缓冲软件流水（预取第 b+1 批的同时校验第 b 批，`cp.async.wait_group 1`）
//
// 安全：无生产者/消费者竞态（数组预写），无等待循环，故无守门狗需求。
//
// 编译：nvcc -O3 -arch=sm_90 src/cpasync_desc_fetch.cu -o /tmp/cpasync_desc_fetch
// 用法：
//   ./cpasync_desc_fetch run <NDESC> <DW> <K> <mode> <NLANES> <reps>
//       mode: 0=scalar 1=vec4 2=cpasync 3=cpasync_pf
//   ./cpasync_desc_fetch bench

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

// 与 MPK load_smem 逐字一致的 16B cp.async（copy_sm80.cuh:44-57）
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

// 从一块连续内存读一条 DW 字描述符，XOR 折叠成 checksum（模拟「路由 + 读 payload」）。
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
  // DW 不是 4 的倍数时补齐（本实验只用 8/16/32，恒为倍数）
  for (int i = nv * 4; i < DW; ++i) x ^= p[i];
  return x;
}

// =====================================================================================
// 取数 kernel：唯一变量 = 取数路径 MODE
// =====================================================================================
template <int MODE>
__global__ void fetch_kernel(int NDESC, int DW, int K, int reps,
                             const uint32_t* __restrict__ descs,
                             unsigned long long* __restrict__ sink) {
  extern __shared__ uint32_t smem[];  // 单缓冲需 K*DW 字；双缓冲需 2*K*DW 字
  const int lane = threadIdx.x;
  const int nlanes = blockDim.x;
  const int nbatch = NDESC / K;
  unsigned long long acc = 0;

  for (int rep = 0; rep < reps; ++rep) {
    if (MODE == 0 || MODE == 1) {
      // ------- 寄存器读：scalar(4B) / vec4(16B) -------
      for (int b = 0; b < nbatch; ++b) {
        const uint32_t* base = descs + (size_t)b * K * DW;
        for (int j = lane; j < K; j += nlanes) {
          const uint32_t* p = base + (size_t)j * DW;
          acc += (MODE == 0) ? fold_words(p, DW) : fold_vec4(p, DW);
        }
      }
    } else if (MODE == 2) {
      // ------- cp.async 单缓冲 -------
      const int chunks = K * DW / 4;  // 16B chunks
      for (int b = 0; b < nbatch; ++b) {
        const uint32_t* base = descs + (size_t)b * K * DW;
        for (int i = lane; i < chunks; i += nlanes)
          load_smem(smem + (size_t)i * 4, base + (size_t)i * 4);
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        for (int j = lane; j < K; j += nlanes)
          acc += fold_words(smem + (size_t)j * DW, DW);
        __syncthreads();  // 复用 SMEM 前清场
      }
    } else {
      // ------- MODE==3：cp.async 双缓冲软件流水 -------
      const int chunks = K * DW / 4;
      uint32_t* buf[2] = {smem, smem + (size_t)K * DW};
      auto issue = [&](int b, int slot) {
        const uint32_t* base = descs + (size_t)b * K * DW;
        for (int i = lane; i < chunks; i += nlanes)
          load_smem(buf[slot] + (size_t)i * 4, base + (size_t)i * 4);
        cp_commit();
      };
      if (nbatch > 0) issue(0, 0);
      for (int b = 0; b < nbatch; ++b) {
        if (b + 1 < nbatch) issue(b + 1, (b + 1) & 1);
        cp_wait<1>();  // 至多 1 组在飞 ⇒ 第 b 组必已完成
        __syncthreads();
        for (int j = lane; j < K; j += nlanes)
          acc += fold_words(buf[b & 1] + (size_t)j * DW, DW);
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
    default: return "cpasync_pf";
  }
}

static int run_fetch(long long NDESC, int DW, int K, int mode, int nlanes, int reps) {
  NDESC = (NDESC / K) * K;  // 取整到 K 的整数倍，保证 host 校验与 kernel 处理范围一致
  uint32_t* descs;
  unsigned long long* sink;
  CK(cudaMalloc(&descs, (size_t)NDESC * DW * 4));
  CK(cudaMalloc(&sink, 8));
  fill_kernel<<<(int)((NDESC + 255) / 256), 256>>>(descs, NDESC, DW);
  CK(cudaDeviceSynchronize());

  int smem_words = (mode == 3) ? 2 * K * DW : K * DW;
  size_t smem_bytes = (size_t)smem_words * 4;
  if (smem_bytes > 48 * 1024) {
    CK(cudaFuncSetAttribute(fetch_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    CK(cudaFuncSetAttribute(fetch_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    CK(cudaFuncSetAttribute(fetch_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    CK(cudaFuncSetAttribute(fetch_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  }

  unsigned long long exp = host_expected(NDESC, DW);
  unsigned long long got = 0;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float ms_accum = 0.f;
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(sink, 0, 8));
    CK(cudaEventRecord(e0));
    switch (mode) {
      case 0: fetch_kernel<0><<<1, nlanes, smem_bytes>>>(NDESC, DW, K, 1, descs, sink); break;
      case 1: fetch_kernel<1><<<1, nlanes, smem_bytes>>>(NDESC, DW, K, 1, descs, sink); break;
      case 2: fetch_kernel<2><<<1, nlanes, smem_bytes>>>(NDESC, DW, K, 1, descs, sink); break;
      default: fetch_kernel<3><<<1, nlanes, smem_bytes>>>(NDESC, DW, K, 1, descs, sink); break;
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
  printf("  fetch %-11s K=%-4d DW=%-2d(%3dB) lanes=%-4d  bad=%d  ~%8.2f ns/item  %7.1f GB/s  (%.3f ms/rep)\n",
         mode_name(mode), K, DW, DW * 4, nlanes, bad, ns_per_item, gbps, ms_accum / reps);

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

  const char* cmd = argc > 1 ? argv[1] : "bench";
  if (strcmp(cmd, "run") == 0) {
    long long NDESC = argc > 2 ? atoll(argv[2]) : 2000000;
    int DW = argc > 3 ? atoi(argv[3]) : 32;
    int K = argc > 4 ? atoi(argv[4]) : 32;
    int mode = argc > 5 ? atoi(argv[5]) : 0;
    int nlanes = argc > 6 ? atoi(argv[6]) : 32;
    int reps = argc > 7 ? atoi(argv[7]) : 5;
    run_fetch(NDESC, DW, K, mode, nlanes, reps);
    return 0;
  }

  // ---- bench：固定总字节 ≈ 256 MB（≫ 50 MB L2），NDESC 随 DW 反比 ----
  const long long BYTES = 256LL << 20;
  auto ndesc = [&](int DW) { return BYTES / (DW * 4); };

  printf("\n===== §A 正确性：四路径 checksum 一致（DW=32, K=32, lanes=32） =====\n");
  for (int m = 0; m < 4; ++m) run_fetch(ndesc(32), 32, 32, m, 32, 3);

  printf("\n===== §B 描述符大小扫描（K=32, lanes=32, 256MB；DW=88=352B 对齐 MPK §6.1） =====\n");
  for (int DW : {8, 16, 32, 88}) {
    for (int m : {0, 1, 2, 3}) run_fetch(ndesc(DW), DW, 32, m, 32, 5);
    printf("\n");
  }

  printf("\n===== §C 批量大小 K 扫描（DW=32=128B, lanes=32, 256MB） =====\n");
  for (int K : {1, 2, 4, 8, 16, 32, 64, 128}) {
    for (int m : {0, 1, 2, 3}) run_fetch(ndesc(32), 32, K, m, 32, 5);
    printf("\n");
  }

  printf("\n===== §D 预取（双缓冲）vs 单缓冲 vs vec4（DW=32, K 扫描, lanes=32） =====\n");
  for (int K : {4, 16, 32, 64}) {
    run_fetch(ndesc(32), 32, K, 1, 32, 5);
    run_fetch(ndesc(32), 32, K, 2, 32, 5);
    run_fetch(ndesc(32), 32, K, 3, 32, 5);
    printf("\n");
  }

  printf("\n===== §E 线程数（warp 数）标度（DW=32, K=32, 256MB） =====\n");
  for (int L : {32, 128, 256, 512}) {
    for (int m : {1, 2, 3}) run_fetch(ndesc(32), 32, 32, m, L, 5);
    printf("\n");
  }
  return 0;
}
