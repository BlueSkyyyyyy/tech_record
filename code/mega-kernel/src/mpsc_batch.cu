// mpsc_batch.cu — MegaKernel 专题 P49-1 小实验：
//   **MPSC in-kernel 队列的「批量出队 + 多线程消费者 + 真实 task 描述符」：
//   单线程消费者每项 ~2.3µs 的同步往返能不能被摊薄？128 B 描述符值多少钱？**
//
// 动机 / 接续：
//   P48-1（docs/04 §37）把 SPSC 队列推广到 MPSC，并把「发布内存序」升级为「发布协议」，
//   但 §37.7 明确留了两条尾巴：
//     (a) **消费者是单线程**，每项 ~2.3 µs 的同步往返主导成本，§C 只能看同配置差值；
//         生产实现（MPK scheduler）会**批量取 task** 把这层摊薄（docs/04 §15 的 K≈W/G）；
//     (b) payload 是纯 `uint32`、**未做真实 task 描述符**。
//
//   真实锚点（mirage 快照 `f9eb70c`，同一快照）：
//     - worker 拉取 task 时**不是逐条**：`threadIdx.x==0` 先读一个 task 区间，再用
//       **128-bit `cp.async` 把 `TaskDesc`（`runtime_header.h:296-336`，`alignas(16)`）
//       批量搬进 shared memory**（`persistent_kernel.cuh:1063-1078`）——即「批量取 task」。
//     - `TaskDesc` 把 `TensorDesc` 压成裸指针数组（`input_ptrs[8]` / `output_ptrs[3]`，
//       `:327-328`），大小约 128 B 量级、按 16 B 对齐——即「真实描述符」。
//     - 队列容量按「速率」拍成固定常量：`per_worker_queue_len=8192`（`persistent_kernel.cuh:1790-1791`）。
//     - scheduler 每处理完一个 batch 走 `prepare_next_batch`（`:1378-1423`）。
//
// 核心问题：
//   Q1（§A）**批量出队（一次读 K 条、一次放行）在数值上仍安全吗？**（逐字校验，应 0 错）
//   Q2（§B）**单线程消费者的每项成本能被「批量」摊薄多少？**（相对 §37 `serial` 基线）
//   Q3（§C）**多线程消费者（一个 warp 并行校验一批 K 条）还能再快多少？**
//       对大描述符（128 B）是否收益更大（并行读 payload）？
//   Q4（§D）**描述符大小（8/16/32 字 = 32/64/128 B）对拆解的影响**——批量/多线程是否随描述符变大
//       越有价值（即「描述符越大、越该批量并行读」）。
//
// 机制：
//   单常驻 kernel，grid = P + 1 block：block 0..P-1 = 生产者（各 1 线程），block P = 消费者。
//   生产者协议与 §37 `mpk` 完全一致：占位(release atomicAdd) → relaxed 写 payload → 连续 CAS 放行。
//   payload 自描述、真实描述符布局（DESC 字）：
//     w[0] = task id (= 占位号 r)；w[1] = op_type = r%4；w[2] = layer = r/4；w[3] = iter = 0；
//     w[i>=4] = mix(r,i)。校验时 w[0] 必查；`checkall` 时逐字再查 w[i]==mix(r,i)。
//   消费者三档（唯一变量）：
//     0 `serial`  : 1 线程，逐条出队（= §37 基线形态）
//     1 `batchK`  : 1 线程，一次等 K 条就绪、串行校验 K 条、**一次 release K 个槽**
//     2 `warpK`   : 1 个 warp（32 线程），一次等 K=32 条就绪、**每线程并行校验 1 条**、
//                   线程 0 一次 release K 个槽（生产实现「批量取 task + 分发给 warp」的微缩）
//
// 安全：所有等待循环带 clock64 守门狗，超时置 dead、kernel 正常返回、host 不挂。
//
// 编译：nvcc -O3 -arch=sm_90 src/mpsc_batch.cu -o /tmp/mpsc_batch
// 用法：
//   ./mpsc_batch run <NS> <CAP> <DW> <P> <cmode> <K> <checkall> <reps>
//       cmode: 0=serial 1=batch 2=warp
//   ./mpsc_batch bench

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

__device__ __forceinline__ uint32_t mix(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 与 MPK mpk_atoms.cuh 同形的内存序原语 ----
__device__ __forceinline__ unsigned long long atom_add_release_u64(unsigned long long* a, unsigned long long v) {
  unsigned long long o;
  asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;" : "=l"(o) : "l"(a), "l"(v) : "memory");
  return o;
}
__device__ __forceinline__ unsigned long long atom_cas_release_u64(unsigned long long* a, unsigned long long c, unsigned long long v) {
  unsigned long long o;
  asm volatile("atom.cas.release.gpu.b64 %0,[%1],%2,%3;" : "=l"(o) : "l"(a), "l"(c), "l"(v) : "memory");
  return o;
}
__device__ __forceinline__ unsigned long long ld_acquire_u64(unsigned long long* a) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.u64 %0,[%1];" : "=l"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_u32(uint32_t* a, uint32_t v) {
  asm volatile("st.relaxed.gpu.u32 [%0],%1;" : : "l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_relaxed_u32(uint32_t* a) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.u32 %0,[%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}

// 校验一条「真实描述符」：w0=task id、w1=op_type(id%4)、w2=layer(id/4)、w3=iter(0)、w[i>=4]=mix。
// checkall=0 时只查 task id（路由用，最省）。
__device__ __forceinline__ int check_desc(uint32_t* p, unsigned long long id, int DW, int checkall) {
  if (ld_relaxed_u32(p) != (uint32_t)id) return 1;
  if (checkall) {
    if (DW > 1 && ld_relaxed_u32(p + 1) != (uint32_t)(id % 4)) return 1;
    if (DW > 2 && ld_relaxed_u32(p + 2) != (uint32_t)(id / 4)) return 1;
    if (DW > 3 && ld_relaxed_u32(p + 3) != 0u) return 1;
    for (int i = 4; i < DW; ++i)
      if (ld_relaxed_u32(p + i) != mix((uint32_t)id, (uint32_t)i)) return 1;
  }
  return 0;
}

// =====================================================================================
// 批量 MPSC 队列 kernel（生产者协议 = §37 mpk）
// =====================================================================================
__global__ void mb_kernel(int NS, int CAP, int DW, int P, int cmode, int K, int checkall,
                          uint32_t* __restrict__ ring,                 // CAP * DW
                          unsigned long long* __restrict__ next_free,  // 占位号分配
                          unsigned long long* __restrict__ last_ready, // 连续就绪游标
                          unsigned long long* __restrict__ consumed,   // 已消费/已放行槽
                          int* __restrict__ bad,
                          int* __restrict__ dead,
                          unsigned long long* __restrict__ cyc) {
  long long t0 = clock64();
  unsigned long long total = (unsigned long long)NS * (unsigned long long)P;

  if (blockIdx.x < (unsigned)P) {                      // ---------------- 生产者 ----------------
    if (threadIdx.x != 0) return;
    for (int k = 0; k < NS; ++k) {
      unsigned long long r = atom_add_release_u64(next_free, 1);   // 占位
      long long w = clock64();
      while (r - ld_acquire_u64(consumed) >= (unsigned long long)CAP) {   // 背压
        if (clock64() - w > (1LL << 32)) { atomicMax(dead, 1); return; }
      }
      int slot = (int)(r % (unsigned long long)CAP);
      uint32_t* p = ring + (size_t)slot * DW;
      st_relaxed_u32(p, (uint32_t)r);                            // 真实描述符前 4 字
      if (DW > 1) st_relaxed_u32(p + 1, (uint32_t)(r % 4));      // op_type
      if (DW > 2) st_relaxed_u32(p + 2, (uint32_t)(r / 4));      // layer
      if (DW > 3) st_relaxed_u32(p + 3, 0u);                     // iter
      for (int i = 4; i < DW; ++i) st_relaxed_u32(p + i, mix((uint32_t)r, (uint32_t)i));
      long long w2 = clock64();
      unsigned long long old;
      do {                                                       // 连续 CAS 放行（MPK 第③步）
        old = atom_cas_release_u64(last_ready, r, r + 1);
        if (old != r && clock64() - w2 > (1LL << 32)) { atomicMax(dead, 1); return; }
      } while (old != r);
    }
  } else if (cmode == 2) {                             // ---------------- 消费者：warp ----------------
    __shared__ unsigned long long sc;
    __shared__ int sbad;
    __shared__ int sdone;
    if (threadIdx.x == 0) { sc = 0; sbad = 0; sdone = 0; }
    __syncthreads();
    long long w = clock64();
    while (true) {
      if (threadIdx.x == 0) {
        if (sc >= total) { sdone = 1; }
        else {
          unsigned long long need = sc + (unsigned long long)K;
          if (need > total) need = total;
          while (ld_acquire_u64(last_ready) < need) {
            if (clock64() - w > (1LL << 32)) { atomicMax(dead, 1); sdone = 1; break; }
          }
        }
      }
      __syncthreads();
      if (sdone) break;
      unsigned long long c = sc;
      int t = threadIdx.x;
      if (c + (unsigned)t < total) {
        int slot = (int)((c + t) % (unsigned long long)CAP);
        uint32_t* p = ring + (size_t)slot * DW;
        if (check_desc(p, c + t, DW, checkall)) atomicAdd(&sbad, 1);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        unsigned long long nxt = sc + (unsigned long long)K;
        if (nxt > total) nxt = total;
        unsigned long long step = nxt - sc;
        sc = nxt;
        __threadfence();
        atomicAdd(consumed, step);
      }
      __syncthreads();
    }
    if (sbad) atomicAdd(bad, sbad);
  } else {                                             // ---------------- 消费者：单线程 ----------------
    if (threadIdx.x != 0) return;
    unsigned long long c = 0;
    int mybad = 0;
    int step = (cmode == 1) ? K : 1;
    long long w = clock64();
    while (c < total) {
      unsigned long long need = c + (unsigned long long)step;
      if (need > total) need = total;
      while (ld_acquire_u64(last_ready) < need) {
        if (clock64() - w > (1LL << 32)) { atomicMax(dead, 1); break; }
      }
      for (unsigned long long j = c; j < need; ++j) {
        int slot = (int)(j % (unsigned long long)CAP);
        uint32_t* p = ring + (size_t)slot * DW;
        if (check_desc(p, j, DW, checkall)) ++mybad;
      }
      unsigned long long prev = c;
      c = need;
      __threadfence();
      atomicAdd(consumed, c - prev);                 // 一次放行 step 个槽
    }
    if (mybad) atomicAdd(bad, mybad);
  }
  if (threadIdx.x == 0) {
    atomicMax(cyc, (unsigned long long)(clock64() - t0));
  }
}

// =====================================================================================
// host
// =====================================================================================
static const char* cmode_name(int m) {
  switch (m) { case 0: return "serial"; case 1: return "batch"; default: return "warp"; }
}

static int run_batch(int NS, int CAP, int DW, int P, int cmode, int K, int checkall, int reps) {
  uint32_t* ring;
  unsigned long long *next_free, *last_ready, *consumed;
  int *bad, *dead; unsigned long long* cyc;
  CK(cudaMalloc(&ring, (size_t)CAP * DW * 4));
  CK(cudaMalloc(&next_free, 8)); CK(cudaMalloc(&last_ready, 8)); CK(cudaMalloc(&consumed, 8));
  CK(cudaMalloc(&bad, 4)); CK(cudaMalloc(&dead, 4)); CK(cudaMalloc(&cyc, 8));
  int total_bad = 0, any_dead = 0;
  unsigned long long maxcyc = 0;

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float ms_accum = 0.f;
  int threads = (cmode == 2) ? K : 1;   // warp 模式一 warp = K 线程（K<=32）

  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(ring, 0xff, (size_t)CAP * DW * 4));
    CK(cudaMemset(next_free, 0, 8)); CK(cudaMemset(last_ready, 0, 8)); CK(cudaMemset(consumed, 0, 8));
    CK(cudaMemset(bad, 0, 4)); CK(cudaMemset(dead, 0, 4)); CK(cudaMemset(cyc, 0, 8));
    CK(cudaEventRecord(e0));
    mb_kernel<<<P + 1, threads>>>(NS, CAP, DW, P, cmode, K, checkall, ring, next_free, last_ready,
                                  consumed, bad, dead, cyc);
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    CK(cudaEventSynchronize(e1));
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1)); ms_accum += ms;
    int b, d; unsigned long long cc;
    CK(cudaMemcpy(&b, bad, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&d, dead, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&cc, cyc, 8, cudaMemcpyDeviceToHost));
    total_bad += b; any_dead |= d;
    if (cc > maxcyc) maxcyc = cc;
  }
  double items = (double)NS * (double)P * (double)reps;
  double ns_per_item = (ms_accum / 1e3) / items * 1e9;
  printf("  mb %-6s K=%-2d DW=%-2d(%3dB) P=%d%s  bad=%d / %.0f  dead=%d  ~%.2f ns/item  (%.3f ms/rep)\n",
         cmode_name(cmode), K, DW, DW * 4, P, checkall ? "" : " route",
         total_bad, items, any_dead, ns_per_item, ms_accum / reps);
  cudaFree(ring); cudaFree(next_free); cudaFree(last_ready);
  cudaFree(consumed); cudaFree(bad); cudaFree(dead); cudaFree(cyc);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return total_bad;
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  printf("device %d: %s  SMs=%d  cc=%d.%d\n", dev, prop.name, prop.multiProcessorCount, prop.major, prop.minor);
  CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

  const char* cmd = argc > 1 ? argv[1] : "bench";
  if (strcmp(cmd, "run") == 0) {
    int NS = argc > 2 ? atoi(argv[2]) : 200000;
    int CAP = argc > 3 ? atoi(argv[3]) : 512;
    int DW = argc > 4 ? atoi(argv[4]) : 8;
    int P = argc > 5 ? atoi(argv[5]) : 8;
    int cmode = argc > 6 ? atoi(argv[6]) : 0;
    int K = argc > 7 ? atoi(argv[7]) : 32;
    int checkall = argc > 8 ? atoi(argv[8]) : 1;
    int reps = argc > 9 ? atoi(argv[9]) : 1;
    run_batch(NS, CAP, DW, P, cmode, K, checkall, reps);
    return 0;
  }

  // ---- bench ----
  const int NS = 100000;
  const int CAP = 1024;

  printf("\n===== §A 正确性（真实描述符 DW=32=128B，三档消费者应 0 bad） =====\n");
  run_batch(NS, CAP, 32, 8, 0, 1, 1, 1);     // serial 全校验
  run_batch(NS, CAP, 32, 8, 1, 32, 1, 1);    // batch K=32
  run_batch(NS, CAP, 32, 8, 2, 32, 1, 1);    // warp K=32

  printf("\n===== §B 单线程消费者：批量出队把「每项往返」摊薄（DW=8=32B，P=8） =====\n");
  run_batch(NS, CAP, 8, 8, 0, 1, 1, 1);      // serial（§37 §C 的 ~1.17us 档）
  run_batch(NS, CAP, 8, 8, 1, 2, 1, 1);
  run_batch(NS, CAP, 8, 8, 1, 4, 1, 1);
  run_batch(NS, CAP, 8, 8, 1, 8, 1, 1);
  run_batch(NS, CAP, 8, 8, 1, 16, 1, 1);
  run_batch(NS, CAP, 8, 8, 1, 32, 1, 1);

  printf("\n===== §C 多线程（warp）消费者：并行校验一批 K=32（扫 DW，P=8） =====\n");
  run_batch(NS, CAP, 8, 8, 0, 1, 1, 1);      // serial baseline
  run_batch(NS, CAP, 8, 8, 2, 32, 1, 1);     // warp
  run_batch(NS, CAP, 16, 8, 0, 1, 1, 1);
  run_batch(NS, CAP, 16, 8, 1, 32, 1, 1);
  run_batch(NS, CAP, 16, 8, 2, 32, 1, 1);
  run_batch(NS, CAP, 32, 8, 0, 1, 1, 1);
  run_batch(NS, CAP, 32, 8, 1, 32, 1, 1);
  run_batch(NS, CAP, 32, 8, 2, 32, 1, 1);

  printf("\n===== §D 描述符大小 × 校验粒度（warp K=32，P=8） =====\n");
  run_batch(NS, CAP, 32, 8, 2, 32, 1, 1);        // 全校验（128B）
  run_batch(NS, CAP, 32, 8, 2, 32, 0, 1);        // 只查 task id（路由）
  run_batch(NS, CAP, 32, 8, 1, 32, 0, 1);        // batch 只查 task id
  run_batch(NS, CAP, 32, 8, 0, 1, 0, 1);         // serial 只查 task id

  printf("\n===== §E 稳定性 / 生产者数（DW=32=128B，batch K=32 与 warp K=32） =====\n");
  run_batch(NS, CAP, 32, 1, 1, 32, 1, 1);
  run_batch(NS, CAP, 32, 4, 1, 32, 1, 1);
  run_batch(NS, CAP, 32, 8, 1, 32, 1, 1);
  run_batch(NS, CAP, 32, 1, 2, 32, 1, 1);
  run_batch(NS, CAP, 32, 4, 2, 32, 1, 1);
  run_batch(NS, CAP, 32, 8, 2, 32, 1, 1);
  return 0;
}
