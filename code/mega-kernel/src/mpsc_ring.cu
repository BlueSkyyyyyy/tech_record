// mpsc_ring.cu — MegaKernel 专题 P48-1 小实验：
//   **多生产者单消费者（MPSC）in-kernel 队列的「发布协议」：MPK 的「占位→写→连续 CAS 放行」
//   到底是不是多生产者下的必要条件？哪种替代会在结构上读错、代价多少？**
//
// 动机 / 接续：
//   P47-1（docs/04 §36）把 in-kernel 队列的「发布内存序」做成了实测，但 §36.6 明确留了一条尾巴：
//   **「环未测多生产者（MPSC）」**。而真实 MPK 的调度队列恰恰是 MPSC——
//   `sched_queues[sched_id]` 由**多个 worker**（128 个）用 `get_rand_sched_id()`（`persistent_kernel.cuh:857`）
//   分配到同一个 scheduler（`num_local_schedulers=16`，故每队列 ~8 个生产者）上推事件
//   （push 路径 `persistent_kernel.cuh:1206-1228`）。其协议是三步，**与 SPSC 的「写 payload→推 tail」不同**：
//     1) `last_event_pos = atom.add.release.gpu.u64(&sched_queue_next_free_event_id[sched_id], 1)`
//        （**占位**，原子返回一个全局分配号）；
//     2) `st.relaxed.gpu.u64(&sched_queues[sched_id][pos % len], event_index)`（**relaxed 写 payload**）；
//     3) `do { old = atom.cas.release.gpu.u64(&sched_queue_last_ready_event_id[sched_id], pos, pos+1); }
//        while (old != pos);`（**连续 CAS 放行**：只有 pos == 当前 last_ready 的生产者能把它 +1）。
//   第 3 步是关键：多生产者的占位号是**乱序**产生的，消费者必须以**连续**方式放行，才能保证「读到
//   last_ready=N 时，槽 0..N-1 全部已写」。
//
// 核心问题：
//   Q1（§A）**MPSC 下，MPK 的「占位→写→连续 CAS 放行」协议正确吗？**（逐槽校验，应 0 错）
//   Q2（§B）**两种「看似合理」但结构错误的替代会不会读错？**
//       (a) `reserve_count`：占位 → 写 payload → `atomicAdd(done,1)`（**非连续完成计数**）——
//           消费者把 done 当「前 done 个槽已写」→ 占位乱序时可能读到「还没写的低号槽」。
//       (b) `pub_then_write`：先占位/发布（把 `next_free` 当就绪游标）→ 后写 payload——§36 的
//           「发布先于写」在 MPSC 下的形态。
//       (c) `pertile`：每个槽自带 `st.release` 的 ready 标志（**逐槽发布**，另一种正确解）。
//   Q3（§C）**连续 CAS 放行 / 逐槽 release 的代价**：相对「无放行的 SPSC 基线」每槽多多少 ns？
//       随生产者数 P 怎么变？
//
// 机制：
//   单常驻 kernel，grid = P + 1 block、每 block 1 线程：block 0..P-1 = 生产者，block P = 消费者。
//   生产者在**全局号 r = 0..P*NS-1** 上工作（每个生产者做 NS 项）；实际占位号由 `next_free` 原子分配，
//   故三项工作乱序落地。payload 自描述：slot 首字 = 其占位号 r、其余字 = mix(r,i)（PL 字）。
//   消费者**按占位号 0,1,2,... 顺序**消费（这正是「连续放行」要保证的），逐字校验首字是否 == c、
//   其余是否 == mix(c,i)；任何不符记一次 bad。这样「读到陈旧复用槽」（首字 = c-CAP）或「未写槽」
//   （首字 = 0xffffffff）都会被抓到。
//   容量背压：生产者占位后等 `r - consumed < CAP` 才写；消费者读完才放行 consumed。
//
// 安全：所有等待循环带 clock64 守门狗，超时置 dead、kernel 正常返回、host 不挂。
//
// 编译：nvcc -O3 -arch=sm_90 src/mpsc_ring.cu -o /tmp/mpsc_ring
// 用法：
//   ./mpsc_ring ring <NS> <CAP> <PL> <P> <mode> <reps>   mode: 0=mpk 1=pertile 2=reserve_count
//                                                                3=pub_then_write 4=pubslp 5=mpk_relaxed 6=spsc
//   ./mpsc_ring bench

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
__device__ __forceinline__ unsigned long long ld_relaxed_u64(unsigned long long* a) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.u64 %0,[%1];" : "=l"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_u32(uint32_t* a, uint32_t v) {
  asm volatile("st.relaxed.gpu.u32 [%0],%1;" : : "l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_u32(uint32_t* a, uint32_t v) {
  asm volatile("st.release.gpu.u32 [%0],%1;" : : "l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_u32(uint32_t* a) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_u32(uint32_t* a) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.u32 %0,[%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}

// =====================================================================================
// MPSC 环形队列 kernel
//   mode 0 `mpk`            : 占位(release atomicAdd) → 写 payload(relaxed) → 连续 CAS 放行(release)
//   mode 1 `pertile`        : 占位 → 写 payload → st.release 逐槽 ready 标志
//   mode 2 `reserve_count`  : 占位 → 写 payload → atomicAdd(done,1) 非连续完成计数（结构错误）
//   mode 3 `pub_then_write` : 占位后立刻当就绪（next_free 即游标）→ 后写 payload（结构错误）
//   mode 4 `pubslp`         : 同 3，但占位后人为空转拉长窗口（检测器正对照）
//   mode 5 `mpk_relaxed`    : 同 0，但 CAS/load 用 relaxed（测 release 是否必需）
//   mode 6 `spsc`           : P=1 时退化为 §36 的「写 payload → 推 tail」无放行基线
// =====================================================================================
__global__ void mpsc_kernel(int NS, int CAP, int PL, int P, int mode, int stall0,
                            uint32_t* __restrict__ ring,       // CAP * PL
                            uint32_t* __restrict__ flags,      // CAP (pertile ready 标志, generation=pos+1)
                            unsigned long long* __restrict__ next_free,   // 占位号分配
                            unsigned long long* __restrict__ last_ready,  // 连续就绪游标 (mpk)
                            unsigned long long* __restrict__ ready_cnt,   // 非连续完成计数 (reserve_count)
                            unsigned long long* __restrict__ consumed,    // 已消费（供背压与终止）
                            int* __restrict__ bad,
                            int* __restrict__ dead,
                            unsigned long long* __restrict__ cyc) {
  long long t0 = clock64();
  unsigned long long total = (unsigned long long)NS * (unsigned long long)P;

  if (blockIdx.x < P) {                                // ---------------- 生产者 ----------------
    if (threadIdx.x != 0) return;
    for (int k = 0; k < NS; ++k) {
      unsigned long long r = atom_add_release_u64(next_free, 1);   // 占位
      long long w = clock64();
      while (r - ld_acquire_u64(consumed) >= (unsigned long long)CAP) {   // 背压
        if (clock64() - w > (1LL << 32)) { atomicMax(dead, 1); return; }
      }
      int slot = (int)(r % (unsigned long long)CAP);
      uint32_t* p = ring + (size_t)slot * PL;

      // 正对照：让「持有前 4 个占位号 r<4」的生产者在其写 payload 前停 ~110us，决定性地制造
      // 「低号槽未写、高号槽（r=4..7）已写」的偏斜。完成门控的协议（mpk/pertile）会把消费者挡在
      // 低号槽外，不受影响；完成计数/占位即就绪会让消费者提前读低号槽 → 读错。
      if (stall0 && r < 4) { long long d = clock64(); while (clock64() - d < 200000) {} }

      if (mode == 3 || mode == 4) {                    // pub_then_write：占位号已是就绪
        if (mode == 4) { long long d = clock64(); while (clock64() - d < 4000) {} }  // ~2.2us 窗口
        st_relaxed_u32(p, (uint32_t)r);
        for (int i = 1; i < PL; ++i) st_relaxed_u32(p + i, mix((uint32_t)r, (uint32_t)i));
        continue;
      }

      st_relaxed_u32(p, (uint32_t)r);                  // relaxed 写 payload（自描述首字 = 占位号）
      for (int i = 1; i < PL; ++i) st_relaxed_u32(p + i, mix((uint32_t)r, (uint32_t)i));

      if (mode == 1) {                                 // pertile：逐槽 release 发布
        st_release_u32(&flags[slot], (uint32_t)(r + 1));
      } else if (mode == 2) {                          // reserve_count：非连续完成计数
        atomicAdd((unsigned long long*)ready_cnt, 1ULL);
      } else if (mode == 6) {                          // spsc：P=1 下 relaxed atomicAdd 推游标（无 release 基线）
        atomicAdd(last_ready, 1ULL);
      } else {                                         // mpk / mpk_relaxed：连续 CAS 放行
        long long w2 = clock64();
        if (mode == 5) {
          unsigned long long old;
          do { old = atomicCAS(last_ready, r, r + 1);
               if (old != r && clock64() - w2 > (1LL << 32)) { atomicMax(dead, 1); return; }
          } while (old != r);
        } else {
          unsigned long long old;
          do { old = atom_cas_release_u64(last_ready, r, r + 1);
               if (old != r && clock64() - w2 > (1LL << 32)) { atomicMax(dead, 1); return; }
          } while (old != r);
        }
      }
    }
  } else {                                             // ---------------- 消费者 ----------------
    if (threadIdx.x != 0) return;
    int mybad = 0;
    unsigned long long c = 0;
    long long w = clock64();
    while (c < total) {
      if (mode == 5) {                                 // relaxed 读游标
        unsigned long long lr = ld_relaxed_u64(last_ready);
        if (lr <= c) { if (clock64() - w > (1LL << 32)) { atomicMax(dead, 1); break; } continue; }
      } else if (mode == 1 || mode == 3 || mode == 4) {
        if (mode == 1) {                               // 逐槽 acquire ready 标志
          int slot = (int)(c % (unsigned long long)CAP);
          if (ld_acquire_u32(&flags[slot]) != (uint32_t)(c + 1)) {
            if (clock64() - w > (1LL << 32)) { atomicMax(dead, 1); break; } continue;
          }
        } else {                                       // pub_then_write：把 next_free 当就绪
          if (ld_acquire_u64(next_free) <= c) {
            if (clock64() - w > (1LL << 32)) { atomicMax(dead, 1); break; } continue;
          }
        }
      } else if (mode == 2) {                          // reserve_count：把 done 当「前 done 个已写」
        if (ld_acquire_u64(ready_cnt) <= c) {
          if (clock64() - w > (1LL << 32)) { atomicMax(dead, 1); break; } continue;
        }
      } else if (mode == 6) {                          // spsc：relaxed 读游标（无 acquire 基线）
        if (ld_relaxed_u64(last_ready) <= c) {
          if (clock64() - w > (1LL << 32)) { atomicMax(dead, 1); break; } continue;
        }
      } else {                                         // mpk：acquire 读连续游标
        if (ld_acquire_u64(last_ready) <= c) {
          if (clock64() - w > (1LL << 32)) { atomicMax(dead, 1); break; } continue;
        }
      }
      int slot = (int)(c % (unsigned long long)CAP);
      uint32_t* p = ring + (size_t)slot * PL;
      uint32_t g = ld_relaxed_u32(p);                  // relaxed 读 payload 首字
      if (g != (uint32_t)c) { ++mybad; }
      else {
        for (int i = 1; i < PL; ++i) {
          if (ld_relaxed_u32(p + i) != mix((uint32_t)c, (uint32_t)i)) { ++mybad; break; }
        }
      }
      c++;
      __threadfence();
      atomicAdd((unsigned long long*)consumed, 1ULL);  // 读完后才放行槽
    }
    if (mybad) atomicAdd(bad, mybad);
  }
  if (threadIdx.x == 0) {
    unsigned long long dt = (unsigned long long)(clock64() - t0);
    atomicMax(cyc, dt);
  }
}

// =====================================================================================
// host
// =====================================================================================
static const char* mode_name(int m) {
  switch (m) {
    case 0: return "mpk";
    case 1: return "pertile";
    case 2: return "reserve_count";
    case 3: return "pub_then_write";
    case 4: return "pubslp";
    case 5: return "mpk_relaxed";
    default: return "spsc";
  }
}

static int run_ring(int NS, int CAP, int PL, int P, int mode, int reps, int stall0) {
  uint32_t* ring; uint32_t* flags;
  unsigned long long *next_free, *last_ready, *ready_cnt, *consumed;
  int *bad, *dead; unsigned long long* cyc;
  CK(cudaMalloc(&ring, (size_t)CAP * PL * 4));
  CK(cudaMalloc(&flags, (size_t)CAP * 4));
  CK(cudaMalloc(&next_free, 8)); CK(cudaMalloc(&last_ready, 8)); CK(cudaMalloc(&ready_cnt, 8));
  CK(cudaMalloc(&consumed, 8)); CK(cudaMalloc(&bad, 4)); CK(cudaMalloc(&dead, 4)); CK(cudaMalloc(&cyc, 8));
  int total_bad = 0, any_dead = 0;
  unsigned long long maxcyc = 0;

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float ms_accum = 0.f;

  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(ring, 0xff, (size_t)CAP * PL * 4));
    CK(cudaMemset(flags, 0, (size_t)CAP * 4));
    CK(cudaMemset(next_free, 0, 8)); CK(cudaMemset(last_ready, 0, 8)); CK(cudaMemset(ready_cnt, 0, 8));
    CK(cudaMemset(consumed, 0, 8)); CK(cudaMemset(bad, 0, 4)); CK(cudaMemset(dead, 0, 4)); CK(cudaMemset(cyc, 0, 8));
    CK(cudaEventRecord(e0));
    mpsc_kernel<<<P + 1, 32>>>(NS, CAP, PL, P, mode, stall0, ring, flags, next_free, last_ready,
                               ready_cnt, consumed, bad, dead, cyc);
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
  double ns_per_item = (ms_accum / 1e3) / items * 1e9;   // ms -> s -> ns
  printf("  ring %-14s NS=%d CAP=%d PL=%d P=%d%s  bad=%d / %.0f  dead=%d  ~%.2f ns/item  (%.3f ms/rep)\n",
         mode_name(mode), NS, CAP, PL, P, stall0 ? " stall0" : "", total_bad, items, any_dead, ns_per_item, ms_accum / reps);
  cudaFree(ring); cudaFree(flags); cudaFree(next_free); cudaFree(last_ready); cudaFree(ready_cnt);
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
  if (strcmp(cmd, "ring") == 0) {
    int NS = argc > 2 ? atoi(argv[2]) : 200000;
    int CAP = argc > 3 ? atoi(argv[3]) : 256;
    int PL = argc > 4 ? atoi(argv[4]) : 8;
    int P = argc > 5 ? atoi(argv[5]) : 8;
    int mode = argc > 6 ? atoi(argv[6]) : 0;
    int reps = argc > 7 ? atoi(argv[7]) : 1;
    int stall0 = argc > 8 ? atoi(argv[8]) : 0;
    run_ring(NS, CAP, PL, P, mode, reps, stall0);
    return 0;
  }

  // ---- bench ----
  // §A 正确性（自然时序）
  printf("\n===== §A 正确协议（应 0 bad）：连续 CAS 放行 / 逐槽 release / 无放行基线 =====\n");
  run_ring(50000, 256, 8, 1, 0, 1, 0);   // mpk P=1
  run_ring(50000, 256, 8, 2, 0, 1, 0);
  run_ring(50000, 256, 8, 4, 0, 1, 0);
  run_ring(50000, 256, 8, 8, 0, 1, 0);
  run_ring(50000, 256, 8, 32, 0, 1, 0);
  run_ring(50000, 256, 8, 8, 1, 1, 0);   // pertile P=8
  run_ring(50000, 32, 8, 8, 0, 1, 0);    // 紧容量
  run_ring(50000, 32, 8, 8, 1, 1, 0);
  run_ring(50000, 256, 8, 1, 6, 1, 0);   // spsc P=1 无 release 基线

  printf("\n===== §B 结构不严谨的替代（自然时序；多数「侥幸」0 错） =====\n");
  run_ring(50000, 256, 8, 8, 2, 1, 0);   // reserve_count（非连续完成计数）
  run_ring(50000, 256, 8, 2, 2, 1, 0);
  run_ring(50000, 256, 8, 8, 3, 1, 0);   // pub_then_write（占位即就绪）
  run_ring(50000, 256, 8, 2, 3, 1, 0);
  printf("  relaxed 版本（CAS/load 无 release/acquire）：\n");
  run_ring(50000, 256, 8, 8, 5, 1, 0);
  run_ring(50000, 256, 8, 2, 5, 1, 0);

  printf("\n===== §B2 决定性正对照 stall0（持有占位号 0 的生产者写前停 ~110us） =====\n");
  printf("  完成计数 / 占位即就绪 → 读错；连续 CAS / 逐槽 release → 仍 0：\n");
  run_ring(20000, 256, 8, 8, 2, 1, 1);   // reserve_count + stall0 → 应错
  run_ring(20000, 256, 8, 4, 2, 1, 1);
  run_ring(20000, 256, 8, 8, 3, 1, 1);   // pub_then_write + stall0 → 应错
  run_ring(20000, 256, 8, 4, 3, 1, 1);
  run_ring(20000, 256, 8, 8, 0, 1, 1);   // mpk + stall0 → 仍 0
  run_ring(20000, 256, 8, 8, 1, 1, 1);   // pertile + stall0 → 仍 0

  printf("\n===== §C 放行代价：连续 CAS / 逐槽 release 相对无放行基线（ns/item，扫 P） =====\n");
  run_ring(400000, 512, 1, 1, 6, 1, 0);  // spsc P=1 基线
  run_ring(400000, 512, 1, 1, 0, 1, 0);  // mpk P=1
  run_ring(400000, 512, 1, 8, 0, 1, 0);  // mpk P=8
  run_ring(400000, 512, 1, 8, 1, 1, 0);  // pertile P=8
  run_ring(400000, 512, 1, 8, 5, 1, 0);  // mpk_relaxed P=8
  printf("  大 payload（放行成本被访存摊薄）：\n");
  run_ring(50000, 256, 256, 8, 0, 1, 0);
  run_ring(50000, 256, 256, 8, 1, 1, 0);
  return 0;
}
