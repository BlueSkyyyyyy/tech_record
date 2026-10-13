// pub_order_ring.cu — MegaKernel 专题 P47-1 小实验：
//   **in-kernel 生产者→消费者队列的「发布内存序」到底要不要 release/acquire？代价多大？**
//
// 动机 / 接续：
//   P46-1（docs/04 §35）把事件驱动调度的 ready 队列从「无界」装进「有界环形缓冲」，但 §35.7
//   明确留了一条尾巴：**入队的内存序是「先 atomicAdd/CAS 推 tail、后写数据」（无 release 发布），
//   理论上消费者可能在数据可见前读到槽；真实 MPK 用 release/acquire 把它做实**——可作后续实验。
//
//   这不是 §35 的笔误，而是实现里真实存在的两种写法：
//     - **P46/§34/§31–§33 的合成实现**：`int slot = atomicAdd(tail,1); pool[slot]=u; __threadfence();`
//       —— tail（发布游标）与 payload **都在一次原子/栅栏里**，「先推游标」。
//     - **MPK（mirage 快照 f9eb70c）的队列协议**（docs/06 §3 / §5.3，`mpk_atoms.cuh:17-96`）：
//       「**relaxed 写数据 + release 推游标 / acquire 读游标 + relaxed 读数据**」——worker 侧
//       `st.relaxed.gpu` 写事件索引（`persistent_kernel.cuh:1215-1218`）后用
//       `atom.cas.release.gpu` 放行读游标（`:1222-1228`）；scheduler 侧 `ld.acquire.gpu` 读
//       `sched_queue_last_ready_event_id`（`:1340`）再 `ld.relaxed.gpu` 取 event_id（`:1357-1360`）；
//       每次派发也是「`st.relaxed.gpu` 写 worker 队列槽 → `atom.add.release.gpu` 放行游标」
//       （`:1495-1502`）。DeepGEMM 的 ring barrier 同理（`red.async.inc.release` + 无符号距离 wait）。
//   所以问题是：**sm90 上「不 release」到底会不会真的读错？release/acquire 值多少代价？**
//
// 核心问题：
//   Q1（§A litmus）**GPU 弱内存模型下，经典的 message-passing litmus 在 sm90 上可观测吗？**
//        生产者「写 data → 写 flag」（都 relaxed），消费者「读 flag==1 → 读 data」，
//        不做任何 fence 时能否观测到 `flag==1 且 data==0`（丢失消息）？加 release/acquire 后是否恒 0？
//   Q2（§B 环）**把这个 litmus 搬到真实的 in-kernel 生产者→消费者环形队列**：写 payload 后推 tail，
//        不 release（`wro`）vs release/acquire（`rel`）vs 故意「先推 tail 后写 payload」（`pub`，坏顺序）
//        vs 只写一半 payload 就推 tail（`half`，检测器正对照）——各自丢/读错多少槽？
//   Q3（§C 代价）**release/acquire 每槽值多少 ns**？环的稳态吞吐（槽/µs）在 `wro` vs `rel` 差多少？
//       对应 MPK 为何要在「relaxed 写 + release 推游标」之间拆分，而不是全程 fence。
//
// 机制：
//   §A litmus：单 kernel，block0=P、block1=Q，各 1 线程（不同 SM，必然跨 SM）。逐 trial 握手：
//     P 复位 data/flag → 用带 fence 的 `setup` 通道放行 Q → 写 data=1 → [mode1: __threadfence()]
//     → 写 flag=1；Q 等 flag==1 → [mode1: __threadfence()] → 读 data；==0 记一次丢失。
//     控制通道（setup/consumed）两模式都带 fence，只有 data→flag 通道被操控。
//   §B 环：单常驻 kernel，block0=生产者，其余 G-1=消费者，全局环形缓冲 CAP 槽 × PL 字 payload，
//     payload[s][i]=mix(s,i) 可验证。生产者按模式写 payload + 推 tail（含容量背压 tail-head<CAP）；
//     消费者 CAS 抢 head、读槽、逐字校验；任何一字不符记一次 `bad`。
//   §C 复用 §B 的计时（每槽 ns）与 §A 的每 trial ns。
//
// 安全：所有等待循环都有 clock64 守门狗，超时置 deadlock 标志、kernel 正常返回、host 不挂。
//
// 编译：nvcc -O3 -arch=sm_90 src/pub_order_ring.cu -o /tmp/pub_order_ring
// 用法：
//   ./pub_order_ring litmus <T> <mode>              mode: 0=relaxed 1=release/acquire
//   ./pub_order_ring ring <NS> <CAP> <PL> <G> <mode> mode: 0=wro 1=rel 2=pub 3=half
//   ./pub_order_ring bench                          扫 litmus 与 ring

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

// =====================================================================================
// §A message-passing litmus：P：data=1; flag=1   Q：while(flag==0); r1=data;
// =====================================================================================
// 控制通道 setup/consumed 两模式都带 __threadfence()，只操控 data→flag 通道。
__global__ void litmus_kernel(int T, int mode,
                              int* __restrict__ data,
                              int* __restrict__ flag,
                              int* __restrict__ setup,
                              int* __restrict__ consumed,
                              int* __restrict__ fail,
                              unsigned long long* __restrict__ cyc) {
  volatile int* vflag = flag;
  volatile int* vsetup = setup;
  volatile int* vconsumed = consumed;
  volatile int* vdata = data;
  long long t0 = clock64();

  if (blockIdx.x == 0) {
    if (threadIdx.x != 0) return;
    for (int t = 0; t < T; ++t) {
      if (t > 0) {                                   // 等 Q 完成上一 trial
        long long w = clock64();
        while (*vconsumed != t) { if (clock64() - w > (1LL << 40)) return; }
      }
      *data = 0; *flag = 0;                          // 复位数据通道
      __threadfence();
      *setup = t + 1;                                // 放行 Q（控制通道带 release fence）
      __threadfence();
      *data = 1;                                     // payload
      if (mode == 1) __threadfence();                // release：把 data 推到 flag 之前可见
      *flag = 1;                                     // 发布
    }
  } else if (blockIdx.x == 1) {
    if (threadIdx.x != 0) return;
    for (int t = 0; t < T; ++t) {
      long long w = clock64();
      while (*vsetup != t + 1) { if (clock64() - w > (1LL << 40)) return; }
      w = clock64();
      while (*vflag == 0) { if (clock64() - w > (1LL << 40)) return; }
      if (mode == 1) __threadfence();                // acquire：把 data 的读排到 flag 的读之后
      int r1 = *vdata;
      if (r1 == 0) atomicAdd(fail, 1);               // flag 已置 1 却没看到 data ⇒ 丢失
      *consumed = t + 1;
      __threadfence();
    }
  }
  if (threadIdx.x == 0 && blockIdx.x < 2) {
    unsigned long long dt = (unsigned long long)(clock64() - t0);
    atomicMax(cyc, dt);
  }
}

// =====================================================================================
// §A2 store-buffering (SB) litmus：P: x=1; r1=y   Q: y=1; r2=x（都 relaxed）
//   观测 r1==0 && r2==0（双方都先读、后 See 对方写）= store→load 重排。这是 GPU 弱内存的
//   正对照：若它可观测、而 §A 的 MP（store→store）不可观测，则说明 sm90 保序的是 store→store。
// =====================================================================================
__device__ __forceinline__ void gbar2(int* bar, int* myb) {
  atomicAdd(bar, 1);
  ++(*myb);
  volatile int* vb = bar;
  while (*vb < (*myb) * 2) {}
}

__global__ void sb_kernel(int T, int* __restrict__ x, int* __restrict__ y,
                          int* __restrict__ r1, int* __restrict__ r2,
                          int* __restrict__ fail, int* __restrict__ bar,
                          unsigned long long* __restrict__ cyc) {
  volatile int* vx = x; volatile int* vy = y;
  if (blockIdx.x > 1 || threadIdx.x != 0) return;
  int myb = 0;
  int local = 0;
  long long t0 = clock64();
  for (int t = 0; t < T; ++t) {
    if (blockIdx.x == 0) { *x = 0; *y = 0; *r1 = 0; *r2 = 0; }
    gbar2(bar, &myb);                       // reset 可见（之后的写/读之间**不**再同步，才能观测 store→load 重排）
    if (blockIdx.x == 0) { *x = 1; int r = *vy; *r1 = r; }   // P: 写 x，读 y（背靠背）
    else                 { *y = 1; int r = *vx; *r2 = r; }   // Q: 写 y，读 x
    gbar2(bar, &myb);                       // 读已落
    if (blockIdx.x == 0 && *r1 == 0 && *r2 == 0) ++local;   // 双方都没看到对方写
    gbar2(bar, &myb);
  }
  if (local) atomicAdd(fail, local);
  unsigned long long dt = (unsigned long long)(clock64() - t0);
  atomicMax(cyc, dt);
}

// =====================================================================================
// §B in-kernel 生产者→消费者环形队列
// =====================================================================================
// mode 0 `wro`  : 写 payload → 推 tail                （P46 写法，无 release）
// mode 1 `rel`  : 写 payload → __threadfence() → 推 tail；消费者 读 tail → __threadfence() → 读 payload（MPK）
// mode 2 `pub`  : 推 tail → 写 payload                （坏顺序，正对照）
// mode 3 `half` : 写一半 payload → 推 tail → 写另一半 （检测器正对照，必错）
__global__ void ring_kernel(int NS, int CAP, int PL, int G, int mode,
                            uint32_t* __restrict__ ring,
                            int* __restrict__ head,
                            int* __restrict__ consumed,
                            int* __restrict__ tail,
                            int* __restrict__ bad,
                            int* __restrict__ done,
                            int* __restrict__ dead,
                            unsigned long long* __restrict__ cyc) {
  volatile int* vhead = head;
  volatile int* vtail = tail;
  volatile int* vconsumed = consumed;
  volatile int* vdone = done;
  volatile int* vdead = dead;
  long long t0 = clock64();

  if (blockIdx.x == 0) {                              // 生产者
    if (threadIdx.x != 0) return;
    for (int s = 0; s < NS; ++s) {
      long long w = clock64();
      // 背压用「已完全消费（读并校验完）」计数，而非 head：head 只是抢槽，读还没发生。
      while (*vtail - *vconsumed >= CAP) {
        if (clock64() - w > (1LL << 40)) { *dead = 1; return; }
      }
      uint32_t* p = ring + (size_t)(s % CAP) * PL;
      if (mode == 2) {                                // pub：先发布后写（坏）
        *vtail = s + 1;
        for (int i = 0; i < PL; ++i) p[i] = mix((uint32_t)s, (uint32_t)i);
      } else if (mode == 4) {                         // pubslp：先发布 → 人为拉长窗口 → 后写（检测器正对照）
        *vtail = s + 1;
        long long d = clock64();
        while (clock64() - d < 4000) {}               // ~2.2µs 窗口（模拟发布者被抢占/延迟）
        for (int i = 0; i < PL; ++i) p[i] = mix((uint32_t)s, (uint32_t)i);
      } else if (mode == 3) {                         // half：写一半就发布
        for (int i = 0; i < PL / 2; ++i) p[i] = mix((uint32_t)s, (uint32_t)i);
        *vtail = s + 1;
        for (int i = PL / 2; i < PL; ++i) p[i] = mix((uint32_t)s, (uint32_t)i);
      } else {
        for (int i = 0; i < PL; ++i) p[i] = mix((uint32_t)s, (uint32_t)i);
        if (mode == 1) __threadfence();               // release：payload 先于 tail 可见
        *vtail = s + 1;
      }
    }
    *vdone = 1;
  } else {                                            // 消费者
    if (threadIdx.x != 0) return;
    int mybad = 0;
    long long w = clock64();
    while (true) {
      int h = *vhead;
      int t = *vtail;
      if (h >= t) {
        if (*vdone && h >= *vtail) break;
        if (clock64() - w > (1LL << 42)) { *vdead = 1; break; }
        continue;
      }
      if (atomicCAS(head, h, h + 1) != h) continue;
      int slot = h % CAP;
      uint32_t* p = ring + (size_t)slot * PL;
      if (mode == 1) __threadfence();                 // acquire：payload 读排到 tail 读之后
      for (int i = 0; i < PL; ++i) {
        if (p[i] != mix((uint32_t)h, (uint32_t)i)) { ++mybad; break; }
      }
      __threadfence();
      atomicAdd(consumed, 1);                          // 读完后才放行槽（否则会被生产者覆盖）
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
static const char* lit_mode_name(int m) { return m == 0 ? "relaxed" : "rel+acq"; }
static const char* ring_mode_name(int m) {
  switch (m) { case 0: return "wro"; case 1: return "rel"; case 2: return "pub"; case 3: return "half"; default: return "pubslp"; }
}

static int run_litmus(int T, int mode, int reps) {
  int *data, *flag, *setup, *consumed, *fail;
  unsigned long long* cyc;
  CK(cudaMalloc(&data, 4)); CK(cudaMalloc(&flag, 4)); CK(cudaMalloc(&setup, 4));
  CK(cudaMalloc(&consumed, 4)); CK(cudaMalloc(&fail, 4)); CK(cudaMalloc(&cyc, 8));
  int total_fail = 0;
  unsigned long long maxcyc = 0;
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(data, 0, 4)); CK(cudaMemset(flag, 0, 4)); CK(cudaMemset(setup, 0, 4));
    CK(cudaMemset(consumed, 0, 4)); CK(cudaMemset(fail, 0, 4)); CK(cudaMemset(cyc, 0, 8));
    litmus_kernel<<<2, 32>>>(T, mode, data, flag, setup, consumed, fail, cyc);
    CK(cudaDeviceSynchronize());
    int f; unsigned long long c;
    CK(cudaMemcpy(&f, fail, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
    total_fail += f;
    if (c > maxcyc) maxcyc = c;
  }
  double ns = maxcyc / 1.8;  // H100 SM clock ~1.8 GHz（见正文口径）
  printf("  litmus %-8s T=%d reps=%d  fail=%d / %d  (%.2e)   cycle=%llu  ~%.2f ns/trial\n",
         lit_mode_name(mode), T, reps, total_fail, T * reps,
         (double)total_fail / (double)(T * reps), maxcyc, ns / T);
  cudaFree(data); cudaFree(flag); cudaFree(setup); cudaFree(consumed); cudaFree(fail); cudaFree(cyc);
  return total_fail;
}

static int run_sb(int T, int reps) {
  int *x, *y, *r1, *r2, *fail, *bar; unsigned long long* cyc;
  CK(cudaMalloc(&x, 4)); CK(cudaMalloc(&y, 4)); CK(cudaMalloc(&r1, 4)); CK(cudaMalloc(&r2, 4));
  CK(cudaMalloc(&fail, 4)); CK(cudaMalloc(&bar, 4)); CK(cudaMalloc(&cyc, 8));
  int total = 0; unsigned long long maxcyc = 0;
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(x, 0, 4)); CK(cudaMemset(y, 0, 4)); CK(cudaMemset(r1, 0, 4)); CK(cudaMemset(r2, 0, 4));
    CK(cudaMemset(fail, 0, 4)); CK(cudaMemset(bar, 0, 4)); CK(cudaMemset(cyc, 0, 8));
    sb_kernel<<<2, 32>>>(T, x, y, r1, r2, fail, bar, cyc);
    CK(cudaDeviceSynchronize());
    int f; unsigned long long c;
    CK(cudaMemcpy(&f, fail, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
    total += f; if (c > maxcyc) maxcyc = c;
  }
  printf("  litmus SB       T=%d reps=%d  both-zero=%d / %d  (%.2e)   cycle=%llu  ~%.2f ns/trial\n",
         T, reps, total, T * reps, (double)total / (double)(T * reps), maxcyc, (maxcyc / 1.8) / T);
  cudaFree(x); cudaFree(y); cudaFree(r1); cudaFree(r2); cudaFree(fail); cudaFree(bar); cudaFree(cyc);
  return total;
}

static int run_ring(int NS, int CAP, int PL, int G, int mode, int reps) {
  uint32_t* ring; int *head, *consumed, *tail, *bad, *done, *dead; unsigned long long* cyc;
  CK(cudaMalloc(&ring, (size_t)CAP * PL * 4));
  CK(cudaMalloc(&head, 4)); CK(cudaMalloc(&consumed, 4)); CK(cudaMalloc(&tail, 4)); CK(cudaMalloc(&bad, 4));
  CK(cudaMalloc(&done, 4)); CK(cudaMalloc(&dead, 4)); CK(cudaMalloc(&cyc, 8));
  int total_bad = 0, any_dead = 0;
  unsigned long long maxcyc = 0;
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(ring, 0xff, (size_t)CAP * PL * 4));
    CK(cudaMemset(head, 0, 4)); CK(cudaMemset(consumed, 0, 4)); CK(cudaMemset(tail, 0, 4)); CK(cudaMemset(bad, 0, 4));
    CK(cudaMemset(done, 0, 4)); CK(cudaMemset(dead, 0, 4)); CK(cudaMemset(cyc, 0, 8));
    ring_kernel<<<G, 1>>>(NS, CAP, PL, G, mode, ring, head, consumed, tail, bad, done, dead, cyc);
    CK(cudaDeviceSynchronize());
    int b, d; unsigned long long c;
    CK(cudaMemcpy(&b, bad, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&d, dead, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
    total_bad += b; any_dead |= d;
    if (c > maxcyc) maxcyc = c;
  }
  double ns = maxcyc / 1.8;
  double slots_per_us = (double)NS / (ns / 1000.0);
  printf("  ring %-5s NS=%d CAP=%d PL=%d G=%d reps=%d  bad=%d / %d  dead=%d  cycle=%llu  "
         "~%.2f ns/slot  %.1f slot/us\n",
         ring_mode_name(mode), NS, CAP, PL, G, reps, total_bad, NS * reps, any_dead, maxcyc,
         ns / NS, slots_per_us);
  cudaFree(ring); cudaFree(head); cudaFree(consumed); cudaFree(tail); cudaFree(bad); cudaFree(done); cudaFree(dead); cudaFree(cyc);
  return total_bad;
}

int main(int argc, char** argv) {
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  printf("device %d: %s  SMs=%d  cc=%d.%d\n", dev, prop.name, prop.multiProcessorCount,
         prop.major, prop.minor);
  CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

  const char* cmd = argc > 1 ? argv[1] : "bench";

  if (strcmp(cmd, "litmus") == 0) {
    int T = argc > 2 ? atoi(argv[2]) : 5000000;
    int mode = argc > 3 ? atoi(argv[3]) : 0;
    int reps = argc > 4 ? atoi(argv[4]) : 1;
    run_litmus(T, mode, reps);
    return 0;
  }
  if (strcmp(cmd, "sb") == 0) {
    int T = argc > 2 ? atoi(argv[2]) : 5000000;
    int reps = argc > 3 ? atoi(argv[3]) : 1;
    run_sb(T, reps);
    return 0;
  }
  if (strcmp(cmd, "ring") == 0) {
    int NS = argc > 2 ? atoi(argv[2]) : 2000000;
    int CAP = argc > 3 ? atoi(argv[3]) : 64;
    int PL = argc > 4 ? atoi(argv[4]) : 256;
    int G = argc > 5 ? atoi(argv[5]) : 34;
    int mode = argc > 6 ? atoi(argv[6]) : 0;
    int reps = argc > 7 ? atoi(argv[7]) : 1;
    run_ring(NS, CAP, PL, G, mode, reps);
    return 0;
  }

  // ---- bench ----
  printf("\n===== §A message-passing litmus (P: data=1; flag=1  Q: while(flag==0); r1=data) =====\n");
  int f_relax = 0, f_rel = 0;
  f_relax += run_litmus(20000000, 0, 1);
  f_relax += run_litmus(20000000, 0, 1);
  f_rel   += run_litmus(20000000, 1, 1);
  f_rel   += run_litmus(20000000, 1, 1);
  printf("  >> relaxed 共 lost=%d（store→store 未重排），release/acquire 共 lost=%d\n", f_relax, f_rel);
  printf("  >> 正对照 store-buffering（应可观测 store→load 重排）：\n");
  int f_sb = 0;
  f_sb += run_sb(5000000, 1);
  f_sb += run_sb(5000000, 1);
  printf("  >> SB both-zero=%d（>0 说明本机弱内存重排可观测、检测器有效）\n", f_sb);

  printf("\n===== §B in-kernel 生产者→消费者环形队列（payload=PL 字，可逐字校验） =====\n");
  printf("  正对照（人为拉长「发布→写 payload」窗口 ~2.2us，检测器应抓错）：\n");
  run_ring(20000, 64, 256, 34, 4, 1);     // pubslp, PL=256
  run_ring(2000, 4, 8192, 2, 4, 1);       // pubslp, PL=8192
  printf("  自然「先发布后写」（= P46 §35 的真实写法，无 fence）：\n");
  run_ring(200000, 256, 256, 34, 2, 1);
  run_ring(20000, 4, 8192, 2, 2, 1);
  printf("  「写一半就发布」（无 fence）：\n");
  run_ring(200000, 256, 256, 34, 3, 1);
  printf("  正确程序序对照（wro = 无 fence / rel = MPK release+acquire）：\n");
  run_ring(200000, 256, 256, 34, 0, 1);
  run_ring(200000, 256, 256, 34, 1, 1);
  run_ring(20000, 4, 8192, 2, 0, 1);
  run_ring(20000, 4, 8192, 2, 1, 1);

  printf("\n===== §C release/acquire 的代价（小 payload 放大队列机制本身，ns/slot 差 = 每槽 fence） =====\n");
  run_ring(400000, 64, 1, 2, 0, 1);
  run_ring(400000, 64, 1, 2, 1, 1);
  run_ring(400000, 64, 1, 34, 0, 1);
  run_ring(400000, 64, 1, 34, 1, 1);
  printf("  大 payload（fence 被访存摊薄）：\n");
  run_ring(200000, 256, 256, 34, 0, 1);
  run_ring(200000, 256, 256, 34, 1, 1);
  return 0;
}
