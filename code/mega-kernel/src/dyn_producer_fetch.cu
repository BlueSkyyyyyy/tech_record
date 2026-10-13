// dyn_producer_fetch.cu — MegaKernel 专题 P58-1 小实验：
//   **取 task × 动态生产者：描述符由设备侧 scheduler 现场生成（有界环 + 背压），还是读预写数组？**
//
// 动机 / 接续（补齐 §39–§46 每一节都明留的同一条尾巴「未叠动态生产者」）：
//   §43–§46 把「worker 取 task（控制面）」与「执行 task（数据面）」的流水、载体、缓冲预算、
//   占用率、HBM 饱和都量透了，但**取数数组全是 host 预写的静态块**（host fill kernel 写一次、
//   之后 TMA 只读）。真实 megakernel 不是这样：MPK 的 in-kernel scheduler 在设备上**现场**把
//   task 推进 ready 队列（`docs/06` §3/§4），worker 从**有界**队列取——描述符从不以完整数组形式
//   往返 HBM。本节第一次把这条补上：
//
//     Q1（§A）动态生产者 + 有界环 + 背压，取 task 的**数值**是否仍与静态预写一致？
//     Q2（§B）**控制面的 HBM 足迹**：静态数组 = `NDESC·352 B`（本节 132 MB ≫ L2）必须从 HBM 读；
//             动态环 = `CAP·352 B`（本节十几 MB，驻 L2），描述符从不落 HBM——省下多少？
//     Q3（§C）**生产者的成本**：设备侧生成描述符不是免费的（要算、要写环）。生产者工作 `PW`
//             加到多大才会成为整条流水的新瓶颈（取代控制面/数据面成关键路径）？
//     Q4（§D）**有界环的容量 `CAP`**：环是「生产-消费窗口」，`CAP` 太小会被背压卡住（甚至
//             死锁，同 §31–§35），多大才够、有没有上限收益？
//     Q5（§E）接上数据面后，动态生产者是否改变 §44–§46 的「控制面被数据面藏住」结论？
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：
//   - device 侧 scheduler 现场入队：`execute_scheduler`（`persistent_kernel.cuh:1254-1523`），
//     事件三档派发 + `prepare_next_batch` 连续服务；worker 侧 `execute_worker`（`:957-1252`）
//     取 task 前 `ld_acquire` 读队列游标、批量 `cp.async` 把 `TaskDesc` 搬进 SMEM（`:1063-1078`）。
//   - 队列是**有界**的：`per_worker_queue_len=8192` / `per_sched_queue_len=4096`
//     （`:1790-1791`）+ 不溢出 `assert`（`:1027`）；注释 `:1782-1789` 明说容量是被**速率**逼上去的。
//   - MPSC push 三步 = 占位(`atom.add.release`) → relaxed 写 payload → **连续 CAS 放行**
//     （`:1206-1228`）；出队 `ld.acquire`（`:1340`/`:1357-1360`）、派发 `:1495-1502`。
//   - 论文 §5.1 事件驱动（行 1175-1231）、§5.2 混合下发/两步同步（行 1232-1308）、
//     §5.3 paged shared memory（行 1313-1325）、`TaskDesc` 352 B（论文 §6.1 行 1365）、
//     worker 数按物理 SM（§5.1 行 1168）。
//
// 机制（与 §44/§45/§46 同骨架，把「描述符来源」当独立维度）：
//   grid = 1 + G：block 0 = **producer**（等价于设备的 scheduler），其余 G 个 = worker。
//   三种「描述符来源」模式（唯一变量）：
//     mode 0 **static**  = host 预写 `NDESC` 槽的大数组（= §44–§46 口径，无生产者、无等待）。
//     mode 2 **sched**   = 描述符仍预写在大数组（HBM 读），但 producer **只动态发布「第 bb 批就绪」
//                          这一条调度信息**（推 `head`），worker 等 `head` 再取——**隔离「设备侧动态
//                          调度」本身的握手成本**（= MPK 真实形态：tGraph 预计算、scheduler 只推 task id）。
//     mode 1 **gen**     = producer 现场**生成整条 352 B 描述符**写进有界环 `ring`（`CAP` 槽），
//                          worker 从环取——**隔离「设备侧现场生成 payload」的成本**（描述符不落 HBM）。
//   发布采用 MPK 式 **release 发布**：写 payload → `fence.proxy.async.global` → `st.release` 推
//   `head`；受**背压**约束——`head - tail ≥ CAP` 时必须等 worker 退休（`tail`）才能继续。
//   worker 取 task 前 `ld.acquire` 等 `head ≥ 本批末尾`，再用 **TMA bulk + 硬件 mbarrier** 把本批
//   `K` 条描述符搬进 SMEM 环（`D` 槽预取）；执行后 `atom.add.release` 推 `tail`（释放环槽）。
//   `PW` = 生产者每条描述符的额外计算量；`EDW` = 数据面流式读字数；`R` = 执行侧 LCG 压舱轮数；
//   `do_fetch=0` 只跑执行。
//
// 安全：默认 `CAP` 取 ≥ 生产-消费窗口（`G·K`）以避免死锁；探针另有 `clock64` 无进度守门狗，
//   一旦生产者等待 `tail` 超过阈值即置 `abort` 并退出，防死锁 case 挂死（脚本再带 timeout）。
//
// 编译：
//   nvcc -O3 -arch=sm_90a src/dyn_producer_fetch.cu -o /tmp/dyn_producer_fetch
// 用法：
//   ./dyn_producer_fetch run <G> <NDESC> <DW> <EDW> <K> <D> <R> <PW> <CAP> <mode> <reps> [T] [do_fetch]
//   ./dyn_producer_fetch bench

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t _e = (x);                                                             \
    if (_e != cudaSuccess) {                                                          \
      printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));   \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

#define MAXD 64

__host__ __device__ __forceinline__ uint32_t mix(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 内存序原语（对齐 MPK 队列协议；快照 mpk_atoms.cuh:17-96） ----
__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long atom_add_release_u64(unsigned long long* p,
                                                                   unsigned long long v) {
  unsigned long long old;
  asm volatile("atom.add.release.gpu.u64 %0, [%1], %2;" : "=l"(old) : "l"(p), "l"(v)
               : "memory");
  return old;
}

// ---- 硬件 mbarrier + TMA bulk 1D ----
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

// 「执行」一个 batch：控制面折叠（读本批 K 条描述符，走 SMEM 环）+ 数据面（整块连续流式读本批
// `K·EDW` 个字，走 HBM）+ R 轮 LCG 计算压舱。
__device__ __forceinline__ unsigned long long exec_batch(const uint32_t* desc, int K, int DW,
                                                         const uint32_t* dbase, int nlanes,
                                                         int lane, int EDW, int R, int bid) {
  uint32_t a = 0x9e3779b9u;
  for (int j = lane; j < K; j += nlanes)
    for (int i = 0; i < DW; ++i) a = a * 2246822519u + desc[(size_t)j * DW + i];
  uint32_t s = 0;
  const size_t totv = ((size_t)K * (size_t)EDW) >> 2;
  const uint4* dv = reinterpret_cast<const uint4*>(dbase);
  for (size_t i = lane; i < totv; i += nlanes) {
    uint4 x = dv[i];
    s += x.x + x.y + x.z + x.w;
  }
  uint32_t v = a * 2654435761u + s + (uint32_t)bid;
  for (int r = 0; r < R; ++r) v = v * 1664525u + 1013904223u;
  return (unsigned long long)a + (unsigned long long)v;
}

// =====================================================================================
// 主 kernel：producer（block 0）+ G 个 worker（其余 block）
//   mode 0 = 静态预写（无生产者，读大数组 static_ring，CAP=NDESC）
//   mode 1 = 动态生产者（读有界环 ring，producer 现场生成 + 背压）
// =====================================================================================
__global__ void dp_kernel(int NDESC, int DW, int EDW, int K, int D, int R, int PW, int CAP,
                          int mode, const uint32_t* __restrict__ static_ring,
                          uint32_t* __restrict__ ring, unsigned long long* __restrict__ head,
                          unsigned long long* __restrict__ tail,
                          const uint32_t* __restrict__ data,
                          unsigned long long* __restrict__ sink,
                          unsigned long long* __restrict__ psink, int* __restrict__ abort) {
  extern __shared__ __align__(16) uint32_t smem[];
  const int G = (mode != 0) ? (gridDim.x - 1) : gridDim.x;
  const int lane = threadIdx.x;
  const int nlanes = blockDim.x;

  // ---------------- producer：block 0（mode 1/2） ----------------
  if (mode != 0 && blockIdx.x == 0) {
    unsigned long long pacc = 0;
    const long long step = nlanes;  // 每轮 T 个线程各生成 1 条描述符
    for (long long base = 0; base < NDESC; base += step) {
      // 背压：需保证 (base+step) ≤ tail + CAP
      if (lane == 0) {
        unsigned long long need = (unsigned long long)(base + step);
        unsigned long long t = ld_acquire_u64(tail);
        long long t0 = clock64();
        while (mode == 1 && t + (unsigned long long)CAP < need) {
          __nanosleep(200);
          t = ld_acquire_u64(tail);
          if (clock64() - t0 > 4000000000LL) {  // ~4e9 cyc 无进度 ⇒ 死锁，置 abort
            *abort = 1;
            break;
          }
        }
        if (*abort) break;
      }
      __syncthreads();
      if (*abort) break;
      if (mode == 1) {  // gen：现场生成整条 payload 写进环
        long long s = base + lane;
        if (s < NDESC) {
          uint32_t* p = ring + (size_t)(s % CAP) * DW;
          p[0] = (uint32_t)s;
          if (DW > 1) p[1] = (uint32_t)(s % 4);
          if (DW > 2) p[2] = (uint32_t)(s / 4);
          if (DW > 3) p[3] = 0u;
          for (int i = 4; i < DW; ++i) p[i] = mix((uint32_t)s, (uint32_t)i);
        }
      }
      unsigned int pv = (unsigned int)(base + lane);
      for (int r = 0; r < PW; ++r) pv = pv * 1664525u + 1013904223u;
      pacc += pv;
      // 让 generic-proxy 写对 async-proxy（TMA）可见，再发布 head（仅 gen 有 payload 要发）
      if (mode == 1) asm volatile("fence.proxy.async.global;" ::: "memory");
      __syncthreads();
      if (lane == 0) {
        long long nx = base + step;
        if (nx > NDESC) nx = NDESC;
        st_release_u64(head, (unsigned long long)nx);
      }
    }
    if (lane == 0 && psink && pacc) atomicAdd(psink, pacc);
    return;
  }

  // ---------------- worker ----------------
  const int w = (mode != 0) ? (blockIdx.x - 1) : blockIdx.x;
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + (size_t)D * K * DW);
  const int nbatch = NDESC / K;
  const int bytes_per_batch = K * DW * 4;
  unsigned long long acc = 0;

  const int nt = (w < nbatch) ? (nbatch - w + G - 1) / G : 0;

  if (lane == 0)
    for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
  __syncthreads();

  // 等 producer 产出到 need 条（acquire）；mode 0 直接可读
  auto wait_head = [&](unsigned long long need) -> bool {
    if (mode == 0) return true;
    unsigned long long h = ld_acquire_u64(head);
    while (h < need) {
      if (abort && *abort) return false;
      __nanosleep(100);
      h = ld_acquire_u64(head);
    }
    return true;
  };

  auto issue = [&](int t) -> bool {
    int slot = t % D;
    int bb = w + t * G;
    unsigned long long need = (unsigned long long)(bb + 1) * (unsigned long long)K;
    if (!wait_head(need)) return false;
    const uint32_t* src;
    if (mode == 1)
      src = ring + (size_t)(((size_t)bb * K) % (size_t)CAP) * DW;
    else
      src = static_ring + (size_t)bb * K * DW;
    if (lane == 0) {
      mbar_expect_tx(&mbar[slot], (uint32_t)bytes_per_batch);
      bulk_1d(smem + (size_t)slot * K * DW, src, (uint32_t)bytes_per_batch, &mbar[slot]);
    }
    return true;
  };

  bool ok = true;
  for (int t = 0; t < D && t < nt; ++t)
    if (!issue(t)) { ok = false; break; }

  for (int t = 0; t < nt && ok; ++t) {
    int slot = t % D;
    uint32_t phase = (uint32_t)((t / D) & 1u);
    mbar_wait(&mbar[slot], phase);
    const uint32_t* buf = smem + (size_t)slot * K * DW;
    int bb = w + t * G;
    const uint32_t* dbase = (EDW > 0) ? data + (size_t)bb * K * EDW : data;
    acc += exec_batch(buf, K, DW, dbase, nlanes, lane, EDW, R, bb);
    __syncthreads();
    if (mode != 0 && lane == 0)
      atom_add_release_u64(tail, (unsigned long long)K);  // 退休本批 ⇒ 释放环槽
    if (t + D < nt)
      if (!issue(t + D)) { ok = false; break; }
  }

  if (acc) atomicAdd(sink, acc);
}

// =====================================================================================
// host
// =====================================================================================
__global__ void fill_desc_kernel(uint32_t* descs, long long NDESC, int DW) {
  long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= NDESC) return;
  uint32_t* p = descs + id * DW;
  p[0] = (uint32_t)id;
  if (DW > 1) p[1] = (uint32_t)(id % 4);
  if (DW > 2) p[2] = (uint32_t)(id / 4);
  if (DW > 3) p[3] = 0u;
  for (int i = 4; i < DW; ++i) p[i] = mix((uint32_t)id, (uint32_t)i);
}
__global__ void fill_data_kernel(uint32_t* data, long long n, int seed) {
  long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) return;
  data[id] = mix((uint32_t)seed, (uint32_t)(id & 0xffffffffu)) ^ (uint32_t)(id >> 32);
}

static uint32_t* g_static = nullptr;   // 大数组（静态对照）
static uint32_t* g_ring = nullptr;     // 有界环（动态生产者）
static unsigned long long* g_head = nullptr;
static unsigned long long* g_tail = nullptr;
static int* g_abort = nullptr;
static unsigned long long* g_sink = nullptr;
static unsigned long long* g_psink = nullptr;
static uint32_t* g_data = nullptr;
static long long g_static_alloc = -1;
static int g_dw = -1;
static long long g_ring_alloc = -1;
static long long g_data_alloc = -1;
static int g_edw_filled = -1;

static void ensure_static(long long NDESC, int DW) {
  if (g_static_alloc == NDESC && g_dw == DW) return;
  if (g_static) cudaFree(g_static);
  CK(cudaMalloc(&g_static, (size_t)NDESC * DW * 4));
  fill_desc_kernel<<<(int)((NDESC + 255) / 256), 256>>>(g_static, NDESC, DW);
  CK(cudaDeviceSynchronize());
  g_static_alloc = NDESC;
  g_dw = DW;
}
static void ensure_ring(long long CAP, int DW) {
  if (g_ring_alloc == CAP && g_dw == DW) return;
  if (g_ring) cudaFree(g_ring);
  CK(cudaMalloc(&g_ring, (size_t)CAP * DW * 4));
  g_ring_alloc = CAP;
  (void)DW;
}
static void ensure_ctl() {
  if (g_head) return;
  CK(cudaMalloc(&g_head, 8));
  CK(cudaMalloc(&g_tail, 8));
  CK(cudaMalloc(&g_sink, 8));
  CK(cudaMalloc(&g_psink, 8));
  CK(cudaMalloc(&g_abort, 4));
}
static void ensure_data(long long NDESC, int EDW) {
  long long n = (EDW > 0) ? NDESC * (long long)EDW : 1;
  if (n == g_data_alloc && EDW == g_edw_filled) return;
  if (g_data) cudaFree(g_data);
  CK(cudaMalloc(&g_data, (size_t)n * 4));
  fill_data_kernel<<<(int)((n + 255) / 256), 256>>>(g_data, n, 12345);
  CK(cudaDeviceSynchronize());
  g_data_alloc = n;
  g_edw_filled = EDW;
}

static size_t g_optin = 0;

struct Res {
  unsigned long long got;
  unsigned long long pgot;
  double ms;
  int aborted;
};

static int max_blocks_per_sm(size_t dyn) {
  if (dyn > 48 * 1024)
    cudaFuncSetAttribute(dp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  int T = 256;
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, dp_kernel, T, dyn);
  if (nb <= 0 && dyn > 0) {
    int byThr = 2048 / T;
    int bySmem = (int)(233472 / (dyn ? dyn : 1));
    nb = byThr < bySmem ? byThr : bySmem;
  }
  return nb;
}

// mode 0=static, 1=dyn；G = worker 数（不含 producer）
static Res run_dp(int G, long long NDESC, int DW, int EDW, int K, int D, int R, int PW,
                  long long CAPin, int mode, int reps, int T, int do_fetch) {
  Res res = {0, 0, 0, 0};
  D = (D < 1) ? 1 : D;
  if (D > MAXD) D = MAXD;
  NDESC = (NDESC / K) * K;
  long long CAP = (mode == 0) ? NDESC : CAPin;
  if (CAP < K) CAP = K;
  const size_t smem_bytes = (size_t)D * K * DW * 4 + (size_t)D * 8;
  if (do_fetch && smem_bytes > g_optin) {
    printf("  EDW=%-5d K=%-4d D=%-3d R=%-6d SMEM=%6zuB  SKIP (exceeds optin)\n", EDW, K, D, R,
           smem_bytes);
    return res;
  }
  size_t req = smem_bytes;
  if (req < 1024) req = 1024;  // producer block 也开一点（无环时）
  if (req > 48 * 1024)
    CK(cudaFuncSetAttribute(dp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req));

  ensure_ctl();
  ensure_data(NDESC, EDW);
  ensure_static(NDESC, DW);  // mode 0/2 读静态数组
  if (mode == 1) ensure_ring(CAP, DW);

  int grid = (mode != 0) ? (G + 1) : G;

  for (int w = 0; w < 2; ++w) {
    CK(cudaMemset(g_head, 0, 8));
    CK(cudaMemset(g_tail, 0, 8));
    CK(cudaMemset(g_abort, 0, 4));
    dp_kernel<<<grid, T, req>>>(NDESC, DW, EDW, K, D, R, PW, (int)CAP, mode, g_static, g_ring,
                                g_head, g_tail, g_data, g_sink, g_psink, g_abort);
    CK(cudaDeviceSynchronize());
  }

  float ms_min = 1e30f;
  unsigned long long got = 0, pgot = 0;
  int aborted = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaMemset(g_psink, 0, 8));
    CK(cudaMemset(g_head, 0, 8));
    CK(cudaMemset(g_tail, 0, 8));
    CK(cudaMemset(g_abort, 0, 4));
    CK(cudaEventRecord(e0));
    dp_kernel<<<grid, T, req>>>(NDESC, DW, EDW, K, D, R, PW, (int)CAP, mode, g_static, g_ring,
                                g_head, g_tail, g_data, g_sink, g_psink, g_abort);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    CK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < ms_min) ms_min = ms;
    CK(cudaMemcpy(&got, g_sink, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&pgot, g_psink, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&aborted, g_abort, 4, cudaMemcpyDeviceToHost));
  }
  res.got = got;
  res.pgot = pgot;
  res.ms = ms_min;
  res.aborted = aborted;
  printf("  EDW=%-5d K=%-4d D=%-3d R=%-6d PW=%-6d CAP=%-7lld mode=%d  %8.4f ms/rep  got=%llu  "
         "psink=%llu%s\n",
         EDW, K, D, R, PW, CAP, mode, res.ms, res.got, res.pgot, aborted ? "  ABORT" : "");
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return res;
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  const int NSM = prop.multiProcessorCount;
  const size_t OPTIN = prop.sharedMemPerBlockOptin;
  printf("device %d: %s  SMs=%d  cc=%d.%d  L2=%d MB  optin=%zu  maxThreads/SM=%d\n", dev,
         prop.name, NSM, prop.major, prop.minor, (int)(prop.l2CacheSize >> 20), OPTIN,
         prop.maxThreadsPerMultiProcessor);
  CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
  g_optin = OPTIN;

  const int DW = 88;                // 352 B, 对齐 MPK §6.1
  const int T = 256;
  const long long NDESC = 393216;   // ×352 B ≈ 132 MB ≫ L2（静态数组口径）
  const int G = NSM - 1;            // worker 数（producer 占 1，共 NSM 块，全共驻）

  const char* cmd = argc > 1 ? argv[1] : "bench";
  if (strcmp(cmd, "run") == 0) {
    int g = argc > 2 ? atoi(argv[2]) : G;
    long long N = argc > 3 ? atoll(argv[3]) : NDESC;
    int dw = argc > 4 ? atoi(argv[4]) : DW;
    int edw = argc > 5 ? atoi(argv[5]) : 256;
    int K = argc > 6 ? atoi(argv[6]) : 32;
    int D = argc > 7 ? atoi(argv[7]) : 2;
    int R = argc > 8 ? atoi(argv[8]) : 0;
    int PW = argc > 9 ? atoi(argv[9]) : 0;
    long long CAP = argc > 10 ? atoll(argv[10]) : 32768;
    int mode = argc > 11 ? atoi(argv[11]) : 1;
    int reps = argc > 12 ? atoi(argv[12]) : 5;
    int t = argc > 13 ? atoi(argv[13]) : T;
    int fetch = argc > 14 ? atoi(argv[14]) : 1;
    run_dp(g, N, dw, edw, K, D, R, PW, CAP, mode, reps, t, fetch);
    return 0;
  }

  const int K = 32, D_ = 2;
  const size_t SMEM = (size_t)D_ * K * DW * 4 + D_ * 8;
  const long long CAP_SAFE = (long long)G * K * 2 + K;  // ~2× 生产-消费窗口
  printf("worker G=%d (producer 另占 1 块，共 %d/%d SM)  K=%d D=%d  SMEM=%zu B "
         "(maxBlocks/SM@T%d=%d)\n",
         G, G + 1, NSM, K, D_, SMEM, T, max_blocks_per_sm(SMEM));
  printf("静态数组 NDESC·DW·4 = %.1f MB (≫ L2) ; 动态环 CAP_SAFE=%lld slots = %.2f MB\n",
         (double)NDESC * DW * 4 / 1e6, CAP_SAFE, (double)CAP_SAFE * DW * 4 / 1e6);

  // §A 正确性：static(mode0) vs sched(mode2) vs gen(mode1)，D=1..4，小 NDESC，EDW=8
  printf("\n===== §A 正确性：static(m0) / sched(m2) / gen(m1)，D=1..4，NDESC=8192，EDW=8，R=0 =====\n");
  {
    long long N = 8192, CAPa = (long long)G * K * 4 + 1024;
    unsigned long long refS = 0;
    for (int D : {1, 2, 3, 4}) {
      Res s = run_dp(G, N, DW, 8, K, D, 0, 0, 0, 0, 1, T, 1);
      Res c = run_dp(G, N, DW, 8, K, D, 0, 0, CAPa, 2, 1, T, 1);
      Res g = run_dp(G, N, DW, 8, K, D, 0, 0, CAPa, 1, 1, T, 1);
      if (D == 1) refS = s.got;
      printf("      D=%d  static%s  sched%s  gen%s  (all vs static-ref)\n", D,
             s.got == refS ? "==" : "!=", c.got == refS ? "==" : "!=",
             g.got == refS ? "==" : "!=");
    }
  }

  // §B 描述符来源：静态 HBM 数组 vs 设备侧「只调度」vs 设备侧「现场生成」[EDW=0, PW=0]
  printf("\n===== §B 描述符来源 = static(HBM 读) / sched(设备只推就绪) / gen(设备生成 payload) =====\n");
  {
    Res s = run_dp(G, NDESC, DW, 0, K, D_, 0, 0, 0, 0, 9, T, 1);
    Res c = run_dp(G, NDESC, DW, 0, K, D_, 0, 0, CAP_SAFE, 2, 9, T, 1);
    Res g = run_dp(G, NDESC, DW, 0, K, D_, 0, 0, CAP_SAFE, 1, 9, T, 1);
    double bd = (double)NDESC * DW * 4.0;
    printf("      static: %9.4f ms  desc=%8.1f GB/s   (HBM 读预写数组)\n", s.ms,
           bd / (s.ms / 1e3) / 1e9);
    printf("      sched : %9.4f ms  desc=%8.1f GB/s   (HBM 读 + 设备侧调度握手)  vs static=%.3fx\n",
           c.ms, bd / (c.ms / 1e3) / 1e9, c.ms / s.ms);
    printf("      gen   : %9.4f ms  desc=%8.1f GB/s   (设备现场生成, 单 producer 块) vs static=%.3fx\n",
           g.ms, bd / (g.ms / 1e3) / 1e9, g.ms / s.ms);
  }

  // §C 生产者成本 PW 扫描（gen 与 sched 各一档），EDW=0
  printf("\n===== §C 生产者每条描述符的额外计算 PW（gen vs sched），static 对照 [EDW=0] =====\n");
  {
    Res s = run_dp(G, NDESC, DW, 0, K, D_, 0, 0, 0, 0, 9, T, 1);
    printf("      (static 对照 = %.4f ms)\n", s.ms);
    for (int PW : {0, 1, 4, 16, 64, 256, 1024, 4096}) {
      Res g = run_dp(G, NDESC, DW, 0, K, D_, 0, PW, CAP_SAFE, 1, 9, T, 1);
      Res c = run_dp(G, NDESC, DW, 0, K, D_, 0, PW, CAP_SAFE, 2, 9, T, 1);
      printf("      PW=%-5d -> gen/static = %8.3fx   sched/static = %6.3fx\n", PW,
             g.ms / s.ms, c.ms / s.ms);
    }
  }

  // §D 有界环容量 CAP 扫描（gen，PW=0，EDW=0）
  printf("\n===== §D 有界环容量 CAP（gen 的生产-消费窗口）：太小 ⇒ 背压/死锁 [EDW=0, PW=0] =====\n");
  {
    Res s = run_dp(G, NDESC, DW, 0, K, D_, 0, 0, 0, 0, 1, T, 1);
    printf("      (static 对照 = %.4f ms)\n", s.ms);
    for (long long CAP : {(long long)K, (long long)G * K / 2, (long long)G * K,
                          (long long)G * K * 2, (long long)G * K * 8, (long long)NDESC}) {
      if (CAP % K) continue;
      Res d = run_dp(G, NDESC, DW, 0, K, D_, 0, 0, CAP, 1, 9, T, 1);
      printf("      CAP=%-7lld (%6.2f MB) -> ms=%9.4f%s\n", CAP,
             (double)CAP * DW * 4 / 1e6, d.ms, d.aborted ? "  (deadlock)" : "");
    }
  }

  // §E 接上数据面：三来源扫 EDW
  printf("\n===== §E 接数据面：static / sched / gen 扫 EDW（K=%d,D=%d,PW=0）=====\n", K, D_);
  {
    printf("  %-6s %11s %11s %11s %11s %9s %9s\n", "EDW", "static", "sched", "gen", "exec_only",
           "gen/static", "g/max");
    for (int EDW : {0, 16, 64, 256, 1024, 4096}) {
      Res s = run_dp(G, NDESC, DW, EDW, K, D_, 0, 0, 0, 0, 9, T, 1);
      Res c = run_dp(G, NDESC, DW, EDW, K, D_, 0, 0, CAP_SAFE, 2, 9, T, 1);
      Res g = run_dp(G, NDESC, DW, EDW, K, D_, 0, 0, CAP_SAFE, 1, 9, T, 1);
      Res e = run_dp(G, NDESC, DW, EDW, K, D_, 0, 0, 0, 0, 9, T, 0);  // exec_only
      double mxg = (s.ms > e.ms ? s.ms : e.ms);
      printf("  %-6d %11.4f %11.4f %11.4f %11.4f %9.3f %9.3f\n", EDW, s.ms, c.ms, g.ms, e.ms,
             g.ms / s.ms, g.ms / mxg);
    }
  }

  if (g_static) cudaFree(g_static);
  if (g_ring) cudaFree(g_ring);
  if (g_data) cudaFree(g_data);
  cudaFree(g_head);
  cudaFree(g_tail);
  cudaFree(g_sink);
  cudaFree(g_psink);
  cudaFree(g_abort);
  return 0;
}
