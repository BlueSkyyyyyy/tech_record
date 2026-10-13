// dyn_producer_scale.cu — MegaKernel 专题 P59-1 小实验：
//   **动态生产者的多块并行：把 §47 的单块 producer 扩成 P 个 SM，调度/生成代价是否按 P 下降？**
//
// 动机 / 接续（补掉 §47.8 明留的前三条尾巴）：
//   §47（P58-1）把「描述符来源」拆成 **调度（sched，只推就绪）** 与 **生成（gen，现场造 352 B payload）**，
//   量出二者差 ~27×（sched 1.36 ms / gen 36.5 ms vs static 0.16 ms），且结论是「动态必须落在调度、
//   不能落在生成」。但那条结论**全部建立在「单块 producer」**上——§47.8 自陈：
//     - 尾巴①「`gen` 的 producer 是单块……真实系统若真要生成会分散到多块/SM（则代价按生产者块数下降）」；
//     - 尾巴③「`sched` 的 1.36 ms 是『单块 producer』的循环吞吐……MPK 的实际 scheduler 是 **16 SM**」。
//   本节把这台 producer 从 1 块扩到 **P 块（独占 P 个 SM）**，回答「§47 的两条结论在 P 上怎么标度、
//   以及 MPK 的 16 scheduler SM 到底够不够」。
//
//   四个问题：
//   **Q1（§A）** P 个 producer 各写一段（有界子环 + 背压 + 分段 `head[p]/tail[p]`），数值是否仍逐位一致？
//   **Q2（§B，headline）** `gen`/`sched` 相对 static 的代价是否按 **1/P** 下降？在哪饱和、最终能不能追平 static？
//   **Q3（§C）** 多块后 `gen` 的「算被存藏住」/`sched` 的「算直接显形」两条还成立吗？
//   **Q4（§D）** P 个 producer 下**有界环容量**的死锁边界怎么变（每个 producer 一个窗口）？
//   **Q5（§E）** 接上数据面后，多块 producer 会不会让 §47「生成盖不住」翻案成「生成被藏住」？
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：
//   - MPK 的 scheduler 不是单块：H100 配置 **128 worker / 16 scheduler**（`utils.py:25-29`），
//     16 个 scheduler SM 并发做「事件→task id」的映射入队（`execute_scheduler` `persistent_kernel.cuh:1254-1523`）；
//     worker 数按物理 SM（论文 §5.1 行 1168）。故「P 个 producer SM」正是 MPK 的物理形态。
//   - 队列有界：`per_worker_queue_len=8192` / `per_sched_queue_len=4096`（`:1790-1791`）；MPSC push 三步
//     = 占位 → relaxed 写 payload → **连续 CAS 放行**（`:1206-1228`）；论文 §5.2 JIT/AOT（行 1232-1308）。
//
// 机制（与 §47 同骨架，新增「producer 块数 P」这一维度）：
//   grid = P + G：block `0..P-1` = **producer**（P 个 SM），其余 G = worker。P+G = NSM 保证全共驻。
//   batch 数 `nb=NDESC/K` 按**连续区间**分给 P 个 producer：producer `p` 拥有 batch `[p·bp,(p+1)·bp)`。
//   环按 producer 分区：`CAPp = CAP/P` 槽/生产者（K 的倍数），producer `p` 只写自己那一段子环，
//   并维护**自己的** `head[p]`（本地已发布描述符数）与 `tail[p]`（worker 退休数）；背压对
//   `head[p]-tail[p] ≤ CAPp` 生效。worker 取 batch `bb` 时按 `owner=bb/bp` 找对应生产者，等
//   `head[owner]`、从该子环 TMA 读、完成后 `atom.add.release tail[owner] += K`。
//   即 **P 条独立的 SPSC 子环**（§37 的 SPSC 协议），P 个 writer / 多个 reader（按 owner 分工）。
//   `mode 0 static`（P 无用）/ `mode 2 sched`（payload 预写大数组，producer 只推 `head[p]`）/
//   `mode 1 gen`（producer 现场生成 352 B payload 写子环）。`PW` = 每条额外算量；`EDW` = 数据面字数。
//
// 安全：§B/§C 的标度用「CAP 足够大（CAPp ≥ 区间长）」关掉背压，只量 producer 吞吐；§D 才按小 CAP 压边界。
//   producer 另有 `clock64` 无进度守门狗，防死锁 case 挂死（脚本再带 timeout）。
//
// 编译：nvcc -O3 -arch=sm_90a src/dyn_producer_scale.cu -o /tmp/dyn_producer_scale
// 用法：./dyn_producer_scale run <P> <G> <NDESC> <DW> <EDW> <K> <D> <R> <PW> <CAP> <mode> <reps> [T] [do_fetch]

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
#define MAXP 132

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

// 执行一个 batch：控制面折叠 + 数据面流式读 + R 轮 LCG 压舱。
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
// 主 kernel：P 个 producer（block 0..P-1）+ G 个 worker（其余 block）
//   mode 0 = 静态预写（无 producer，读大数组 static_ring）
//   mode 1 = gen（producer 现场生成 payload 写自己的子环 + 背压）
//   mode 2 = sched（payload 预写大数组，producer 只推 head[p]）
//   bp = 每个 producer 拥有的 batch 数（连续区间）
// =====================================================================================
__global__ void dps_kernel(int P, int NDESC, int DW, int EDW, int K, int D, int R, int PW,
                           int CAP, int mode, int bp, const uint32_t* __restrict__ static_ring,
                           uint32_t* __restrict__ ring, unsigned long long* __restrict__ head,
                           unsigned long long* __restrict__ tail,
                           const uint32_t* __restrict__ data,
                           unsigned long long* __restrict__ sink,
                           unsigned long long* __restrict__ psink, int* __restrict__ abort) {
  extern __shared__ __align__(16) uint32_t smem[];
  const int lane = threadIdx.x;
  const int nlanes = blockDim.x;
  const int nb = NDESC / K;
  const int CAPp = (mode == 0) ? 0 : (CAP / P);  // 每 producer 子环槽数（K 的倍数）

  // ---------------- producer：block 0..P-1（mode 1/2） ----------------
  if (mode != 0 && blockIdx.x < P) {
    const int p = blockIdx.x;
    int b0 = p * bp;
    if (b0 > nb) b0 = nb;
    int b1 = (p + 1) * bp;
    if (b1 > nb) b1 = nb;
    const long long off = (long long)b0 * K;
    const long long len = (long long)(b1 - b0) * K;
    unsigned long long pacc = 0;
    const long long step = nlanes;  // 每轮 T 个线程各生成/发布 1 条描述符
    for (long long base = 0; base < len; base += step) {
      if (lane == 0) {
        unsigned long long need = (unsigned long long)(base + step);
        unsigned long long t = ld_acquire_u64(&tail[p]);
        long long t0 = clock64();
        while (mode == 1 && t + (unsigned long long)CAPp < need) {
          __nanosleep(200);
          t = ld_acquire_u64(&tail[p]);
          if (clock64() - t0 > 4000000000LL) {  // ~4e9 cyc 无进度 ⇒ 死锁，置 abort
            *abort = 1;
            break;
          }
        }
        if (*abort) break;
      }
      __syncthreads();
      if (*abort) break;
      if (mode == 1) {  // gen：现场生成整条 payload 写进自己的子环
        long long s = base + lane;
        if (s < len) {
          uint32_t* ptr = ring + (size_t)((size_t)p * CAPp + (size_t)(s % CAPp)) * DW;
          uint32_t gid = (uint32_t)(off + s);
          ptr[0] = gid;
          if (DW > 1) ptr[1] = (uint32_t)(gid % 4);
          if (DW > 2) ptr[2] = (uint32_t)(gid / 4);
          if (DW > 3) ptr[3] = 0u;
          for (int i = 4; i < DW; ++i) ptr[i] = mix(gid, (uint32_t)i);
        }
      }
      unsigned int pv = (unsigned int)(base + lane);
      for (int r = 0; r < PW; ++r) pv = pv * 1664525u + 1013904223u;
      pacc += pv;
      if (mode == 1) asm volatile("fence.proxy.async.global;" ::: "memory");
      __syncthreads();
      if (lane == 0) {
        long long nx = base + step;
        if (nx > len) nx = len;
        st_release_u64(&head[p], (unsigned long long)nx);
      }
    }
    if (lane == 0 && psink && pacc) atomicAdd(psink, pacc);
    return;
  }

  // ---------------- worker ----------------
  const int w = (mode != 0) ? (blockIdx.x - P) : blockIdx.x;
  const int Gw = (mode != 0) ? (gridDim.x - P) : gridDim.x;
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + (size_t)D * K * DW);
  const int nbatch = nb;
  const int bytes_per_batch = K * DW * 4;
  unsigned long long acc = 0;
  const int nt = (w < nbatch) ? (nbatch - w + Gw - 1) / Gw : 0;

  if (lane == 0)
    for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
  __syncthreads();

  auto owner_of = [&](int bb) -> int {
    int o = (bp > 0) ? (bb / bp) : 0;
    if (o >= P) o = P - 1;
    if (o < 0) o = 0;
    return o;
  };
  auto wait_head = [&](int own, unsigned long long need) -> bool {
    if (mode == 0) return true;
    unsigned long long h = ld_acquire_u64(&head[own]);
    while (h < need) {
      if (abort && *abort) return false;
      __nanosleep(100);
      h = ld_acquire_u64(&head[own]);
    }
    return true;
  };

  auto issue = [&](int t) -> bool {
    int slot = t % D;
    int bb = w + t * Gw;
    int own = owner_of(bb);
    int b0 = own * bp;
    unsigned long long need = (unsigned long long)(bb - b0 + 1) * (unsigned long long)K;
    if (!wait_head(own, need)) return false;
    const uint32_t* src;
    if (mode == 1)
      src = ring + (size_t)((size_t)own * CAPp + (size_t)(((bb - b0) * K) % CAPp)) * DW;
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
    int bb = w + t * Gw;
    const uint32_t* dbase = (EDW > 0) ? data + (size_t)bb * K * EDW : data;
    acc += exec_batch(buf, K, DW, dbase, nlanes, lane, EDW, R, bb);
    __syncthreads();
    if (mode == 1 && lane == 0) {
      int own = owner_of(bb);
      atom_add_release_u64(&tail[own], (unsigned long long)K);  // 退休本批 ⇒ 释放该子环槽
    }
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

static uint32_t* g_static = nullptr;
static uint32_t* g_ring = nullptr;
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
static int g_ctl_p = -1;

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
static void ensure_ctl(int P) {
  if (g_ctl_p >= P && g_head) return;
  if (g_head) { cudaFree(g_head); cudaFree(g_tail); }
  if (P < 1) P = 1;
  CK(cudaMalloc(&g_head, sizeof(unsigned long long) * P));
  CK(cudaMalloc(&g_tail, sizeof(unsigned long long) * P));
  if (!g_sink) {
    CK(cudaMalloc(&g_sink, 8));
    CK(cudaMalloc(&g_psink, 8));
    CK(cudaMalloc(&g_abort, 4));
  }
  g_ctl_p = P;
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
    cudaFuncSetAttribute(dps_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  int T = 256;
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, dps_kernel, T, dyn);
  return nb;
}

// P = producer 块数；G = worker 数（grid = P+G）。CAP = 总环槽数（P 等分）。
static Res run_scale(int P, int G, long long NDESC, int DW, int EDW, int K, int D, int R, int PW,
                     long long CAPin, int mode, int reps, int T, int do_fetch) {
  Res res = {0, 0, 0, 0};
  D = (D < 1) ? 1 : D;
  if (D > MAXD) D = MAXD;
  NDESC = (NDESC / K) * K;
  if (P > MAXP) P = MAXP;
  if (P < 1) P = 1;
  const int nb = (int)(NDESC / K);
  const int bp = (nb + P - 1) / P;
  long long CAP = (mode == 0) ? NDESC : CAPin;
  // 保证 CAP 能被 P 整除且每块 ≥ K（子环 K 的倍数）
  long long CAPp = CAP / P;
  CAPp = (CAPp / K) * K;
  if (CAPp < K) CAPp = K;
  CAP = CAPp * P;
  const size_t smem_bytes = (size_t)D * K * DW * 4 + (size_t)D * 8;
  if (do_fetch && smem_bytes > g_optin) {
    printf("  P=%-3d EDW=%-5d K=%-4d D=%-3d R=%-6d SMEM=%6zuB  SKIP (exceeds optin)\n", P, EDW,
           K, D, R, smem_bytes);
    return res;
  }
  size_t req = smem_bytes;
  if (req < 1024) req = 1024;  // producer block 也开一点（无环时）
  if (req > 48 * 1024)
    CK(cudaFuncSetAttribute(dps_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req));

  ensure_ctl(P);
  ensure_data(NDESC, EDW);
  ensure_static(NDESC, DW);
  if (mode == 1) ensure_ring(CAP, DW);

  int grid = (mode != 0) ? (P + G) : G;

  for (int w = 0; w < 2; ++w) {
    CK(cudaMemset(g_head, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_tail, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_abort, 0, 4));
    dps_kernel<<<grid, T, req>>>(P, (int)NDESC, DW, EDW, K, D, R, PW, (int)CAP, mode, bp, g_static,
                                 g_ring, g_head, g_tail, g_data, g_sink, g_psink, g_abort);
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
    CK(cudaMemset(g_head, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_tail, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_abort, 0, 4));
    CK(cudaEventRecord(e0));
    dps_kernel<<<grid, T, req>>>(P, (int)NDESC, DW, EDW, K, D, R, PW, (int)CAP, mode, bp, g_static,
                                 g_ring, g_head, g_tail, g_data, g_sink, g_psink, g_abort);
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
  printf("  P=%-3d G=%-3d EDW=%-5d K=%-4d D=%-3d R=%-6d PW=%-6d CAPp=%-6lld mode=%d  %9.4f ms/rep  "
         "got=%llu%s\n",
         P, G, EDW, K, D, R, PW, CAPp, mode, res.ms, res.got, aborted ? "  ABORT" : "");
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
  const int K = 32, D_ = 2;

  const char* cmd = argc > 1 ? argv[1] : "bench";
  if (strcmp(cmd, "run") == 0) {
    int P = argc > 2 ? atoi(argv[2]) : 1;
    int G = argc > 3 ? atoi(argv[3]) : (NSM - P);
    long long N = argc > 4 ? atoll(argv[4]) : NDESC;
    int dw = argc > 5 ? atoi(argv[5]) : DW;
    int edw = argc > 6 ? atoi(argv[6]) : 256;
    int Kk = argc > 7 ? atoi(argv[7]) : K;
    int D = argc > 8 ? atoi(argv[8]) : D_;
    int R = argc > 9 ? atoi(argv[9]) : 0;
    int PW = argc > 10 ? atoi(argv[10]) : 0;
    long long CAP = argc > 11 ? atoll(argv[11]) : 32768;
    int mode = argc > 12 ? atoi(argv[12]) : 1;
    int reps = argc > 13 ? atoi(argv[13]) : 5;
    int t = argc > 14 ? atoi(argv[14]) : T;
    int fetch = argc > 15 ? atoi(argv[15]) : 1;
    run_scale(P, G, N, dw, edw, Kk, D, R, PW, CAP, mode, reps, t, fetch);
    return 0;
  }

  const size_t SMEM = (size_t)D_ * K * DW * 4 + D_ * 8;
  printf("worker/生产者 全共驻：grid = P + G = %d SM。K=%d D=%d SMEM=%zu B (maxBlocks/SM@T%d=%d)\n",
         NSM, K, D_, SMEM, T, max_blocks_per_sm(SMEM));
  printf("静态数组 NDESC·DW·4 = %.1f MB (≫ L2)\n", (double)NDESC * DW * 4 / 1e6);

  // §A 正确性：P∈{1,2,4}，mode static/sched/gen，D=1..2，有界子环 + 背压，小 NDESC
  printf("\n===== §A 正确性：P 个 producer（分段子环+背压+head[p]/tail[p]），NDESC=8192, EDW=8, R=0 =====\n");
  {
    long long N = 8192;
    unsigned long long ref = 0;
    int G = NSM - 4;
    for (int P : {1, 2, 4}) {
      // 窗口既要有界（背压/wrap），又须 ≥ 预取领先量 (D−1)·G·K（否则单 producer 下 worker 的
      // 第 2 次投机预取会等一个超出窗口的远批 → 死锁，见文档说明）。
      (void)0;
      for (int D : {1, 2}) {
        long long CAP = (long long)P * K * 8;                 // 每 producer 8 批窗口（强制 wrap + 背压）
        long long lead = (long long)(D) * NSM * K;            // ≥ 预取领先量
        if (CAP < lead) CAP = lead;
        Res s = run_scale(P, G, N, DW, 8, K, D, 0, 0, 0, 0, 1, T, 1);
        Res c = run_scale(P, G, N, DW, 8, K, D, 0, 0, CAP, 2, 1, T, 1);
        Res g = run_scale(P, G, N, DW, 8, K, D, 0, 0, CAP, 1, 1, T, 1);
        if (P == 1 && D == 1) ref = s.got;
        printf("      P=%d D=%d  static%s  sched%s  gen%s  (all vs static-ref)\n", P, D,
               s.got == ref ? "==" : "!=", c.got == ref ? "==" : "!=", g.got == ref ? "==" : "!=");
      }
    }
  }

  // §B headline：gen/sched 相对 static 的代价随 P 标度 [EDW=0, PW=0]
  printf("\n===== §B 生产者块数 P 的标度（EDW=0, PW=0；grid=P+G=NSM 全共驻）=====\n");
  {
    long long CAPbig = NDESC;  // 足够大：CAPp ≥ 区间长 ⇒ 关掉背压，只量 producer 吞吐
    Res s = run_scale(1, NSM, NDESC, DW, 0, K, D_, 0, 0, 0, 0, 9, T, 1);
    double bd = (double)NDESC * DW * 4.0;
    printf("      static(G=%d, 无 producer) = %9.4f ms  desc=%8.1f GB/s\n", NSM, s.ms,
           bd / (s.ms / 1e3) / 1e9);
    printf("  %-4s %-4s %11s %9s %9s %11s %9s %9s %12s\n", "P", "G", "gen(ms)", "gen/st", "gen*P",
           "sched(ms)", "sch/st", "sch*P", "gen Mdesc/s");
    for (int P : {1, 2, 4, 8, 16, 32, 64, 128}) {
      if (P >= NSM) break;
      int G = NSM - P;
      Res g = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAPbig, 1, 9, T, 1);
      Res c = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAPbig, 2, 9, T, 1);
      printf("  %-4d %-4d %11.4f %9.3f %9.3f %11.4f %9.3f %9.3f %12.1f\n", P, G, g.ms,
             g.ms / s.ms, g.ms * P, c.ms, c.ms / s.ms, c.ms * P, NDESC / (g.ms / 1e3) / 1e6);
    }
  }

  // §C 每条描述符额外计算 PW：P=1 vs P=16（算被存藏住 / 算直接显形）
  printf("\n===== §C 额外计算 PW 随 P（gen 与 sched），static 对照 [EDW=0] =====\n");
  {
    long long CAPbig = NDESC;
    Res s = run_scale(1, NSM, NDESC, DW, 0, K, D_, 0, 0, 0, 0, 9, T, 1);
    printf("      (static = %.4f ms)\n", s.ms);
    for (int P : {1, 4, 16}) {
      int G = NSM - P;
      printf("      --- P=%d ---\n", P);
      for (int PW : {0, 64, 1024, 4096}) {
        Res g = run_scale(P, G, NDESC, DW, 0, K, D_, 0, PW, CAPbig, 1, 9, T, 1);
        Res c = run_scale(P, G, NDESC, DW, 0, K, D_, 0, PW, CAPbig, 2, 9, T, 1);
        printf("        PW=%-5d -> gen=%9.4f (gen/st %7.3f)   sched=%9.4f (sch/st %6.3f)\n", PW,
               g.ms, g.ms / s.ms, c.ms, c.ms / s.ms);
      }
    }
  }

  // §D 有界环容量：P=1 vs P=4，gen，EDW=0
  printf("\n===== §D 多 producer 的有界环容量（每 producer 窗口 CAPp）：太小 ⇒ 背压/死锁 [EDW=0,PW=0] =====\n");
  {
    long long CAPbig = NDESC;
    Res s = run_scale(1, NSM, NDESC, DW, 0, K, D_, 0, 0, 0, 0, 1, T, 1);
    printf("      (static = %.4f ms)\n", s.ms);
    for (int P : {1, 4}) {
      int G = NSM - P;
      const int nb = (int)(NDESC / K);
      const int bp = (nb + P - 1) / P;
      printf("      --- P=%d (每 producer 窗口 = bp·K = %d 条) ---\n", P, bp * K);
      for (long long CAPp : {(long long)K, (long long)bp * K / 8, (long long)bp * K / 4,
                             (long long)bp * K / 2, (long long)bp * K,
                             (long long)bp * K * 2}) {
        if (CAPp % K) continue;
        long long CAP = CAPp * P;
        if (CAP > CAPbig) continue;
        Res d = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAP, 1, 9, T, 1);
        printf("        CAPp=%-7lld (%6.2f MB total) -> %9.4f ms%s\n", CAPp,
               (double)CAP * DW * 4 / 1e6, d.ms, d.aborted ? "  (deadlock)" : "");
      }
    }
  }

  // §E 接数据面：P=1 vs P=16，gen 扫 EDW
  printf("\n===== §E 接数据面：gen 扫 EDW（P=1 vs P=16；K=%d,D=%d,PW=0）=====\n", K, D_);
  {
    long long CAPbig = NDESC;
    printf("  %-6s %11s %11s %11s %11s %9s %9s\n", "EDW", "gen_P1", "gen_P16", "exec_only",
           "static", "gP1/max", "gP16/max");
    for (int EDW : {0, 16, 64, 256, 1024, 4096}) {
      Res s = run_scale(1, NSM, NDESC, DW, EDW, K, D_, 0, 0, 0, 0, 5, T, 1);
      Res g1 = run_scale(1, NSM - 1, NDESC, DW, EDW, K, D_, 0, 0, CAPbig, 1, 5, T, 1);
      Res g16 = run_scale(16, NSM - 16, NDESC, DW, EDW, K, D_, 0, 0, CAPbig, 1, 5, T, 1);
      Res e = run_scale(1, NSM, NDESC, DW, EDW, K, D_, 0, 0, 0, 0, 5, T, 0);
      double mx = (s.ms > e.ms ? s.ms : e.ms);
      printf("  %-6d %11.4f %11.4f %11.4f %11.4f %9.3f %9.3f\n", EDW, g1.ms, g16.ms, e.ms, s.ms,
             g1.ms / mx, g16.ms / mx);
    }
  }

  if (g_static) cudaFree(g_static);
  if (g_ring) cudaFree(g_ring);
  if (g_data) cudaFree(g_data);
  if (g_head) { cudaFree(g_head); cudaFree(g_tail); }
  cudaFree(g_sink);
  cudaFree(g_psink);
  cudaFree(g_abort);
  return 0;
}
