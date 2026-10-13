// dyn_producer_vector.cu — MegaKernel 专题 P60-1 小实验：
//   **动态生产者的写入向量化：把 §48 的标量 strided 写换成向量化/协作写后，「生成」的代价能压回多少？**
//
// 动机 / 接续（补掉 §48.8 明留的尾巴②）：
//   §48（P59-1）把「描述符来源」的单块 producer 扩成 **P 块**，量出 `gen` 相对 static 的代价随 P 按
//   ~1/P 下降但**只到 P≈8 就饱和**，自陈瓶颈已从「单 SM 标量写」转到**整机写带宽/发射**；同时留尾巴②：
//   「`gen` 仍是**标量 strided 写**（每线程顺序写 88 字、跨线程步长 352 B、非 coalesced）——向量化/协作写
//   未单独做，向量化至多改变饱和点、不改『生成 ≫ 预计算读』的量级」。
//   本节把这条尾巴实测出来：**同一套协议/环/背压/数值**，只换 producer 的写入形态，看两件事——
//   (a) 写入指令数（4 B→16 B store）与 (b) **访存 coalescing**（逐线程 strided vs 跨线程连续），
//   各自对 `gen` 吞吐/饱和点的贡献，并检验 §48「生成 ≫ 调度」的结论是否仍然成立。
//
//   四个问题：
//   **Q1（§A）** 三种写入形态（标量 / 逐线程 v4 / 协作 coalesced）写出的 payload 是否逐位一致？
//   **Q2（§B，headline）** 向量化后 `gen` 相对 static 的代价是否下降？饱和点是否右移？能追平 static 吗？
//   **Q3（§C）** 纯写带宽（无 worker 消费、grid 只有 P 个 producer）下三种形态的单 SM/整机写吞吐差多少？
//   **Q4（§D）** 接上数据面（EDW>0）后，向量化是否让「生成盖不住执行」进一步收窄？
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：
//   - MPK 的 scheduler 不是单块：H100 配置 **128 worker / 16 scheduler**（`utils.py:25-29`）；
//     但 scheduler 只做「事件→task id」的映射入队（`execute_scheduler` `persistent_kernel.cuh:1254-1523`），
//     payload 来自**编译期预计算的 `FullTaskDesc` 数组**（`runtime.cc` 一次性 emit），**不在设备侧现场生成**。
//   - 队列有界 `per_worker_queue_len=8192` / `per_sched_queue_len=4096`（`:1790-1791`）；MPSC push 三步
//     = 占位 → relaxed 写 payload → **连续 CAS 放行**（`:1206-1228`）。故「现场生成 payload」是 MPK
//     **没有采用**的路线，本节正是量它为什么不该用、以及即使向量化也救不回多少。
//
// 机制（与 §48 同骨架，唯一新变量 = producer 写入形态 `gvar`）：
//   grid = P + G：block `0..P-1` = **producer**（P 个 SM），其余 G = worker。batch 按连续区间分给 P 个
//   producer；环按 producer 分区（`CAPp=CAP/P` 槽、自己的 `head[p]/tail[p]`、P 条独立 SPSC 子环）。
//   **gvar 0** = 标量 strided 写（§48 原样，每线程顺序写 DW 个字）；
//   **gvar 1** = 逐线程 v4 写（每线程仍写自己那条 desc，但用 `uint4` = 4 B→16 B store，指令数 ÷4）；
//   **gvar 2** = 协作 coalesced 写（一个 block 的线程合作写同一批连续 desc，lane 写连续 `uint4`，
//                32 lane 拼成 512 B 连续事务）。三者的 payload **内容完全相同**（按 `gen_word` 生成）。
//   `mode 5 pure` = 只 launch P 个 producer（无 worker、无背压），量**纯写吞吐**。
//
// 安全：§B/§C 的大 CAP 关掉背压只量 producer 吞吐；producer 有 `clock64` 无进度守门狗防死锁挂死，
//   脚本再带 `timeout`。
//
// 编译：nvcc -O3 -arch=sm_90a src/dyn_producer_vector.cu -o /tmp/dyn_producer_vector
// 用法：./dyn_producer_vector run <P> <G> <NDESC> <DW> <EDW> <K> <D> <R> <PW> <CAP> <mode> <reps> [T] [gvar]

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

// 第 gid 条描述符的第 i 个 32-bit 字（三种写入形态共用，保证 payload 内容一致）。
__host__ __device__ __forceinline__ uint32_t gen_word(uint32_t gid, int i) {
  if (i == 0) return gid;
  if (i == 1) return gid % 4u;
  if (i == 2) return gid / 4u;
  if (i == 3) return 0u;
  return mix(gid, (uint32_t)i);
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

// 现场写一条描述符（gvar 0 标量 / gvar 1 v4），ptr 指向该 desc 起点。
__device__ __forceinline__ void write_desc(uint32_t* ptr, uint32_t gid, int DW, int gvar) {
  if (gvar == 0 || (DW & 3)) {
    for (int i = 0; i < DW; ++i) ptr[i] = gen_word(gid, i);
    return;
  }
  uint4* p4 = reinterpret_cast<uint4*>(ptr);
  const int NWU = DW >> 2;
  for (int u = 0; u < NWU; ++u) {
    uint32_t w0 = gen_word(gid, 4 * u), w1 = gen_word(gid, 4 * u + 1);
    uint32_t w2 = gen_word(gid, 4 * u + 2), w3 = gen_word(gid, 4 * u + 3);
    p4[u] = make_uint4(w0, w1, w2, w3);
  }
}

// 执行一个 batch：控制面折叠 + 数据面流式读 + R 轮 LCG 压舱。
__device__ __forceinline__ unsigned long long exec_batch(const uint32_t* desc, int K, int DW,
                                                         const uint32_t* dbase, int nlanes,
                                                         int lane, int EDW, int R, int bid,
                                                         int df) {
  uint32_t a = 0x9e3779b9u;
  if (df)
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
//   mode 0 = 静态预写（读大数组 static_ring）
//   mode 1 = gen（producer 现场生成 payload，写入形态由 gvar 选）
//   mode 2 = sched（payload 预写大数组，producer 只推 head[p]）
//   mode 5 = pure（只有 P 个 producer，无 worker/背压，量纯写吞吐）
// =====================================================================================
__global__ void dpv_kernel(int P, int NDESC, int DW, int EDW, int K, int D, int R, int PW,
                           int CAP, int mode, int bp, int gvar, int df,
                           const uint32_t* __restrict__ static_ring,
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
  const bool is_gen = (mode == 1 || mode == 5);

  // ---------------- producer：block 0..P-1（mode 1/2/5） ----------------
  if (mode != 0 && blockIdx.x < P) {
    const int p = blockIdx.x;
    int b0 = p * bp;
    if (b0 > nb) b0 = nb;
    int b1 = (p + 1) * bp;
    if (b1 > nb) b1 = nb;
    const long long off = (long long)b0 * K;
    const long long len = (long long)(b1 - b0) * K;
    unsigned long long pacc = 0;
    const long long step = nlanes;  // 每轮 T 个线程各生成/发布 1 条描述符（gvar 2 时协作写）
    const int NWU = DW >> 2;
    for (long long base = 0; base < len; base += step) {
      if (lane == 0) {
        unsigned long long need = (unsigned long long)(base + step);
        unsigned long long t = ld_acquire_u64(&tail[p]);
        long long t0 = clock64();
        while (is_gen && t + (unsigned long long)CAPp < need) {
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
      if (is_gen) {  // 现场生成整条 payload 写进自己的子环
        if (gvar == 2 && !(DW & 3)) {
          // 协作 coalesced：一个 block 的线程合作写 base..base+step 这批连续 desc 的 uint4。
          // 全文 uint4 数 = step*NWU，lane 取连续下标 ⇒ 相邻 lane 写相邻 16 B（coalesced）。
          long long tot_u = (long long)step * NWU;
          for (long long gu = lane; gu < tot_u; gu += nlanes) {
            long long s = base + gu / NWU;       // 段内描述符下标
            if (s >= len) break;
            int wu = (int)(gu % NWU);            // 该 desc 内第几个 uint4
            uint32_t gid = (uint32_t)(off + s);
            uint32_t* ptr = ring + (size_t)((size_t)p * CAPp + (size_t)(s % CAPp)) * DW + wu * 4;
            uint4 v = make_uint4(gen_word(gid, 4 * wu), gen_word(gid, 4 * wu + 1),
                                 gen_word(gid, 4 * wu + 2), gen_word(gid, 4 * wu + 3));
            *reinterpret_cast<uint4*>(ptr) = v;
          }
        } else {
          long long s = base + lane;
          if (s < len) {
            uint32_t* ptr = ring + (size_t)((size_t)p * CAPp + (size_t)(s % CAPp)) * DW;
            write_desc(ptr, (uint32_t)(off + s), DW, gvar);
          }
        }
      }
      unsigned int pv = (unsigned int)(base + lane);
      for (int r = 0; r < PW; ++r) pv = pv * 1664525u + 1013904223u;
      pacc += pv;
      if (is_gen) asm volatile("fence.proxy.async.global;" ::: "memory");
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
    if (is_gen)
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
    acc += exec_batch(buf, K, DW, dbase, nlanes, lane, EDW, R, bb, df);
    __syncthreads();
    if (is_gen && lane == 0) {
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
  for (int i = 0; i < DW; ++i) p[i] = gen_word((uint32_t)id, i);
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


struct Res {
  unsigned long long got;
  unsigned long long pgot;
  double ms;
  int aborted;
};

// P = producer 块数；G = worker 数（grid = P+G）。CAP = 总环槽数（P 等分）。
// mode 5（pure）忽略 G，grid = P。
static Res run_scale(int P, int G, long long NDESC, int DW, int EDW, int K, int D, int R, int PW,
                     long long CAPin, int mode, int reps, int T, int gvar, int df = 1) {
  Res res = {0, 0, 0, 0};
  D = (D < 1) ? 1 : D;
  if (D > MAXD) D = MAXD;
  NDESC = (NDESC / K) * K;
  if (P > MAXP) P = MAXP;
  if (P < 1) P = 1;
  const int nb = (int)(NDESC / K);
  const int bp = (nb + P - 1) / P;
  long long CAP = (mode == 0) ? NDESC : CAPin;
  long long CAPp = CAP / P;
  CAPp = (CAPp / K) * K;
  if (CAPp < K) CAPp = K;
  CAP = CAPp * P;
  const size_t smem_bytes = (size_t)D * K * DW * 4 + (size_t)D * 8;
  size_t req = smem_bytes;
  if (req < 1024) req = 1024;
  if (req > 48 * 1024)
    CK(cudaFuncSetAttribute(dpv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req));

  ensure_ctl(P);
  ensure_data(NDESC, EDW);
  ensure_static(NDESC, DW);
  if (mode == 1 || mode == 5) ensure_ring(CAP, DW);

  int grid;
  if (mode == 0) grid = G;
  else if (mode == 5) grid = P;   // 纯写：只有 producer
  else grid = P + G;

  for (int w = 0; w < 2; ++w) {
    CK(cudaMemset(g_head, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_tail, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_abort, 0, 4));
    dpv_kernel<<<grid, T, req>>>(P, (int)NDESC, DW, EDW, K, D, R, PW, (int)CAP, mode, bp, gvar, df,
                                 g_static, g_ring, g_head, g_tail, g_data, g_sink, g_psink, g_abort);
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
    dpv_kernel<<<grid, T, req>>>(P, (int)NDESC, DW, EDW, K, D, R, PW, (int)CAP, mode, bp, gvar, df,
                                 g_static, g_ring, g_head, g_tail, g_data, g_sink, g_psink, g_abort);
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
  printf("  P=%-3d G=%-3d EDW=%-5d K=%-4d D=%-3d R=%-6d PW=%-6d CAPp=%-6lld mode=%d gvar=%d  "
         "%9.4f ms/rep  got=%llu%s\n",
         P, G, EDW, K, D, R, PW, CAPp, mode, gvar, res.ms, res.got, aborted ? "  ABORT" : "");
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
    int gvar = argc > 15 ? atoi(argv[15]) : 0;
    int df = argc > 16 ? atoi(argv[16]) : 1;
    run_scale(P, G, N, dw, edw, Kk, D, R, PW, CAP, mode, reps, t, gvar, df);
    return 0;
  }

  const size_t SMEM = (size_t)D_ * K * DW * 4 + D_ * 8;
  printf("worker/生产者 全共驻：grid = P + G = %d SM。K=%d D=%d SMEM=%zu B\n", NSM, K, D_, SMEM);
  printf("静态数组 NDESC·DW·4 = %.1f MB (≫ L2)\n", (double)NDESC * DW * 4 / 1e6);

  // §A 正确性：P∈{1,2}，gvar∈{0,1,2}，mode static/sched/gen，D=1，有界子环 + 背压
  printf("\n===== §A 正确性：三种写入形态（gvar 0 标量 / 1 v4 / 2 协作 coalesced），NDESC=8192, EDW=8, R=0 =====\n");
  {
    long long N = 8192;
    unsigned long long ref = 0;
    int G = NSM - 4;
    for (int P : {1, 2}) {
      for (int gvar : {0, 1, 2}) {
        long long CAP = (long long)P * K * 8;
        long long lead = (long long)1 * NSM * K;
        if (CAP < lead) CAP = lead;
        Res s = run_scale(P, G, N, DW, 8, K, 1, 0, 0, 0, 0, 1, T, gvar);
        Res c = run_scale(P, G, N, DW, 8, K, 1, 0, 0, CAP, 2, 1, T, gvar);
        Res g = run_scale(P, G, N, DW, 8, K, 1, 0, 0, CAP, 1, 1, T, gvar);
        if (P == 1 && gvar == 0) ref = s.got;
        printf("      P=%d gvar=%d  static%s  sched%s  gen%s  (all vs static-ref %llu)\n", P, gvar,
               s.got == ref ? "==" : "!=", c.got == ref ? "==" : "!=", g.got == ref ? "==" : "!=",
               ref);
      }
    }
  }

  // §B headline：三种写入形态下 gen 相对 static 的代价随 P 标度 [EDW=0, PW=0]
  printf("\n===== §B gen 写入形态 × 生产者块数 P 的标度（EDW=0, PW=0；grid=P+G=NSM 全共驻）=====\n");
  {
    long long CAPbig = NDESC;
    Res s = run_scale(1, NSM, NDESC, DW, 0, K, D_, 0, 0, 0, 0, 7, T, 0);
    double bd = (double)NDESC * DW * 4.0;
    printf("      static(G=%d, 无 producer) = %9.4f ms  desc=%8.1f GB/s\n", NSM, s.ms,
           bd / (s.ms / 1e3) / 1e9);
    printf("  %-4s %-4s %11s %9s %11s %9s %11s %9s %11s %9s\n", "P", "G", "gen_s(ms)", "g_s/st",
           "gen_v4(ms)", "v4/st", "gen_co(ms)", "co/st", "v4/s", "co/s");
    for (int P : {1, 2, 4, 8, 16, 32, 64}) {
      if (P >= NSM) break;
      int G = NSM - P;
      Res g0 = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAPbig, 1, 7, T, 0);
      Res g1 = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAPbig, 1, 7, T, 1);
      Res g2 = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAPbig, 1, 7, T, 2);
      printf("  %-4d %-4d %11.4f %9.3f %11.4f %9.3f %11.4f %9.3f %11.3f %11.3f\n", P, G, g0.ms,
             g0.ms / s.ms, g1.ms, g1.ms / s.ms, g2.ms, g2.ms / s.ms, g1.ms / g0.ms, g2.ms / g0.ms);
    }
  }

  // §C 纯写吞吐（mode 5：只有 P 个 producer、无 worker、无背压）——隔离写入路径
  printf("\n===== §C 纯写吞吐（mode 5 pure，grid=P，无 worker 消费）：三种写入形态 × P =====\n");
  {
    double bd = (double)NDESC * DW * 4.0;
    printf("  %-4s %12s %12s %12s %10s %10s\n", "P", "w_s(GB/s)", "w_v4(GB/s)", "w_co(GB/s)",
           "v4/s", "co/s");
    for (int P : {1, 2, 4, 8, 16, 32, 64, 128}) {
      if (P > NSM) break;
      Res w0 = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 5, 7, T, 0);
      Res w1 = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 5, 7, T, 1);
      Res w2 = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 5, 7, T, 2);
      double bw0 = bd / (w0.ms / 1e3) / 1e9, bw1 = bd / (w1.ms / 1e3) / 1e9,
             bw2 = bd / (w2.ms / 1e3) / 1e9;
      printf("  %-4d %12.1f %12.1f %12.1f %10.3f %10.3f\n", P, bw0, bw1, bw2, bw1 / bw0, bw2 / bw0);
    }
  }

  // §D 接数据面：P=16 固定，标量 vs 协作，扫 EDW
  printf("\n===== §D 接数据面：gen 扫 EDW（P=16，gvar 0 vs 2；K=%d,D=%d,PW=0）=====\n", K, D_);
  {
    long long CAPbig = NDESC;
    printf("  %-6s %11s %11s %11s %11s %9s %9s\n", "EDW", "gen_s", "gen_co", "exec_only",
           "static", "s/max", "co/max");
    for (int EDW : {0, 16, 64, 256, 1024, 4096}) {
      Res s = run_scale(1, NSM, NDESC, DW, EDW, K, D_, 0, 0, 0, 0, 5, T, 0);
      Res g0 = run_scale(16, NSM - 16, NDESC, DW, EDW, K, D_, 0, 0, CAPbig, 1, 5, T, 0);
      Res g2 = run_scale(16, NSM - 16, NDESC, DW, EDW, K, D_, 0, 0, CAPbig, 1, 5, T, 2);
      Res e = run_scale(1, NSM, NDESC, DW, EDW, K, D_, 0, 0, 0, 0, 5, T, 0, 0);  // df=0 纯执行
      double mx = (s.ms > e.ms ? s.ms : e.ms);
      printf("  %-6d %11.4f %11.4f %11.4f %11.4f %9.3f %9.3f\n", EDW, g0.ms, g2.ms, e.ms, s.ms,
             g0.ms / mx, g2.ms / mx);
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
