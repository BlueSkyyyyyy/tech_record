// ============================================================================
// P68-1 (§57)：补上 §55.7 / §56.7 两处反复点名、却一直没实现的「出路」——
//             「保证一个 producer ↔ 一个 consumer 的拓扑 / 每 worker 独立环」
//             （= MPK 快照 per_worker_queue_len=8192 的真实形态）
//
// 背景：§55（P66-1）与 §56（P67-1）把「worker 取 task」量化到：预取深度 D 不是
//   自由性能旋钮，而是被**环协议/拓扑**锁死的自由度——
//     - 前缀 tail 环（prefix）下，全局 strided 分配 + D>1 死锁；
//     - 全局 contig 分配（worker 边界与 producer 边界 chunk≠bp 不对齐）+ prefix
//       **静默错**；换成每槽回收（perslot）只解锁「正确性」、且慢。
//   §55.7/§56.7 两处都给出同一句话的出路：「保证『一个 producer ↔ 一个 consumer』
//   的拓扑，或**每 worker 独立环**」——本机此前从未跑过这个配置，本节补上。
//
// 关键观察：当 **P == G** 时，全局 contig 分配（bb = w*chunk + t）里每个 worker 的
//   连续区间 [w*chunk,(w+1)*chunk) 恰好 == 一个 producer 的区间 [w*bp,(w+1)*bp)
//   （chunk==bp），于是 `own = bb/bp == w`：**每个环恰好一个 producer、一个 consumer**，
//   前缀计数器「恰好等于该环完成数」→ 语义精确 → **D>1 可用且正确**。这就是 §55.7/
//   §56.7 说的「每 worker 独立环 / 一对一对齐拓扑」，本节称 `pair`。
//
// 本节在同一骨架 / 同一工作 / 同一 prefix 协议上，唯一变量 = **worker→batch 映射**
//   （strided vs contig）与 **是否对齐**（P==G 的 contig = pair）：
//     topo A `shared-strided`：bb=w+t*Gw（§55 D>1 死锁基线）
//     topo B `shared-contig` ：bb=w*chunk+t，P≠G、chunk≠bp（§55 静默错）
//     topo C `pair`          ：bb=w*chunk+t，P==G、chunk==bp（一对一对齐，新配置）
//   回答：D 的解锁是不是「对齐/每 worker 独立环」，性能上 pair+D>1 值多少。
//
// 骨架：P 个 producer 块 + G 个 worker 块（grid = P + G 全共驻），单常驻 kernel。
//   producer p 负责连续 NB 批中的 [p*bp,(p+1)*bp)（bp=ceil(NB/P)），写进环 p（CAPp 槽）。
//   worker 用 TMA bulk load 把槽搬进 SMEM、折叠描述符（控制面）；sink 与 host 参考逐位比。
//   环协议固定为**共享前缀**（prefix head/tail），隔离「拓扑」这一唯一变量：
//     背压门 `tail[own]+CAPp >= bl+1`；发布 `head[own]=bl+1`；就绪门 `head[own] >= bl+1`；
//     消费后 `atom.add.release tail[own] += 1`。
//   预取深度 D：worker 在消费第 t 批前先发第 t+D 批的 TMA bulk load（D=1=单缓冲）。
// 正确性哨兵 = `sink`（全 worker 折叠之和 vs host 参考）；死锁哨兵 = clock64 守门狗 + abort。
//
// 用法：./perworker_ring run <P> <G> <NB> <DW> <K> <D> <CAPp> <wassign> <reps>
//       ./perworker_ring bench          # 本节全部 Section
// 目标卡 H100 / sm90a。安全：脚本带 timeout；死锁配置由守门狗兜底返回。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cstring>

#define CK(x)                                                                        \
  do {                                                                               \
    cudaError_t _e = (x);                                                            \
    if (_e != cudaSuccess) {                                                         \
      printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));  \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

#define MAXT 256
#define MAXD 64

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
__host__ __device__ __forceinline__ uint32_t gen_word(uint32_t gid, int i) {
  if (i == 0) return gid;
  if (i == 1) return gid % 4u;
  if (i == 2) return gid / 4u;
  if (i == 3) return 0u;
  return mix32(gid, (uint32_t)i);
}
__host__ __device__ __forceinline__ unsigned long long fold_desc(uint32_t gid, int DW) {
  uint32_t a = 0x9e3779b9u;
  for (int i = 0; i < DW; ++i) a = a * 2246822519u + gen_word(gid, i);
  return (unsigned long long)a;
}

// ---- 内存序原语（对齐 MPK 队列协议；快照 mpk_atoms.cuh:17-96）----
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

// ---- 硬件 mbarrier + TMA bulk load（worker 取 task）----
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes)
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
__device__ __forceinline__ void bulk_load_1d(void* smem_ptr, const void* gmem_ptr,
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
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;\n" ::: "memory");
}

// ============================================================================
// kernel（单常驻：block 0..P-1 = producer，block P..P+G-1 = worker）
//   mode：0 = static（无生产者，worker 只读 host 预填数组，给参考）
//         1 = prefix 环（共享前缀 head/tail；本节唯一协议）
//   wassign：0 = strided（bb = w + t*Gw）；1 = contig（bb = w*chunk + t）
// ============================================================================
__global__ void pwr_kernel(int P, int NB, int DW, int K, int D, int CAPp, int mode,
                           int wassign, const uint32_t* __restrict__ g_static,
                           uint32_t* __restrict__ ring, unsigned long long* head,
                           unsigned long long* tail, unsigned long long* sink, int* abort,
                           int* dbg_done) {
  const int tid = threadIdx.x;
  const int nlanes = blockDim.x;
  extern __shared__ char smem_raw[];
  uint32_t* smem = reinterpret_cast<uint32_t*>(smem_raw);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + (size_t)D * K * DW);
  const int bytes_per_batch = K * DW * 4;

  const int bp = (NB + P - 1) / P;
  const int nb = NB;

  // ---------------- producer：block 0..P-1 ----------------
  if (mode != 0 && blockIdx.x < P) {
    const int p = blockIdx.x;
    int b0 = p * bp;
    if (b0 > nb) b0 = nb;
    int b1 = (p + 1) * bp;
    if (b1 > nb) b1 = nb;
    const int nbp = b1 - b0;
    for (int bj = 0; bj < nbp; ++bj) {
      const int slot = bj % CAPp;
      if (tid == 0) {
        long long t0 = clock64();
        unsigned long long need = (unsigned long long)(bj + 1);
        unsigned long long t = ld_acquire_u64(tail + p);
        while (t + (unsigned long long)CAPp < need) {
          if (*abort) break;
          if (clock64() - t0 > 4000000000LL) { *abort = 1; break; }
          __nanosleep(200);
          t = ld_acquire_u64(tail + p);
        }
      }
      __syncthreads();
      if (*abort) return;
      const int total = K * DW;
      uint32_t* dst = (uint32_t*)ring + (size_t)((size_t)p * CAPp + slot) * K * DW;
      for (int i = tid; i < total; i += nlanes) {
        int d = i / DW, w = i - d * DW;
        uint32_t gid = (uint32_t)((size_t)(b0 + bj) * K + d);
        dst[i] = gen_word(gid, w);
      }
      __syncthreads();
      if (tid == 0) {
        fence_proxy_async_global();
        st_release_u64(head + p, (unsigned long long)(bj + 1));
      }
      __syncthreads();
    }
    if (tid == 0 && dbg_done) atomicAdd(dbg_done, 1);
    return;
  }

  // ---------------- worker ----------------
  const int w = (mode != 0) ? (blockIdx.x - P) : blockIdx.x;
  const int Gw = (mode != 0) ? (gridDim.x - P) : gridDim.x;
  const int wchunk = (nb + Gw - 1) / Gw;
  int nt, b0w;
  if (mode == 0) {
    b0w = w; nt = (w < nb) ? (nb - w + Gw - 1) / Gw : 0;
  } else if (wassign == 0) {
    b0w = w; nt = (w < nb) ? (nb - w + Gw - 1) / Gw : 0;
  } else {
    b0w = w * wchunk;
    nt = (b0w >= nb) ? 0 : (nb - b0w < wchunk ? nb - b0w : wchunk);
  }
  auto bb_of = [&](int t) -> int { return (wassign == 1) ? (w * wchunk + t) : (w + t * Gw); };

  if (tid == 0)
    for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
  __syncthreads();

  unsigned long long acc = 0;

  auto wait_ready = [&](int bb) -> bool {
    if (mode == 0) return true;
    int own = bb / bp;
    if (own >= P) own = P - 1;
    int bl = bb - own * bp;
    long long t0 = clock64();
    unsigned long long need = (unsigned long long)(bl + 1);
    unsigned long long h = ld_acquire_u64(head + own);
    while (h < need) {
      if (abort && *abort) return false;
      if (clock64() - t0 > 2000000000LL) { if (abort) *abort = 1; return false; }
      __nanosleep(100);
      h = ld_acquire_u64(head + own);
    }
    return true;
  };

  auto issue = [&](int t) -> bool {
    int slot = t % D;
    int bb = bb_of(t);
    if (!wait_ready(bb)) return false;
    int own = (bb / bp < P) ? bb / bp : P - 1;
    int bl = bb - own * bp;
    const uint32_t* src;
    if (mode == 0)
      src = g_static + (size_t)bb * K * DW;
    else
      src = ring + (size_t)((size_t)own * CAPp + (bl % CAPp)) * K * DW;
    if (tid == 0) {
      mbar_expect_tx(&mbar[slot], (uint32_t)bytes_per_batch);
      bulk_load_1d(smem + (size_t)slot * K * DW, src, (uint32_t)bytes_per_batch, &mbar[slot]);
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
    int bb = bb_of(t);
    unsigned long long part = 0;
    for (int j = tid; j < K; j += nlanes) {
      uint32_t a = 0x9e3779b9u;
      for (int i = 0; i < DW; ++i) a = a * 2246822519u + buf[(size_t)j * DW + i];
      part += (unsigned long long)a;
    }
    __shared__ unsigned long long red[MAXT];
    red[tid] = part;
    __syncthreads();
    if (tid == 0) {
      unsigned long long ssum = 0;
      for (int i = 0; i < nlanes; ++i) ssum += red[i];
      acc += ssum;
    }
    __syncthreads();
    if (mode != 0 && tid == 0) {
      int own = (bb / bp < P) ? bb / bp : P - 1;
      atom_add_release_u64(tail + own, (unsigned long long)1);
    }
    if (t + D < nt)
      if (!issue(t + D)) { ok = false; break; }
  }

  if (acc && tid == 0) atomicAdd(sink, acc);
  if (tid == 0 && dbg_done) atomicAdd(dbg_done, 1);
}

// ============================================================================
// host
// ============================================================================
__global__ void fill_static_kernel(uint32_t* d, long long nb, int K, int DW) {
  long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long tot = nb * K;
  if (gid >= tot) return;
  uint32_t* p = d + gid * DW;
  for (int i = 0; i < DW; ++i) p[i] = gen_word((uint32_t)gid, i);
}

struct Res {
  double ms;
  unsigned long long sink;
  int aborted;
  int done;
};

static uint32_t* g_static = nullptr;
static uint32_t* g_ring = nullptr;
static unsigned long long* g_head = nullptr;
static unsigned long long* g_tail = nullptr;
static unsigned long long* g_sink = nullptr;
static int* g_abort = nullptr;
static int* g_done = nullptr;
static long long g_static_cap = 0, g_ring_cap = 0, g_head_cap = 0;

static void ensure(long long static_words, long long ring_words, int P) {
  if (static_words > g_static_cap) {
    if (g_static) CK(cudaFree(g_static));
    CK(cudaMalloc(&g_static, static_words * 4));
    g_static_cap = static_words;
  }
  if (ring_words > g_ring_cap) {
    if (g_ring) CK(cudaFree(g_ring));
    CK(cudaMalloc(&g_ring, ring_words * 4));
    g_ring_cap = ring_words;
  }
  if (P > g_head_cap) {
    if (g_head) CK(cudaFree(g_head));
    if (g_tail) CK(cudaFree(g_tail));
    CK(cudaMalloc(&g_head, (size_t)P * 8));
    CK(cudaMalloc(&g_tail, (size_t)P * 8));
    g_head_cap = P;
  }
  if (!g_sink) CK(cudaMalloc(&g_sink, 8));
  if (!g_abort) CK(cudaMalloc(&g_abort, 4));
  if (!g_done) CK(cudaMalloc(&g_done, 4));
}

static unsigned long long host_ref(int NB, int K, int DW) {
  unsigned long long s = 0;
  for (long long g = 0; g < (long long)NB * K; ++g) s += fold_desc((uint32_t)g, DW);
  return s;
}

static Res run(int P, int G, int NB, int DW, int K, int D, int CAPp, int mode, int wassign,
               int reps, int T) {
  static bool attr_set = false;
  if (!attr_set) {
    CK(cudaFuncSetAttribute(pwr_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));
    attr_set = true;
  }
  const long long desc_words = (long long)NB * K * DW;
  const long long ring_words = (long long)P * CAPp * K * DW;
  ensure(desc_words, ring_words, P);
  if (mode == 0)
    fill_static_kernel<<<(int)(((long long)NB * K + 255) / 256), 256>>>(g_static, NB, K, DW);
  CK(cudaMemset(g_head, 0, P * 8));
  CK(cudaMemset(g_tail, 0, P * 8));
  CK(cudaMemset(g_sink, 0, 8));
  CK(cudaMemset(g_abort, 0, 4));
  CK(cudaMemset(g_done, 0, 4));

  int grid = (mode == 0) ? G : P + G;
  size_t req = (size_t)D * K * DW * 4 + (size_t)D * 8 + 64;
  unsigned long long sink = 0;
  double best = 1e30;
  int aborted = 0, done = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaMemset(g_abort, 0, 4));
    CK(cudaMemset(g_done, 0, 4));
    CK(cudaMemset(g_head, 0, P * 8));
    CK(cudaMemset(g_tail, 0, P * 8));
    CK(cudaEventRecord(e0));
    pwr_kernel<<<grid, T, req>>>(P, NB, DW, K, D, CAPp, mode, wassign, g_static, g_ring, g_head,
                                 g_tail, g_sink, g_abort, g_done);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) best = ms;
    CK(cudaMemcpy(&aborted, g_abort, 4, cudaMemcpyDeviceToHost));
    if (aborted) break;
  }
  CK(cudaMemcpy(&sink, g_sink, 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&done, g_done, 4, cudaMemcpyDeviceToHost));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  Res res;
  res.ms = best;
  res.sink = sink;
  res.aborted = aborted;
  res.done = done;
  return res;
}

static const char* topo_name(int P, int G, int wassign) {
  if (wassign == 0) return "shared-strided";
  return (P == G) ? "pair" : "shared-contig";
}

static void run_line(int P, int G, int NB, int DW, int K, int D, int CAPp, int mode,
                     int wassign, int reps, unsigned long long ref, int T) {
  Res r = run(P, G, NB, DW, K, D, CAPp, mode, wassign, reps, T);
  const char* verdict;
  if (r.aborted)
    verdict = "ABORT";
  else if (mode == 0)
    verdict = "REF";
  else if (r.sink == ref)
    verdict = "OK";
  else
    verdict = "WRONG";
  double gbs = (double)((long long)NB * K * (long long)DW * 4) / (r.ms * 1e-3) / 1e9;
  printf("  P=%-3d G=%-3d NB=%-6d K=%-3d DW=%-3d D=%-2d CAPp=%-4d topo=%-14s "
         "-> %-6s ms=%.4f descGB/s=%.1f sink=%s done=%d\n",
         P, G, NB, K, DW, D, CAPp, topo_name(P, G, wassign), verdict, r.ms, gbs,
         (r.sink == ref ? "ref" : (r.aborted ? "-" : "MISMATCH")), r.done);
}

static void bench() {
  const int DW = 88;  // 352 B，对齐 MPK §6.1
  const int K = 32;
  const int T = 256;

  printf("########################################################################\n");
  printf("# §57 每 worker 独立环（pair：P==G，一对一对齐）解锁 prefix+D>1\n");
  printf("# 目标卡 H100/sm90a，DW=%d(352B)，K=%d，T=%d，prefix 协议，grid=P+G=132\n", DW, K, T);
  printf("########################################################################\n");

  // pair：P=G=66（chunk==bp=63）；shared：P=32,G=100（§55/§56 配置）。
  const int Pp = 66, Gp = 66;
  const int Ps = 32, Gs = 100;

  // ===== §A 宽松环 CAPp=bp：无 wrap、隔离协议正确性 =====
  {
    const int NB = 4096;
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §A 正确性（宽松环 CAPp=bp：不 wrap、无停顿）=====\n");
    printf("  host_ref = %llu\n", ref);
    for (int ws = 0; ws <= 1; ++ws) {
      int bp = (NB + Pp - 1) / Pp;
      for (int D = 1; D <= 8; D *= 2)
        run_line(Pp, Gp, NB, DW, K, D, bp, 1, ws, 3, ref, T);
    }
    {
      int bp = (NB + Ps - 1) / Ps;
      for (int D = 1; D <= 8; D *= 2)
        run_line(Ps, Gs, NB, DW, K, D, bp, 1, 1, 3, ref, T);
    }
  }

  // ===== §B 紧环 CAPp=4：wrap 打开 → 拓扑决定 D 能否用 =====
  {
    const int NB = 4096;
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §B 紧环 CAPp=4（背压真正触发，唯一变量拓扑）=====\n");
    printf("  host_ref = %llu\n", ref);
    printf("  --- pair (P=G=%d, chunk==bp) ---\n", Pp);
    for (int D = 1; D <= 8; D *= 2)
      run_line(Pp, Gp, NB, DW, K, D, 4, 1, 1, 3, ref, T);
    printf("  --- shared-strided (P=%d G=%d) ---\n", Ps, Gs);
    for (int D = 1; D <= 8; D *= 2)
      run_line(Ps, Gs, NB, DW, K, D, 4, 1, 0, 3, ref, T);
    printf("  --- shared-contig (P=%d G=%d, chunk!=bp) ---\n", Ps, Gs);
    for (int D = 1; D <= 8; D *= 2)
      run_line(Ps, Gs, NB, DW, K, D, 4, 1, 1, 3, ref, T);
  }

  // ===== §C pair 的解锁阈值 CAPp >= D（单变量 CAPp 扫描）=====
  {
    const int NB = 4096;
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §C pair 解锁阈值：CAPp 扫描（P=G=%d，D 固定）=====\n", Pp);
    for (int di = 0; di < 3; ++di) {
      int D = (di == 0) ? 2 : (di == 1 ? 4 : 8);
      printf("  --- D=%d（预测 CAPp* = D = %d）---\n", D, D);
      for (int CAPp = 1; CAPp <= 16; CAPp *= 2)
        run_line(Pp, Gp, NB, DW, K, D, CAPp, 1, 1, 2, ref, T);
    }
  }

  // ===== §D 性能：pair 的 D 是否真把串行缝藏掉（吞吐随 D）=====
  {
    const int NB = 8192;
    printf("\n===== §D pair 吞吐随 D（宽松环 CAPp=bp）：D>1 是否真值钱 =====\n");
    for (int D = 1; D <= 8; D *= 2)
      run_line(Pp, Gp, NB, DW, K, D, (NB + Pp - 1) / Pp, 1, 1, 5, host_ref(NB, K, DW), T);
  }

  // ===== §E 对照：同工作量、同 grid，三条拓扑 =====
  {
    const int NB = 8192;
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §E 三条拓扑对照（同 grid=132、同工作量）=====\n");
    printf("  --- pair (P=G=%d) ---\n", Pp);
    for (int D = 1; D <= 4; D *= 2)
      run_line(Pp, Gp, NB, DW, K, D, (NB + Pp - 1) / Pp, 1, 1, 5, ref, T);
    printf("  --- shared-strided (P=%d G=%d) ---\n", Ps, Gs);
    for (int D = 1; D <= 4; D *= 2)
      run_line(Ps, Gs, NB, DW, K, D, (NB + Ps - 1) / Ps, 1, 0, 5, ref, T);
    printf("  --- shared-contig (P=%d G=%d) ---\n", Ps, Gs);
    for (int D = 1; D <= 4; D *= 2)
      run_line(Ps, Gs, NB, DW, K, D, (NB + Ps - 1) / Ps, 1, 1, 5, ref, T);
  }

  printf("\n########## §57 完成 ##########\n");
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 11) { printf("usage: run P G NB DW K D CAPp wassign reps\n"); return 1; }
    int P = atoi(argv[2]), G = atoi(argv[3]), NB = atoi(argv[4]), DW = atoi(argv[5]);
    int K = atoi(argv[6]), D = atoi(argv[7]), CAPp = atoi(argv[8]), wassign = atoi(argv[9]);
    int reps = atoi(argv[10]);
    run_line(P, G, NB, DW, K, D, CAPp, 1, wassign, reps, host_ref(NB, K, DW), 256);
    return 0;
  }
  printf("usage: perworker_ring {bench|run ...}\n");
  return 1;
}
