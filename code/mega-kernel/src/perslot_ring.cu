// ============================================================================
// P67-1 (§56)：把「共享前缀 tail」换成「每槽独立回收/就绪」——环协议对分配形态与
//            预取深度 D 的解锁（端到端有界环 + 动态生产者）
//
// 背景：§55（P66-1）把 §54 端到端流水里被钉死为 D=1 的「worker 取 task」维度放开，
//   得到结论「D 不是自由性能旋钮，而是被环协议锁死的自由度」：
//     - strided（跨步 bb=w+t*Gw）+ D>1 在紧环下**死锁**（issue(t+D) 跨出 producer 子环）；
//     - contig（连续分块 bb=w*chunk+t）**不阻塞但静默错**（got ≠ 参考、且"更快"）。
//   §55.8 明留尾巴：「contig 的静默错是『共享前缀 tail 协议』的性质，不是『连续分配』
//   固有的——换 per-slot 回收协议后连续分配可正确；本节隔离的是『分配 × 前缀计数器』
//   这一组合，不外推为『contig 一定错』」，并把「死锁/错误的具体边界未做闭式扫描」
//   列为另一条尾巴。
//
// 本节把这两条一起补掉：实现一个**每槽独立回收/就绪**（per-slot reclaim/ready）的有界环，
// 与 §55 的**共享前缀计数器**（prefix head/tail）在同一骨架、同一工作、同一分配形态上
// 对跑，回答：
//   (1) per-slot 回收是否真的让 contig 正确、并让 D>1 可用（§55.8 的预测）；
//   (2) prefix 协议下 strided+D>1 的死锁边界是否有闭式（CAPp vs D·Gw / bp）；
//   (3) 两种协议的协议开销差多少（per-slot 多一张数组 + 逐槽 release）。
//
// 骨架：P 个 producer 块 + G 个 worker 块（grid = P + G 全共驻），单常驻 kernel。
//   producer p 负责 NB 个 batch 中的 [p*bp, (p+1)*bp)；每个 batch = K 条描述符、每条 DW 字。
//   环按 producer 分区：producer p 的子环 = CAPp 个槽、每槽一个 batch（K*DW 字）。
//   worker 用 TMA bulk load 把槽搬进 SMEM，折叠描述符内容（控制面）；sink 与 host 参考逐位比。
//
// 两种环协议（唯一机制变量 proto）：
//   proto 0 = prefix：producer 背压门 = `tail[p] + CAPp >= (bj+1)*K`（**前缀**已消费计数）；
//                     producer 发布 = `head[p] = (bj+1)*K`（**前缀**就绪计数）；
//                     worker 就绪门 = `head[p] >= (bl+1)*K`；消费后 `atom.add.release tail[p] += K`。
//   proto 1 = perslot：每槽一个 `reclaim[p][s]`（上一占用者消费后写 `bl+1`）与
//                     `ready[p][s]`（producer 写完写 `bj+1`）；
//                     producer 写槽前等 `reclaim[p][s] > bj - CAPp`（**该槽**的前一占用者已回收）；
//                     worker 就绪门 = `ready[own][s] >= bl+1`；消费后 `st.release reclaim[p][s] = bl+1`。
//
// 分配形态 wassign：0 = strided（bb = w + t*Gw）；1 = contig（bb = w*chunk + t）。
// 预取深度 D：worker 在消费第 t 批前先发第 t+D 批的 TMA bulk load（D=1 = 单缓冲）。
//
// 正确性哨兵 = `sink`（全 worker 折叠之和 vs host 参考）；死锁哨兵 = 设备侧 clock64 守门狗 + abort。
//
// 用法：./perslot_ring run <P> <G> <NB> <DW> <K> <D> <CAPp> <proto> <wassign> <reps>
//       ./perslot_ring bench           # 本节全部 Section
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

// ---- payload 内容（与 §48–§55 同：确定性、host/device 一致）----
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
// 一条描述符的折叠（host 参考与 device 用同一函数）
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
// ============================================================================
// mode：0 = static（无生产者，worker 只读 host 预填数组，给参考）
//       1 = proto prefix（共享前缀 head/tail）
//       2 = proto perslot（每槽 reclaim/ready）
__global__ void psr_kernel(int P, int NB, int DW, int K, int D, int CAPp, int mode,
                           int wassign, const uint32_t* __restrict__ g_static,
                           uint32_t* __restrict__ ring, unsigned long long* head,
                           unsigned long long* tail, unsigned long long* ready,
                           unsigned long long* reclaim, unsigned long long* sink,
                           int* abort, int* dbg_done) {
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
      unsigned long long* reclaim_p = reclaim + (size_t)p * CAPp + slot;
      unsigned long long* ready_p = ready + (size_t)p * CAPp + slot;
      // ---- 等槽可用 ----
      if (tid == 0) {
        long long t0 = clock64();
        if (mode == 1) {  // 前缀门：tail[p] + CAPp >= bj+1（均为 batch 计）
          unsigned long long need = (unsigned long long)(bj + 1);
          unsigned long long t = ld_acquire_u64(tail + p);
          while (t + (unsigned long long)CAPp < need) {
            if (*abort) break;
            if (clock64() - t0 > 4000000000LL) { *abort = 1; break; }
            __nanosleep(200);
            t = ld_acquire_u64(tail + p);
          }
        } else {  // 每槽门：reclaim[p][slot] > bj - CAPp
          long long need = (long long)bj - (long long)CAPp;
          unsigned long long r = ld_acquire_u64(reclaim_p);
          while ((long long)r <= need) {
            if (*abort) break;
            if (clock64() - t0 > 4000000000LL) { *abort = 1; break; }
            __nanosleep(200);
            r = ld_acquire_u64(reclaim_p);
          }
        }
      }
      __syncthreads();
      if (*abort) return;
      // ---- 现场生成 K 条描述符写进槽（普通 store → proxy fence → release 发布）----
      const int total = K * DW;
      uint32_t* dst = (uint32_t*)ring + (size_t)((size_t)p * CAPp + slot) * K * DW;
      for (int i = tid; i < total; i += nlanes) {
        int d = i / DW, w = i - d * DW;
        uint32_t gid = (uint32_t)((size_t)(b0 + bj) * K + d);
        dst[i] = gen_word(gid, w);
      }
      __syncthreads();
      if (tid == 0) {
        fence_proxy_async_global();  // 让 async proxy（TMA 读端）看到普通 store
        if (mode == 1) {
          st_release_u64(head + p, (unsigned long long)(bj + 1));
        } else {
          st_release_u64(ready_p, (unsigned long long)(bj + 1));
        }
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
  int nt;
  int b0w;
  if (wassign == 0) {
    b0w = w;
    nt = (w < nb) ? (nb - w + Gw - 1) / Gw : 0;
  } else {
    b0w = w * wchunk;
    nt = (b0w >= nb) ? 0 : (nb - b0w < wchunk ? nb - b0w : wchunk);
  }
  auto bb_of = [&](int t) -> int { return (wassign == 0) ? (w + t * Gw) : (w * wchunk + t); };

  if (tid == 0)
    for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
  __syncthreads();

  unsigned long long acc = 0;

  auto wait_ready = [&](int bb) -> bool {
    if (mode == 0) return true;
    int own = bb / bp;
    if (own >= P) own = P - 1;
    int bl = bb - own * bp;
    int slot = bl % CAPp;
    long long t0 = clock64();
    if (mode == 1) {
      unsigned long long need = (unsigned long long)(bl + 1);
      unsigned long long h = ld_acquire_u64(head + own);
      while (h < need) {
        if (abort && *abort) return false;
        if (clock64() - t0 > 2000000000LL) { if (abort) *abort = 1; return false; }
        __nanosleep(100);
        h = ld_acquire_u64(head + own);
      }
    } else {
      unsigned long long need = (unsigned long long)(bl + 1);
      unsigned long long r = ld_acquire_u64(ready + (size_t)own * CAPp + slot);
      while (r < need) {
        if (abort && *abort) return false;
        if (clock64() - t0 > 2000000000LL) { if (abort) *abort = 1; return false; }
        __nanosleep(100);
        r = ld_acquire_u64(ready + (size_t)own * CAPp + slot);
      }
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
    // 折叠本 batch 的 K 条描述符（lane 并行、块内求和）
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
      unsigned long long s = 0;
      for (int i = 0; i < nlanes; ++i) s += red[i];
      acc += s;
    }
    __syncthreads();
    // ---- 回收 ----
    if (mode != 0 && tid == 0) {
      int own = (bb / bp < P) ? bb / bp : P - 1;
      int bl = bb - own * bp;
      if (mode == 1)
        atom_add_release_u64(tail + own, (unsigned long long)1);
      else
        st_release_u64(reclaim + (size_t)own * CAPp + (bl % CAPp),
                       (unsigned long long)(bl + 1));
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
static unsigned long long* g_ready = nullptr;
static unsigned long long* g_reclaim = nullptr;
static unsigned long long* g_sink = nullptr;
static int* g_abort = nullptr;
static int* g_done = nullptr;
static long long g_static_cap = 0, g_ring_cap = 0, g_head_cap = 0, g_slot_cap = 0;

static void ensure(long long static_words, long long ring_words, int P, int CAPp) {
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
  long long slots = (long long)P * CAPp;
  if (slots > g_slot_cap) {
    if (g_ready) CK(cudaFree(g_ready));
    if (g_reclaim) CK(cudaFree(g_reclaim));
    CK(cudaMalloc(&g_ready, (size_t)slots * 8 + 8));
    CK(cudaMalloc(&g_reclaim, (size_t)slots * 8 + 8));
    g_slot_cap = slots;
  }
  if (!g_sink) CK(cudaMalloc(&g_sink, 8));
  if (!g_abort) CK(cudaMalloc(&g_abort, 4));
  if (!g_done) CK(cudaMalloc(&g_done, 4));
}

// 参考 = Σ fold_desc(gid)，gid ∈ [0, NB*K)
static unsigned long long host_ref(int NB, int K, int DW) {
  unsigned long long s = 0;
  for (long long g = 0; g < (long long)NB * K; ++g) s += fold_desc((uint32_t)g, DW);
  return s;
}

static Res run(int P, int G, int NB, int DW, int K, int D, int CAPp, int mode, int wassign,
               int reps, int T) {
  static bool attr_set = false;
  if (!attr_set) {
    CK(cudaFuncSetAttribute(psr_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));
    attr_set = true;
  }
  const long long desc_words = (long long)NB * K * DW;
  const long long ring_words = (long long)P * CAPp * K * DW;
  ensure(desc_words, ring_words, P, CAPp);
  if (mode == 0)
    fill_static_kernel<<<(int)(((long long)NB * K + 255) / 256), 256>>>(g_static, NB, K, DW);
  // 控制复位
  CK(cudaMemset(g_head, 0, P * 8));
  CK(cudaMemset(g_tail, 0, P * 8));
  CK(cudaMemset(g_ready, 0, (long long)P * CAPp * 8));
  CK(cudaMemset(g_reclaim, 0, (long long)P * CAPp * 8));
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
    // 复位控制（每 rep 都清，避免上轮残留）
    CK(cudaMemset(g_head, 0, P * 8));
    CK(cudaMemset(g_tail, 0, P * 8));
    if (mode == 2) {
      CK(cudaMemset(g_ready, 0, (long long)P * CAPp * 8));
      CK(cudaMemset(g_reclaim, 0, (long long)P * CAPp * 8));
    }
    CK(cudaEventRecord(e0));
    psr_kernel<<<grid, T, req>>>(P, NB, DW, K, D, CAPp, mode, wassign, g_static, g_ring,
                                 g_head, g_tail, g_ready, g_reclaim, g_sink, g_abort, g_done);
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

static const char* proto_name(int mode) {
  return mode == 0 ? "static" : (mode == 1 ? "prefix" : "perslot");
}
static const char* asg_name(int ws) { return ws == 0 ? "strided" : "contig"; }

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
  double gbs = (double)(NB * K * (long long)DW * 4) / (r.ms * 1e-3) / 1e9;
  printf("  P=%-3d G=%-3d NB=%-6d K=%-3d DW=%-3d D=%-2d CAPp=%-5d proto=%-7s asg=%-7s "
         "-> %-6s ms=%.4f descGB/s=%.1f sink=%s done=%d\n",
         P, G, NB, K, DW, D, CAPp, proto_name(mode), asg_name(wassign), verdict, r.ms, gbs,
         (r.sink == ref ? "ref" : (r.aborted ? "-" : "MISMATCH")), r.done);
}

static void bench() {
  const int DW = 88;  // 352 B，对齐 MPK §6.1
  const int K = 32;
  const int T = 256;
  const int P = 32, G = 100;  // grid = 132 = #SM

  printf("########################################################################\n");
  printf("# §56 per-slot 回收 vs 共享前缀：有界环 × 分配形态 × 预取深度 D\n");
  printf("# 目标卡 H100/sm90a，DW=%d(352B)，K=%d，T=%d，P=%d producer + G=%d worker (grid=%d)\n",
         DW, K, T, P, G, P + G);
  printf("########################################################################\n");

  // ===== §A 正确性（宽松环 CAPp >= bp：无 producer 停顿，隔离协议正确性）=====
  {
    const int NB = 4096;
    const int bp = (NB + P - 1) / P;  // 128
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §A 正确性（宽松环：CAPp=%d >= bp=%d，无停顿）=====\n", bp, bp);
    printf("  host_ref = %llu\n", ref);
    for (int mode = 0; mode <= 2; ++mode)
      for (int ws = 0; ws <= 1; ++ws)
        for (int D = 1; D <= 4; D *= 2)
          run_line(P, G, NB, DW, K, D, bp, mode, ws, 3, ref, T);
  }

  // ===== §B 紧环 CAPp=16 批：§55 复现 + per-slot 解锁 =====
  {
    const int NB = 4096;
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §B 紧环（CAPp=16 批，背压真正触发）=====\n");
    printf("  host_ref = %llu\n", ref);
    for (int mode = 1; mode <= 2; ++mode)
      for (int ws = 0; ws <= 1; ++ws)
        for (int D = 1; D <= 8; D *= 2)
          run_line(P, G, NB, DW, K, D, 16, mode, ws, 3, ref, T);
  }

  // ===== §C 紧环下的边界对齐（contig 的 chunk vs bp）=====
  {
    const int NB = 4096;
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §C 紧环 CAPp=16，contig 边界对齐敏感性（chunk vs bp）=====\n");
    // (P,G): (32,100) split(chunk=41<bp=128); (100,32) split(chunk=128>bp=41); (66,66) aligned
    int cfgs[3][2] = {{32, 100}, {100, 32}, {66, 66}};
    for (int i = 0; i < 3; ++i) {
      int p = cfgs[i][0], g = cfgs[i][1];
      int bp = (NB + p - 1) / p;
      int chunk = (NB + g - 1) / g;
      printf("  --- P=%d G=%d bp=%d chunk=%d (%s) ---\n", p, g, bp, chunk,
             chunk == bp ? "aligned" : "split");
      for (int mode = 1; mode <= 2; ++mode)
        run_line(p, g, NB, DW, K, 1, 16, mode, 1, 3, ref, T);
    }
  }

  // ===== §D strided+D>1 死锁边界的闭式（CAPp 扫描）=====
  {
    const int NB = 4096;
    const int bp = (NB + P - 1) / P;         // 128
    const int chunk = (NB + G - 1) / G;      // 41
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §D 边界闭式：CAPp 扫描（P=%d G=%d，Gw=%d，bp=%d，chunk=%d）=====\n", P, G, G,
           bp, chunk);
    int Ds[3] = {2, 4, 8};
    for (int di = 0; di < 3; ++di) {
      int D = Ds[di];
      printf("  --- D=%d（strided 预测边界 CAPp > D*Gw=%d；contig 预测边界 CAPp > D=%d）---\n", D,
             D * G, D);
      for (int CAPp = 4; CAPp <= 512; CAPp *= 2)
        run_line(P, G, NB, DW, K, D, CAPp, 2, 0, 2, ref, T);  // perslot strided
      for (int CAPp = 4; CAPp <= 64; CAPp *= 2)
        run_line(P, G, NB, DW, K, D, CAPp, 2, 1, 2, ref, T);  // perslot contig
    }
    printf("  --- prefix 对照（strided D=2）---\n");
    for (int CAPp = 4; CAPp <= 512; CAPp *= 2)
      run_line(P, G, NB, DW, K, 2, CAPp, 1, 0, 2, ref, T);
  }

  // ===== §E 协议开销（宽松环同工作、同分配）：prefix vs perslot 吞吐 =====
  {
    const int NB = 8192;
    const int bp = (NB + P - 1) / P;
    printf("\n===== §E 协议开销（宽松环 CAPp=bp=%d、strided、D=1/2）：prefix vs perslot =====\n",
           bp);
    unsigned long long ref = host_ref(NB, K, DW);
    (void)ref;
    for (int mode = 1; mode <= 2; ++mode)
      for (int D = 1; D <= 2; ++D)
        run_line(P, G, NB, DW, K, D, bp, mode, 0, 5, host_ref(NB, K, DW), T);
  }

  printf("\n########## §56 完成 ##########\n");
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 12) { printf("usage: run P G NB DW K D CAPp mode wassign reps\n"); return 1; }
    int P = atoi(argv[2]), G = atoi(argv[3]), NB = atoi(argv[4]), DW = atoi(argv[5]);
    int K = atoi(argv[6]), D = atoi(argv[7]), CAPp = atoi(argv[8]), mode = atoi(argv[9]);
    int wassign = atoi(argv[10]), reps = atoi(argv[11]);
    run_line(P, G, NB, DW, K, D, CAPp, mode, wassign, reps, host_ref(NB, K, DW), 256);
    return 0;
  }
  printf("usage: perslot_ring {bench|run ...}\n");
  return 1;
}
