// ============================================================================
// P69-1 (§58)：补掉 §57.8 明留的头号尾巴——真实 MPK 形态是「**少 producer（scheduler）
//             × 多 per-worker 环**」：一个生产者顺序/轮转填多条环、每条环仍单生产者
//             单消费者（前缀计数仍精确），producer 侧不随 G 膨胀。§57 的 `pair` 靠
//             `P==G` 拿到「一消费者一环」，于是 `grid=2G`，producer 侧随 G 膨胀，是本机
//             一直缺的「真实形态」。
//
// 背景（接 §55/§56/§57）：
//   - §55：worker 取 task 的预取深度 D 被「共享前缀环 + 全局 strided/contig 分配」锁死
//          （strided D>1 死锁、contig 静默错）。
//   - §56：换每槽回收（perslot）只解锁「正确性」，且把阈值收紧到 CAPp>=bp。
//   - §57：把 worker→batch 映射对齐成 P==G（每环一个生产者一个消费者）后，prefix + D>1
//          从死锁/静默错变成正确（阈值 CAPp*=D），且同 P/G 下比 shared-strided 快 1.45×。
//          但 §57.8 明留：`pair` 强制 `P==G` → `grid=2G`，producer 侧随 G 膨胀；**真实
//          MPK/DeepGEMM 是「少 producer（scheduler）× 多 per-worker 环」**（MPK 16 scheduler
//          SM → 128 per-worker 队列，`per_worker_queue_len=8192`；DeepGEMM 静态 scheduler
//          warp 填 ring），producer 侧不随 G 膨胀。**本节把这条实现出来。**
//
// 关键设计：把「环所有权」与「block」解耦——
//   - G 个 worker 块，每个拥有**自己的**环 `w`（per-worker ring，前缀计数精确）；
//   - **PS 个 producer（scheduler）块**（PS << G），每个拥有**连续一段** worker 环
//     `[s*wpp,(s+1)*wpp)`（wpp=ceil(G/PS)），在段内**轮转（round-robin）**填环：
//     每轮对段内每条未填完的环写一批 `bl=r`、发布 `head[w]=r+1`。
//   - 每条环**仍只有一个生产者（其 owner producer）与一个消费者（worker w）**：前缀
//     `head[w]/tail[w]` 精确等于该环完成数 → D>1 语义仍成立。
//   - grid = **PS + G**（而非 §57 pair 的 2G）。PS==G 时退化为 §57 的 `pair`。
//   环协议固定为**共享前缀（prefix）**，与 §55–§57 完全一致，隔离「producer 数」这一唯一变量。
//
// 回答三个问题：
//   (1) **正确性**：少 producer + per-worker 环下，prefix + D>1 是否仍从死锁/静默错变正确
//       （即 §57 的解锁是**拓扑**性质、与 producer 数无关）？
//   (2) **producer 侧代价**：PS << G 时生产者成为共享瓶颈——吞吐随 PS 的标度与拐点，
//       grid 从 2G 降到 PS+G 省下多少 SM。
//   (3) **阈值**：per-ring 的 CAPp*=D 是否与 PS 无关。
//
// 骨架与 §57 同：producer 写描述符到 global ring（标量 store）、worker 用 TMA bulk load
//   搬进 SMEM 折叠；sink 与 host 参考逐位比；clock64 守门狗 + abort 兜死锁。
//
// 用法：./sched_perworker_ring run <PS> <G> <NB> <DW> <K> <D> <CAPp> <reps>
//       ./sched_perworker_ring bench          # 本节全部 Section
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
// kernel（单常驻：block 0..PS-1 = producer/scheduler，block PS..PS+G-1 = worker）
//   mode：0 = static（无生产者，worker 只读 host 预填数组，给参考）
//         1 = per-worker 环：worker w 拥有环 w，producer owner(w)=w/wpp 填环
//   PS 个 producer 每个拥有连续一段 worker 环，段内轮转填。
// ============================================================================
__global__ void spr_kernel(int PS, int G, int NB, int DW, int K, int D, int CAPp, int mode,
                           int CW, const uint32_t* __restrict__ g_static,
                           uint32_t* __restrict__ ring, unsigned long long* head,
                           unsigned long long* tail, unsigned long long* sink, int* abort,
                           int* dbg_done) {
  const int tid = threadIdx.x;
  const int nlanes = blockDim.x;
  extern __shared__ char smem_raw[];
  uint32_t* smem = reinterpret_cast<uint32_t*>(smem_raw);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + (size_t)D * K * DW);
  const int bytes_per_batch = K * DW * 4;

  const int chunk = (NB + G - 1) / G;
  const int nb = NB;

  auto count_of = [&](int w) -> int {
    int b0 = w * chunk;
    if (b0 >= nb) return 0;
    int c = nb - b0;
    return (c < chunk) ? c : chunk;
  };

  // ---------------- producer / scheduler：block 0..PS-1 ----------------
  if (mode != 0 && blockIdx.x < PS) {
    const int s = blockIdx.x;
    const int wpp = (G + PS - 1) / PS;
    int r0 = s * wpp;
    int r1 = (s + 1) * wpp;
    if (r1 > G) r1 = G;
    int maxr = 0;
    for (int w = r0; w < r1; ++w) {
      int c = count_of(w);
      if (c > maxr) maxr = c;
    }
    // 段内轮转填：每轮给每条未填完的环写一批 bl=r。
    for (int r = 0; r < maxr; ++r) {
      for (int w = r0; w < r1; ++w) {
        int c = count_of(w);
        if (r >= c) continue;
        const int slot = r % CAPp;
        if (tid == 0) {
          long long t0 = clock64();
          unsigned long long need = (unsigned long long)(r + 1);
          unsigned long long t = ld_acquire_u64(tail + w);
          while (t + (unsigned long long)CAPp < need) {
            if (*abort) break;
            if (clock64() - t0 > 4000000000LL) { *abort = 1; break; }
            __nanosleep(200);
            t = ld_acquire_u64(tail + w);
          }
        }
        __syncthreads();
        if (*abort) return;
        const int total = K * DW;
        uint32_t* dst = (uint32_t*)ring + (size_t)((size_t)w * CAPp + slot) * K * DW;
        const int bb = w * chunk + r;  // 该环第 r 批 = 全局批 w*chunk+r
        for (int i = tid; i < total; i += nlanes) {
          int d = i / DW, wi = i - d * DW;
          uint32_t gid = (uint32_t)((size_t)bb * K + d);
          dst[i] = gen_word(gid, wi);
        }
        __syncthreads();
        if (tid == 0) {
          fence_proxy_async_global();
          st_release_u64(head + w, (unsigned long long)(r + 1));
        }
        __syncthreads();
      }
    }
    if (tid == 0 && dbg_done) atomicAdd(dbg_done, 1);
    return;
  }

  // ---------------- worker：block PS..PS+G-1 ----------------
  const int w = (mode != 0) ? (blockIdx.x - PS) : blockIdx.x;
  const int nt = (w < G) ? count_of(w) : 0;
  const int b0w = w * chunk;

  if (tid == 0)
    for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
  __syncthreads();

  unsigned long long acc = 0;

  auto wait_ready = [&](int bl) -> bool {
    if (mode == 0) return true;
    long long t0 = clock64();
    unsigned long long need = (unsigned long long)(bl + 1);
    unsigned long long h = ld_acquire_u64(head + w);
    while (h < need) {
      if (abort && *abort) return false;
      if (clock64() - t0 > 2000000000LL) { if (abort) *abort = 1; return false; }
      __nanosleep(100);
      h = ld_acquire_u64(head + w);
    }
    return true;
  };

  auto issue = [&](int t) -> bool {
    int slot = t % D;
    if (!wait_ready(t)) return false;
    const uint32_t* src;
    if (mode == 0)
      src = g_static + (size_t)(b0w + t) * K * DW;
    else
      src = ring + (size_t)((size_t)w * CAPp + (t % CAPp)) * K * DW;
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
    unsigned long long part = 0;
    for (int j = tid; j < K; j += nlanes) {
      uint32_t a = 0x9e3779b9u;
      for (int i = 0; i < DW; ++i) a = a * 2246822519u + buf[(size_t)j * DW + i];
      part += (unsigned long long)a;
    }
    if (CW > 0) {  // 模拟「计算型 task」：纯 ALU 延迟，不改 part（不改 sink）
      uint32_t b = (uint32_t)part + (uint32_t)(tid * 2654435761u);
      for (int c = 0; c < CW; ++c) b = b * 1664525u + 1013904223u;
      if (b == 0x1234567u) part += b;
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
    if (mode != 0 && tid == 0) atom_add_release_u64(tail + w, (unsigned long long)1);
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
static long long g_static_cap = 0, g_ring_cap = 0, g_ht_cap = 0;

static void ensure(long long static_words, long long ring_words, int G) {
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
  if (G > g_ht_cap) {
    if (g_head) CK(cudaFree(g_head));
    if (g_tail) CK(cudaFree(g_tail));
    CK(cudaMalloc(&g_head, (size_t)G * 8));
    CK(cudaMalloc(&g_tail, (size_t)G * 8));
    g_ht_cap = G;
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

static int g_CW = 0;  // worker 每 batch 的「计算重量」（纯 ALU 延迟，模拟计算型 task）

static Res run(int PS, int G, int NB, int DW, int K, int D, int CAPp, int mode, int reps,
               int T) {
  static bool attr_set = false;
  if (!attr_set) {
    CK(cudaFuncSetAttribute(spr_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            200 * 1024));
    attr_set = true;
  }
  const long long desc_words = (long long)NB * K * DW;
  const long long ring_words = (long long)G * CAPp * K * DW;
  ensure(desc_words, ring_words, G);
  if (mode == 0)
    fill_static_kernel<<<(int)(((long long)NB * K + 255) / 256), 256>>>(g_static, NB, K, DW);
  CK(cudaMemset(g_head, 0, G * 8));
  CK(cudaMemset(g_tail, 0, G * 8));
  CK(cudaMemset(g_sink, 0, 8));
  CK(cudaMemset(g_abort, 0, 4));
  CK(cudaMemset(g_done, 0, 4));

  int grid = (mode == 0) ? G : PS + G;
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
    CK(cudaMemset(g_head, 0, G * 8));
    CK(cudaMemset(g_tail, 0, G * 8));
    CK(cudaEventRecord(e0));
    spr_kernel<<<grid, T, req>>>(PS, G, NB, DW, K, D, CAPp, mode, g_CW, g_static, g_ring,
                                 g_head, g_tail, g_sink, g_abort, g_done);
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

static void run_line(int PS, int G, int NB, int DW, int K, int D, int CAPp, int mode,
                     int reps, unsigned long long ref, int T) {
  Res r = run(PS, G, NB, DW, K, D, CAPp, mode, reps, T);
  const char* verdict;
  if (r.aborted)
    verdict = "ABORT";
  else if (mode == 0)
    verdict = "REF";
  else if (g_CW > 0)
    verdict = "RUN";  // CW>0：sink 语义与 ref 不同（仅是耗时实验），只查不死锁
  else if (r.sink == ref)
    verdict = "OK";
  else
    verdict = "WRONG";
  double gbs = (double)((long long)NB * K * (long long)DW * 4) / (r.ms * 1e-3) / 1e9;
  int wpp = (G + PS - 1) / PS;
  printf("  PS=%-3d G=%-3d wpp=%-3d NB=%-6d K=%-3d DW=%-3d D=%-2d CAPp=%-4d CW=%-6d "
         "grid=%-4d -> %-6s ms=%.4f descGB/s=%.1f%s\n",
         PS, G, wpp, NB, K, DW, D, CAPp, g_CW, PS + G, verdict, r.ms, gbs,
         (g_CW > 0) ? "" : (r.sink == ref ? " (sink=ref)" : (r.aborted ? "" : " (sink=MISMATCH)")));
}

static void bench() {
  const int DW = 88;  // 352 B，对齐 MPK §6.1
  const int K = 32;
  const int T = 256;

  printf("########################################################################\n");
  printf("# §58 少 producer(scheduler) × 多 per-worker 环：真实 MPK 形态\n");
  printf("# 目标卡 H100/sm90a，DW=%d(352B)，K=%d，T=%d，prefix 协议，grid=PS+G\n", DW, K, T);
  printf("########################################################################\n");

  const int G = 100;  // per-worker 环数
  const int Ps[] = {1, 2, 4, 8, 16, 32};  // grid = PS+G <= 132，全共驻（tight 环需共驻）
  const int NP = 6;

  // ===== §A 正确性（宽松环 CAPp=chunk）：per-ring 前缀精确，少 producer 不动数值 =====
  {
    const int NB = 4096;
    const int chunk = (NB + G - 1) / G;
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §A 正确性（宽松环 CAPp=chunk=%d，无停顿）：PS 扫描 × D =====\n", chunk);
    printf("  host_ref = %llu\n", ref);
    for (int i = 0; i < NP; ++i) {
      printf("  --- PS=%d ---\n", Ps[i]);
      for (int D = 1; D <= 8; D *= 2)
        run_line(Ps[i], G, NB, DW, K, D, chunk, 1, 3, ref, T);
    }
  }

  // ===== §B 紧环 CAPp=4：D>1 是否仍被拓扑解锁（与 producer 数无关？）=====
  {
    const int NB = 4096;
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §B 紧环 CAPp=4（背压真正触发，唯一变量 PS）=====\n");
    printf("  host_ref = %llu\n", ref);
    for (int i = 0; i < NP; ++i) {
      printf("  --- PS=%d ---\n", Ps[i]);
      for (int D = 1; D <= 8; D *= 2)
        run_line(Ps[i], G, NB, DW, K, D, 4, 1, 3, ref, T);
    }
  }

  // ===== §C 吞吐随 PS：少数 producer 是否成为共享瓶颈（grid=PS+G）=====
  {
    const int NB = 8192;
    const int chunk = (NB + G - 1) / G;
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §C 吞吐随 PS（宽松环 CAPp=chunk=%d，D=2，NB=%d）=====\n", chunk, NB);
    for (int i = 0; i < NP; ++i)
      run_line(Ps[i], G, NB, DW, K, 2, chunk, 1, 5, ref, T);
  }

  // ===== §D headline：计算型 task 下，需要几个 producer 才喂饱 G 个 worker =====
  {
    const int NB = 8192;
    const int chunk = (NB + G - 1) / G;
    unsigned long long ref = host_ref(NB, K, DW);
    const int Pd[] = {1, 2, 4, 8, 16, 32};  // grid = PS+G <= 132，全共驻
    const int CWs[] = {0, 4096, 16384, 65536, 262144};
    printf("\n===== §D headline：计算重量 CW × producer 数 PS（G=%d，宽松环，D=2，NB=%d）=====\n",
           G, NB);
    for (int ci = 0; ci < 5; ++ci) {
      g_CW = CWs[ci];
      printf("  --- CW=%d（worker 每 batch 纯 ALU 延迟）---\n", g_CW);
      for (int i = 0; i < 6; ++i)
        run_line(Pd[i], G, NB, DW, K, 2, chunk, 1, 3, ref, T);
    }
    g_CW = 0;
  }

  // ===== §E 阈值 CAPp*=D 是否与 PS 无关（per-ring 性质）=====
  {
    const int NB = 4096;
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §E 阈值：绑死 D 扫 CAPp（PS=4 与 PS=16）=====\n");
    for (int pi = 0; pi < 2; ++pi) {
      int PS = (pi == 0) ? 4 : 16;
      printf("  --- PS=%d ---\n", PS);
      for (int di = 0; di < 3; ++di) {
        int D = (di == 0) ? 2 : (di == 1 ? 4 : 8);
        printf("    D=%d（预测 CAPp*=D=%d）--- \n", D, D);
        for (int CAPp = 1; CAPp <= 16; CAPp *= 2)
          run_line(PS, G, NB, DW, K, D, CAPp, 1, 2, ref, T);
      }
    }
  }

  // ===== §F 与 §57 pair 对照：同 G、同工作，PS=G vs PS<<G =====
  {
    const int NB = 8192;
    const int Gp = 66;  // pair 配置：PS=G=66、chunk==bp
    const int chunk = (NB + Gp - 1) / Gp;
    unsigned long long ref = host_ref(NB, K, DW);
    printf("\n===== §F 与 §57 pair 对照（G=%d，同工作 NB=%d，宽松环，D=1/2/4）=====\n", Gp,
           NB);
    const int Ps2[] = {1, 2, 4, 8, 16, 33, 66};
    for (int i = 0; i < 7; ++i) {
      printf("  --- PS=%d (grid=%d) ---\n", Ps2[i], Ps2[i] + Gp);
      for (int D = 1; D <= 4; D *= 2)
        run_line(Ps2[i], Gp, NB, DW, K, D, chunk, 1, 5, ref, T);
    }
  }

  printf("\n########## §58 完成 ##########\n");
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 10) { printf("usage: run PS G NB DW K D CAPp reps [CW]\n"); return 1; }
    int PS = atoi(argv[2]), G = atoi(argv[3]), NB = atoi(argv[4]), DW = atoi(argv[5]);
    int K = atoi(argv[6]), D = atoi(argv[7]), CAPp = atoi(argv[8]), reps = atoi(argv[9]);
    if (argc >= 11) g_CW = atoi(argv[10]);
    run_line(PS, G, NB, DW, K, D, CAPp, 1, reps, host_ref(NB, K, DW), 256);
    return 0;
  }
  printf("usage: sched_perworker_ring {bench|run ...}\n");
  return 1;
}
