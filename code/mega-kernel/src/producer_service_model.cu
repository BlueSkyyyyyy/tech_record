// ============================================================================
// P75-1 (§63)：producer 服务模型——「每环服务成本 vs 环数」的隔离测量
//
// 背景（接 §62）：
//   - §62 在**完整端到端骨架**（producer 写 per-worker 环 + worker TMA 取 task + SMEM 折叠 +
//     HBM 数据面）上发现：固定 NB=2048、固定 PS=8 时，端到端 taskGB/s 随 worker 环数 G 增大
//     **反而下降**（2058@G=528 → 1768@G=1040），并把成本归因到「**少 producer 轮转服务多
//     per-worker 环的扇出**」（§62.7）。
//   - 但 §62 的骨架里 producer 与 worker 同时在场，故那笔成本**无法区分是 producer 写侧
//     的扇出，还是 worker 读侧等更多环**——§62.10 明留尾巴：「`G` 与 `chunk` 天然耦合
//     （`chunk=⌈NB/G⌉`）……**更细的 producer 服务模型（每环服务成本 vs 环数）留待后续**」。
//
// 本节把 **producer 写端单独抽出来**（§A–§D：无 worker、无数据面、无背压）：PS 个 producer、
//   R 个环、每环 `chunk` 个描述符批次（批次 = K*DW 个字 = 352 B），producer s 连续服务
//   `wpp=⌈R/PS⌉` 个环、逐「轮」r 轮转写 + `fence + st.release` 发布 head。
//   唯一控制变量 = **每个 producer 服务的环数 `wpp`**，并**固定每个 producer 写的总批数
//   `Btot = chunk*wpp`**（= 每 producer 的描述符工作量恒定）——于是吞吐随 `wpp` 的变化
//   就**纯粹**是「环数」的代价，把 §62 的归因钉死在 producer 侧（或证伪）。
// 再用 **§F 对照**：同一 (PS,R,chunk) 下加回 consumer（plain-load 消费 + tail 回收、无 TMA/
//   无数据面），看「环数代价」是否只在**有消费者时**才出现。
//
// Section：
//   §A 正确性：producer 写出内容逐位对 host 参考（sink 校验，gid 连续覆盖）。
//   §B headline：固定 PS、固定每 producer 总批数 Btot，扫环数 wpp（chunk=Btot/wpp）。
//   §C 对照：固定 wpp=1（每 producer 一环），扫 Btot（chunk）——纯「每批」成本标度（基线）。
//   §D PS 标度：固定 wpp、Btot，扫 producer 数 PS。
//   §E clock64：lane0 每 producer 的总 cycles 与「每批 cycles」。
//   §F 对照：producer-only vs +consumer，同 (PS,R,chunk) 扫环数 R。
//
// 用法：./producer_service_model bench
//       ./producer_service_model run <PS> <R> <chunk> <K> <DW> <T> <reps> [Gmode]
// 目标卡 H100 / sm90a。安全：脚本带 timeout；设备侧 clock64 守门狗兜死锁。
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

// ---- 内存序原语（对齐 MPK 队列协议 / §62 骨架）----
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
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;\n" ::: "memory");
}

// ============================================================================
// kernel：mode0 = producer-only（grid=PS）；mode1 = producer + consumer（grid=PS+R）
//   R = 环数；chunk = 每环批次数；wpp = ⌈R/PS⌉ = 每 producer 连续服务的环数
//   mode0：gid = s*Btot + (r*wpp + (w-r0))，Btot=chunk*wpp（连续覆盖）
//   mode1：gid = w*chunk + r（= §62 的按环连续），consumer 逐环消费并回收 tail
// 每个批次 = K*DW 个字（=352 B）
// ============================================================================
__global__ void psm_kernel(int PS, int R, int chunk, int wpp, int K, int DW, uint32_t* ring,
                           unsigned long long* head, unsigned long long* tail,
                           unsigned long long* sink, unsigned long long* cyc, int mode,
                           int* abort) {
  const int tid = threadIdx.x;
  const int nl = blockDim.x;
  const int KW = K * DW;
  const int CAPp = chunk;  // 环容量（无背压，与 §62 一致）

  // ---------------- producer：block 0..PS-1 ----------------
  if (blockIdx.x < PS) {
    const int s = blockIdx.x;
    int r0 = s * wpp;
    int r1 = (s + 1) * wpp;
    if (r1 > R) r1 = R;
    unsigned long long part = 0;
    long long t0 = clock64();
    for (int r = 0; r < chunk; ++r) {
      for (int w = r0; w < r1; ++w) {
        const int slot = r % CAPp;
        if (mode == 1 && tid == 0) {
          long long w0 = clock64();
          unsigned long long need = (unsigned long long)(r + 1);
          unsigned long long t = ld_acquire_u64(tail + w);
          while (t + (unsigned long long)CAPp < need) {
            if (abort && *abort) return;
            if (clock64() - w0 > 4000000000LL) {
              if (abort) *abort = 1;
              return;
            }
            __nanosleep(200);
            t = ld_acquire_u64(tail + w);
          }
        }
        __syncthreads();
        uint32_t* dst = ring + (size_t)((size_t)w * CAPp + slot) * (size_t)KW;
        int gid = (mode == 1) ? (w * chunk + r) : (s * chunk * wpp + r * wpp + (w - r0));
        for (int i = tid; i < KW; i += nl) {
          int d = i / DW, wi = i - d * DW;
          uint32_t v = gen_word((uint32_t)gid, wi);
          dst[i] = v;
          part += (unsigned long long)v;
        }
        __syncthreads();
        if (tid == 0) {
          fence_proxy_async_global();
          st_release_u64(head + w, (unsigned long long)(r + 1));
        }
        __syncthreads();
      }
    }
    long long t1 = clock64();
    if (tid == 0 && cyc) cyc[s] = (unsigned long long)(t1 - t0);
    // block 归约
    __shared__ unsigned long long wred[32];
    const int nwarp = (nl + 31) >> 5;
    unsigned long long v = part;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((tid & 31) == 0) wred[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
      unsigned long long ssum = 0;
      for (int i = 0; i < nwarp; ++i) ssum += wred[i];
      atomicAdd(sink, ssum);
    }
    return;
  }

  // ---------------- consumer：block PS..PS+R-1（仅 mode1）----------------
  if (mode == 1) {
    const int w = blockIdx.x - PS;
    unsigned long long part = 0;
    for (int t = 0; t < chunk; ++t) {
      if (tid == 0) {
        long long w0 = clock64();
        unsigned long long need = (unsigned long long)(t + 1);
        unsigned long long h = ld_acquire_u64(head + w);
        while (h < need) {
          if (abort && *abort) return;
          if (clock64() - w0 > 2000000000LL) {
            if (abort) *abort = 1;
            return;
          }
          __nanosleep(100);
          h = ld_acquire_u64(head + w);
        }
      }
      __syncthreads();
      const uint32_t* src = ring + (size_t)((size_t)w * CAPp + (t % CAPp)) * (size_t)KW;
      for (int i = tid; i < KW; i += nl) part += (unsigned long long)src[i];
      __syncthreads();
      if (tid == 0) atom_add_release_u64(tail + w, 1ull);
    }
    __shared__ unsigned long long wred[32];
    const int nwarp = (nl + 31) >> 5;
    unsigned long long v = part;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((tid & 31) == 0) wred[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
      unsigned long long ssum = 0;
      for (int i = 0; i < nwarp; ++i) ssum += wred[i];
      atomicAdd(sink, ssum);
    }
  }
}

// ============================================================================
// host
// ============================================================================
static uint32_t* g_ring = nullptr;
static unsigned long long* g_head = nullptr;
static unsigned long long* g_tail = nullptr;
static unsigned long long* g_sink = nullptr;
static unsigned long long* g_cyc = nullptr;
static int* g_abort = nullptr;
static long long g_ring_cap = 0, g_head_cap = 0, g_cyc_cap = 0;

static void ensure(long long ring_words, int nhead) {
  if (ring_words > g_ring_cap) {
    if (g_ring) CK(cudaFree(g_ring));
    CK(cudaMalloc(&g_ring, (size_t)ring_words * 4));
    g_ring_cap = ring_words;
  }
  if ((long long)nhead > g_head_cap) {
    if (g_head) CK(cudaFree(g_head));
    if (g_tail) CK(cudaFree(g_tail));
    CK(cudaMalloc(&g_head, (size_t)nhead * 8));
    CK(cudaMalloc(&g_tail, (size_t)nhead * 8));
    g_head_cap = nhead;
  }
  if ((long long)nhead > g_cyc_cap) {
    if (g_cyc) CK(cudaFree(g_cyc));
    CK(cudaMalloc(&g_cyc, (size_t)nhead * 8));
    g_cyc_cap = nhead;
  }
  if (!g_sink) CK(cudaMalloc(&g_sink, 8));
  if (!g_abort) CK(cudaMalloc(&g_abort, 4));
}

// 每 gid 的 sink 贡献 = Σ_i gen_word(gid, i%DW)（与 kernel 一致）
static unsigned long long host_ref(long long total_gid, int K, int DW) {
  unsigned long long s = 0;
  int KW = K * DW;
  for (long long g = 0; g < total_gid; ++g)
    for (int i = 0; i < KW; ++i) s += (unsigned long long)gen_word((uint32_t)g, i % DW);
  return s;
}

struct Res {
  double ms;
  double cyc_mean;
  unsigned long long sink;
  int aborted;
};

static Res run(int PS, int R, int chunk, int K, int DW, int T, int reps, int mode) {
  const int wpp = (R + PS - 1) / PS;
  const int KW = K * DW;
  const int CAPp = chunk;
  ensure((long long)R * CAPp * KW, R > PS ? R : PS);
  int grid = (mode == 1) ? (PS + R) : PS;

  Res res;
  res.ms = 1e30;
  res.cyc_mean = 0;
  res.sink = 0;
  res.aborted = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int rr = 0; rr < reps; ++rr) {
    CK(cudaMemset(g_head, 0, (size_t)grid * 8));
    CK(cudaMemset(g_tail, 0, (size_t)grid * 8));
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaMemset(g_cyc, 0, (size_t)grid * 8));
    CK(cudaMemset(g_abort, 0, 4));
    CK(cudaEventRecord(e0));
    psm_kernel<<<grid, T>>>(PS, R, chunk, wpp, K, DW, g_ring, g_head, g_tail, g_sink, g_cyc,
                            mode, g_abort);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < res.ms) res.ms = ms;
    CK(cudaMemcpy(&res.aborted, g_abort, 4, cudaMemcpyDeviceToHost));
    if (res.aborted) break;
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  CK(cudaMemcpy(&res.sink, g_sink, 8, cudaMemcpyDeviceToHost));
  static unsigned long long host_cyc[4096];
  int n = PS > 4096 ? 4096 : PS;
  CK(cudaMemcpy(host_cyc, g_cyc, (size_t)n * 8, cudaMemcpyDeviceToHost));
  double acc = 0;
  for (int i = 0; i < n; ++i) acc += (double)host_cyc[i];
  res.cyc_mean = acc / n;
  return res;
}

static void line(int PS, int R, int chunk, int K, int DW, int T, int reps, int mode,
                 const char* tag) {
  int wpp = (R + PS - 1) / PS;
  int Btot = chunk * wpp;
  Res r = run(PS, R, chunk, K, DW, T, reps, mode);
  long long total_gid = (mode == 1) ? ((long long)R * chunk) : ((long long)PS * Btot);
  double desc_bytes = (double)total_gid * (double)(K * DW) * 4.0;
  double gbs = desc_bytes / (r.ms * 1e-3) / 1e9;
  double cyc_per_batch = r.cyc_mean / (double)Btot;
  const char* v = r.aborted ? "ABORT" : "OK";
  printf("  %-11s %-4s PS=%-3d R=%-5d wpp=%-3d chunk=%-4d -> descGB/s=%-7.1f "
         "cyc/prod=%-9.0f cyc/batch=%-7.1f (%s)\n",
         tag, (mode == 1) ? "+cons" : "prod", PS, R, wpp, chunk, gbs, r.cyc_mean,
         cyc_per_batch, v);
}

static unsigned long long ref_for(int PS, int R, int chunk, int mode, int K, int DW) {
  int wpp = (R + PS - 1) / PS;
  long long total_gid = (mode == 1) ? ((long long)R * chunk) : ((long long)PS * chunk * wpp);
  return host_ref(total_gid, K, DW);
}

static void bench() {
  const int K = 32, DW = 88, T = 256;
  const int PS0 = 32, Btot0 = 64;
  const int reps = 20;

  printf("########################################################################\n");
  printf("# §63 producer 服务模型：每环服务成本 vs 环数（target H100/sm90a）\n");
  printf("# DW=%d(352B/batch)，K=%d，T=%d，纯 producer（§A–§E 无 worker/背压/数据面）\n", DW, K, T);
  printf("########################################################################\n");

  // ===== §A 正确性 =====
  {
    int PS = 8, R = 32, chunk = 8;
    Res r = run(PS, R, chunk, K, DW, T, 3, 0);
    unsigned long long ref = ref_for(PS, R, chunk, 0, K, DW);
    printf("\n===== §A 正确性（PS=%d,R=%d,chunk=%d）：producer-only sink 对 host 参考 =====\n",
           PS, R, chunk);
    printf("  sink=%llu ref=%llu -> %s\n", r.sink, ref, (r.sink == ref) ? "OK" : "WRONG");
  }

  // ===== §B headline：固定每 producer 总批数 Btot，扫环数 wpp =====
  printf("\n===== §B 固定 Btot=%d（每 producer 总批数恒定），扫每 producer 环数 wpp（producer-only）=====\n",
         Btot0);
  {
    int ws[] = {1, 2, 4, 8, 16, 32, 64};
    for (int i = 0; i < 7; ++i) {
      int wpp = ws[i];
      int chunk = Btot0 / wpp;
      int R = PS0 * wpp;
      line(PS0, R, chunk, K, DW, T, reps, 0, "wpp-scan");
    }
  }

  // ===== §C 对照：固定 wpp=1（每 producer 一环），扫 Btot（chunk）=====
  printf("\n===== §C 对照：wpp=1（每 producer 一环），扫 Btot=chunk（纯每批成本标度）=====\n");
  {
    int chunks[] = {8, 16, 32, 64, 128, 256};
    for (int i = 0; i < 6; ++i)
      line(PS0, PS0, chunks[i], K, DW, T, reps, 0, "chunk-scan");
  }

  // ===== §D PS 标度：固定 wpp=8、Btot=64，扫 PS =====
  printf("\n===== §D producer 数标度：固定 wpp=8、Btot=64（chunk=8），扫 PS =====\n");
  {
    int pss[] = {1, 2, 4, 8, 16, 32};
    for (int i = 0; i < 6; ++i) {
      int PS = pss[i];
      line(PS, PS * 8, 8, K, DW, T, reps, 0, "PS-scan");
    }
  }

  // ===== §F 对照：producer-only vs +consumer，同 (PS,R,chunk) 扫环数 R =====
  printf("\n===== §F 对照：PS=8、NB=2048（R*chunk=2048），扫环数 R；同配置 producer-only vs +consumer =====\n");
  {
    int Gs[] = {128, 256, 512, 1024};
    for (int i = 0; i < 4; ++i) {
      int R = Gs[i];
      int chunk = 2048 / R;  // 16,8,4,2
      // 正确性：producer 与 consumer 都累加进 sink，故 full = 2×ref（写值 + 读回值）
      unsigned long long refc = ref_for(8, R, chunk, 1, K, DW) * 2ull;
      Res rc = run(8, R, chunk, K, DW, T, 3, 1);
      line(8, R, chunk, K, DW, T, reps, 0, "F-prod");
      line(8, R, chunk, K, DW, T, reps, 1, "F-cons");
      printf("     (+consumer 校验 sink=%llu 2*ref=%llu -> %s)\n", rc.sink, refc,
             (rc.sink == refc) ? "OK" : "WRONG");
    }
  }

  printf("\n########## §63 完成 ##########\n");
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "bench")) {
    bench();
    return 0;
  }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 9) {
      printf("usage: run PS R chunk K DW T reps [mode]\n");
      return 1;
    }
    int PS = atoi(argv[2]), R = atoi(argv[3]), chunk = atoi(argv[4]), K = atoi(argv[5]);
    int DW = atoi(argv[6]), T = atoi(argv[7]), reps = atoi(argv[8]);
    int mode = (argc >= 10) ? atoi(argv[9]) : 0;
    line(PS, R, chunk, K, DW, T, reps, mode, "run");
    return 0;
  }
  printf("usage: producer_service_model {bench|run ...}\n");
  return 1;
}
