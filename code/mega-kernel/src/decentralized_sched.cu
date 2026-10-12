// decentralized_sched.cu — MegaKernel 专题 P15-1 小实验：
//   **最小「去中心化 in-kernel 调度」megakernel**：设备侧原子 ready-queue + 每 task 事件计数器
//
// 目的（目标卡 H100 / sm90 实测）：
//   MPK 论文的核心运行时机制是**去中心化调度**（decentralized scheduling，§5，快照行 1147–1308；
//   docs/01 §4、docs/03 §3.3）：整张算子图被切成 task，每个 task 完成时原子地更新后继 task 的
//   事件计数器，计数器到阈值即把后继 task 推入**设备侧 ready 队列**（atomicAdd 环形缓冲），worker
//   自取自执行——**没有全局栅栏（grid.sync）**。本专题此前的 persistent kernel 实验（P5-2/P13-1）
//   都停在「cooperative grid.sync() 的集中式会合」，明确标注「不满足 C5」；P10-1 又实测出
//   `grid.sync()` 有 ~1.1µs 硬地板。本实验用一个小而完整的 **dataflow 调度器**把 C5 机制真正跑起来，
//   并与 `grid.sync()` 集中式 barrier 版、kernel-per-operator（eager）版做公平对照。
//
// 任务图（忠实对应「多链依赖」这一最简 DAG）：
//   C 条独立链，每条链 L 个 task 串行依赖：task(c,l) 依赖 task(c,l-1)。
//   task(c,l) 把输入第 c 条链的第 l 段归约，累加到该链的运行和 chain_sum[c]（真依赖，不可跳过）。
//   ---- 三种执行策略，总工作量完全相同 ----
//   eager  : L 次 kernel launch，每次 kernel 以 grid-stride 处理该层全部 C 个 task（流序保证依赖）。
//   barrier: 单 persistent cooperative kernel，逐层处理，每层后 grid.sync()（集中式，= 本专题此前形态）。
//   decen  : 单 persistent kernel，设备侧 ready-queue + 每 task 事件计数器，worker 自取（去中心化，C5）。
//
// 三个 Section：
//   A 扫 L（层数）：固定 C/seg，看三种策略随依赖深度缩放；barrier 每层 +1 次 grid.sync。
//   B 扫 C（链数）：固定 L/seg，看 ready-queue 的调度开销与并行度。
//   C 负载不均衡：各链每段长度不同，看 decen 能否「快链先跑」而不被 barrier 的逐层最慢链拖住。
//
// 正确性：三种策略都应得到同一 chain_sum[c]（同序累加），与串行参考逐位相等 → 打印 max|Δ| 与 fail 数。
//
// 编译：nvcc -O3 -arch=sm_90 decentralized_sched.cu -o decentralized_sched
// 用法：./decentralized_sched [iters] [C] [L] [seg]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
      printf("CUDA ERROR %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,         \
             cudaGetErrorString(_e));                                           \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static const int T = 256;  // 每 block 线程数

// 段归约：block 内 grid-stride 求和 + warp/shared 归约，结果广播给 block 所有线程。
__device__ __forceinline__ float seg_sum(const float* __restrict__ in,
                                         long long b, long long e) {
  float s = 0.f;
  for (long long i = b + threadIdx.x; i < e; i += blockDim.x) s += in[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ float sm[T / 32];
  __shared__ float bres;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) sm[warp] = s;
  __syncthreads();
  if (warp == 0) {
    s = (threadIdx.x < T / 32) ? sm[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = T / 64; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (threadIdx.x == 0) bres = s;
  }
  __syncthreads();
  return bres;
}

// ---------------- 去中心化：设备侧原子 ready-queue（MPK §5.2 的 ring buffer 微缩版） ----------
__device__ __forceinline__ void q_enqueue(int* __restrict__ q_buf,
                                          int* __restrict__ q_ready,
                                          int* __restrict__ q_tail, int task) {
  int idx = atomicAdd(q_tail, 1);
  q_buf[idx] = task;
  __threadfence();               // 值先可见
  atomicExch(&q_ready[idx], 1);  // 再发布 ready 标志
}

// 返回 true 并写 *out；队列空且全部 task 已完成时返回 false。
__device__ __forceinline__ bool q_dequeue(const int* __restrict__ q_buf,
                                          const int* __restrict__ q_ready,
                                          int* __restrict__ q_head,
                                          const int* __restrict__ q_tail,
                                          const int* __restrict__ done_count,
                                          int total, int* out) {
  long long spins = 0;
  while (true) {
    int h = *((volatile int*)q_head);
    int t = *((volatile int*)q_tail);
    if (h < t) {
      if (atomicCAS(q_head, h, h + 1) == h) {
        while (atomicAdd((int*)&q_ready[h], 0) == 0) __nanosleep(64);
        __threadfence();
        *out = q_buf[h];
        return true;
      }
      continue;
    }
    if (atomicAdd((int*)done_count, 0) >= total) return false;
    if (++spins > 200000000LL) { *out = -2; return false; }  // watchdog（不应触发）
    __nanosleep(128);
  }
}

// decen kernel：C*L 个 task，task id = c*L + l。初始只入队 l==0。
__global__ void decen_kernel(const float* __restrict__ in, float* __restrict__ out_sum,
                             int* __restrict__ dep_count, int* __restrict__ q_buf,
                             int* __restrict__ q_ready, int* __restrict__ q_head,
                             int* __restrict__ q_tail, int* __restrict__ done_count,
                             float* __restrict__ chain_sum,
                             const long long* __restrict__ offset,
                             const int* __restrict__ seg, int C, int L) {
  const int total = C * L;
  __shared__ int stask;
  while (true) {
    if (threadIdx.x == 0) {
      int t;
      if (q_dequeue(q_buf, q_ready, q_head, q_tail, done_count, total, &t)) stask = t;
      else stask = -1;
    }
    __syncthreads();
    if (stask < 0) break;
    int task = stask, c = task / L, l = task % L;
    long long b = offset[c] + (long long)l * seg[c];
    float part = seg_sum(in, b, b + seg[c]);
    __syncthreads();  // 等 seg_sum 的 shared 用完
    if (threadIdx.x == 0) {
      float base = (l == 0) ? 0.f : chain_sum[c];
      float sv = base + part;
      chain_sum[c] = sv;  // 同链同刻只有一个 task 活跃，无竞争
      __threadfence();
      if (l == L - 1) out_sum[c] = sv;
      int succ = (l + 1 < L) ? (c * L + (l + 1)) : -1;
      if (succ >= 0 && atomicSub(&dep_count[succ], 1) == 1) q_enqueue(q_buf, q_ready, q_tail, succ);
      atomicAdd(done_count, 1);
    }
    __syncthreads();
  }
}

// ---------------- 集中式 barrier：单 cooperative persistent kernel，逐层 grid.sync() ----------
__global__ void barrier_kernel(const float* __restrict__ in, float* __restrict__ out_sum,
                               float* __restrict__ chain_sum,
                               const long long* __restrict__ offset,
                               const int* __restrict__ seg, int C, int L) {
  for (int l = 0; l < L; ++l) {
    for (int c = blockIdx.x; c < C; c += gridDim.x) {
      long long b = offset[c] + (long long)l * seg[c];
      float part = seg_sum(in, b, b + seg[c]);
      if (threadIdx.x == 0) {
        float sv = ((l == 0) ? 0.f : chain_sum[c]) + part;
        chain_sum[c] = sv;
        if (l == L - 1) out_sum[c] = sv;
      }
    }
    cooperative_groups::this_grid().sync();
  }
}

// ---------------- eager：每层一次 kernel launch（kernel-per-operator-ish） ----------
__global__ void eager_kernel(const float* __restrict__ in, float* __restrict__ out_sum,
                             float* __restrict__ chain_sum,
                             const long long* __restrict__ offset,
                             const int* __restrict__ seg, int C, int L, int l) {
  for (int c = blockIdx.x; c < C; c += gridDim.x) {
    long long b = offset[c] + (long long)l * seg[c];
    float part = seg_sum(in, b, b + seg[c]);
    if (threadIdx.x == 0) {
      float sv = ((l == 0) ? 0.f : chain_sum[c]) + part;
      chain_sum[c] = sv;
      if (l == L - 1) out_sum[c] = sv;
    }
  }
}

// 串行参考：逐链逐层累加（同序，应与三者逐位一致）。
static void reference(const float* h_in, const long long* offset, const int* seg,
                      int C, int L, float* ref) {
  for (int c = 0; c < C; ++c) {
    float s = 0.f;
    for (int l = 0; l < L; ++l) {
      long long b = offset[c] + (long long)l * seg[c];
      for (int i = 0; i < seg[c]; ++i) s += h_in[b + i];
    }
    ref[c] = s;
  }
}

struct Case {
  int C, L, seg;
  const char* name;
};

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 50;
  setvbuf(stdout, NULL, _IONBF, 0);
  CUDA_CHECK(cudaSetDevice(0));

  int dev = 0, numSM = 0;
  cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
  int coop = 0;
  cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
  const int G = (argc > 4) ? atoi(argv[4]) : numSM;  // barrier/decen 的常驻 block 数
  printf("== decentralized_sched (目标卡 H100/sm90) ==\n");
  printf("iters=%d  设备: %d SM  cooperativeLaunch=%d\n", iters, numSM, coop);
  printf("T(block)=%d  G(decen/barrier)=%d\n\n", T, G);

  Case cases[] = {
      {64, 4, 1 << 15, "A: C=64 L=4  seg=32768"},
      {64, 16, 1 << 15, "A: C=64 L=16 seg=32768"},
      {64, 64, 1 << 15, "A: C=64 L=64 seg=32768"},
      {16, 32, 1 << 15, "B: C=16 L=32 seg=32768"},
      {256, 32, 1 << 15, "B: C=256 L=32 seg=32768"},
      {528, 16, 1 << 13, "C: C=528 L=16 seg=8192 不均衡(随机 1-8×)"},
  };
  int only = (argc > 2) ? atoi(argv[2]) : -1;
  int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

  for (auto& cs : cases) {
    int C = cs.C, L = cs.L, segb = (argc > 3) ? atoi(argv[3]) : cs.seg;
    if (only >= 0 && only < ncases && (&cs - cases) != only) continue;
    int total = C * L;
    // seg[c] 不均衡：C 组 case 让每链段长按伪随机 (1..8)× 变化（同一链各层等长）。
    bool imbalance = (cs.name[0] == 'C');

    int* seg = (int*)malloc(sizeof(int) * C);
    long long* offset = (long long*)malloc(sizeof(long long) * C);
    long long ntotal = 0;
    for (int c = 0; c < C; ++c) {
      unsigned h = (unsigned)c * 2654435761u;
      seg[c] = imbalance ? segb * (1 + (int)((h >> 5) % 8)) : segb;
      offset[c] = ntotal;
      ntotal += (long long)seg[c] * L;
    }

    float* h_in = (float*)malloc(sizeof(float) * ntotal);
    for (long long i = 0; i < ntotal; ++i) h_in[i] = (float)((i % 1000)) * 1e-3f;
    float* ref = (float*)malloc(sizeof(float) * C);
    reference(h_in, offset, seg, C, L, ref);

    float *d_in, *d_out, *d_csum;
    int *d_dep, *d_qbuf, *d_qready, *d_qhead, *d_qtail, *d_done;
    long long* d_offset;
    int* d_seg;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * ntotal));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * C));
    CUDA_CHECK(cudaMalloc(&d_csum, sizeof(float) * C));
    CUDA_CHECK(cudaMalloc(&d_dep, sizeof(int) * total));
    CUDA_CHECK(cudaMalloc(&d_qbuf, sizeof(int) * total));
    CUDA_CHECK(cudaMalloc(&d_qready, sizeof(int) * total));
    CUDA_CHECK(cudaMalloc(&d_qhead, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_qtail, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_done, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset, sizeof(long long) * C));
    CUDA_CHECK(cudaMalloc(&d_seg, sizeof(int) * C));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * ntotal, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset, offset, sizeof(long long) * C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg, seg, sizeof(int) * C, cudaMemcpyHostToDevice));

    // dep_count：l>0 为 1，l==0 为 0。初始 ready 队列：【只】放 C 个 l==0 的 task（id = c*L）。
    int* h_dep = (int*)malloc(sizeof(int) * total);
    int* h_qready = (int*)calloc(total, sizeof(int));
    int* h_qbuf = (int*)malloc(sizeof(int) * total);
    for (int s = 0; s < total; ++s) h_qbuf[s] = s;
    for (int c = 0; c < C; ++c)
      for (int l = 0; l < L; ++l) h_dep[c * L + l] = (l == 0) ? 0 : 1;
    for (int c = 0; c < C; ++c) { h_qbuf[c] = c * L; h_qready[c] = 1; }

    // 计时辅助。
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    auto reset = [&]() {
      // 重置 decen 状态（每个 iter 一次）。放在事件 record 之前，不计入计时。
      cudaMemcpyAsync(d_dep, h_dep, sizeof(int) * total, cudaMemcpyHostToDevice, 0);
      cudaMemcpyAsync(d_qready, h_qready, sizeof(int) * total, cudaMemcpyHostToDevice, 0);
      cudaMemcpyAsync(d_qbuf, h_qbuf, sizeof(int) * total, cudaMemcpyHostToDevice, 0);
      cudaMemsetAsync(d_qhead, 0, sizeof(int), 0);
      cudaMemsetAsync(d_done, 0, sizeof(int), 0);
      int val = C;
      cudaMemcpyAsync(d_qtail, &val, sizeof(int), cudaMemcpyHostToDevice, 0);
      cudaMemsetAsync(d_csum, 0, sizeof(float) * C, 0);
      cudaMemsetAsync(d_out, 0, sizeof(float) * C, 0);
    };
    auto reset_out = [&]() { cudaMemsetAsync(d_out, 0, sizeof(float) * C, 0);
                             cudaMemsetAsync(d_csum, 0, sizeof(float) * C, 0); };

    // 三种策略均用 max|Δ| 与 fail 校验。
    float t_eager = 0, t_barrier = 0, t_decen = 0;
    int fail_e = 0, fail_b = 0, fail_d = 0;
    float maxd_e = 0, maxd_b = 0, maxd_d = 0;
    float *h_e = nullptr, *h_b = nullptr, *h_d = nullptr;

    // --- eager ---
    {
      reset_out();      CUDA_CHECK(cudaEventRecord(e0, 0));
      for (int it = 0; it < iters; ++it) {
        for (int l = 0; l < L; ++l)
          eager_kernel<<<G, T, 0, 0>>>(d_in, d_out, d_csum, d_offset, d_seg, C, L, l);
        if (it < iters - 1) reset_out();
      }
      CUDA_CHECK(cudaEventRecord(e1, 0));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&t_eager, e0, e1));
      float* h = (float*)malloc(sizeof(float) * C);
      CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(float) * C, cudaMemcpyDeviceToHost));
      for (int c = 0; c < C; ++c) { float d = fabsf(h[c] - ref[c]) / fmaxf(1.f, fabsf(ref[c])); maxd_e = fmaxf(maxd_e, d); if (d > 1e-3f) fail_e++; }
      h_e = h;
    }
    // --- barrier (cooperative) ---
    {
      reset_out();
      void* args[] = {&d_in, &d_out, &d_csum, &d_offset, &d_seg, &C, &L};
      CUDA_CHECK(cudaEventRecord(e0, 0));
      for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)barrier_kernel, dim3(G), dim3(T), args, 0, 0));
        if (it < iters - 1) reset_out();
      }
      CUDA_CHECK(cudaEventRecord(e1, 0));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&t_barrier, e0, e1));
      float* h = (float*)malloc(sizeof(float) * C);
      CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(float) * C, cudaMemcpyDeviceToHost));
      for (int c = 0; c < C; ++c) { float d = fabsf(h[c] - ref[c]) / fmaxf(1.f, fabsf(ref[c])); maxd_b = fmaxf(maxd_b, d); if (d > 1e-3f) fail_b++; }
      h_b = h;
    }
    // --- decen ---
    {
      reset();
      CUDA_CHECK(cudaEventRecord(e0, 0));
      for (int it = 0; it < iters; ++it) {
        decen_kernel<<<G, T, 0, 0>>>(d_in, d_out, d_dep, d_qbuf, d_qready, d_qhead, d_qtail,
                                     d_done, d_csum, d_offset, d_seg, C, L);
        if (it < iters - 1) reset();
      }
      CUDA_CHECK(cudaEventRecord(e1, 0));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&t_decen, e0, e1));
      float* h = (float*)malloc(sizeof(float) * C);
      CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(float) * C, cudaMemcpyDeviceToHost));
      for (int c = 0; c < C; ++c) { float d = fabsf(h[c] - ref[c]) / fmaxf(1.f, fabsf(ref[c])); maxd_d = fmaxf(maxd_d, d); if (d > 1e-3f) fail_d++; }
      h_d = h;
    }
    // 三策略互比（应与 eager 逐位一致）。
    float xd = 0, xb = 0;
    for (int c = 0; c < C; ++c) {
      xd = fmaxf(xd, fabsf(h_d[c] - h_e[c]) / fmaxf(1.f, fabsf(h_e[c])));
      xb = fmaxf(xb, fabsf(h_b[c] - h_e[c]) / fmaxf(1.f, fabsf(h_e[c])));
    }
    free(h_e); free(h_b); free(h_d);

    double per = (double)iters;
    printf("--- %s ---\n", cs.name);
    printf("  C=%3d L=%3d tasks=%5d  数据=%.1f MB\n", C, L, total, ntotal * 4.0 / 1e6);
    printf("  eager  : %9.3f ms  (=%7.3f us/case, %d launch/case)\n", t_eager / per, t_eager / per * 1000, L);
    printf("  barrier: %9.3f ms  (%6.3f×, %d grid.sync/case)\n", t_barrier / per, t_eager / t_barrier, L);
    printf("  decen  : %9.3f ms  (%6.3f×)\n", t_decen / per, t_eager / t_decen);
    printf("  正确性 vs 串行参考 max rel|Δ|: eager=%.2e fail=%d | barrier=%.2e fail=%d | decen=%.2e fail=%d\n",
           maxd_e, fail_e, maxd_b, fail_b, maxd_d, fail_d);
    printf("  一致性 vs eager max rel|Δ|: barrier=%.2e | decen=%.2e\n\n", xb, xd);
    fflush(stdout);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_csum); cudaFree(d_dep); cudaFree(d_qbuf);
    cudaFree(d_qready); cudaFree(d_qhead); cudaFree(d_qtail); cudaFree(d_done);
    cudaFree(d_offset); cudaFree(d_seg);
    free(seg); free(offset); free(h_in); free(ref); free(h_dep); free(h_qready); free(h_qbuf);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("done.\n");
  return 0;
}
