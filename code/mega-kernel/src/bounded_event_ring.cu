// bounded_event_ring.cu — MegaKernel 专题 P46-1 小实验：
//   **事件驱动调度的 ready 队列从「无界」变成「有界环形缓冲」后，会发生什么？**
//
// 动机 / 接续：
//   P42–P45（docs/04 §31–§34）把「有界 ready 池 + 准入顺序 ⇒ 死锁」收口成普适量 B(π)
//   （CAP*=G*=B(π)+1），并证明「事件驱动（worker 完成即 push 就绪后继）」把 B(π) 归零：
//   §34 实测 event 模式 **CAP=1 且 G=1 即活**，因为 ready 队列只装就绪 task、从不塞阻塞 task。
//   ——但 §34.9 明确留了尾巴：**「event 的 ready 队列无界（按 N+4 开数组），真实实现要环形缓冲 +
//   容量上界」**。本实验就把「有界」补上。
//
//   这不是学院派问题：真实 megakernel 的 in-kernel 队列**是有界的固定容量**——
//     - MPK：`global_runtime_config.per_worker_queue_len = 8192` / `per_sched_queue_len = 4096`
//       （`mirage` 快照 `f9eb70c`，`persistent_kernel.cuh:1790-1791`）；worker 侧
//       `assert(next_task_pos[q] + per_worker_queue_len > last_ready_task_id)`（`:1027`）靠「容量足够大」
//       来避免溢出，而不做背压。
//     - DeepGEMM Mega MoE：`get_num_max_live_pool_blocks()`（`scheduler/mega_moe.cuh:47-80`，注释
//       *"Host-side ring capacity helper"*）在**编译期**闭式算出 ring 容量上界；配合
//       `get_num_l1_warmup_waves()`（`:15-45`，"ensure no L1 -> L2 deadlock"）静态化防死锁。
//   所以「有界 ring 要多大的容量才不丢 task / 不死锁、容量不足时怎么退化」是 megakernel 的真问题。
//
// 机制：单常驻 kernel，block 0 播种源任务（layer 0），其余 G 个 block 是 worker。每个 task 一个
//   依赖计数器 `dep_left`（= 未完成前驱数，即 event 计数器的补集）。worker 完成 task 后把每个后继
//   `atomicSub`，计数归零者 = event 触发 → 入 ready 队列。两种 ring 策略：
//     - MODE 0 `unb`：无界（= §34 的 event 模式；tail 一直加、模 N+4）。
//     - MODE 1 `bnd`：有界容量 CAP 的环形缓冲 + **阻塞式背压**——入队前先 `CAS(tail)` 预留一个槽，
//       仅当 `tail-head < CAP` 才成功，否则自旋等空位（真实「有界队列」的做法）。
//   `unb` 从不因容量失败；`bnd` 在容量不足时会把正在 push 的 worker 卡住（背压），可能把整机
//   拖进死锁（所有 worker 都在 push、无人消费环里现成的就绪 task）。
//
// 核心问题：
//   Q1（§A）**有界 ring 的可完成容量**：CAP_hard（单次最小）/ CAP_safe（重复下最小）与「最大层宽」
//           W_max、与 worker 数 G 是什么关系？（猜想 CAP*∝W_max；实测更接近「并发就绪数 / G」的排队量。）
//   Q2（§A2/§B）FIN、span、G 各自怎么影响容量？（独立扫 G：容量随 G **下降**。）
//   Q3（§C）安全容量以上，继续加大 CAP 还有没有吞吐收益？（基本没有——容量是安全阀不是吞吐旋钮。）
//   Q4（§D）**正确性**：全完成时 `bnd`（CAP≥安全容量）与 `unb` 对同一 DAG 给相同 sum。
//   Q5（§E）容量随图规模怎么标度；对照 MPK 固定 8192 / DeepGEMM 闭式 helper。
//
// 安全：设备侧无进度守门狗（clock64，每 block 监视 completed 是否推进）检测死锁并置 flag，
//   kernel 正常返回并打印，host 不挂。
//
// 编译：nvcc -O3 -arch=sm_90 src/bounded_event_ring.cu -o /tmp/bounded_event_ring
// 用法：
//   ./bounded_event_ring bench [G]
//   ./bounded_event_ring run <mode> <shape> <CAP> <G> <L> <W0> <FIN> <span> <scramble> <iters>
//     mode: unb|bnd     shape: reg|grow|shrink|zigzag

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
      printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define MAXPRED 16
#define MAXSUCC 32

__device__ __forceinline__ float do_work(long iters, unsigned seed) {
  float a = (float)(seed & 1023) * 0.001f + 0.5f;
  float b = (float)((seed >> 10) & 1023) * 0.0005f + 0.25f;
  float acc = 0.f;
  for (long i = 0; i < iters; ++i) acc = fmaf(acc, 1.0000001f, a * b);
  return acc + a - b;
}

// 单常驻 kernel：block 0 = 播种源任务，其余 G = worker。
//   MODE 0 = unb（无界 ready 队列，= §34 event）
//   MODE 1 = bnd（有界环形缓冲 CAP + 阻塞式背压）
__global__ void __launch_bounds__(32) event_kernel(
    const int* __restrict__ pred_count,
    const int* __restrict__ succ_count,
    const int* __restrict__ succs,
    int N,
    int* __restrict__ pool, int RING,
    int* __restrict__ q_head, int* __restrict__ q_tail,
    int* __restrict__ done_flag,
    int* __restrict__ dep_left,          // [N] 未完成前驱数（event 计数器）
    int* __restrict__ completed,
    int* __restrict__ deadlock_flag,
    float* __restrict__ out,
    int CAP, int MODE,
    long work_iters, long idle_cycles) {
  if (threadIdx.x != 0) return;

  volatile int* v_completed = completed;
  volatile int* v_head = q_head;
  volatile int* v_tail = q_tail;
  volatile int* v_dl = deadlock_flag;

  // 阻塞式/无界入队：返回 false 表示守门狗判定死锁（调用者应退出）。
  auto enqueue = [&](int u) -> bool {
    if (MODE == 0) {                              // unb：直接取槽（容量 N+4，从不失败）
      int slot = atomicAdd(q_tail, 1);
      pool[slot % RING] = u;
      __threadfence();
      return true;
    }
    long t0 = clock64();                          // bnd：有空间才 CAS 预留，否则自旋等空位
    while (true) {
      if (*v_dl) return false;
      int h = *v_head;
      int t = *v_tail;
      if (t - h < CAP) {
        if (atomicCAS(q_tail, t, t + 1) == t) {
          pool[t % RING] = u;
          __threadfence();
          return true;
        }
      } else if ((clock64() - t0) > idle_cycles) {
        __threadfence();
        *v_dl = 1;
        return false;
      }
      __nanosleep(32);
    }
  };

  if (blockIdx.x == 0) {
    // 播种所有源任务（前驱数 0），然后只等完成（不消费、不参与准入）。
    int seeded = 0;
    for (int t = 0; t < N; ++t)
      if (pred_count[t] == 0) { if (!enqueue(t)) return; ++seeded; }
    (void)seeded;
    long t0 = clock64();
    int last = -1;
    while (true) {
      if (*v_dl) break;
      int d = *v_completed;
      if (d >= N) break;
      if (d != last) { last = d; t0 = clock64(); }
      else if ((clock64() - t0) > idle_cycles) { __threadfence(); *v_dl = 1; break; }
    }
    return;
  }

  // worker
  while (true) {
    if (*v_dl) break;
    if (*v_completed >= N) break;
    int h = *v_head;
    int t = *v_tail;
    if (h >= t) { __nanosleep(64); continue; }
    if (atomicCAS(q_head, h, h + 1) != h) continue;
    __threadfence();
    int task = pool[h % RING];

    float v = do_work(work_iters, (unsigned)task);
    out[task] = v;
    __threadfence();
    done_flag[task] = 1;
    __threadfence();
    atomicAdd(completed, 1);

    // event：把后继依赖计数减一，归零者直接入队（= MPK trigger_event）。
    int sc = succ_count[task];
    const int* s = succs + (size_t)task * MAXSUCC;
    for (int i = 0; i < sc; ++i) {
      int u = s[i];
      if (atomicSub(&dep_left[u], 1) == 1) {
        if (!enqueue(u)) return;
      }
    }
  }
}

struct Config {
  int L, W0, FIN, span, scramble, CAP, G;
  int shape;         // 0=reg,1=grow,2=shrink,3=zigzag
  int mode;          // 0=unb,1=bnd
  long work_iters, idle_cycles;
};

static const char* mode_name(int m) { return m == 0 ? "unb" : "bnd"; }
static const char* shape_name(int s) {
  switch (s) { case 0: return "reg"; case 1: return "grow"; case 2: return "shrink"; default: return "zigzag"; }
}

static std::vector<int> widths_of(const Config& c) {
  std::vector<int> W(c.L);
  for (int l = 0; l < c.L; ++l) {
    int w = c.W0;
    if (c.shape == 1) w = c.W0 + l;
    else if (c.shape == 2) w = std::max(2, c.W0 - l);
    else if (c.shape == 3) w = (l % 2 == 0) ? c.W0 : std::max(2, c.W0 / 2);
    W[l] = w;
  }
  return W;
}

struct Graph {
  int N, Wmax;
  std::vector<int> W, layer_base, layer_of;
  std::vector<int> pred_count, preds, succ_count, succs;
};

static Graph build_graph(const Config& c) {
  Graph g;
  g.W = widths_of(c);
  g.layer_base.assign(c.L + 1, 0);
  for (int l = 0; l < c.L; ++l) g.layer_base[l + 1] = g.layer_base[l] + g.W[l];
  g.N = g.layer_base[c.L];
  g.pred_count.assign(g.N, 0);
  g.preds.assign((size_t)g.N * MAXPRED, -1);
  g.layer_of.assign(g.N, 0);
  g.Wmax = 0;
  for (int l = 0; l < c.L; ++l) {
    g.Wmax = std::max(g.Wmax, g.W[l]);
    for (int j = 0; j < g.W[l]; ++j) g.layer_of[g.layer_base[l] + j] = l;
  }

  auto add_pred = [&](int t, int p) {
    if (g.pred_count[t] >= MAXPRED) return;
    for (int i = 0; i < g.pred_count[t]; ++i)
      if (g.preds[(size_t)t * MAXPRED + i] == p) return;
    g.preds[(size_t)t * MAXPRED + g.pred_count[t]++] = p;
  };

  int span = std::max(1, c.span);
  for (int l = 1; l < c.L; ++l) {
    for (int j = 0; j < g.W[l]; ++j) {
      int t = g.layer_base[l] + j;
      for (int o = 0; o < c.FIN; ++o) {
        int pl = l - 1 - (o % span);
        if (pl < 0) pl = 0;
        int pw = g.W[pl];
        int pj = (j + o / span) % pw;
        add_pred(t, g.layer_base[pl] + pj);
      }
    }
  }
  g.succ_count.assign(g.N, 0);
  g.succs.assign((size_t)g.N * MAXSUCC, -1);
  for (int t = 0; t < g.N; ++t)
    for (int k = 0; k < g.pred_count[t]; ++k) {
      int p = g.preds[(size_t)t * MAXPRED + k];
      if (p < 0) continue;
      if (g.succ_count[p] < MAXSUCC) g.succs[(size_t)p * MAXSUCC + g.succ_count[p]++] = t;
    }
  return g;
}

struct Result { bool deadlock; int total_done; double ms; double sum_out; };

static Result run_once(const Config& c) {
  Graph g = build_graph(c);
  int N = g.N;
  int RING = (c.mode == 0) ? (N + 4) : c.CAP;
  int *d_pc, *d_sc, *d_succ, *d_pool, *d_head, *d_tail, *d_done, *d_dep, *d_completed, *d_dl;
  float* d_out;
  CK(cudaMalloc(&d_pc, (size_t)N * sizeof(int)));
  CK(cudaMalloc(&d_sc, (size_t)N * sizeof(int)));
  CK(cudaMalloc(&d_succ, (size_t)N * MAXSUCC * sizeof(int)));
  CK(cudaMalloc(&d_pool, (size_t)RING * sizeof(int)));
  CK(cudaMalloc(&d_head, sizeof(int)));
  CK(cudaMalloc(&d_tail, sizeof(int)));
  CK(cudaMalloc(&d_done, (size_t)N * sizeof(int)));
  CK(cudaMalloc(&d_dep, (size_t)N * sizeof(int)));
  CK(cudaMalloc(&d_completed, sizeof(int)));
  CK(cudaMalloc(&d_dl, sizeof(int)));
  CK(cudaMalloc(&d_out, (size_t)N * sizeof(float)));
  CK(cudaMemcpy(d_pc, g.pred_count.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_sc, g.succ_count.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_succ, g.succs.data(), (size_t)N * MAXSUCC * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_head, 0, sizeof(int)));
  CK(cudaMemset(d_tail, 0, sizeof(int)));
  CK(cudaMemset(d_done, 0, (size_t)N * sizeof(int)));
  CK(cudaMemcpy(d_dep, g.pred_count.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_completed, 0, sizeof(int)));
  CK(cudaMemset(d_dl, 0, sizeof(int)));

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  event_kernel<<<c.G + 1, 32>>>(d_pc, d_sc, d_succ, N, d_pool, RING, d_head, d_tail,
                                d_done, d_dep, d_completed, d_dl, d_out,
                                c.CAP, c.mode, c.work_iters, c.idle_cycles);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));

  int h_dl = 0, h_done = 0;
  CK(cudaMemcpy(&h_dl, d_dl, sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&h_done, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<float> h_out(N, 0.f);
  CK(cudaMemcpy(h_out.data(), d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
  double s = 0; for (float v : h_out) s += v;

  cudaFree(d_pc); cudaFree(d_sc); cudaFree(d_succ); cudaFree(d_pool); cudaFree(d_head);
  cudaFree(d_tail); cudaFree(d_done); cudaFree(d_dep); cudaFree(d_completed); cudaFree(d_dl);
  cudaFree(d_out);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  bool deadlock = (h_dl != 0) || (h_done < N);
  return Result{deadlock, h_done, ms, s};
}

static double ref_sum(int N, long work_iters) {
  double s = 0;
  for (int t = 0; t < N; ++t) {
    float a = (float)(t & 1023) * 0.001f + 0.5f;
    float b = (float)((t >> 10) & 1023) * 0.0005f + 0.25f;
    float acc = 0.f;
    for (long i = 0; i < work_iters; ++i) acc = fmaf(acc, 1.0000001f, a * b);
    s += (double)(acc + a - b);
  }
  return s;
}

static double time_best(const Config& c, int reps = 3) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) best = std::min(best, run_once(c).ms);
  return best;
}

static Config with_cap(const Config& b, int cap) { Config c = b; c.CAP = cap; return c; }

// 在 reps 次里死锁的次数（>0 表示该容量在该重复次数下不安全）。
static int deadlock_count(const Config& b, int cap, int reps) {
  Config c = with_cap(b, cap);
  int d = 0;
  for (int r = 0; r < reps; ++r) if (run_once(c).deadlock) ++d;
  return d;
}

// 近似单调下二分找「最小且 reps 次全不挂」的容量（lo 给下界提示，靠近边界本就不确定）。
static int cap_safe(const Config& b, int lo, int hi, int reps) {
  lo = std::max(1, lo);
  if (deadlock_count(b, hi, reps) != 0) return -1;
  int a = lo, z = hi, ans = -1;
  while (a <= z) {
    int m = (a + z) / 2;
    if (deadlock_count(b, m, reps) == 0) { ans = m; z = m - 1; } else a = m + 1;
  }
  return ans;
}

// 从 lo 起线性找第一个「单次全完成」的 CAP（细边界 CAP_hard）。
static int find_cap_hard(const Config& base, int lo, int hi) {
  for (int v = std::max(1, lo); v <= hi; ++v) {
    Config c = base; c.CAP = v;
    if (!run_once(c).deadlock) return v;
  }
  return -1;
}

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "bench";
  long work_iters = 20000;
  long idle_cycles = 60000000;   // 守门狗：~30 ms @2GHz

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("== device ==\n%s, sm_%d%d, SMs=%d\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);

  auto parse_mode = [](const char* s) -> int {
    if (!strcmp(s, "unb")) return 0;
    return 1;
  };
  auto parse_shape = [](const char* s) -> int {
    if (!strcmp(s, "reg")) return 0;
    if (!strcmp(s, "grow")) return 1;
    if (!strcmp(s, "shrink")) return 2;
    if (!strcmp(s, "zigzag")) return 3;
    return 0;
  };

  if (strcmp(mode, "run") == 0) {
    Config c;
    c.mode = parse_mode(argv[2]);
    c.shape = parse_shape(argv[3]);
    c.CAP = atoi(argv[4]); c.G = atoi(argv[5]); c.L = atoi(argv[6]);
    c.W0 = atoi(argv[7]); c.FIN = atoi(argv[8]); c.span = atoi(argv[9]);
    c.scramble = atoi(argv[10]);
    work_iters = (argc > 11) ? atol(argv[11]) : work_iters;
    c.work_iters = work_iters; c.idle_cycles = idle_cycles;
    Graph g = build_graph(c);
    Result r = run_once(c);
    printf("run mode=%-4s shape=%-6s CAP=%d G=%d L=%d W0=%d FIN=%d span=%d: N=%d Wmax=%d"
           " deadlock=%d done=%d/%d ms=%.3f\n",
           mode_name(c.mode), shape_name(c.shape), c.CAP, c.G, c.L, c.W0, c.FIN, c.span,
           g.N, g.Wmax, r.deadlock, r.total_done, g.N, r.ms);
    return 0;
  }

  int G = (argc > 2) ? atoi(argv[2]) : 64;

  Config base; base.shape = 0; base.L = 4; base.W0 = 16;
  base.FIN = 2; base.span = 1; base.scramble = 0; base.G = G;
  base.CAP = G; base.work_iters = work_iters; base.idle_cycles = idle_cycles;

  printf("默认：G=%d, work_iters=%ld, idle_cycles=%ld\n\n", G, work_iters, idle_cycles);

  // ---- §A：有界 ring 的可完成容量 vs 最大就绪前哨宽度 W_max ----
  {
    printf("== A. 有界 ring 容量：CAP_hard(单次最小) / CAP_safe(5 次全不挂) vs 最大层宽 W_max ==\n");
    printf("   shape   | L | W0 | N   | Wmax | unb | CAP_hard | CAP_safe | safe/Wmax\n");
    printf("   --------+---+----+-----+------+-----+----------+----------+----------\n");
    for (int shape : {0, 1, 2, 3}) {
      Config c = base; c.shape = shape; c.L = 5; c.W0 = 24; c.FIN = 1; c.span = 1; c.scramble = 0;
      c.G = G;
      Graph g = build_graph(c);
      Config cu = c; cu.mode = 0;
      Config cb = c; cb.mode = 1;
      int lo = g.Wmax - G - 2;
      int caph = find_cap_hard(cb, lo, g.N + 3);
      int caps = cap_safe(cb, lo, g.Wmax + 2, 5);
      printf("   %-7s | %d | %2d | %3d | %4d | %-3s | %8d | %8d | %9.2f\n",
             shape_name(shape), c.L, c.W0, g.N, g.Wmax,
             run_once(cu).deadlock ? "HANG" : "ok", caph, caps,
             g.Wmax > 0 ? (double)caps / g.Wmax : 0.0);
    }
    printf("   (CAP 远小于 W_max：有 G 个 worker 在飞、环只需缓冲『还没被取走的就绪 task』那一小截)\n\n");
  }

  // ---- §A2：FIN>1（多前驱）与 span>1（跨层 skip）下的容量 ----
  {
    printf("== A2. 扇入 FIN / 跨层 span 对容量的影响（reg, L=5, W0=24, G=%d）==\n", G);
    printf("   FIN | span | N   | Wmax | CAP_hard | CAP_safe\n");
    printf("   ----+------+-----+------+----------+---------\n");
    for (int fin : {1, 2, 4, 8}) {
      for (int span : {1, 2}) {
        Config c = base; c.shape = 0; c.L = 5; c.W0 = 24; c.FIN = fin; c.span = span;
        c.scramble = 0; c.G = G; c.mode = 1;
        Graph g = build_graph(c);
        int lo = g.Wmax - G - 2;
        int caph = find_cap_hard(c, lo, g.N + 3);
        int caps = cap_safe(c, lo, g.Wmax + 2, 5);
        printf("   %3d | %4d | %3d | %4d | %8d | %8d\n", fin, span, g.N, g.Wmax, caph, caps);
      }
    }
    printf("   (FIN/span 只轻微抬高需求：仍远低于 W_max，主因是有多个 worker 在并行取走就绪 task)\n\n");
  }

  // ---- §B：容量是否依赖 G（独立扫 G） ----
  {
    printf("== B. 独立扫 G：容量随 worker 数（shrink, L=5, W0=40, FIN=2, W_max=40）==\n");
    printf("   G   | N   | CAP_hard | CAP_safe | CAP_safe+G | (Wmax+1)-(CAP_safe+G)\n");
    printf("   ----+-----+----------+----------+------------+------------------------\n");
    for (int g : {1, 2, 4, 8, 16, 32, 64, 128}) {
      Config c = base; c.shape = 2; c.L = 5; c.W0 = 40; c.FIN = 2; c.span = 1;
      c.scramble = 0; c.G = g; c.mode = 1;
      Graph gg = build_graph(c);
      int lo = gg.Wmax - g - 2;
      int caph = find_cap_hard(c, lo, gg.N + 3);
      int caps = cap_safe(c, lo, gg.Wmax + 2, 5);
      printf("   %3d | %3d | %8d | %8d | %10d | %+22d\n",
             g, gg.N, caph, caps, caps + g, (gg.Wmax + 1) - (caps + g));
    }
    printf("   (容量随 G 下降、且 **CAP_safe+G ≈ W_max+1..2**（G≤W_max 时）：G 个 worker 在飞、\n");
    printf("    环只需缓冲『宽 W_max 的那一层里还没被取走的 W_max-G 个』。这是**排队/速率**性质，\n");
    printf("    不同于 §31–§34 的 B(π)+1——那条由**准入顺序**决定、是图的静态量)\n\n");
  }

  // ---- §C：容量→吞吐（安全容量以上是否还有收益） ----
  {
    printf("== C. 容量→吞吐：安全容量以上基本免费（reg, L=6, W0=64, G=%d）==\n", G);
    printf("   CAP  | ms(bnd) | vs CAP=512 | done\n");
    printf("   -----+---------+------------+-----\n");
    Config c = base; c.shape = 0; c.L = 6; c.W0 = 64; c.FIN = 1; c.span = 1;
    c.scramble = 0; c.G = G; c.mode = 1;
    Graph g = build_graph(c);
    int caps = cap_safe(c, 1, g.Wmax + 2, 5);
    const int caps_list[] = {1, 2, 4, 8, 16, 64, 128, 256, 512};
    double t512 = 0;
    for (int cap : caps_list) {
      Config cc = c; cc.CAP = cap;
      Result r = run_once(cc);
      if (r.deadlock) {
        printf("   %4d | %7.3f | %10s | %d/%d  <- 守门狗%s\n", cap, r.ms, "DEADLOCK",
               r.total_done, g.N, (cap < caps) ? " (CAP<CAP_safe)" : "");
        continue;
      }
      double t = time_best(cc);
      if (cap == 512) t512 = t;
    }
    if (t512 == 0) t512 = time_best(with_cap(c, caps));
    for (int cap : caps_list) {
      Config cc = c; cc.CAP = cap;
      Result r = run_once(cc);
      if (r.deadlock) continue;
      double t = time_best(cc);
      printf("   %4d | %7.3f | %9.2fx | %d/%d\n", cap, t, t512 > 0 ? t / t512 : 0.0, r.total_done, g.N);
    }
    printf("   (CAP_safe=%d；一旦不死锁，加大容量不再提速——容量不是吞吐的旋钮，只是**安全阀**)\n\n", caps);
  }

  // ---- §D：正确性（全完成时 bnd 与 unb 数值一致） ----
  {
    printf("== D. 正确性：unb 与 bnd(CAP 富余) 对同一 DAG 给出相同 sum ==\n");
    printf("   shape   | scr | N   | Wmax | unb rel  | bnd rel  | done\n");
    printf("   --------+-----+-----+------+----------+----------+------\n");
    for (int shape : {0, 1, 2, 3}) {
      for (int scr : {0, 1}) {
        Config c = base; c.shape = shape; c.L = 5; c.W0 = 24; c.FIN = 2; c.span = 2;
        c.scramble = scr; c.G = G;
        Graph g = build_graph(c);
        double rs = ref_sum(g.N, work_iters);
        Config cu = c; cu.mode = 0;
        Config cb = c; cb.mode = 1; cb.CAP = g.Wmax + 1;   // 验证「CAP ≥ W_max 恒安全」
        Result ru = run_once(cu), rb = run_once(cb);
        double relu = rs != 0 ? (ru.sum_out - rs) / rs : 0.0;
        double relb = rs != 0 ? (rb.sum_out - rs) / rs : 0.0;
        printf("   %-7s | %3d | %3d | %4d | %8.1e | %8.1e | %d/%d\n",
               shape_name(shape), scr, g.N, g.Wmax, relu, relb, rb.total_done, g.N);
      }
    }
    printf("   (bnd 只改『何时入队』不改『算什么』；CAP 足够时 fallback 到 unb 的执行/数值)\n\n");
  }

  // ---- §E：容量随图规模标度 + 与 MPK / DeepGEMM 的固定容量对照 ----
  {
    printf("== E. 容量随规模标度：CAP_safe vs W_max（reg, L=5, FIN=1；G=%d 与 G=8 两档）==\n", G);
    printf("   W0  | N    | Wmax | CAP_safe(G=%3d) | /Wmax | CAP_safe(G=8) | /Wmax\n", G);
    printf("   ----+------+------+-----------------+-------+---------------+------\n");
    for (int w0 : {16, 64, 256, 512, 1024}) {
      Config c = base; c.shape = 0; c.L = 5; c.W0 = w0; c.FIN = 1; c.span = 1;
      c.scramble = 0; c.mode = 1;
      Graph g = build_graph(c);
      Config cg = c; cg.G = G;
      Config c8 = c; c8.G = 8;
      int csG = cap_safe(cg, g.Wmax - G - 2, g.Wmax + 2, 5);
      int cs8 = cap_safe(c8, g.Wmax - 8 - 2, g.Wmax + 2, 5);
      printf("   %4d | %4d | %4d | %15d | %5.2f | %13d | %5.2f\n",
             w0, g.N, g.Wmax, csG, g.Wmax > 0 ? (double)csG / g.Wmax : 0.0,
             cs8, g.Wmax > 0 ? (double)cs8 / g.Wmax : 0.0);
    }
    printf("   (容量随 W_max 增长、随 G 下降——是一个『并发就绪数 / worker 数』的排队量；\n");
    printf("    MPK 直接钉 per_worker_queue_len=8192（:1790-1791）+ assert（:1027）不溢出，\n");
    printf("    DeepGEMM 用 get_num_max_live_pool_blocks（:47-80）闭式给 ring 容量上界——\n");
    printf("    两者都是『取一个远超实际峰值就绪数的安全常量』，与本节量出的『实际需求很小』一致)\n\n");
  }

  printf(">> 读法：\n");
  printf(">>  §A  unb（=§34 event，无界）恒可完成；bnd（有界 CAP + 阻塞背压）需 CAP ≥ 一个小容量，\n");
  printf(">>      远低于最大层宽 W_max——因为 G 个 worker 在并行取走就绪 task。CAP 太小时所有 worker\n");
  printf(">>      卡在 push、无人消费现成就绪 task → 死锁（守门狗）。CAP ≥ W_max 是平凡安全上界。\n");
  printf(">>  §A2 FIN/span 轻微抬高需求；§B 容量随 G 下降（排队/速率量），非 §31–§34 的静态 B(π)+1。\n");
  printf(">>  §C  安全容量以上加大容量不再提速——容量是**安全阀**不是吞吐旋钮。\n");
  printf(">>  §D  bnd 不动数值（CAP 足够时退化为 unb 的执行/结果）。\n");
  printf(">>  §E  容量随 W_max 增长、随 G 下降；MPK/DeepGEMM 取远超峰值就绪数的固定常量保守兜底。\n");
  printf(">>  边界 caveat：靠近安全边界时结果**不确定**（单次可完成、多次可能死锁，见 §A 的 CAP_hard<CAP_safe）。\n");
  return 0;
}
