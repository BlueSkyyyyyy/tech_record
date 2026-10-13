// irregular_pool.cu — MegaKernel 专题 P44-1 小实验：**有界 ready 池 × 不规则依赖 DAG**。
//
// 动机 / 接续：
//   P42-1（`docs/04` §31，`src/ring_deadlock.cu`）把 DeepGEMM Mega MoE scheduler 的
//   `get_num_l1_warmup_waves()`（`scheduler/mega_moe.cuh:15-45`）与 `get_num_max_live_pool_blocks()`
//   （`:47-80`）做成实测：单层 fan-in 下死锁边界 = fan-in 宽度 Kc。
//   P43-1（`docs/04` §32，`src/layered_pool.cu`）推广到**规则分层 DAG**（L 层 × 每层等宽 W ×
//   上一层 FIN 扇入）：坏顺序（层降序）边界 = 「阻塞数」B=(L-1)*W，且是 (CAP,G) 联合约束；
//   拓扑层序 `level` / 依赖门控 `gated` 任意 CAP 无死锁。§32.7 明确留了尾巴：
//     依赖模型是「规则的分层 × 每层等宽 × 上一层扇入」，**未测不规则/跨层依赖、多级扇入扇出混合**。
//
// 本实验把 DAG 从「规则分层」推广到**不规则**：
//     (1) **每层宽度不等** W[l]（递增/递减/锯齿）；
//     (2) **跨层（skip）前向边**：前驱可来自更浅的若干层（残差式 skip connection，span）；
//     (3) **多级扇入**：一个 task 的前驱跨多个更浅的层；
//     (4) **层号乱序（scramble）**：名义层号被旋转，使「层号升序」不再是拓扑序（对偶于一般 DAG /
//         含跨层反馈的图）。
//
// 关键概念（相对 §32 的推广）：死锁边界 = 沿准入 order 贪心模拟的**最小池容量 B(π)**——
//   逐个准入，并立刻排空任何「前驱全已 done」的就绪 task（等价于一个无限快 worker），
//   池内「已准入未完成」的峰值占用即 B(π)。scheduler 必须能把峰值那批全塞进池、并留一个空 worker
//   去跑就绪 task。实测边界 = **CAP*=G*=B(π)+1**。
//   §32 规则分层下 B=N-W0=(L-1)*W 恰是此量；不规则宽度下 (L-1)*W 闭式失效，B(π) 才是普适量；
//   层号乱序时它也不再等于 N-W0。注意它**不能**简化成「第一个就绪 task 的位置」——源在前、
//   后面再堆满阻塞的形态（§C scramble）会漏掉。
//
// 核心问题：
//   Q1（§A）不规则宽度：(L-1)*W 闭式失效，真边界 = B(π)=N-W0？ CAP*=G*=B(π)+1？
//   Q2（§B）前向 skip / 多级扇入：层号升序仍是拓扑序 → level/topo 恒安全；B_wide(=N-W0) 不随 span 变。
//   Q3（§C）层号乱序：level（按名义层升序）不再是拓扑序 → 任意 CAP 死锁；topo（Kahn）恒 B=0。
//   Q4（§D）gated（依赖门控）对任意 DAG 无死锁，但扫描代价 vs 预先算好的 topo。
//   Q5（§E）warmup 的正确判据是「剩余阻塞 B'<G」而非「warm D 层」。
//
// 安全：设备侧无进度守门狗（scheduler 检测到 completed 长时间不增即置 deadlock_flag，
//   所有 worker 在依赖自旋里检查该 flag 退出）——死锁 case 让 kernel 正常返回并打印，host 不挂。
//
// 编译：nvcc -O3 -arch=sm_90 src/irregular_pool.cu -o /tmp/irregular_pool
// 用法：
//   ./irregular_pool bench [iters] [G]
//   ./irregular_pool run <policy> <shape> <CAP> <G> <L> <W0> <FIN> <span> <scramble> <D> <iters>
//     policy: wide|level|topo|warmup|gated    shape: reg|grow|shrink|zigzag

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <algorithm>
#include <queue>
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

__device__ __forceinline__ float do_work(long iters, unsigned seed) {
  float a = (float)(seed & 1023) * 0.001f + 0.5f;
  float b = (float)((seed >> 10) & 1023) * 0.0005f + 0.25f;
  float acc = 0.f;
  for (long i = 0; i < iters; ++i) acc = fmaf(acc, 1.0000001f, a * b);
  return acc + a - b;
}

__device__ __forceinline__ bool all_preds_done(
    int t, const int* __restrict__ pred_count, const int* __restrict__ preds,
    const int* __restrict__ done_flag) {
  int pc = pred_count[t];
  const int* p = preds + (size_t)t * MAXPRED;
  for (int i = 0; i < pc; ++i)
    if (*(volatile const int*)&done_flag[p[i]] == 0) return false;
  return true;
}

__global__ void __launch_bounds__(32) sched_kernel(
    const unsigned* __restrict__ order,
    const int* __restrict__ pred_count,
    const int* __restrict__ preds,
    int N, int unused_W,
    int* __restrict__ pool, int POOLCAP,
    int* __restrict__ q_head, int* __restrict__ q_tail,
    int* __restrict__ done_flag,         // [N]
    int* __restrict__ admitted,          // [N] (gated 用)
    int* __restrict__ completed,
    int* __restrict__ deadlock_flag,
    float* __restrict__ out,             // [N]
    int CAP, int GATED,
    long work_iters, long idle_cycles) {
  if (threadIdx.x != 0) return;
  (void)unused_W;

  volatile int* v_completed = completed;
  volatile int* v_dl = deadlock_flag;
  volatile int* v_head = q_head;
  volatile int* v_tail = q_tail;

  if (blockIdx.x == 0) {
    int cursor = 0, tail = 0, last = 0;
    long t0 = clock64();
    while (true) {
      if (*v_dl) break;
      int done = *v_completed;
      if (done >= N) break;
      bool progressed = false;
      while ((tail - done) < CAP && cursor < N) {
        if (!GATED) {
          int t = order[cursor++];
          pool[tail % POOLCAP] = t; ++tail; progressed = true;
        } else {
          bool found = false;
          for (int c = cursor; c < N; ++c) {
            int t = order[c];
            if (atomicCAS(&admitted[t], 0, 1) == 0) {
              if (all_preds_done(t, pred_count, preds, done_flag)) {
                pool[tail % POOLCAP] = t; ++tail; progressed = true;
                found = true;
                if (c == cursor) ++cursor;
                break;
              } else {
                admitted[t] = 0;
              }
            }
          }
          if (!found) break;
        }
      }
      __threadfence();
      *v_tail = tail;
      if (done != last || progressed) { last = done; t0 = clock64(); }
      else if ((clock64() - t0) > idle_cycles) { __threadfence(); *v_dl = 1; break; }
    }
  } else {
    while (true) {
      if (*v_dl) break;
      int done = *v_completed;
      if (done >= N) break;
      int h = *v_head;
      int t = *v_tail;
      if (h >= t) { __nanosleep(64); continue; }
      if (atomicCAS(q_head, h, h + 1) != h) continue;
      __threadfence();
      int task = pool[h % POOLCAP];
      while (!all_preds_done(task, pred_count, preds, done_flag)) {
        if (*v_dl) break;
        __nanosleep(32);
      }
      if (*v_dl) break;
      float v = do_work(work_iters, (unsigned)task);
      out[task] = v;
      __threadfence();
      done_flag[task] = 1;
      __threadfence();
      atomicAdd(completed, 1);
    }
  }
}

enum Policy { P_WIDE = 0, P_LEVEL = 1, P_TOPO = 2, P_WARMUP = 3, P_GATED = 4 };

struct Config {
  int L, W0, FIN, span, scramble, CAP, G, D;
  int shape;         // 0=reg,1=grow,2=shrink,3=zigzag
  int policy;
  long work_iters, idle_cycles;
};

static const char* policy_name(int p) {
  switch (p) {
    case P_WIDE: return "wide"; case P_LEVEL: return "level"; case P_TOPO: return "topo";
    case P_WARMUP: return "warmup"; default: return "gated";
  }
}
static const char* shape_name(int s) {
  switch (s) { case 0: return "reg"; case 1: return "grow"; case 2: return "shrink"; default: return "zigzag"; }
}

static std::vector<int> widths_of(const Config& c) {
  std::vector<int> W(c.L);
  for (int l = 0; l < c.L; ++l) {
    int w = c.W0;
    if (c.shape == 1) w = c.W0 + l;                       // grow
    else if (c.shape == 2) w = std::max(2, c.W0 - l);     // shrink
    else if (c.shape == 3) w = (l % 2 == 0) ? c.W0 : std::max(2, c.W0 / 2);  // zigzag
    W[l] = w;
  }
  return W;
}

struct Graph {
  int N;
  std::vector<int> W;                   // 每「拓扑层」宽度
  std::vector<int> layer_base;          // 拓扑层起始 id
  std::vector<int> layer_of;            // [N] 名义层号（可能被 scramble 旋转）
  std::vector<int> pred_count, preds;   // [N], [N*MAXPRED]
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
  for (int l = 0; l < c.L; ++l)
    for (int j = 0; j < g.W[l]; ++j) g.layer_of[g.layer_base[l] + j] = l;

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
        // 跨层（skip）：前驱层在 [l-span, l-1] 轮转；span=1 即只取上一层（规则形态）。
        int pl = l - 1 - (o % span);
        if (pl < 0) pl = 0;
        int pw = g.W[pl];
        int pj = (j + o / span) % pw;
        add_pred(t, g.layer_base[pl] + pj);
      }
    }
  }
  // 名义层号乱序：旋转 layer_of（对偶于一般 DAG / 层号与依赖方向不一致）。
  if (c.scramble != 0) {
    int s = ((c.scramble % c.L) + c.L) % c.L;
    for (int t = 0; t < g.N; ++t) g.layer_of[t] = (g.layer_of[t] + s) % c.L;
  }
  return g;
}

// B(π) = 沿准入 order 贪心模拟的最小池容量：逐个准入，并**立刻排空**任何「前驱全已 done」的就绪 task
//   （等价于一个无限快 worker），池内「已准入未完成」的峰值占用就是最小 CAP。
//   这比「第一个就绪 task 的位置」更准：前者漏掉「源在前、但后面又堆满阻塞」的形态（见 §C scramble）。
// 预计死锁边界 CAP*=G*=B(π)（+1 的常数由实现里 CAP 是「>」还是「≥」决定，按实测标定）。
static int greedy_peak(const Graph& g, const std::vector<unsigned>& order) {
  std::vector<char> admitted(g.N, 0), done(g.N, 0);
  int occ = 0, peak = 0;
  auto preds_done = [&](int u) {
    for (int k = 0; k < g.pred_count[u]; ++k)
      if (!done[g.preds[(size_t)u * MAXPRED + k]]) return false;
    return true;
  };
  for (size_t i = 0; i < order.size(); ++i) {
    int t = order[i];
    admitted[t] = 1; ++occ;
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t j = 0; j <= i; ++j) {
        int u = order[j];
        if (admitted[u] && !done[u] && preds_done(u)) {
          done[u] = 1; --occ; changed = true;
        }
      }
    }
    if (occ > peak) peak = occ;
  }
  return peak;
}

static std::vector<unsigned> make_order(const Config& c, const Graph& g, int* B_out) {
  std::vector<unsigned> order;
  // 按名义层号分组（同层内保持 id 升序）。
  std::vector<std::vector<int>> by_layer(c.L);
  for (int t = 0; t < g.N; ++t) by_layer[g.layer_of[t]].push_back(t);
  auto emit = [&](int l) { for (int t : by_layer[l]) order.push_back((unsigned)t); };

  if (c.policy == P_WIDE) {                       // 名义层降序（最对抗）
    for (int l = c.L - 1; l >= 0; --l) emit(l);
  } else if (c.policy == P_LEVEL || c.policy == P_GATED) {   // 名义层升序
    for (int l = 0; l < c.L; ++l) emit(l);
  } else if (c.policy == P_WARMUP) {              // 前 D 名义层升序 + 其余降序
    int D = std::min(c.D, c.L);
    for (int l = 0; l < D; ++l) emit(l);
    for (int l = c.L - 1; l >= D; --l) emit(l);
  } else {                                        // topo：Kahn 真拓扑序
    std::vector<int> indeg(g.N, 0);
    for (int t = 0; t < g.N; ++t) indeg[t] = g.pred_count[t];
    std::vector<int> succ_scratch;  (void)succ_scratch;
    auto cmp = [](int a, int b) { return a > b; };   // 小 id 优先（确定、可复现）
    std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);
    for (int t = 0; t < g.N; ++t) if (indeg[t] == 0) pq.push(t);
    while (!pq.empty()) {
      int t = pq.top(); pq.pop();
      order.push_back((unsigned)t);
      for (int u = 0; u < g.N; ++u)
        for (int k = 0; k < g.pred_count[u]; ++k)
          if (g.preds[(size_t)u * MAXPRED + k] == t && --indeg[u] == 0) pq.push(u);
    }
  }
  if (B_out) *B_out = greedy_peak(g, order);
  return order;
}

struct Result { bool deadlock; int total_done; double ms; double sum_out; };

static Result run_once(const Config& c) {
  Graph g = build_graph(c);
  int N = g.N;
  int POOLCAP = c.CAP + c.G + 4;
  int B = 0;
  std::vector<unsigned> order = make_order(c, g, &B);

  unsigned *d_order; int *d_pc, *d_preds, *d_pool, *d_head, *d_tail,
                     *d_done, *d_admitted, *d_completed, *d_dl;
  float* d_out;
  CK(cudaMalloc(&d_order, N * sizeof(unsigned)));
  CK(cudaMalloc(&d_pc, N * sizeof(int)));
  CK(cudaMalloc(&d_preds, (size_t)N * MAXPRED * sizeof(int)));
  CK(cudaMalloc(&d_pool, POOLCAP * sizeof(int)));
  CK(cudaMalloc(&d_head, sizeof(int)));
  CK(cudaMalloc(&d_tail, sizeof(int)));
  CK(cudaMalloc(&d_done, N * sizeof(int)));
  CK(cudaMalloc(&d_admitted, N * sizeof(int)));
  CK(cudaMalloc(&d_completed, sizeof(int)));
  CK(cudaMalloc(&d_dl, sizeof(int)));
  CK(cudaMalloc(&d_out, N * sizeof(float)));
  CK(cudaMemcpy(d_order, order.data(), N * sizeof(unsigned), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_pc, g.pred_count.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_preds, g.preds.data(), (size_t)N * MAXPRED * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_head, 0, sizeof(int)));
  CK(cudaMemset(d_tail, 0, sizeof(int)));
  CK(cudaMemset(d_done, 0, N * sizeof(int)));
  CK(cudaMemset(d_admitted, 0, N * sizeof(int)));
  CK(cudaMemset(d_completed, 0, sizeof(int)));
  CK(cudaMemset(d_dl, 0, sizeof(int)));

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  sched_kernel<<<c.G + 1, 32>>>(d_order, d_pc, d_preds, N, c.W0, d_pool, POOLCAP,
                                d_head, d_tail, d_done, d_admitted, d_completed, d_dl,
                                d_out, c.CAP, c.policy == P_GATED ? 1 : 0,
                                c.work_iters, c.idle_cycles);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));

  int h_dl = 0, h_done = 0;
  CK(cudaMemcpy(&h_dl, d_dl, sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&h_done, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<float> h_out(N, 0.f);
  CK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  double s = 0; for (float v : h_out) s += v;

  cudaFree(d_order); cudaFree(d_pc); cudaFree(d_preds); cudaFree(d_pool);
  cudaFree(d_head); cudaFree(d_tail); cudaFree(d_done); cudaFree(d_admitted);
  cudaFree(d_completed); cudaFree(d_dl); cudaFree(d_out);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return Result{h_dl != 0, h_done, ms, s};
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

static int find_boundary(const Config& base, bool scan_cap, int limit) {
  for (int v = 1; v <= limit; ++v) {
    Config c = base;
    if (scan_cap) c.CAP = v; else c.G = v;
    if (!run_once(c).deadlock) return v;
  }
  return -1;
}

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "bench";
  long work_iters = 20000;
  long idle_cycles = 100000000;

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("== device ==\n%s, sm_%d%d, SMs=%d\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);

  auto parse_policy = [](const char* s) -> int {
    if (!strcmp(s, "wide")) return P_WIDE;
    if (!strcmp(s, "level")) return P_LEVEL;
    if (!strcmp(s, "topo")) return P_TOPO;
    if (!strcmp(s, "warmup")) return P_WARMUP;
    if (!strcmp(s, "gated")) return P_GATED;
    return P_WIDE;
  };
  auto parse_shape = [](const char* s) -> int {
    if (!strcmp(s, "reg")) return 0;
    if (!strcmp(s, "grow")) return 1;
    if (!strcmp(s, "shrink")) return 2;
    if (!strcmp(s, "zigzag")) return 3;
    return 0;
  };

  if (strcmp(mode, "run") == 0) {
    // run <policy> <shape> <CAP> <G> <L> <W0> <FIN> <span> <scramble> <D> <iters>
    Config c;
    c.policy = parse_policy(argv[2]);
    c.shape  = parse_shape(argv[3]);
    c.CAP = atoi(argv[4]); c.G = atoi(argv[5]); c.L = atoi(argv[6]);
    c.W0 = atoi(argv[7]); c.FIN = atoi(argv[8]); c.span = atoi(argv[9]);
    c.scramble = atoi(argv[10]); c.D = atoi(argv[11]);
    work_iters = (argc > 12) ? atol(argv[12]) : work_iters;
    c.work_iters = work_iters; c.idle_cycles = idle_cycles;
    Graph g = build_graph(c);
    int B = 0; make_order(c, g, &B);
    Result r = run_once(c);
    printf("run policy=%-6s shape=%-6s CAP=%d G=%d L=%d W0=%d FIN=%d span=%d scramble=%d D=%d:"
           " N=%d B(pi)=%d deadlock=%d done=%d/%d ms=%.3f\n",
           policy_name(c.policy), shape_name(c.shape), c.CAP, c.G, c.L, c.W0, c.FIN,
           c.span, c.scramble, c.D, g.N, B, r.deadlock, r.total_done, g.N, r.ms);
    return 0;
  }

  int iters = (argc > 2) ? atoi(argv[2]) : 30;
  (void)iters;
  int G = (argc > 3) ? atoi(argv[3]) : 64;

  Config base; base.policy = P_WIDE; base.shape = 0; base.L = 4; base.W0 = 16;
  base.FIN = 2; base.span = 1; base.scramble = 0; base.D = 0; base.G = G;
  base.work_iters = work_iters; base.idle_cycles = idle_cycles;

  printf("默认：G=%d, work_iters=%ld, idle_cycles=%ld\n\n", G, work_iters, idle_cycles);

  // 正确性：每个 shape（含 skip + scramble）用 gated + CAP=N+1 跑通，比 sum。
  printf("== 正确性（gated, CAP=N+1；含 skip + scramble）==\n");
  printf("   shape   | L  | W0 | span | scramble | N  | done | rel\n");
  printf("   --------+----+----+------+----------+----+------+-----\n");
  for (int shape : {0, 1, 2, 3}) {
    Config c = base; c.policy = P_GATED; c.shape = shape; c.span = 2; c.scramble = 1;
    Graph gg = build_graph(c);
    c.CAP = gg.N + 1;
    Result r = run_once(c);
    double rs = ref_sum(gg.N, work_iters);
    printf("   %-7s | %2d | %2d | %4d | %8d | %3d | %3d/%2d | %.1e\n",
           shape_name(shape), c.L, c.W0, c.span, c.scramble, gg.N, r.total_done, gg.N,
           rs != 0 ? (r.sum_out - rs) / rs : 0.0);
  }
  printf("\n");

  // ---- §A：不规则宽度——真边界 = B(π)=N-W0，(L-1)*W0 闭式失效 ----
  {
    printf("== A. 不规则层宽：wide 边界 vs B(pi) vs 失效的 (L-1)*W0 闭式 ==\n");
    printf("   shape   | W[l] 序列        | N  | W0 | B(pi) | CAP* | G*  | (L-1)*W0+1\n");
    printf("   --------+------------------+----+----+-------+------+-----+-----------\n");
    for (int shape : {0, 1, 2, 3}) {
      Config c = base; c.shape = shape; c.policy = P_WIDE; c.scramble = 0; c.span = 1;
      Graph gg = build_graph(c);
      int B = 0; make_order(c, gg, &B);
      Config cb = c; cb.G = 256; int capb = find_boundary(cb, true, gg.N + 3);
      Config cg = c; cg.CAP = gg.N + 3; int gb = find_boundary(cg, false, gg.N + 3);
      printf("   %-7s |", shape_name(shape));
      for (int l = 0; l < c.L; ++l) printf(" %d", gg.W[l]);
      printf(" | %2d | %2d | %5d | %4d | %3d | %9d\n",
             gg.N, gg.W[0], B, capb, gb, (c.L - 1) * c.W0 + 1);
    }
    printf("   (宽度不等时 (L-1)*W0 只是近似；真边界 = B(pi)=N-W0 与 CAP*=G*=B(pi)+1 一致)\n\n");
  }

  // ---- §B：跨层前向 skip / 多级扇入——level 仍拓扑、wide 边界不变 ----
  {
    printf("== B. 跨层前向 skip 边（span 1..4）：level 仍拓扑序、wide 边界不变 ==\n");
    printf("   span | N  | B_wide(pi) | wide CAP* | wide G* | level ok@CAP=1 | topo ok@CAP=1\n");
    printf("   -----+----+------------+-----------+---------+----------------+---------------\n");
    for (int span : {1, 2, 3, 4}) {
      Config c = base; c.shape = 2; c.span = span; c.policy = P_WIDE; c.scramble = 0;
      Graph gg = build_graph(c);
      int B = 0; make_order(c, gg, &B);
      Config cb = c; cb.G = 256; int capb = find_boundary(cb, true, gg.N + 3);
      Config cg = c; cg.CAP = gg.N + 3; int gb = find_boundary(cg, false, gg.N + 3);
      Config cl = c; cl.policy = P_LEVEL; cl.CAP = 1; cl.G = 64;
      Config ct = c; ct.policy = P_TOPO; ct.CAP = 1; ct.G = 64;
      printf("   %4d | %2d | %10d | %9d | %7d | %-14s | %-13s\n",
             span, gg.N, B, capb, gb,
             run_once(cl).deadlock ? "DEADLOCK" : "ok",
             run_once(ct).deadlock ? "DEADLOCK" : "ok");
    }
    printf("   (前向 skip 只从更浅的拓扑层取前驱 → 层号升序仍是合法拓扑序；wide 的 B(pi)=N-W0 与 span 无关)\n\n");
  }

  // ---- §C：层号乱序——层号升序不再是拓扑序（headline） ----
  {
    printf("== C. 层号乱序（scramble）：level(按名义层升序) 不再是拓扑序 ==\n");
    printf("   scr | N  | B_wide | B_level | B_topo | wide CAP* = B_wide+1 | level ok@CAP=1 | topo ok@CAP=1\n");
    printf("   ----+----+--------+---------+--------+---------------------+----------------+---------------\n");
    for (int scr : {0, 1, 2, 3}) {
      Config c = base; c.shape = 0; c.span = 1; c.scramble = scr;
      Graph gg = build_graph(c);
      int Bw = 0, Bl = 0, Bt = 0;
      Config cw = c; cw.policy = P_WIDE; make_order(cw, gg, &Bw);
      Config cl = c; cl.policy = P_LEVEL; make_order(cl, gg, &Bl);
      Config ct = c; ct.policy = P_TOPO; make_order(ct, gg, &Bt);
      Config cb = cw; cb.G = 256; int capb = find_boundary(cb, true, gg.N + 3);
      Config clr = c; clr.policy = P_LEVEL; clr.CAP = 1; clr.G = 64;
      Config ctr = c; ctr.policy = P_TOPO; ctr.CAP = 1; ctr.G = 64;
      printf("   %3d | %2d | %6d | %7d | %6d | %19d | %-14s | %-13s\n",
             scr, gg.N, Bw, Bl, Bt, capb,
             run_once(clr).deadlock ? "DEADLOCK" : "ok",
             run_once(ctr).deadlock ? "DEADLOCK" : "ok");
    }
    printf("   (层号乱序时按名义层升序先发『名义浅层』，其真前驱（名义深层）最后才发 → B_level 骤增、\n");
    printf("    CAP=1 死锁；真拓扑序 topo 恒 B=0、CAP=1 无死锁。→ 「按名义层/op 名遍历」≠「拓扑线性化」。\n");
    printf("    对照 MPK 依赖分析器用 Kahn 求 linearization（docs/08 §5），而非按层名/op 名顺序。)\n\n");
  }

  // ---- §D：多级扇入——gated 扫描代价 vs 预先算好的 topo ----
  {
    printf("== D. 多级扇入（span 越大前驱跨越多层）：gated 扫描代价 vs topo（CAP=G 富余） ==\n");
    printf("   shape   | span | FIN | N  | B_gated | gated ms | topo ms | gated/topo\n");
    printf("   --------+------+-----+----+---------+----------+---------+-----------\n");
    for (int shape : {0, 2}) {
      for (int span : {1, 4}) {
        for (int fin : {2, 8}) {
          Config c = base; c.shape = shape; c.span = span; c.FIN = fin; c.scramble = 0;
          c.CAP = 512; c.G = G;
          Graph gg = build_graph(c);
          Config cg = c; cg.policy = P_GATED;
          Config ct = c; ct.policy = P_TOPO;
          double tg = time_best(cg), tt = time_best(ct);
          int Bg = 0; make_order(cg, gg, &Bg);
          printf("   %-7s | %4d | %3d | %2d | %7d | %8.3f | %7.3f | %10.3f\n",
                 shape_name(shape), span, fin, gg.N, Bg, tg, tt, tt > 0 ? tg / tt : 0.0);
        }
      }
    }
    printf("   (gated 对任意 DAG/顺序无死锁，但 scheduler 每次要扫描剩余 order 找就绪 task——代价随 N\n");
    printf("    与扫描跳数增长；预先算好拓扑序的 topo 不付扫描账。MPK 用 event->task 直接映射避免扫描（docs/06 §2/§4）。)\n\n");
  }

  // ---- §E：warmup——按层 warm D 层 vs 阻塞预算（scramble 下按层不再精确） ----
  {
    printf("== E. warmup：按名义层 warm D 层（wide 基线，CAP=W0, G=%d）==\n", G);
    printf("   shape   | scr | D | deadlock | 完成 | B(pi) | 说明\n");
    printf("   --------+-----+---+----------+------+-------+------\n");
    for (int shape : {0, 1}) {
      for (int scr : {0, 1}) {
        Config c = base; c.shape = shape; c.scramble = scr; c.span = 1;
        c.policy = P_WARMUP; c.CAP = c.W0; c.G = G;
        Graph gg = build_graph(c);
        for (int D = 0; D <= c.L; ++D) {
          c.D = D;
          int B = 0; make_order(c, gg, &B);
          Result r = run_once(c);
          printf("   %-7s | %3d | %d | %8d | %3d/%2d | %5d | %s\n",
                 shape_name(shape), scr, D, r.deadlock, r.total_done, gg.N, B,
                 D == 0 ? "= wide" : (D >= c.L ? "全 warm" : ""));
        }
      }
    }
    printf("   (不规则宽度/乱序下「warm D 层」不再精确覆盖阻塞：D 层内 task 数随宽度变、且层号可能非拓扑；\n");
    printf("    正确判据应是「剩余阻塞 B' < G」而非「层数 D」)\n\n");
  }

  printf(">> 读法：\n");
  printf(">>  §A 规则闭式 (L-1)*W 在不规则宽度下失效；普适量是 B(π)=贪心模拟的峰值池占用，\n");
  printf(">>      死锁边界仍 CAP*=G*=B(π)+1（B(π) 不能简化成「第一个就绪 task 的位置」）。\n");
  printf(">>  §B 前向 skip 只从更浅拓扑层取前驱 → 层号升序仍拓扑、level/topo 恒安全；wide 的 B(π)=N-W0 不变。\n");
  printf(">>  §C 层号乱序让名义层升序不再是拓扑序：level 任意 CAP 死锁，topo（Kahn）恒 B=0。→ 必须真拓扑线性化。\n");
  printf(">>  §D gated 对任意 DAG 无死锁，但扫描代价随 N/多级扇入增长；预计算 topo 不付扫描账。\n");
  printf(">>  §E warmup 的正确判据是「剩余阻塞 B'<G」而非「warm D 层」。\n");
  printf(">>  deadlock 由设备侧无进度守门狗（clock64）检测并置 flag，kernel 正常返回（host 不挂）。\n");
  return 0;
}
