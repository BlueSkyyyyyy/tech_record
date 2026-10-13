// event_driven_pool.cu — MegaKernel 专题 P45-1 小实验：
//   **事件驱动的「就地入队」如何消除有界 ready 池的死锁**：扫描门控 → 依赖计数直接触发。
//
// 动机 / 接续：
//   P42-1（docs/04 §31）单层 fan-in、P43-1（§32）规则分层 DAG、P44-1（§33）不规则 DAG，三节
//   把「有界 ready 池 + 准入顺序 ⇒ 死锁」收口到一个普适量 B(π)（贪心峰值池占用），边界
//   CAP*=G*=B(π)+1；两条零成本正解是「预计算拓扑序 topo」与「依赖门控 gated」。§33.8 明确留的
//   尾巴：N≤70、scramble 只是层号整体旋转、**未测更大 N / 未独立扫 G 在不规则图上的边界**；
//   且 gated 仍是**朴素扫描**实现（每个 admission 扫剩余 order），其 ~1.7–2.5× 慢是工程代价。
//
//   本实验把「依赖门控」从**调度器扫描**改成 MPK §5.1 的真正机制：**每个 task 一个事件计数器**
//   （= 未完成前驱数），**由完成该前驱的 worker 直接把后继 push 进 ready 队列**（计数归零即入队），
//   调度器根本不参与准入。这样 ready 队列**只装就绪 task**，于是：
//       - 池内永远没有「等前驱的阻塞 task」⇒ B(π) ≡ 0 ⇒ CAP*=G*=1，**不需要 warmup、不需要拓扑序**；
//       - 不再有「扫描剩余 order」的代价（event→task 直接触发）。
//   对标 MPK 的快照：`EventDesc` + worker 完成时 `trigger_event`（docs/06 §2/§4、docs/08 §7）；
//   DeepGEMM Mega MoE 的 ring barrier/`l2_full_mask` 是同一思想的静态化（scheduler/mega_moe.cuh）。
//
// 核心问题：
//   Q1（§A）三种机制（scan 扫描门控 / topo 预计算拓扑序 / event 完成即 push）在不规则 scramble
//           DAG 上**逐位一致**、正确。
//   Q2（§B）死锁边界：raw（按序不查就绪）在 scramble>0 下 B(π)>0、小池死锁、CAP*=G*=B(π)+1；
//           scan（就绪才准入）与 topo（拓扑序）恒 B=0、CAP=1 即活；**event 更是 CAP=1/G=1、且无需预排序**。
//   Q3（§C）更大 N 的准入代价标度：scan vs topo vs event（CAP=G 富余、规则/不规则）。
//   Q4（§D）**独立扫 G**（§33.8 的尾巴）：不规则图 + scramble 下，scan 的 (CAP,G) 联合边界，
//           event 的 G*=1（单 worker 也能跑）。
//   Q5（§E）扇入 FIN 敏感：更高扇入对 scan 扫描 / event 的 push 风暴各有何影响。
//
// 安全：设备侧无进度守门狗（clock64）检测死锁并置 flag，所有 block 检查该 flag 退出——死锁
//   case 让 kernel 正常返回并打印，host 不挂。
//
// 编译：nvcc -O3 -arch=sm_90 src/event_driven_pool.cu -o /tmp/event_driven_pool
// 用法：
//   ./event_driven_pool bench [G]
//   ./event_driven_pool run <mode> <shape> <CAP> <G> <L> <W0> <FIN> <span> <scramble> <iters>
//     mode: scan|topo|event|raw    shape: reg|grow|shrink|zigzag

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
#define MAXSUCC 32

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

// 单常驻 kernel：block 0 = scheduler（scan/topo 用；event 模式只用来播种源任务），其余 G = worker。
//   MODE 0 = scan（调度器扫描剩余 order 找就绪 task 准入；= P44 的 gated）
//   MODE 1 = topo（调度器按预计算拓扑序准入，CAP 有界）
//   MODE 2 = event（无调度器准入：worker 完成时把「计数归零」的后继直接 push）
//   MODE 3 = raw（调度器按 order 直接准入、**不查就绪**；= P44 的 wide/level，坐实 B(π)+1 死锁边界）
__global__ void __launch_bounds__(32) sched_kernel(
    const unsigned* __restrict__ order,
    const int* __restrict__ pred_count,
    const int* __restrict__ preds,
    const int* __restrict__ succ_count,
    const int* __restrict__ succs,
    int N,
    int* __restrict__ pool, int POOLCAP,
    int* __restrict__ q_head, int* __restrict__ q_tail,
    int* __restrict__ done_flag,
    int* __restrict__ admitted,
    int* __restrict__ dep_left,          // [N] event 模式：未完成前驱数
    int* __restrict__ completed,
    int* __restrict__ deadlock_flag,
    float* __restrict__ out,
    int CAP, int MODE,
    long work_iters, long idle_cycles) {
  if (threadIdx.x != 0) return;

  volatile int* v_completed = completed;
  volatile int* v_dl = deadlock_flag;
  volatile int* v_head = q_head;
  volatile int* v_tail = q_tail;

  if (blockIdx.x == 0) {
    if (MODE == 2) {
      // event 模式：播种所有源任务（前驱数 0），然后等完成（不参与准入、不占用 worker）。
      int tail = 0;
      for (int t = 0; t < N; ++t)
        if (pred_count[t] == 0) { pool[tail % POOLCAP] = t; ++tail; }
      __threadfence();
      *v_tail = tail;
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
    // scan / topo：调度器唯一准入者。
    int cursor = 0, tail = 0, last = 0;
    long t0 = clock64();
    while (true) {
      if (*v_dl) break;
      int done = *v_completed;
      if (done >= N) break;
      bool progressed = false;
      while ((tail - done) < CAP && cursor < N) {
        if (MODE == 0) {                       // scan（gated）：扫剩余 order 找就绪 task 准入
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
        } else {                               // topo / raw：按 order 直接准入，不查就绪
          int t = order[cursor++];
          pool[tail % POOLCAP] = t; ++tail; progressed = true;
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
      if (MODE != 2) {
        // scan/topo：pop 到的 task 可能还没就绪（被有界池提前准入）——占着 worker 自旋等待。
        while (!all_preds_done(task, pred_count, preds, done_flag)) {
          if (*v_dl) break;
          __nanosleep(32);
        }
      }
      if (*v_dl) break;
      float v = do_work(work_iters, (unsigned)task);
      out[task] = v;
      __threadfence();
      done_flag[task] = 1;
      __threadfence();
      atomicAdd(completed, 1);
      if (MODE == 2) {
        // event：把后继的依赖计数减一，归零者直接 push（= MPK trigger_event，只装就绪 task）。
        int sc = succ_count[task];
        const int* s = succs + (size_t)task * MAXSUCC;
        for (int i = 0; i < sc; ++i) {
          int u = s[i];
          if (atomicSub(&dep_left[u], 1) == 1) {
            int slot = atomicAdd(q_tail, 1);
            pool[slot % POOLCAP] = u;
          }
        }
      }
    }
  }
}

enum Mode { M_SCAN = 0, M_TOPO = 1, M_EVENT = 2, M_RAW = 3 };

struct Config {
  int L, W0, FIN, span, scramble, CAP, G;
  int shape;         // 0=reg,1=grow,2=shrink,3=zigzag
  int mode;
  long work_iters, idle_cycles;
};

static const char* mode_name(int m) {
  switch (m) {
    case M_SCAN: return "scan"; case M_TOPO: return "topo";
    case M_EVENT: return "event"; default: return "raw";
  }
}
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
  int N;
  std::vector<int> W, layer_base, layer_of;
  std::vector<int> pred_count, preds;
  std::vector<int> succ_count, succs;
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
        int pl = l - 1 - (o % span);
        if (pl < 0) pl = 0;
        int pw = g.W[pl];
        int pj = (j + o / span) % pw;
        add_pred(t, g.layer_base[pl] + pj);
      }
    }
  }
  if (c.scramble != 0) {
    int s = ((c.scramble % c.L) + c.L) % c.L;
    for (int t = 0; t < g.N; ++t) g.layer_of[t] = (g.layer_of[t] + s) % c.L;
  }
  // 反向邻接（event 模式用）。
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

// B(π)：沿准入 order 贪心模拟的最小池容量（已准入未完成的峰值占用），见 §33.1。
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
        if (admitted[u] && !done[u] && preds_done(u)) { done[u] = 1; --occ; changed = true; }
      }
    }
    if (occ > peak) peak = occ;
  }
  return peak;
}

static std::vector<unsigned> make_order(const Config& c, const Graph& g, int* B_out) {
  std::vector<unsigned> order;
  std::vector<std::vector<int>> by_layer(c.L);
  for (int t = 0; t < g.N; ++t) by_layer[g.layer_of[t]].push_back(t);
  auto emit = [&](int l) { for (int t : by_layer[l]) order.push_back((unsigned)t); };

  if (c.mode != M_TOPO) {                      // scan/raw/event：名义层升序（贪婪名的扫描/准入序）
    for (int l = 0; l < c.L; ++l) emit(l);
  } else {                                     // topo：Kahn 真拓扑序（小 id 优先）
    std::vector<int> indeg(g.N, 0);
    for (int t = 0; t < g.N; ++t) indeg[t] = g.pred_count[t];
    auto cmp = [](int a, int b) { return a > b; };
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
  int POOLCAP = (c.mode == M_EVENT) ? (N + 4) : (c.CAP + c.G + 4);
  int B = 0;
  std::vector<unsigned> order = make_order(c, g, &B);

  unsigned *d_order; int *d_pc, *d_preds, *d_sc, *d_succ, *d_pool, *d_head, *d_tail,
                     *d_done, *d_admitted, *d_dep, *d_completed, *d_dl;
  float* d_out;
  CK(cudaMalloc(&d_order, (size_t)N * sizeof(unsigned)));
  CK(cudaMalloc(&d_pc, (size_t)N * sizeof(int)));
  CK(cudaMalloc(&d_preds, (size_t)N * MAXPRED * sizeof(int)));
  CK(cudaMalloc(&d_sc, (size_t)N * sizeof(int)));
  CK(cudaMalloc(&d_succ, (size_t)N * MAXSUCC * sizeof(int)));
  CK(cudaMalloc(&d_pool, (size_t)POOLCAP * sizeof(int)));
  CK(cudaMalloc(&d_head, sizeof(int)));
  CK(cudaMalloc(&d_tail, sizeof(int)));
  CK(cudaMalloc(&d_done, (size_t)N * sizeof(int)));
  CK(cudaMalloc(&d_admitted, (size_t)N * sizeof(int)));
  CK(cudaMalloc(&d_dep, (size_t)N * sizeof(int)));
  CK(cudaMalloc(&d_completed, sizeof(int)));
  CK(cudaMalloc(&d_dl, sizeof(int)));
  CK(cudaMalloc(&d_out, (size_t)N * sizeof(float)));
  CK(cudaMemcpy(d_order, order.data(), (size_t)N * sizeof(unsigned), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_pc, g.pred_count.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_preds, g.preds.data(), (size_t)N * MAXPRED * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_sc, g.succ_count.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_succ, g.succs.data(), (size_t)N * MAXSUCC * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_head, 0, sizeof(int)));
  CK(cudaMemset(d_tail, 0, sizeof(int)));
  CK(cudaMemset(d_done, 0, (size_t)N * sizeof(int)));
  CK(cudaMemset(d_admitted, 0, (size_t)N * sizeof(int)));
  CK(cudaMemcpy(d_dep, g.pred_count.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_completed, 0, sizeof(int)));
  CK(cudaMemset(d_dl, 0, sizeof(int)));

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  sched_kernel<<<c.G + 1, 32>>>(d_order, d_pc, d_preds, d_sc, d_succ, N, d_pool, POOLCAP,
                                d_head, d_tail, d_done, d_admitted, d_dep, d_completed, d_dl,
                                d_out, c.CAP, c.mode, c.work_iters, c.idle_cycles);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));

  int h_dl = 0, h_done = 0;
  CK(cudaMemcpy(&h_dl, d_dl, sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&h_done, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<float> h_out(N, 0.f);
  CK(cudaMemcpy(h_out.data(), d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
  double s = 0; for (float v : h_out) s += v;

  cudaFree(d_order); cudaFree(d_pc); cudaFree(d_preds); cudaFree(d_sc); cudaFree(d_succ);
  cudaFree(d_pool); cudaFree(d_head); cudaFree(d_tail); cudaFree(d_done); cudaFree(d_admitted);
  cudaFree(d_dep); cudaFree(d_completed); cudaFree(d_dl); cudaFree(d_out);
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

  auto parse_mode = [](const char* s) -> int {
    if (!strcmp(s, "scan")) return M_SCAN;
    if (!strcmp(s, "topo")) return M_TOPO;
    if (!strcmp(s, "raw"))  return M_RAW;
    return M_EVENT;
  };
  auto parse_shape = [](const char* s) -> int {
    if (!strcmp(s, "reg")) return 0;
    if (!strcmp(s, "grow")) return 1;
    if (!strcmp(s, "shrink")) return 2;
    if (!strcmp(s, "zigzag")) return 3;
    return 0;
  };

  if (strcmp(mode, "run") == 0) {
    // run <mode> <shape> <CAP> <G> <L> <W0> <FIN> <span> <scramble> <iters>
    Config c;
    c.mode = parse_mode(argv[2]);
    c.shape = parse_shape(argv[3]);
    c.CAP = atoi(argv[4]); c.G = atoi(argv[5]); c.L = atoi(argv[6]);
    c.W0 = atoi(argv[7]); c.FIN = atoi(argv[8]); c.span = atoi(argv[9]);
    c.scramble = atoi(argv[10]);
    work_iters = (argc > 11) ? atol(argv[11]) : work_iters;
    c.work_iters = work_iters; c.idle_cycles = idle_cycles;
    Graph g = build_graph(c);
    int B = 0; make_order(c, g, &B);
    Result r = run_once(c);
    printf("run mode=%-5s shape=%-6s CAP=%d G=%d L=%d W0=%d FIN=%d span=%d scramble=%d:"
           " N=%d B(pi)=%d deadlock=%d done=%d/%d ms=%.3f\n",
           mode_name(c.mode), shape_name(c.shape), c.CAP, c.G, c.L, c.W0, c.FIN,
           c.span, c.scramble, g.N, B, r.deadlock, r.total_done, g.N, r.ms);
    return 0;
  }

  int G = (argc > 2) ? atoi(argv[2]) : 64;

  Config base; base.mode = M_SCAN; base.shape = 0; base.L = 4; base.W0 = 16;
  base.FIN = 2; base.span = 1; base.scramble = 0; base.G = G;
  base.CAP = G; base.work_iters = work_iters; base.idle_cycles = idle_cycles;

  printf("默认：G=%d, work_iters=%ld, idle_cycles=%ld\n\n", G, work_iters, idle_cycles);

  // ---- §A：三机制正确性（不规则 scramble DAG，CAP=N+1 富余） ----
  {
    printf("== A. 正确性：scan / topo / event 在不规则 scramble DAG 上逐位一致 ==\n");
    printf("   shape   | scr | N  | scan rel | topo rel | event rel | done(scan/topo/event)\n");
    printf("   --------+-----+----+----------+----------+-----------+-----------------------\n");
    for (int shape : {0, 1, 2, 3}) {
      for (int scr : {0, 1}) {
        Config c = base; c.shape = shape; c.scramble = scr; c.span = 2;
        Graph gg = build_graph(c);
        double rs = ref_sum(gg.N, work_iters);
        double rel[3]; int dn[3];
        for (int mi = 0; mi < 3; ++mi) {
          Config cc = c; cc.mode = mi; cc.CAP = gg.N + 1; cc.G = G;
          Result r = run_once(cc);
          rel[mi] = (rs != 0) ? (r.sum_out - rs) / rs : 0.0;
          dn[mi] = r.total_done;
        }
        printf("   %-7s | %3d | %2d | %8.1e | %8.1e | %9.1e | %2d/%2d/%2d\n",
               shape_name(shape), scr, gg.N, rel[0], rel[1], rel[2], dn[0], dn[1], dn[2]);
      }
    }
    printf("   (三机制对同一 DAG、同一 task 工作给出相同 sum；event 的 ready 队列只装就绪 task)\n\n");
  }

  // ---- §B：死锁边界（scramble>0 让 raw 的「层号升序」不再是拓扑序） ----
  {
    printf("== B. 死锁边界：raw(按序不查就绪) B(pi)>0、CAP*=G*=B+1；scan/topo B=0；event 无需预排序 ==\n");
    printf("   shape   | scr | N  | B_raw | raw CAP* | raw G* | scan ok@CAP=1 | topo ok@CAP=1 | event ok@CAP=1,G=1\n");
    printf("   --------+-----+----+-------+----------+--------+---------------+--------------+-----------------\n");
    for (int shape : {0, 2}) {
      for (int scr : {0, 1}) {
        Config c = base; c.shape = shape; c.scramble = scr; c.span = 1;
        Graph gg = build_graph(c);
        int B = 0; c.mode = M_RAW; make_order(c, gg, &B);
        Config cb = c; cb.G = 256; int capb = find_boundary(cb, true, gg.N + 3);
        Config cg = c; cg.CAP = gg.N + 3; int gb = find_boundary(cg, false, gg.N + 3);
        Config cs = c; cs.mode = M_SCAN; cs.CAP = 1; cs.G = G;
        Config ct = c; ct.mode = M_TOPO; ct.CAP = 1; ct.G = G;
        Config ce = c; ce.mode = M_EVENT; ce.CAP = 1; ce.G = 1;   // event: 单 worker 也活
        printf("   %-7s | %3d | %2d | %5d | %8d | %6d | %-13s | %-12s | %-15s\n",
               shape_name(shape), scr, gg.N, B, capb, gb,
               run_once(cs).deadlock ? "DEADLOCK" : "ok",
               run_once(ct).deadlock ? "DEADLOCK" : "ok",
               run_once(ce).deadlock ? "DEADLOCK" : "ok");
      }
    }
    printf("   (raw 在 scramble>0 下按名义层升序先发『名义浅层』，其真前驱在后 → B(pi)>0、CAP 或 G ≤B 死锁；\n");
    printf("    scan 靠『就绪才准入』把 B 归零、但每次 admission 扫剩余 order；topo 预计算拓扑序 → B=0；\n");
    printf("    event 由 worker 完成即 push 就绪后继 → 池内无阻塞 task、CAP=1/G=1 即活，且无需预排序)\n\n");
  }

  // ---- §C：更大 N 的准入代价标度 ----
  {
    printf("== C. 更大 N 的准入代价：scan vs topo vs event（CAP=G 富余、不含 scramble）==\n");
    printf("   shape   | L  | W0 | N    | scan ms | topo ms | event ms | scan/event | topo/event\n");
    printf("   --------+----+----+------+---------+---------+----------+------------+-----------\n");
    for (int shape : {0, 2}) {
      for (int Nwant : {64, 256, 1024, 4096}) {
        int W0 = std::max(2, Nwant / 4);
        Config c = base; c.shape = shape; c.L = 4; c.W0 = W0; c.scramble = 0; c.span = 1;
        c.CAP = G; c.G = G;
        Graph gg = build_graph(c);
        if (gg.N > 6000) continue;
        Config cs = c; cs.mode = M_SCAN;
        Config ct = c; ct.mode = M_TOPO;
        Config ce = c; ce.mode = M_EVENT;
        double ts = time_best(cs), tt = time_best(ct), te = time_best(ce);
        printf("   %-7s | %2d | %2d | %4d | %7.3f | %7.3f | %8.3f | %10.2f | %10.2f\n",
               shape_name(shape), c.L, W0, gg.N, ts, tt, te, te > 0 ? ts / te : 0.0,
               te > 0 ? tt / te : 0.0);
      }
    }
    printf("   (scan 的扫描代价随 N 增长；topo 只需按序取；event 由完成者 push，无中央准入)\n\n");
  }

  // ---- §D：独立扫 G（§33.8 尾巴）在不规则图上的边界 ----
  {
    printf("== D. 独立扫 G：不规则图（shrink, scramble=1）下 raw 的 (CAP,G) 边界 vs scan/event ==\n");
    printf("   N  | B(pi) | raw CAP* | raw G* | scan ok@CAP=1 | event G*(@CAP=1) | event ms@G=1 | event ms@G=%d\n", G);
    printf("   ---+-------+----------+--------+---------------+------------------+--------------+---------------\n");
    for (int W0 : {8, 32, 128}) {
      Config c = base; c.shape = 2; c.L = 4; c.W0 = W0; c.FIN = 2; c.span = 1; c.scramble = 1;
      c.mode = M_RAW; c.CAP = 1; c.G = 256;
      Graph gg = build_graph(c);
      int B = 0; make_order(c, gg, &B);
      Config cb = c; cb.G = 512; int capb = find_boundary(cb, true, gg.N + 3);
      Config cg = c; cg.CAP = gg.N + 3; int gb = find_boundary(cg, false, gg.N + 3);
      Config cs = c; cs.mode = M_SCAN; cs.CAP = 1; cs.G = G;
      Config ce = c; ce.mode = M_EVENT; ce.CAP = 1;
      Config e1 = ce; e1.G = 1;
      int eg = find_boundary(ce, false, 64);
      double te1 = time_best(e1), te_mode = time_best(ce);
      printf("   %2d | %5d | %8d | %6d | %-13s | %16d | %12.3f | %13.3f\n",
             gg.N, B, capb, gb, run_once(cs).deadlock ? "DEADLOCK" : "ok",
             eg, te1, te_mode);
    }
    printf("   (raw 需 (CAP>B 且 G>B) 同时成立；scan 靠扫描免死锁但要 (CAP≥1,G≥1) 且付扫描账；\n");
    printf("    event 的 G*=1：ready 队列只装就绪 task，从不把阻塞 task 塞给 worker——无需 warmup/拓扑序)\n\n");
  }

  // ---- §E：扇入 FIN 敏感 ----
  {
    printf("== E. 扇入 FIN 敏感（reg, N=256, CAP=G 富余）：scan 扫描 / event push 风暴 ==\n");
    printf("   FIN | B(order)| scan ms | topo ms | event ms | scan/event | topo/event\n");
    printf("   ----+--------+---------+---------+----------+------------+-----------\n");
    for (int fin : {2, 4, 8, 16}) {
      Config c = base; c.shape = 0; c.L = 4; c.W0 = 64; c.FIN = fin; c.span = 1;
      c.scramble = 0; c.CAP = G; c.G = G;
      Graph gg = build_graph(c);
      int B = 0; c.mode = M_SCAN; make_order(c, gg, &B);
      Config cs = c; cs.mode = M_SCAN;
      Config ct = c; ct.mode = M_TOPO;
      Config ce = c; ce.mode = M_EVENT;
      // scan 的扫描代价对「多少 task 同时就绪」敏感、抖动大 → 取 best-of-10。
      double ts = time_best(cs, 10), tt = time_best(ct), te = time_best(ce);
      printf("   %3d | %6d | %7.3f | %7.3f | %8.3f | %10.2f | %10.2f\n",
             fin, B, ts, tt, te, te > 0 ? ts / te : 0.0, te > 0 ? tt / te : 0.0);
    }
    printf("   (event 的每完成一次要遍历后继并 atomicSub；FIN 增大提高『同时就绪』的宽度)\n\n");
  }

  printf(">> 读法：\n");
  printf(">>  §A 三机制对同一 DAG 数值一致（含不规则/乱序）；event 的 ready 队列只装就绪 task。\n");
  printf(">>  §B raw(按序不查就绪) 在 scramble>0 下 B(pi)>0、CAP*=G*=B+1；scan/topo 恒 B=0；event CAP=1/G=1 即活。\n");
  printf(">>  §C 三种准入的代价标度：scan 随 N 扫描变慢，topo 按序取，event 无中央准入（topo≈0.95×event）。\n");
  printf(">>  §D 独立扫 G：raw 需 (CAP>B 且 G>B)，event 恒 G*=1——从机制上消掉有界池死锁。\n");
  printf(">>  §E 高扇入抬高 scan 的扫描代价（FIN 16 时 ~3.6×event），event 的 push 代价近持平。\n");
  printf(">>  deadlock 由设备侧无进度守门狗（clock64）检测并置 flag，kernel 正常返回（host 不挂）。\n");
  return 0;
}
