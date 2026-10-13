// layered_pool.cu — MegaKernel 专题 P43-1 小实验：**有界 ready 池 × 多级依赖 DAG**。
//
// 动机 / 接续：
//   P42-1（`docs/04` §31，源码 `src/ring_deadlock.cu`）把 DeepGEMM Mega MoE scheduler 的
//   `get_num_l1_warmup_waves()`（`scheduler/mega_moe.cuh:15-45`，注释 *"ensure no L1 -> L2 deadlock"*）
//   与 `get_num_max_live_pool_blocks()`（`:47-80`，"Host-side ring capacity helper"）做成实测：
//   单层 fan-in 模型下，**有界 ready 池的「准入顺序」决定死锁**——naive（块内 consumer 先于 producer）
//   在 `CAP≤Kc`（首个 m-block 的 consumer 数 = fan-in **宽度**）时死锁，`warmup`（先发前 W 块 producer）
//   任意 CAP 皆可达，但宽度须 W=M（把所有块都 warm 一遍）。
//   §31.6 明确留了尾巴：模型「只测 `Kp=8`、依赖是「每块 fan-in」……未测**多级扇入扇出 + 有界池**的
//   组合」（见 `docs/04` §31.6 第 1/3 条）。
//
// 本实验把依赖从「单层 fan-in」推广到**分层 DAG**（L 层 × 每层 W 个 task，第 l 层每个 task 依赖
//   第 l-1 层的 FIN 个前驱），在有界 ready 池上对照四种**准入策略**：
//     - `wide`   : 深度优先（层 L-1 → 0 降序），最对抗的顺序（把「等前驱的 task」先塞满池）。
//     - `level`  : 严格拓扑层序（层 0 → L-1 升序，每层按列序）。
//     - `warmup` : 先把前 D 层（升序）发出，其余按 `wide` 降序（= P42 `warmup` 的多级版，warmup 深度 D）。
//     - `gated`  : **依赖门控**——scheduler 只在某 task 的全部前驱**已 done** 时才把它准入（对应
//                  MPK §5.1 事件驱动入队 / DeepGEMM L1→L2 依赖检查 `get_next_task()`，见
//                  `docs/04` §31.1 与 `docs/06` §3；`gated` 用 `level` 的扫描次序）。
//
// 四个问题：
//   Q1（§A）多级 + 有界池 + 坏顺序（wide）的死锁边界：是随**深度 L**、**层宽 W**、还是二者乘积？
//      与 P42 单层的 `Kc`（宽度）对照。
//   Q2（§B）warmup 深度 D 需要多大？是否如 P42「须覆盖全部」那样须覆盖**整条依赖深度**（D≈L-1）？
//   Q3（§C）`level`（拓扑层序）与 `gated`（依赖门控）是否**任意 CAP 都无死锁**？二者的**吞吐**
//      随池容量的标度、容量下界是否仍 ≈ worker 数 G（对上 `get_num_max_live_pool_blocks` 的 "max live"）？
//   Q4（§D）worker 数 / 层数 / 扇入 FIN 的敏感性；正确性。
//
// 安全：设备侧**无进度守门狗**（scheduler 检测到 `completed` 长时间不增即置 `deadlock_flag`，
//   所有 worker 在依赖自旋里检查该 flag 退出）——死锁 case 让 kernel 正常返回并打印，host 不挂。
//
// 编译：nvcc -O3 -arch=sm_90 src/layered_pool.cu -o /tmp/layered_pool
// 用法：
//   ./layered_pool run <wide|level|warmup|gated> <CAP> <G> <L> <W> <FIN> <D> <iters>
//   ./layered_pool bench [iters]

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

__device__ __forceinline__ float do_work(long iters, unsigned seed) {
  float a = (float)(seed & 1023) * 0.001f + 0.5f;
  float b = (float)((seed >> 10) & 1023) * 0.0005f + 0.25f;
  float acc = 0.f;
  for (long i = 0; i < iters; ++i) acc = fmaf(acc, 1.0000001f, a * b);
  return acc + a - b;
}

// task id 编码：t = l*W + j。
__device__ __forceinline__ bool all_preds_done(
    int t, const int* __restrict__ pred_count, const int* __restrict__ preds,
    const int* __restrict__ done_flag) {
  int pc = pred_count[t];
  const int* p = preds + (size_t)t * MAXPRED;
  for (int i = 0; i < pc; ++i)
    if (*(volatile const int*)&done_flag[p[i]] == 0) return false;
  return true;
}

// order[cursor] = task id。GATED 模式下 scheduler 只准入「前驱全 done」的 task（按 order 扫描）。
__global__ void __launch_bounds__(32) sched_kernel(
    const unsigned* __restrict__ order,
    const int* __restrict__ pred_count,
    const int* __restrict__ preds,
    int N, int W,
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

  volatile int* v_completed = completed;
  volatile int* v_dl = deadlock_flag;
  volatile int* v_head = q_head;
  volatile int* v_tail = q_tail;

  if (blockIdx.x == 0) {
    // ---------------- scheduler ----------------
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
          // 依赖门控：从 cursor 起找「未准入且前驱全 done」的 task
          bool found = false;
          for (int c = cursor; c < N; ++c) {
            int t = order[c];
            if (atomicCAS(&admitted[t], 0, 1) == 0) {
              if (all_preds_done(t, pred_count, preds, done_flag)) {
                pool[tail % POOLCAP] = t; ++tail; progressed = true;
                found = true;
                // 把 cursor 推进到已准入连续段之后（简单起见只前移一格）
                if (c == cursor) ++cursor;
                break;
              } else {
                // 此 task 尚不可准入，回滚标记，继续找后面的
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
    // ---------------- worker ----------------
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
      // 等全部前驱 done（fan-in），自旋期间占着 worker + 池槽（对应 ring block / SMEM 页）
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

struct Config {
  int L, W, FIN, CAP, G, D;
  int policy;   // 0=wide,1=level,2=warmup,3=gated
  long work_iters, idle_cycles;
};

static const char* policy_name(int p) {
  switch (p) { case 0: return "wide"; case 1: return "level"; case 2: return "warmup"; default: return "gated"; }
}

// 生成准入 order（task id = l*W + j）与前驱表。
static void build_graph(const Config& c, std::vector<unsigned>& order,
                        std::vector<int>& pred_count, std::vector<int>& preds) {
  int N = c.L * c.W;
  pred_count.assign(N, 0);
  preds.assign((size_t)N * MAXPRED, -1);
  for (int l = 1; l < c.L; ++l) {
    for (int j = 0; j < c.W; ++j) {
      int t = l * c.W + j;
      for (int o = 0; o < c.FIN; ++o) {
        int pj = (j + o) % c.W;
        preds[(size_t)t * MAXPRED + o] = (l - 1) * c.W + pj;
      }
      pred_count[t] = c.FIN;
    }
  }
  order.clear();
  auto emit = [&](int l) { for (int j = 0; j < c.W; ++j) order.push_back((unsigned)(l * c.W + j)); };
  if (c.policy == 0) {                    // wide: 层降序
    for (int l = c.L - 1; l >= 0; --l) emit(l);
  } else if (c.policy == 1 || c.policy == 3) {  // level / gated: 层升序
    for (int l = 0; l < c.L; ++l) emit(l);
  } else {                                // warmup: 前 D 层升序，其余降序
    int D = std::min(c.D, c.L);
    for (int l = 0; l < D; ++l) emit(l);
    for (int l = c.L - 1; l >= D; --l) emit(l);
  }
}

struct Result { bool deadlock; int total_done; double ms; double sum_out; };

static Result run_once(const Config& c) {
  int N = c.L * c.W;
  int POOLCAP = c.CAP + c.G + 4;

  std::vector<unsigned> order; std::vector<int> pred_count, preds;
  build_graph(c, order, pred_count, preds);

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
  CK(cudaMemcpy(d_pc, pred_count.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_preds, preds.data(), (size_t)N * MAXPRED * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_head, 0, sizeof(int)));
  CK(cudaMemset(d_tail, 0, sizeof(int)));
  CK(cudaMemset(d_done, 0, N * sizeof(int)));
  CK(cudaMemset(d_admitted, 0, N * sizeof(int)));
  CK(cudaMemset(d_completed, 0, sizeof(int)));
  CK(cudaMemset(d_dl, 0, sizeof(int)));

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  sched_kernel<<<c.G + 1, 32>>>(d_order, d_pc, d_preds, N, c.W, d_pool, POOLCAP,
                                d_head, d_tail, d_done, d_admitted, d_completed, d_dl,
                                d_out, c.CAP, c.policy == 3 ? 1 : 0,
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

// host 解析参考（与 device 同公式，逐 task）。
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

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "bench";
  long work_iters = 20000;
  long idle_cycles = 100000000;  // ~0.05s @2GHz，无进度阈值（task ~0.1ms，余量 ~500×）

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("== device ==\n%s, sm_%d%d, SMs=%d\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);

  auto parse_policy = [](const char* s) -> int {
    if (!strcmp(s, "wide")) return 0;
    if (!strcmp(s, "level")) return 1;
    if (!strcmp(s, "warmup")) return 2;
    if (!strcmp(s, "gated")) return 3;
    return 0;
  };

  if (strcmp(mode, "run") == 0) {
    // run <policy> <CAP> <G> <L> <W> <FIN> <D> <iters>
    Config c;
    c.policy = parse_policy(argv[2]);
    c.CAP = atoi(argv[3]); c.G = atoi(argv[4]); c.L = atoi(argv[5]);
    c.W = atoi(argv[6]); c.FIN = atoi(argv[7]); c.D = atoi(argv[8]);
    work_iters = (argc > 9) ? atol(argv[9]) : work_iters;
    c.work_iters = work_iters; c.idle_cycles = idle_cycles;
    Result r = run_once(c);
    printf("run policy=%s CAP=%d G=%d L=%d W=%d FIN=%d D=%d: deadlock=%d done=%d/%d ms=%.3f\n",
           policy_name(c.policy), c.CAP, c.G, c.L, c.W, c.FIN, c.D,
           r.deadlock, r.total_done, c.L * c.W, r.ms);
    return 0;
  }

  // ---------------- bench ----------------
  int iters = (argc > 2) ? atoi(argv[2]) : 30;
  (void)iters;
  int G = 64, L = 4, W = 16, FIN = 2;
  Config base; base.G = G; base.L = L; base.W = W; base.FIN = FIN; base.D = 0;
  base.work_iters = work_iters; base.idle_cycles = idle_cycles;
  int N = L * W;

  printf("默认：L=%d, W=%d, FIN=%d, N=%d, G=%d, work_iters=%ld, idle_cycles=%ld\n\n",
         L, W, FIN, N, G, work_iters, idle_cycles);

  // 正确性参考（gated, CAP=N+1 必然收敛）
  {
    Config c = base; c.policy = 3; c.CAP = N + 1;
    Result r = run_once(c);
    double rs = ref_sum(N, work_iters);
    printf("== 正确性（gated, CAP=N+1）: deadlock=%d done=%d/%d, sum=%.6f ref=%.6f (rel=%.2e)\n\n",
           r.deadlock, r.total_done, N, r.sum_out, rs,
           rs != 0 ? (r.sum_out - rs) / rs : 0.0);
  }

  // ---- §A：死锁边界（wide vs level vs gated，扫 CAP）----
  printf("== A. 有界池 + 准入策略 vs 死锁（扫 CAP；L=%d W=%d FIN=%d）==\n", L, W, FIN);
  printf("   CAP | wide(层降序) | level(层升序) | gated(依赖门控)\n");
  printf("   ----+--------------+---------------+----------------\n");
  int wide_boundary = -1, level_boundary = -1, gated_boundary = -1;
  for (int CAP = 1; CAP <= 64; CAP *= 2) {
    Config c = base; c.CAP = CAP;
    c.policy = 0; Result rw = run_once(c);
    c.policy = 1; Result rl = run_once(c);
    c.policy = 3; Result rg = run_once(c);
    if (wide_boundary < 0 && !rw.deadlock) wide_boundary = CAP;
    if (level_boundary < 0 && !rl.deadlock) level_boundary = CAP;
    if (gated_boundary < 0 && !rg.deadlock) gated_boundary = CAP;
    printf("   %3d | %-12s | %-13s | %s\n", CAP,
           rw.deadlock ? "DEADLOCK" : "ok", rl.deadlock ? "DEADLOCK" : "ok",
           rg.deadlock ? "DEADLOCK" : "ok");
  }
  printf("   (wide 采样下首次 ok @ CAP=%d；联合边界 CAP>B 且 G>B，B=(L-1)*W=%d，此处 G=%d；\n",
         wide_boundary, (L - 1) * W, G);
  printf("    level/gated 首个 ok @ %d/%d —— 任意 CAP 皆无死锁)\n\n", level_boundary, gated_boundary);

  // ---- §A2：wide 死锁边界随深度 L 的标度（资源富余，分别扫 CAP 与 G）----
  {
    printf("== A2. wide 边界 = 「阻塞在前的前驱等待 task 数」B=(L-1)*W（W=%d, FIN=%d）==\n", W, FIN);
    printf("   L  | B=(L-1)*W | CAP*(G=256) | G*(CAP=256) | B+1\n");
    printf("   ---+-----------+-------------+-------------+-----\n");
    for (int l : {1, 2, 3, 4, 6, 8}) {
      int B = (l - 1) * W;
      int capb = -1, gb = -1;
      for (int CAP = 1; CAP <= B + 3; ++CAP) {
        Config c = base; c.policy = 0; c.L = l; c.CAP = CAP; c.G = 256;
        if (!run_once(c).deadlock) { capb = CAP; break; }
      }
      for (int G = 1; G <= B + 3; ++G) {
        Config c = base; c.policy = 0; c.L = l; c.CAP = 256; c.G = G;
        if (!run_once(c).deadlock) { gb = G; break; }
      }
      printf("   %2d | %9d | %11d | %11d | %d\n", l, B, capb, gb, B + 1);
    }
    printf("   (资源分量各自富余时，两个边界都 = B+1=(L-1)*W+1：CAP 必须 > B 才能把一个可运行的\n");
    printf("    task 塞进池；G 必须 > B 才有空闲 worker 去跑它——**池容量与 worker 数是联合约束**)\n\n");
  }

  // ---- §A3：(CAP,G) 联合平面（L=8，B=112；G=#SM 固定时的部署含义）----
  {
    int l = 8; int B = (l - 1) * W;
    printf("== A3. (CAP,G) 联合死锁平面（wide, L=%d B=%d）==\n", l, B);
    printf("   CAP\\G |");
    int gs[] = {16, 32, 64, 128, 256};
    for (int g : gs) printf(" G=%-3d", g);
    printf("\n   ------+");
    for (size_t i = 0; i < sizeof(gs) / sizeof(int); ++i) printf("------");
    printf("\n");
    for (int CAP : {32, 64, 96, 112, 113, 128}) {
      printf("   %5d |", CAP);
      for (int g : gs) {
        Config c = base; c.policy = 0; c.L = l; c.CAP = CAP; c.G = g;
        Result r = run_once(c);
        printf(" %-5s", r.deadlock ? "DEAD" : "ok");
      }
      printf("\n");
    }
    printf("   (只有当 CAP>B 且 G>B 二者同时成立才达 → 真实 megakernel G=#SM 时，单靠加大 ring 容量不救；\n");
    printf("    必须先 warmup/拓扑序把 B 压下去)\n\n");
  }

  // ---- §B：warmup 深度 D 的下界（固定小 CAP）----
  {
    int CAP = W;  // = 层宽，wide 在 L>=2 必死（(L-1)*W+1 > W 当 L>2）
    printf("== B. warmup 深度 D 的下界（CAP=%d=W，G=%d；wide 必死；L=%d）==\n", CAP, G, L);
    printf("   D  | deadlock | done | 剩余阻塞 B'=(L-1-D)*W\n");
    printf("   ---+----------+------+----------------------\n");
    for (int D = 0; D <= L; ++D) {
      Config c = base; c.policy = 2; c.CAP = CAP; c.D = D;
      Result r = run_once(c);
      int bp = (L - 1 - D) * W; if (bp < 0) bp = 0;
      printf("   %2d | %8d | %4d | %d\n", D, r.deadlock, r.total_done, bp);
    }
    printf("   (D=0 退化为 wide；D 每 warm 一层把 B 降 W；须 D≈L-1（warm 完整依赖深度）才在 CAP=W\n");
    printf("    且 G 富余时全解。注意 B' 同时决定「池容量」和「worker 占用」两笔账)\n\n");
  }

  // ---- §C：吞吐 vs 池容量 CAP（level vs gated）----
  {
    printf("== C. 吞吐 vs 池容量 CAP（level vs gated，G=%d workers）==\n", G);
    printf("   CAP | level ms | gated ms | level/gated\n");
    printf("   ----+----------+----------+------------\n");
    double tb = 0;
    for (int CAP = 2; CAP <= 128; CAP *= 2) {
      Config c = base; c.CAP = CAP;
      c.policy = 1; double tl = time_best(c);
      c.policy = 3; double tg = time_best(c);
      if (CAP == G) tb = tl;
      printf("   %3d | %8.3f | %8.3f | %10.3f\n", CAP, tl, tg, tg > 0 ? tl / tg : 0.0);
    }
    printf("   (level 基准 CAP=G=%d：%.3f ms；二者都在 CAP≥G 饱和 —— 容量下界 = worker 数)\n\n", G, tb);
  }

  // ---- §D：worker 数 G 扫描（gated, CAP 富余）----
  {
    int CAP = 128;
    printf("== D. worker 数 G 扫描（gated, CAP=%d 富余, L=%d, N=%d）==\n", CAP, L, N);
    printf("   G  |    ms    | us/task | speedup vs G=16\n");
    printf("   ---+----------+---------+---------------\n");
    auto time_g = [&](int g) -> double {
      Config c = base; c.policy = 3; c.CAP = CAP; c.G = g;
      return time_best(c);
    };
    double t16 = time_g(16);
    for (int g : {1, 2, 4, 8, 16, 32, 64, 128}) {
      double best = time_g(g);
      printf("   %3d | %8.3f | %7.2f | %13.3f\n", g, best, best * 1000.0 / N, t16 / best);
    }
    printf("\n");
  }

  // ---- §E：扇入 FIN 敏感性（边界是否只由 L,W 决定）----
  {
    printf("== E. 扇入 FIN 对 wide 边界的影响（L=%d, W=%d, G=256 富余）==\n", L, W);
    printf("   FIN | CAP* (首次 ok) | B+1=(L-1)*W+1\n");
    printf("   ----+---------------+---------------\n");
    for (int fin : {1, 2, 4, 8}) {
      int bound = -1;
      for (int CAP = 1; CAP <= (L - 1) * W + 3; ++CAP) {
        Config c = base; c.policy = 0; c.FIN = fin; c.CAP = CAP; c.G = 256;
        if (!run_once(c).deadlock) { bound = CAP; break; }
      }
      printf("   %3d | %13d | %d\n", fin, bound, (L - 1) * W + 1);
    }
    printf("   (wide 的边界只由层序几何 L,W 决定，与扇入宽度 FIN 无关 → 边界由「阻塞数 B=(L-1)*W」定，\n");
    printf("    对照 P42 单层 fan-in 的边界 = fan-in 宽度 Kc（同为「阻塞在前的前驱等待 task 数」）)\n\n");
  }

  printf(">> 读法：\n");
  printf(">>  §A/§A2/§A3：坏顺序（层降序）把「阻塞在前的前驱等待 task」B=(L-1)*W 个先塞进有界池；\n");
  printf(">>         只有当 CAP>B 且 G>B 同时成立才达 → 池容量与 worker 数**联合**决定死锁。\n");
  printf(">>         对照 P42 单层：B=fan-in 宽度 Kc（同一「阻塞数」量）。\n");
  printf(">>  §B：warmup 深度 D 每 warm 一层把 B 降 W；须 warm 完整依赖深度（D≈L-1）才在 CAP=W 全解。\n");
  printf(">>  §C：严格拓扑层序（level）与依赖门控（gated）**任意 CAP 皆无死锁**；gated 大 CAP 下慢 ~2×\n");
  printf(">>         （scheduler 逐次扫描就绪 task 的开销）→ 拓扑层序是「零成本防死锁」的正解。\n");
  printf(">>  §E：边界只由层序几何 L,W 决定，与扇入宽度 FIN 无关。\n");
  printf(">>  deadlock 由设备侧无进度守门狗（clock64）检测并置 flag，kernel 正常返回（host 不挂）。\n");
  return 0;
}
