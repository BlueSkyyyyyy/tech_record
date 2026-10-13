// ring_deadlock.cu — MegaKernel 专题 P42-1 小实验：**有界 ready 池（ring/pool）+ warmup wave
// 的防死锁**。
//
// 动机：
//   DeepGEMM Mega MoE 的 in-kernel scheduler 有两件与「死锁」直接相关的机制（一手核对 `057ca59`）：
//     (1) `scheduler/mega_moe.cuh:15-45` 的 `get_num_l1_warmup_waves()`——注释原文
//         *"Get minimal L1 warmup waves to ensure no L1 -> L2 deadlock"*：在交错调度 L1/L2 task 前，
//         **先把前若干个 m-block 的 L1 task 全部发出去**（warmup wave），否则 L2 task 会先把
//         ring/pool 槽位占住、却在等永远发不出来的 L1 前置，形成死锁。
//     (2) `scheduler/mega_moe.cuh:47-80` 的 `get_num_max_live_pool_blocks()`——**host 侧闭式**给出
//         ring 容量上界（"Host-side ring capacity helper"）：ring 必须大到能装下「warmup 之后的活跃
//         block 数 + frontier 增长 + 一个 wave margin」，小了会 deadlock，大了浪费（`l2_full_mask`
//         的奇偶 generation 也依赖 ring 容量，见 `:140-166`）。
//   本专题此前测的是**调度策略**（P15-1 去中心化、P22-1 扇入/扇出四机制、P23-1 跨层流水），
//   都没测「**有界 ready 池的容量下界 + warmup wave 的防死锁必要性**」——这正是 checklist
//   （`docs/05` §2.3「防死锁」）点名、且只作资料转引的一条。
//
// 模型（单常驻 kernel，scheduler/worker 角色分离，对应 MPK §5.1）：
//   - `M` 个 m-block，每块 `Kp` 个 producer task（L1，可立即跑）与 `Kc` 个 consumer task（L2，
//     **必须等本块全部 Kp 个 producer 完成**——fan-in 依赖，对应 `L2KBlockDependency::wait`）。
//   - block 0 = **scheduler**（唯一 producer），其余 `G` 个 block = **worker**（消费者）。
//   - scheduler 按 `order[]` 顺序把 task **准入**一个有界 ready 池；池内「已准入未完成」的 task
//     数不能超过 `CAP`（占用语义 = task 在整条生命周期内占一个池槽，对应 ring block / SMEM 页）。
//     worker 从池里 pop task；遇到 consumer 且前置未齐就**自旋等待（占着 worker + 池槽）**。
//   - 两个准入顺序：`naive`（每块 **consumer 先于 producer**，对抗性交错）vs
//     `warmup`（先把前 `W` 个 m-block 的 producer 发出去 = warmup wave），其余同 naive。
//
// 三个问题：
//   Q1（§A）有界池 + 坏顺序是否会**真死锁**？死锁边界是否 = 「该 m-block 内 consumer 数（先于
//      producer 被准入）」，即 `CAP ≲ Kc`？
//   Q2（§B）warmup wave 是否**无条件消解**死锁（任意 CAP）？warmup 宽度 `W` 需要多大（对上
//      闭式 `get_num_l1_warmup_waves`）？
//   Q3（§C）池容量对**吞吐**的标度：即便不死锁，CAP 太小会让 worker 饿着（流水填不满），
//     阈值是否 ≈ worker 数 `G`（对上 `get_num_max_live_pool_blocks` 的 "max live"）？
//
// 安全：设备侧**无进度守门狗**（scheduler 检测到 `completed` 长时间不增即置 `deadlock_flag`，
//   所有 worker 在依赖自旋里检查该 flag 退出）——所以死锁 case 会让 kernel **正常返回**并打印，不会挂住 host。
//
// 编译：nvcc -O3 -arch=sm_90 src/ring_deadlock.cu -o /tmp/ring_deadlock
// 用法：
//   ./ring_deadlock run <naive|warmup> <CAP> <G> <M> <Kp> <Kc> <W> <iters>
//   ./ring_deadlock bench [iters]        # §A/§B/§C 默认矩阵

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

__device__ __forceinline__ float do_work(long iters, unsigned seed) {
  float a = (float)(seed & 1023) * 0.001f + 0.5f;
  float b = (float)((seed >> 10) & 1023) * 0.0005f + 0.25f;
  float acc = 0.f;
  for (long i = 0; i < iters; ++i) acc = fmaf(acc, 1.0000001f, a * b);
  return acc + a - b;
}

// order[cursor] = task id；task_prod[t]/task_mb[t] 给每个 task 的类型与 m-block。
__global__ void __launch_bounds__(32) sched_kernel(
    const unsigned* __restrict__ order,
    const int* __restrict__ task_prod,
    const int* __restrict__ task_mb,
    int N, int M,
    int* __restrict__ pool, int POOLCAP,
    int* __restrict__ q_head, int* __restrict__ q_tail,
    int* __restrict__ producers_done,   // [M]
    int* __restrict__ completed,
    int* __restrict__ deadlock_flag,
    float* __restrict__ out,            // [N]
    int CAP, int Kp, long work_iters,
    long idle_cycles) {
  if (threadIdx.x != 0) return;
  const int G_workers = gridDim.x - 1;

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
      while (cursor < N && (tail - done) < CAP) {
        int t = order[cursor++];
        pool[tail % POOLCAP] = t;
        ++tail;
      }
      __threadfence();
      *v_tail = tail;
      if (done != last) { last = done; t0 = clock64(); }
      else if ((clock64() - t0) > idle_cycles) { __threadfence(); *v_dl = 1; break; }
    }
  } else {
    // ---------------- worker ----------------
    (void)G_workers;
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
      int mb = task_mb[task];
      if (!task_prod[task]) {
        // consumer: 等本块全部 producer 完成（fan-in），自旋期间占着 worker + 池槽
        while (*(volatile int*)&producers_done[mb] < Kp) {
          if (*v_dl) break;
          __nanosleep(32);
        }
        if (*v_dl) break;
      }
      float v = do_work(work_iters, (unsigned)task);
      out[task] = v;
      if (task_prod[task]) atomicAdd(&producers_done[mb], 1);
      __threadfence();
      atomicAdd(completed, 1);
    }
  }
}

struct Config {
  int M, Kp, Kc, CAP, G, W;
  bool warmup;
  long work_iters, idle_cycles;
};

// 生成 order、prod、mb。naive = 每块 consumer 先于 producer；warmup 先把前 W 块的 producer 发出。
static void build_order(const Config& c, std::vector<unsigned>& order,
                        std::vector<int>& prod, std::vector<int>& mb) {
  int id = 0;
  mb.clear(); prod.clear();
  auto add = [&](int is_prod, int m) {
    mb.push_back(m); prod.push_back(is_prod);
  };
  // 先给所有 task 分配 id（按块顺序），order 决定准入次序
  struct Task { int id, is_prod, m; };
  std::vector<Task> tasks;
  for (int m = 0; m < c.M; ++m) {
    for (int j = 0; j < c.Kp; ++j) tasks.push_back({id++, 1, m});
    for (int j = 0; j < c.Kc; ++j) tasks.push_back({id++, 0, m});
  }
  for (auto& t : tasks) { add(t.is_prod, t.m); }
  order.clear();
  if (!c.warmup) {
    // naive: 每块 consumer 先、producer 后（对抗性交错）
    for (int m = 0; m < c.M; ++m) {
      for (int j = 0; j < c.Kc; ++j) order.push_back((unsigned)(m * (c.Kp + c.Kc) + c.Kp + j));
      for (int j = 0; j < c.Kp; ++j) order.push_back((unsigned)(m * (c.Kp + c.Kc) + j));
    }
  } else {
    // warmup: 先把前 W 块的 producer 全部发出，再按 naive 交错剩余
    std::vector<char> emitted((size_t)c.M * (c.Kp + c.Kc), 0);
    for (int m = 0; m < c.W && m < c.M; ++m)
      for (int j = 0; j < c.Kp; ++j) {
        unsigned t = (unsigned)(m * (c.Kp + c.Kc) + j);
        order.push_back(t); emitted[t] = 1;
      }
    for (int m = 0; m < c.M; ++m) {
      for (int j = 0; j < c.Kc; ++j) {
        unsigned t = (unsigned)(m * (c.Kp + c.Kc) + c.Kp + j);
        if (!emitted[t]) { order.push_back(t); emitted[t] = 1; }
      }
      for (int j = 0; j < c.Kp; ++j) {
        unsigned t = (unsigned)(m * (c.Kp + c.Kc) + j);
        if (!emitted[t]) { order.push_back(t); emitted[t] = 1; }
      }
    }
  }
}

struct Result { bool deadlock; int total_done; double ms; double sum_out; };

static Result run_once(const Config& c) {
  int N = c.M * (c.Kp + c.Kc);
  int POOLCAP = c.CAP + c.G + 4;

  std::vector<unsigned> order; std::vector<int> prod, mb;
  build_order(c, order, prod, mb);

  unsigned *d_order; int *d_prod, *d_mb, *d_pool, *d_head, *d_tail,
                     *d_pdone, *d_completed, *d_dl;
  float* d_out;
  CK(cudaMalloc(&d_order, N * sizeof(unsigned)));
  CK(cudaMalloc(&d_prod, N * sizeof(int)));
  CK(cudaMalloc(&d_mb, N * sizeof(int)));
  CK(cudaMalloc(&d_pool, POOLCAP * sizeof(int)));
  CK(cudaMalloc(&d_head, sizeof(int)));
  CK(cudaMalloc(&d_tail, sizeof(int)));
  CK(cudaMalloc(&d_pdone, c.M * sizeof(int)));
  CK(cudaMalloc(&d_completed, sizeof(int)));
  CK(cudaMalloc(&d_dl, sizeof(int)));
  CK(cudaMalloc(&d_out, N * sizeof(float)));
  CK(cudaMemcpy(d_order, order.data(), N * sizeof(unsigned), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_prod, prod.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_mb, mb.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_head, 0, sizeof(int)));
  CK(cudaMemset(d_tail, 0, sizeof(int)));
  CK(cudaMemset(d_pdone, 0, c.M * sizeof(int)));
  CK(cudaMemset(d_completed, 0, sizeof(int)));
  CK(cudaMemset(d_dl, 0, sizeof(int)));

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  sched_kernel<<<c.G + 1, 32>>>(d_order, d_prod, d_mb, N, c.M, d_pool, POOLCAP,
                                d_head, d_tail, d_pdone, d_completed, d_dl, d_out,
                                c.CAP, c.Kp, c.work_iters, c.idle_cycles);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));

  int h_dl = 0, h_done = 0;
  CK(cudaMemcpy(&h_dl, d_dl, sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&h_done, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<float> h_out(N, 0.f);
  CK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  double s = 0; for (float v : h_out) s += v;

  cudaFree(d_order); cudaFree(d_prod); cudaFree(d_mb); cudaFree(d_pool);
  cudaFree(d_head); cudaFree(d_tail); cudaFree(d_pdone); cudaFree(d_completed);
  cudaFree(d_dl); cudaFree(d_out);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return Result{h_dl != 0, h_done, ms, s};
}

// 解析参考：每个 task 的 out[] 由 do_work 决定，host 无浮点重算（device 用 float，host 用同一公式）。
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

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "bench";
  long work_iters = 20000;
  long idle_cycles = 1000000000;  // scheduler 无进度阈值（clock64 周期；~0.5s @2GHz）

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("== device ==\n%s, sm_%d%d, SMs=%d\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);

  if (strcmp(mode, "run") == 0) {
    // run <naive|warmup> <CAP> <G> <M> <Kp> <Kc> <W> <iters>
    Config c;
    c.warmup = strcmp(argv[2], "warmup") == 0;
    c.CAP = atoi(argv[3]); c.G = atoi(argv[4]); c.M = atoi(argv[5]);
    c.Kp = atoi(argv[6]); c.Kc = atoi(argv[7]); c.W = atoi(argv[8]);
    work_iters = (argc > 9) ? atol(argv[9]) : work_iters;
    c.work_iters = work_iters; c.idle_cycles = idle_cycles;
    Result r = run_once(c);
    printf("run order=%s CAP=%d G=%d M=%d Kp=%d Kc=%d W=%d: deadlock=%d done=%d/%d ms=%.3f\n",
           c.warmup ? "warmup" : "naive", c.CAP, c.G, c.M, c.Kp, c.Kc, c.W,
           r.deadlock, r.total_done, c.M * (c.Kp + c.Kc), r.ms);
    return 0;
  }

  // ---------------- bench ----------------
  int iters = (argc > 2) ? atoi(argv[2]) : 30;
  int G = 64, M = 32, Kp = 8, Kc = 4;
  Config base; base.G = G; base.M = M; base.Kp = Kp; base.Kc = Kc;
  base.work_iters = work_iters; base.idle_cycles = idle_cycles;
  int N = M * (Kp + Kc);

  printf("默认：G=%d, M=%d, Kp=%d, Kc=%d, N=%d, work_iters=%ld, idle_cycles=%ld\n\n",
         G, M, Kp, Kc, N, work_iters, idle_cycles);

  // 正确性参考（unbounded CAP=巨大 + warmup 必然收敛）
  {
    Config c = base; c.warmup = true; c.CAP = N + 1; c.W = M;
    Result r = run_once(c);
    double rs = ref_sum(N, work_iters);
    printf("== 正确性（warmup, CAP=N+1）: deadlock=%d done=%d/%d, sum=%.6f ref=%.6f (rel=%.2e)\n\n",
           r.deadlock, r.total_done, N, r.sum_out, rs,
           rs != 0 ? (r.sum_out - rs) / rs : 0.0);
  }

  // ---- §A：死锁边界（naive vs warmup，扫 CAP）----
  printf("== A. 有界池 + 准入顺序 vs 死锁（扫 CAP；M=%d Kp=%d Kc=%d，块内 consumer 先于 producer）==\n", M, Kp, Kc);
  printf("   CAP | naive(cons 先) | warmup(W=M=%d)\n", M);
  printf("   ----+----------------+----------------\n");
  for (int CAP = 1; CAP <= 12; ++CAP) {
    Config c = base; c.CAP = CAP; c.warmup = false;
    Result rn = run_once(c);
    c.warmup = true; c.W = M;
    Result rw = run_once(c);
    printf("   %3d | %-14s | %s\n", CAP,
           rn.deadlock ? "DEADLOCK" : "ok",
           rw.deadlock ? "DEADLOCK" : "ok");
  }
  printf("   (naive 在 CAP≤Kc=%d（首个 m-block 的 consumer 数）时把池堵死；CAP≥Kc+1 有 producer 挤进来即脱困；warmup 任意 CAP 皆 ok)\n\n",
         Kc);

  // ---- §B：warmup 宽度 W 的下界（固定 CAP 小）----
  {
    int CAP = Kc;  // 恰好 = 块内 consumer 数，naive 必死
    printf("== B. warmup 宽度 W 的下界（CAP=%d=Kc，naive 必死）==\n", CAP);
    printf("   W  | deadlock | done\n");
    printf("   ---+----------+------\n");
    for (int w : {0, 1, 2, 4, 8, 16, 32}) {
      Config c = base; c.CAP = CAP; c.warmup = true; c.W = w;
      Result r = run_once(c);
      printf("   %2d | %8d | %d/%d\n", w, r.deadlock, r.total_done, N);
    }
    printf("   (W=0 退化为 naive；W 每增大一档多救一个 block 的 12 个 task，需 W=M=%d 才全解)\n\n", M);
  }

  // ---- §C：吞吐 vs CAP（warmup，G=worker 数）----
  {
    printf("== C. 吞吐 vs 池容量 CAP（warmup，W=M，G=%d workers）==\n", G);
    printf("   CAP |    ms    | us/task |  vs CAP=G\n");
    printf("   ----+----------+---------+----------\n");
    auto time_cap = [&](int CAP) -> double {
      double best = 1e30;
      for (int r = 0; r < 3; ++r) {
        Config c = base; c.CAP = CAP; c.warmup = true; c.W = M;
        Result res = run_once(c);
        best = std::min(best, res.ms);
      }
      return best;
    };
    double t_base = time_cap(G);   // 基准 = CAP=G
    for (int CAP = 2; CAP <= 128; CAP *= 2) {
      double best = time_cap(CAP);
      printf("   %3d | %8.3f | %7.2f | %9.3f\n", CAP, best, best * 1000.0 / N,
             best / t_base);
    }
    printf("   (基准 CAP=G=%d：%.3f ms；CAP 小于 G 时池不够同时在飞 → 饿着变慢，CAP≥G 后饱和)\n\n",
           G, t_base);
  }

  // ---- §D：worker 数 G 扫描（固定 CAP 富余）----
  {
    int CAP = 128;
    printf("== D. worker 数 G 扫描（warmup, W=M, CAP=%d 富余）==\n", CAP);
    printf("   G  |    ms    | us/task | speedup vs G=16\n");
    printf("   ---+----------+---------+---------------\n");
    auto time_g = [&](int g) -> double {
      double best = 1e30;
      for (int r = 0; r < 3; ++r) {
        Config c = base; c.CAP = CAP; c.warmup = true; c.W = M; c.G = g;
        Result res = run_once(c);
        best = std::min(best, res.ms);
      }
      return best;
    };
    double t16 = time_g(16);
    for (int g : {1, 2, 4, 8, 16, 32, 64, 128}) {
      double best = time_g(g);
      printf("   %3d | %8.3f | %7.2f | %13.3f\n", g, best, best * 1000.0 / N,
             t16 / best);
    }
    printf("\n");
  }

  printf(">> 读法：\n");
  printf(">>  §A：坏顺序（块内 consumer 先于 producer）+ 有界池 → 池被「等前置的 consumer」堵死 → deadlock；\n");
  printf(">>      warmup（producer 先）在任意 CAP 皆可达。\n");
  printf(">>  §B：warmup 必须**够宽**——W 每档多救一个 block；W<M 则只推进到下一个未 warmup 的块再死。\n");
  printf(">>  §C：即便不死锁，CAP 太小 → worker 饿着（同时在飞不足 G）；CAP≥G 吞吐饱和（= ring 容量下界）。\n");
  printf(">>  §D：worker 数越多越快，直到任务数/池容量成为瓶颈。\n");
  printf(">>  deadlock 由设备侧无进度守门狗（clock64）检测并置 flag，kernel 正常返回（host 不挂）。\n");
  return 0;
}
