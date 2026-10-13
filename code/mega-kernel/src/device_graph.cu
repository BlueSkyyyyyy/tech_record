// P37-1: Device Graph Launch 作为「in-kernel 分发」原语 —— 设备侧图启动能不能替代
// host 派发/单 kernel 内循环？（补齐 P4-1 §1.6「device graph launch 仍停在 kernel 级」的尾巴，
// 并与 P36-1 §25 的 in-kernel `kinternal` / 条件节点 `cwhile` 对照。）
//
// 编译（目标卡 H100 / sm90a，CUDA 13.2；device 侧 runtime 需 -rdc=true -lcudadevrt）：
//   nvcc -arch=sm_90a -O2 -rdc=true device_graph.cu -o device_graph -lcudadevrt
// 运行： CUDA_VISIBLE_DEVICES=3 ./device_graph [N] [reps]
//
// 一条「单元」= 一次 tick（对 sink 做一次 atomicAdd）。要跑 N 个单元，对照四种「分发缝」：
//   eager    : host 循环 N 次 <<<>>> 直接 launch kernel          （host 派发 kernel）
//   hgraph   : host 循环 N 次 cudaGraphLaunch(1-node graph)      （host 派发 graph）
//   srelaunch: device-launchable graph，节点用 cudaGraphLaunch(cudaGetCurrentGraphExec(),
//              TailLaunch) 自重启 N 次；host 只提交 1 次         （device 派发 graph）
//   kinternal: 单 kernel 内 for N 次循环                          （常驻 kernel 自己派发）
// 外加 Section C：host-graph 里的 launcher 节点 TailLaunch 一个子图，测「一次 device 图启动」往返。

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static __device__ __forceinline__ unsigned long long gt() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));} }while(0)

__global__ void tick(int *sink) { if (threadIdx.x == 0) atomicAdd(sink, 1); }

__global__ void kinternal_kernel(int *sink, unsigned long long N) {
  if (threadIdx.x != 0) return;
  for (unsigned long long i = 0; i < N; i++) atomicAdd(sink, 1);
}

__global__ void srelaunch_kernel(unsigned long long *cnt, unsigned long long N, int *sink) {
  if (threadIdx.x != 0) return;
  atomicAdd(sink, 1);
  unsigned long long c = atomicAdd(cnt, 1ULL);
  if (c + 1 < N)
    cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
}

__global__ void fireonce_launcher(cudaGraphExec_t child, unsigned long long *t0, int *errcode) {
  if (threadIdx.x != 0) return;
  t0[0] = gt();
  cudaError_t e = cudaGraphLaunch(child, cudaStreamGraphTailLaunch);
  if (errcode) errcode[0] = (int)e;
}
__global__ void fireonce_child(unsigned long long *t1, int *sink) {
  if (threadIdx.x != 0) return;
  t1[0] = gt();
  atomicAdd(sink, 1);
}

// 同一 kernel 内连续两次 device launch 同一 exec，记录两次返回值。
__global__ void dual_launch_kernel(cudaGraphExec_t exec, int *codes) {
  if (threadIdx.x != 0) return;
  codes[0] = (int)cudaGraphLaunch(exec, cudaStreamGraphFireAndForget);
  codes[1] = (int)cudaGraphLaunch(exec, cudaStreamGraphFireAndForget);
}

// host 循环派发（mode 0 = 直接 launch kernel；mode 1 = cudaGraphLaunch）。返回每单元 ns。
static double time_host_loop(int mode, cudaGraphExec_t exec, int *sink,
                             unsigned long long N, int reps, int *errout) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, 0);
  for (int r = 0; r < reps; r++)
    for (unsigned long long i = 0; i < N; i++) {
      if (mode == 0) tick<<<1,1>>>(sink);
      else { cudaError_t e = cudaGraphLaunch(exec, 0); if (e && errout) *errout = (int)e; }
    }
  cudaEventRecord(b, 0); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms * 1e6 / (double)(N * reps);
}

int main(int argc, char **argv) {
  unsigned long long N = (argc > 1) ? strtoull(argv[1], 0, 10) : 20000ULL;
  int reps = (argc > 2) ? atoi(argv[2]) : 5;
  cudaSetDevice(0);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  printf("=== P37-1 device graph launch ===\n");
  printf("device=%s sm_%d%d  N=%llu reps=%d  total_units=%llu\n",
         prop.name, prop.major, prop.minor, N, reps, N * (unsigned long long)reps);

  int *sink; cudaMalloc(&sink, 4);
  unsigned long long *cnt; cudaMalloc(&cnt, 8);
  unsigned long long *t0, *t1; cudaMalloc(&t0, 8); cudaMalloc(&t1, 8);
  int *errcode; cudaMalloc(&errcode, 4); cudaMemset(errcode, 0, 4);

  // (1) 1-node tick graph：host-graph exec（普通 instantiate）与 device-launch exec。
  cudaGraph_t g1; cudaGraphCreate(&g1, 0);
  cudaGraphNode_t n1; cudaKernelNodeParams kp{};
  void *a1[] = {&sink};
  kp.func = (void*)tick; kp.gridDim = dim3(1); kp.blockDim = dim3(1); kp.kernelParams = a1;
  CK(cudaGraphAddKernelNode(&n1, g1, nullptr, 0, &kp));
  cudaGraphExec_t e_host;   // host 派发用（普通 instantiate）
  CK(cudaGraphInstantiate(&e_host, g1, 0));
  cudaGraphExec_t e_dev;    // device 自重启用（device launch flag）
  CK(cudaGraphInstantiateWithFlags(&e_dev, g1, cudaGraphInstantiateFlagDeviceLaunch));

  // (2) srelaunch graph（device-launchable）：单节点自重启。
  cudaGraph_t gs; cudaGraphCreate(&gs, 0);
  cudaGraphNode_t ns; cudaKernelNodeParams kps{};
  unsigned long long NN = N;
  void *as[] = {&cnt, &NN, &sink};
  kps.func = (void*)srelaunch_kernel; kps.gridDim = dim3(1); kps.blockDim = dim3(1); kps.kernelParams = as;
  CK(cudaGraphAddKernelNode(&ns, gs, nullptr, 0, &kps));
  cudaGraphExec_t es;
  CK(cudaGraphInstantiateWithFlags(&es, gs, cudaGraphInstantiateFlagDeviceLaunch));
  CK(cudaGraphUpload(es, 0));
  CK(cudaGraphUpload(e_dev, 0));

  // (3) Section C：host graph 里的 launcher 节点 TailLaunch 子图。
  cudaGraph_t gc; cudaGraphCreate(&gc, 0);
  cudaGraphNode_t nc; cudaKernelNodeParams kpc{};
  void *ac[] = {&t1, &sink};
  kpc.func = (void*)fireonce_child; kpc.gridDim = dim3(1); kpc.blockDim = dim3(1); kpc.kernelParams = ac;
  CK(cudaGraphAddKernelNode(&nc, gc, nullptr, 0, &kpc));
  cudaGraphExec_t ec;
  CK(cudaGraphInstantiateWithFlags(&ec, gc, cudaGraphInstantiateFlagDeviceLaunch));
  CK(cudaGraphUpload(ec, 0));
  cudaGraph_t gl; cudaGraphCreate(&gl, 0);
  cudaGraphNode_t nl; cudaKernelNodeParams kpl{};
  void *al[] = {&ec, &t0, &errcode};
  kpl.func = (void*)fireonce_launcher; kpl.gridDim = dim3(1); kpl.blockDim = dim3(1); kpl.kernelParams = al;
  CK(cudaGraphAddKernelNode(&nl, gl, nullptr, 0, &kpl));
  cudaGraphExec_t el;
  CK(cudaGraphInstantiate(&el, gl, 0));

  // (4) Section A2：同一 kernel 内连续两次 device launch 同一 exec。
  cudaGraph_t gd; cudaGraphCreate(&gd, 0);
  cudaGraphNode_t nd; cudaKernelNodeParams kpd{};
  int *codes; cudaMalloc(&codes, 8); cudaMemset(codes, 0xFF, 8);
  void *ad[] = {&e_dev, &codes};
  kpd.func = (void*)dual_launch_kernel; kpd.gridDim = dim3(1); kpd.blockDim = dim3(1); kpd.kernelParams = ad;
  CK(cudaGraphAddKernelNode(&nd, gd, nullptr, 0, &kpd));
  cudaGraphExec_t ed;
  CK(cudaGraphInstantiate(&ed, gd, 0));

  // ---- Section A：约束 ----
  printf("\n[Section A] 设备侧 cudaGraphLaunch 的使用约束\n");
  {
    // A1：直接 <<<>>> 启动一个调用 device cudaGraphLaunch 的 kernel（非 graph 节点）。
    cudaMemset(sink, 0, 4); cudaMemset(t0, 0, 8); cudaMemset(t1, 0, 8); cudaMemset(errcode, 0, 4);
    fireonce_launcher<<<1,1>>>(ec, t0, errcode);
    cudaError_t e = cudaGetLastError();
    cudaDeviceSynchronize();
    printf("  A1 plain <<<>>> launcher 直接调 device cudaGraphLaunch -> %s\n", cudaGetErrorString(e));

    // A2：把调用者放进 host graph 后，同 kernel 内连续两次 device launch 同一 exec。
    cudaMemset(codes, 0xFF, 8); cudaMemset(sink, 0, 4);
    CK(cudaGraphLaunch(ed, 0)); CK(cudaDeviceSynchronize());
    int c0, c1; cudaMemcpy(&c0, codes, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&c1, codes + 1, 4, cudaMemcpyDeviceToHost);
    printf("  A2 dual_launch (host graph 节点内连发两次): 1st=%s 2nd=%s\n",
           cudaGetErrorString((cudaError_t)c0), cudaGetErrorString((cudaError_t)c1));
  }

  // ---- Section B：每单元分发缝成本 ----
  printf("\n[Section B] 每个单元的分发缝成本（ns/unit）\n");
  for (int i = 0; i < 100; i++) tick<<<1,1>>>(sink);
  cudaDeviceSynchronize();

  int err_eager = 0, err_hgraph = 0;
  cudaMemset(sink, 0, 4);
  double e_eager = time_host_loop(0, e_host, sink, N, reps, &err_eager);
  int s_eager; cudaMemcpy(&s_eager, sink, 4, cudaMemcpyDeviceToHost);

  cudaMemset(sink, 0, 4);
  double e_hgraph = time_host_loop(1, e_host, sink, N, reps, &err_hgraph);
  int s_hgraph; cudaMemcpy(&s_hgraph, sink, 4, cudaMemcpyDeviceToHost);

  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaMemset(sink, 0, 4);
  cudaEventRecord(a, 0);
  for (int r = 0; r < reps; r++) kinternal_kernel<<<1,1>>>(sink, N);
  cudaEventRecord(b, 0); cudaEventSynchronize(b);
  float ms_ki; cudaEventElapsedTime(&ms_ki, a, b);
  double e_kinternal = ms_ki * 1e6 / (double)(N * reps);
  int s_kinternal; cudaMemcpy(&s_kinternal, sink, 4, cudaMemcpyDeviceToHost);

  cudaMemset(sink, 0, 4);
  cudaEventRecord(a, 0);
  for (int r = 0; r < reps; r++) { cudaMemset(cnt, 0, 8); CK(cudaGraphLaunch(es, 0)); }
  cudaEventRecord(b, 0); cudaEventSynchronize(b);
  float ms_sr; cudaEventElapsedTime(&ms_sr, a, b);
  double e_srelaunch = ms_sr * 1e6 / (double)(N * reps);
  int s_srelaunch; cudaMemcpy(&s_srelaunch, sink, 4, cudaMemcpyDeviceToHost);
  unsigned long long c_dev; cudaMemcpy(&c_dev, cnt, 8, cudaMemcpyDeviceToHost);

  unsigned long long exp = N * (unsigned long long)reps;
  printf("  %-10s %10.1f    sink=%d (exp %llu)   [host loop %s]\n",
         "eager", e_eager, s_eager, exp, err_eager ? cudaGetErrorString((cudaError_t)err_eager) : "ok");
  printf("  %-10s %10.1f    sink=%d (exp %llu)   [host loop %s]\n",
         "hgraph", e_hgraph, s_hgraph, exp, err_hgraph ? cudaGetErrorString((cudaError_t)err_hgraph) : "ok");
  printf("  %-10s %10.1f    sink=%d (exp %llu)\n", "kinternal", e_kinternal, s_kinternal, exp);
  printf("  %-10s %10.1f    sink=%d (exp %llu)   cnt/rep=%llu\n",
         "srelaunch", e_srelaunch, s_srelaunch, exp, c_dev);
  printf("  ratios vs eager: hgraph=%.2fx  kinternal=%.4fx  srelaunch=%.2fx\n",
         e_hgraph / e_eager, e_kinternal / e_eager, e_srelaunch / e_eager);

  // ---- Section C：一次 device 图启动（TailLaunch）往返 ----
  printf("\n[Section C] host-graph launcher 节点 TailLaunch 一个子图：\n");
  double best = 1e18; int feas = 0;
  for (int i = 0; i < 50; i++) {
    cudaMemset(sink, 0, 4); cudaMemset(t0, 0, 8); cudaMemset(t1, 0, 8); cudaMemset(errcode, 0, 4);
    CK(cudaGraphLaunch(el, 0)); CK(cudaDeviceSynchronize());
    unsigned long long h0, h1; int he;
    cudaMemcpy(&h0, t0, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h1, t1, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&he, errcode, 4, cudaMemcpyDeviceToHost);
    if (h1 > h0 && he == 0) { double d = (double)(h1 - h0); if (d < best) best = d; feas++; }
  }
  printf("  device-graph-launch round-trip = %.1f ns (feasible=%d/50)\n", best, feas);

  return 0;
}
