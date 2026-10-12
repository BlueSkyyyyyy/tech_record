// P42-1: DeepGEMM / DeepJIT 的 kernel 发射层实验
//   - Programmatic Dependent Launch (PDL)：cudaGridDependencySynchronize /
//     cudaTriggerProgrammaticLaunchCompletion 的重叠收益与正确性；缺同步的「坏变体」。
//   - cluster / non-portable cluster / dynamic shared memory 的 launch attribute。
// 说明：DeepJIT 的 Kernel::launch 走 driver API cuLaunchKernelEx + CUlaunchAttribute
//   （third-party/deep_jit/include/deep_jit/backend/cuda/kernel.hpp:135-170）。
//   本实验用 runtime API cudaLaunchKernelEx + cudaLaunchAttribute 做同一件事
//   （两者填的是同一组 CUlaunchAttribute），在 H100(sm90) 上可跑。
// LaunchOptions 的 host 语义（default/override）另见 src/pdl_launch_host.cpp。
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); } } while (0)

static const int kThreads = 256;
static const int kTail = 200000;
static const int kPrologue = 200000;

// ------------------------- PDL producer / consumer -------------------------
// producer: 先写 out[i]（本轮 round）并 fence，然后可提前触发 programmatic completion，
//           再做一段 tail（模拟 epilogue / 收尾）。
__global__ void k_producer(int* out, int n, int round, int tail_iters, long long* sink, int early_trigger) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = round * 1000003 + i;
    __threadfence();
    if (early_trigger) cudaTriggerProgrammaticLaunchCompletion();
    long long s = 0;
    for (int k = 0; k < tail_iters; ++k) s += (long long)k * (i + 1);
    if (s == 0x123456789abcdefLL) *sink = s;
}

// consumer: prologue（独立工作，可与 producer tail 重叠）→ 可选 grid 依赖同步 →
//           读 out[i] 并校验（期望本轮 round）。
__global__ void k_consumer(const int* in, int n, int round, int prologue_iters,
                           int do_grid_sync, int* fail, long long* sink) {
    long long s = 0;
    for (int k = 0; k < prologue_iters; ++k) s += (long long)k * (threadIdx.x + 1);
    if (s == 0x123456789abcdefLL) *sink = s;
    if (do_grid_sync) cudaGridDependencySynchronize();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int exp = round * 1000003 + i;
        if (in[i] != exp) atomicAdd(fail, 1);
    }
}

// 「坏用法」producer：提前触发，但数据在 tail 之后才写 —— 用来暴露「consumer 缺 griddepsync」。
__global__ void k_producer_late(int* out, int n, int round, int tail_iters, long long* sink) {
    if (threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
    long long s = 0;
    for (int k = 0; k < tail_iters; ++k) s += (long long)k * (threadIdx.x + 1);
    if (s == 0x123456789abcdefLL) *sink = s;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = round * 1000003 + i;
}

// 短 prologue 的 consumer（用于暴露 stale 读）。
__global__ void k_consumer_short(const int* in, int n, int round, int do_grid_sync, int* fail) {
    if (do_grid_sync) cudaGridDependencySynchronize();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int exp = round * 1000003 + i;
        if (in[i] != exp) atomicAdd(fail, 1);
    }
}

// ------------------------- cluster / dyn smem -------------------------
__global__ void k_cluster(int* out) {
    unsigned rank = 0, ncta = 0;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm("mov.u32 %0, %%cluster_nctarank;" : "=r"(ncta));
    out[blockIdx.x] = (int)(rank * 1000 + ncta);
}

extern __shared__ int dyn_smem[];
__global__ void k_dynsmem(int* out, int num_ints) {
    for (int i = threadIdx.x; i < num_ints; i += blockDim.x) dyn_smem[i] = i;
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = dyn_smem[num_ints - 1];
}

// ------------------------- timing helpers -------------------------
static float time_pair(int pb, int cb, int n, int round, int tail, int prologue,
                       int early_trigger, int use_pdl, int do_grid_sync, int* out, int* fail,
                       long long* sink) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    CK(cudaMemset(fail, 0, sizeof(int)));
    cudaEventRecord(e0);
    k_producer<<<pb, kThreads>>>(out, n, round, tail, sink, early_trigger);
    if (use_pdl) {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(cb); cfg.blockDim = dim3(kThreads);
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr; cfg.numAttrs = 1;
        CK(cudaLaunchKernelEx(&cfg, k_consumer, (const int*)out, n, round, prologue, do_grid_sync, fail, sink));
    } else {
        k_consumer<<<cb, kThreads>>>(out, n, round, prologue, do_grid_sync, fail, sink);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

static float bench(int pb, int cb, int n, int tail, int prologue,
                   int early_trigger, int use_pdl, int do_grid_sync, int* out, int* fail,
                   long long* sink) {
    float best = 1e30f;
    for (int r = 0; r < 40; ++r) {
        float ms = time_pair(pb, cb, n, r + 1, tail, prologue,
                             early_trigger, use_pdl, do_grid_sync, out, fail, sink);
        if (ms < best) best = ms;
    }
    return best;
}

static float bench_producer_only(int pb, int n, int* out, long long* sink) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int r = 0; r < 40; ++r) {
        cudaEventRecord(e0);
        k_producer<<<pb, kThreads>>>(out, n, r + 1, kTail, sink, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return best;
}

int main() {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    const int num_sms = prop.multiProcessorCount;
    printf("DEVICE %s cc=%d.%d sm_count=%d\n", prop.name, prop.major, prop.minor, num_sms);

    const int n = 1 << 20;                 // 4 MiB output
    int *out, *fail; long long* sink;
    CK(cudaMalloc(&out, n * sizeof(int)));
    CK(cudaMalloc(&fail, sizeof(int)));
    CK(cudaMalloc(&sink, sizeof(long long)));
    CK(cudaMemset(out, 0xff, n * sizeof(int)));

    // ---- 分量计时 ----
    float t_prod_full = bench_producer_only(num_sms, n, out, sink);
    float t_prod_half = bench_producer_only(num_sms / 2, n, out, sink);
    float t_serial = bench(num_sms, num_sms, n, kTail, kPrologue, 1, 0, 1, out, fail, sink);
    printf("PDL_COMPONENT producer_full_ms=%.3f producer_half_ms=%.3f pair_serial_full_ms=%.3f tail=%d prologue=%d\n",
           t_prod_full, t_prod_half, t_serial, kTail, kPrologue);

    // ---- 全格（grid=132）：无空闲 SM，PDL 收益有限 ----
    {
        float ser = bench(num_sms, num_sms, n, kTail, kPrologue, 1, 0, 1, out, fail, sink);
        float pdl = bench(num_sms, num_sms, n, kTail, kPrologue, 1, 1, 1, out, fail, sink);
        int f; CK(cudaMemcpy(&f, fail, sizeof(int), cudaMemcpyDeviceToHost));
        printf("PDL full_grid serial_ms=%.3f overlap_ms=%.3f speedup=%.2fx fail=%d\n",
               ser, pdl, ser / pdl, f);
    }
    // ---- 半格（producer=66）：有空闲 SM，prologue 真正并发 ----
    {
        float ser = bench(num_sms / 2, num_sms / 2, n, kTail, kPrologue, 1, 0, 1, out, fail, sink);
        float pdl = bench(num_sms / 2, num_sms / 2, n, kTail, kPrologue, 1, 1, 1, out, fail, sink);
        int f; CK(cudaMemcpy(&f, fail, sizeof(int), cudaMemcpyDeviceToHost));
        printf("PDL half_grid serial_ms=%.3f overlap_ms=%.3f speedup=%.2fx fail=%d\n",
               ser, pdl, ser / pdl, f);
    }
    // ---- 触发器位置：提前触发 vs 隐式（kernel 结束）----
    {
        float early = bench(num_sms / 2, num_sms / 2, n, kTail, kPrologue, 1, 1, 1, out, fail, sink);
        float late  = bench(num_sms / 2, num_sms / 2, n, kTail, kPrologue, 0, 1, 1, out, fail, sink);
        printf("PDL trigger early_ms=%.3f implicit_end_ms=%.3f\n", early, late);
    }
    // ---- 设备端同步的必要性：缺 griddepsync 会读到 stale ----
    {
        const int rounds = 100;
        for (int gs = 1; gs >= 0; --gs) {
            CK(cudaMemset(fail, 0, sizeof(int)));
            for (int r = 0; r < rounds; ++r) {
                k_producer_late<<<num_sms / 2, kThreads>>>(out, n, r + 1, kTail, sink);
                cudaLaunchConfig_t cfg{};
                cfg.gridDim = dim3(num_sms / 2); cfg.blockDim = dim3(kThreads);
                cudaLaunchAttribute attr[1];
                attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                attr[0].val.programmaticStreamSerializationAllowed = 1;
                cfg.attrs = attr; cfg.numAttrs = 1;
                CK(cudaLaunchKernelEx(&cfg, k_consumer_short, (const int*)out, n, r + 1, gs, fail));
            }
            cudaDeviceSynchronize();
            int f; CK(cudaMemcpy(&f, fail, sizeof(int), cudaMemcpyDeviceToHost));
            printf("PDL stale grid_sync=%d fail=%d / %lld\n", gs, f, (long long)rounds * n);
        }
    }

    // ------------------- dynamic shared memory -------------------
    int *dout; CK(cudaMalloc(&dout, 4 * sizeof(int)));
    const int num_ints = 100 * 1024 / 4;          // 100 KiB dynamic smem
    const int smem_bytes = num_ints * sizeof(int);
    cudaFuncAttributes fa_before; cudaFuncGetAttributes(&fa_before, k_dynsmem);
    printf("DYN maxDynamicSharedMemorySize_default=%u\n", (unsigned)fa_before.maxDynamicSharedSizeBytes);
    k_dynsmem<<<1, 256, smem_bytes>>>(dout, num_ints);
    cudaError_t e_noattr = cudaGetLastError();
    printf("DYN launch_without_attr err=%d (%s)\n", (int)e_noattr, cudaGetErrorString(e_noattr));
    cudaGetLastError();
    cudaError_t e_set = cudaFuncSetAttribute(k_dynsmem, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaFuncAttributes fa_after; cudaFuncGetAttributes(&fa_after, k_dynsmem);
    k_dynsmem<<<1, 256, smem_bytes>>>(dout, num_ints);
    cudaError_t e_with = cudaGetLastError();
    int dval; CK(cudaMemcpy(&dval, dout, sizeof(int), cudaMemcpyDeviceToHost));
    printf("DYN set_attr_err=%d (%s) maxDynamicSharedMemorySize_after=%u launch_err=%d (%s) readback=%d expect=%d\n",
           (int)e_set, cudaGetErrorString(e_set), (unsigned)fa_after.maxDynamicSharedSizeBytes,
           (int)e_with, cudaGetErrorString(e_with), dval, num_ints - 1);

    // ------------------- cluster -------------------
    int *cout; CK(cudaMalloc(&cout, num_sms * sizeof(int)));
    for (int cs : {2, 4}) {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3((num_sms / cs) * cs); cfg.blockDim = dim3(128);
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = cs; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr; cfg.numAttrs = 1;
        cudaError_t e = cudaLaunchKernelEx(&cfg, k_cluster, cout);
        cudaDeviceSynchronize();
        int linear = cfg.gridDim.x;
        int* h = (int*)malloc(linear * sizeof(int));
        CK(cudaMemcpy(h, cout, linear * sizeof(int), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int b = 0; b < linear; ++b)
            if (h[b] != (b % cs) * 1000 + cs) ++bad;
        printf("CLUSTER cs=%d grid=%d launch_err=%d bad=%d sample=%d,%d\n",
               cs, linear, (int)e, bad, h[0], h[1]);
        free(h);
    }
    // non-portable cluster（H100 便携 cluster 上限 8，16 需显式允许）
    {
        const int cs = 16;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3((num_sms / cs) * cs); cfg.blockDim = dim3(128);
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = cs; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr; cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, k_cluster, cout);
        cudaError_t e_no = cudaGetLastError();
        cudaGetLastError();
        cudaError_t e_set = cudaFuncSetAttribute(k_cluster, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        cudaLaunchKernelEx(&cfg, k_cluster, cout);
        cudaDeviceSynchronize();
        cudaError_t e_with = cudaGetLastError();
        printf("CLUSTER nonportable cs=%d without_attr_err=%d (%s) set_attr_err=%d with_attr_err=%d (%s)\n",
               cs, (int)e_no, cudaGetErrorString(e_no), (int)e_set, (int)e_with, cudaGetErrorString(e_with));
    }

    printf("DONE\n");
    return 0;
}
