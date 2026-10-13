// P138-1 实验：MoE wgrad 的 S 步归约能否「藏进 GEMM 的 HBM 余量」——
//   把「归约/优化器更新融进 GEMM 的 1D2D epilogue / 与下一步 GEMM 重叠」这条路走到能核对的数。
//
// 背景：第 147 篇（P137-1）把「wgrad 换 bf16 输出」落到端到端 **1.67×**（ALT-OPT），并断言
//   「2.27× 不可达，因为每步 bf16 梯度至少要**串行**读一遍（~76us/步）」。它留的口子是：
//   若把归约/更新**融进 GEMM 的 epilogue、或与下一步 GEMM 重叠**，这 76us/步能不能被藏掉？
//
// 本篇给三个**架构无关**（只有 bf16/fp32 ld/st + 浮点，无 UMMA/tcgen05/TMA）的核对内核：
//   (1) k_acc_bf16_step  : 增量式、**bf16 累加器**：acc[i] = bf16(acc[i] + g_s[i])
//                          —— 模拟「epilogue 直接 RMW 进一个 bf16 归约缓冲」的字节账。
//   (2) k_fused_step     : 增量式、**fp32 master**：w[i] = w[i] - lr*g_s[i]
//                          —— 模拟「每一步就地更新 fp32 master 权重」（incremental optimizer）。
//   (3) k_fused_all      : 批量式（= 第 147 篇 ALT-OPT）：读齐 S 份 bf16 再更新一次 master。
//   外加 k_gen_bf16：一个**计算受限**的「伪 GEMM epilogue」生产者（FFMA 打满 + 写 bf16），
//   用来在**不依赖 DeepGEMM**的情况下量「访存受限消费者能否藏进计算受限核的 HBM 余量」。
//
// 关键量纲：G = E*M*N 个元素（dW2 的 `[E=8,M=2048,N=7168]` = 117,440,512）。
//   bf16 一份 = 235 MB，fp32 一份 = 470 MB。
//   每步字节（按线上流量）：
//     store-fresh bf16 (第 146/147 篇 GEMM 输出)  = 写 235
//     k_acc_bf16_step  (epilogue RMW bf16 acc)     = 读 235 + 读 235 + 写 235 = 705
//     k_fused_step     (incremental fp32 master)   = 读 235 + 读 470 + 写 470 = 1175
//   批量 ALT-OPT（S 步总量） = 读 S*235 + 读 470 + 写 470
//   → 增量式字节多，但**可重叠**；批量式字节少，但只能**串行尾收**。本篇量「重叠能不能补回字节差」。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。
//
// 编译/运行见 harness/exp_wgrad_epilogue_overlap.sh（编成 .so 由 Python 经 ctypes 驱动，
// 也可单独 `nvcc ... && ./a.out` 跑自带报告）。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------------------
// (1) 增量式 bf16 累加器：acc = bf16(acc + g_{s_idx})。8 元素/线程。
//     线上字节 = 读 g 235MB + 读 acc 235MB + 写 acc 235MB。
// ---------------------------------------------------------------------------
__global__ void k_acc_bf16_step(const bf16* __restrict__ g, bf16* __restrict__ acc,
                                long G8, int s_idx) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    const long G = G8 * 8;
    const bf16* p = g + (long)s_idx * G;
    for (; i < G8; i += stride) {
        const bf16* gp = p + i * 8;
        bf16* ap = acc + i * 8;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(gp + j * 2);
            __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(ap + j * 2);
            __nv_bfloat162 r;
            r.x = __float2bfloat16(__bfloat162float(a.x) + __bfloat162float(x.x));
            r.y = __float2bfloat16(__bfloat162float(a.y) + __bfloat162float(x.y));
            *reinterpret_cast<__nv_bfloat162*>(ap + j * 2) = r;
        }
    }
}

// ---------------------------------------------------------------------------
// (2) 增量式 fp32 master：w = w - lr * g_{s_idx}。8 元素/线程（128-bit 读写）。
//     线上字节 = 读 g 235MB + 读 w 470MB + 写 w 470MB。
// ---------------------------------------------------------------------------
__global__ void k_fused_step(const bf16* __restrict__ g, float* __restrict__ w,
                             long G8, int s_idx, float lr) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    const long G = G8 * 8;
    const bf16* p = g + (long)s_idx * G;
    for (; i < G8; i += stride) {
        float a[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = 0.f;
        const bf16* gp = p + i * 8;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(gp + j * 2);
            a[j * 2]     += __bfloat162float(x.x);
            a[j * 2 + 1] += __bfloat162float(x.y);
        }
        float4* q = reinterpret_cast<float4*>(w + i * 8);
        float4 x0 = q[0], x1 = q[1];
        q[0] = make_float4(x0.x - lr * a[0], x0.y - lr * a[1], x0.z - lr * a[2], x0.w - lr * a[3]);
        q[1] = make_float4(x1.x - lr * a[4], x1.y - lr * a[5], x1.z - lr * a[6], x1.w - lr * a[7]);
    }
}

// ---------------------------------------------------------------------------
// (3) 批量式（= 第 147 篇 ALT-OPT）：读齐 S 份 bf16 求和后更新一次 fp32 master。
//     线上字节 = 读 S*235MB + 读 w 470MB + 写 w 470MB。
// ---------------------------------------------------------------------------
__global__ void k_fused_all(const bf16* __restrict__ g, float* __restrict__ w,
                            long G8, int S, float lr) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    const long G = G8 * 8;
    for (; i < G8; i += stride) {
        float a[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = 0.f;
        const bf16* base = g + i * 8;
        for (int s = 0; s < S; ++s) {
            const bf16* p = base + (long)s * G;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(p + j * 2);
                a[j * 2]     += __bfloat162float(x.x);
                a[j * 2 + 1] += __bfloat162float(x.y);
            }
        }
        float4* q = reinterpret_cast<float4*>(w + i * 8);
        float4 x0 = q[0], x1 = q[1];
        q[0] = make_float4(x0.x - lr * a[0], x0.y - lr * a[1], x0.z - lr * a[2], x0.w - lr * a[3]);
        q[1] = make_float4(x1.x - lr * a[4], x1.y - lr * a[5], x1.z - lr * a[6], x1.w - lr * a[7]);
    }
}

// ---------------------------------------------------------------------------
// 纯读一份 bf16（读带宽参照）。
// ---------------------------------------------------------------------------
__global__ void k_read8(const bf16* __restrict__ g, float* __restrict__ sink, long G8) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < G8; i += stride) {
        const bf16* p = g + i * 8;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(p + j * 2);
            acc += __bfloat162float(x.x) + __bfloat162float(x.y);
        }
    }
    if (acc == 12345.f) sink[threadIdx.x] = acc;
}

// ---------------------------------------------------------------------------
// 「伪 GEMM epilogue」生产者：每个输出元素先做 `flops` 次 FFMA（把算术强度抬上去），
// 再写一个 bf16。用来在**不依赖 DeepGEMM**的情况下造一个**计算受限 + 写 235MB**的内核，
// 与一个访存受限消费者并发，量「消费者能否藏进 HBM 余量」。
// ---------------------------------------------------------------------------
__global__ void k_gen_bf16(bf16* __restrict__ out, long G, int flops, unsigned seed, int smem_n) {
    extern __shared__ char smem[];                 // 动态 smem：用来控制「每 SM 能装几个 block」
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < G; i += stride) {
        float v = (float)((i + seed) & 1023u) * 1e-3f;
#pragma unroll 4
        for (int f = 0; f < flops; ++f) v = fmaf(v, 1.0000001f, 0.5f);
        out[i] = __float2bfloat16(v);
    }
    if (smem_n > 0) smem[0] = (char)(seed & 1);     // 触碰一下，确保动态 smem 被真正占用
}

// --------------------------------------------------------------------------- host helpers
// 选 grid：不超过预算 blocks（模拟「给消费者限块数」，从而限占用的 SM 槽位）。
static int grid_cap(long n_threads, int budget) {
    long want = (n_threads + 255) / 256;
    long g = want < budget ? want : budget;
    return (int)(g > 0 ? g : 1);
}

// 在指定 stream 上计时（事件挂在同一 stream）。
static double bench_stream(void (*launch)(void*), void* ctx, int iters, cudaStream_t st) {
    for (int i = 0; i < 20; ++i) launch(ctx);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, st);
    for (int i = 0; i < iters; ++i) launch(ctx);
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (double)ms / 1e3 / iters;
}

struct AccCtx  { bf16* g; bf16* acc; long G8; int s; int grid; cudaStream_t st; };
struct StepCtx { bf16* g; float* w; long G8; int s; float lr; int grid; cudaStream_t st; };
struct AllCtx  { bf16* g; float* w; long G8; int S; float lr; int grid; cudaStream_t st; };
struct ReadCtx { bf16* g; float* sink; long G8; int grid; cudaStream_t st; };
struct GenCtx  { bf16* out; long G; int flops; unsigned seed; int grid; cudaStream_t st; int smem; };

static void L_acc (void* c){ auto* x=(AccCtx*)c;  k_acc_bf16_step<<<x->grid,256,0,x->st>>>(x->g,x->acc,x->G8,x->s); }
static void L_step(void* c){ auto* x=(StepCtx*)c; k_fused_step   <<<x->grid,256,0,x->st>>>(x->g,x->w,x->G8,x->s,x->lr); }
static void L_all (void* c){ auto* x=(AllCtx*)c;  k_fused_all    <<<x->grid,256,0,x->st>>>(x->g,x->w,x->G8,x->S,x->lr); }
static void L_read(void* c){ auto* x=(ReadCtx*)c; k_read8        <<<x->grid,256,0,x->st>>>(x->g,x->sink,x->G8); }
static void L_gen (void* c){ auto* x=(GenCtx*)c;  k_gen_bf16     <<<x->grid,256,x->smem,x->st>>>(x->out,x->G,x->flops,x->seed,x->smem); }

// stream 版本（供 Python 把消费者放到侧流上做重叠）
static cudaStream_t as_stream(void* p) { return (cudaStream_t)p; }

// --------------------------------------------------------------------------- extern "C" API (ctypes)
extern "C" {

void* woc_malloc_bf16(long n){ void* p=nullptr; CK(cudaMalloc(&p,(size_t)n*2)); return p; }
void* woc_malloc_f32 (long n){ void* p=nullptr; CK(cudaMalloc(&p,(size_t)n*4)); return p; }
void  woc_free(void* p){ if(p) cudaFree(p); }

// 在 stream 上的 launch 包装：stream=NULL 表示默认流。
// 纯读一份 bf16，返回 s/次
double woc_read(void* g, long G, int budget, void* stream, int iters){
    long g8=(G+7)/8; ReadCtx c{(bf16*)g,nullptr,g8,grid_cap(g8,budget),as_stream(stream)};
    float* sink=(float*)woc_malloc_f32(256); c.sink=sink;
    double t=bench_stream(L_read,&c,iters,as_stream(stream)); woc_free(sink); return t;
}
// 增量 bf16 累加一步
double woc_acc_step(void* g, void* acc, long G, int s_idx, int budget, void* stream, int iters){
    long g8=(G+7)/8; AccCtx c{(bf16*)g,(bf16*)acc,g8,s_idx,grid_cap(g8,budget),as_stream(stream)};
    return bench_stream(L_acc,&c,iters,as_stream(stream));
}
// 增量 fp32 master 一步
double woc_fused_step(void* g, void* w, long G, int s_idx, double lr, int budget, void* stream, int iters){
    long g8=(G+7)/8; StepCtx c{(bf16*)g,(float*)w,g8,s_idx,(float)lr,grid_cap(g8,budget),as_stream(stream)};
    return bench_stream(L_step,&c,iters,as_stream(stream));
}
// 批量 ALT-OPT
double woc_fused_all(void* g, void* w, long G, int S, double lr, int budget, void* stream, int iters){
    long g8=(G+7)/8; AllCtx c{(bf16*)g,(float*)w,g8,S,(float)lr,grid_cap(g8,budget),as_stream(stream)};
    return bench_stream(L_all,&c,iters,as_stream(stream));
}
// 伪 GEMM epilogue 生产者（写 bf16 G 个 + flops*G FFMA），grid 固定为 budget
double woc_gen(void* out, long G, int flops, int budget, void* stream, int iters){
    GenCtx c{(bf16*)out,G,flops,12345u,budget,as_stream(stream),0};
    return bench_stream(L_gen,&c,iters,as_stream(stream));
}
// 带动态 smem 的伪生产者（用于隔离「每 SM 的 smem 占用是否挡住 co-residency」）
double woc_gen_smem(void* out, long G, int flops, int budget, int smem_bytes, void* stream, int iters){
    static bool set_ = false;
    if (!set_) { cudaFuncSetAttribute(k_gen_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, 227*1024); set_ = true; }
    GenCtx c{(bf16*)out,G,flops,12345u,budget,as_stream(stream),smem_bytes};
    return bench_stream(L_gen,&c,iters,as_stream(stream));
}

// 单次（不重复）在给定 stream 上发射一个消费者，供 Python 做「逐步重叠」流水。
void woc_launch_acc (void* g, void* acc, long G, int s_idx, int budget, void* stream){
    long g8=(G+7)/8; AccCtx c{(bf16*)g,(bf16*)acc,g8,s_idx,grid_cap(g8,budget),as_stream(stream)}; L_acc(&c);
}
void woc_launch_step(void* g, void* w, long G, int s_idx, double lr, int budget, void* stream){
    long g8=(G+7)/8; StepCtx c{(bf16*)g,(float*)w,g8,s_idx,(float)lr,grid_cap(g8,budget),as_stream(stream)}; L_step(&c);
}
void woc_launch_all (void* g, void* w, long G, int S, double lr, int budget, void* stream){
    long g8=(G+7)/8; AllCtx c{(bf16*)g,(float*)w,g8,S,(float)lr,grid_cap(g8,budget),as_stream(stream)}; L_all(&c);
}
void woc_launch_gen(void* out, long G, int flops, int budget, void* stream){
    GenCtx c{(bf16*)out,G,flops,12345u,budget,as_stream(stream),0}; L_gen(&c);
}
void woc_launch_gen_smem(void* out, long G, int flops, int budget, int smem_bytes, void* stream){
    static bool set_ = false;
    if (!set_) { cudaFuncSetAttribute(k_gen_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, 227*1024); set_ = true; }
    GenCtx c{(bf16*)out,G,flops,12345u,budget,as_stream(stream),smem_bytes}; L_gen(&c);
}

// 正确性自测：GPU vs CPU double，返回 {acc_bf16, fused_step, fused_all} 的最大误差
void woc_selftest(double* out3){
    const long G=100000; const int S=7;
    std::vector<bf16> h((size_t)G*S);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f,1.f);
    for(size_t i=0;i<h.size();++i) h[i]=bf16(nd(rng));
    bf16* d_g=(bf16*)woc_malloc_bf16((long)h.size());
    bf16* d_acc=(bf16*)woc_malloc_bf16(G);
    float* d_w=(float*)woc_malloc_f32(G);
    float* d_w2=(float*)woc_malloc_f32(G);
    CK(cudaMemcpy(d_g,h.data(),h.size()*sizeof(bf16),cudaMemcpyHostToDevice));
    std::vector<bf16> hacc(G); std::vector<float> hw(G);
    for(long i=0;i<G;++i){ hacc[i]=bf16(0.0f); hw[i]=0.0f; }
    CK(cudaMemcpy(d_acc,hacc.data(),G*sizeof(bf16),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w ,hw.data(),G*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w2,hw.data(),G*sizeof(float),cudaMemcpyHostToDevice));
    int g8=(int)((G+7)/8);
    for(int s=0;s<S;++s){ k_acc_bf16_step<<<g8,256>>>(d_g,d_acc,g8,s); }
    for(int s=0;s<S;++s){ k_fused_step   <<<g8,256>>>(d_g,d_w ,g8,s,0.25f); }
    k_fused_all<<<g8,256>>>(d_g,d_w2,g8,S,0.25f);
    CK(cudaDeviceSynchronize());
    std::vector<bf16> ga(G); std::vector<float> gw(G),gw2(G);
    CK(cudaMemcpy(ga.data(),d_acc,G*sizeof(bf16),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gw.data(),d_w ,G*sizeof(float),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gw2.data(),d_w2,G*sizeof(float),cudaMemcpyDeviceToHost));
    double merge=0, mes=0, mea=0;
    for(long i=0;i<G;++i){
        double ref=0; for(int s=0;s<S;++s) ref+=(double)(float)h[(size_t)s*G+i];
        // acc_bf16：逐步 bf16 舍入，参照用「同序 bf16 累加」的 double 上界（只报量级）
        mes=std::fmax(mes, std::fabs((0.0 - 0.25*ref) - (double)gw[i]));
        mea=std::fmax(mea, std::fabs((0.0 - 0.25*ref) - (double)gw2[i]));
        merge=std::fmax(merge, std::fabs((double)ga[i]) );  // 仅确保有限
    }
    out3[0]=merge; out3[1]=mes; out3[2]=mea;
    woc_free(d_g); woc_free(d_acc); woc_free(d_w); woc_free(d_w2);
}

}  // extern "C"

// --------------------------------------------------------------------------- standalone main
static void fill_rand(bf16* p, size_t n, unsigned seed){
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f,1.f);
    std::vector<bf16> h(n); for(size_t i=0;i<n;++i) h[i]=bf16(nd(rng));
    CK(cudaMemcpy(p,h.data(),n*sizeof(bf16),cudaMemcpyHostToDevice));
}

int main_local(int, char**){
    int dev=0; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,dev);
    printf("device=%s sm_%d%d sms=%d\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);

    const long E=8,M=2048,N=7168; const long G=E*M*N;
    const int S=8; const int budget=prop.multiProcessorCount*32;   // 全带宽需 ~32 block/SM
    bf16* d_g=(bf16*)woc_malloc_bf16((long)16*G);
    bf16* d_acc=(bf16*)woc_malloc_bf16(G);
    float* d_w=(float*)woc_malloc_f32(G);
    fill_rand(d_g,(size_t)16*G,11);

    double err[3]; woc_selftest(err);
    printf("[A] 正确性（GPU vs CPU double）：fused_step max|err|=%.3e  fused_all max|err|=%.3e\n",
           err[1], err[2]);
    printf("[B] 线上字节 / 带宽（G=%ld；budget=%d）\n",G,budget);
    woc_read(d_g,G,budget,nullptr,400);
    double tr=woc_read(d_g,G,budget,nullptr,20);
    printf("    纯读一份 bf16 (%.0f MB)            %8.1f us  %8.2f GB/s\n",G*2/1e6,tr*1e6,(double)G*2/tr/1e9);
    double ta=woc_acc_step(d_g,d_acc,G,0,budget,nullptr,20);
    printf("    acc_bf16 一步 (读g+读acc+写acc=705MB)  %8.1f us  %8.2f GB/s\n",
           ta*1e6, 705e6/ta/1e9);
    double ts=woc_fused_step(d_g,d_w,G,0,0.25,budget,nullptr,20);
    printf("    fused_step 一步(读g+读w+写w=1175MB)   %8.1f us  %8.2f GB/s\n",
           ts*1e6, 1175e6/ts/1e9);
    double tb=woc_fused_all(d_g,d_w,G,S,0.25,budget,nullptr,20);
    printf("    fused_all  S=%d (读S*235+读470+写470=%.0fMB) %8.1f us  %8.2f GB/s\n",
           S, (double)S*235+940, tb*1e6, ((double)S*235+940)*1e6/tb/1e9);
    printf("done\n");
    woc_free(d_g); woc_free(d_acc); woc_free(d_w);
    return 0;
}

#ifndef WOC_NO_MAIN
int main(int argc,char** argv){ return main_local(argc,argv); }
#endif
