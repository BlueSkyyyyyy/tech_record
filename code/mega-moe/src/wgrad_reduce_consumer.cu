// P137-1 实验：MoE wgrad 的「fp32 物化 / 归约税」回收——三种消费协议的**下界内核**。
//
// 背景：第 146 篇（P136-1）把「wgrad 换 bf16 输出」落地：GEMM 本身快 **2.27×**
//   （真实 `k_grouped` 1D1D fp32/累积 500.5us vs `E×` 稠密 1D2D bf16 220.8us），
//   但端到端在 `S≈8–16` 封顶 **~1.55×**——因为「把每步 fresh 的 bf16 梯度再 Σ 成一份 fp32
//   参数梯度」要付一笔**归约税**（`S=8` 时 838.6us，读 S 份 bf16 + 写一份 fp32）。
//   它给出的结论是：**只有让优化器直接吃 bf16 梯度、不再物化 fp32，2.2× 才拿得回来**——
//   但**没有量**「不物化 fp32」到底能拉回多少。
//
// 本篇补上这个量。三种「把 S 份 per-micro-batch 梯度交给优化器」的协议，按**线上字节**排：
//   REF      : 每步 1D1D 累积（读 C fp32 4B + 写 D fp32 4B = 8B/elem/步），最后优化器读 fp32 4B
//   ALT-R    : 每步 1D2D bf16 写 2B；末次 reduce 读 S·2B + 写 4B；优化器读 4B
//   ALT-OPT  : 每步 1D2D bf16 写 2B；**融合**内核读 S·2B + 读 w 4B + 写 w 4B（**无 fp32 中间**）
// 差异集中在「reduce 写 fp32 那一下 + 优化器再读回」（8B/elem）——这正是「物化税」。
//
// 本文件给的是这条路径能达到的**下界**：手写、读齐 S 份 bf16 就地求和的向量化内核，
// 与 `torch.sum` 对照，证明「物化税」是**带宽硬账**、不是实现问题；并给出「融合进优化器」的
// 真实带宽。GEMM 段（真实 DeepGEMM kernel）与协议组装在 harness/exp_wgrad_reduce_consumer.py。
// Mega MoE 主体是 SM100 专属；本文件全部架构无关（只有 bf16 ld/st + 浮点），H100(sm90) 真机可跑。
//
// 编译/运行见 harness/exp_wgrad_reduce_consumer.sh（编成 .so 由 Python 经 ctypes 驱动，
// 也可单独 `nvcc ... && ./a.out` 跑自带的 [A]/[B] 报告）。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。

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
// 内核 1：reduce over S —— bf16[S][G] -> fp32[G]。
//   8 元素/线程（128-bit 载入 16B），每线程沿 S 循环累加在寄存器里。
//   线上字节 = 读 S·G·2B + 写 G·4B —— 这就是「末次归约」的**下界**。
// ---------------------------------------------------------------------------
__global__ void k_reduce8(const bf16* __restrict__ g, float* __restrict__ out,
                          long G8, int S) {
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
        float4* q = reinterpret_cast<float4*>(out + i * 8);   // 2×128-bit 向量化写
        q[0] = make_float4(a[0], a[1], a[2], a[3]);
        q[1] = make_float4(a[4], a[5], a[6], a[7]);
    }
}

// ---------------------------------------------------------------------------
// 内核 2：atomicAdd 变体 —— 先清零 out，再让每一步 s 用 atomicAdd 累加。
//   线上字节 = 读 S·G·2B + **S 次 RMW G·4B**（每次 atomicAdd 读-改-写）。
//   用来证伪「把 reduce 拆成 S 个 atomicAdd 就更快」——RMW 反而多付。
// ---------------------------------------------------------------------------
__global__ void k_reduce_atomic(const bf16* __restrict__ g, float* __restrict__ out,
                                long G, int s_idx) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    const bf16* p = g + (long)s_idx * G;
    for (; i < G; i += stride) atomicAdd(out + i, __bfloat162float(p[i]));
}

// ---------------------------------------------------------------------------
// 内核 3：融合优化器 —— w = w - lr * Σ_s g_s。
//   读 S·G·2B(bf16 梯度) + 读 G·4B(w) + 写 G·4B(w)，**不落地任何 fp32 梯度**。
//   这就是「优化器直接吃 bf16 梯度」的真实字节账（SGD 形态；AdamW 只多各自一份状态）。
// ---------------------------------------------------------------------------
__global__ void k_fused_sgd8(const bf16* __restrict__ g, float* __restrict__ w,
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
        float4* q = reinterpret_cast<float4*>(w + i * 8);     // 2×128-bit 向量化读改写
        float4 x0 = q[0], x1 = q[1];
        q[0] = make_float4(x0.x - lr * a[0], x0.y - lr * a[1], x0.z - lr * a[2], x0.w - lr * a[3]);
        q[1] = make_float4(x1.x - lr * a[4], x1.y - lr * a[5], x1.z - lr * a[6], x1.w - lr * a[7]);
    }
}

// ---------------------------------------------------------------------------
// 内核 0：纯读带宽（读一份 bf16，不做归约）——为上面几个数给「峰值读」参照。
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
    if (acc == 12345.f) sink[threadIdx.x] = acc;  // 防优化
}

// --------------------------------------------------------------------------- host helpers
static int grid_for(long n_threads) {
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    long want = (n_threads + 255) / 256;
    long cap = (long)sms * 32;
    long g = want < cap ? want : cap;
    return (int)(g > 0 ? g : 1);
}

static double now_bench(void (*launch)(void*), void* ctx, int iters) {
    for (int i = 0; i < 20; ++i) launch(ctx);   // 足够 warmup：盖过 H100 空闲降频爬坡
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) launch(ctx);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (double)ms / 1e3 / iters;  // s/iter
}

struct ReduceCtx { bf16* g; float* out; long G8; int S; int g8; };
static void launch_reduce(void* c) { auto* x = (ReduceCtx*)c; k_reduce8<<<x->g8, 256>>>(x->g, x->out, x->G8, x->S); }
struct AtomicCtx { bf16* g; float* out; long G; int S; int g1; float* zero; };
static void launch_atomic(void* c) {
    auto* x = (AtomicCtx*)c;
    CK(cudaMemcpyAsync(x->out, x->zero, x->G * sizeof(float), cudaMemcpyHostToDevice));
    for (int s = 0; s < x->S; ++s) k_reduce_atomic<<<x->g1, 256>>>(x->g, x->out, x->G, s);
}
struct FusedCtx { bf16* g; float* w; long G8; int S; float lr; int g8; };
static void launch_fused(void* c) { auto* x = (FusedCtx*)c; k_fused_sgd8<<<x->g8, 256>>>(x->g, x->w, x->G8, x->S, x->lr); }
struct ReadCtx { bf16* g; float* sink; long G8s; int g8; };
static void launch_read(void* c) { auto* x = (ReadCtx*)c; k_read8<<<x->g8, 256>>>(x->g, x->sink, x->G8s); }

// --------------------------------------------------------------------------- extern "C" API (ctypes)
extern "C" {

// 分配 bf16 缓冲（device）
void* wrc_malloc_bf16(long n) { void* p = nullptr; CK(cudaMalloc(&p, (size_t)n * 2)); return p; }
void* wrc_malloc_f32(long n) { void* p = nullptr; CK(cudaMalloc(&p, (size_t)n * 4)); return p; }
void  wrc_free(void* p) { if (p) cudaFree(p); }

// 纯读一份 bf16 [G]，返回秒/次
double wrc_read(void* g, long G, int iters) {
    long g8 = (G + 7) / 8; int gr = grid_for(g8);
    float* sink = (float*)wrc_malloc_f32(256);
    ReadCtx c{(bf16*)g, sink, g8, gr};
    double t = now_bench(launch_read, &c, iters);
    wrc_free(sink);
    return t;
}

// reduce bf16[S][G] -> fp32[G]，返回秒/次
double wrc_reduce(void* g, void* out, long G, int S, int iters) {
    long g8 = (G + 7) / 8; int gr = grid_for(g8);
    ReduceCtx c{(bf16*)g, (float*)out, g8, S, gr};
    return now_bench(launch_reduce, &c, iters);
}

// atomicAdd 变体（含清零），返回秒/次
double wrc_atomic(void* g, void* out, long G, int S, int iters) {
    int gr = grid_for(G);
    float* zero = (float*)malloc((size_t)G * 4); for (long i = 0; i < G; ++i) zero[i] = 0.f;
    AtomicCtx c{(bf16*)g, (float*)out, G, S, gr, zero};
    double t = now_bench(launch_atomic, &c, iters);
    free(zero);
    return t;
}

// 融合优化器 w = w - lr*Σ_s g_s，返回秒/次
double wrc_fused(void* g, void* w, long G, int S, double lr, int iters) {
    long g8 = (G + 7) / 8; int gr = grid_for(g8);
    FusedCtx c{(bf16*)g, (float*)w, g8, S, (float)lr, gr};
    return now_bench(launch_fused, &c, iters);
}

// 正确性自测（GPU vs CPU double），返回最大误差对 {reduce, fused} 到 out2
void wrc_selftest(double* out2) {
    const long G = 100000;   // 8 的倍数（内核按 G8*8 取址；真实 G=E*M*N 恒可被 8 整除）
    const int S = 7;
    std::vector<bf16> h((size_t)G * S);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t i = 0; i < h.size(); ++i) h[i] = bf16(nd(rng));
    bf16* d_g = (bf16*)wrc_malloc_bf16((long)h.size());
    float* d_out = (float*)wrc_malloc_f32(G);
    float* d_w = (float*)wrc_malloc_f32(G);
    CK(cudaMemcpy(d_g, h.data(), h.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    int g8 = grid_for((G + 7) / 8);
    k_reduce8<<<g8, 256>>>(d_g, d_out, (G + 7) / 8, S);
    std::vector<float> hw(G); for (long i = 0; i < G; ++i) hw[i] = nd(rng);
    CK(cudaMemcpy(d_w, hw.data(), G * sizeof(float), cudaMemcpyHostToDevice));
    k_fused_sgd8<<<g8, 256>>>(d_g, d_w, (G + 7) / 8, S, 0.25f);
    CK(cudaDeviceSynchronize());
    std::vector<float> go(G), gw(G);
    CK(cudaMemcpy(go.data(), d_out, G * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gw.data(), d_w, G * sizeof(float), cudaMemcpyDeviceToHost));
    double me_r = 0, me_f = 0;
    for (long i = 0; i < G; ++i) {
        double ref = 0; for (int s = 0; s < S; ++s) ref += (double)(float)h[(size_t)s * G + i];
        me_r = std::fmax(me_r, std::fabs(ref - (double)go[i]));
        me_f = std::fmax(me_f, std::fabs(((double)hw[i] - 0.25 * ref) - (double)gw[i]));
    }
    out2[0] = me_r; out2[1] = me_f;
    wrc_free(d_g); wrc_free(d_out); wrc_free(d_w);
}

}  // extern "C"

// --------------------------------------------------------------------------- standalone main
static void fill_rand(bf16* p, size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<bf16> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = bf16(nd(rng));
    CK(cudaMemcpy(p, h.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
}

int main_local(int, char**) {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("device=%s sm_%d%d sms=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    printf("[A] 正确性（小案例，GPU vs CPU double 参考）\n");
    double err[2]; wrc_selftest(err);
    printf("    reduce8    max|GPU-CPU_double| = %.3e\n", err[0]);
    printf("    fused_sgd  max|GPU-CPU_double| = %.3e\n", err[1]);
    printf("    -> %s\n", (err[0] < 1e-2 && err[1] < 1e-2) ? "OK（<=fp32 求和舍入+bf16 地板）" : "FAIL");

    const long E = 8, M = 2048, N = 7168;   // DeepSeek 量级 dW2：[E,M,N]
    const long G = E * M * N;
    int S_list[] = {1, 2, 4, 8, 16};
    long Gmax = 16;
    bf16* d_g = (bf16*)wrc_malloc_bf16((long)Gmax * G);
    float* d_out = (float*)wrc_malloc_f32(G);
    float* d_w = (float*)wrc_malloc_f32(G);
    fill_rand(d_g, (size_t)Gmax * G, 11);
    std::vector<float> hw(G, 0.001f); CK(cudaMemcpy(d_w, hw.data(), G * sizeof(float), cudaMemcpyHostToDevice));

    printf("[B] 线上带宽下界（G=E*M*N=%ld 元素；真实 wgrad dW2 形状）\n", G);
    wrc_read(d_g, G, 400);   // 时钟爬坡：H100 空闲降频，先热机再报数
    double tr = wrc_read(d_g, G, 20);
    printf("    纯读 bf16 一份 (%.0f MB)            %8.1f us  %8.2f GB/s\n",
           G * 2 / 1e6, tr * 1e6, (double)G * 2 / tr / 1e9);
    printf("    %-30s %10s %10s %12s\n", "协议 / 核", "时间us", "读MB", "总GB/s");
    for (int i = 0; i < 5; ++i) {
        int S = S_list[i];
        double t = wrc_reduce(d_g, d_out, G, S, 10);
        double rb = (double)S * G * 2, wb = (double)G * 4;
        printf("    custom reduce8 S=%-2d           %10.1f %10.0f %12.2f\n", S, t * 1e6, (rb + wb) / 1e6, (rb + wb) / t / 1e9);
        t = wrc_fused(d_g, d_w, G, S, 0.25, 10);
        rb = (double)S * G * 2 + (double)G * 4; wb = (double)G * 4;
        printf("    custom fused_sgd S=%-2d         %10.1f %10.0f %12.2f\n", S, t * 1e6, (rb + wb) / 1e6, (rb + wb) / t / 1e9);
        if (S == 1) {   // atomic 走 RMW，极慢；只留一个 S 说明趋势
            t = wrc_atomic(d_g, d_out, G, S, 3);
            rb = (double)S * G * 2; wb = (double)S * G * 4;  // RMW
            printf("    custom atomic   S=%-2d         %10.1f %10.0f %12.2f\n", S, t * 1e6, (rb + wb) / 1e6, (rb + wb) / t / 1e9);
        }
    }
    printf("done\n");
    return 0;
}

#ifndef WRC_NO_MAIN
int main(int argc, char** argv) { return main_local(argc, argv); }
#endif
