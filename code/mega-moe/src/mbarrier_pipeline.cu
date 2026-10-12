// P3-2 实验：在 H100（sm90）上复现 Mega MoE 的「warp specialization + mbarrier 多级流水」。
//
// Mega MoE 的 gemm 主循环（deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh）
// 用两组 mbarrier 把生产者（TMA loader warp）和消费者（MMA/epilogue warp）解耦：
//   - full_barriers[stage]:  生产者 arrive(+expect_tx)，消费者 wait(phase)
//   - empty_barriers[stage]: 消费者 arrive，生产者 wait(phase ^ 1)
//   阶段指针与 phase 的翻转逻辑见 kernel 的 advance_pipeline()（:307-314）。
//
// 本机 H100 跑不了 SM100 的 tcgen05/TMA-2SM，但 **mbarrier + cp.async.bulk 1D 的相位语义**
// 与 SM90 完全一致。这里用 1 个 producer warp + 1 个 consumer warp 搭一条 NSTAGES 级流水：
//   producer: 对每个 stage 先 wait empty，再 cp.async.bulk(global -> shared) 并发 expect_tx；
//   consumer: wait full，校验数据，arrive empty。
// 对比 NSTAGES=1（串行，无重叠）与 NSTAGES=4（流水，重叠）的墙钟时间与有效带宽。
//
// 编译：nvcc -arch=sm_90 -O3 -std=c++17 mbarrier_pipeline.cu -o /tmp/mbarrier_pipeline
// 运行：/tmp/mbarrier_pipeline
//
// 注意：本文件仅实验用途，不进入 kernel 实现。

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                              \
    do {                                                                           \
        cudaError_t e = (x);                                                       \
        if (e != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                         cudaGetErrorString(e), __FILE__, __LINE__);               \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                 "r"(smem_addr(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_addr(bar)));
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                 "r"(smem_addr(bar)), "r"(bytes));
}

// 返回 try_wait.parity(phase) 是否立即完成；用于探测 mbarrier 初值相位语义。
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t phase) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred P1;\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
                 "selp.b32 %0, 1, 0, P1;\n\t}"
                 : "=r"(ok) : "r"(smem_addr(bar)), "r"(phase));
    return ok != 0;
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    asm volatile("{\n\t.reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra DONE;\n\t"
                 "bra LAB_WAIT;\n\t"
                 "DONE:\n\t}"
                 :: "r"(smem_addr(bar)), "r"(phase), "r"(0x989680));
}

__device__ __forceinline__ void bulk_load(void* dst_smem, const void* src_gmem,
                                          uint32_t bytes, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(smem_addr(dst_smem)), "l"(src_gmem), "r"(bytes), "r"(smem_addr(bar))
        : "memory");
}

// ---------------------------------------------------------------------------
// 实验 1：探测 mbarrier 初值相位 —— 全新 mbarrier 上 try_wait.parity(0) / (1) 谁立即成功
// ---------------------------------------------------------------------------
__global__ void probe_phase_kernel(int* out) {
    __shared__ alignas(16) uint64_t bar;
    if (threadIdx.x == 0) {
        mbar_init(&bar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // 全新（未 arrive）mbarrier：当前处于第 0 相位。
        const bool w0 = mbar_try_wait(&bar, 0);
        const bool w1 = mbar_try_wait(&bar, 1);
        out[0] = w0 ? 1 : 0;
        out[1] = w1 ? 1 : 0;

        // 一次 arrive 后（完成第 0 相位），再探测。
        mbar_arrive(&bar);
        const bool a0 = mbar_try_wait(&bar, 0);
        const bool a1 = mbar_try_wait(&bar, 1);
        out[2] = a0 ? 1 : 0;
        out[3] = a1 ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// 实验 2：NSTAGES 级 producer/consumer 流水
//   producer (warp 0): wait empty[stage](phase^1) -> bulk_load -> arrive_expect_tx(full[stage])
//   consumer (warp 1): wait full[stage](phase)   -> 校验 -> arrive(empty[stage])
//   advance_pipeline 语义与 kernel:307-314 一致：stage 回绕到 0 时 phase 翻转。
// ---------------------------------------------------------------------------
template <int NSTAGES, int BYTES>
__global__ void pipeline_kernel(const uint32_t* __restrict__ src, int iters, int verify,
                                unsigned long long* __restrict__ out_checksum) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint64_t* full = reinterpret_cast<uint64_t*>(smem);                       // NSTAGES
    uint64_t* empty = full + NSTAGES;                                         // NSTAGES
    uint8_t* buf = smem + 2 * NSTAGES * sizeof(uint64_t);                     // NSTAGES * BYTES

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    if (threadIdx.x == 0) {
        for (int i = 0; i < NSTAGES; ++i) {
            mbar_init(&full[i], 1);
            mbar_init(&empty[i], 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    if (warp == 0) {
        // ---- producer：elect one ----
        if (lane == 0) {
            uint32_t phase = 0;
            for (int it = 0; it < iters; ++it) {
                const int stage = it % NSTAGES;
                // 生产者等「空」，相位与消费者相反（kernel:729 / :783 用 phase ^ 1）
                mbar_wait(&empty[stage], phase ^ 1);
                bulk_load(buf + stage * BYTES, src + (size_t)it * (BYTES / 4), BYTES, &full[stage]);
                mbar_arrive_expect_tx(&full[stage], BYTES);
                if (stage == NSTAGES - 1) phase ^= 1;   // 回绕 -> 翻转
            }
        }
    } else if (warp == 1) {
        // ---- consumer：整 warp 校验 ----
        uint32_t phase = 0;
        unsigned long long local = 0;
        for (int it = 0; it < iters; ++it) {
            const int stage = it % NSTAGES;
            mbar_wait(&full[stage], phase);             // 消费者等「满」（kernel:886）
            const uint32_t* p = reinterpret_cast<const uint32_t*>(buf + stage * BYTES);
            if (verify) {
                // 每个 lane 抽查若干元素；期望值 = it*BYTES/4 + offset
                for (int k = lane; k < BYTES / 4; k += 32) {
                    const uint32_t expect = (uint32_t)(it * (BYTES / 4) + k);
                    if (p[k] != expect) local |= 1ull << 32;
                }
            } else if (lane == 0) {
                local = p[0] == (uint32_t)(it * (BYTES / 4)) ? local : (local | (1ull << 32));
            }
            local += 1;
            __syncwarp();
            // 保证 generic-proxy 读取完成后，才允许 producer 的 async 写入覆盖本 stage
            asm volatile("fence.proxy.async.shared::cta;");
            if (lane == 0) mbar_arrive(&empty[stage]);
            if (stage == NSTAGES - 1) phase ^= 1;
        }
        if (lane == 0) atomicAdd(out_checksum, local);
    }
}

int main() {
    // ---- 实验 1 ----
    {
        int* d_out; int h_out[4] = {0, 0, 0, 0};
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * 4));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int) * 4));
        probe_phase_kernel<<<1, 32>>>(d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(int) * 4, cudaMemcpyDeviceToHost));
        std::printf("[probe] fresh mbarrier: try_wait(parity=0)=%d try_wait(parity=1)=%d\n",
                    h_out[0], h_out[1]);
        std::printf("[probe] after 1 arrive: try_wait(parity=0)=%d try_wait(parity=1)=%d\n",
                    h_out[2], h_out[3]);
        cudaFree(d_out);
    }

    // ---- 实验 2 ----
    constexpr int BYTES = 48 * 1024;   // 每 stage 48 KiB
    constexpr int ITERS = 2048;        // 总共 96 MiB 拷贝
    const int total = ITERS * (BYTES / 4);
    uint32_t* h_src = static_cast<uint32_t*>(std::malloc((size_t)total * 4));
    for (int i = 0; i < total; ++i) h_src[i] = (uint32_t)i;

    uint32_t* d_src; unsigned long long* d_ck;
    CUDA_CHECK(cudaMalloc(&d_src, (size_t)total * 4));
    CUDA_CHECK(cudaMalloc(&d_ck, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, (size_t)total * 4, cudaMemcpyHostToDevice));

    auto run = [&](auto tag, int nstages, int smem_bytes, int verify, int reps) {
        (void)tag;
        if (nstages == 1)
            CUDA_CHECK(cudaFuncSetAttribute(pipeline_kernel<1, BYTES>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        else
            CUDA_CHECK(cudaFuncSetAttribute(pipeline_kernel<4, BYTES>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            CUDA_CHECK(cudaMemset(d_ck, 0, sizeof(unsigned long long)));
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a);
            if (nstages == 1)
                pipeline_kernel<1, BYTES><<<1, 64, smem_bytes>>>(d_src, ITERS, verify, d_ck);
            else
                pipeline_kernel<4, BYTES><<<1, 64, smem_bytes>>>(d_src, ITERS, verify, d_ck);
            CUDA_CHECK(cudaGetLastError());
            cudaEventRecord(b);
            CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0; cudaEventElapsedTime(&ms, a, b);
            if (ms < best) best = ms;
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        unsigned long long ck = 0;
        CUDA_CHECK(cudaMemcpy(&ck, d_ck, sizeof(ck), cudaMemcpyDeviceToHost));
        const double gib = (double)total * 4 / (1024.0 * 1024.0 * 1024.0);
        const bool ok = (ck == (unsigned long long)ITERS);
        std::printf("[pipe NSTAGES=%d verify=%d] best=%.3f ms  eff_bw=%.1f GiB/s  checksum_ok=%s\n",
                    nstages, verify, best, gib / (best / 1e3), ok ? "yes" : "NO");
    };

    // smem: 2*NSTAGES*8 bytes barriers + NSTAGES*BYTES buffers
    const int smem1 = (2 * 1 * 8 + 1 * BYTES + 127) / 128 * 128;
    const int smem4 = (2 * 4 * 8 + 4 * BYTES + 127) / 128 * 128;
    std::printf("[setup] per-stage=%d KiB iters=%d total=%.1f MiB\n",
                BYTES / 1024, ITERS, (double)ITERS * BYTES / (1024.0 * 1024.0));
    run("serial",    1, smem1, 1, 20);
    run("pipelined", 4, smem4, 1, 20);
    run("serial",    1, smem1, 0, 20);
    run("pipelined", 4, smem4, 0, 20);

    std::free(h_src);
    cudaFree(d_src); cudaFree(d_ck);
    return 0;
}
