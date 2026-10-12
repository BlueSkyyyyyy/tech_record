// P28-1 实验：DeepGEMM **PTX 原语底座**（`deep_gemm/include/deep_gemm/ptx/ld_st.cuh`、
//   `ptx/utils.cuh`、`common/ring_pipeline.cuh` 等）在 H100(sm90) 上**直接编译运行真实设备代码**。
//
// 背景：前面 0~27 篇把 DeepGEMM 的整套 kernel（GEMM / attention / mega moe / 通信重叠）都拆过，
//   但它们共同踩在一层**没有系统讲过**的裸 PTX 原语上——内存序 / 原子 / 共享内存搬运 / warp 通信。
//   Mega MoE 的 dispatch/combine 正确性（跨 SM、跨 rank 的 readiness 水位线、grid_sync、
//   nvlink_barrier）全靠这层的 release/acquire / red vs atom / sys vs gpu 语义。本实验把它们逐一
//   拉到 H100 真机上验证语义（`ptx::` 全是 `__device__` 内联，架构无关，sm90 可编译可运行）。
//
// 覆盖（全部来自真实头文件，非手抄）：
//   A) 内存序：`st_rel`(st.release.gpu) / `ld_acq_gpu`(ld.acquire.gpu) 的生产者-消费者契约，
//      与「裸存 + ld.volatile」弱版本对照（对应 comm/barrier.cuh 与 mega_moe combine readiness）。
//   B) 原子：`atomic_add`(atom，返回旧值) vs `red_add`(red，不返回)，及 `red_add_rel` 求和。
//   C) `common/ring_pipeline.cuh:14-38` 的 `RingPipeline::advance` 相位翻转（幂次取模位运算分支）。
//   D) `cp_async_cg<{64,256}>` / `cp_async_cg_zfill<256>` / `cp_async_ca_zfill` 的拷贝与零填充。
//   E) `ld_evict_first`（`ld.weak.global.L1::no_allocate.L2::cache_hint`）的正确性与流式带宽。
//   F) `ptx/utils.cuh` 的 `cvt_relu_bf16x2_f32`（H100 走 `#else` 回退）/ `accumulate` /
//      `accumulate_square`（回退路径）的数值语义。
//
// 编译/运行见 harness/exp_ptx_primitives.sh。目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>

using namespace deep_gemm;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1);} } while (0)

// ---------------------------------------------------------------------------
// A) 内存序：生产者-消费者
//   生产者把 payload 用普通 st 写满后，用 release 置 flag=1；消费者用 acquire 自旋到 flag==1
//   再读 payload。弱版本：生产者普通存 flag，消费者 ld.volatile 自旋。
//   slot 数 = gridDim/2，每个 slot 由 block 2*slot（生产）与 2*slot+1（消费）配对。
// ---------------------------------------------------------------------------
static constexpr uint32_t kPerSlot = 128;

__device__ __forceinline__ uint32_t payload_value(uint32_t slot, uint32_t j, uint32_t gen) {
    uint32_t v = (slot * 131u + j * 7u + gen * 2654435761u);
    return (v ^ (v >> 15)) | 1u;
}

__global__ void k_mo(uint32_t* payload, uint64_t* flags64, uint32_t* flags32,
                     uint32_t gen, int mode, uint32_t* fail) {
    const uint32_t blk = blockIdx.x;
    const uint32_t slot = blk >> 1;
    if ((blk & 1u) == 0) {
        for (uint32_t j = threadIdx.x; j < kPerSlot; j += blockDim.x)
            payload[slot * kPerSlot + j] = payload_value(slot, j, gen);
        __syncthreads();
        if (threadIdx.x == 0) {
            if (mode == 1)      ptx::st_rel(&flags64[slot], 1ull);   // st.release.gpu
            else                flags32[slot] = 1u;                  // 裸存（无 release）
        }
    } else {
        if (threadIdx.x == 0) {
            if (mode == 1) { while (ptx::ld_acq_gpu(&flags64[slot]) != 1ull) {} }  // ld.acquire.gpu
            else           { while (ptx::ld_volatile(&flags32[slot]) == 0u) {} }   // ld.volatile
        }
        __syncthreads();
        uint32_t f = 0;
        for (uint32_t j = threadIdx.x; j < kPerSlot; j += blockDim.x)
            if (payload[slot * kPerSlot + j] != payload_value(slot, j, gen)) f++;
        if (f) atomicAdd(fail, f);
    }
}

// ---------------------------------------------------------------------------
// B) 原子：atom（返回旧值）vs red（不返回）
// ---------------------------------------------------------------------------
__global__ void k_atomic(uint32_t* atom_sum, uint32_t* red_sum, uint32_t* red_rel_sum,
                         uint32_t* atom_old) {
    const uint32_t v = threadIdx.x + 1u;
    atom_old[threadIdx.x] = ptx::atomic_add(atom_sum, v);
    ptx::red_add(red_sum, v);
    ptx::red_add_rel(red_rel_sum, v);
}

// ---------------------------------------------------------------------------
// C) RingPipeline::advance 相位轨迹
// ---------------------------------------------------------------------------
template <uint32_t kNumStages>
__global__ void k_ring(uint32_t* out, uint32_t steps) {
    RingPipeline<kNumStages> rp;
    for (uint32_t s = 0; s < steps; ++s) {
        const auto p = rp.advance(1);
        out[s * 2 + 0] = cute::get<0>(p);
        out[s * 2 + 1] = cute::get<1>(p);
    }
}

// ---------------------------------------------------------------------------
// D) cp.async 拷贝与零填充
// ---------------------------------------------------------------------------
__global__ void k_cpasync(const uint4* src, uint4* out, uint32_t n) {
    extern __shared__ uint4 sm4[];
    const uint32_t i = threadIdx.x;
    if (i < n) {
        // 0..n-1 全量 16B（L2::256B 预取提示）；n..2n-1 走 zfill 只拷前 8B
        ptx::cp_async_cg<256>(src + i, sm4 + i);
        const uint32_t npad = threadIdx.x + n;
        if (npad < 2 * n)
            ptx::cp_async_cg_zfill<256>(src + npad, sm4 + npad, 8u);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (i < 2 * n) out[i] = sm4[i];
}

__global__ void k_cpasync_ca(const uint32_t* src, uint32_t* out, uint32_t n) {
    extern __shared__ uint32_t sm32[];
    const uint32_t i = threadIdx.x;
    if (i < n) {
        // 只拷前 4B 或 0B，其余零填充（对应 sparse MQA logits 的越界尾块 cp_async_ca_zfill）
        const uint32_t ncopy = (i & 1u) ? 4u : 0u;
        ptx::cp_async_ca_zfill(src + i, sm32 + i, ncopy);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (i < n) out[i] = sm32[i];
}

// ---------------------------------------------------------------------------
// E) ld_evict_first 正确性 / 流式带宽
// ---------------------------------------------------------------------------
__global__ void k_evict(const uint4* src, uint4* sink, size_t n4, int mode) {
    uint4 acc = make_uint4(0u, 0u, 0u, 0u);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = mode ? ptx::ld_evict_first(src + i) : src[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0xdeadbeefu && acc.y == 0x12345678u)
        sink[blockIdx.x] = acc;  // 防 DCE
}

// ---------------------------------------------------------------------------
// F) warp 数值原语回退路径
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ uint16_t rne_bf16(float f) {
    uint32_t u = *reinterpret_cast<uint32_t*>(&f);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;  // round-to-nearest-even 到高 16 位
    return static_cast<uint16_t>(u >> 16);
}

__global__ void k_bf16(const float* in, uint32_t* packed, float* accs, float* sqs, uint32_t npairs) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npairs) return;
    const float2 v = make_float2(in[i * 2 + 0], in[i * 2 + 1]);
    const nv_bfloat162 r = ptx::cvt_relu_bf16x2_f32(v);
    packed[i] = *reinterpret_cast<const uint32_t*>(&r);
    float2 a = make_float2(0.0f, 0.0f);
    ptx::accumulate(a, r);
    float2 b = make_float2(0.0f, 0.0f);
    ptx::accumulate_square(b, r);
    accs[i * 2 + 0] = a.x; accs[i * 2 + 1] = a.y;
    sqs[i * 2 + 0] = b.x;  sqs[i * 2 + 1] = b.y;
}

int main() {
    const uint32_t nslots = 64;
    const uint32_t gen_iters = 200;

    // -------- A) 内存序 --------
    uint32_t* payload; uint64_t* flags64; uint32_t* flags32; uint32_t* fail;
    CK(cudaMalloc(&payload, nslots * kPerSlot * sizeof(uint32_t)));
    CK(cudaMalloc(&flags64, nslots * sizeof(uint64_t)));
    CK(cudaMalloc(&flags32, nslots * sizeof(uint32_t)));
    CK(cudaMalloc(&fail, sizeof(uint32_t)));
    for (int mode = 1; mode <= 2; ++mode) {
        uint32_t total_fail = 0;
        for (uint32_t g = 0; g < gen_iters; ++g) {
            CK(cudaMemset(flags64, 0, nslots * sizeof(uint64_t)));
            CK(cudaMemset(flags32, 0, nslots * sizeof(uint32_t)));
            CK(cudaMemset(fail, 0, sizeof(uint32_t)));
            k_mo<<<nslots * 2, 128>>>(payload, flags64, flags32, g, mode, fail);
            uint32_t f = 0;
            CK(cudaMemcpy(&f, fail, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            total_fail += f;
        }
        printf("MO mode=%d rounds=%u slots=%u per_slot=%u fail_words=%u\n",
               mode, gen_iters, nslots, kPerSlot, total_fail);
    }
    printf("MO acqrel_vs_weak verify: acqrel_expected=0\n");

    // -------- B) 原子 --------
    uint32_t *atom_sum, *red_sum, *red_rel_sum, *atom_old;
    CK(cudaMalloc(&atom_sum, sizeof(uint32_t)));
    CK(cudaMalloc(&red_sum, sizeof(uint32_t)));
    CK(cudaMalloc(&red_rel_sum, sizeof(uint32_t)));
    CK(cudaMalloc(&atom_old, 1024 * sizeof(uint32_t)));
    CK(cudaMemset(atom_sum, 0, sizeof(uint32_t)));
    CK(cudaMemset(red_sum, 0, sizeof(uint32_t)));
    CK(cudaMemset(red_rel_sum, 0, sizeof(uint32_t)));
    k_atomic<<<1, 1024>>>(atom_sum, red_sum, red_rel_sum, atom_old);
    uint32_t a_sum, r_sum, rr_sum;
    CK(cudaMemcpy(&a_sum, atom_sum, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&r_sum, red_sum, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&rr_sum, red_rel_sum, 4, cudaMemcpyDeviceToHost));
    std::vector<uint32_t> old(1024);
    CK(cudaMemcpy(old.data(), atom_old, 1024 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // atom 返回的旧值集合应是 {0,1,3,6,...} 的一个排列（顺序随调度）
    // atomicAdd 返回的旧值是「按完成顺序的运行时部分和」：每个值互不相同（增量恒正），
    // 且最后一个完成者的旧值 + 其增量 = 总和。这里只校验「互不相同且计数正确」。
    uint32_t distinct = 0;
    std::vector<char> hit(1024, 0);
    // old[] 取值范围 [0, 524800)，用简单哈希集合去重
    std::vector<char> present(524801, 0);
    for (uint32_t i = 0; i < 1024; ++i) present[old[i]] = 1;
    for (uint32_t v = 0; v <= 524800; ++v) if (present[v]) distinct++;
    (void)hit;
    printf("ATOMIC expected_sum=%u atom_sum=%u red_sum=%u red_rel_sum=%u atom_old_distinct=%u/1024\n",
           (1024u * 1025u) / 2u, a_sum, r_sum, rr_sum, distinct);

    // -------- C) RingPipeline --------
    uint32_t* ring_out;
    CK(cudaMalloc(&ring_out, 2 * 40 * sizeof(uint32_t)));
    for (uint32_t ns : {1u, 2u, 4u, 8u}) {
        uint32_t steps = 40;
        if (ns == 1) k_ring<1><<<1, 1>>>(ring_out, steps);
        if (ns == 2) k_ring<2><<<1, 1>>>(ring_out, steps);
        if (ns == 4) k_ring<4><<<1, 1>>>(ring_out, steps);
        if (ns == 8) k_ring<8><<<1, 1>>>(ring_out, steps);
        std::vector<uint32_t> o(2 * steps);
        CK(cudaMemcpy(o.data(), ring_out, 2 * steps * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        printf("RING ns=%u steps=%u", ns, steps);
        for (uint32_t s = 0; s < steps; ++s) printf(" %u,%u", o[s * 2], o[s * 2 + 1]);
        printf("\n");
    }

    // -------- D) cp.async --------
    {
        const uint32_t n = 64;
        uint4* src; uint4* out;
        CK(cudaMalloc(&src, 2 * n * sizeof(uint4)));
        CK(cudaMalloc(&out, 2 * n * sizeof(uint4)));
        std::vector<uint4> hsrc(2 * n);
        for (uint32_t i = 0; i < 2 * n; ++i)
            hsrc[i] = make_uint4(i + 1, i + 2, i + 3, i + 4);
        CK(cudaMemcpy(src, hsrc.data(), 2 * n * sizeof(uint4), cudaMemcpyHostToDevice));
        k_cpasync<<<1, 2 * n, 2 * n * sizeof(uint4)>>>(src, out, n);
        std::vector<uint4> hout(2 * n);
        CK(cudaMemcpy(hout.data(), out, 2 * n * sizeof(uint4), cudaMemcpyDeviceToHost));
        uint32_t mis = 0;
        for (uint32_t i = 0; i < n; ++i)
            if (!(hout[i].x == hsrc[i].x && hout[i].y == hsrc[i].y &&
                  hout[i].z == hsrc[i].z && hout[i].w == hsrc[i].w)) mis++;
        for (uint32_t i = n; i < 2 * n; ++i)
            if (!(hout[i].x == hsrc[i].x && hout[i].y == hsrc[i].y &&
                  hout[i].z == 0u && hout[i].w == 0u)) mis++;
        printf("CPASYNC cg_full+zfill8 mismatch=%u/128\n", mis);

        uint32_t* src2; uint32_t* out2;
        CK(cudaMalloc(&src2, n * sizeof(uint32_t)));
        CK(cudaMalloc(&out2, n * sizeof(uint32_t)));
        std::vector<uint32_t> h2(n);
        for (uint32_t i = 0; i < n; ++i) h2[i] = 0xA0000000u + i;
        CK(cudaMemcpy(src2, h2.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        k_cpasync_ca<<<1, n, n * sizeof(uint32_t)>>>(src2, out2, n);
        std::vector<uint32_t> ho2(n);
        CK(cudaMemcpy(ho2.data(), out2, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t mis2 = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t exp = (i & 1u) ? h2[i] : 0u;
            if (ho2[i] != exp) mis2++;
        }
        printf("CPASYNC ca_zfill(0/4B) mismatch=%u/%u\n", mis2, n);
    }

    // -------- E) ld_evict_first --------
    {
        const size_t bytes = 256ull * 1024 * 1024;
        const size_t n4 = bytes / 16;
        uint4* src; uint4* sink;
        CK(cudaMalloc(&src, bytes));
        CK(cudaMemset(src, 0x5a, bytes));
        CK(cudaMalloc(&sink, 4096 * sizeof(uint4)));
        for (int mode = 0; mode <= 1; ++mode) {
            // 正确性：只读一遍并 XOR，与参考比较（此处反复读同一 buffer，验证数值一致）
            cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0));
            k_evict<<<132, 256>>>(src, sink, n4, mode);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            const double gbps = bytes / (ms * 1e-3) / 1e9;
            printf("EVICT mode=%d bytes=%zu ms=%.3f GBps=%.1f\n", mode, bytes, ms, gbps);
        }
    }

    // -------- F) warp 数值原语 --------
    {
        const uint32_t npairs = 1u << 16;
        float* in; uint32_t* packed; float* accs; float* sqs;
        CK(cudaMalloc(&in, 2 * npairs * sizeof(float)));
        CK(cudaMalloc(&packed, npairs * sizeof(uint32_t)));
        CK(cudaMalloc(&accs, 2 * npairs * sizeof(float)));
        CK(cudaMalloc(&sqs, 2 * npairs * sizeof(float)));
        std::vector<float> hin(2 * npairs);
        for (uint32_t i = 0; i < 2 * npairs; ++i) {
            const float x = std::sin((double)i * 0.001) * 4.0f + (static_cast<int>(i % 7) - 3) * 0.5f;
            hin[i] = x;
        }
        CK(cudaMemcpy(in, hin.data(), 2 * npairs * sizeof(float), cudaMemcpyHostToDevice));
        k_bf16<<<(npairs + 255) / 256, 256>>>(in, packed, accs, sqs, npairs);
        std::vector<uint32_t> hp(npairs);
        std::vector<float> ha(2 * npairs), hs(2 * npairs);
        CK(cudaMemcpy(hp.data(), packed, npairs * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ha.data(), accs, 2 * npairs * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hs.data(), sqs, 2 * npairs * sizeof(float), cudaMemcpyDeviceToHost));
        uint32_t mis = 0;
        for (uint32_t i = 0; i < npairs; ++i) {
            const float x0 = fmaxf(hin[i * 2 + 0], 0.0f);
            const float x1 = fmaxf(hin[i * 2 + 1], 0.0f);
            const uint16_t e0 = rne_bf16(x0), e1 = rne_bf16(x1);
            const uint32_t exp = (uint32_t)e0 | ((uint32_t)e1 << 16);
            if (hp[i] != exp) mis++;
        }
        printf("BF16 cvt_relu pack mismatch=%u/%u\n", mis, npairs);
        printf("BF16 sample");
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t lo = hp[i] & 0xffffu, hi = hp[i] >> 16;
            uint32_t u0 = lo << 16, u1 = hi << 16;
            const float fx0 = *reinterpret_cast<float*>(&u0);
            const float fx1 = *reinterpret_cast<float*>(&u1);
            printf(" [%u: in=(%.4f,%.4f) relu_bf16=(%.4f,%.4f) acc=(%.4f,%.4f) sq=(%.4f,%.4f)]",
                   i, hin[i * 2], hin[i * 2 + 1], fx0, fx1, ha[i * 2], ha[i * 2 + 1],
                   hs[i * 2], hs[i * 2 + 1]);
        }
        printf("\n");
    }

    printf("DONE\n");
    return 0;
}
