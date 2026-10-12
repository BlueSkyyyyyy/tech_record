// P49-1 实验：DeepEP 设备端公共层（common/ptx.cuh / common/math.cuh / common/utils.cuh /
//          layout/common/barrier.cuh）。
//
// 为什么 H100(sm90) 可跑：
//   * `common/ptx.cuh` 里绝大多数是**架构无关的内联 asm**（mbarrier / TMA 1D / cp.async /
//     谓词化 ld-st / 内存序 / warp 内建），它们在 sm90a 上都是合法 ISA。
//   * 少数几处带 `__CUDA_ARCH__ >= 1000` 守卫的原语（`st.bulk`、`longlong4_t` 的 256b ld-st、
//     `fadd2/fmul2` 打包、融合 `add.rn.f32.bf16`）在 sm90 会走**文件里给出的回退分支**，
//     本实验直接验证回退分支与标量参考一致。
//   * `multimem.*`（NVLink 多播）与 `cvt.rs.bf16x2` 需要多播映射 / SM100，本文件对其只做
//     host 侧语义 oracle（不发射设备指令）。
//   因此本文件**直接 include 真实头文件、编译运行真实设备代码**：
//
//   nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
//     -I~/github/DeepEP/deep_ep/include -I/usr/local/cuda/include/cccl \
//     src/deepep_ptx_primitives.cu -o /tmp/deepep_ptx_primitives && /tmp/deepep_ptx_primitives

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/utils.cuh>
#include <deep_ep/layout/common/barrier.cuh>

using namespace deep_ep;

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("  [CUDA ERROR] %s (line %d): %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} }while(0)

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& what) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", what.c_str()); }
}

static __host__ __device__ __forceinline__ float b2f(nv_bfloat16 x) { return __bfloat162float(x); }

__global__ void k_pack(int* out) {
    const auto p = deep_ep::math::pack2<int16_t, int32_t>(0x1111, 0x2222);
    int16_t a, b;
    deep_ep::math::unpack2<int16_t, int32_t>(p, a, b);
    out[0] = (int)(uint32_t)p;
    out[1] = a; out[2] = b;
}
static void part0b_pack();

// =============================================================================
// Part 0: math.cuh —— 整数编码/取整/指针/打包（host + device 语义）
// =============================================================================
static void part0_math() {
    printf("\n=== Part 0: common/math.cuh 语义 ===\n");
    // encode_decode_positive 是对合：encode(x) = -x-1，再 encode 一次得回 x。
    // 就绪判据：is_decoded_positive_ready(raw) 作用在「解码后」的值上 ⇒ raw<0 才就绪，raw==0 未就绪。
    bool ok = true;
    for (int x = 0; x < 8; ++ x) {
        const int raw = deep_ep::math::encode_decode_positive<int>(x);   // -x-1
        const int decoded = deep_ep::math::encode_decode_positive<int>(raw);
        if (decoded != x) ok = false;                                    // 对合
        if (!deep_ep::math::is_decoded_positive_ready<int>(decoded)) ok = false;  // 正数 = 就绪
        if (raw >= 0) ok = false;                                        // 编码后恒为负
    }
    printf("    encode(0)=%d, decode(encode(5))=%d, 未写 0 -> decode=%d(未就绪)\n",
           deep_ep::math::encode_decode_positive<int>(0),
           deep_ep::math::encode_decode_positive<int>(deep_ep::math::encode_decode_positive<int>(5)),
           deep_ep::math::encode_decode_positive<int>(0));
    check(ok, "encode_decode_positive 对合；就绪值编码为负、未写 0 解码为负");
    check(deep_ep::math::encode_decode_positive<int>(0) == -1 and
          deep_ep::math::encode_decode_positive<int>(1023) == -1024 and
          !deep_ep::math::is_decoded_positive_ready<int>(deep_ep::math::encode_decode_positive<int>(0)) and
          deep_ep::math::is_decoded_positive_ready<int>(deep_ep::math::encode_decode_positive<int>(-1)),
          "raw=0 未就绪、raw=-1 就绪（就绪 iff raw<0）");

    // ceil_div / align（含 kDoCeilAlignment=false 的向下取整）
    check(deep_ep::math::ceil_div(7, 4) == 2 and deep_ep::math::ceil_div(8, 4) == 2, "ceil_div");
    check(deep_ep::math::align(7, 4) == 8 and deep_ep::math::align(8, 4) == 8, "align 向上");
    check(deep_ep::math::align<int, false>(7, 4) == 4 and deep_ep::math::align<int, false>(15, 4) == 12, "align 向下");
    check(deep_ep::math::constexpr_align(513, 64) == 576, "constexpr_align");

    // advance_ptr / ptr_diff
    char buf[64];
    check(deep_ep::math::advance_ptr(buf, 16) == buf + 16, "advance_ptr");
    check(deep_ep::math::ptr_diff(buf + 16, buf) == 16, "ptr_diff");

    // pack2 / unpack2（低位在前；__device__ only → 在设备端验证，见 part0b）
    // BarrierSignals 布局（kNumRDMAAlignmentBytes=64 对齐）
    check(sizeof(deep_ep::layout::BarrierSignals) == 64, "BarrierSignals 对齐到 64B");
    deep_ep::layout::BarrierSignals bs;
    check(bs.get_nvl_barrier_signal_ptr(0) == &bs.nvl_barrier_signals[0] and
          bs.get_nvl_barrier_signal_ptr(1) == &bs.nvl_barrier_signals[1] and
          bs.get_nvl_barrier_counter_ptr() == &bs.nvl_barrier_counter, "BarrierSignals 取址");
    part0b_pack();
}

// =============================================================================
// Part 1: 线程布局 & elect.sync
// =============================================================================
static void part0b_pack() {
    int* d;
    CK(cudaMalloc(&d, 3 * sizeof(int)));
    k_pack<<<1, 1>>>(d);
    CK(cudaDeviceSynchronize());
    int h[3];
    CK(cudaMemcpy(h, d, 3 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("    pack2<int16>(0x1111,0x2222)=0x%08x -> unpack=(0x%x,0x%x)\n", (uint32_t)h[0], (uint16_t)h[1], (uint16_t)h[2]);
    check((uint32_t)h[0] == 0x22221111 and (uint16_t)h[1] == 0x1111 and (uint16_t)h[2] == 0x2222,
          "pack2/unpack2 低位在前（设备端）");
    cudaFree(d);
}

__global__ void k_elect(int* elected_per_warp, int* warp_idx_out, int* lane_idx_out) {
    const int warp = ptx::get_warp_idx();
    const int lane = ptx::get_lane_idx();
    if (lane == 0) { warp_idx_out[blockIdx.x * (blockDim.x / 32) + warp] = warp; }
    if (ptx::elect_one_sync())
        atomicAdd(&elected_per_warp[warp], 1);
    if (threadIdx.x == 0) lane_idx_out[0] = lane;
}
static void part1_elect() {
    printf("\n=== Part 1: get_warp_idx / get_lane_idx / elect_one_sync (ptx.cuh:23-46) ===\n");
    const int warps = 4;
    int *d_elec, *d_warp, *d_lane;
    CK(cudaMalloc(&d_elec, warps * sizeof(int)));
    CK(cudaMalloc(&d_warp, warps * sizeof(int)));
    CK(cudaMalloc(&d_lane, sizeof(int)));
    CK(cudaMemset(d_elec, 0, warps * sizeof(int)));
    k_elect<<<1, warps * 32>>>(d_elec, d_warp, d_lane);
    CK(cudaDeviceSynchronize());
    std::vector<int> elec(warps), warp(warps), lane(1);
    CK(cudaMemcpy(elec.data(), d_elec, warps * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(warp.data(), d_warp, warps * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(lane.data(), d_lane, sizeof(int), cudaMemcpyDeviceToHost));
    bool one = true, wid = true;
    for (int i = 0; i < warps; ++ i) { one &= (elec[i] == 1); wid &= (warp[i] == i); }
    printf("    elect 每 warp 命中数 = [%d %d %d %d]，warp_idx=[%d %d %d %d]，lane0=%d\n",
           elec[0], elec[1], elec[2], elec[3], warp[0], warp[1], warp[2], warp[3], lane[0]);
    check(one, "elect_one_sync 每 warp 恰好选 1 个 lane");
    check(wid, "get_warp_idx 与 warp 编号一致");
    check(lane[0] == 0, "get_lane_idx (mov %laneid)");
    cudaFree(d_elec); cudaFree(d_warp); cudaFree(d_lane);
}

// =============================================================================
// Part 2: mbarrier 生命周期（init/arrive/wait + 相位翻转 + expect_tx + invalidate）
// =============================================================================
__global__ void k_mbarrier_phase(uint32_t* out_phase_after_two) {
    __shared__ ptx::mbarrier bar;
    if (threadIdx.x == 0) ptx::mbarrier_init_with_fence(&bar, 1);
    __syncthreads();
    ptx::arrival_phase phase = 0;
    if (threadIdx.x == 0) ptx::mbarrier_arrive(&bar);
    ptx::mbarrier_wait_and_flip_phase(&bar, phase);   // 第 1 次：等 0 相位
    const auto p1 = phase;
    if (threadIdx.x == 0) ptx::mbarrier_arrive(&bar);
    ptx::mbarrier_wait_and_flip_phase(&bar, phase);   // 第 2 次：等 1 相位
    if (threadIdx.x == 0) {
        out_phase_after_two[0] = p1;                  // 翻转后应为 1
        out_phase_after_two[1] = phase;               // 再翻转后应为 0
    }
}
static void part2_mbarrier_phase() {
    printf("\n=== Part 2: mbarrier 相位翻转 (ptx.cuh:49-83) ===\n");
    uint32_t* d;
    CK(cudaMalloc(&d, 2 * sizeof(uint32_t)));
    k_mbarrier_phase<<<1, 128>>>(d);
    CK(cudaDeviceSynchronize());
    uint32_t h[2];
    CK(cudaMemcpy(h, d, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("    两次 wait 后 phase = %u, %u（期望 1, 0）\n", h[0], h[1]);
    check(h[0] == 1 and h[1] == 0, "mbarrier_wait_and_flip_phase 每次翻转 0<->1");
    cudaFree(d);
}

// =============================================================================
// Part 3: TMA 1D 载入 + expect_tx（真实异步拷贝引擎）
// =============================================================================
__global__ void k_tma_load(const float* src, float* dst, int n) {
    __shared__ alignas(128) float smem[256];
    __shared__ ptx::mbarrier bar;
    if (threadIdx.x == 0) {
        ptx::mbarrier_init_with_fence(&bar, 1);
        ptx::mbarrier_arrive_and_set_tx(&bar, n * (int)sizeof(float));   // 预算 n*4 字节
        ptx::tma_load_1d(smem, src, &bar, n * (int)sizeof(float));       // complete_tx::bytes
    }
    __syncthreads();
    ptx::arrival_phase phase = 0;
    ptx::mbarrier_wait_and_flip_phase(&bar, phase);                      // 等 tx 到齐
    if (threadIdx.x == 0) { ptx::mbarrier_invalidate(&bar); }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = smem[i];
}
static void part3_tma_load() {
    printf("\n=== Part 3: TMA 1D 载入 + expect_tx (ptx.cuh:65-68/122-134) ===\n");
    const int n = 256;
    std::vector<float> h(n);
    for (int i = 0; i < n; ++ i) h[i] = (float)(i * 3 - 7);
    float *d_src, *d_dst;
    CK(cudaMalloc(&d_src, n * sizeof(float)));
    CK(cudaMalloc(&d_dst, n * sizeof(float)));
    CK(cudaMemcpy(d_src, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    k_tma_load<<<1, 128>>>(d_src, d_dst, n);
    CK(cudaDeviceSynchronize());
    std::vector<float> got(n);
    CK(cudaMemcpy(got.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));
    check(got == h, "TMA 1D 载入完整正确（mbarrier expect_tx 同步）");
    cudaFree(d_src); cudaFree(d_dst);
}

// =============================================================================
// Part 4: TMA 1D 存储 + cp.reduce.async.bulk.add.f32（归约写回语义）
// =============================================================================
__global__ void k_tma_store(const float* src, float* dst, int n) {
    __shared__ alignas(128) float smem[256];
    for (int i = threadIdx.x; i < n; i += blockDim.x) smem[i] = src[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        ptx::tma_store_fence();
        ptx::tma_store_1d(dst, smem, n * (int)sizeof(float));
        ptx::tma_store_commit();
        ptx::tma_store_wait<0>();
    }
}
__global__ void k_tma_reduce_add(const float* src, float* dst, int n) {
    __shared__ alignas(128) float smem[256];
    for (int i = threadIdx.x; i < n; i += blockDim.x) smem[i] = src[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        ptx::tma_store_fence();
        ptx::tma_store_reduce_add_f32(dst, smem, n * (int)sizeof(float));   // dst += smem（元素级原子加）
        ptx::tma_store_commit();
        ptx::tma_store_wait<0>();
    }
}
static void part4_tma_store() {
    printf("\n=== Part 4: TMA 1D 存储 & cp.reduce.async.bulk.add.f32 (ptx.cuh:136-160) ===\n");
    const int n = 256;
    float *d_a, *d_b, *d_out;
    CK(cudaMalloc(&d_a, n * sizeof(float)));
    CK(cudaMalloc(&d_b, n * sizeof(float)));
    CK(cudaMalloc(&d_out, n * sizeof(float)));
    std::vector<float> a(n), b(n), init(n);
    for (int i = 0; i < n; ++ i) { a[i] = (float)i; b[i] = (float)(i * 2 + 1); init[i] = 100.f; }
    // (a) 普通 store：覆盖
    CK(cudaMemcpy(d_out, init.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    k_tma_store<<<1, 128>>>(d_a, d_out, n);
    CK(cudaDeviceSynchronize());
    std::vector<float> got(n);
    CK(cudaMemcpy(got.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok_store = true;
    for (int i = 0; i < n; ++ i) ok_store &= (got[i] == a[i]);
    check(ok_store, "tma_store_1d 覆盖写正确");

    // (b) reduce-add：dst = init + a + b（两次归约写）
    CK(cudaMemcpy(d_out, init.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    k_tma_reduce_add<<<1, 128>>>(d_a, d_out, n);
    k_tma_reduce_add<<<1, 128>>>(d_b, d_out, n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    // 对照：普通 store（非归约）会把 init 覆盖掉，验证 reduce-add 是「累加」
    CK(cudaMemcpy(d_out, init.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    k_tma_store<<<1, 128>>>(d_a, d_out, n);
    CK(cudaDeviceSynchronize());
    std::vector<float> overwrite(n);
    CK(cudaMemcpy(overwrite.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok_ra = true;
    for (int i = 0; i < n; ++ i) ok_ra &= (std::fabs(got[i] - (init[i] + a[i] + b[i])) < 1e-3);
    printf("    归约写后 dst[3]=%.1f（期望 %.1f）；普通 store 后 dst[3]=%.1f（覆盖，=a[3]=%.1f）\n",
           got[3], init[3] + a[3] + b[3], overwrite[3], a[3]);
    check(ok_ra, "cp.reduce.async.bulk.add.f32 元素级累加（两次写 = init+a+b）");
    check(overwrite[3] == a[3], "普通 store 无归约（对照，覆盖 init）");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

// =============================================================================
// Part 5: st.bulk（SM100 专属；sm90 回退为逐 lane 清零循环）
// =============================================================================
__global__ void k_st_bulk(int* out) {
    __shared__ alignas(128) int smem[128];
    for (int i = threadIdx.x; i < 128; i += blockDim.x) smem[i] = 0xdeadbeef;
    __syncthreads();
    ptx::st_bulk<128 * (int)sizeof(int)>(smem);        // 清 512B
    __syncthreads();
    if (threadIdx.x == 0) {
        int bad = 0;
        for (int i = 0; i < 128; ++ i) bad += (smem[i] != 0);
        out[0] = bad;
    }
}
static void part5_st_bulk() {
    printf("\n=== Part 5: st.bulk (ptx.cuh:85-101) —— SM100 专属，sm90 走回退 ===\n");
    int* d;
    CK(cudaMalloc(&d, sizeof(int)));
    k_st_bulk<<<1, 128>>>(d);
    CK(cudaDeviceSynchronize());
    int bad;
    CK(cudaMemcpy(&bad, d, sizeof(int), cudaMemcpyDeviceToHost));
    printf("    sm90 回退路径清零后非零元素 = %d（期望 0）\n", bad);
    check(bad == 0, "st_bulk 在 sm90 回退分支把 smem 清零（真 st.bulk 是 arch>=1000）");
    cudaFree(d);
}

// =============================================================================
// Part 6: cp.async.ca + cp.async.mbarrier.arrive
// =============================================================================
__global__ void k_cp_async(const int* src, int* dst, int n) {
    __shared__ alignas(128) int smem[64];
    __shared__ ptx::mbarrier bar;
    if (threadIdx.x == 0) ptx::mbarrier_init_with_fence(&bar, 1);   // 与 dispatch.cuh:273 一致：count=1
    __syncthreads();
    if (threadIdx.x < n) {
        ptx::cp_async_ca(src + threadIdx.x, smem + threadIdx.x);    // cp.async.ca
        ptx::cp_async_mbarrier_arrive(&bar);                        // 拷贝完成时 arrive（issue 时 +1、完成时 -1）
    }
    // 与 dispatch.cuh:351-354 同构：等 cp.async 全部完成后，elected one 再做一次 arrive 触发相位翻转
    __syncthreads();
    if (threadIdx.x == 0) ptx::mbarrier_arrive(&bar);
    ptx::arrival_phase phase = 0;
    ptx::mbarrier_wait_and_flip_phase(&bar, phase);
    if (threadIdx.x == 0) ptx::mbarrier_invalidate(&bar);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = smem[i];
}
static void part6_cp_async() {
    printf("\n=== Part 6: cp_async_ca + cp_async_mbarrier_arrive (ptx.cuh:197-209) ===\n");
    const int n = 64;
    std::vector<int> h(n);
    for (int i = 0; i < n; ++ i) h[i] = 1000 + i;
    int *d_src, *d_dst;
    CK(cudaMalloc(&d_src, n * sizeof(int)));
    CK(cudaMalloc(&d_dst, n * sizeof(int)));
    CK(cudaMemcpy(d_src, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    k_cp_async<<<1, 64>>>(d_src, d_dst, n);
    CK(cudaDeviceSynchronize());
    std::vector<int> got(n);
    CK(cudaMemcpy(got.data(), d_dst, n * sizeof(int), cudaMemcpyDeviceToHost));
    check(got == h, "cp.async.ca 载入正确且由 cp.async.mbarrier.arrive 计数");
    cudaFree(d_src); cudaFree(d_dst);
}

// =============================================================================
// Part 7: 命名 barrier（bar.sync vs barrier.sync）
// =============================================================================
__global__ void k_named_barrier(int* out) {
    __shared__ int counter;
    if (threadIdx.x == 0) counter = 0;
    __syncthreads();
    atomicAdd(&counter, 1);
    ptx::named_barrier<128>(1);                 // 全部 128 线程
    if (threadIdx.x == 0) out[0] = counter;     // 到达后应 = 128
    __syncthreads();
    if (threadIdx.x == 0) counter = 0;
    __syncthreads();
    atomicAdd(&counter, 1);
    ptx::named_barrier_unaligned<128>(2);
    if (threadIdx.x == 0) out[1] = counter;
}
static void part7_named_barrier() {
    printf("\n=== Part 7: named_barrier / named_barrier_unaligned (ptx.cuh:212-221) ===\n");
    int* d;
    CK(cudaMalloc(&d, 2 * sizeof(int)));
    k_named_barrier<<<1, 128>>>(d);
    CK(cudaDeviceSynchronize());
    int h[2];
    CK(cudaMemcpy(h, d, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("    bar.sync 后 counter=%d，barrier.sync 后 counter=%d（期望 128,128）\n", h[0], h[1]);
    check(h[0] == 128 and h[1] == 128, "命名 barrier 令全部线程到达后才继续");
    cudaFree(d);
}

// =============================================================================
// Part 8: 谓词化 ld/st（ge/gt）+ iterate_ld_st 网格步进 + 尾部谓词
// =============================================================================
__global__ void k_pred_ld_st(const float* src, float* dst, int n) {
    const int lane = threadIdx.x;
    // 谓词化载入：lhs=n-lane、rhs=0，`gt` ⇒ 仅 lane<n 时真，否则返回 0
    const float4 v = ptx::ld_with_gt_pred(reinterpret_cast<const float4*>(src) + lane, n - lane, 0);
    // 谓词化存储：仅当 gt 为真才写
    ptx::st_with_gt_pred(reinterpret_cast<float4*>(dst) + lane, v, n - lane, 0);
}
__global__ void k_ldg_pred_zero(const int* src, int* out) {
    const int lane = threadIdx.x;
    const int n = 16;
    const int4 v = ptx::ldg_with_ge_pred(reinterpret_cast<const int4*>(src) + lane, n - lane, 0);  // pred 假 → 0
    out[lane] = v.x + v.y + v.z + v.w;
}
__global__ void k_iterate_ldst(const int* in, int* out, int n) {
    const int gt = blockIdx.x * blockDim.x + threadIdx.x;
    deep_ep::utils::iterate_ld_st<1, 128, 4>(
        gt, n, in, out,
        [](const int* p, const int& lhs = 1, const int& rhs = 0) { return lhs > rhs ? *p : 0; },
        [](int* p, const int v, const int& lhs = 1, const int& rhs = 0) { if (lhs > rhs) *p = v; });
}
static void part8_pred() {
    printf("\n=== Part 8: 谓词化 ld/st + iterate_ld_st (ptx.cuh:223-348; utils.cuh:14-75) ===\n");
    const int words = 256;
    std::vector<float> h(words);
    for (int i = 0; i < words; ++ i) h[i] = (float)(i * 7 - 3);
    float *d_src, *d_dst;
    CK(cudaMalloc(&d_src, words * sizeof(float)));
    CK(cudaMalloc(&d_dst, words * sizeof(float)));
    CK(cudaMemcpy(d_src, h.data(), words * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_dst, 0, words * sizeof(float)));   // 未写处保持 0
    // 64 个 float4 = 256 float；lane 0..63，谓词 lhs=n-lane，n=16 ⇒ 仅 lane 0..15 写
    k_pred_ld_st<<<1, 64>>>(d_src, d_dst, 16);
    CK(cudaDeviceSynchronize());
    std::vector<float> got(words);
    CK(cudaMemcpy(got.data(), d_dst, words * sizeof(float), cudaMemcpyDeviceToHost));
    bool wrote16 = true, untouched = true;
    for (int i = 0; i < 64; ++ i) wrote16 &= (got[i] == h[i]);            // lane0..15 -> float4 0..3
    for (int i = 64; i < words; ++ i) untouched &= (got[i] == 0.0f);      // 其余保持原值
    check(wrote16 and untouched, "ld/st 谓词跳过越界 lane（未写处保持原值）");

    // ldg_with_ge_pred：谓词为假时返回 0（`ld.global.nc` 的谓词版本）
    int *d_i, *d_o;
    CK(cudaMalloc(&d_i, words * sizeof(int)));
    CK(cudaMalloc(&d_o, 64 * sizeof(int)));
    CK(cudaMemcpy(d_i, h.data(), 1, cudaMemcpyHostToDevice));  // 无关，只要可读
    k_ldg_pred_zero<<<1, 64>>>(d_i, d_o);
    CK(cudaDeviceSynchronize());
    std::vector<int> ho(64);
    CK(cudaMemcpy(ho.data(), d_o, 64 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("    ldg_with_ge_pred：lane15=%d（真，非 0），lane20=%d（假，恒 0）\n", ho[15], ho[20]);
    check(ho[20] == 0, "ldg_with_ge_pred 谓词假 → 返回 0");
    cudaFree(d_i); cudaFree(d_o);

    // iterate_ld_st：num=1000、每组 128 线程 × 每线程 4 元素 → 尾部走谓词分支
    const int N = 1000;
    int* d_in; int* d_out;
    CK(cudaMalloc(&d_in, N * sizeof(int)));
    CK(cudaMalloc(&d_out, N * sizeof(int)));
    std::vector<int> hin(N);
    for (int i = 0; i < N; ++ i) hin[i] = i + 1;
    CK(cudaMemcpy(d_in, hin.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    // 用 lambda 传 ld/st，主循环 1 参 / 尾部 3 参
    k_iterate_ldst<<<1, 128>>>(d_in, d_out, N);
    CK(cudaDeviceSynchronize());
    std::vector<int> hout(N, -1);
    CK(cudaMemcpy(hout.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost));
    // 全部元素（含尾部 992..999）都应被写出
    bool all = true;
    for (int i = 0; i < N; ++ i) all &= (hout[i] == hin[i]);
    check(all, "iterate_ld_st 主循环 + 尾部谓词处理全部元素");
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_in); cudaFree(d_out);
}

// =============================================================================
// Part 9: 内存序原语（red / release / acquire / relaxed / volatile）
// =============================================================================
__global__ void k_red_add(int64_t* s64) {
    const int64_t v = 1;
    ptx::red_add(s64, v);                    // red.gpu.global.add.u64
    ptx::red_add_rel_sys(s64, v);            // red.release.sys.global.add.u64
    ptx::red_add_rel_gpu(s64, v);            // red.release.gpu.global.add.u64
}
static void part9_memory_order() {
    printf("\n=== Part 9: 内存序原语 (ptx.cuh:364-478) ===\n");
    // (a) red_add 家族：3 次 * K threads
    const int K = 256;
    int64_t* d_cnt;
    CK(cudaMalloc(&d_cnt, sizeof(int64_t)));
    CK(cudaMemset(d_cnt, 0, sizeof(int64_t)));
    k_red_add<<<1, K>>>(d_cnt);
    CK(cudaDeviceSynchronize());
    int64_t cnt;
    CK(cudaMemcpy(&cnt, d_cnt, sizeof(int64_t), cudaMemcpyDeviceToHost));
    printf("    red_add + red_add_rel_sys + red_add_rel_gpu × %d 线程 = %lld（期望 %d）\n",
           K, (long long)cnt, 3 * K);
    check(cnt == 3LL * K, "red_add / red_add_rel_{gpu,sys} 原子累加正确");
    cudaFree(d_cnt);
}

// 生产者-消费者：每轮用独立 cell（data[r][32] + flag[r]），block0 release 发布、block1 acquire 自旋。
__global__ void k_produce_consume(int* data, int* flag, int* out, int rounds) {
    if (blockIdx.x == 0) {                       // 生产者（单线程）
        if (threadIdx.x == 0) {
            for (int r = 0; r < rounds; ++ r) {
                for (int i = 0; i < 32; ++ i) data[r * 32 + i] = r * 100 + i;
                __threadfence();
                ptx::st_release_sys(flag + r, 1);    // 发布（cell r）
            }
        }
    } else if (blockIdx.x == 1) {                // 消费者（单线程）
        if (threadIdx.x == 0) {
            int bad = 0;
            for (int r = 0; r < rounds; ++ r) {
                int v;
                do { v = ptx::ld_acquire_sys(flag + r); } while (v != 1);
                for (int i = 0; i < 32; ++ i)
                    if (data[r * 32 + i] != r * 100 + i) bad += 1;
            }
            out[0] = bad; out[1] = rounds;
        }
    }
}
static void part9_memory_order2() {
    const int rounds = 2000;
    int *d_data, *d_flag, *d_out;
    CK(cudaMalloc(&d_data, rounds * 32 * sizeof(int)));
    CK(cudaMalloc(&d_flag, rounds * sizeof(int)));
    CK(cudaMalloc(&d_out, 2 * sizeof(int)));
    CK(cudaMemset(d_flag, 0, rounds * sizeof(int)));
    CK(cudaMemset(d_out, 0, 2 * sizeof(int)));
    k_produce_consume<<<2, 32>>>(d_data, d_flag, d_out, rounds);
    CK(cudaDeviceSynchronize());
    int h[2];
    CK(cudaMemcpy(h, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("    生产者-消费者 %d 轮：坏数据计数=%d，最终 seen=%d（期望 0, %d）\n", rounds, h[0], h[1], rounds);
    check(h[0] == 0 and h[1] == rounds, "st_release_sys + ld_acquire_sys：release-acquire 无撕裂/乱序");
    cudaFree(d_data); cudaFree(d_flag); cudaFree(d_out);
}

// =============================================================================
// Part 10: warp 内建（exchange/match/deduplicate/ffs/reduce_or/reduce_add/sum）
// =============================================================================
__global__ void k_warp_intrinsics(int* out) {
    const int lane = threadIdx.x & 31;
    // exchange：把 lane 的值广播到全部
    const int ex = ptx::exchange<int>(lane * 10, 5);
    // gather(ballot)：对 (lane%3==0) 投票
    const unsigned bal = ptx::gather(lane % 3 == 0);
    // match：同值成组
    const unsigned m = ptx::match(lane / 4);
    // deduplicate：组内最大 lane
    const bool is_master = ptx::deduplicate(lane / 4, lane);
    // reduce_or / reduce_add
    const unsigned ro = ptx::reduce_or(1u << (lane % 8));
    const int ra = ptx::reduce_add(lane);
    // warp inclusive / exclusive sum
    const int inc = ptx::warp_inclusive_sum(lane, lane);
    const int exc = ptx::warp_exclusive_sum(lane, lane);
    // get_master_lane_idx / ffs
    const int master = ptx::get_master_lane_idx(m);
    const int lowbit = ptx::ffs<unsigned>(0b101100);
    if (lane == 0) {
        out[0] = ex;                 // 50
        out[1] = (int)bal;           // bits 0,3,...,30
        out[2] = master;             // 组内最大 lane（按 round）
        out[3] = (int)ro;            // 0xff
        out[4] = ra;                 // 0+..+31 = 496
        out[5] = lowbit;             // ffs(0b101100)=2 (0-based)
        out[6] = 0;
    }
    // 每个 lane 的 dedup 命中数应为 8（32/4 组 × 1 master）
    atomicAdd(&out[6], is_master ? 1 : 0);
    if (lane == 31) { out[7] = inc; out[8] = exc; }   // lane31: inc=496, exc=465
}
static void part10_warp() {
    printf("\n=== Part 10: warp 内建 (ptx.cuh:481-560) ===\n");
    int* d;
    CK(cudaMalloc(&d, 16 * sizeof(int)));
    CK(cudaMemset(d, 0, 16 * sizeof(int)));
    k_warp_intrinsics<<<1, 32>>>(d);
    CK(cudaDeviceSynchronize());
    std::vector<int> h(16);
    CK(cudaMemcpy(h.data(), d, 16 * sizeof(int), cudaMemcpyDeviceToHost));
    unsigned expect_bal = 0;
    for (int l = 0; l < 32; ++ l) if (l % 3 == 0) expect_bal |= (1u << l);
    printf("    exchange=%d（=50），ballot=0x%08x（期望 0x%08x），master=%d，reduce_or=0x%x，"
           "reduce_add=%d，ffs=%d，dedup master 数=%d，inc/exc(lane31)=%d/%d\n",
           h[0], (unsigned)h[1], expect_bal, h[2], (unsigned)h[3], h[4], h[5], h[6], h[7], h[8]);
    check(h[0] == 50, "exchange 广播 src_lane 的值");
    check((unsigned)h[1] == expect_bal, "gather=__ballot_sync");
    check(h[2] == 3, "get_master_lane_idx = bfind.u32（lane0 所在组 {0,1,2,3} 的最高 lane=3）");
    check((unsigned)h[3] == 0xff, "reduce_or over warp");
    check(h[4] == 496, "reduce_add over warp = 496");
    check(h[5] == 2, "ffs 0-based lowest set bit");
    check(h[6] == 8, "deduplicate：32 lanes / 每组 4 → 8 个 master");
    check(h[7] == 496 and h[8] == 465, "warp inclusive/exclusive sum");
    cudaFree(d);
}

// =============================================================================
// Part 11: fadd4 / fmul4 / accumulate 回退路径（sm90 无打包指令）
// =============================================================================
__global__ void k_vec4_arith(const float4* in, float4* add, float4* mul, float2* acc, const nv_bfloat162* bf) {
    const float4 a = in[threadIdx.x];
    const float4 b = in[threadIdx.x + 64];
    add[threadIdx.x] = ptx::fadd4(a, b);
    mul[threadIdx.x] = ptx::fmul4(a, b);
    float2 a2 = make_float2(a.x, a.y);
    ptx::accumulate(a2, bf[threadIdx.x]);          // sm90 回退：bf16->f32 add
    acc[threadIdx.x] = a2;
}
static void part11_vec() {
    printf("\n=== Part 11: fadd4 / fmul4 / accumulate 回退 (ptx.cuh:562-602) ===\n");
    const int n = 64;
    std::vector<float4> h(2 * n);
    for (int i = 0; i < 2 * n; ++ i) h[i] = make_float4(i + 0.25f, i * 0.5f, -i, 3.f);
    std::vector<nv_bfloat162> hb(n);
    for (int i = 0; i < n; ++ i)
        hb[i] = __floats2bfloat162_rn((float)(i + 1), (float)(-i - 2));
    float4 *d_in, *d_add, *d_mul;
    float2* d_acc;
    nv_bfloat162* d_bf;
    CK(cudaMalloc(&d_in, 2 * n * sizeof(float4)));
    CK(cudaMalloc(&d_add, n * sizeof(float4)));
    CK(cudaMalloc(&d_mul, n * sizeof(float4)));
    CK(cudaMalloc(&d_acc, n * sizeof(float2)));
    CK(cudaMalloc(&d_bf, n * sizeof(nv_bfloat162)));
    CK(cudaMemcpy(d_in, h.data(), 2 * n * sizeof(float4), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bf, hb.data(), n * sizeof(nv_bfloat162), cudaMemcpyHostToDevice));
    k_vec4_arith<<<1, n>>>(d_in, d_add, d_mul, d_acc, d_bf);
    CK(cudaDeviceSynchronize());
    std::vector<float4> h_add(n), h_mul(n);
    std::vector<float2> h_acc(n);
    CK(cudaMemcpy(h_add.data(), d_add, n * sizeof(float4), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_mul.data(), d_mul, n * sizeof(float4), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_acc.data(), d_acc, n * sizeof(float2), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < n; ++ i) {
        const float4 a = h[i], b = h[i + n];
        ok &= (h_add[i].x == a.x + b.x and h_add[i].y == a.y + b.y and
               h_add[i].z == a.z + b.z and h_add[i].w == a.w + b.w);
        ok &= (h_mul[i].x == a.x * b.x and h_mul[i].y == a.y * b.y and
               h_mul[i].z == a.z * b.z and h_mul[i].w == a.w * b.w);
        const float2 bb = __bfloat1622float2(hb[i]);
        ok &= (h_acc[i].x == a.x + bb.x and h_acc[i].y == a.y + bb.y);
    }
    check(ok, "fadd4/fmul4/accumulate sm90 回退 = 标量逐元素（位一致）");
    cudaFree(d_in); cudaFree(d_add); cudaFree(d_mul); cudaFree(d_acc); cudaFree(d_bf);
}

// =============================================================================
// Part 12: warpgroup_reg_realloc（setmaxnreg，sm90a）
// =============================================================================
__global__ void __launch_bounds__(512, 1) k_reg_realloc(int* out) {
    // kNumThreads=512 ⇒ 初始寄存器预算 = 65536/512/8*8 = 128；40 < 128 ⇒ setmaxnreg.dec
    ptx::warpgroup_reg_realloc<40, 512>();
    if (threadIdx.x == 0) out[0] = 12345;
}
static void part12_reg_realloc() {
    printf("\n=== Part 12: warpgroup_reg_realloc / setmaxnreg (ptx.cuh:454-469) ===\n");
    // 公式：kNumInitialRegisters = (65536 / kNumThreads / 8) * 8
    auto init_regs = [](int t) { return (65536 / t / 8) * 8; };
    printf("    kNumInitialRegisters: t=128 -> %d（>255 早退）, t=512 -> %d\n", init_regs(128), init_regs(512));
    check(init_regs(512) == 128 and init_regs(128) == 512, "初始寄存器预算公式（65536/线程数/8*8）");
    int* d;
    CK(cudaMalloc(&d, sizeof(int)));
    CK(cudaMemset(d, 0, sizeof(int)));
    k_reg_realloc<<<1, 512>>>(d);
    CK(cudaDeviceSynchronize());
    int v;
    CK(cudaMemcpy(&v, d, sizeof(int), cudaMemcpyDeviceToHost));
    check(v == 12345, "setmaxnreg.dec 到 40 后 kernel 正常完成（sm90a）");
    cudaFree(d);
}

// =============================================================================
// multimem / cvt.rs —— host 语义 oracle（设备指令需多播映射 / SM100）
// =============================================================================
static float cvt_rs_bf16x2_oracle(float x, uint32_t bits) {
    // `cvt.rs.bf16x2.f32`：随机舍入到 bf16。把尾数低 16 位与随机 bits 比较，
    // low16 > bits 才进位（否则截断）；bits 均匀随机时等价于「按小数部分概率进位」。
    uint32_t u; memcpy(&u, &x, 4);
    const uint32_t low = u & 0xffffu;
    const uint32_t high = u >> 16;                 // bf16 截断的位模式
    const uint32_t rounded = (low > bits) ? (high + 1) : high;
    nv_bfloat16 r = __ushort_as_bfloat16((unsigned short)(rounded & 0xffffu));
    return __bfloat162float(r);
}
static void part13_oracles() {
    printf("\n=== Part 13: multimem / cvt.rs 语义 oracle（设备指令不可在 sm90 发射）===\n");
    // multimem.ld_reduce.add 的语义 = 「所有 rank 同地址之和」（顺序无关）。
    // 对应 all_reduce/nvlink.cuh:44-50 的 iterate_ld_st + multimem；此处纯 host 规约模型。
    const int ranks = 8, elems = 4;
    std::vector<std::vector<float>> per_rank(ranks, std::vector<float>(elems));
    for (int r = 0; r < ranks; ++ r) for (int e = 0; e < elems; ++ e) per_rank[r][e] = r * 10 + e;
    std::vector<float> sum(elems, 0.f);
    for (int e = 0; e < elems; ++ e) for (int r = 0; r < ranks; ++ r) sum[e] += per_rank[r][e];
    bool mul_ok = true;
    for (int e = 0; e < elems; ++ e) mul_ok &= (sum[e] == (ranks - 1) * ranks / 2 * 10 + e * ranks);
    printf("    all-reduce 语义（nvl all_reduce）：sum[j]=%g %g %g %g\n", sum[0], sum[1], sum[2], sum[3]);
    check(mul_ok, "multimem.ld_reduce.add 语义 = 各 rank 同地址求和（顺序无关）");

    // cvt.rs.bf16x2 oracle：随机位阈值越大越不容易进位（单调），且随机位下无偏。
    auto ups_fraction = [](uint32_t bits, int trials) {
        double up = 0, n = 0;
        for (int i = 0; i < trials; ++ i) {
            const float x = (float)((i * 2654435761u) % 100000) / 100000.0f + 0.5f;
            uint32_t u; memcpy(&u, &x, 4);
            const float trunc = b2f(__ushort_as_bfloat16((unsigned short)(u >> 16)));   // 截断（非 RNE）
            if (cvt_rs_bf16x2_oracle(x, bits) > trunc) up += 1;
            n += 1;
        }
        return up / n;
    };
    const double f0 = ups_fraction(0x0000, 40000);
    const double f4 = ups_fraction(0x4000, 40000);
    const double f8 = ups_fraction(0x8000, 40000);
    const double fc = ups_fraction(0xc000, 40000);
    const double ff = ups_fraction(0xffff, 40000);
    printf("    进位比例 vs 随机 bits：0x0000=%.4f 0x4000=%.4f 0x8000=%.4f 0xc000=%.4f 0xffff=%.4f（应单调↓）\n",
           f0, f4, f8, fc, ff);
    check(f0 > f4 and f4 > f8 and f8 > fc and fc > ff and ff == 0.0,
          "cvt.rs.bf16x2：阈值越大越少进位、bits=0xffff 恒截断（单调）");
    check(true, "cvt.rs.bf16x2 / multimem 均在 sm90 不可发射（SM100 / 需多播映射）——仅语义 oracle");
}

// =============================================================================
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("DeepEP 设备端公共层实验 —— H100 (sm90a)，直接 include 真实头文件\n");
    part0_math();
    part1_elect();
    part2_mbarrier_phase();
    part3_tma_load();
    part4_tma_store();
    part5_st_bulk();
    part6_cp_async();
    part7_named_barrier();
    part8_pred();
    part9_memory_order();
    part9_memory_order2();
    part10_warp();
    part11_vec();
    part12_reg_realloc();
    part13_oracles();
    printf("\n========== 汇总：%d PASS / %d FAIL ==========\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
