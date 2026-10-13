// P67-1 DeepGEMM 的共享 TMA 拷贝层 + 环形流水线：
//   deep_gemm/include/deep_gemm/common/tma_copy.cuh
//   deep_gemm/include/deep_gemm/common/ring_pipeline.cuh
// 真机实验（H100 sm90，nvcc 直接 include 真实头文件跑真实设备代码）：
//   Part 1 host constexpr —— get_inner_block_atom_size / get_smem_pack_factor（含 FP4 打包因子）
//   Part 2 设备端 RingPipeline<kNumStages>::advance 状态机（含非 2 的幂 + step<=N）
//   Part 3 真实 tma::copy 2D 单原子 / 多原子（atom 循环）/ 3D 路径，逐字节对比预测布局
//   Part 4 真实 tma::copy 多播（cluster=2，仅 rank0 发 LoadMulticast，两 CTA 各自的 barrier 收到 tx）
//
// 编译：见 harness/exp_tma_copy_ring.sh
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>

namespace DG = deep_gemm::tma;

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& tag) {
    if (ok) { g_pass++; } else { g_fail++; printf("  !! FAIL %s\n", tag.c_str()); }
}

#define CU_CHECK(x) do { auto e = (x); if (e != cudaSuccess) { \
    printf("  !! CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); g_fail++; } } while (0)

// ---------------------------------------------------------------------------
// tensormap 构造（直连 driver）
// ---------------------------------------------------------------------------
static CUtensorMap make_map_2d(void* ptr, uint64_t inner, uint64_t outer,
                               uint32_t box_inner, uint32_t box_outer,
                               CUtensorMapDataType dt, CUtensorMapSwizzle swz) {
    CUtensorMap m;
    uint64_t dims[2] = {inner, outer};
    uint64_t strides[1] = {inner * /*storage bytes*/ (dt == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 ? 2 : 1)};
    uint32_t box[2] = {box_inner, box_outer};
    uint32_t es[2] = {1, 1};
    cuTensorMapEncodeTiled(&m, dt, 2, ptr, dims, strides, box, es,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return m;
}

static CUtensorMap make_map_3d(void* ptr, uint64_t inner, uint64_t outer, uint64_t batch,
                               uint32_t box_inner, uint32_t box_outer,
                               CUtensorMapDataType dt, CUtensorMapSwizzle swz) {
    CUtensorMap m;
    uint64_t dims[3] = {inner, outer, batch};
    uint64_t strides[2] = {inner * (dt == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 ? 2ULL : 1ULL),
                           inner * outer * (dt == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 ? 2ULL : 1ULL)};
    uint32_t box[3] = {box_inner, box_outer, 1};
    uint32_t es[3] = {1, 1, 1};
    cuTensorMapEncodeTiled(&m, dt, 3, ptr, dims, strides, box, es,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return m;
}

// ---------------------------------------------------------------------------
// Part 1：host constexpr（atom size 与打包因子）
// ---------------------------------------------------------------------------
static void part1_constexpr() {
    printf("== Part 1: get_inner_block_atom_size / get_smem_pack_factor（host constexpr）==\n");
    // 独立公式：swizzle==0 ? BLOCK_INNER/pack : swizzle_bytes/sizeof(dtype)
    auto ref_atom = [](uint32_t BI, uint32_t swz, uint32_t pack, uint32_t elem) -> uint32_t {
        return swz == 0 ? BI / pack : swz / elem;
    };
    using bf16 = cutlass::bfloat16_t;
    using e4m3 = cutlass::float_e4m3_t;
    using e2m1 = cutlass::float_e2m1_t;
    check(DG::get_inner_block_atom_size<256, 0,   1, bf16>() == ref_atom(256, 0,   1, 2), "atom_bf16_sw0");
    check(DG::get_inner_block_atom_size<256, 128, 1, bf16>() == ref_atom(256, 128, 1, 2), "atom_bf16_sw128");
    check(DG::get_inner_block_atom_size<256, 64,  1, bf16>() == ref_atom(256, 64,  1, 2), "atom_bf16_sw64");
    check(DG::get_inner_block_atom_size<256, 32,  1, bf16>() == ref_atom(256, 32,  1, 2), "atom_bf16_sw32");
    check(DG::get_inner_block_atom_size<256, 0,   1, e4m3>() == ref_atom(256, 0,   1, 1), "atom_fp8_sw0");
    check(DG::get_inner_block_atom_size<256, 128, 1, e4m3>() == ref_atom(256, 128, 1, 1), "atom_fp8_sw128");
    // FP4：storage 1 字节，logical 因子 2
    check(DG::get_inner_block_atom_size<256, 0,   2, e2m1>() == ref_atom(256, 0,   2, 1), "atom_fp4_sw0");
    check(DG::get_inner_block_atom_size<256, 128, 2, e2m1>() == ref_atom(256, 128, 2, 1), "atom_fp4_sw128");
    check(deep_gemm::get_smem_pack_factor<bf16>() == 1, "pack_bf16");
    check(deep_gemm::get_smem_pack_factor<e4m3>() == 1, "pack_fp8");
    check(deep_gemm::get_smem_pack_factor<e2m1>() == 2, "pack_fp4");
    printf("  bf16: sw0->%u sw32->%u sw64->%u sw128->%u (logical)\n",
           DG::get_inner_block_atom_size<256, 0, 1, bf16>(), DG::get_inner_block_atom_size<256, 32, 1, bf16>(),
           DG::get_inner_block_atom_size<256, 64, 1, bf16>(), DG::get_inner_block_atom_size<256, 128, 1, bf16>());
    printf("  fp8 : sw0->%u sw128->%u | fp4: sw0->%u sw128->%u storage, pack=%u\n",
           DG::get_inner_block_atom_size<256, 0, 1, e4m3>(), DG::get_inner_block_atom_size<256, 128, 1, e4m3>(),
           DG::get_inner_block_atom_size<256, 0, 2, e2m1>(), DG::get_inner_block_atom_size<256, 128, 2, e2m1>(),
           (uint32_t)deep_gemm::get_smem_pack_factor<e2m1>());
}

// ---------------------------------------------------------------------------
// Part 2：RingPipeline<kNumStages>::advance 状态机
// ---------------------------------------------------------------------------
template <uint32_t kNumStages>
__global__ void ring_kernel(const uint32_t num_steps, const uint32_t step_size,
                            uint32_t* stage_out, uint32_t* phase_out) {
    deep_gemm::RingPipeline<kNumStages> p;
    if (threadIdx.x == 0) {
        for (uint32_t s = 0; s < num_steps; ++ s) {
            auto ret = p.advance(step_size);
            stage_out[s] = cute::get<0>(ret);
            phase_out[s] = cute::get<1>(ret);
        }
    }
}

template <uint32_t kNumStages>
static void run_ring(const uint32_t step_size, const std::string& tag) {
    const uint32_t steps = 5 * kNumStages;   // 足够跨多圈
    uint32_t *ds = nullptr, *dp = nullptr;
    CU_CHECK(cudaMalloc(&ds, steps * sizeof(uint32_t)));
    CU_CHECK(cudaMalloc(&dp, steps * sizeof(uint32_t)));
    ring_kernel<kNumStages><<<1, 32>>>(steps, step_size, ds, dp);
    CU_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> s(steps), ph(steps);
    cudaMemcpy(s.data(), ds, steps * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ph.data(), dp, steps * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // oracle：第 i 次调用前状态 = (pos, (i/kNumStages)&1)，其中 pos=i%kNumStages；返回旧状态
    int mism = 0;
    for (uint32_t i = 0; i < steps; ++ i) {
        const uint32_t pos = (i * step_size) % kNumStages;
        const uint32_t par = ((i * step_size) / kNumStages) & 1u;
        if (s[i] != pos or ph[i] != par) {
            if (mism < 4) printf("    step=%u call=%u got(%u,%u) exp(%u,%u)\n", step_size, i, s[i], ph[i], pos, par);
            mism++;
        }
    }
    check(mism == 0, tag);
    printf("  N=%-2u step=%u :: %u calls mismatch=%d\n", kNumStages, step_size, steps, mism);
    cudaFree(ds); cudaFree(dp);
}

static void part2_ring() {
    printf("== Part 2: RingPipeline::advance 状态机 vs oracle（返回旧 (stage,phase)）==\n");
    run_ring<1>(1, "ring_N1_s1");
    run_ring<2>(1, "ring_N2_s1");
    run_ring<3>(1, "ring_N3_s1");   // 非 2 的幂
    run_ring<4>(1, "ring_N4_s1");
    run_ring<5>(1, "ring_N5_s1");   // 非 2 的幂
    run_ring<8>(1, "ring_N8_s1");
    // step <= N：整数步进（仍在「最多一圈」约束内）
    run_ring<4>(2, "ring_N4_s2");
    run_ring<8>(3, "ring_N8_s3");
    run_ring<8>(8, "ring_N8_s8");   // 恰好一整圈
}

// ---------------------------------------------------------------------------
// Part 3：真实 tma::copy —— 2D / 3D / 多原子
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode,
          typename dtype_t, bool kIs3DTMA>
__global__ void tma_copy_kernel(const __grid_constant__ CUtensorMap tmap,
                                const int c_inner, const int c_outer, const int c_batch,
                                unsigned char* __restrict__ out) {
    constexpr int kElem = sizeof(dtype_t);
    __shared__ __align__(1024) unsigned char smem[BLOCK_INNER * BLOCK_OUTER * kElem];
    __shared__ __align__(8) uint64_t bar_storage;
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(&bar_storage);
    if (threadIdx.x == 0) {
        bar->init(1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        deep_gemm::tma::copy<BLOCK_INNER, BLOCK_OUTER, kSwizzleMode, dtype_t, kIs3DTMA>(
            &tmap, bar, reinterpret_cast<dtype_t*>(smem), c_inner, c_outer, 1, c_batch);
        bar->arrive_and_expect_tx(BLOCK_INNER * BLOCK_OUTER * kElem);
        bar->wait(0);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < BLOCK_INNER * BLOCK_OUTER * kElem; i += blockDim.x) out[i] = smem[i];
}

// 预测物理 smem 偏移：TMA swizzle XOR 键取自绝对 smem 地址 bits[7:10)（见 docs/56 §Part2）
static int swz_offset(const int swz_bytes, const int linear_byte) {
    if (swz_bytes == 0) return linear_byte;
    const int bits = swz_bytes == 128 ? 3 : (swz_bytes == 64 ? 2 : 1);
    const int mask = (1 << bits) - 1;
    const int key = (linear_byte >> 7) & mask;
    return linear_byte ^ (key << 4);
}

// 跑一次 tma::copy，返回 mismatch（对比预测；smem 基址 1024 对齐故 swizzle base_row_bits=0）
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode, bool kIs3DTMA>
static int run_tma_copy_raw(const int inner_total, const int outer_total, const int batch_total,
                            const int c_inner, const int c_outer, const int c_batch, const std::string& tag) {
    using dtype_t = cutlass::bfloat16_t;
    constexpr int kElem = sizeof(dtype_t);
    constexpr int kAtomLogical = deep_gemm::tma::get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, 1, dtype_t>();
    const int box_inner = kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / kElem;
    const int box_outer = BLOCK_OUTER;

    const int64_t n_elem = (int64_t)inner_total * outer_total * batch_total;
    std::vector<int16_t> host(n_elem);
    for (int64_t i = 0; i < n_elem; i++) host[i] = (int16_t)(i & 0x7fff);
    int16_t* src = nullptr;
    CU_CHECK(cudaMalloc(&src, n_elem * kElem));
    CU_CHECK(cudaMemcpy(src, host.data(), n_elem * kElem, cudaMemcpyHostToDevice));

    CUtensorMap tmap;
    if constexpr (kIs3DTMA)
        tmap = make_map_3d(src, inner_total, outer_total, batch_total, box_inner, box_outer,
                           CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                           kSwizzleMode == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
                           kSwizzleMode == 64 ? CU_TENSOR_MAP_SWIZZLE_64B :
                           kSwizzleMode == 32 ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE);
    else
        tmap = make_map_2d(src, inner_total, outer_total, box_inner, box_outer,
                           CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                           kSwizzleMode == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
                           kSwizzleMode == 64 ? CU_TENSOR_MAP_SWIZZLE_64B :
                           kSwizzleMode == 32 ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE);

    unsigned char* out = nullptr;
    const int out_bytes = BLOCK_INNER * BLOCK_OUTER * kElem;
    CU_CHECK(cudaMalloc(&out, out_bytes));
    tma_copy_kernel<BLOCK_INNER, BLOCK_OUTER, kSwizzleMode, dtype_t, kIs3DTMA>
        <<<1, 128>>>(tmap, c_inner, c_outer, c_batch, out);
    CU_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned char> got(out_bytes);
    CU_CHECK(cudaMemcpy(got.data(), out, out_bytes, cudaMemcpyDeviceToHost));

    // 预测：全局值 = (c_outer+r)*inner_total + (c_inner+c) + c_batch*inner_total*outer_total
    std::vector<unsigned char> exp(out_bytes, 0);
    for (int r = 0; r < BLOCK_OUTER; r++) {
        for (int c = 0; c < BLOCK_INNER; c++) {
            const int atom = c / kAtomLogical;
            const int c_in_atom = c % kAtomLogical;
            const int64_t g = (int64_t)(c_outer + r) * inner_total + (c_inner + c)
                              + (int64_t)c_batch * inner_total * outer_total;
            const uint16_t v = (uint16_t)(g & 0x7fff);
            const int atom_base = atom * BLOCK_OUTER * kAtomLogical * kElem;
            const int lin = r * (kAtomLogical * kElem) + c_in_atom * kElem;
            const int phys = atom_base + swz_offset(kSwizzleMode, lin);
            exp[phys] = v & 0xff;
            exp[phys + 1] = (v >> 8) & 0xff;
        }
    }
    int mism = 0;
    for (int i = 0; i < out_bytes; i++) if (got[i] != exp[i]) mism++;
    check(mism == 0, tag + "_mismatch");
    printf("  %-28s box=%ux%u atom=%u atoms=%u mismatch=%d/%d\n",
           tag.c_str(), box_inner, box_outer, kAtomLogical, BLOCK_INNER / kAtomLogical, mism, out_bytes);
    cudaFree(src); cudaFree(out);
    return mism;
}

static void part3_real_copy() {
    printf("== Part 3: 真实 tma::copy（2D 单原子 / 多原子 atom 循环 / 3D 路径）==\n");
    using bf16 = cutlass::bfloat16_t;
    // 2D：swizzle=0，BLOCK_INNER 一整个 box
    run_tma_copy_raw<64, 8, 0, false>(256, 64, 1, 0, 0, 0, "2d_sw0_1atom");
    // 2D：swizzle=128，BLOCK_INNER == atom（64）
    run_tma_copy_raw<64, 8, 128, false>(256, 64, 1, 8, 4, 0, "2d_sw128_1atom");
    // 2D：swizzle=128，BLOCK_INNER=256 -> 4 个 atom，靠 copy_nd 的循环拼接
    run_tma_copy_raw<256, 8, 128, false>(512, 64, 1, 16, 8, 0, "2d_sw128_4atom");
    // 2D：swizzle=64，BLOCK_INNER=64 -> 2 个 atom
    run_tma_copy_raw<64, 8, 64, false>(256, 64, 1, 0, 0, 0, "2d_sw64_2atom");
    // 3D：kIs3DTMA=true（batch 维 = 独立矩阵），取第 2 个 batch
    run_tma_copy_raw<64, 8, 128, true>(256, 64, 4, 8, 4, 2, "3d_sw128_batch2");
}

// ---------------------------------------------------------------------------
// Part 4：真实 tma::copy 多播（cluster=2）
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode, uint32_t kNumMulticast>
__global__ void __cluster_dims__(kNumMulticast, 1, 1)
tma_multicast_kernel(const __grid_constant__ CUtensorMap tmap,
                     const int c_inner, const int c_outer,
                     unsigned char* __restrict__ out) {
    using dtype_t = cutlass::bfloat16_t;
    constexpr int kElem = sizeof(dtype_t);
    __shared__ __align__(1024) unsigned char smem[BLOCK_INNER * BLOCK_OUTER * kElem];
    __shared__ __align__(8) uint64_t bar_storage;
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(&bar_storage);
    const uint32_t rank = cute::block_rank_in_cluster();
    if (threadIdx.x == 0) {
        bar->init(1);
        cutlass::arch::fence_barrier_init();
    }
    cute::cluster_sync();
    if (threadIdx.x == 0) {
        // 只有 rank0 真发（copy_nd 内部对 rank!=0 不再发 LoadMulticast）
        deep_gemm::tma::copy<BLOCK_INNER, BLOCK_OUTER, kSwizzleMode, dtype_t, false>(
            &tmap, bar, reinterpret_cast<dtype_t*>(smem), c_inner, c_outer, kNumMulticast, 0);
        bar->arrive_and_expect_tx(BLOCK_INNER * BLOCK_OUTER * kElem);
        bar->wait(0);
    }
    __syncthreads();
    unsigned char* my_out = out + rank * (BLOCK_INNER * BLOCK_OUTER * kElem);
    for (int i = threadIdx.x; i < BLOCK_INNER * BLOCK_OUTER * kElem; i += blockDim.x) my_out[i] = smem[i];
}

static void part4_multicast() {
    printf("== Part 4: 真实 tma::copy 多播（cluster=2，LoadMulticast，两 CTA 各自 barrier 收 tx）==\n");
    using dtype_t = cutlass::bfloat16_t;
    constexpr int BLOCK_INNER = 64, BLOCK_OUTER = 8, kElem = sizeof(dtype_t), kNumMulticast = 2;
    const int inner_total = 256, outer_total = 64;
    const int64_t n_elem = (int64_t)inner_total * outer_total;
    std::vector<int16_t> host(n_elem);
    for (int64_t i = 0; i < n_elem; i++) host[i] = (int16_t)(i & 0x7fff);
    int16_t* src = nullptr;
    CU_CHECK(cudaMalloc(&src, n_elem * kElem));
    CU_CHECK(cudaMemcpy(src, host.data(), n_elem * kElem, cudaMemcpyHostToDevice));
    auto tmap = make_map_2d(src, inner_total, outer_total, BLOCK_INNER /*swz128 box inner=64*/, BLOCK_OUTER,
                            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B);
    const int per_cta = BLOCK_INNER * BLOCK_OUTER * kElem;
    unsigned char* out = nullptr;
    CU_CHECK(cudaMalloc(&out, kNumMulticast * per_cta));
    tma_multicast_kernel<BLOCK_INNER, BLOCK_OUTER, 128, kNumMulticast>
        <<<kNumMulticast, 128>>>(tmap, 8, 4, out);
    CU_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> got(kNumMulticast * per_cta);
    CU_CHECK(cudaMemcpy(got.data(), out, kNumMulticast * per_cta, cudaMemcpyDeviceToHost));
    for (uint32_t rank = 0; rank < kNumMulticast; rank++) {
        int mism = 0;
        for (int r = 0; r < BLOCK_OUTER; r++)
            for (int c = 0; c < BLOCK_INNER; c++) {
                const int64_t g = (int64_t)(4 + r) * inner_total + (8 + c);
                const uint16_t v = (uint16_t)(g & 0x7fff);
                const int lin = r * (BLOCK_INNER * kElem) + c * kElem;   // swz128, atom=64 => 单原子
                const int phys = swz_offset(128, lin);
                if (got[rank * per_cta + phys] != (v & 0xff) or
                    got[rank * per_cta + phys + 1] != ((v >> 8) & 0xff)) mism++;
            }
        check(mism == 0, "multicast_rank" + std::to_string(rank));
        printf("  rank=%u mismatch=%d/%d\n", rank, mism, per_cta);
    }
    cudaFree(src); cudaFree(out);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("tma_copy.cuh / ring_pipeline.cuh 实验（H100 sm90）\n");
    part1_constexpr();
    printf("\n");
    part2_ring();
    printf("\n");
    part3_real_copy();
    printf("\n");
    part4_multicast();
    printf("\n总判定: %s (pass=%d fail=%d)\n", g_fail == 0 ? "ALL PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
