// P25-1 实验：DeepGEMM **SM100 FP8/FP4 稠密 GEMM 主 kernel**
//   （`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`）里
//   **架构无关的几何/描述符设备代码**在 H100(sm90) 上直接跑真机验证。
//
// 背景：SM100 主 kernel（含 Mega MoE / 全部 SM100 GEMM 的统一后端）在 sm90 上跑不了
//   （`sm100_fp8_fp4_gemm_1d1d.cuh:47` 整段 `#if __CUDA_ARCH__ >= 1000`；UMMA/UTCCP/TMEM 皆 SM100）。
//   但 kernel 里有几块**纯整数/共享内存**的设备代码与 UMMA 硬件无关，可原样抽取后在 H100 运行：
//     A) `utccp_required_smem_warp_transpose`（kernel:464-472）—— 把 SF 从 TMA 落下的
//        `[4][32]` 行主序块在 smem 里**原地转成 UTCCP 需要的 `[32][4]`**（每 lane 连续 4 个）。
//     B) `advance_umma_desc_lo` / `get_umma_desc_stride_k` / `get_inner_block_atom_size`
//        （`mma/sm100.cuh:83-96`、`common/tma_copy.cuh:13`）—— UMMA smem 描述符的
//        **字节偏移推进**（K-major 步长 1、MN-major 步长 = swizzle atom），FP4 还要 /pack_factor。
//     C) `make_umma_desc` / `make_smem_desc`（`mma/sm100.cuh:14-142`）—— 描述符位布局
//        （start_address/SBO/LBO/layout_type）。
//     D) `make_runtime_instr_desc_with_sf_id` / `update_instr_desc_with_umma_n`
//        （`mma/sm100.cuh:144-158`）+ cutlass `make_instr_desc_block_scaled`—— **指令描述符**
//        的 scale-format / n_dim / sf_id 位布局（kernel:305-315/352-356/410-444 用到）。
//
// 本实验把上述**真实设备函数**（或 cutlass 真实类型）编译进 sm90 kernel 运行，输出原始数值，
// 由 harness 的 Python 侧独立复刻并逐项对拍。见 harness/exp_sm100_fp8_fp4_geometry.sh。
//
// 编译/运行见 harness/exp_sm100_fp8_fp4_geometry.sh（nvcc -arch=sm_90 --expt-relaxed-constexpr）。
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

using namespace deep_gemm;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1);} } while (0)

// ---------------------------------------------------------------------------
// A) SF warp 内原地转置：kernel:465-471 逐行原样抽取到 /tmp/utccp_body.inc，
//    包进一个设备函数（其余为 harness 从源码 sed 抽取，非手抄）。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void utccp_warp_transpose(const uint32_t* smem_ptr, uint32_t lane_idx) {
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
#include "/tmp/utccp_body.inc"
}

__global__ void probe_transpose(const uint32_t* in, uint32_t* out) {
    extern __shared__ uint32_t sm[];
    const uint32_t lane = threadIdx.x & 31;
    for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x)
        sm[i] = in[i];
    __syncthreads();
    if (lane < 32)
        utccp_warp_transpose(sm, lane);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x)
        out[i] = sm[i];
}

// ---------------------------------------------------------------------------
// B)/C)/D) 描述符几何：单线程把结果写进 out[]，host 侧按固定 layout 解读。
// ---------------------------------------------------------------------------
// out 布局（uint32）：
//   [0]        n_desc_entries
//   [1 + 6*i]  case i: lo
//   [2 + 6*i]  case i: hi
//   [3 + 6*i]  case i: advance_lo(offset=0, k=0)
//   [4 + 6*i]  case i: advance_lo(offset=0, k=1)
//   [5 + 6*i]  case i: advance_lo(offset=0, k=2)
//   [6 + 6*i]  case i: advance_lo(offset=32, k=0)
static __device__ void push_desc(uint32_t* out, uint32_t& idx, cute::UMMA::SmemDescriptor d,
                                 uint32_t adv0, uint32_t adv1, uint32_t adv2, uint32_t adv32) {
    out[1 + 6 * idx] = d.lo;
    out[2 + 6 * idx] = d.hi;
    out[3 + 6 * idx] = adv0;
    out[4 + 6 * idx] = adv1;
    out[5 + 6 * idx] = adv2;
    out[6 + 6 * idx] = adv32;
    ++idx;
}

__global__ void probe_desc(uint32_t* out) {
    extern __shared__ __align__(1024) uint8_t smem[];
    uint32_t idx = 0;
    using cute::UMMA::Major;

    // case 0: K-major, BLOCK_MN=128, swizzle=128, FP8 (pack=1)
    {
        using dt = cutlass::float_e4m3_t;
        constexpr uint32_t BM = 128, BK = 128, SW = 128;
        auto d = mma::sm100::make_umma_desc<Major::K, BM, BK, SW>(reinterpret_cast<dt*>(smem), 0, 0);
        uint32_t b = d.lo;
        push_desc(out, idx, d,
            mma::sm100::advance_umma_desc_lo<Major::K, BM, SW, dt>(b, 0, 0),
            mma::sm100::advance_umma_desc_lo<Major::K, BM, SW, dt>(b, 0, 1),
            mma::sm100::advance_umma_desc_lo<Major::K, BM, SW, dt>(b, 0, 2),
            mma::sm100::advance_umma_desc_lo<Major::K, BM, SW, dt>(b, 32, 0));
    }
    // case 1: MN-major, BLOCK_MN=128, swizzle=128, FP8
    {
        using dt = cutlass::float_e4m3_t;
        constexpr uint32_t BM = 128, BK = 128, SW = 128;
        auto d = mma::sm100::make_umma_desc<Major::MN, BM, BK, SW>(reinterpret_cast<dt*>(smem), 0, 0);
        uint32_t b = d.lo;
        push_desc(out, idx, d,
            mma::sm100::advance_umma_desc_lo<Major::MN, BM, SW, dt>(b, 0, 0),
            mma::sm100::advance_umma_desc_lo<Major::MN, BM, SW, dt>(b, 0, 1),
            mma::sm100::advance_umma_desc_lo<Major::MN, BM, SW, dt>(b, 0, 2),
            mma::sm100::advance_umma_desc_lo<Major::MN, BM, SW, dt>(b, 32, 0));
    }
    // case 2: K-major, BLOCK_MN=128, swizzle=64, FP4 (pack=2；K-major FP4 满足
    //         `kSwizzleMode*pack == BLOCK_K*sizeof` -> 64*2 == 128*1)
    {
        using dt = cutlass::float_e2m1_t;
        constexpr uint32_t BM = 128, BK = 128, SW = 64;
        auto d = mma::sm100::make_umma_desc<Major::K, BM, BK, SW>(reinterpret_cast<dt*>(smem), 0, 0);
        uint32_t b = d.lo;
        push_desc(out, idx, d,
            mma::sm100::advance_umma_desc_lo<Major::K, BM, SW, dt>(b, 0, 0),
            mma::sm100::advance_umma_desc_lo<Major::K, BM, SW, dt>(b, 0, 1),
            mma::sm100::advance_umma_desc_lo<Major::K, BM, SW, dt>(b, 0, 2),
            mma::sm100::advance_umma_desc_lo<Major::K, BM, SW, dt>(b, 32, 0));
    }
    // case 3: MN-major, BLOCK_MN=256, swizzle=64, FP8
    {
        using dt = cutlass::float_e4m3_t;
        constexpr uint32_t BM = 256, BK = 128, SW = 64;
        auto d = mma::sm100::make_umma_desc<Major::MN, BM, BK, SW>(reinterpret_cast<dt*>(smem), 0, 0);
        uint32_t b = d.lo;
        push_desc(out, idx, d,
            mma::sm100::advance_umma_desc_lo<Major::MN, BM, SW, dt>(b, 0, 0),
            mma::sm100::advance_umma_desc_lo<Major::MN, BM, SW, dt>(b, 0, 1),
            mma::sm100::advance_umma_desc_lo<Major::MN, BM, SW, dt>(b, 0, 2),
            mma::sm100::advance_umma_desc_lo<Major::MN, BM, SW, dt>(b, 32, 0));
    }
    out[0] = idx;
}

// ---------------------------------------------------------------------------
// D) 指令描述符：真实 cutlass `make_instr_desc_block_scaled` + DeepGEMM 的
//    `make_runtime_instr_desc_with_sf_id`。单线程写 out[]（每项 4 个 uint32：
//    {raw32, sfa_id, sfb_id, n_dim_after_update}）。
// ---------------------------------------------------------------------------
template <typename a_t, typename b_t, uint32_t M, uint32_t N, cute::UMMA::Major MA, cute::UMMA::Major MB>
__device__ void push_instr(uint32_t* out, uint32_t& idx, const uint32_t sfa, const uint32_t sfb) {
    auto instr = cute::UMMA::make_instr_desc_block_scaled<a_t, b_t, float, cutlass::float_ue8m0_t, M, N, MA, MB>();
    const uint64_t packed = mma::sm100::make_runtime_instr_desc_with_sf_id(instr, sfa, sfb);
    const uint32_t lo32 = static_cast<uint32_t>(packed & 0xffffffffu);
    const uint32_t hi32 = static_cast<uint32_t>(packed >> 32);

    auto instr2 = cute::UMMA::make_instr_desc_block_scaled<a_t, b_t, float, cutlass::float_ue8m0_t, M, N, MA, MB>();
    mma::sm100::update_instr_desc_with_umma_n(instr2, 96);
    out[1 + 4 * idx] = hi32;
    out[2 + 4 * idx] = sfa | (sfb << 8);       // 输入 id（打包便于对拍）
    out[3 + 4 * idx] = lo32;                    // 必须为 0
    out[4 + 4 * idx] = instr2.n_dim_;            // update 后 n_dim_ == 96>>3 == 12
    ++idx;
}

__global__ void probe_instr(uint32_t* out) {
    uint32_t idx = 0;
    using cute::UMMA::Major;
    // MXF8F6F4：UMMA_M=128/256, UMMA_N=128；A/B 可为 E4M3/E2M1
    push_instr<cutlass::float_e4m3_t, cutlass::float_e4m3_t, 256, 128, Major::K, Major::K>(out, idx, 0, 0);
    push_instr<cutlass::float_e4m3_t, cutlass::float_e4m3_t, 256, 128, Major::K, Major::K>(out, idx, 1, 2);
    push_instr<cutlass::float_e4m3_t, cutlass::float_e2m1_t, 256, 128, Major::K, Major::K>(out, idx, 3, 3);
    push_instr<cutlass::float_e4m3_t, cutlass::float_e4m3_t, 128,  64, Major::MN, Major::K>(out, idx, 2, 1);
    // MXF4：M=128, N=128, K-major only
    push_instr<cutlass::float_e2m1_t, cutlass::float_e2m1_t, 128, 128, Major::K, Major::K>(out, idx, 3, 0);
    out[0] = idx;
}

// ---------------------------------------------------------------------------

int main() {
    const int N = 512;
    std::vector<uint32_t> host(N, 0);

    // --- A) transpose ---
    {
        std::vector<uint32_t> in(128);
        for (uint32_t i = 0; i < 128; ++i) in[i] = 0xAAAA0000u ^ (i * 2654435761u);
        uint32_t *din, *dout;
        CK(cudaMalloc(&din, 128 * 4)); CK(cudaMalloc(&dout, 128 * 4));
        CK(cudaMemcpy(din, in.data(), 128 * 4, cudaMemcpyHostToDevice));
        CK(cudaFuncSetAttribute(probe_transpose, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096));
        probe_transpose<<<1, 32, 4096>>>(din, dout);
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> out(128);
        CK(cudaMemcpy(out.data(), dout, 128 * 4, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < 128; ++i) printf("TR %u %u\n", i, out[i]);
        cudaFree(din); cudaFree(dout);
    }
    // --- B/C) descriptors ---
    {
        uint32_t* dout;
        CK(cudaMalloc(&dout, N * 4));
        CK(cudaMemset(dout, 0, N * 4));
        CK(cudaFuncSetAttribute(probe_desc, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024));
        probe_desc<<<1, 32, 128 * 1024>>>(dout);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(host.data(), dout, N * 4, cudaMemcpyDeviceToHost));
        const uint32_t ncase = host[0];
        printf("DESC_NCASE %u\n", ncase);
        for (uint32_t i = 0; i < ncase; ++i) {
            printf("DESC %u %u %u %u %u %u %u\n", i,
                   host[1 + 6 * i], host[2 + 6 * i], host[3 + 6 * i],
                   host[4 + 6 * i], host[5 + 6 * i], host[6 + 6 * i]);
        }
        cudaFree(dout);
    }
    // --- D) instr descriptors ---
    {
        uint32_t* dout;
        CK(cudaMalloc(&dout, N * 4));
        CK(cudaMemset(dout, 0, N * 4));
        probe_instr<<<1, 32>>>(dout);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(host.data(), dout, N * 4, cudaMemcpyDeviceToHost));
        const uint32_t ncase = host[0];
        printf("INSTR_NCASE %u\n", ncase);
        for (uint32_t i = 0; i < ncase; ++i) {
            printf("INSTR %u %u %u %u %u\n", i,
                   host[1 + 4 * i], host[2 + 4 * i], host[3 + 4 * i], host[4 + 4 * i]);
        }
    }
    printf("DONE\n");
    return 0;
}
