// P44-1 实验：DeepGEMM **设备端数学与工具层**（`deep_gemm/include/deep_gemm/common/
//   {math,utils,types,packing,compile,exception,cute_tie}.cuh`）在 H100(sm90) 上
//   **直接编译运行真实设备代码**。
//
// 背景：前面 0~43 篇把一个个 kernel（GEMM / attention / mega moe / mega gate / mHC /
//   backward grouped GEMM）都拆过，但每个 kernel 都在调同一层**没有系统讲过**的设备端
//   数学与工具函数：UE8M0 scale 的整数进位取整、bf16 打包、warp/CTA 归约与前缀和、
//   向量化零值、TMEM 列对齐、GemmType 判定…… 本实验逐一把它们拉到 H100 真机上验证数值
//   与语义。`common/math.cuh` 除 `fma2` 的 SM100 快路（`__ffma2_rn`）外全部架构无关，
//   sm90 可编译可运行。
//
// 覆盖（全部来自真实头文件，非手抄）：
//   A) `get_ue8m0_sf_exp`/`get_ue8m0_sf_inv`（E4M3 与 E2M1、FP32 与 BF16 输入）——
//      与 `deep_gemm/utils/math.py:ceil_to_ue8m0` 参考逐点对拍；SF*SF_inv 精确为 1。
//   B) `get_packed_bf16_amax<kNumPacked>` 的 HMNMX2 树归约。
//   C) `cast_into_bf16_and_pack` / `cast_into_bf16x2`（RNE bf16 打包）。
//   D) `scale_bf16x2_into_fp8x4`（幂次 scale 精确 + E4M3 打包）。
//   E) `fma2`（SM100 `__ffma2_rn` vs sm90 回退）与 `fast_rcp`（rcp.approx.ftz）。
//   F) `warp_inclusive_sum` / `cta_exclusive_sum`（块内互斥前缀和）。
//   G) `warp_reduce<kNumLanesPerGroup, kIntergroupReduce>`（xor 蝶形分组归约）。
//   H) `align`/`ceil_div`/`constexpr_*`、`get_num_aligned_tmem_cols`、`get_smem_pack_factor`、
//      `GemmType` 判定、`Vectorized<kNumBytes>::zeros`。
//
// 编译/运行见 harness/exp_device_math_utils.sh；raw 二进制写 /tmp/dmutils_*.bin，
// 判定在 harness/exp_device_math_utils.sh 内的 Python（用真实 deep_gemm 参考 + torch）。
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/packing.cuh>

using namespace deep_gemm;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1);} } while (0)

static void dump(const char* path, const void* ptr, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("cannot open %s\n", path); std::exit(1); }
    fwrite(ptr, 1, bytes, f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// A) UE8M0 SF 指数：`math.cuh:114-147`
//    `get_ue8m0_sf_exp<quant_dtype_t>` 用整数加法进位实现「把 amax/quant_max
//    向上取到 2 的幂」；`get_ue8m0_sf_inv` 给出倒数（幂次，精确）。
// ---------------------------------------------------------------------------
__global__ void k_sfexp(const float* amax, const nv_bfloat16* amax_bf16, uint32_t n,
                        uint32_t* sf8_exp, uint32_t* sf4_exp, float* sf8_inv, float* sf4_inv,
                        uint32_t* sf8_exp_bf16, uint32_t* sf4_exp_bf16,
                        float* sf8_prod, float* sf4_prod) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = amax[i];
    sf8_exp[i] = math::get_ue8m0_sf_exp<cutlass::float_e4m3_t>(a);
    sf4_exp[i] = math::get_ue8m0_sf_exp<cutlass::float_e2m1_t>(a);
    const float i8 = math::get_ue8m0_sf_inv<float>(sf8_exp[i]);
    const float i4 = math::get_ue8m0_sf_inv<float>(sf4_exp[i]);
    sf8_inv[i] = i8;
    sf4_inv[i] = i4;
    // sf = 2^(exp-127) broadcast as a float, product must be exactly 1
    sf8_prod[i] = __uint_as_float(sf8_exp[i] << 23) * i8;
    sf4_prod[i] = __uint_as_float(sf4_exp[i] << 23) * i4;
    const nv_bfloat16 ab = amax_bf16[i];
    sf8_exp_bf16[i] = math::get_ue8m0_sf_exp<cutlass::float_e4m3_t>(ab);
    sf4_exp_bf16[i] = math::get_ue8m0_sf_exp<cutlass::float_e2m1_t>(ab);
}

// ---------------------------------------------------------------------------
// B) get_packed_bf16_amax<kNumPacked>：`math.cuh:88-103`
// ---------------------------------------------------------------------------
template <uint32_t kNumPacked>
__global__ void k_bf16_amax(const nv_bfloat16* x, uint32_t n,
                            nv_bfloat16* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    nv_bfloat162 vals[kNumPacked];
    #pragma unroll
    for (uint32_t j = 0; j < kNumPacked; ++ j)
        vals[j] = reinterpret_cast<const nv_bfloat162*>(x)[i * kNumPacked + j];
    out[i] = math::get_packed_bf16_amax<kNumPacked>(vals);
}

// ---------------------------------------------------------------------------
// C) cast_into_bf16_and_pack：`math.cuh:75-84`（RNE）
// ---------------------------------------------------------------------------
__global__ void k_cvt_bf16_pack(const float2* x, uint32_t n, int* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = x[i].x, b = x[i].y;
    out[i] = math::cast_into_bf16_and_pack(a, b);
}

// ---------------------------------------------------------------------------
// D) scale_bf16x2_into_fp8x4：`math.cuh:106-109`
// ---------------------------------------------------------------------------
__global__ void k_scale_fp8x4(const nv_bfloat162* lower, const nv_bfloat162* upper,
                              const nv_bfloat162* inv_l, const nv_bfloat162* inv_u,
                              uint32_t n, uint32_t* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = math::scale_bf16x2_into_fp8x4(lower[i], upper[i], inv_l[i], inv_u[i]);
}

// ---------------------------------------------------------------------------
// E) fma2 与 fast_rcp：`math.cuh:57-72`
// ---------------------------------------------------------------------------
__global__ void k_fma2(const float2* a, const float2* b, const float2* c, uint32_t n,
                       float2* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = math::fma2(a[i], b[i], c[i]);
}

__global__ void k_rcp(const float* x, uint32_t n, float* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = math::fast_rcp(x[i]);
}

// ---------------------------------------------------------------------------
// F) warp_inclusive_sum / cta_exclusive_sum：`math.cuh:150-184`
// ---------------------------------------------------------------------------
__global__ void k_scan(const uint32_t* x, uint32_t n, uint32_t* incl, uint32_t* excl, uint32_t* total) {
    const uint32_t i = threadIdx.x;
    const uint32_t v = x[i % n];
    incl[i] = math::warp_inclusive_sum(v, i % 32);
    __shared__ uint32_t warp_sums[8];
    uint32_t tot;
    excl[i] = math::cta_exclusive_sum<256>(v, warp_sums, tot);
    if (i == 0) total[0] = tot;
}

// ---------------------------------------------------------------------------
// G) warp_reduce<kNumLanesPerGroup, kIntergroupReduce>：`math.cuh:200-232`
// ---------------------------------------------------------------------------
template <uint32_t kNumLanesPerGroup, bool kInter>
__global__ void k_warp_reduce(const float* base, float* sum_out, float* max_out) {
    // one warp per block; input value chosen from the lane index
    const uint32_t lane = threadIdx.x;
    const float v = base[lane];
    sum_out[lane] = math::warp_reduce_sum<kNumLanesPerGroup, kInter>(v);
    max_out[lane] = math::warp_reduce_max<kNumLanesPerGroup, kInter>(v);
}

// ---------------------------------------------------------------------------
// H) host/constexpr 工具
// ---------------------------------------------------------------------------
__global__ void k_utils(uint32_t* tmem_cols, uint32_t* vec_bytes, uint32_t* gemm_flags) {
    if (threadIdx.x != 0) return;
    const uint32_t cols[] = {1, 32, 33, 64, 65, 128, 129, 256, 257, 512};
    for (uint32_t i = 0; i < 10; ++ i) {
        switch (cols[i]) {
            case   1: tmem_cols[i] = utils::get_num_aligned_tmem_cols<  1>(); break;
            case  32: tmem_cols[i] = utils::get_num_aligned_tmem_cols< 32>(); break;
            case  33: tmem_cols[i] = utils::get_num_aligned_tmem_cols< 33>(); break;
            case  64: tmem_cols[i] = utils::get_num_aligned_tmem_cols< 64>(); break;
            case  65: tmem_cols[i] = utils::get_num_aligned_tmem_cols< 65>(); break;
            case 128: tmem_cols[i] = utils::get_num_aligned_tmem_cols<128>(); break;
            case 129: tmem_cols[i] = utils::get_num_aligned_tmem_cols<129>(); break;
            case 256: tmem_cols[i] = utils::get_num_aligned_tmem_cols<256>(); break;
            case 257: tmem_cols[i] = utils::get_num_aligned_tmem_cols<257>(); break;
            case 512: tmem_cols[i] = utils::get_num_aligned_tmem_cols<512>(); break;
        }
    }
    vec_bytes[0] = sizeof(utils::Vectorized<16>::vec_t);
    vec_bytes[1] = sizeof(utils::Vectorized<8>::vec_t);
    vec_bytes[2] = sizeof(utils::Vectorized<4>::vec_t);
    vec_bytes[3] = get_smem_pack_factor<cutlass::float_e2m1_t>();
    vec_bytes[4] = get_smem_pack_factor<cutlass::float_e4m3_t>();
    gemm_flags[0] = is_m_grouped_contiguous(GemmType::MGroupedContiguous) &&
                    is_m_grouped_contiguous(GemmType::MGroupedContiguousWithPsumLayout) &&
                    (not is_m_grouped_contiguous(GemmType::KGroupedContiguous));
    gemm_flags[1] = is_k_grouped_contiguous(GemmType::KGroupedContiguous) &&
                    is_k_grouped_contiguous(GemmType::KGroupedContiguousWithPsumLayout) &&
                    (not is_k_grouped_contiguous(GemmType::MGroupedMasked));
}

int main() {
    const uint32_t N = 1u << 16;  // 65536 samples

    // ---- A) SF exponent ----
    std::vector<float> amax(N);
    std::vector<nv_bfloat16> amax_bf16(N);
    uint32_t seed = 12345u;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    for (uint32_t i = 0; i < N; ++ i) {
        float v;
        if (i % 8 == 0) {
            // exact powers of two and boundary mantissas (1.75 / 1.5 family)
            const uint32_t k = rnd() % 40;
            v = std::ldexp(1.0f, (int)k - 20);
        } else if (i % 8 == 1) {
            v = std::ldexp(1.75f, (int)(rnd() % 30) - 15);   // E4M3 boundary
        } else if (i % 8 == 2) {
            v = std::ldexp(1.5f, (int)(rnd() % 30) - 15);    // E2M1 boundary
        } else if (i % 8 == 3) {
            v = 0.0f;                                        // zero
        } else {
            // uniform in [2^-24, 2^10)
            const float u = (float)(rnd() % 1000000u) / 1000000.0f;
            v = std::ldexp(0.5f + u, (int)(rnd() % 34) - 24);
        }
        amax[i] = std::fabs(v);
        amax_bf16[i] = __float2bfloat16_rn(amax[i]);
    }
    float *d_amax; nv_bfloat16 *d_amax_bf16;
    uint32_t *d_sf8e, *d_sf4e, *d_sf8eb, *d_sf4eb;
    float *d_sf8i, *d_sf4i, *d_p8, *d_p4;
    CK(cudaMalloc(&d_amax, N * 4)); CK(cudaMalloc(&d_amax_bf16, N * 2));
    CK(cudaMalloc(&d_sf8e, N * 4)); CK(cudaMalloc(&d_sf4e, N * 4));
    CK(cudaMalloc(&d_sf8eb, N * 4)); CK(cudaMalloc(&d_sf4eb, N * 4));
    CK(cudaMalloc(&d_sf8i, N * 4)); CK(cudaMalloc(&d_sf4i, N * 4));
    CK(cudaMalloc(&d_p8, N * 4)); CK(cudaMalloc(&d_p4, N * 4));
    CK(cudaMemcpy(d_amax, amax.data(), N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_amax_bf16, amax_bf16.data(), N * 2, cudaMemcpyHostToDevice));
    k_sfexp<<<(N + 255) / 256, 256>>>(d_amax, d_amax_bf16, N, d_sf8e, d_sf4e, d_sf8i, d_sf4i,
                                       d_sf8eb, d_sf4eb, d_p8, d_p4);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> sf8e(N), sf4e(N), sf8eb(N), sf4eb(N);
    std::vector<float> sf8i(N), sf4i(N), p8(N), p4(N);
    CK(cudaMemcpy(sf8e.data(), d_sf8e, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sf4e.data(), d_sf4e, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sf8eb.data(), d_sf8eb, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sf4eb.data(), d_sf4eb, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sf8i.data(), d_sf8i, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sf4i.data(), d_sf4i, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(p8.data(), d_p8, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(p4.data(), d_p4, N * 4, cudaMemcpyDeviceToHost));
    dump("/tmp/dmutils_amax.bin", amax.data(), N * 4);
    dump("/tmp/dmutils_amax_bf16.bin", amax_bf16.data(), N * 2);
    dump("/tmp/dmutils_sf8e.bin", sf8e.data(), N * 4);
    dump("/tmp/dmutils_sf4e.bin", sf4e.data(), N * 4);
    dump("/tmp/dmutils_sf8eb.bin", sf8eb.data(), N * 4);
    dump("/tmp/dmutils_sf4eb.bin", sf4eb.data(), N * 4);
    dump("/tmp/dmutils_sf8i.bin", sf8i.data(), N * 4);
    dump("/tmp/dmutils_sf4i.bin", sf4i.data(), N * 4);
    dump("/tmp/dmutils_p8.bin", p8.data(), N * 4);
    dump("/tmp/dmutils_p4.bin", p4.data(), N * 4);

    // ---- B) bf16 packed amax ----
    constexpr uint32_t kPack = 8;   // power of two
    const uint32_t NB = 1u << 15;
    // each element consumes `kPack` packed BF16 pairs = 2*kPack BF16 scalars
    std::vector<nv_bfloat16> bx(NB * kPack * 2);
    for (uint32_t i = 0; i < NB * kPack * 2; ++ i) {
        const float u = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        bx[i] = __float2bfloat16_rn(std::ldexp(u, (int)(rnd() % 20) - 12));
    }
    nv_bfloat16 *d_bx, *d_bamax;
    CK(cudaMalloc(&d_bx, NB * kPack * 2 * 2)); CK(cudaMalloc(&d_bamax, NB * 2));
    CK(cudaMemcpy(d_bx, bx.data(), NB * kPack * 2 * 2, cudaMemcpyHostToDevice));
    k_bf16_amax<kPack><<<(NB + 255) / 256, 256>>>(d_bx, NB, d_bamax);
    CK(cudaDeviceSynchronize());
    std::vector<nv_bfloat16> bamax(NB);
    CK(cudaMemcpy(bamax.data(), d_bamax, NB * 2, cudaMemcpyDeviceToHost));
    dump("/tmp/dmutils_bx.bin", bx.data(), NB * kPack * 2 * 2);
    dump("/tmp/dmutils_bamax.bin", bamax.data(), NB * 2);

    // ---- C) cast_into_bf16_and_pack ----
    const uint32_t NC = 1u << 16;
    std::vector<float2> cx(NC);
    for (uint32_t i = 0; i < NC; ++ i) {
        const float u = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        const float v = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        cx[i] = make_float2(std::ldexp(u, (int)(rnd() % 30) - 15),
                            std::ldexp(v, (int)(rnd() % 30) - 15));
    }
    float2* d_cx; int* d_cout;
    CK(cudaMalloc(&d_cx, NC * 8)); CK(cudaMalloc(&d_cout, NC * 4));
    CK(cudaMemcpy(d_cx, cx.data(), NC * 8, cudaMemcpyHostToDevice));
    k_cvt_bf16_pack<<<(NC + 255) / 256, 256>>>(d_cx, NC, d_cout);
    CK(cudaDeviceSynchronize());
    std::vector<int> cout_(NC);
    CK(cudaMemcpy(cout_.data(), d_cout, NC * 4, cudaMemcpyDeviceToHost));
    dump("/tmp/dmutils_cx.bin", cx.data(), NC * 8);
    dump("/tmp/dmutils_cout.bin", cout_.data(), NC * 4);

    // ---- D) scale_bf16x2_into_fp8x4 ----
    const uint32_t NS = 1u << 15;
    std::vector<nv_bfloat162> lo(NS), up(NS), il(NS), iu(NS);
    std::vector<uint32_t> sfexp_s(NS);
    for (uint32_t i = 0; i < NS; ++ i) {
        const float e = (float)(int)(rnd() % 16) - 8;   // power-of-two inv exponent
        const nv_bfloat16 inv = __float2bfloat16(std::ldexp(1.0f, (int)e));
        il[i] = nv_bfloat162(inv, inv);
        iu[i] = nv_bfloat162(inv, inv);
        const float a = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        const float b = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        const float c = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        const float d = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        // values within +/-1 so scaled magnitudes stay in E4M3 finite range
        lo[i] = nv_bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(b));
        up[i] = nv_bfloat162(__float2bfloat16_rn(c), __float2bfloat16_rn(d));
        sfexp_s[i] = (uint32_t)(int)e;
    }
    nv_bfloat162 *d_lo, *d_up, *d_il, *d_iu; uint32_t* d_sout;
    CK(cudaMalloc(&d_lo, NS * 4)); CK(cudaMalloc(&d_up, NS * 4));
    CK(cudaMalloc(&d_il, NS * 4)); CK(cudaMalloc(&d_iu, NS * 4));
    CK(cudaMalloc(&d_sout, NS * 4));
    CK(cudaMemcpy(d_lo, lo.data(), NS * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_up, up.data(), NS * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_il, il.data(), NS * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_iu, iu.data(), NS * 4, cudaMemcpyHostToDevice));
    k_scale_fp8x4<<<(NS + 255) / 256, 256>>>(d_lo, d_up, d_il, d_iu, NS, d_sout);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> sout(NS);
    CK(cudaMemcpy(sout.data(), d_sout, NS * 4, cudaMemcpyDeviceToHost));
    dump("/tmp/dmutils_lo.bin", lo.data(), NS * 4);
    dump("/tmp/dmutils_up.bin", up.data(), NS * 4);
    dump("/tmp/dmutils_sfexp_s.bin", sfexp_s.data(), NS * 4);
    dump("/tmp/dmutils_sout.bin", sout.data(), NS * 4);

    // ---- E) fma2 / fast_rcp ----
    const uint32_t NF = 1u << 16;
    std::vector<float2> fa(NF), fb(NF), fc(NF);
    std::vector<float> rx(NF);
    for (uint32_t i = 0; i < NF; ++ i) {
        const float u1 = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        const float u2 = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        const float u3 = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        const float u4 = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        const float u5 = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        const float u6 = ((float)(rnd() % 2000000u) / 1000000.0f) - 1.0f;
        fa[i] = make_float2(u1 * 4.0f, u2 * 4.0f);
        fb[i] = make_float2(u3 * 4.0f, u4 * 4.0f);
        fc[i] = make_float2(u5, u6);
        rx[i] = std::ldexp(1.0f + std::fabs(u1), (int)(rnd() % 20) - 10) + 0.01f;
    }
    float2 *d_fa, *d_fb, *d_fc, *d_fout; float *d_rx, *d_rout;
    CK(cudaMalloc(&d_fa, NF * 8)); CK(cudaMalloc(&d_fb, NF * 8)); CK(cudaMalloc(&d_fc, NF * 8));
    CK(cudaMalloc(&d_fout, NF * 8)); CK(cudaMalloc(&d_rx, NF * 4)); CK(cudaMalloc(&d_rout, NF * 4));
    CK(cudaMemcpy(d_fa, fa.data(), NF * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_fb, fb.data(), NF * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_fc, fc.data(), NF * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rx, rx.data(), NF * 4, cudaMemcpyHostToDevice));
    k_fma2<<<(NF + 255) / 256, 256>>>(d_fa, d_fb, d_fc, NF, d_fout);
    k_rcp<<<(NF + 255) / 256, 256>>>(d_rx, NF, d_rout);
    CK(cudaDeviceSynchronize());
    std::vector<float2> fout(NF); std::vector<float> rout(NF);
    CK(cudaMemcpy(fout.data(), d_fout, NF * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rout.data(), d_rout, NF * 4, cudaMemcpyDeviceToHost));
    dump("/tmp/dmutils_fa.bin", fa.data(), NF * 8);
    dump("/tmp/dmutils_fb.bin", fb.data(), NF * 8);
    dump("/tmp/dmutils_fc.bin", fc.data(), NF * 8);
    dump("/tmp/dmutils_fout.bin", fout.data(), NF * 8);
    dump("/tmp/dmutils_rx.bin", rx.data(), NF * 4);
    dump("/tmp/dmutils_rout.bin", rout.data(), NF * 4);

    // ---- F) scan ----
    const uint32_t NW = 256;
    std::vector<uint32_t> scan_in(NW);
    for (uint32_t i = 0; i < NW; ++ i) scan_in[i] = (rnd() % 1000u);
    uint32_t *d_sin, *d_incl, *d_excl, *d_tot;
    CK(cudaMalloc(&d_sin, NW * 4)); CK(cudaMalloc(&d_incl, NW * 4));
    CK(cudaMalloc(&d_excl, NW * 4)); CK(cudaMalloc(&d_tot, 4));
    CK(cudaMemcpy(d_sin, scan_in.data(), NW * 4, cudaMemcpyHostToDevice));
    k_scan<<<1, NW>>>(d_sin, NW, d_incl, d_excl, d_tot);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> incl(NW), excl(NW), tot(1);
    CK(cudaMemcpy(incl.data(), d_incl, NW * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(excl.data(), d_excl, NW * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tot.data(), d_tot, 4, cudaMemcpyDeviceToHost));
    dump("/tmp/dmutils_scan_in.bin", scan_in.data(), NW * 4);
    dump("/tmp/dmutils_incl.bin", incl.data(), NW * 4);
    dump("/tmp/dmutils_excl.bin", excl.data(), NW * 4);
    dump("/tmp/dmutils_total.bin", tot.data(), 4);

    // ---- G) warp_reduce ----
    float base[32];
    for (uint32_t i = 0; i < 32; ++ i) base[i] = (float)((i * 7u + 3u) % 23u) - 11.0f;
    float *d_base; CK(cudaMalloc(&d_base, 32 * 4));
    CK(cudaMemcpy(d_base, base, 32 * 4, cudaMemcpyHostToDevice));
    #define RUN_WR(TAG, ...) do { \
        float *d_s, *d_m; CK(cudaMalloc(&d_s, 32 * 4)); CK(cudaMalloc(&d_m, 32 * 4)); \
        __VA_ARGS__<<<1, 32>>>(d_base, d_s, d_m); \
        CK(cudaDeviceSynchronize()); \
        std::vector<float> s(32), m(32); \
        CK(cudaMemcpy(s.data(), d_s, 32 * 4, cudaMemcpyDeviceToHost)); \
        CK(cudaMemcpy(m.data(), d_m, 32 * 4, cudaMemcpyDeviceToHost)); \
        char p[128]; \
        snprintf(p, sizeof(p), "/tmp/dmutils_wr_%s_sum.bin", TAG); dump(p, s.data(), 32 * 4); \
        snprintf(p, sizeof(p), "/tmp/dmutils_wr_%s_max.bin", TAG); dump(p, m.data(), 32 * 4); \
        cudaFree(d_s); cudaFree(d_m); \
    } while (0)
    RUN_WR("4f", k_warp_reduce<4, false>);
    RUN_WR("8f", k_warp_reduce<8, false>);
    RUN_WR("16f", k_warp_reduce<16, false>);
    RUN_WR("32f", k_warp_reduce<32, false>);
    RUN_WR("4t", k_warp_reduce<4, true>);
    RUN_WR("8t", k_warp_reduce<8, true>);
    RUN_WR("16t", k_warp_reduce<16, true>);
    dump("/tmp/dmutils_wr_base.bin", base, 32 * 4);

    // ---- H) utils / constexpr ----
    uint32_t *d_tc, *d_vb, *d_gf;
    CK(cudaMalloc(&d_tc, 10 * 4)); CK(cudaMalloc(&d_vb, 5 * 4)); CK(cudaMalloc(&d_gf, 2 * 4));
    k_utils<<<1, 32>>>(d_tc, d_vb, d_gf);
    CK(cudaDeviceSynchronize());
    uint32_t tc[10], vb[5], gf[2];
    CK(cudaMemcpy(tc, d_tc, 40, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vb, d_vb, 20, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gf, d_gf, 8, cudaMemcpyDeviceToHost));
    dump("/tmp/dmutils_tmem_cols.bin", tc, 40);
    dump("/tmp/dmutils_vec_bytes.bin", vb, 20);
    dump("/tmp/dmutils_gemm_flags.bin", gf, 8);

    // host-only constexpr checks
    const uint32_t host_checks[8] = {
        math::ceil_div<uint32_t>(1000, 128),          // 8
        math::align<uint32_t>(1000, 128),             // 1024
        math::align<uint32_t, false>(1000, 128),      // 896
        math::constexpr_ceil_div<uint32_t>(1000, 128),
        math::constexpr_align<uint32_t>(1000, 128),
        math::constexpr_gcd<uint32_t>(48, 36),        // 12
        math::constexpr_min<uint32_t>(17, 42),        // 17
        (uint32_t)kNumDeviceLocalityDomains,          // 2
    };
    dump("/tmp/dmutils_host_checks.bin", host_checks, 32);

    cudaFree(d_amax);
    printf("dmutils done N=%u NB=%u NC=%u NS=%u NF=%u NW=%u\n", N, NB, NC, NS, NF, NW);
    return 0;
}
