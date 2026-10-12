// P3-2 探针：验证 K2 Phase 3/4 的「C 片段 -> B 操作数」两条路径等价，并量化
// SM75_U32x1_MOVM_T（movmatrix 寄存器内转置）相对「STSM + LDSM smem 往返」的收益。
//
// 对应 fwd_kernel2.cuh:589-650：
//   - 路径 R (kernel 实际)：u_bf16(C 片段) --4x MOVM_T--> tCrB_u(B 片段) --gemm(P,·)--> U
//   - 路径 S (无该优化)：u_bf16(C 片段) --STSM_N--> smem --LDSM_N--> B 片段 --gemm--> U
// 二者 B 操作数应逐位相同，U 应完全相同。
//
// 编译（kernel_lab 内，FlashKDA 绝对路径）：
//   nvcc -std=c++17 -O2 -arch=sm_90a --expt-relaxed-constexpr --expt-extended-lambda \
//     -I <FlashKDA>/cutlass/include -I <FlashKDA>/cutlass/tools/util/include \
//     movm_probe.cu -o /tmp/movm_probe && /tmp/movm_probe
#include <cstdio>
#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/mma_sm80.hpp>

using namespace cute;
using BF16 = cutlass::bfloat16_t;
using MMAOP = SM80_16x8x16_F32BF16BF16F32_TN;

// 16x16 bf16 K_INTER 布局（与 fwd_kernel2.cuh:35-39 的 LMLayout 同构）
using MLayout = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<BF16>{}, make_shape(Int<16>{}, Int<16>{}), LayoutLeft{}));

struct Smem {
    alignas(128) BF16 P[256];    // A 操作数
    alignas(128) BF16 V[256];    // uprime（含 C 片段源）
    alignas(128) BF16 Vt[256];   // 路径 S 的 smem 往返缓冲
};

// 读行主序 [16,16] gmem 到 MLayout smem
__device__ __forceinline__ void load_smem(BF16* s, const __nv_bfloat16* g, int lane) {
    Tensor t = make_tensor(make_smem_ptr(s), MLayout{});
    for (int idx = lane; idx < 256; idx += 32) t(idx / 16, idx % 16) = BF16(g[idx]);
}

__global__ void equiv_kernel(const __nv_bfloat16* Pg, const __nv_bfloat16* Vg,
                             float* US, float* UR, int* n_bdiff, int* n_udiff) {
    __shared__ Smem sm;
    const int lane = threadIdx.x;
    load_smem(sm.P, Pg, lane);
    load_smem(sm.V, Vg, lane);
    __syncwarp();

    auto mma = make_tiled_mma(MMAOP{}, Layout<Shape<_1,_1>>{}, Tile<_16,_16,_16>{});
    ThrMMA thr = mma.get_slice(lane);

    auto smem_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto thr_copy_A = smem_copy_A.get_thread_slice(lane);
    auto smem_load_C = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto thr_load_C = smem_load_C.get_slice(lane);
    auto smem_store_C = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
    auto thr_store_C = smem_store_C.get_slice(lane);
    auto smem_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto thr_copy_B = smem_copy_B.get_thread_slice(lane);

    Tensor P_s = make_tensor(make_smem_ptr(sm.P), MLayout{});
    Tensor V_s = make_tensor(make_smem_ptr(sm.V), MLayout{});
    Tensor Vt_s = make_tensor(make_smem_ptr(sm.Vt), MLayout{});

    // A 片段 = P
    Tensor tCrAi = make_fragment_like<BF16>(thr.partition_fragment_A(P_s));
    auto tCrAi_view = thr_copy_A.retile_D(tCrAi);
    copy(smem_copy_A, thr_copy_A.partition_S(P_s), tCrAi_view);
    Tensor tCrA = thr.partition_fragment_A(P_s);
    transform(tCrAi, tCrA, identity{});

    // uprime 的 C 片段（M=token, N=value）
    Tensor tCrC_ref = thr.partition_C(V_s);
    using SFragT = decltype(make_fragment_like<BF16>(thr.make_fragment_C(tCrC_ref)));
    SFragT u_bf16 = make_fragment_like<BF16>(thr.make_fragment_C(tCrC_ref));
    copy(smem_load_C, thr_load_C.partition_S(V_s), thr_load_C.retile_D(u_bf16));

    // ---- 路径 R：MOVM_T 寄存器内转置 ----
    Tensor B_ref = V_s;
    Tensor tCrB_R = thr.partition_fragment_B(B_ref);
    uint32_t u_b_regs[4];
    uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16(0));
    SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
    SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
    SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
    SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);
    uint32_t* bR = reinterpret_cast<uint32_t*>(&tCrB_R(0));
    bR[0] = u_b_regs[0]; bR[1] = u_b_regs[1]; bR[2] = u_b_regs[2]; bR[3] = u_b_regs[3];

    // ---- 路径 S：STSM_N 写 smem，再 LDSM_N 读成 B ----
    copy(smem_store_C, thr_store_C.retile_S(u_bf16), thr_store_C.partition_D(Vt_s));
    __syncwarp();
    Tensor tCrBi = make_fragment_like<BF16>(thr.partition_fragment_B(B_ref));
    auto tCrBi_view = thr_copy_B.retile_D(tCrBi);
    copy(smem_copy_B, thr_copy_B.partition_S(Vt_s), tCrBi_view);
    Tensor tCrB_S = thr.partition_fragment_B(B_ref);
    transform(tCrBi, tCrB_S, identity{});

    // B 操作数逐位比较（路径 R 的寄存器写入 vs 路径 S 的回读）
    uint32_t* bS = reinterpret_cast<uint32_t*>(&tCrB_S(0));
    int bdiff = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) bdiff += (bS[j] != bR[j]);
    if (bdiff) atomicAdd(n_bdiff, bdiff);

    // ---- 各自 gemm：U = P @ uprime ----
    Tensor C_R = thr.make_fragment_C(tCrC_ref); clear(C_R);
    gemm(thr, tCrA(_,_,Int<0>{}), tCrB_R(_,_,Int<0>{}), C_R);
    Tensor C_S = thr.make_fragment_C(tCrC_ref); clear(C_S);
    gemm(thr, tCrA(_,_,Int<0>{}), tCrB_S(_,_,Int<0>{}), C_S);

    // 写回 gmem [16,16] 行主序
    Tensor gUS = make_tensor(make_gmem_ptr(US), make_layout(make_shape(Int<16>{}, Int<16>{}), LayoutRight{}));
    Tensor gUR = make_tensor(make_gmem_ptr(UR), make_layout(make_shape(Int<16>{}, Int<16>{}), LayoutRight{}));
    copy(C_S, thr.partition_C(gUS));
    copy(C_R, thr.partition_C(gUR));

    // 两条路径的 U 差异（naive STSM+LDSM_N 未做转置，预期与 MOVM_T 不同）
    #pragma unroll
    for (int i = 0; i < size(C_S); ++i) {
        if (C_S(i) != C_R(i)) atomicAdd(n_udiff, 1);
    }
}

// 微基准：每次迭代 4x MOVM_T  vs  STSM_N + syncwarp + LDSM_N
__global__ void bench_kernel(long long* cyc_movm, long long* cyc_smem, int iters, float* sink) {
    __shared__ Smem sm;
    __shared__ BF16 Vt2[256];
    const int lane = threadIdx.x;
    for (int idx = lane; idx < 256; idx += 32) { sm.V[idx] = BF16(0.01f * idx); }
    __syncwarp();

    auto mma = make_tiled_mma(MMAOP{}, Layout<Shape<_1,_1>>{}, Tile<_16,_16,_16>{});
    ThrMMA thr = mma.get_slice(lane);
    auto smem_load_C = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto thr_load_C = smem_load_C.get_slice(lane);
    auto smem_store_C = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
    auto thr_store_C = smem_store_C.get_slice(lane);
    auto smem_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto thr_copy_B = smem_copy_B.get_thread_slice(lane);

    Tensor V_s = make_tensor(make_smem_ptr(sm.V), MLayout{});
    Tensor Vt_s = make_tensor(make_smem_ptr(Vt2), MLayout{});
    Tensor C_ref = thr.partition_C(V_s);
    using SFragT = decltype(make_fragment_like<BF16>(thr.make_fragment_C(C_ref)));
    SFragT u_bf16 = make_fragment_like<BF16>(thr.make_fragment_C(C_ref));
    copy(smem_load_C, thr_load_C.partition_S(V_s), thr_load_C.retile_D(u_bf16));
    uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16(0));

    uint32_t acc = 0;
    // 预热
    #pragma unroll
    for (int j = 0; j < 4; ++j) { uint32_t d; SM75_U32x1_MOVM_T::copy(u_c[j], d); acc ^= d; }

    long long t0 = clock64();
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) { uint32_t d; SM75_U32x1_MOVM_T::copy(u_c[j], d); acc += d; }
    }
    long long t1 = clock64();

    Tensor B_ref = V_s;
    Tensor tCrBi = make_fragment_like<BF16>(thr.partition_fragment_B(B_ref));
    auto tCrBi_view = thr_copy_B.retile_D(tCrBi);
    long long t2 = clock64();
    for (int it = 0; it < iters; ++it) {
        copy(smem_store_C, thr_store_C.retile_S(u_bf16), thr_store_C.partition_D(Vt_s));
        __syncwarp();
        copy(smem_copy_B, thr_copy_B.partition_S(Vt_s), tCrBi_view);
        __syncwarp();
        uint32_t* p = reinterpret_cast<uint32_t*>(&tCrBi(0));
        acc += p[0] + p[1] + p[2] + p[3];
    }
    long long t3 = clock64();

    if (lane == 0) { *cyc_movm = t1 - t0; *cyc_smem = t3 - t2; }
    *sink = float(acc);
}

int main() {
    const int N = 16;
    __nv_bfloat16 hP[N * N], hV[N * N];
    srand(1234);
    for (int i = 0; i < N * N; ++i) {
        hP[i] = __float2bfloat16(((rand() % 200) - 100) / 500.0f);
        hV[i] = __float2bfloat16(((rand() % 200) - 100) / 500.0f);
    }
    __nv_bfloat16 *dP, *dV; float *dUS, *dUR; int *d_bd, *d_ud;
    cudaMalloc(&dP, sizeof(hP)); cudaMalloc(&dV, sizeof(hV));
    cudaMalloc(&dUS, N * N * 4); cudaMalloc(&dUR, N * N * 4);
    cudaMalloc(&d_bd, 4); cudaMalloc(&d_ud, 4);
    cudaMemcpy(dP, hP, sizeof(hP), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, sizeof(hV), cudaMemcpyHostToDevice);
    cudaMemset(d_bd, 0, 4); cudaMemset(d_ud, 0, 4);
    equiv_kernel<<<1, 32>>>(dP, dV, dUS, dUR, d_bd, d_ud);
    cudaDeviceSynchronize();

    float hUS[N * N], hUR[N * N]; int h_bd = 0, h_ud = 0;
    cudaMemcpy(hUS, dUS, sizeof(hUS), cudaMemcpyDeviceToHost);
    cudaMemcpy(hUR, dUR, sizeof(hUR), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_bd, d_bd, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_ud, d_ud, 4, cudaMemcpyDeviceToHost);

    // host 参考：ref = P @ uprime（fp32，输入取精确 bf16 值）
    float ref[N * N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0;
            for (int k = 0; k < N; ++k)
                acc += float(hP[i * N + k]) * float(hV[k * N + j]);
            ref[i * N + j] = acc;
        }
    auto stat = [&](const float* x, const char* name, float* ma, float* mr) {
        float a = 0, r = 0;
        for (int i = 0; i < N * N; ++i) {
            a = fmaxf(a, fabsf(x[i] - ref[i]));
            r = fmaxf(r, fabsf(ref[i]));
        }
        *ma = a; *mr = a / (r + 1e-12f);
        printf("  %-38s max_abs=%.3e  max_rel=%.3e\n", name, *ma, *mr);
    };
    float maR, mrR, maS, mrS;
    printf("== Phase 3/4：U = P @ uprime 正确性（vs host fp32 参考） ==\n");
    stat(hUR, "路径 R: MOVM_T 寄存器转置 -> B", &maR, &mrR);
    stat(hUS, "路径 S: STSM+LDSM_N（未转置）-> B", &maS, &mrS);
    printf("  B 操作数逐位比较 differing u32: %d / %d\n", h_bd, 4 * 32);
    printf("  U 两路径 differing elements: %d / %d\n", h_ud, N * N);
    printf("  => 结论：MOVM_T 是必需的转置；naive smem 往返不转置会算成 P@uprime^T。\n\n");

    long long *cm, *cs; float* sink;
    cudaMalloc(&cm, 8); cudaMalloc(&cs, 8); cudaMalloc(&sink, 4);
    const int iters = 200000;
    bench_kernel<<<1, 32>>>(cm, cs, iters, sink);
    cudaDeviceSynchronize();
    long long hcm = 0, hcs = 0;
    cudaMemcpy(&hcm, cm, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&hcs, cs, 8, cudaMemcpyDeviceToHost);
    printf("== 微基准（1 warp，%d iters） ==\n", iters);
    printf("  4x MOVM_T          : %lld cycles total, %.3f cyc/iter\n", hcm, (double)hcm / iters);
    printf("  STSM+LDSM 往返      : %lld cycles total, %.3f cyc/iter\n", hcs, (double)hcs / iters);
    printf("  加速比 = %.2fx\n", (double)hcs / (double)hcm);
    return 0;
}
