// P2-2 探针：量化 K1 的 smem 占用与 union 复用收益。
// 对应 csrc/smxx/fwd_kernel1.cuh:5-84 的 K1Layouts / SharedStorageK1。
// 编译（在 kernel_lab 内）：
//   nvcc -std=c++17 -O2 -arch=sm_90a --expt-relaxed-constexpr --expt-extended-lambda \
//     -I <FlashKDA>/cutlass/include -I <FlashKDA>/cutlass/tools/util/include -I <FlashKDA>/csrc \
//     smem_probe.cu -o /tmp/smem_probe && /tmp/smem_probe
#include "smxx/fwd_kernel1.cuh"
#include <cstdio>

int main() {
    using L = K1Layouts<128, 16>;
    using BF16 = cutlass::bfloat16_t;

    auto c = [](auto l) { return (int)cute::cosize_v<decltype(l)>; };

    int qk  = c(typename L::QKLayout{});     // bf16, 16x128
    int gl  = c(typename L::GLayout{});      // fp32, 16x128
    int mma = c(typename L::MMALayout{});    // bf16, 16x128 swizzled
    int lm  = c(typename L::LMLayout{});     // bf16, 16x16
    int lf  = c(typename L::LF32Layout{});   // fp32, 16x16
    int bt  = c(typename L::BetaSmemLayout{});
    int gt  = c(typename L::GTotalLayout{});

    printf("cosize: QK=%d G=%d MMA=%d LM=%d LF32=%d Beta=%d GTotal=%d\n",
           qk, gl, mma, lm, lf, bt, gt);
    printf("bytes : QK=%d G=%d MMA=%d LM=%d LF32=%d Beta=%d GTotal=%d\n",
           qk*2, gl*4, mma*2, lm*2, lf*4, bt*2, gt*4);

    long phaseA = 2L*qk*2 + 1L*gl*4;
    long phaseB = 3L*mma*2 + 1L*lf*4 + 2L*lm*2;
    printf("PhaseA (q,k,g)                     = %ld bytes\n", phaseA);
    printf("PhaseB (k_dec,q_dec,k_inv,L,INV,M) = %ld bytes\n", phaseB);
    printf("union saves min(A,B)               = %ld bytes\n", phaseA < phaseB ? phaseA : phaseB);

    long gbf16_krest = 1L*qk*2;   // union g_bf16 / k_restored
    long dtb_gtotal  = 1L*gt*4;   // union dt_bias / g_total
    long beta_buf    = 1L*bt*2;
    printf("union g_bf16/k_restored            = %ld bytes\n", gbf16_krest);
    printf("union dt_bias/g_total              = %ld bytes\n", dtb_gtotal);
    printf("beta                               = %ld bytes\n", beta_buf);

    printf("\nsizeof(SharedStorageK1<128,16>)    = %zu bytes\n", sizeof(SharedStorageK1<L>));

    // __launch_bounds__(256,8) 可行性：8 CTA/SM
    size_t ss = sizeof(SharedStorageK1<L>);
    printf("\n8 CTA/SM total smem = %zu bytes (%.1f KB)\n", 8 * ss, 8.0 * ss / 1024.0);
    return 0;
}
