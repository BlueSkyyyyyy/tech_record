// P3-1 探针：量化 K2 的 smem 占用、pipeline stage 预算与 union 复用。
// 对应 csrc/smxx/fwd_kernel2.cuh:9-112 的 K2Layouts / SharedStorageK2。
// 编译（在 kernel_lab 内，FlashKDA 绝对路径）：
//   nvcc -std=c++17 -O2 -arch=sm_90a --expt-relaxed-constexpr --expt-extended-lambda \
//     -I <FlashKDA>/cutlass/include -I <FlashKDA>/cutlass/tools/util/include -I <FlashKDA>/csrc \
//     smem_k2_probe.cu -o /tmp/smem_k2_probe && /tmp/smem_k2_probe
#include "smxx/fwd_kernel2.cuh"
#include <cstdio>

int main() {
    using L = K2Layouts<128, 16>;
    using BF16 = cutlass::bfloat16_t;

    auto c = [](auto l) { return (int)cute::cosize_v<decltype(l)>; };

    int vol = c(typename L::VOLayout{});                 // bf16, 16x128 (== MMALayout)
    int bt  = c(typename L::BetaSmemLayout{});           // bf16, 32
    int st  = c(typename L::StateSmemLayout{});          // bf16, 128x128 swizzled
    int gt  = c(typename L::GTotalLayout{});             // fp32, 128
    int lm  = c(typename L::LMLayout{});                 // bf16, 16x16

    printf("cosize: VO=%d Beta=%d State=%d GTotal=%d LM=%d\n", vol, bt, st, gt, lm);
    printf("bytes : VO=%d Beta=%d State=%d GTotal=%d LM=%d\n",
           vol*2, bt*2, st*2, gt*4, lm*2);

    long input_per_stage = 4L*vol*2 + 1L*bt*2 + 1L*gt*4 + 2L*lm*2;
    // v + k_decayed + q_decayed + k_restored = 4x VO; beta; g_total; INV+Mqk
    long output_per_stage = 1L*vol*2;
    printf("\nInputStorage per stage  = 4*%d + %d + %d + 2*%d = %ld bytes\n",
           vol*2, bt*2, gt*4, lm*2, input_per_stage);
    printf("OutputStorage per stage = %ld bytes\n", output_per_stage);

    constexpr int In = 3, Out = 2;
    long pipeline_region = In*input_per_stage + Out*output_per_stage;
    long fp32_state_buf  = 1L*st*4;   // state_fp32_buf = cosize(StateSmemLayout)*sizeof(float)
    printf("\nInput[3]+Output[2]      = %ld bytes\n", pipeline_region);
    printf("state_fp32_buf          = %ld bytes\n", fp32_state_buf);
    printf("union takes max         = %ld bytes (winner: %s)\n",
           pipeline_region > fp32_state_buf ? pipeline_region : fp32_state_buf,
           pipeline_region > fp32_state_buf ? "pipeline" : "fp32_state");

    printf("state_acc (outside union)= %ld bytes\n", 1L*st*2);
    printf("sizeof(SharedStorageK2<L,3,2>) = %zu bytes\n",
           sizeof(SharedStorageK2<L, In, Out>));

    // kTmaTransactionBytes (fwd_kernel2.cuh:172-181) = v + beta + 3*MMA + g_total + 2*LM
    long txn = 1L*vol*2 + 32L*2 + 3L*vol*2 + 1L*gt*4 + 2L*lm*2;
    printf("\nkTmaTransactionBytes per stage = %ld bytes\n", txn);

    // warp partition: kComputeThreads=128, +1 LOAD +1 STORE (fwd_kernel2.cuh:169-197)
    printf("warps: MMA=4 (threads 0-127) LOAD_QKG=1 (128-159) STORE=1 (160-191), total=192\n");

    size_t ss = sizeof(SharedStorageK2<L, In, Out>);
    printf("\n1 CTA/SM smem  = %zu bytes (%.1f KB); H100 max=228KB -> %s\n",
           ss, ss / 1024.0, ss <= 228*1024 ? "fits" : "no");
    printf("2 CTA/SM smem  = %zu bytes (%.1f KB)\n", 2*ss, 2.0*ss/1024.0);
    return 0;
}
