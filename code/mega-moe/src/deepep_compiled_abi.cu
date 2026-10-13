// P88-1 实验 Part 1：直接 include DeepEP 真实编译期配置头
//   deep_ep/include/deep_ep/common/compiled.cuh
// 验证它的 ABI / 类型映射 / 对齐 / 通信上限常量。
// 同一份源在两个编译单元各跑一次：
//   nvcc ... src/deepep_compiled_abi.cu                    （默认 EP_NUM_TOPK_IDX_BITS=64）
//   nvcc ... -DEP_NUM_TOPK_IDX_BITS=32 src/deepep_compiled_abi.cu （覆盖成 32）
// 注意：这是 host/设备通用头，架构无关，sm90 即可编译运行。
#include <cstdio>
#include <cstdint>
#include <type_traits>

#include <deep_ep/common/compiled.cuh>

using namespace deep_ep;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) { ++g_pass; std::printf("  [PASS] %s\n", msg); }             \
        else      { ++g_fail; std::printf("  [FAIL] %s\n", msg); }             \
    } while (0)

// 编译期断言：类型映射、union 大小、对齐整除关系（与头文件内 static_assert 独立复核）
static_assert(std::is_same_v<int_with_bits<8>::type,  int8_t>,  "int_with_bits<8>");
static_assert(std::is_same_v<int_with_bits<16>::type, int16_t>, "int_with_bits<16>");
static_assert(std::is_same_v<int_with_bits<32>::type, int32_t>, "int_with_bits<32>");
static_assert(std::is_same_v<int_with_bits<64>::type, int64_t>, "int_with_bits<64>");
static_assert(sizeof(sf_pack_t) == 4, "sf_pack_t is a 4-byte union (fp32 / ue8m0x4)");
static_assert(kNumAlignedSFPacks == 16 / sizeof(sf_pack_t), "kNumAlignedSFPacks == 16/sizeof");
static_assert(kNumRDMAAlignmentBytes % kNumTMAAlignmentBytes == 0, "RDMA % TMA");
static_assert(kNumAllocationAlignmentBytes % kNumRDMAAlignmentBytes == 0, "ALLOC % RDMA");

int main() {
    std::printf("== Part 1: compiled.cuh ABI (EP_NUM_TOPK_IDX_BITS=%d) ==\n", EP_NUM_TOPK_IDX_BITS);

    // 1) topk_idx_t 由 EP_NUM_TOPK_IDX_BITS 决定
#if EP_NUM_TOPK_IDX_BITS == 64
    CHECK((std::is_same_v<topk_idx_t, int64_t>), "topk_idx_t == int64_t (default)");
#elif EP_NUM_TOPK_IDX_BITS == 32
    CHECK((std::is_same_v<topk_idx_t, int32_t>), "topk_idx_t == int32_t (overridden)");
#else
#error "unexpected EP_NUM_TOPK_IDX_BITS"
#endif
    std::printf("  sizeof(topk_idx_t) = %zu\n", sizeof(topk_idx_t));

    // 2) int_with_bits 四个特化（运行时打印，编译期已 static_assert）
    CHECK(sizeof(int_with_bits<8>::type)  == 1, "int_with_bits<8>  -> 1 byte");
    CHECK(sizeof(int_with_bits<16>::type) == 2, "int_with_bits<16> -> 2 bytes");
    CHECK(sizeof(int_with_bits<32>::type) == 4, "int_with_bits<32> -> 4 bytes");
    CHECK(sizeof(int_with_bits<64>::type) == 8, "int_with_bits<64> -> 8 bytes");

    // 3) sf_pack_t：UE8M0 打包的最小单元 = 4 字节（4 个 UE8M0 指数挤进一个 int）
    CHECK(sizeof(sf_pack_t) == 4 && alignof(sf_pack_t) == 4, "sf_pack_t == 4B (fp32|ue8m0x4)");
    CHECK(kNumAlignedSFPacks == 4, "kNumAlignedSFPacks == 4");
    std::printf("  fp32=%.1f -> ue8m0x4=0x%08x (bit-reinterpret, size %zu)\n",
                1.0f, (unsigned)[](float f) { sf_pack_t s; s.fp32 = f; return s.ue8m0x4; }(1.0f),
                sizeof(sf_pack_t));

    // 4) 256-bit LD/ST 兼容类型
    std::printf("  CUDART_VERSION=%d longlong4_t: sizeof=%zu alignof=%zu\n",
                CUDART_VERSION, sizeof(longlong4_t), alignof(longlong4_t));
    CHECK(sizeof(longlong4_t) == 32 && alignof(longlong4_t) == 32, "longlong4_t == 32B/32-align");

    // 5) 对齐三档
    std::printf("  align: TMA=%d RDMA=%d ALLOC=%d\n",
                kNumTMAAlignmentBytes, kNumRDMAAlignmentBytes, kNumAllocationAlignmentBytes);
    CHECK(kNumTMAAlignmentBytes == 32 && kNumRDMAAlignmentBytes == 64 &&
          kNumAllocationAlignmentBytes == 2 * 1024 * 1024, "alignment triple 32/64/2MiB");
    CHECK(kNumRDMAAlignmentBytes % kNumTMAAlignmentBytes == 0, "64 % 32 == 0");
    CHECK(kNumAllocationAlignmentBytes % kNumRDMAAlignmentBytes == 0, "2MiB % 64 == 0");

    // 6) 通信上限（对称 buffer 的静态上界）
    std::printf("  limits: maxRanks=%d maxRDMARanks=%d maxNVLRanks=%d scaleout=%d scaleup=%d "
                "ctx=%d QPs=%d QPDepth=%d signalBytes=%d\n",
                kNumMaxRanks, kNumMaxRDMARanks, kNumMaxNVLRanks,
                kNumMaxScaleoutRanks, kNumMaxScaleupRanks,
                kNumMaxContexts, kNumMaxQPs, kDefaultQPDepth, kNumMaxSignalBytes);
    CHECK(kNumMaxRanks == 1024 && kNumMaxRDMARanks == 1024 && kNumMaxNVLRanks == 1024 &&
          kNumMaxScaleoutRanks == 1024 && kNumMaxScaleupRanks == 1024, "rank caps all 1024");
    CHECK(kNumMaxContexts == 8, "kNumMaxContexts == 8");
    CHECK(kNumMaxQPs == 1024 && kDefaultQPDepth == 1024, "QP caps 1024/1024");
    CHECK(kNumMaxSignalBytes == 16 * 1024 * 1024, "kNumMaxSignalBytes == 16 MiB");

    std::printf("Part 1 RESULT: %d PASS / %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
