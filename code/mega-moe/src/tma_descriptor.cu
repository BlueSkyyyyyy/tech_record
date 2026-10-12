// P46-1 DeepGEMM 的 TMA 描述符构造层 csrc/jit_kernels/impls/runtime_utils.hpp
// 真机实验（H100 sm90）：
//   Part 1 参数翻译等价 —— 独立复刻 make_tma_* 的编码参数，直连 cuTensorMapEncodeTiled，逐字节对比
//   Part 2 真实 TMA 载入   —— 用真实 make_tma_2d_desc 造描述符，发 cp.async.bulk.tensor.2d，
//                            与 cute::Swizzle 预测的物理 smem 布局逐字节对比（swizzle 0/32/64/128）
//   Part 3 描述符字节指纹 —— 变一个输入，看 128 字节里哪些字/位变化（反推字段布局）
//
// 编译：见 harness/exp_tma_descriptor.sh（nvcc 直接 include 真实 runtime_utils.hpp + torch）
#include <torch/python.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cute/swizzle.hpp>

#include "jit_kernels/impls/runtime_utils.hpp"

using deep_gemm::make_tma_2d_desc;
using deep_gemm::make_tma_3d_desc;
using deep_gemm::make_tma_4d_desc;
using deep_gemm::make_tma_a_desc;
using deep_gemm::make_tma_b_desc;
using deep_gemm::make_tma_cd_desc;
using deep_gemm::make_tma_sf_desc;
using deep_gemm::make_tma_weights_2d_desc;
using deep_gemm::make_tma_weights_3d_desc;
using deep_gemm::get_inner_outer_dims;

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& tag) {
    if (ok) { g_pass++; } else { g_fail++; printf("  !! FAIL %s\n", tag.c_str()); }
}

static std::string hex128(const CUtensorMap& m) {
    const auto* p = reinterpret_cast<const unsigned char*>(&m);
    std::string s;
    for (int i = 0; i < 128; i++) { char b[4]; snprintf(b, 4, "%02x", p[i]); s += b; }
    return s;
}
static bool eq128(const CUtensorMap& a, const CUtensorMap& b) {
    return memcmp(&a, &b, 128) == 0;
}
static std::string diff_words(const CUtensorMap& a, const CUtensorMap& b) {
    const auto* pa = reinterpret_cast<const uint64_t*>(&a);
    const auto* pb = reinterpret_cast<const uint64_t*>(&b);
    std::string s;
    for (int i = 0; i < 16; i++) {
        if (pa[i] != pb[i]) { char buf[64]; snprintf(buf, 64, "w%d:%016llx->%016llx ", i,
            (unsigned long long)pa[i], (unsigned long long)pb[i]); s += buf; }
    }
    return s;
}

// ---------------------------------------------------------------------------
// 独立复刻：按 rank 直接调 driver 编码（不做任何 make_tma_* 的封装逻辑）
// ---------------------------------------------------------------------------
static CUtensorMap ref_encode(const torch::Tensor& t, const int rank,
                              const std::vector<cuuint64_t>& gd,
                              const std::vector<cuuint64_t>& gs_elem,   // 元素单位，rank-1 个
                              const std::vector<cuuint32_t>& sd,
                              const int swz, const int base,
                              const bool tf32, const bool fp4_un) {
    const int elem = static_cast<int>(t.element_size());
    std::vector<cuuint64_t> gsb;
    for (auto s : gs_elem) gsb.push_back(s * static_cast<cuuint64_t>(elem));
    std::vector<cuuint32_t> es(rank, 1);
    CUtensorMap m;
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &m, deep_gemm::aten_dtype_to_tensor_map_dtype(t.scalar_type(), tf32, fp4_un),
        rank, t.data_ptr(), gd.data(), rank > 1 ? gsb.data() : nullptr, sd.data(), es.data(),
        CU_TENSOR_MAP_INTERLEAVE_NONE, deep_gemm::mode_into_tensor_map_swizzle(swz, base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return m;
}

// make_tma_2d_desc 的输入规整（swizzle 覆盖 inner、FP4 特殊）
static std::vector<cuuint32_t> ref_2d_smem(const torch::Tensor& t, int si, int so, int swz, bool fp4_un) {
    const int elem = static_cast<int>(t.element_size());
    if (swz != 0) si = swz / elem;
    if (t.scalar_type() == deep_gemm::kPackedFP4 and not fp4_un and swz != 0) si = swz * 2;
    return {(cuuint32_t)si, (cuuint32_t)so};
}

// ---------------------------------------------------------------------------
// Part 1：参数翻译等价
// ---------------------------------------------------------------------------
static void part1_equivalence() {
    printf("== Part 1: 参数翻译等价（wrapper vs 直连编码，128 字节逐字节）==\n");
    auto bf16 = torch::ones({64, 128}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    auto fp32 = torch::ones({64, 128}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto e4m3 = torch::ones({64, 128}, torch::dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
    auto fp4  = torch::ones({64, 128}, torch::dtype(torch::kInt8).device(torch::kCUDA));  // kPackedFP4

    // ---- make_tma_2d_desc：多种 swizzle / dtype / base ----
    struct C2 { const char* name; torch::Tensor* t; int gi, go, si, so, stride, swz, base; bool tf32, fp4un; };
    std::vector<C2> c2 = {
        {"2d_bf16_sw128",  &bf16, 128, 64, 0,  8, 128, 128, 0, false, true},
        {"2d_bf16_sw64",   &bf16, 128, 64, 0,  8, 128,  64, 0, false, true},
        {"2d_bf16_sw32",   &bf16, 128, 64, 0,  8, 128,  32, 0, false, true},
        {"2d_bf16_sw0",    &bf16, 128, 64, 64, 8, 128,   0, 0, false, true},
        {"2d_bf16_sw16",   &bf16, 128, 64, 0,  8, 128,  16, 0, false, true},
        {"2d_fp32_tf32",   &fp32, 128, 64, 0,  8, 128, 128, 0, true,  true},
        {"2d_e4m3_sw128",  &e4m3, 128, 64, 0,  8, 128, 128, 0, false, true},
        {"2d_sw128_base32",&bf16, 128, 64, 0,  8, 128, 128, 32, false, true},  // SM100 专属 swizzle
    };
    for (auto& c : c2) {
        CUtensorMap got;
        try {
            got = make_tma_2d_desc(*c.t, c.gi, c.go, c.si, c.so, c.stride, c.swz, c.base, c.tf32, c.fp4un);
        } catch (const std::exception& e) {
            printf("  %-18s THREW (%s)  [base=%d 的 ATOM swizzle 在 H100 被驱动拒绝]\n", c.name,
                   std::string(e.what()).substr(0, 40).c_str(), c.base);
            continue;
        }
        auto sd = ref_2d_smem(*c.t, c.si, c.so, c.swz, c.fp4un);
        auto ref = ref_encode(*c.t, 2, {(cuuint64_t)c.gi, (cuuint64_t)c.go}, {(cuuint64_t)c.stride}, sd,
                              c.swz, c.base, c.tf32, c.fp4un);
        check(eq128(got, ref), std::string(c.name));
        printf("  %-18s %s  %s\n", c.name, eq128(got, ref) ? "MATCH" : "DIFF",
               eq128(got, ref) ? hex128(got).substr(0, 32).c_str() : diff_words(got, ref).c_str());
    }

    // ---- make_tma_a_desc / make_tma_b_desc：major K/MN，num_groups ----
    struct CAB { const char* name; int is_a; int major; int block, k, groups, swz; };
    std::vector<CAB> cab = {
        {"a_K_g1_sw128", 1, 0 /*K*/, 128, 64, 1, 128},
        {"a_MN_g1_sw128",1, 1 /*MN*/,128, 64, 1, 128},
        {"a_K_g2_sw128", 1, 0,       128, 64, 2, 128},
        {"b_K_g2_sw128", 0, 0,       128, 64, 2, 128},
        {"b_MN_g1_sw128",0, 1,       128, 64, 1, 128},
    };
    for (auto& c : cab) {
        auto major = c.major == 0 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
        CUtensorMap got;
        if (c.is_a) got = make_tma_a_desc(major, bf16, c.block, c.k, c.block, c.k, c.k, c.groups, c.swz);
        else         got = make_tma_b_desc(major, bf16, c.block, c.k, c.block, c.k, c.k, c.groups, c.swz);
        auto [gi, go] = get_inner_outer_dims(major, c.k, c.is_a ? c.block * c.groups : c.block);
        if (not c.is_a) go *= c.groups;
        auto [si, so] = get_inner_outer_dims(major, c.k, c.block);
        auto sd = ref_2d_smem(bf16, si, so, c.swz, true);
        auto ref = ref_encode(bf16, 2, {(cuuint64_t)gi, (cuuint64_t)go}, {(cuuint64_t)c.k}, sd, c.swz, 0, false, true);
        check(eq128(got, ref), std::string(c.name));
        printf("  %-18s %s\n", c.name, eq128(got, ref) ? "MATCH" : "DIFF");
    }

    // ---- make_tma_cd_desc：输出恒 N-major ----
    {
        auto got = make_tma_cd_desc(bf16, 256, 128, 128, 128, 128, 2, 128);
        auto sd = ref_2d_smem(bf16, 128, 128, 128, true);
        auto ref = ref_encode(bf16, 2, {128, 512}, {128}, sd, 128, 0, false, true);
        check(eq128(got, ref), "cd_Nmajor_g2");
        printf("  %-18s %s\n", "cd_Nmajor_g2", eq128(got, ref) ? "MATCH" : "DIFF");
    }

    // ---- make_tma_sf_desc：仅 MN，无 swizzle，按 gran_k 分组 ----
    {
        // fp32 SF: gran_k 有效 = gran_k * 1
        auto got = make_tma_sf_desc(cute::UMMA::Major::MN, fp32, 131, 512, 128, 128, 1, 0);
        const int shape_mn = deep_gemm::get_tma_aligned_size(131, 4);
        auto ref = ref_encode(fp32, 2, {(cuuint64_t)shape_mn, (cuuint64_t)deep_gemm::ceil_div(512, 128)},
                              {(cuuint64_t)shape_mn}, {(cuuint32_t)128, (cuuint32_t)1}, 0, 0, false, true);
        check(eq128(got, ref), "sf_mn_fp32");
        printf("  %-18s %s (shape_mn=%d)\n", "sf_mn_fp32", eq128(got, ref) ? "MATCH" : "DIFF", shape_mn);
    }

    // ---- dtype → tensor map dtype 枚举（含 FP4 的 unpacked/packed 两态）----
    {
        auto fp4 = torch::ones({64, 128}, torch::dtype(torch::kInt8).device(torch::kCUDA));
        struct DM { const char* n; at::ScalarType dt; bool tf32; bool un; CUtensorMapDataType exp; };
        std::vector<DM> dm = {
            {"bf16",   torch::kBFloat16,      false, true,  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16},
            {"fp32",   torch::kFloat,         false, true,  CU_TENSOR_MAP_DATA_TYPE_FLOAT32},
            {"tf32",   torch::kFloat,         true,  true,  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32},
            {"e4m3",   torch::kFloat8_e4m3fn, false, true,  CU_TENSOR_MAP_DATA_TYPE_UINT8},
            {"fp4_un", deep_gemm::kPackedFP4, false, true,  CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B},
            {"fp4_pk", deep_gemm::kPackedFP4, false, false, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B},
        };
        for (auto& d : dm) {
            auto got = deep_gemm::aten_dtype_to_tensor_map_dtype(d.dt, d.tf32, d.un);
            check(got == d.exp, std::string("dtype_") + d.n);
            printf("  %-18s dtype_enum=%d\n", d.n, (int)got);
        }
    }

    // ---- make_tma_weights_2d_desc：4D (k, n_per_domain, num_domains, 1) ----
    {
        const int nd = deep_gemm::LocalityDomainAllocator::get_num_locality_domains();
        auto w = torch::ones({256, 128}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));  // (N,K)
        auto got = make_tma_weights_2d_desc(w, 64 /*block_k*/, 128 /*load_block_n*/, 128);
        const int n_per = 256 / nd;
        std::vector<cuuint64_t> gd = {128, (cuuint64_t)n_per, (cuuint64_t)nd, 1};
        std::vector<cuuint64_t> gse = {128, (cuuint64_t)(n_per * 128), (cuuint64_t)(n_per * 128 * nd)};
        std::vector<cuuint32_t> sd = {64, 128, 1, 1};
        auto ref = ref_encode(w, 4, gd, gse, sd, 128, 0, false, true);
        check(eq128(got, ref), "weights2d_4d");
        printf("  %-18s %s (num_domains=%d)\n", "weights2d_4d", eq128(got, ref) ? "MATCH" : "DIFF", nd);
    }
}

// ---------------------------------------------------------------------------
// Part 2：真实 TMA 载入，对比 cute::Swizzle 预测的物理 smem 布局
// ---------------------------------------------------------------------------
__global__ void tma_load_kernel(const __grid_constant__ CUtensorMap tmap,
                                int box_inner, int box_outer, int elem_bytes,
                                int c0, int c1, int smem_off, unsigned char* out) {
    __shared__ __align__(1024) unsigned char smem[4096];   // swizzle 的 XOR 键含 smem 地址高位
    __shared__ __align__(8) uint64_t mbar;
    unsigned char* dst = smem + smem_off;
    const int tid = threadIdx.x;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&mbar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int box_bytes = box_inner * box_outer * elem_bytes;
    if (tid == 0) {
        const uint32_t sp = (uint32_t)__cvta_generic_to_shared(dst);
        const uint32_t mp = (uint32_t)__cvta_generic_to_shared(&mbar);
        const uint64_t dp = reinterpret_cast<uint64_t>(&tmap);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mp), "r"(box_bytes));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4}], [%2];" ::
            "r"(sp), "l"(dp), "r"(mp), "r"(c0), "r"(c1) : "memory");
    }
    __syncthreads();
    if (tid == 0) {
        const uint32_t mp = (uint32_t)__cvta_generic_to_shared(&mbar);
        const uint32_t phase = 0;
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
            "@!p bra WAIT;\n\t}" :: "r"(mp), "r"(phase));
    }
    __syncthreads();
    for (int i = tid; i < box_bytes; i += blockDim.x) out[i] = dst[i];
}

// 计算物理 smem 字节偏移：TMA swizzle 用 cute::Swizzle<B,4,3>（B=1/2/3 对应 32/64/128B）。
// 注意：hardware 的 XOR 键取自**绝对** smem 地址的 [7:10) 位，故 smem 基址的 (base>>7)&7
// 会整体平移 swizzle —— 用 base_row_bits 参数显式建模。
static int swizzle_offset(const int swz_bytes, const int linear_byte_offset, const int base_row_bits) {
    if (swz_bytes == 0) return linear_byte_offset;
    const int bits = swz_bytes == 128 ? 3 : (swz_bytes == 64 ? 2 : 1);
    const int mask = (1 << bits) - 1;
    const int key = ((linear_byte_offset >> 7) + base_row_bits) & mask;   // XOR 键 = 绝对地址 bits[7:10)
    return linear_byte_offset ^ (key << 4);
}

// 运行一次 TMA 载入，返回 mismatch（对比 base_row_bits 预测）
static int run_tma_load(const torch::Tensor& t, const int inner, const int outer,
                        const int swz, const int box_outer, const int smem_off,
                        const int base_row_bits) {
    const int elem = static_cast<int>(t.element_size());
    const int box_inner = swz == 0 ? 64 : swz / elem;
    auto desc = make_tma_2d_desc(t, inner, outer, box_inner, box_outer, inner, swz);
    auto out = torch::zeros({box_inner * box_outer * elem},
                torch::dtype(torch::kUInt8).device(torch::kCUDA));
    tma_load_kernel<<<1, 128, 0>>>(
        desc, box_inner, box_outer, elem, 0, 0, smem_off, out.data_ptr<unsigned char>());
    check(cudaDeviceSynchronize() == cudaSuccess, "tma_load_run_sw" + std::to_string(swz));
    std::vector<unsigned char> exp(box_inner * box_outer * elem, 0);
    for (int r = 0; r < box_outer; r++)
        for (int c = 0; c < box_inner; c++) {
            const int lin = r * (box_inner * elem) + c * elem;
            const int phys = swizzle_offset(swz, lin, base_row_bits);
            const uint16_t v = static_cast<uint16_t>(r * inner + c);
            exp[phys] = v & 0xff; exp[phys + 1] = v >> 8;
        }
    auto got = out.to(torch::kCPU);
    const auto* g = got.data_ptr<unsigned char>();
    int mism = 0;
    for (int i = 0; i < box_inner * box_outer * elem; i++) if (g[i] != exp[i]) mism++;
    return mism;
}

static void part2_real_load() {
    printf("== Part 2: 真实 TMA 载入 vs cute::Swizzle<B,4,3> 预测布局 ==\n");
    const int inner = 128, outer = 64;         // bf16 tensor [outer, inner]
    const int elem = 2;
    auto t = torch::arange(inner * outer, torch::dtype(torch::kInt16))
                 .reshape({outer, inner}).to(torch::kCUDA)
                 .view(torch::kBFloat16);   // 位模式=row*inner+col（dtype 需为 bf16 才被接受）
    for (int swz : {0, 32, 64, 128}) {
        const int mism = run_tma_load(t, inner, outer, swz, 8, 0, 0);
        check(mism == 0, "tma_load_layout_sw" + std::to_string(swz));
        printf("  swizzle=%-3d box_outer=8 smem_off=0    mismatch=%d/%d\n", swz, mism,
               swz == 0 ? 1024 : swz / elem * 8 * elem);
    }
    // 关键：把 smem 目标从原子对齐的 +0 挪到只满足「128B 对齐」的 +128，
    // swizzle 的 XOR 键随绝对地址整体平移 1 个原子行。
    const int m0 = run_tma_load(t, inner, outer, 128, 8, 128, 0);   // 用 base=0 预测
    const int m1 = run_tma_load(t, inner, outer, 128, 8, 128, 1);   // 用 base=1 预测
    check(m0 != 0 and m1 == 0, "swizzle_base_address_shift");
    printf("  swizzle=128 smem_off=+128(仅128B对齐)  base预测0 mismatch=%d/1024, base预测1 mismatch=%d/1024\n",
           m0, m1);
}

// ---------------------------------------------------------------------------
// Part 3：描述符字节指纹
// ---------------------------------------------------------------------------
static void part3_fingerprint() {
    printf("== Part 3: 128 字节描述符指纹（变一个输入 → XOR 掩码）==\n");
    auto bf16 = torch::ones({64, 128}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    auto fp32 = torch::ones({64, 128}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto e4m3 = torch::ones({64, 128}, torch::dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
    auto base = make_tma_2d_desc(bf16, 128, 64, 0, 8, 128, 128);
    const auto* pb = reinterpret_cast<const unsigned char*>(&base);
    printf("  base_dump ");
    for (int i = 0; i < 128; i++) printf("%02x", pb[i]);
    printf("\n");
    auto report = [&](const char* name, const CUtensorMap& v) {
        const auto* pv = reinterpret_cast<const unsigned char*>(&v);
        printf("  diff %-16s bytes:", name);
        for (int i = 0; i < 128; i++) if (pv[i] != pb[i]) printf(" %d", i);
        printf("\n");
    };
    report("swz128->64", make_tma_2d_desc(bf16, 128, 64, 0, 8, 128, 64));
    report("swz128->0",  make_tma_2d_desc(bf16, 128, 64, 64, 8, 128, 0));
    report("dtype->fp32", make_tma_2d_desc(fp32, 128, 64, 0, 8, 128, 128));
    report("dtype->e4m3", make_tma_2d_desc(e4m3, 128, 64, 0, 8, 128, 128));
    report("gi128->256",  make_tma_2d_desc(bf16, 256, 64, 0, 8, 128, 128));
    report("go64->128",   make_tma_2d_desc(bf16, 128, 128, 0, 8, 128, 128));
    report("so8->16",     make_tma_2d_desc(bf16, 128, 64, 0, 16, 128, 128));
    report("stride128->256", make_tma_2d_desc(bf16, 128, 64, 0, 8, 256, 128));
    // 地址变化：同一形状、同 stride，仅基址不同（隔离出地址字段）
    auto big = torch::ones({128, 128}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    auto view = big.narrow(0, 64, 64);
    report("addr(ptr)", make_tma_2d_desc(view, 128, 64, 0, 8, 128, 128));
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("nfp4=%d kPackedFP4_is_int8=%d\n", (int)deep_gemm::kPackedFP4, (int)torch::kInt8);
    printf("swizzle_enum: 0->%d 16->%d 32->%d 64->%d 128->%d base32->%d\n",
           (int)deep_gemm::mode_into_tensor_map_swizzle(0, 0),
           (int)deep_gemm::mode_into_tensor_map_swizzle(16, 0),
           (int)deep_gemm::mode_into_tensor_map_swizzle(32, 0),
           (int)deep_gemm::mode_into_tensor_map_swizzle(64, 0),
           (int)deep_gemm::mode_into_tensor_map_swizzle(128, 0),
           (int)deep_gemm::mode_into_tensor_map_swizzle(128, 32));
    part1_equivalence();
    printf("\n");
    part2_real_load();
    printf("\n");
    part3_fingerprint();
    printf("\n总判定: %s (pass=%d fail=%d)\n", g_fail == 0 ? "ALL PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
