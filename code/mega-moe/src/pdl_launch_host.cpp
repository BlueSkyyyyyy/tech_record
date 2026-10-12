// P42-1: DeepJIT LaunchOptions 的 host 语义（直接 include 真实头文件）。
//   对应 third-party/deep_jit/include/deep_jit/backend/cuda/options.hpp:178-233
//   与 runtime/runtime.hpp:31/42/71-73。
// 纯 host 逻辑，架构无关；用 g++ 编译运行。
#include <cstdio>
#include <string>
#include <deep_jit/backend/cuda/options.hpp>

using namespace deep_jit;
using namespace deep_jit::cuda;

static int g_fail = 0;
#define CHECK(cond, ...) do { bool _c = (cond); if (!_c) { ++g_fail; \
    std::printf("  FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static void dump(const char* tag, const LaunchOptions& o) {
    std::printf("%s smem=%s grid=%s block=%s cluster=(%u,1,1) coop=%s pdl=%s nonportable=%s stream=%s\n",
        tag,
        o.num_smem_bytes ? std::to_string(*o.num_smem_bytes).c_str() : "null",
        o.grid_dim ? std::to_string(o.grid_dim->x).c_str() : "null",
        o.block_dim ? std::to_string(o.block_dim->x).c_str() : "null",
        o.cluster_dim->x,
        o.cooperative ? (*o.cooperative ? "true" : "false") : "null",
        o.enable_pdl ? (*o.enable_pdl ? "true" : "false") : "null",
        o.nonportable_cluster_size_allowed ? (*o.nonportable_cluster_size_allowed ? "true" : "false") : "null",
        o.stream ? "set" : "null");
}

int main() {
    Env env("DG");

    // 1) 默认值
    auto d = LaunchOptions::default_options(env);
    dump("DEFAULT", d);
    CHECK(!d.stream.has_value(), "default stream should be nullopt");
    CHECK(d.num_smem_bytes && *d.num_smem_bytes == 0, "default smem=0");
    CHECK(!d.grid_dim.has_value() && !d.block_dim.has_value(), "default grid/block nullopt");
    CHECK(d.cluster_dim && d.cluster_dim->x == 1, "default cluster=1");
    CHECK(d.cooperative && !*d.cooperative, "default coop=false");
    CHECK(d.enable_pdl && !*d.enable_pdl, "default pdl=false");
    CHECK(d.nonportable_cluster_size_allowed && !*d.nonportable_cluster_size_allowed, "default nonportable=false");

    // 2) DeepGEMM 每 kernel 只填 4 个字段（见 csrc/jit_kernels/impls/*.hpp）
    LaunchOptions per_kernel{
        .num_smem_bytes = 232448,
        .grid_dim = dim3(132, 1, 1),
        .block_dim = dim3(512, 1, 1),
        .cluster_dim = dim3(2, 1, 1),
    };
    auto merged = d.override_with(per_kernel);
    dump("MERGED", merged);
    CHECK(*merged.num_smem_bytes == 232448 && merged.grid_dim->x == 132 &&
          merged.block_dim->x == 512 && merged.cluster_dim->x == 2, "per-kernel fields applied");
    CHECK(merged.enable_pdl && !*merged.enable_pdl, "pdl inherited false");
    CHECK(merged.cooperative && !*merged.cooperative, "coop inherited false");
    CHECK(merged.nonportable_cluster_size_allowed && !*merged.nonportable_cluster_size_allowed,
          "nonportable inherited false");

    // 3) set_pdl(true) → 运行时默认置位 → 不吃 override 的 enable_pdl 也变 true
    //    对应 csrc/apis/config.hpp:27-31 与 runtime.hpp:71-73
    LaunchOptions runtime_default = d;
    runtime_default.enable_pdl = true;
    auto merged_pdl = runtime_default.override_with(per_kernel);
    dump("SET_PDL", merged_pdl);
    CHECK(*merged_pdl.enable_pdl, "pdl=true propagated to per-kernel launch");

    // 4) 显式 false/0 是「有值」，会覆盖默认；default 里被置 true 的项也能被 per-kernel 关掉
    LaunchOptions base = d;
    base.cooperative = true;
    base.nonportable_cluster_size_allowed = true;
    LaunchOptions off{
        .num_smem_bytes = 0,                                   // 显式 0
        .cooperative = false,                                  // 显式 false
        .nonportable_cluster_size_allowed = false,             // 显式 false
    };
    auto m2 = base.override_with(off);
    dump("EXPLICIT_OFF", m2);
    CHECK(*m2.num_smem_bytes == 0 && !*m2.cooperative && !*m2.nonportable_cluster_size_allowed,
          "engaged false/0 overrides base true");
    CHECK(m2.grid_dim->x == 1 || !m2.grid_dim.has_value(), "unset keeps base nullopt");

    // 5) override 的链式顺序：后者覆盖前者
    LaunchOptions a{ .grid_dim = dim3(10, 1, 1) };
    LaunchOptions b{ .grid_dim = dim3(20, 1, 1) };
    auto chained = d.override_with(a).override_with(b);
    CHECK(chained.grid_dim->x == 20, "last override wins");

    // 6) cluster_dim 默认努力给 (1,1,1)，per-kernel 覆盖为 2
    CHECK(base.override_with(per_kernel).cluster_dim->x == 2, "cluster override");

    std::printf("HOST_JUDGE %s (fail=%d)\n", g_fail == 0 ? "ALL PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
