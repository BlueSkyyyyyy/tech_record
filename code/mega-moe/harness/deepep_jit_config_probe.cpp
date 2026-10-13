// P88-1 实验 Part 2：纯 host 探针
//   ① 真实执行 DeepEP 的 csrc/utils/system.hpp::get_env（含「不支持类型」断言路径）
//   ② 真实构造 DeepEP 在 csrc/runtime/jit.hpp:22-27 用的 deep_jit::Config（含三条负样本）
//   ③ 按 jit.hpp:34-44 复刻 8 条 nvcc flag，并用 deep_jit 真实的 FNV1a/str::join 演示
//      「NCCL include 路径进缓存键、DeepEP 自身 include 路径不进 flag」。
// 架构无关，g++/nvcc host 编译即可。
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <exception>

#include <deep_jit/runtime/config.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/str.hpp>

#include "csrc/utils/system.hpp"   // deep_ep::get_env 真实实现

using namespace deep_ep;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                               \
    do {                                                               \
        if (cond) { ++g_pass; std::printf("  [PASS] %s\n", msg); }    \
        else      { ++g_fail; std::printf("  [FAIL] %s\n", msg); }    \
    } while (0)

int main() {
    ::setenv("EP_PROBE_INT", "42", 1);
    ::setenv("EP_PROBE_INT_PREFIX", "123abc", 1);
    ::setenv("EP_PROBE_STR", "hello world", 1);
    ::unsetenv("EP_PROBE_UNSET");

    std::printf("== Part 2: get_env / Config / 缓存键 ==\n");

    // ① get_env
    CHECK(get_env<int>("EP_PROBE_UNSET", 7) == 7, "get_env unset -> default 7");
    CHECK(get_env<int>("EP_PROBE_INT", 0) == 42, "get_env int -> 42");
    CHECK(get_env<int>("EP_PROBE_INT_PREFIX", 0) == 123, "get_env int sscanf 取前缀 -> 123");
    CHECK(get_env<std::string>("EP_PROBE_STR", "x") == "hello world", "get_env string -> \"hello world\"");

    bool thrown = false;
    try { (void)get_env<bool>("EP_PROBE_INT", false); }
    catch (const EPException& e) { thrown = true; std::printf("  get_env<bool> threw: %s\n", e.what()); }
    CHECK(thrown, "get_env<bool> -> EP_HOST_ASSERT(false) throws");

    // ② Config（与 jit.hpp:22-27 同参）
    const std::string root = "/tmp/deepep_pkg";
    auto make = [&](const std::string& r, const std::string& prefix) {
        return deep_jit::Config(r, prefix, {}, {std::string(r) + "/include"}, {"deep_ep/"});
    };
    const auto cfg = make(root, "EP");
    const auto json = cfg.to_json().dump();
    std::printf("  config.to_json = %s\n", json.c_str());
    CHECK(json.find("\"extra_signature\":\"\"") != std::string::npos,
          "extra_signature is empty (DeepGEMM 则是 \"cutlass-<ver>\")");
    CHECK(json.find("\"include_prefixes\":[\"deep_ep/\"]") != std::string::npos,
          "include_prefixes == {\"deep_ep/\"}");
    CHECK(cfg.get_python_path("mega/mega.cuh") == root + "/mega/mega.cuh",
          "get_python_path 相对拼接到 library_root");

    auto throws = [&](const std::string& r, const std::string& p) {
        try { (void)make(r, p); } catch (const std::exception&) { return true; }
        return false;
    };
    CHECK(throws("relative/path", "EP"), "relative python_library_root -> DJ_HOST_ASSERT");
    CHECK(throws(root, ""),            "empty env_prefix -> DJ_HOST_ASSERT");
    CHECK(throws(root, "DJ"),          "\"DJ\" reserved -> DJ_HOST_ASSERT");

    // ③ flag 复刻 + 缓存键演示
    auto build_flags = [&](const std::string& nccl_include, int topk_bits, bool debug) {
        std::vector<std::string> f = {
            "--diag-suppress=39,161,174,177,186,940,3012",
            "--compiler-options=-Wno-deprecated-declarations,-Wno-abi",
            "-I" + nccl_include,
            "-DEP_NUM_TOPK_IDX_BITS=" + std::to_string(topk_bits),
            "-DNCCL_GIN_GDAKI_ENABLE=1",
            "-DNCCL_GIN_PROXY_ENABLE=0",
            "-DNCCL_GIN_GPI_ENABLE=0",
            "-DNCCL_GIN_EFA_GDA_ENABLE=0",
        };
        if (debug) f.emplace_back("-DNCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG=1");
        return f;
    };

    const auto fA = build_flags("/opt/nccl-A/include", 64, false);
    const auto fB = build_flags("/opt/nccl-B/include", 64, false);
    const auto fD = build_flags("/opt/nccl-A/include", 64, true);
    const auto joinA = deep_jit::str::join(fA);
    std::printf("  flags(base, nccl-A) = %s\n", joinA.c_str());
    std::printf("  flags(base, nccl-B) = %s\n", deep_jit::str::join(fB).c_str());
    std::printf("  flags(debug, nccl-A) = %s\n", deep_jit::str::join(fD).c_str());
    CHECK(fA.size() == 8, "8 条 unconditional flag");
    CHECK(fD.size() == 9, "EP_GIN_GDAKI_DEBUG 打开后 9 条");

    const auto hA = deep_jit::hash::get_hex_digest(joinA);
    const auto hB = deep_jit::hash::get_hex_digest(deep_jit::str::join(fB));
    const auto hD = deep_jit::hash::get_hex_digest(deep_jit::str::join(fD));
    std::printf("  hash(nccl-A) = %s\n", hA.c_str());
    std::printf("  hash(nccl-B) = %s\n", hB.c_str());
    std::printf("  hash(debug)  = %s\n", hD.c_str());
    CHECK(hA != hB, "换 NCCL include 路径 -> flags 哈希变化（进缓存键）");
    CHECK(hA != hD, "开 debug flag -> flags 哈希变化（进缓存键）");
    // DeepEP 自身 include 路径不在 flag 里（由 parser 按 deep_ep/ 前缀跟踪）
    CHECK(joinA.find(root + "/include") == std::string::npos,
          "DeepEP 自身 include 路径不出现在 flags（保持路径无关）");

    std::printf("Part 2 RESULT: %d PASS / %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
