// P107-1 实验 Part A：纯 host 探针，验证 DeepEP 的 `get_env<T>`（csrc/utils/system.hpp:12-29）
//   ① 真实执行 `deep_ep::get_env<int/string>`：未设->默认、int 前缀解析、string 直取；
//   ② 「不支持类型」走 EP_HOST_ASSERT 抛 EPException；
//   ③ **解析陷阱**：畸形/空值的 `sscanf("%d")` 返回值被忽略 -> 返回未初始化的本地 `value`（UB）。
//      真实 `get_env<int>` 之外，另用一个逐行复刻 + 哨兵初始值的函数把语义固定下来。
//   ④ 跨语言对照：C++ `get_env<int>("EP_BUFFER_DEBUG")` 把 "0" 解析成 0（关），
//      而 Python `os.environ.get('EP_BUFFER_DEBUG', 0)` 对非空字符串 "0" 恒为真（开）。
// 架构无关，g++ host 编译即可；只依赖 deep_ep/include 下的 exception.cuh。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <exception>

#include "csrc/utils/system.hpp"   // deep_ep::get_env 真实实现

using namespace deep_ep;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                               \
    do {                                                               \
        if (cond) { ++g_pass; std::printf("  [PASS] %s\n", msg); }    \
        else      { ++g_fail; std::printf("  [FAIL] %s\n", msg); }    \
    } while (0)

// 逐行复刻 system.hpp:12-29 的真实语义，但把本地 value 预置哨兵，
// 以便观察「sscanf 未写入时返回原样未初始化值」这一行为（避开读 UB 的随机性）。
static int get_env_int_with_sentinel(const std::string& name, int sentinel) {
    const auto c_str = std::getenv(name.c_str());
    if (c_str == nullptr) return -1;
    if constexpr (true) {              // 只走 int 分支，镜像真实代码
        int value = sentinel;
        std::sscanf(c_str, "%d", &value);
        return value;
    }
}

int main() {
    std::printf("== Part A: get_env<T> 语义（真实 csrc/utils/system.hpp）==\n");

    // ---- ① 正常语义 ----
    ::setenv("EP_PROBE_UNSET", "", 1); ::unsetenv("EP_PROBE_UNSET");
    ::setenv("EP_PROBE_INT", "42", 1);
    ::setenv("EP_PROBE_INT_PREFIX", "123abc", 1);
    ::setenv("EP_PROBE_STR", "hello world", 1);
    CHECK(get_env<int>("EP_PROBE_UNSET", 7) == 7, "unset -> default 7");
    CHECK(get_env<int>("EP_PROBE_INT", 0) == 42, "int \"42\" -> 42");
    CHECK(get_env<int>("EP_PROBE_INT_PREFIX", 0) == 123, "int \"123abc\" -> 123（sscanf 取前缀）");
    CHECK(get_env<std::string>("EP_PROBE_STR", "x") == "hello world", "string -> \"hello world\"");

    // ---- ② 不支持类型：EP_HOST_ASSERT(false) ----
    bool threw = false;
    try {
        (void)get_env<bool>("EP_PROBE_INT", false);
    } catch (const EPException& e) {
        threw = true;
        std::printf("  [INFO] get_env<bool> 抛 EPException: %s\n", e.what());
    }
    CHECK(threw, "get_env<bool> -> EPException（system.hpp:27 不支持类型）");

    // ---- ③ 解析陷阱：sscanf 返回值被忽略 ----
    std::printf("  -- 畸形/空值解析（sscanf 返回值被忽略）--\n");
    ::setenv("EP_PROBE_BAD", "abc", 1);
    ::setenv("EP_PROBE_EMPTY", "", 1);
    int r_bad = get_env<int>("EP_PROBE_BAD", -999);
    int r_empty = get_env<int>("EP_PROBE_EMPTY", -999);
    std::printf("  [INFO] get_env<int>(\"abc\")   = %d (默认值 -999 未生效，未设才算未设)\n", r_bad);
    std::printf("  [INFO] get_env<int>(\"\")      = %d\n", r_empty);
    // sscanf 对 "abc"/"" 返回 0 且不写 value —— 真实返回值来自未初始化栈值（UB）。
    // 用哨兵复刻证明「value 原样返回、不是默认值也不是 0」：
    for (int sent : {0x11111111, 0x22222222}) {
        int got = get_env_int_with_sentinel("EP_PROBE_BAD", sent);
        std::printf("  [INFO] 哨兵复刻 sentinel=0x%08x -> 0x%08x（未变）\n", sent,
                    static_cast<unsigned>(got));
        CHECK(got == sent, "畸形值下 sscanf 不写 value -> 返回哨兵（坐实未初始化）");
    }
    int c_unset;
    { const auto c = std::getenv("EP_PROBE_NEVER"); c_unset = (c == nullptr); }
    CHECK(c_unset, "未设时 getenv 返回 nullptr -> 走 default_value（安全路径）");

    // ---- ④ 跨语言对照：C++ int 解析 vs Python 真值 ----
    ::setenv("EP_BUFFER_DEBUG", "0", 1);
    int cpp = get_env<int>("EP_BUFFER_DEBUG", 0);
    CHECK(cpp == 0, "C++: EP_BUFFER_DEBUG=\"0\" -> get_env<int>=0（关）");
    std::printf("  [INFO] C++ 侧 = %d（关）；Python 侧 os.environ.get('EP_BUFFER_DEBUG',0) 对 \"0\" 为真（开）\n", cpp);

    std::printf("\nPart A 汇总: %d PASS / %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
