// P160-1 实验（纯 host，架构无关）：直接 include 真实 DeepJIT 支撑工具层头文件并真机跑。
//   filesystem.hpp / command.hpp / str.hpp / uuid.hpp / json.hpp / no_ref_ptr.hpp
// 这些头在哪儿被用：见 harness/exp_deepjit_utils.sh 打印的 grep 命中。
// 编译：g++ -std=c++20 -O2 -g -I<deep_jit/include> src/deepjit_utils.cpp -o /tmp/deepjit_utils -ldw -ldl
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/json.hpp>
#include <deep_jit/utils/no_ref_ptr.hpp>
#include <deep_jit/utils/str.hpp>
#include <deep_jit/utils/uuid.hpp>

namespace fs = std::filesystem;
using deep_jit::json;

static int g_pass = 0, g_fail = 0;
static void ok(const bool cond, const std::string& name, const std::string& detail = "") {
    if (cond) { ++g_pass; std::printf("[A] PASS %s%s%s\n", name.c_str(), detail.empty() ? "" : " : ", detail.c_str()); }
    else      { ++g_fail; std::printf("[A] FAIL %s%s%s\n", name.c_str(), detail.empty() ? "" : " : ", detail.c_str()); }
}
static bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}
// 逐字复刻 backend/cuda/kernel.hpp:24-30 的 kernel_arg_pointer
template <typename T>
static void* kernel_arg_pointer(const T& value) {
    if constexpr (std::is_base_of_v<deep_jit::NoRefPtr, std::decay_t<T>>)
        return value.ptr;
    else
        return const_cast<void*>(static_cast<const void*>(&value));
}

// ---------------------------------------------------------------- json
static void test_json() {
    const json basic = json::object_t{
        {"a", 1}, {"b", "x"}, {"c", json::array_t{1, 2, 3}}, {"d", nullptr}, {"e", true}};
    const std::string dumped = basic.dump();
    std::printf("JSON_BASIC %s\n", dumped.c_str());
    ok(dumped == R"({"a":1,"b":"x","c":[1,2,3],"d":null,"e":true})", "json.basic_dump", dumped);

    // 有符号 / 无符号分开：int64_t 与 uint64_t 各一成员
    std::printf("JSON_INT %s\n", json(int64_t(-5)).dump().c_str());
    std::printf("JSON_UINT %s\n", json(uint64_t(18446744073709551615ULL)).dump().c_str());
    ok(json(int64_t(-5)).dump() == "-5", "json.int64", json(int64_t(-5)).dump());
    ok(json(uint64_t(18446744073709551615ULL)).dump() == "18446744073709551615",
       "json.uint64", json(uint64_t(18446744073709551615ULL)).dump());
    ok(json(true).dump() == "true", "json.bool_not_int");
    ok(json(3).dump() == "3", "json.int_literal");

    // 字符串转义：引号 / 反斜杠 / \b\f\n\r\t / 控制字符 \x01
    std::string evil = "\"\\\b\f\n\r\t\x01";
    const std::string edump = json(evil).dump();
    std::printf("JSON_ESC %s\n", edump.c_str());
    ok(edump == "\"\\\"\\\\\\b\\f\\n\\r\\t\\u0001\"", "json.escape", edump);

    // optional / vector 便捷构造
    ok(json(std::optional<int>(3)).dump() == "3", "json.optional_value");
    ok(json(std::optional<int>{}).dump() == "null", "json.optional_empty");
    ok(json(std::vector<int>{1, 2}).dump() == "[1,2]", "json.vector_ctor");

    // object_t 是 vector<pair>：保序 + 允许重复键（与 std::map 语义不同）
    const std::string dup = json(json::object_t{{"k", 1}, {"k", 2}}).dump();
    std::printf("JSON_DUP %s\n", dup.c_str());
    ok(dup == R"({"k":1,"k":2})", "json.object_is_ordered_vector", dup);

    // 嵌套
    const json nested = json::object_t{{"x", json::array_t{json::object_t{{"y", nullptr}}}}};
    std::printf("JSON_NEST %s\n", nested.dump().c_str());
    ok(nested.dump() == R"({"x":[{"y":null}]})", "json.nested", nested.dump());
}

// ---------------------------------------------------------------- str::join
static void test_str() {
    using deep_jit::str::join;
    ok(join({}) == "", "str.join.empty", join({}));
    ok(join({"a"}) == "a", "str.join.single", join({"a"}));
    ok(join({"a", "b", "c"}) == "a b c", "str.join.default_sep", join({"a", "b", "c"}));
    ok(join({"a", "b", "c"}, ", ") == "a, b, c", "str.join.custom_sep", join({"a", "b", "c"}, ", "));
    ok(join({"a", "b"}, "") == "ab", "str.join.empty_sep", join({"a", "b"}, ""));
    std::printf("STR_JOIN %s\n", join({"-O3", "--use_fast_math", "-std=c++17"}).c_str());
}

// ---------------------------------------------------------------- uuid
static void test_uuid() {
    const std::string u = deep_jit::get_uuid();
    std::printf("UUID %s\n", u.c_str());
    // 前缀必须是本进程 pid
    const std::string pid_prefix = std::to_string(::getpid()) + "-";
    ok(u.rfind(pid_prefix, 0) == 0, "uuid.pid_prefix", u);
    ok(u.size() == pid_prefix.size() + 3 * 8 + 2, "uuid.length", u);

    // 同线程 10 万次唯一
    std::set<std::string> seen;
    for (int i = 0; i < 100000; ++i) seen.insert(deep_jit::get_uuid());
    ok(seen.size() == 100000, "uuid.unique_100k", std::to_string(seen.size()));

    // 多线程：generator 是 thread_local（uuid.hpp:15-20），跨线程也不撞
    std::mutex m;
    std::vector<std::string> all;
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&]() {
            std::vector<std::string> local;
            local.reserve(20000);
            for (int i = 0; i < 20000; ++i) local.push_back(deep_jit::get_uuid());
            std::lock_guard<std::mutex> lock(m);
            for (auto& s : local) all.push_back(std::move(s));
        });
    }
    for (auto& th : threads) th.join();
    std::set<std::string> uniq(all.begin(), all.end());
    ok(uniq.size() == all.size(), "uuid.thread_local_no_collision",
       std::to_string(uniq.size()) + "/" + std::to_string(all.size()));
}

// ---------------------------------------------------------------- command
static void test_command() {
    using deep_jit::call_external_command;
    const std::string out = call_external_command("echo hi");
    std::printf("CMD_SUCCESS %s", out.c_str());
    ok(out == "hi\n", "command.success", out);

    // 2>&1：stderr 被合并进 stdout（command.hpp:18）。
    // 注意追加的 " 2>&1" 在内层重定向之后，故必须用 sh -c 把 >&2 关在子 shell 里才测得到合并。
    const std::string err = call_external_command("sh -c \"echo to_stderr >&2\"");
    std::printf("CMD_STDERR %s", err.c_str());
    ok(contains(err, "to_stderr"), "command.stderr_merged", err);

    // 非零退出码 → 抛异常，消息含退出码
    std::string msg;
    try { call_external_command("exit 3"); } catch (const std::exception& e) { msg = e.what(); }
    ok(contains(msg, "exit code 3"), "command.nonzero_throws", msg.substr(0, 60));

    // 被信号杀死：WIFEXITED 为假 → 128+SIGTERM(15)=143（command.hpp:31-35）
    std::string smsg;
    try { call_external_command("kill -TERM $$"); } catch (const std::exception& e) { smsg = e.what(); }
    ok(contains(smsg, "143"), "command.signal_death_143", smsg.substr(0, 60));

    // 空命令 → 断言
    std::string emsg;
    try { call_external_command(""); } catch (const std::exception& e) { emsg = e.what(); }
    ok(contains(emsg, "must not be empty"), "command.empty_asserts", emsg.substr(0, 60));

    // which nvcc（backend.hpp:187 也这么用）——环境里有 nvcc
    try {
        const std::string which = call_external_command("which nvcc");
        std::printf("CMD_WHICH_NVCC %s", which.c_str());
        ok(!which.empty() && which.back() == '\n', "command.which_nvcc", which);
    } catch (const std::exception& e) {
        ok(false, "command.which_nvcc", e.what());
    }
}

// ---------------------------------------------------------------- filesystem
static void test_fs() {
    using namespace deep_jit;
    const fs::path root = "/tmp/dj_utils_test";
    safe_remove_all(root);
    fs::create_directories(root);

    // write_file_sync + read 往返。
    // 坑：直接传字符串字面量会退化成 string_view(const char*)，遇 NUL 截断（只写 6 字节）；
    // 这里必须显式给长度。
    const std::string payload("hello \x00 binary", 14);
    const auto f = root / "sub" / "data.bin";
    make_dirs(f.parent_path());                       // filesystem.hpp:53-59
    write_file_sync(f, payload);                      // filesystem.hpp:73-83
    const std::string back = read(f);                 // filesystem.hpp:18-22
    std::printf("FS_ROUNDTRIP len=%zu\n", back.size());
    ok(back == payload, "fs.write_read_roundtrip", std::to_string(back.size()));
    ok(fs::exists(root / "sub"), "fs.make_dirs_nested");

    // 读不存在 → 断言
    std::string rmsg;
    try { read(root / "nope"); } catch (const std::exception& e) { rmsg = e.what(); }
    ok(contains(rmsg, "failed to open for reading"), "fs.read_missing_throws", rmsg.substr(0, 60));

    // try_update_mtime：存在→true，不存在→false，且 noexcept（filesystem.hpp:65-69）
    static_assert(noexcept(try_update_mtime(f)), "try_update_mtime must be noexcept");
    ok(try_update_mtime(f) == true, "fs.try_update_mtime_existing");
    ok(try_update_mtime(root / "nope") == false, "fs.try_update_mtime_missing_false");

    // fsync_dir 递归（filesystem.hpp:43-51）
    fs::create_directories(root / "a" / "b");
    write_file_sync(root / "a" / "b" / "x", "1");
    try { fsync_dir(root); ok(true, "fs.fsync_dir_recursive"); }
    catch (const std::exception& e) { ok(false, "fs.fsync_dir_recursive", e.what()); }

    // safe_remove_all 递归删；不存在时静默（filesystem.hpp:88-108）
    safe_remove_all(root / "sub");
    ok(!fs::exists(root / "sub"), "fs.safe_remove_all_nested");
    try { safe_remove_all(root / "does_not_exist"); ok(true, "fs.safe_remove_all_missing_silent"); }
    catch (const std::exception& e) { ok(false, "fs.safe_remove_all_missing_silent", e.what()); }

    // normalize_path：相对→绝对且 lexically_normal；nullopt/空串→nullopt（filesystem.hpp:110-114）
    const auto norm = normalize_path(fs::path("a/b/../c"));
    std::printf("FS_NORMALIZE %s\n", norm ? norm->string().c_str() : "(nullopt)");
    ok(norm.has_value() && norm->is_absolute() && norm->filename() == "c", "fs.normalize_relative");
    ok(!normalize_path(std::nullopt).has_value(), "fs.normalize_nullopt");
    ok(!normalize_path(fs::path("")).has_value(), "fs.normalize_empty");

    // is_executable（filesystem.hpp:116-118）
    ok(is_executable("/bin/true") == true, "fs.is_executable_true");
    const auto plain = root / "plain.txt";
    write_file_sync(plain, "x");
    fs::permissions(plain, fs::perms::owner_read | fs::perms::owner_write);
    ok(is_executable(plain) == false, "fs.is_executable_non_exec_file");
    ok(is_executable(root / "missing") == false, "fs.is_executable_missing");

    safe_remove_all(root);
}

// ---------------------------------------------------------------- no_ref_ptr
struct FakeArgs : deep_jit::NoRefPtr { int filler = 7; };
static void test_no_ref_ptr() {
    int x = 42;
    deep_jit::NoRefPtr marker{&x};
    ok(kernel_arg_pointer(marker) == static_cast<void*>(&x), "no_ref_ptr.pointer_is_storage");
    ok(kernel_arg_pointer(x) == static_cast<void*>(&x), "no_ref_ptr.plain_takes_address");
    ok(sizeof(deep_jit::NoRefPtr) == sizeof(void*), "no_ref_ptr.sizeof_is_pointer");
    static_assert(std::is_base_of_v<deep_jit::NoRefPtr, FakeArgs>, "derivable");
}

int main() {
    std::printf("== DeepJIT utils 工具层：纯 host 真机 ==");
    std::printf("\n--- json ---\n");   test_json();
    std::printf("--- str ---\n");      test_str();
    std::printf("--- uuid ---\n");     test_uuid();
    std::printf("--- command ---\n");  test_command();
    std::printf("--- filesystem ---\n"); test_fs();
    std::printf("--- no_ref_ptr ---\n"); test_no_ref_ptr();
    std::printf("CPP_SUMMARY pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
