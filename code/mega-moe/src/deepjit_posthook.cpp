// P37-1 实验源码：直接嵌入 DeepJIT 的 CUDA 后端（header-only），在 H100(sm90) 上真机验证
//   「post_hook 后处理」这一 DeepGEMM 默认不用、但 DeepJIT 提供的可插拔编译后处理机制，
//   以及它如何进入缓存键、如何影响磁盘产物（meta.json / cubin）。
// 参考：~/github/DeepGEMM/third-party/deep_jit/include/deep_jit/{backend/cuda, runtime, cache}/
//
// 编译（见 harness/exp_deepjit_posthook.sh）：g++ -std=c++20 链接 torch（kernel.hpp 需要
//   ATen/cuda/CUDAContext.h 取当前流）与 libpython（gil.hpp 需要 Py_IsInitialized）。
//
// 输出行以固定前缀给出，由 harness 里的 Python 解析并判定。
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <deep_jit/backend/cuda/backend.hpp>

namespace fs = std::filesystem;
using deep_jit::CUDA;

static const char* kKernel = R"(
extern "C" __global__ void noop() {}
)";

// 各 hook 脚本都是「改 cubin 就地」的 Python 文件；这里统一定义内容。
static const char* kHookMark =
    "import sys, os\n"
    "path = os.path.dirname(sys.argv[1]) + '/hook_mark.txt'\n"
    "open(path, 'a').write('ran\\n')\n";           // 仅留副作用：证明 hook 跑过，不破坏 cubin

static const char* kHookCorrupt =
    "import sys\n"
    "open(sys.argv[1], 'wb').write(b'HOOK')\n";     // 就地破坏 cubin，用于验证「在发布前生效」

static const char* kHookFail =
    "import sys\n"
    "sys.exit(7)\n";                                 // 非零退出 → 编译失败、不得发布

static void write_file(const fs::path& p, const std::string& s) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

static std::string read_head(const fs::path& p, const size_t n) {
    std::ifstream in(p, std::ios::binary);
    std::string s(n, '\0');
    in.read(s.data(), static_cast<std::streamsize>(n));
    s.resize(static_cast<size_t>(in.gcount()));
    return s;
}

static int count_lines(const fs::path& p) {
    std::ifstream in(p);
    if (not in.is_open())
        return -1;
    int n = 0;
    std::string line;
    while (std::getline(in, line))
        ++n;
    return n;
}

static std::string digest_of(const fs::path& entry_path) {
    // <tag>.<digest> → 取 '<digest>'
    const auto name = entry_path.filename().string();
    const auto dot = name.find('.');
    return dot == std::string::npos ? name : name.substr(dot + 1);
}

int main(int argc, char** argv) {
    const fs::path root = "/tmp/deepjit_posthook_root";
    const fs::path hooks = root / "hooks";
    fs::create_directories(hooks);

    // 跨进程「冷缓存」模式：写一个指定内容的 h_cold.py，然后只算 hook 哈希 + 缓存键。
    // 由 harness 在两次独立进程调用之间改内容，验证新进程冷读取会得到不同键。
    if (argc >= 3 and std::string(argv[1]) == "coldwrite") {
        const auto content = std::string(argv[2]);
        write_file(hooks / "h_cold.py", "import sys\n" + content + "\n");
        setenv("EX_JIT_CACHE_DIR", (root / "cache").c_str(), 1);
        deep_jit::Runtime<CUDA> jit(deep_jit::Config(root, "EX", "sig", {}, {}));
        const deep_jit::cuda::CompilerOptions ov{ .post_hook = "hooks/h_cold.py" };
        const auto opts = jit.default_compiler_options.override_with(ov);
        std::printf("COLDWRITE hookhash=%s digest=%s\n",
                    opts.get_post_hook_hash(jit.config).c_str(),
                    jit.cache_key(kKernel, opts).c_str());
        return 0;
    }

    // 标准 hook 集
    write_file(hooks / "h_mark.py", kHookMark);
    write_file(hooks / "h_mark2.py", std::string(kHookMark) + "# v2\n");
    write_file(hooks / "h_corrupt.py", kHookCorrupt);
    write_file(hooks / "h_fail.py", kHookFail);

    setenv("EX_JIT_CACHE_DIR", (root / "cache").c_str(), 1);
    deep_jit::Runtime<CUDA> jit(deep_jit::Config(root, "EX", "sig", {}, {}));
    std::printf("ENV python_library_root=%s cache_root=%s\n",
                jit.config.python_library_root.c_str(), (root / "cache").c_str());

    const auto base = jit.default_compiler_options;

    // ---- 1) 无 hook：基线产物 + meta.json post_hook 字段 ----
    const auto no_hook = jit.compile_without_load("noop", kKernel);
    const auto d0 = digest_of(no_hook);
    std::printf("NOHOOK digest=%s\n", d0.c_str());
    std::printf("NOHOOK files cu=%d cubin=%d meta=%d committed=%d\n",
                int(fs::exists(no_hook / "kernel.cu")),
                int(fs::exists(no_hook / "kernel.cubin")),
                int(fs::exists(no_hook / "meta.json")),
                int(fs::exists(no_hook / ".committed")));
    std::printf("NOHOOK meta_path=%s\n", (no_hook / "meta.json").c_str());

    // ---- 2) 良性 hook：digest 变化、hook 真的跑过、meta 记录路径 ----
    const deep_jit::cuda::CompilerOptions ov_mark{ .post_hook = "hooks/h_mark.py" };
    const auto with_mark = jit.compile_without_load("noop", kKernel, ov_mark);
    const auto d1 = digest_of(with_mark);
    std::printf("MARK digest=%s distinct_from_nohook=%d mark_lines=%d\n",
                d1.c_str(), int(d1 != d0), count_lines(with_mark / "hook_mark.txt"));
    std::printf("MARK meta_path=%s\n", (with_mark / "meta.json").c_str());
    std::printf("MARK runs_as_nvcc_user=%d\n", int(fs::exists(with_mark / "hook_mark.txt")));

    // ---- 3) hook 内容哈希：进缓存键；线程内按路径缓存（同进程改文件不生效） ----
    const auto resolved_mark = base.override_with(ov_mark);
    const auto hh_before = resolved_mark.get_post_hook_hash(jit.config);
    write_file(hooks / "h_mark.py", std::string(kHookMark) + "# modified-in-process\n");
    const auto hh_after = resolved_mark.get_post_hook_hash(jit.config);
    std::printf("HOOKHASH before=%s after_inproc_modify=%s threadlocal_cached=%d\n",
                hh_before.c_str(), hh_after.c_str(), int(hh_before == hh_after));

    // 新路径 = 新内容 → 新哈希/新键（还原 h_mark.py 内容避免干扰后续）
    write_file(hooks / "h_mark.py", kHookMark);
    const auto with_mark2 = jit.compile_without_load(
        "noop", kKernel, deep_jit::cuda::CompilerOptions{ .post_hook = "hooks/h_mark2.py" });
    std::printf("MARK2 digest=%s distinct=%d\n", digest_of(with_mark2).c_str(),
                int(digest_of(with_mark2) != d1));

    // ---- 4) 覆盖语义（纯 host 逻辑）：nullopt 继承默认；'' 仍是「已配置的 hook」 ----
    deep_jit::cuda::CompilerOptions defaults;
    defaults.post_hook = "hooks/h_mark.py";
    const auto inherit = defaults.override_with(deep_jit::cuda::CompilerOptions{ .post_hook = std::nullopt });
    const auto empty = defaults.override_with(deep_jit::cuda::CompilerOptions{ .post_hook = "" });
    std::printf("OVERRIDE default=hook nullopt_inherits=%d empty_is_configured=%d empty_value='%s'\n",
                int(inherit.post_hook.has_value() and *inherit.post_hook == "hooks/h_mark.py"),
                int(empty.post_hook.has_value() and empty.post_hook->empty()),
                empty.post_hook.has_value() ? empty.post_hook->c_str() : "<null>");

    // ---- 5) 失败 hook：抛异常、不得发布（无 .committed） ----
    int fail_threw = 0;
    try {
        jit.compile_without_load("noop", kKernel,
                                 deep_jit::cuda::CompilerOptions{ .post_hook = "hooks/h_fail.py" });
    } catch (const std::exception&) {
        fail_threw = 1;
    }
    const auto d_fail = jit.cache_key(
        kKernel, base.override_with(deep_jit::cuda::CompilerOptions{ .post_hook = "hooks/h_fail.py" }));
    const auto fail_entry = jit.disk_cache.entry("noop", d_fail);
    std::printf("FAILHOOK threw=%d published_hit=%d\n", fail_threw, int(fail_entry.hit));

    // ---- 6) 破坏 hook：cubin 在「发布前」被就地改写 → 产物里的 cubin 就是坏的；load 失败 ----
    const deep_jit::cuda::CompilerOptions ov_corrupt{ .post_hook = "hooks/h_corrupt.py" };
    const auto corrupt = jit.compile_without_load("noop", kKernel, ov_corrupt);
    std::printf("CORRUPT digest=%s cubin_head=%s\n", digest_of(corrupt).c_str(),
                read_head(corrupt / "kernel.cubin", 8).c_str());
    int corrupt_load_threw = 0;
    try {
        CUDA::load(corrupt, jit.env);
    } catch (const std::exception&) {
        corrupt_load_threw = 1;
    }
    std::printf("CORRUPT load_threw=%d\n", corrupt_load_threw);

    // ---- 7) 真负载：compile() 加载 + launch；cache 命中不重跑 hook ----
    const auto kernel = jit.compile("noop", kKernel, ov_mark);
    deep_jit::cuda::LaunchOptions lo;
    lo.grid_dim = dim3(1, 1, 1);
    lo.block_dim = dim3(1, 1, 1);
    jit.launch(kernel, lo);
    cudaDeviceSynchronize();
    std::printf("LOAD launched=%d mark_lines_after_load=%d\n", 1, count_lines(with_mark / "hook_mark.txt"));

    // 清空内存缓存 → 强制走磁盘命中路径，hook 不应再跑
    jit.mem_cache.cache.clear();
    const auto kernel2 = jit.compile("noop", kKernel, ov_mark);
    std::printf("LOAD disk_hit_same_obj=%d mark_lines_after_disk_hit=%d\n",
                int(kernel2.get() != nullptr), count_lines(with_mark / "hook_mark.txt"));

    // 元数据：打印生效的 post_hook 路径与编译产物清单
    std::printf("META_JSON %s\n", (with_mark / "meta.json").c_str());
    return 0;
}
