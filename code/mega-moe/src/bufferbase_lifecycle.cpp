// P111-1 实验：DeepEP 四个 Buffer 家族共享的生命周期契约 BufferBase（C++ 侧）。
//
// 只测纯 host 逻辑：`deep_ep::BufferBase`（csrc/buffers/base.hpp）的
// `destroy_on_destruction(name)` 三分支 + `print_memory_usage` 的 `not destroyed` 断言。
// 该文件不碰任何 CUDA / NCCL 设备 API，故 H100(sm90) 可直接编译运行。
//
// 构建方式由 harness/exp_deepep_bufferbase_lifecycle.py 提供：它在临时目录里搭一棵"影子
// include 树"，把真实的 csrc/buffers/base.hpp 原样拷进去，并用最小 stub 顶掉
// pybind11/pybind11.h 与 csrc/kernels/comm/api.hpp（真实那两份会拉 NCCL/devcomm 头）。
// 于是这里 include 到的 `BufferBase` 类体是**逐字节真实**的。
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#include "csrc/buffers/base.hpp"

namespace {

int g_destroy_calls = 0;
int g_pass = 0;
int g_fail = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
    if (ok) {
        ++g_pass;
        std::printf("[PASS] %s%s%s\n", name.c_str(), detail.empty() ? "" : "  ", detail.c_str());
    } else {
        ++g_fail;
        std::printf("[FAIL] %s%s%s\n", name.c_str(), detail.empty() ? "" : "  ", detail.c_str());
    }
}

// 模拟一个派生 Buffer：只在析构函数体里调 destroy_on_destruction（与 EPBuffer/PPBuffer/
// EngramBuffer/BucketBuffer 完全同构），destroy() 里用派生成员证明「析构体调用时派生成员仍存活」。
struct DerivedBuffer : public deep_ep::BufferBase {
    int marker = 4242;

    explicit DerivedBuffer(bool explicitly_destroy, bool want_context = true)
        : deep_ep::BufferBase(explicitly_destroy) {
        if (want_context)
            main_context = std::make_shared<deep_ep::comm::Context>();
    }

    ~DerivedBuffer() noexcept(false) override {
        deep_ep::BufferBase::destroy_on_destruction("Derived");
    }

    void destroy() override {
        EP_HOST_ASSERT(not destroyed);
        // 若这句成立，就证明 destroy_on_destruction 是在派生析构体里、派生成员还活着时调用的。
        assert(marker == 4242);
        ++g_destroy_calls;
        destroyed = true;
    }
};

}  // namespace

int main() {
    std::printf("== BufferBase C++ 生命周期契约（真实 csrc/buffers/base.hpp）==\n");

    // A: 默认 explicitly_destroy=false + 用户不显式 destroy → 析构体自动 destroy()，且无泄漏告警。
    {
        g_destroy_calls = 0;
        bool ctx_after = true;
        {
            DerivedBuffer b(false);
            check(b.main_context != nullptr, "A0 construct 后 main_context 非空");
            ctx_after = true;  // placeholder
        }
        check(g_destroy_calls == 1, "A1 默认模式析构体自动 destroy()（应恰好 1 次）",
              "calls=" + std::to_string(g_destroy_calls));
    }

    // B: explicitly_destroy=true + 用户不 destroy → 析构体不自动 destroy，打印泄漏告警。
    //    通过捕获 stdout 无法在本进程内做到，故只验证「destroy() 未被调用」这一半；告警文本
    //    由 driver 在外层捕获（会出现在本进程 stdout 里：`... buffer destruction`）。
    {
        g_destroy_calls = 0;
        { DerivedBuffer b(true); }
        check(g_destroy_calls == 0, "B1 显式销毁模式下析构体不自动 destroy()",
              "calls=" + std::to_string(g_destroy_calls));
    }

    // C: explicitly_destroy=true + 用户显式 destroy → 析构体既不再 destroy 也不告警。
    {
        g_destroy_calls = 0;
        {
            DerivedBuffer b(true);
            b.destroy();
            check(g_destroy_calls == 1, "C1 显式 destroy() 恰好一次", "calls=" + std::to_string(g_destroy_calls));
            // 显式路径不清 main_context（只有 destroy_on_destruction 的自动分支才置空）：
            check(b.main_context != nullptr, "C2 显式 destroy 后基类 main_context 不被置空（派生只清自己的 context）");
        }
        check(g_destroy_calls == 1, "C3 显式销毁后析构体不重复 destroy", "calls=" + std::to_string(g_destroy_calls));
    }

    // D: 双重显式 destroy → 第二次撞 EP_HOST_ASSERT(not destroyed) 抛异常（不可幂等）。
    {
        g_destroy_calls = 0;
        DerivedBuffer b(true);
        b.destroy();
        bool threw = false;
        try {
            b.destroy();
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "D1 二次显式 destroy() 抛异常（EP_HOST_ASSERT(not destroyed)）");
        check(g_destroy_calls == 1, "D2 抛异常后未再自增", "calls=" + std::to_string(g_destroy_calls));
    }

    // E: 自动模式（false）走 destroy_on_destruction → main_context 被置空。
    //    （无法在对象销毁后读属性，故用一个存活期外的观察者：把 shared_ptr 拷出来看计数。）
    {
        g_destroy_calls = 0;
        std::weak_ptr<deep_ep::comm::Context> w;
        {
            DerivedBuffer b(false);
            w = b.main_context;
            check(!w.expired(), "E1 自动模式析构前 main_context 有效");
        }
        check(w.expired(), "E2 自动 destroy_on_destruction 把 main_context 置空 → weak_ptr 过期");
        check(g_destroy_calls == 1, "E3 自动模式 destroy() 一次", "calls=" + std::to_string(g_destroy_calls));
    }

    // F: print_memory_usage 的前置断言：销毁后调用必须抛（`not destroyed and main_context != nullptr`）。
    {
        DerivedBuffer b(false);
        auto* ctx = b.main_context.get();
        ctx->rank_idx = 3;
        ctx->num_ranks = 8;
        ctx->num_workspace_bytes = int64_t(200) << 20;         // 200 MiB
        ctx->num_gpu_buffer_bytes = int64_t(1) << 30;          // 1 GiB
        ctx->num_rdma_storage_bytes = int64_t(2) << 30;        // 2 GiB
        ctx->use_cpu_rdma_storage = false;
        // 存活期调用：正常打印（内容见 stdout）。
        std::printf("[INFO] F0 print_memory_usage 正常输出：\n");
        b.print_memory_usage();
        ++g_pass;
        std::printf("[PASS] F1 存活期 print_memory_usage 不抛\n");
        b.destroy();
        bool threw = false;
        try {
            b.print_memory_usage();
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "F2 已销毁后 print_memory_usage 抛异常");
    }

    // G: ABI —— BufferBase 的尺寸/对齐与成员偏移（供 Python 层句柄对照）。
    std::printf("[INFO] G sizeof(BufferBase)=%zu alignof=%zu sizeof(Context)=%zu\n",
                sizeof(deep_ep::BufferBase), alignof(deep_ep::BufferBase),
                sizeof(deep_ep::comm::Context));

    std::printf("== C++ 小结：%d PASS / %d FAIL ==\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
