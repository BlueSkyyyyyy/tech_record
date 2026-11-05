################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

# 【中文批注副本】只插入了整行注释，没有改动、删除或重排任何一行代码；
# 可以用 tools/verify_annotations.py 校验（过滤注释行后与上游逐行一致）。
#
# 上游：ByteDance-Seed/Triton-distributed @ 7908e4e
#       FlashComm/python/flash_comm/ep_overlap/_ops/topk_reduce.py
# 对照副本（只读）：~/github/Triton-distributed/
# 背景：UniEP 两个 MegaKernel 之后唯一「把散落结果收回来」的一步。combine 把
#       每个 token 的 topk 份输出推散到 staging（行号 token*topk+k），本 op
#       按 topk_indices 取回、定序相加，写出 (num_tokens, hidden)。
#       device 逻辑在 kernels/cutedsl_topk_reduce.py。
#
# 【一句话定位】确定性 top-k 归约的 host 封装：一次写者（combine）+ 单一
# 归约者（本 kernel）= 与到达顺序无关。
#
# 【本文件的注释只讲三件事】
#   1. 为什么必须是独立 kernel：塞回 combine 的 epilogue 只剩 atomicAdd 或
#      「最后到达者求和」两条路，两条都把数值绑到时序上。
#   2. 编译期/运行期怎么切：这是全链路最小的 op，边界看得最清楚。
#   3. dummy 的「最小」有下限：动态轴可以给 1，constexpr 轴必须给真值。
# 英文注释和 docstring 保持上游原样；中文只做补充，不逐句翻译。
#
# 【建议阅读顺序】（行号＝本副本行号，点进去即可）
#   ① 行 76 起  「为什么要单独一个 kernel」：两条错误的路
#   ② 行 83 起  变体键只有四项，num_tokens 不在里面
#   ③ 行 146 起  dummy + 参数表：hidden 为什么不能用 1
#   ④ 行 162 起  运行期只传 4 个实参
# =============================================================================
import torch

# 这里用的是 mark_compact_dynamic，而不是 GEMM 那边的 mark_dynamic：
# compact-shape-dynamic 只放开 mode-0（token 轴），其余维度保持静态。
# 这对本 kernel 是必需的 —— hidden 要进 constexpr 决定 num_atoms / num_tiles，
# topk_indices 的列数要决定常量循环上界。用错 helper 会把 hidden 也标成动态，
# 128-bit 原子的分块就失去编译期上界。
from ._base import (
    CuTeDSLEPOverlapOpBase,
    cute_compile_options,
    mark_compact_dynamic,
)


# 【为什么要单独一个 kernel】combine 的 epilogue 是 push：每个源 rank 把
# (token, slot) 这一行写进目标 rank 的 staging，一次写者、互不重叠。此时目标侧
# 若想在 kernel 内直接求和，只有两条路：
#   (a) atomicAdd 到 output —— 求和顺序由到达顺序决定，确定性没了；
#   (b) per-token 计数 + 最后一个到达者求和 —— 顺序又变成「谁最后到」。
# 两条路都把「数值」绑到「时序」上。本 op 的解法是把归约从通信里摘出来：
# 【一次写者（combine）+ 单一归约者（本 kernel）= 与到达顺序无关】。
# 代价：多一次 kernel launch + 重读一遍 staging（带宽 ≈ (topk+1) 倍输出）；
# 换来的是求和顺序与串行执行完全一致。
class CuTeDSLTopkReduceOp(CuTeDSLEPOverlapOpBase):
    """Cached topk-reduce launcher."""

    # 【编译期】变体键只有四项：dtype / hidden_size / topk / num_experts。
    # 它们全部决定循环上界或指令形状：topk 展开成常量循环、hidden 决定
    # num_atoms/num_tiles、num_experts 是丢弃哨兵的比较上界。
    # num_tokens 不在键里 —— 它是唯一的动态轴，每拍都能变、不需要重编译。
    def _compile(self, dtype: torch.dtype, hidden_size: int, topk: int, num_experts: int):
        # 只看这四项，不看 num_tokens：这张元组就是「哪些量属于布局、哪些量属于
        # 数据」的分界线。
        variant_args = (
            dtype,
            int(hidden_size),
            int(topk),
            int(num_experts),
        )

        # factory 同样只在 cache miss 时调用。
        def factory():
            return self._build(dtype, hidden_size, topk, num_experts)

        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
        )

    # 【构建期】三个 dummy 都极小，理由与 GEMM 那边相同：JIT 只读 rank / dtype /
    # 哪个 mode 的 stride==1，不读数值，也不读 dynamic 轴大小。
    def _build(self, dtype: torch.dtype, hidden_size: int, topk: int, num_experts: int):
        import cutlass
        import cutlass.cute as cute
        # kernel 类从 kernels/ 下取；它同时是 kernel 定义与 launch 描述。
        from ..kernels.cutedsl_topk_reduce import MoETopkReduceBlockPerToken

        # dtype 白名单：只有 bf16 / fp16（128-bit 原子要求元素宽度确定）。
        self._assert_supported_dtype(dtype, "CuTeDSL topk reduce")
        # dummy 的列数用真实 hidden_size —— 这是与 GEMM 那边最重要的区别：
        # hidden 是 constexpr，JIT 要按它算 num_atoms/num_tiles 并展开循环，
        # 所以这一维不能写成 1。
        # 【对「最小 dummy」的修正】最小化只对「动态轴」成立，对 constexpr 轴必须
        # 给真值 —— 否则要么编不过，要么编出一个循环上界错的 kernel。
        dummy_staging = torch.empty(
            (1, hidden_size),
            dtype=dtype,
            device="cuda",
        )
        # topk_indices dummy：(1, topk)，int32。列数用真 topk（常量循环上界），
        # 行数是动态轴，给 1 就够。
        dummy_topk = torch.empty(
            (1, topk),
            dtype=torch.int32,
            device="cuda",
        )
        # output dummy：(1, hidden_size)，只提供 rank / dtype / stride-1。
        dummy_output = torch.empty(
            (1, hidden_size),
            dtype=dtype,
            device="cuda",
        )
        # 【契约清单】实参顺序 = kernel.__call__ 的位置参数：
        #   ① 三个运行期张量：staging / topk_indices / output。都用
        #      mark_compact_dynamic 包过，assumed_align=16 对应 128-bit 原子，
        #      topk_indices 用 4（int32 的对齐）；
        #   ② 三个 constexpr：hidden_size / topk / num_experts；
        #   ③ 一个 cutlass.Int32(0)：num_tokens 的编译期占位 —— 它是动态标量，
        #      编译时只需要确定类型，值无意义；
        #   ④ fake stream：launch 时由 tvm-ffi 环境提供。
        # 【assumed_align=16 不是可选项】它是对齐承诺：每行起点 16 B 对齐，
        # LDG.128 才不会跨行拆分。hidden % 8 == 0（kernel 里有 assert）保证
        # 行宽 hidden×2 B 是 16 B 的倍数，这条承诺才成立。
        return cute.compile(
            MoETopkReduceBlockPerToken(),
            mark_compact_dynamic(dummy_staging, assumed_align=16, enable_tvm_ffi=True),
            mark_compact_dynamic(dummy_topk, assumed_align=4, enable_tvm_ffi=True),
            mark_compact_dynamic(dummy_output, assumed_align=16, enable_tvm_ffi=True),
            int(hidden_size),
            int(topk),
            int(num_experts),
            cutlass.Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=cute_compile_options(enable_tvm_ffi=True),
        )

    # 【运行期入口】run() 只传 3 个张量 + 1 个动态标量：三个 constexpr 已经烧进
    # 产物，stream 由 tvm-ffi 环境提供 —— 这是「编译期/运行期解耦」在最小 op 上的
    # 完整样子：签名里能变的只剩数据。
    def run(self, *, staging: torch.Tensor, topk_indices: torch.Tensor, output: torch.Tensor, num_experts: int) -> None:
        """Run topk reduce. ``hidden_size`` is read from ``output.shape[1]``."""
        # hidden 从 output 反推（它是 constexpr 候选），并立刻校验 staging 的列宽。
        # 【不查会怎样】kernel 按 hidden 算出的 num_atoms/num_tiles 与 staging 的
        # 实际行宽不符 → 128-bit 原子跨行读，读到下一个 token 的数据：
        # 数值错、但不崩，且只在「staging 比 output 宽/窄」时出现。
        hidden_size = int(output.shape[1])
        # 这是本条链路上唯一一处「形状一致性」断言：staging 与 output 必须同宽，
        # 否则同一次 copy 的源和目的是两套坐标。
        if staging.shape[1] != hidden_size:
            raise ValueError(f"staging hidden ({staging.shape[1]}) != output hidden "
                             f"({hidden_size})")
        # 取 cache handle。topk 从 topk_indices 的列数取（而不是另立参数），
        # 避免出现「布局按 8 算、归约按 4 算」这种错配。
        compiled = self._compile(
            dtype=output.dtype,
            hidden_size=hidden_size,
            topk=int(topk_indices.shape[1]),
            num_experts=num_experts,
        )
        # 【4 个实参】staging（按 token*topk+k 行排布）、topk_indices（含丢弃
        # 哨兵 num_experts）、output、以及唯一动态标量 num_tokens。
        # grid 不在这里：它由 kernel 内部按 num_tokens 决定（一个 CTA 一个 token），
        # host 侧完全不碰 grid —— 少一个可以配错的旋钮。
        compiled(
            staging,
            topk_indices,
            output,
            int(output.shape[0]),
        )
