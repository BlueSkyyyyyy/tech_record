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
#       FlashComm/python/flash_comm/ep_overlap/_ops/group_gemm_combine.py
# 对照副本（只读）：~/github/Triton-distributed/
# 背景：UniEP 第二段 MegaKernel（FC2 group-GEMM + push 式 combine）的 host
#       侧 API。device 逻辑在 kernels/cutedsl_group_gemm_combine*.py；本文件
#       与 dispatch_group_gemm.py 是镜像关系：一个把通信进来的数据交给 GEMM，
#       一个把 GEMM 的结果推回通信。
#
# 【一句话定位】FC2 group-GEMM 的 epilogue 直接把结果 push 进目标 rank 的
# staging，本 rank 不做归约 —— 归约留给随后的 topk-reduce kernel。
#
# 【本文件的注释只讲三件事】
#   1. 对称与差异：同样的三段式，但少两项编译期参数（expert_alignment、
#      dispatch_num_stages）、多一个指针表输出。
#   2. 输出不是本地张量：output_ptrs + tsri 决定「写到谁家的哪一行」，
#      staging 的行布局（token*topk+k）就在 launch 实参那里被钉死。
#   3. 谁负责清零：本 op 一个字节都不填，正确性靠两层哨兵 —— 说清代价与边界。
# 英文注释和 docstring 保持上游原样；中文只做补充，不逐句翻译。
#
# 【建议阅读顺序】（行号＝本副本行号，点进去即可）
#   ① 行 92 起  变体键：为什么 topk 在里面、expert_alignment 不在
#   ② 行 148 起  几何必须与 dispatch 一致（同一个 256）
#   ③ 行 178 起  num_sm → persistent grid 的 cluster 数
#   ④ 行 234 起  18 个实参：注意 problem_sizes_m 夹在 constexpr 中间
#   ⑤ 行 260 起  运行期入口
#   ⑥ 行 288 起  【不清零】的责任划分与两层哨兵
#   ⑦ 行 360 起  那块永远用不到的本地 C：为什么留着、代价多少
# =============================================================================
from typing import Optional

import torch

# 与 dispatch 共用同一套设施。注意这里**没有**引入 GEMM_CLUSTER_TILE_M：
# expert_alignment 的对齐在上游 owner（dispatch 阶段）就已经做完了，combine 只是
# 消费 A_padded 的打包结果，不再需要知道补齐粒度 —— 少一个编译期参数，
# 也就少一个「两边算得不一样」的机会。
from ._base import (
    CuTeDSLEPOverlapOpBase,
    cute_compile_options,
    make_moe_jit_dummies,
    mark_dynamic,
    resolve_nvcc_opt_level,
)


# 【对称性】与 CuTeDSLDispatchGroupGemmOp 完全同构的三段式：
#   _compile 选变体 → _build 编 kernel → run 搬指针。
# 差异集中在两处：
#   * 没有 __init__：本 op 不需要额外几何常量（rank/world_size 由基类吞下），
#     对齐信息也不归它管；
#   * 输出不是本地张量，而是 output_ptrs —— 一块 world_size 长的对称指针表。
#     epilogue 把每个 (token, slot) 行**直接写进目标 rank 的 staging buffer**，
#     本 rank 一行都不落。这就是 push 式 combine。
# 于是本地少了「读出来再写一遍」的搬运，代价是本地没有结果可看：
# 必须等 topk-reduce 把 staging 收回来（这也是为什么要单独一个归约 kernel）。
class CuTeDSLGroupGemmCombineOp(CuTeDSLEPOverlapOpBase):
    """Fused FC2 group-GEMM + push-mode combine (no local reduce)."""

    # 【编译期】变体键比 dispatch 那边少两项（expert_alignment、
    # dispatch_num_stages）：combine 没有 dispatch worker，也不再补对齐。
    # 但多了一个 topk —— epilogue 用它算目的槽位
    #   dst_slot = src_token_idx * topk + src_topk_idx
    # topk 一变，生成的地址算术就变了，所以它必须是变体键的一部分。
    def _compile(self, dtype: torch.dtype, experts_per_rank: int, n_out: int, hidden_in: int, topk: int, num_sm: int,
                 weight_dtype: torch.dtype, has_weight: bool):
        # 同 dispatch：arch 取自当前 device，保证 variant 反映真正要跑的卡。
        arch_major, arch_minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        # variant 元组 = 「改了就重编译」的清单。逐项为什么进键，见
        # dispatch_group_gemm 的同名注释；这里只补两处差异：
        #   num_sm → grid 的 cluster 数（persistent GEMM 的 SM 预算）；
        #   topk   → epilogue 的目的槽位算术（见上）。
        variant_args = (
            dtype,
            int(experts_per_rank),
            int(n_out),
            int(hidden_in),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(num_sm),
            weight_dtype,
            bool(has_weight),
            int(arch_major),
            int(arch_minor),
        )

        # factory 延迟到 cache miss 才调用：命中路径不 import cutlass。
        def factory():
            return self._build(dtype, experts_per_rank, n_out, hidden_in, topk, num_sm, weight_dtype, has_weight,
                               arch_major, arch_minor)

        # 同 dispatch：opt-level 按 CUDA 版本选，并且是 cache 身份的一部分。
        opt_level = resolve_nvcc_opt_level()
        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
            compile_options=("--enable-tvm-ffi", f"--opt-level {opt_level}"),
        )

    # 【构建期】cache miss 才进来；重量级 import 全在函数内。
    def _build(self, dtype: torch.dtype, experts_per_rank: int, n_out: int, hidden_in: int, topk: int, num_sm: int,
               weight_dtype: torch.dtype, has_weight: bool, arch_major: int, arch_minor: int):
        # 四个 import 与 dispatch 相同：dtype/常量、编译、torch↔cutlass dtype、
        # HardwareInfo（算可驻留 cluster 数）。
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        # dtype 白名单（bf16 / fp16）在编译期就炸。
        self._assert_supported_dtype(dtype, "MegaMoEGroupGEMMCombine")

        # 【几何必须与 dispatch 一致】sm100：(256,256)+(2,2)+2CTA；
        # sm90：(128,256)+(2,1)。每 cluster 的 M tile 都是 256，理由在 dispatch
        # 那边已经说过：A_padded 是按 256 行一块打的包，combine 遍历 group 必须用
        # 同一把尺子，否则一个 cluster tile 会横跨两个 expert，
        # problem_sizes_m 的边界失效。
        # 【补一点】这里的 N 是 FC2 的 n_out（回传 hidden），与 FC1 的 N 不同；
        # 但 M 侧的打包格式完全继承，所以几何常量不能各自为政。
        if arch_major >= 10:
            from ..kernels.cutedsl_group_gemm_combine import MegaMoEGroupGEMMCombine

            mma_tiler = (256, 256)
            cluster = (2, 2)
            use_2cta = True
        elif arch_major == 9:
            from ..kernels.cutedsl_group_gemm_combine_sm90 import MegaMoEGroupGEMMCombine

            mma_tiler = (128, 256)
            # Match GEMM_CLUSTER_TILE_M=256 used by dispatch packing.
            cluster = (2, 1)
            use_2cta = False
        else:
            raise RuntimeError(
                f"CuTeDSL group_gemm_combine requires Hopper (sm90) or newer, got sm{arch_major}{arch_minor}")

        # c_dtype 只用来造 dummy C：combine 的真实输出被推到 peer 的 staging，
        # 本地这块 C 是「签名占位」—— kernel 里 epilogue 仍按它的 layout 分块，
        # 但结果不落这里（见 run 里的 c_scratch）。
        ab_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
        c_dtype = ab_dtype

        # 与 dispatch 同一个 grid 公式：
        #   min(硬件可驻留 cluster 数, num_sm // cluster_size)。
        # 差别在语义：combine 的 kernel 里没有 dispatch worker，全部 SM 都在做
        # GEMM + push，所以 num_sm 就是这个 MegaKernel 的计算预算。
        # 改它 → 重编译（grid 是编译期常量）；所以 overlap 调参时要先把 num_sm
        # 的取值域枚举好，别在训练中途反复重编。
        hardware_info = utils.HardwareInfo()
        max_active_clusters = min(
            hardware_info.get_max_active_clusters(cluster[0] * cluster[1]),
            max(1,
                int(num_sm) // (cluster[0] * cluster[1])),
        )

        # 【同一套最小 dummy 原则】JIT 只读 rank / dtype / 哪个 mode 的 stride==1；
        # 数值与 dynamic 轴大小一律不读，所以 A/B/C/psm 都可以是「形状能钉住
        # 不变式」的极小张量。生产形状下真实分配要十几 GB，这里几 MB 够用。
        # JIT only inspects dtype / rank / mode-1-stride-1 of these tensors;
        # values and dynamic-axis sizes are never read.
        dummy_A, dummy_B, dummy_C, dummy_psm = make_moe_jit_dummies(
            num_experts=experts_per_rank,
            n=n_out,
            hidden_in=hidden_in,
            ab_dtype=cutlass_torch.dtype(ab_dtype),
            c_dtype=cutlass_torch.dtype(c_dtype),
        )
        # 四个一维表 dummy，各对应 kernel 的一个真实指针表：
        #   output_ptrs        → 目标 rank 的 staging 基址表（push 的落点）；
        #   tsri               → 行级元数据（src_rank / src_topk_idx / src_token_idx）；
        #   dispatched_weights → 每行一个 top-k 标量（has_weight 路径）；
        #   weight_output_ptrs → 目标 rank 的 (max_m, topk) 权重表基址。
        # 它们只提供 dtype + rank，内容永远不被读。
        dummy_output_ptrs = torch.empty(self.world_size, dtype=torch.int64, device="cuda")
        dummy_tsri = torch.empty((1, ), dtype=torch.int64, device="cuda")
        dummy_dispatched_weights = torch.empty(
            (1, ),
            dtype=weight_dtype,
            device="cuda",
        )
        dummy_weight_output_ptrs = torch.empty(
            self.world_size,
            dtype=torch.int64,
            device="cuda",
        )

        opt_level = resolve_nvcc_opt_level()
        # kernel 构造参数：accum dtype、是否 2-CTA、MMA tile、cluster 形状。
        # 注意这里没有 dispatch_num_stages —— combine 没有 dispatch 那条流水。
        kernel = MegaMoEGroupGEMMCombine(
            cutlass.Float32,
            use_2cta,
            mma_tiler,
            cluster,
        )
        # 【契约清单】18 个实参（17 个数据/常量 + 1 个 stream），顺序严格对应
        # kernel.__call__ 的位置参数：
        #   ① 8 个运行期张量：A / B / C / output_ptrs / tsri / dispatched_weights /
        #      weight_output_ptrs / psm；
        #   ② 9 个 constexpr：experts_per_rank、n_out、hidden_in、topk、rank、
        #      world_size、total_num_clusters、max_active_clusters、has_weight；
        #   ③ stream（fake stream，launch 时由 tvm-ffi 环境提供）。
        # 【注意顺序】problem_sizes_m 夹在 constexpr 中间（rank/world_size 之后）——
        # 因为它在 kernel 签名里就排在那儿。顺序错一位，JIT 会按错的描述符解释
        # 内存，通常不报错，只是数值全错。
        return cute.compile(
            kernel,
            mark_dynamic(dummy_A, enable_tvm_ffi=True),
            mark_dynamic(dummy_B, enable_tvm_ffi=True),
            mark_dynamic(dummy_C, enable_tvm_ffi=True),
            mark_dynamic(dummy_output_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_tsri, enable_tvm_ffi=True),
            mark_dynamic(dummy_dispatched_weights, enable_tvm_ffi=True),
            mark_dynamic(dummy_weight_output_ptrs, enable_tvm_ffi=True),
            int(experts_per_rank),
            int(n_out),
            int(hidden_in),
            int(topk),
            int(self.rank),
            int(self.world_size),
            mark_dynamic(dummy_psm, assumed_align=16, enable_tvm_ffi=True),
            int(max_active_clusters),
            int(max_active_clusters),
            int(1 if has_weight else 0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=cute_compile_options(enable_tvm_ffi=True, opt_level=opt_level),
        )

    # 【运行期入口】和 dispatch 一样：反推变体参数 → 取 cache handle →
    # 封送指针 → launch。不同点：这里没有本地 output 张量可校验（输出直接进
    # peer staging），所以校验几乎全部落在「指针表 + weight 侧信道」上。
    def run(self, *, A_padded: torch.Tensor, B: torch.Tensor, token_src_rank_topk_and_indices: torch.Tensor,
            recv_expert_counts: torch.Tensor, output_ptrs: torch.Tensor, num_sm: int, topk: int,
            weight_dtype: torch.dtype, dispatched_weights: Optional[torch.Tensor] = None,
            weight_output_ptrs: Optional[torch.Tensor] = None) -> None:
        """Launch the fused groupgemm+combine kernel.

        Note: we do NOT zero the staging buffer between calls. The
        topk-reduce kernel skips slots whose ``topk_indices`` carry the
        drop sentinel, so stale bytes never reach the accumulator (saves
        a full-bandwidth fill per call).

        Optional ``dispatched_weights`` (1D ``weight_dtype`` sized
        to ``A_padded.shape[0]``) and ``weight_output_ptrs`` enable the
        side-loaded weight push: the epilogue writes one scalar per
        row to the source rank's symmetric ``(max_m, topk)`` weight
        buffer. Both must be supplied together or both omitted.
        """
        # 【谁负责清零】本 op 一个字节都不清零，这是刻意的：
        #   * staging（目标 rank 的 (max_m*topk, n_out)）**不 memset**：多一次
        #     全带宽 fill 就是每拍多搬 hidden×max_m×topk 字节，正好把 push 的
        #     带宽优势吃掉；
        #   * 安全性靠两层哨兵：combine 侧 tsri 的 -1（rank==-1 的槽位不写），
        #     reduce 侧 topk_indices 的 num_experts（`expert_idx < num_experts`
        #     不成立就不 load、不累加）—— 陈旧字节永远进不了累加器；
        #   * 需要预置的东西全在上游：token_src_rank_topk_and_indices 由 host
        #     填 -1、expert_signals / counters 由 context.reset_expert_signals()
        #     归零、dispatch 侧 staging 由 _prepare_dispatch_buffers 拷入并 barrier。
        # 【一句话】「不清零」不是省事，而是把正确性责任交给哨兵协议。
        experts_per_rank = int(recv_expert_counts.shape[0])
        # 与 dispatch 一致：experts_per_rank / hidden_in / n_out 都从张量形状
        # 反推，反推值直接成为 constexpr —— 推错不会报错，只会编出一个几何不对的
        # kernel。
        hidden_in = int(A_padded.shape[1])
        n_out = int(B.shape[0])

        # 只校验 B：必须 (N, K, L) 且 K == A_padded.shape[1]、
        # L == experts_per_rank。
        # 【为什么这里没有 output 校验】本 kernel 不写本地 output（写的是 peer 的
        # 地址，由 output_ptrs + tsri 决定），本地没有形状可查。越界风险因此转移到
        # 「指针表 + 槽位算术」上，靠下面几条校验和 tsri 的哨兵兜。
        if B.shape != (n_out, hidden_in, experts_per_rank):
            raise ValueError(f"B must be (N, K={hidden_in}, L={experts_per_rank}); "
                             f"got {tuple(B.shape)}")

        # 【必须成对】dispatched_weights 与 weight_output_ptrs 要么都给、要么都
        # 不给：has_weight 是 constexpr（决定编哪个 cubin），而两者分别是「读哪」
        # 和「写到哪」，缺一个就会编出「读一个不存在的源 / 写一个不存在的目标」的
        # 变体。用 `!=` 做异或判断，比两条独立的 if 更不容易漏。
        has_weight = dispatched_weights is not None
        if has_weight != (weight_output_ptrs is not None):
            raise ValueError("dispatched_weights and weight_output_ptrs must be "
                             "supplied together")
        if has_weight:
            # 逐条对应一个编译期假设：
            #   dtype == weight_dtype → 变体键，错了会按错误位宽解引用；
            #   1D                    → epilogue 按行号线性索引；
            #   长度 == M_padded      → 同上，短了就是越界读；
            #   contiguous            → 唯一一条 dispatch 那边没有的检查：
            #     epilogue 直接用「行号 × 元素宽度」算地址，非连续视图会让它读到
            #     错行（形状全对、结果错位，属于最难查的那类）。
            if dispatched_weights.dtype != weight_dtype:
                raise TypeError(f"dispatched_weights dtype must be {weight_dtype}; got {dispatched_weights.dtype}")
            if dispatched_weights.ndim != 1:
                raise ValueError("dispatched_weights must be 1D; got shape "
                                 f"{tuple(dispatched_weights.shape)}")
            if dispatched_weights.shape[0] != A_padded.shape[0]:
                raise ValueError("dispatched_weights.shape[0] != A_padded.shape[0]: "
                                 f"{dispatched_weights.shape[0]} vs {A_padded.shape[0]}")
            if not dispatched_weights.is_contiguous():
                raise ValueError("dispatched_weights must be contiguous")
            # 指针表的 dtype 检查（int64）。
            # 【边界】这里没有校验 shape == (world_size,)，而 dispatch 那边对
            # input_weight_ptrs 是查了的 —— 如果上游给了一张短表，
            # weight_output_ptrs[rank] 就会越界读。目前靠 context 里
            # SymmetricTensor.ptrs 的长度保证；要更稳可以在这一行前面补一条。
            if weight_output_ptrs.dtype != torch.int64:
                raise TypeError("weight_output_ptrs must be int64; got "
                                f"{weight_output_ptrs.dtype}")

        # 取 cache handle（首次调用才真编）。topk 也参与变体键，所以这里传错会
        # 编出另一个 kernel，而不是静默沿用旧 kernel 的槽位算术。
        compiled = self._compile(
            dtype=A_padded.dtype,
            experts_per_rank=experts_per_rank,
            n_out=n_out,
            hidden_in=hidden_in,
            topk=topk,
            num_sm=num_sm,
            weight_dtype=weight_dtype,
            has_weight=has_weight,
        )

        # ``c_scratch`` is unused (epilogue pushes go to peer GMEM) but
        # the kernel signature still requires a local C for layout
        # consistency with the standalone GEMM.
        # 【代价】这块 C 完全用不到，但每次调用都按 (M, N) 分配一次：
        # M_padded × n_out × 2 B，生产上可达上百 MB。走 caching allocator 不会
        # 真清零，但它确实占着显存，也让「零本地输出」这个卖点打了折扣。
        # 保留它的原因：kernel 签名与 standalone GEMM 共用同一套 epilogue 分块与
        # layout，去掉一个参数就要 fork 一份 kernel。
        c_scratch = torch.empty((A_padded.shape[0], n_out), dtype=A_padded.dtype, device=A_padded.device)

        # 两条路径的指针封送。注意 else 分支复用 output_ptrs 当
        # weight_output_ptrs：两者都是 (world_size,) int64，且 has_weight=0 时该
        # 实参被 DCE，内容无关紧要 —— 省一次分配。
        if has_weight:
            launch_dispatched_weights = dispatched_weights
            launch_weight_output_ptrs = weight_output_ptrs
        else:
            # ``has_weight=0`` DCEs every read of these args; just give
            # the JIT a tensor of the right shape/dtype.
            launch_dispatched_weights = torch.empty(
                (A_padded.shape[0], ),
                dtype=weight_dtype,
                device=A_padded.device,
            )
            launch_weight_output_ptrs = output_ptrs

        # 【真正 launch】8 个实参：
        #   A_padded / B / c_scratch → FC2 的输入、权重、占位输出；
        #   output_ptrs              → 目标 rank staging 的基址表（push 落点）；
        #   tsri                     → 行级元数据：这一行来自哪个 rank 的哪个 token、
        #                              第几个 topk 槽，以及 -1 丢弃哨兵；
        #   launch_dispatched_weights / launch_weight_output_ptrs → 权重侧信道；
        #   recv_expert_counts       → 每个 expert 的真实 token 数 = problem_sizes_m。
        # 【staging 布局在这里被钉死】目的行 = src_token_idx * topk + src_topk_idx，
        # 必须与 topk-reduce 读侧的 stage_row_idx = token_idx * topk + k 逐字一致。
        # 两侧只要有一处改成「k-major」，结果就整段错位，而且不报错。
        compiled(
            A_padded,
            B,
            c_scratch,
            output_ptrs,
            token_src_rank_topk_and_indices,
            launch_dispatched_weights,
            launch_weight_output_ptrs,
            recv_expert_counts,
        )
