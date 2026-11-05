#!/usr/bin/env python3
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

# =============================================================================
# 【本文件】MegaMoEGroupGEMMCombine —— FC2 GroupGEMM + push 模式 combine 的融合 MegaKernel
#
# 上游：ByteDance-Seed/Triton-distributed，commit 7908e4e
#       路径 FlashComm/python/flash_comm/ep_overlap/kernels/cutedsl_group_gemm_combine.py
#       只读上游副本（对照用）：~/github/Triton-distributed/ 下同一相对路径
# 一句话定位：UniEP（arXiv 2604.19241）两个 MegaKernel 中的第二个。FC2 是 MoE 的
#   down-projection：A(total_padded_tokens, K=hidden) x B(N=hidden_out, K, L=num_experts)，
#   结果一边算出来、一边按 (token, topk) 的路由关系「推」到目标 rank 的 staging buffer；
#   kernel 结束时 combine 已经完成，没有本地 C 的落地，也没有第二轮 all-to-all。
#   「MegaKernel」= GEMM 与 EP 通信在同一个 kernel、同一批 warp 内完成。
#
# 本文件的注释只讲三件事：
#   1. 6 个 warp 的分工，以及为什么比 dispatch 版（8 warp）少两个；
#   2. push 模式的数据路径 TMEM → RMEM → SMEM → 对端 GMEM，以及每级 buffer 的复用条件；
#   3. 调度输入（1D problem_sizes_m / constexpr N,K）、SMEM 预算与 stage 数之间的取舍。
#
# 建议阅读顺序：类 docstring 的 warp 表 → __init__ / _setup_attributes（常量怎么来）
#   → __call__ 的参数（谁是编译期的）→ kernel 的三段（TMA / MMA / EPILOGUE）
#   → 最后回到 _compute_stages 看 SMEM 账本。
# =============================================================================
from typing import Type, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    create_initial_search_state, )

from .m_contig_group_tile_scheduler import MContiguousGroupTileScheduler

from .cutedsl_utils import decode_token_src_rank_topk_and_indices


# 【与第一个 MegaKernel 的关系】同目录下还有 cutedsl_dispatch_group_gemm.py（FC1 前的
# dispatch + GroupGEMM）：它把「收 token」的 all-to-all 与 FC1 的 GEMM 融在一起。
# 本文件是对称的另一半：「发 token」的 all-to-all 与 FC2 的 GEMM 融在一起。
# 两者共用同一个 tile 调度器（m_contig_group_tile_scheduler.py）与同一套 cluster/TMEM 骨架，
# 区别只在通信方向：dispatch 版 = 拉（pull）+ 等信号；本文件 = 推（push）+ 不等信号。
class MegaMoEGroupGEMMCombine:
    """Fused M-contig grouped GEMM + EP push-combine (Blackwell SM100).

    Warp layout (6 warps, 192 threads):
      * Warps 0-3: epilogue (TMEM -> RMEM -> SMEM -> peer GMEM SIMT push).
      * Warp 4:    MMA (tcgen05 UMMA mainloop).
      * Warp 5:    TMA (A/B GMEM -> SMEM).

    Epilogue: TMEM -> RMEM -> single-slot SMEM -> 16-lane cooperative
    SIMT push.
    """
    # ---------------------------------------------------------------------------
    # 【6 个 warp 的分工：192 线程 = 1 个完整 warpgroup + 2 个 warp】
    #   warp 0-3 → epilogue：TMEM→RMEM→SMEM→对端 GMEM 的 SIMT push。
    #              必须正好 4 个：TMEM 有 128 条 lane，一个 warp 32 条 lane，4 个 warp 才能一对一
    #              覆盖 CTA tile 的全部 128 行 —— 每行的搬运者就是它在 TMEM 里的所有者。
    #   warp 4   → MMA：tcgen05 UMMA mainloop，2-CTA 指令下只有 leader CTA 真正发指令。
    #   warp 5   → TMA：A/B 的 GMEM→SMEM 生产者，也是唯一 prefetch TMA 描述符的 warp。
    #
    # 【为什么比 dispatch 版少两个 warp】dispatch 版是 8 warp：epilogue 4 + dispatch 2
    # （warp 4-5，负责 pull 的 TMA G2S/S2G 与 per-expert 的 release-store）+ MMA（warp 6）
    # + TMA（warp 7）。这里少掉的正是那 2 个「通信 warp」，原因不在省资源，而在通信的驱动方式：
    #   * dispatch 的通信要**被 GEMM 等**（A 没到齐不能算），所以必须独立成 warp，用 mbarrier 和
    #     信号把「到齐」这件事显式表达出来，才能和 GEMM 解耦；
    #   * combine 的通信要**被 GEMM 的结果驱动**（先有 C 才谈得上推），而且目标地址逐行不同，
    #     TMA 描述符表达不了 → 只能由已经持有数据的那批 lane 自己发 SIMT store。
    #   → 【一句话】通信没有独立成 warp，而是并进 epilogue 的寄存器持有者；少 2 个 warp 是结果，
    #     不是优化目标。顺带的收益是 4 个 epi warp 拿到更多寄存器预算。
    # 【代价】push 是同步 SIMT store，发射与完成不分离；epi warp 的 store 队列深度直接决定它
    #   能不能跟上 TMEM 的产出（见 epilogue 段的单槽讨论）。

    kMaxWorldSize: int = 128
    reserved_smem_bytes = 1024

    # ---------------------------------------------------------------------------
    # 【成员与两个 NamedBarrier】
    #   epilog_sync_barrier(barrier 1)：只含 4 个 epi warp，用在「TMEM 分配完成」与收尾两个汇合点。
    #     MMA/TMA warp 不参与，否则它们会被卷进 epilogue 的私有同步里白等一轮。
    #   tmem_alloc_barrier(barrier 2)：epi + MMA。因为 MMA warp 也要 wait_for_alloc 才能拿到
    #     TMEM 基址；真正执行 allocate 的只有 epi 的 warp 0（allocator_warp_id）。
    #   reserved_smem_bytes=1024：mbarrier 区与结构体对齐的固定开销，先扣掉再算 stage 数。
    # 【命名 barrier 的坑】num_threads 必须精确等于参与线程数，多算一个就死锁 ——
    # 这也是「warp 数一改，两处 barrier 定义必须跟着改」的原因。
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple,
        cluster_shape_mn: tuple,
    ):
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.cta_group = (tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE)

        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.occupancy = 1

        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len((self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id))
        self.num_threads_per_warp: int = 32

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tma_load_bytes = 0

    # ---------------------------------------------------------------------------
    # 【这一段是整份 kernel 的常量推导链，全部在 JIT 期完成】
    #   dtype/layout → tiled_mma → mma_tiler 的 K（= 指令 K x 4）→ cta_tile（2-CTA 时 M 对半）
    #   → cluster_tile（再乘 cluster 形状）→ epi_tile → stage 数 → SMEM layout → TMEM 列数。
    # kernel 后面出现的所有循环边界、断言、取模都来自这条链，编译后不留指令。
    # 【为什么 K 取指令 K 的 4 倍】UMMA 的 K 步长固定，一次 mainloop 迭代多喂几个 K block 可以
    # 摊薄 mbarrier 往返；4 是 CUTLASS 在这类 GEMM 上的惯用值，也是 num_kblocks 内层循环长度。
    def _setup_attributes(self):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mnk = tuple(x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1)))
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape, ),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        cta_tile_n = self.cta_tile_shape_mnk[1]
        cta_tile_m = self.cta_tile_shape_mnk[0]
        # epi_tile 的 N 取 128 而不是 cta_tile_n(=256)：N 方向切成 2 个 subtile，
        # subtile_cnt = cta_tile_n / epi_tile_n = 2（epilogue 里是 constexpr）。
        # 切两段的意义不是省 SMEM，而是让「R2S 写 SMEM」与「push 读 SMEM」以 subtile 为粒度交错，
        # 从而只用 1 个 SMEM 槽就转得起来（见 epilogue 的单槽论证）。
        # 当 cta_tile_n 不是 128 的倍数时退回 CUTLASS 通用启发式，代价是 epi_tile 与行宽不再整齐，
        # 下面「lane 即行」的假设也就不再成立。
        epi_tile_n_target = 128
        if cta_tile_n % epi_tile_n_target == 0:
            self.epi_tile = (
                cute.make_layout(cta_tile_m),
                cute.make_layout(epi_tile_n_target),
            )
        else:
            self.epi_tile = utils.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.use_2cta_instrs,
                self.c_layout,
                self.c_dtype,
            )

        # stage 数由 SMEM 预算反推（公式在文件末尾 _compute_stages）：先扣 1KB 保留 + epi 槽，
        # 余下全部切成 ab stage。所以 epi 槽每多留一个就少一个 ab stage —— 这正是 epi 只留单槽的
        # 直接原因。acc stage 与 SMEM 无关，它吃的是 TMEM。
        (self.num_acc_stage, self.num_ab_stage, self.num_epi_stage) = (self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
        ))

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )

        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_epi_stage,
        )

        # TMEM 分配列数 = 累加器每 stage 的 N 列数 x acc stage 数。默认 (128,256) 的 fp32 累加器
        # 每 stage 占 256 列，2 个 stage 正好 512 列 = 单 SM 的全部 TMEM 列。
        # → 【num_acc_stage 只能是 2】，再大就分不出列。这也是「epi 必须及时释放 acc stage」的压力
        #   来源：2 个 stage 里只要有一个被 epi 占着不放，MMA 就只剩一个可用。
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma,
            self.mma_tiler,
            self.num_acc_stage,
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        output_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        dispatched_weights: cute.Tensor,
        weight_output_ptrs: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_n: cutlass.Constexpr[int],
        problem_shape_k: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        problem_sizes_m: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        max_active_clusters: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        """Launch fused grouped GEMM + EP push-combine.

        :param mA: ``(total_tokens_padded, K)`` BF16.
        :param mB: ``(N, K, num_experts)`` BF16.
        :param mC: ``(total_tokens_padded, N)`` BF16 -- used only for the C
            TMA descriptor / tile coords; the kernel never writes to it.
        :param output_ptrs: Int64 ``(world_size,)`` symmetric base pointers
            for the peer combine staging buffers.
        :param token_src_rank_topk_and_indices: Int64
            ``(total_tokens_padded,)`` encoding
            ``(src_rank << 48) | (src_topk_idx << 32) | src_token_idx`` (or
            all-ones for dropped rows).
        :param problem_sizes_m: ``(G,)`` Int32 real (un-padded) per-group
            token counts; host pre-pads A on M to ``cluster_tile_m``.
        """
        # ---------------------------------------------------------------------------
        # 【参数分两类：编译期常量与运行时张量】
        # 运行时（由 TVM-FFI 动态传入）：mA / mB / mC / output_ptrs / token_src_rank_topk_and_indices /
        #   dispatched_weights / weight_output_ptrs / problem_sizes_m。
        # 编译期（cutlass.Constexpr）：group_count、problem_shape_n/k、topk、rank、world_size、
        #   total_num_clusters、max_active_clusters、has_weight。
        #
        # 【为什么 N/K 可以是 constexpr】EP 里每个 expert 的权重形状固定，只有「每个 expert 收到多少
        # token」随流量变。把 N/K 提到编译期换来三件事：
        #   1) K 的 tile 数 → mainloop 的 trip count 直接折叠，ab_pipeline 状态机没有运行时上界；
        #   2) N 的 tile 数 → 调度器只读 1D 的 problem_sizes_m，grid.z 的上界也是常量；
        #   3) A/B/C 各只需一张 tiled TMA 描述符（描述符的 tile 网格 host 期就定了）。
        # 【代价】换模型（hidden 变）就要重新 JIT；所以 ops 层把 (N,K,topk,rank,world_size) 全塞进
        # cache key（见 _ops/group_gemm_combine.py 的 variant_args）。
        #
        # 【几个只为「语义」存在的参数】
        #   mC（ops 层叫 c_scratch）：kernel 从不写它，只为拿到 C 的 layout、构造 tma_atom_c、
        #      让 partition_C 有合法坐标；真正落地的是 output_ptrs 指向的对端 buffer。
        #   rank：本文件体内没有读取，保留是为了与其它变体（如 inter 版）共用同一签名。
        #   total_num_clusters：只用于算 grid.z，见 _compute_grid。
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.c_dtype = mC.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(mC)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(world_size > self.kMaxWorldSize):
            raise ValueError(f"world_size={world_size} exceeds kMaxWorldSize="
                             f"{self.kMaxWorldSize}")

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn,
            tiled_mma.thr_id,
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        # 【单张描述符覆盖整块 A】gA 的 tiling 在下面用 local_tile 做，但描述符是「全局」的：
        # make_tiled_tma_atom_A 拿整个 mA 与 mma_tiler 生成一个 tiled atom，per-tile 只换 mode-1 的
        # tile 下标（kernel 里 tAgA[(None, global_m_mma_tile_idx, None)]）。
        # 前提是每个 expert 的 A 段都从 cluster tile 的整数倍处开始 —— 即 host 侧「padded_m 是 Cm 的
        # 倍数」这条不变量（见调度器文件）；没有它就必须为每个 expert 各建一张描述符。
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn,
            tiled_mma.thr_id,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # tx_count 的来历：每级 full barrier 预期到达的字节数 = 一级 A+B 的 SMEM 字节 x atom_thr_size。
        # 乘 atom_thr_size 是 2-CTA 指令下的约定（一对 CTA 合起来算一个 tile 的字节量）；这个数必须与
        # TMA 实际投递的字节严格一致 —— 算多了 MMA 永远等不到 full，算少了会提前放行、读到半张 tile。
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        # tma_atom_c 是 S2G 的 bulk 描述符，同样只为「让 C 的 layout 与标准 GEMM 一致」而建，
        # 本 kernel 一次都不会用它发射（push 走 row_tiled_copy + 128b store）。保留它的副作用是：
        # mC 的 shape/stride 在 JIT 期就被检查，写错会直接编译期报错而不是算错。
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mC,
            epi_smem_layout,
            self.epi_tile,
        )

        self.tile_sched_params, grid = self._compute_grid(
            total_num_clusters,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # 【SMEM 布局】一个 cute.struct 描述整块动态 SMEM：mbarrier 区 + 三块数据区。
        # 数据区各自 1024B 对齐（buffer_align_bytes）：TMA 要求 128B，epi 的 128b store 与 swizzle
        # 要求更强对齐，统一给 1024B 最省心。
        # sC 放在 sA/sB 之前不影响功能，只是让「唯一的单槽」出现在最前面，便于阅读。
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # launch：grid 来自 _compute_grid（z = 并发 cluster 数），block = 192 线程，cluster = (2,2,1)，
        # min_blocks_per_mp=1 表示每个 SM 只驻留一个 CTA。这既是 SMEM 预算决定的（sA/sB/sC 已接近
        # 满载），也是 TMEM 决定的（512 列被 2 个 acc stage 吃光，第二个 CTA 抢不到 TMEM）。
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            output_ptrs,
            token_src_rank_topk_and_indices,
            dispatched_weights,
            weight_output_ptrs,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            group_count,
            problem_shape_n,
            problem_shape_k,
            problem_sizes_m,
            topk,
            rank,
            world_size,
            has_weight,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ---------------------------------------------------------------------------
    # 【kernel 主体 = 三段互斥的 warp 角色 + 一条「隐式同步」约定】
    #   TMA warp  → 为每个 (tile, k_stage) 发 A/B 拷贝；
    #   MMA warp  → 消费 ab stage，产出 acc stage；
    #   epi warps → 消费 acc stage，把结果推到对端。
    # 【三段之间怎么同步】只有两条 mbarrier pipeline（ab 的 full/empty、acc 的 full/empty）
    # 加一次 NamedBarrier 汇合，没有别的通信。
    # 【三段怎么保证看的是同一个 tile】三个角色各持一份**完全相同**的调度器副本，用相同的确定性
    # 函数推进同一个序号序列 —— 靠「算法相同」而不是「互相通知」对齐。
    # 这是 M 连续调度的直接红利：输入相同则三个 warp 算出的 (group, m, n) 必然相同。
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mn: cute.Tensor,
        output_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        dispatched_weights: cute.Tensor,
        weight_output_ptrs: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Constexpr[int],
        problem_shape_n: cutlass.Constexpr[int],
        problem_shape_k: cutlass.Constexpr[int],
        problem_sizes_m: cute.Tensor,
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
    ):
        # make_warp_uniform：把 warp_idx 标成 warp 级均匀值，让编译器把后面的分支识别成「整 warp
        # 走同一条路径」，从而真正把不同角色分到不同 warp 上（否则每个 warp 都保留全部分支的寄存器压力）。
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # TMA 描述符预取：把描述符拉进硬件描述符缓存，避免第一次 TMA 多一次长延迟。
        # 只由 TMA warp 做，且必须早于任何 cute.copy —— 描述符是 per-CTA 的硬件资源。
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            # tma_atom_c is never used to store; kept so the JIT signature
            # matches the standalone GEMM.
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # 动态 SMEM 分配器：按 struct 声明顺序切分，返回带类型的视图。
        # sC/sA/sB 的视图要到所有 barrier 初始化之后再取（见下），因为 mbarrier 区在最前面。
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # ---------------------------------------------------------------------------
        # 【ab_pipeline（A/B 的 GMEM→SMEM 流水）= kernel 内的 per-stage 信号协议】
        # 每次 TMA 拷贝完成 → 硬件把字节数记到 full mbarrier；MMA 消费完 → 释放 empty mbarrier。
        # producer 是 TMA warp（用 tx_count 精确计字节，所以与线程数无关），consumer 是 UMMA
        # （用一条线程代表，因为 tcgen05 的完成由它自己 arrive）。
        # num_tma_producer = na + nb - 1：多播时 A/B 可能由不同 CTA 送达，每级 empty barrier 需要这些
        # 生产者各释放一次；手改成 1 会在多播配置下卡死。
        # defer_sync=True：barrier 初始化推迟，与描述符预取 / TMEM 分配重叠（见下面的 init_arrive/wait）。
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            num_tma_producer,
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # 【acc_pipeline（TMEM 的累加器流水）】producer 是 UMMA，consumer 是 4 个 epi warp；
        # 2-CTA 指令下累加器横跨两个 CTA 的 TMEM，所以消费者「线程数」= 4 warp x 2 CTA = 8 ——
        # 每个 epi warp 的 elect_one 各发一次 release，凑满 8 次这一级才空出来。
        # 【谁写 / 谁等】UMMA 的 producer_commit → acc_full（epi 等）；epi 的 consumer_release
        # → acc_empty（UMMA 等）。acquire 语义由 mbarrier 的 arrive/wait 承担，scope 是 CTA 内。
        # 【这一对 mbarrier 就是「per-tile 信号」的全部】，kernel 内没有别的跨 warp 计数。
        #
        # 【与 dispatch 侧对称的「per-expert 等待」协议，在 combine 方向并不存在】
        #   dispatch 版：dispatch warp 在「某 expert 的 A 全部落进本地 A」时 release-store 一个 int32
        #     信号（expert_signals[i] + 计数器），GEMM 的 TMA warp 用 GPU scope 的 acquire load 自旋，
        #     且只在进入新 expert 时等一次（配合 M 连续调度）。谁写 / 谁等 / scope 都写得很明确。
        #   本文件：没有任何跨 rank 信号。为什么可以没有 ——
        #     * 写的人不需要排序：push 的目标是唯一生产者的槽位（slot = token*topk + k），
        #       不存在两个 rank 抢同一行，不需要用信号定序；
        #     * 读的人在下一个 kernel：本地 C 从不落地，staging buffer 由下游 reduce kernel 在
        #       **同一个 stream 上**接着读，kernel 边界本身就给了 device 级可见性；
        #     * 唯一需要的语义是「我的 store 对另一块 GPU 可见」，由 GPU scope 的 store atom 承担
        #       （见 epilogue 的 store_atom），而不是由信号承担。
        #   → 对称性在「两端都只保留一次同步」：dispatch 压成 per-expert 的 acquire，combine 压成
        #     kernel 边界。代价是 combine 侧**没有跨 kernel 重叠**：reduce 必须等整个 GEMM 结束；
        #     要更细的重叠就得像 inter 版那样加 phase switch。
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            num_acc_consumer_threads,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # TMEM 分配器：allocate 只有 allocator_warp_id（epi warp 0）真正做，其余 warp 只 wait_for_alloc
        # 拿基址。改 warp 数时这三处（allocator_warp_id、barrier、kernel 里的分配点）必须一起动。
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        # cluster 级 barrier 初始化的两段式：arrive 在这里（relaxed），wait 放在下面所有单 CTA 初始化
        # 之后。→ 重叠了「mbarrier 在 cluster 内的可见性传播」与「shared storage 视图 / fragment 构造」，
        # 省掉一次纯等待。
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # NOTE on sC swizzle: the default ``make_smem_layout_epi``
        # picks ``K_SW128`` (``Swizzle<3,4,3>``) for this BF16 +
        # (M,N)=(128,128) epi tile. With ``num_dp=32`` forcing
        # ``CopyUniversalOp`, that yields bank conflicts for BF16.
        # A custom swizzle can reduce bank conflicts, but it is
        # not on the critical path for BF16. We therefore keep
        # the default heuristic.
        # sC 视图。紧邻的 NOTE 是上游留下的 swizzle 取舍说明：默认 K_SW128（Swizzle<3,4,3>）在 BF16 下
        # 对 16-lane 的 SMEM 读有 bank conflict，但不在关键路径上，所以保持默认。
        # 【本文件只取 1 个 epi 槽】sC 的 stage 维长度是 num_epi_stage=1，下面所有写/读都落在 slot 0。
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer,
            swizzle=epi_smem_layout_staged.inner,
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer,
            swizzle=b_smem_layout_staged.inner,
        )

        # 多播掩码：A 沿 cluster 的 N 方向广播（mcast_mode=2，同一 m 的 CTA 共享同一块 A）、
        # B 沿 M 方向广播（mcast_mode=1），同一份数据只从 GMEM 读一次，其余 CTA 从对端 SMEM 直接收。
        # 2-CTA 时还要算「对端掩码」：empty barrier 的释放必须也投到对端 CTA，否则对端永远等不到那一级。
        # 【注意】下面构造的 ab_empty_mcast_mask 在本文件里**没有被使用**（真正的释放语义在
        # PipelineTmaUmma 内部）；保留它是为了与 dispatch 版共享同一段骨架，删掉不改变行为。
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        ab_empty_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mcast_mode=2,
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mcast_mode=1,
            )
            ab_empty_mcast_mask = a_full_mcast_mask | b_full_mcast_mask
        if cutlass.const_expr(use_2cta_instrs):
            block_in_cluster_coord_vmnk_peer = (
                block_in_cluster_coord_vmnk[0] ^ 1,
                *block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk_peer,
                mcast_mode=2,
            )
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk_peer,
                mcast_mode=1,
            )
            ab_empty_mcast_mask = (a_full_mcast_mask_peer
                                   | b_full_mcast_mask_peer
                                   | cutlass.Int16(0 if ab_empty_mcast_mask is None else ab_empty_mcast_mask))

        # A/C: 2D; B: 3D with L=num_experts as a bystander dim indexed by
        # ``group_idx``.
        # A/C 是 2D（token,K / token,N），B 是 3D：(N, K, L=num_experts)。
        # local_tile 把整块张量切成 (mma_tile, tile 网格)，得到「按 tile 索引取片」的视图：
        #   tCgA (mma_tile, m_tiles, k_tiles)
        #   tCgB (mma_tile, n_tiles, k_tiles, experts)
        #   tCgC (mma_tile, m_tiles, n_tiles)
        # B 的专家维作为「旁观维」（bystander）挂在最后，于是取一个 B 片就是
        # tBgB[(None, n_tile, None, group_idx)] —— 既不需要 per-expert 描述符，也不必重算基址。
        gA_mk = cute.local_tile(
            mA_mk,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        gC_mn = cute.local_tile(
            mC_mn,
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None),
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mk)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mn)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # tBgB rank-4: (grouped_mma_tile, num_n_tiles, num_k_tiles,
        # num_experts).
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        grid_dim = cute.arch.grid_dim()

        # ---------------------------------------------------------------------------
        # 【接入 M 连续调度器】传进去的是 1D 的 problem_sizes_m 与 constexpr 的 N/K，于是调度器
        # 每专家只读 1 个 int32、不做运行时 ceil_div；group_count 是搜索的组上界（warp 一轮覆盖 32 组）。
        # 【initial_work_tile_info() 在 warp 分支之外调用】三类 warp 各算一次「我的第一个 tile 是谁」，
        # 三份独立但完全相同的计算 —— 靠确定性对齐（见下）。
        tile_sched = MContiguousGroupTileScheduler.create(
            tile_sched_params,
            bid,
            grid_dim,
            self.cluster_tile_shape_mnk,
            create_initial_search_state(),
            group_count,
            problem_sizes_m,
            problem_shape_n,
            problem_shape_k,
        )
        initial_work_tile_info = tile_sched.initial_work_tile_info()

        # Scheduler returns group-local indices in cta-tile units; the
        # mma_tiler partitions need global mma-tile units. Conversion:
        #   global_m_mma_tile = (tile_count_prev_group / ncluster_tile_n)
        #                       * cluster_to_mma_m
        #                       + cta_tile_idx_m / cta_group_size
        # cluster_to_mma_m collapses to 1 for (2cta, Cm=2) but stays
        # explicit so other cluster shapes remain correct.
        # ---------------------------------------------------------------------------
        # 【坐标换算：调度器给「组内 cta-tile 坐标」，mainloop 要「全局 mma-tile 坐标」】
        # 三个因子各自的来历：
        #   tile_count_prev_group / ncluster_tile_n
        #     = 本组之前所有组的 M 方向 cluster 数（组内 M-major，mn 总数除以 n 即 m 数），
        #       也就是本 expert 在 padded A 里的段起点，单位是 cluster tile。
        #   x cluster_to_mma_m
        #     cluster tile → mma tile。默认 (Cm=2, mma_m=256, cta_tile_m=128) 时正好为 1，
        #     但显式保留它，换 cluster 形状时才不会错。
        #   + cta_tile_idx_m / cta_group_size
        #     2-CTA 指令下两个 CTA 合算一个 mma tile，所以 cta-tile 坐标要除掉 thr 数。
        # 【代价】整段只有两次常量除法 + 一次乘法：ncluster_tile_n / cta_k_tile_cnt 都是 constexpr。
        # 【它和调度器不变量是同一件事的两面】段起点能这么算，正因为 padded_m 是 Cm 的倍数，
        # 否则真实 A 的段起点与 tile 累计数会错位。
        cta_group_size = cute.size(tiled_mma.thr_id.shape)
        cluster_to_mma_m = self.cluster_tile_shape_mnk[0] // self.mma_tiler[0]
        ncluster_tile_n = cutlass.const_expr(
            (problem_shape_n + self.cluster_tile_shape_mnk[1] - 1) // self.cluster_tile_shape_mnk[1])
        cta_k_tile_cnt_constexpr = cutlass.const_expr(
            (problem_shape_k + self.cluster_tile_shape_mnk[2] - 1) // self.cluster_tile_shape_mnk[2])

        # ===== TMA WARP =====
        # 【TMA warp：A/B 的 GMEM→SMEM 生产者】每个 tile 三件事：算 (global_m_tile, n_tile, group)
        # → 取 A/B 切片 → 逐 K stage 发拷贝。
        # 【与 dispatch 版唯一的差别】这里没有「等 expert 信号」那一步：A 要么在 dispatch kernel 里
        # 已经就位，要么由 host 保证，本 kernel 只读本 rank 本地的 A，不存在「别人还没写完」。
        # → dispatch 的 TMA warp 每次换 expert 要 GPU-scope acquire 自旋；这里不用，这也是能少一个
        #   warp 的原因之一（没有信号要维护）。
        # 【重叠】TMA warp 跑在 MMA 前面 num_ab_stage 拍：只要 empty barrier 够，它就一路往前装填。
        if warp_idx == self.tma_warp_id and initial_work_tile_info.is_valid_tile:
            work_tile = initial_work_tile_info
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_ab_stage,
            )

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                # K is constexpr in this variant, so cur_k_tile_cnt folds
                # at JIT.
                cur_k_tile_cnt = cta_k_tile_cnt_constexpr
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx

                m_cluster_tile_base = (tile_sched.search_state.tile_count_prev_group // ncluster_tile_n)
                global_m_mma_tile_idx = (m_cluster_tile_base * cluster_to_mma_m +
                                         grouped_gemm_cta_tile_info.cta_tile_idx_m // cta_group_size)
                n_tile_idx = grouped_gemm_cta_tile_info.cta_tile_idx_n

                tAgA_slice = tAgA[(None, global_m_mma_tile_idx, None)]
                tBgB_slice = tBgB[(None, n_tile_idx, None, cur_group_idx)]

                # reset_count + peek 的组合：先不带阻塞地试取下一级的 empty（producer_try_acquire），
                # 把「下一级是否空」的检查提前到上一次拷贝之后 —— 相当于手写一层软件流水，省掉每次
                # producer_acquire 里「先查后等」的往返。peek 的结果必须紧接着被下一个 producer_acquire 用掉
                # （它是「已经问过」的凭证），否则状态机错拍。
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < cur_k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state, )

                for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(
                        ab_producer_state,
                        peek_ab_empty_status,
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state, ),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state, ),
                        mcast_mask=b_full_mcast_mask,
                    )
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state, )

                # 【三个 warp 各自推进各自的调度器】这里没有「通知 epi 我算完了」这类消息：每个角色按相同
                # 顺序重复 advance_to_next_work()，序号序列天然一致。
                # 一旦有人在某个分支里跳过 advance，三者序号就错位，而错误会以「结果写到别人 tile 上」的
                # 形式出现 —— 这是本设计最脆的地方。
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_pipeline.producer_tail(ab_producer_state)

        # ===== MMA WARP =====
        # 【MMA warp：唯一发 tcgen05 指令的地方】只有 leader CTA（v==0）发 UMMA，2-CTA 指令由硬件把
        # 另一半 CTA 的 SMEM/TMEM 拉进来。
        # 每个 tile 的节奏：producer_acquire(acc stage) → 逐 K stage 的 consumer_wait/gemm/release
        # → producer_commit(acc stage)。
        # 【acc stage 的 acquire 放在 K 循环之前】等于「拿不到空累加器就不开始这个 tile」，epi 慢下来
        # 时压力会一路回传到 TMA —— 整条流水线自动限速，这是 epi 侧压力回传的唯一路径。
        if warp_idx == self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            work_tile = initial_work_tile_info
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_ab_stage,
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_acc_stage,
            )

            while work_tile.is_valid_tile:
                cur_k_tile_cnt = cta_k_tile_cnt_constexpr

                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if ab_consumer_state.count < cur_k_tile_cnt:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state, )
                    acc_pipeline.producer_acquire(acc_producer_state)

                    # 每个 tile 的第一个 kblock 用 ACCUMULATE=False（覆盖写），之后全 True（累加）。
                    # 【这就是「累积顺序确定」的落点】一个 tile 的累加严格按 K 递增，且只发生在这个 TMEM 槽内，
                    # 不跨 CTA、不跨 kernel 调用 → 同样输入必然得到同样输出。
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        ab_pipeline.consumer_wait(
                            ab_consumer_state,
                            peek_ab_full_status,
                        )
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < cur_k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state, )

                    # producer_commit：把这一级 acc stage 交给 epi，然后 advance。紧接着就进入下一个 tile 的 ab
                    # 消费 —— 「MMA 算 tile t+1」与「epi 搬 tile t」在时间上重叠，这是 kernel 内的第一条重叠轴：
                    # 计算 vs 搬运。
                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # ===== EPILOGUE WARPS =====
        # 【epilogue：combine 的「push 模式」就发生在这里】
        # 传统做法：GEMM 写本地 C → kernel 结束 → 再发一轮 all-to-all 把 C 发到该去的 rank。
        # push 模式把两步合成一步：epi 每算完一个 subtile，就按该行的路由关系（目标 rank + 目标槽位）
        # 直接把结果写进对端 staging buffer。kernel 结束时数据已经在最终会被 reduce 的位置上，
        # 中间没有本地 C 的落地，也没有第二次通信。
        #
        # 数据路径与每级 buffer 的存在理由：
        #   TMEM : 累加器，fp32，只由 UMMA 写、epi 读。双 stage 是「MMA 算下一个 / epi 搬上一个」的
        #          前提；TMEM 列数已被 2 stage 吃满，没法再深。
        #   RMEM : tcgen05.ld 的落点（128b 粒度），在这里把 fp32 转成 bf16。**寄存器持有者就是后面
        #          push 的发起者** —— 这是 push 不需要额外通信 warp 的根本原因。
        #   SMEM : 单槽，只起「重组地址」的作用 —— tcgen05.ld 的返回布局不是「一整行连续 128 个 bf16」，
        #          必须先摆成行主序，才能用 16 lane x 128b 的规整 store 打出去。
        #   peer GMEM : output_ptrs[dst_rank] + slot * N * 2B，即对端 rank 上 (num_tokens*topk, N) 的
        #          staging buffer，也是下游 reduce kernel 会读的位置。
        #
        # 【每级 buffer 的复用条件】
        #   TMEM 槽：只有 epi 发过 consumer_release 才回到 MMA 手里（见下面「提前释放」）。
        #   RMEM  ：每个 subtile 覆写一次，天然复用；代价是 fp32→bf16 必须在 push 之前完成。
        #   SMEM 槽：唯一跨角色共享的 buffer，靠「写者 = 读者 = 同一 warp 的同批 lane」把 WAR 压到
        #          warp 内 —— 于是只需要 sync_warp，不需要 CTA barrier，也不需要第二槽。
        if warp_idx < self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx

            (tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc, per_subtile_acc_shape) = (self._epilog_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base,
                tCgC,
                epi_tile,
                use_2cta_instrs,
            ))

            # subtile_cnt = cta_tile_n / epi_tile_n (constexpr). With the
            # single-slot SMEM optimisation, Phase 1 (R2S) and Phase 2
            # (cooperative push) are interleaved per-subtile so the one
            # epi-stage is freed for the next subtile immediately after
            # the push reads complete (per-warp program order guards the
            # WAR). This frees ~ ``(subtile_cnt - 1) * epi_bytes_per_stage``
            # SMEM which funds an extra ab-stage on M=128/N=256/K=2560
            # bf16, the dominant MFU-limited workload.
            # subtile_cnt = 2（cta_tile_n=256 / epi_tile_n=128）。下面的循环对每个 subtile 做
            # 「TMEM→RMEM→SMEM→push」，两段之间没有 barrier，靠单槽的顺序复用推进。
            subtile_cnt_const: cutlass.Constexpr[int] = cutlass.const_expr(self.cta_tile_shape_mnk[1] //
                                                                           cute.size(epi_tile[1]))

            # 4 epi warps x 32 lanes cover epi_tile_m=cta_tile_m=128 rows:
            # lane (warp, lane_idx) owns row ``warp*32 + lane_idx``.
            epi_tile_m = cutlass.const_expr(cute.size(epi_tile[0]))
            epi_tile_n = cutlass.const_expr(cute.size(epi_tile[1]))
            cta_tile_m = cutlass.const_expr(self.cta_tile_shape_mnk[0])
            cta_tile_n = cutlass.const_expr(self.cta_tile_shape_mnk[1])

            atom_v: cutlass.Constexpr[int] = 8  # 16 bytes / bf16 (2 bytes)
            assert epi_tile_n % atom_v == 0, (f"epi_tile_n={epi_tile_n} must be a multiple of atom_v=8 bf16")
            atoms_per_row: cutlass.Constexpr[int] = epi_tile_n // atom_v

            num_epi_warps = cutlass.const_expr(len(self.epilog_warp_id))
            cpy_m_per_warp = cutlass.const_expr(epi_tile_m // num_epi_warps)
            assert epi_tile_m == cpy_m_per_warp * num_epi_warps, ("epi_tile_m must be a multiple of num_epi_warps=4")
            assert epi_tile_m == cta_tile_m, ("v1 fused kernel assumes epi_tile_m == cta_tile_m")
            assert cpy_m_per_warp == self.num_threads_per_warp, ("lane-per-row push requires cpy_m_per_warp == 32")

            # 128b peer-GMEM SIMT atom (GPU scope, L1::no_allocate).
            # 【push 的原子：128b、GPU scope、L1 不分配】
            #   num_bits_per_copy=128 → 一条指令搬 8 个 bf16 = 16B，是单条 store 的自然宽度，也是「16 lane 拼一行 = 256B」的基本单元；
            #   memory_scope=GPU      → 目标是**另一块 GPU** 的显存（P2P 映射过来的地址）。scope 必须提到
            #                           GPU 级，写才会离开本 SM 的 L1、以 L2 为汇聚点被对端看见；
            #                           CTA 级 scope 的 store 没有这个保证，而对端没有任何 fence 机会。
            #   l1c_evict_priority=NO_ALLOCATE → 每行只写一次，进 L1 纯属污染，直接绕开。
            # 【代价】SIMT store 由 lane 逐条发射，没有 TMA「一个描述符搬一块」的压缩；换来的是完全动态的
            # 目标地址（每行的 dst_rank/slot 都不同），这恰恰是 TMA 描述符表达不了的。
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.c_dtype,
                num_bits_per_copy=128,
                memory_scope=cute.nvgpu.MemoryScope.GPU,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )

            dtype_bytes = cutlass.const_expr(self.c_dtype.width // 8)
            nbytes_per_row = cutlass.const_expr(problem_shape_n * dtype_bytes)

            weight_bytes: cutlass.Constexpr[int] = (dispatched_weights.element_type.width // 8)

            epi_lane = epi_tidx % self.num_threads_per_warp
            epi_warp_idx = epi_tidx // self.num_threads_per_warp

            # Cooperative-push layout invariants used to build the
            # 16-lane cooperative ``row_tiled_copy`` over the SMEM slot.
            # 【16 lane 一行】一整行 = epi_tile_n 个 bf16 = 16 个 128b atom；一个 warp 32 lane，
            # 于是 32/16 = 2 个「行组」，每拍并行推 2 行；一个 warp 负责 32 行（cpy_m_per_warp），
            # 要 32/2 = 16 拍（cycles_per_warp）推完。
            # 【为什么是 16 而不是 32】用 32 lane 推一行会让一半 lane 空转；16 lane 正好占满 128b 事务
            # 宽度，剩下 16 lane 去推第二行 —— 发射带宽用满。
            lanes_per_row = cutlass.const_expr(atoms_per_row)
            assert (self.num_threads_per_warp %
                    lanes_per_row == 0), "num_threads_per_warp must be a multiple of lanes_per_row"
            row_groups_per_warp = cutlass.const_expr(self.num_threads_per_warp // lanes_per_row)
            cycles_per_warp = cutlass.const_expr(cpy_m_per_warp // row_groups_per_warp)
            assert (cpy_m_per_warp == cycles_per_warp * row_groups_per_warp), ("cpy_m_per_warp must equal "
                                                                               "cycles_per_warp * row_groups_per_warp")
            group_idx = epi_lane // lanes_per_row
            lane_in_group = epi_lane % lanes_per_row

            # Rendezvous all 4 epi warps after TMEM alloc.
            self.epilog_sync_barrier.arrive_and_wait()

            work_tile = initial_work_tile_info
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_acc_stage,
            )

            # Single-slot SMEM epilogue, two-phase per work-tile:
            #
            #   For each work-tile:
            #     For s in subtile_cnt:
            #       TMEM → RMEM (cast to bf16)
            #       on the LAST subtile only: release acc-stage early
            #         so MMA can refill while we still drain the slot
            #       R2S bf16 → SMEM slot 0
            #       cooperative SIMT push slot 0 → peer GMEM
            # 这里建的是 **c_dtype 的 RMEM 张量**（另一个 fp32 的 tTR_rAcc 在 partition 里建）。
            # 它的生命周期是一个 subtile：fp32 读进 tTR_rAcc → 转 bf16 写进这个 tTR_rC → 进 SMEM，
            # 之后不再需要，所以每个 subtile 反复覆写同一批寄存器，不额外占空间。
            tTR_rC = cute.make_rmem_tensor(
                per_subtile_acc_shape,
                self.c_dtype,
            )
            tiled_copy_r2s, tRS_rC, tRS_sC = (self._epilog_smem_copy_and_partition(
                tiled_copy_t2r,
                tTR_rC,
                epi_tidx,
                sC,
            ))
            assert self.num_epi_stage == 1, ("single-slot SMEM optimisation requires "
                                             f"num_epi_stage==1; got {self.num_epi_stage}")

            row_tiled_copy = cute.make_tiled_copy_tv(
                store_atom,
                cute.make_layout(atoms_per_row),
                cute.make_layout(atom_v),
            )
            row_thr_copy = row_tiled_copy.get_slice(lane_in_group)

            # ---------------------------------------------------------------------------
            # 【每个 tile 开始前先算「这一行的结果该去哪」】三段：
            #   1) 行号 → 编码值：my_global_row 是这一行在 (total_tokens_padded, K) 里的行号，
            #      token_src_rank_topk_and_indices 逐行给出它的来历；
            #   2) 编码值 → (src_token_idx, dst_rank, src_topk_idx)；
            #   3) (token, topk) → 对端槽位 slot = src_token * topk + src_topk。
            # 【为什么在 consumer_wait 之前算】这一段是标量与指针运算，不依赖 C，正好和「等 acc stage」
            # 的停顿重叠；挪到后面就等于白等一趟。
            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result

                m_cluster_tile_base = (tile_sched.search_state.tile_count_prev_group // ncluster_tile_n)
                global_m_mma_tile_idx = (m_cluster_tile_base * cluster_to_mma_m +
                                         grouped_gemm_cta_tile_info.cta_tile_idx_m // cta_group_size)
                n_tile_idx = grouped_gemm_cta_tile_info.cta_tile_idx_n

                mma_tile_m = cutlass.const_expr(self.mma_tiler[0])
                global_m_base = (global_m_mma_tile_idx * mma_tile_m + mma_tile_coord_v * cta_tile_m)
                global_n_base = n_tile_idx * cta_tile_n

                # 行号的三段拼法：mma tile 起点 + 本 CTA 在 2-CTA 对里的那一半（v=0 管 0-127，
                # v=1 管 128-255）+ 本 warp 的 32 行 + lane。于是「lane 即行」：每条 lane 独占一行，
                # 该行的对端地址只存在它的寄存器里（push 时再用 shuffle 广播给同组 16 条 lane）。
                my_global_row = (global_m_base + epi_warp_idx * cpy_m_per_warp + epi_lane)
                # 解码技巧：编码是 (src_rank << 48) | (src_topk << 32) | src_token_idx，单个 int64。
                # decode_token_src_rank_topk_and_indices 不用移位/掩码，而是把 int64 重解释成 4 个 int16
                # 再直读高两个字（buf[2]/buf[3]）—— 省两次移位两次掩码，还顺手拿到 int16 的符号扩展：
                # 全 1 的编码因此自然解出 -1，不需要额外的「是不是哨兵」判断。
                encoded = token_src_rank_topk_and_indices[my_global_row]
                src_token_idx, my_dst_rank, src_topk_idx = (decode_token_src_rank_topk_and_indices(encoded))
                # 【push 模式的地址学】slot = 源 token 号 x topk + 源 topk 序号。这不是随便一个缓冲区下标，
                # 而是**下游 reduce kernel 会去读的那个槽位**（(num_tokens*topk, N) staging 的第 slot 行）。
                # → 每个 (token, topk) 有唯一生产者、唯一槽位：接收侧不需要 gather，也不需要「先写中间 buffer
                #   再搬一次」的第二跳。这就是「省掉一轮 all-to-all」的具体实现方式。
                my_dst_slot = (src_token_idx * cutlass.Int32(topk) + src_topk_idx)
                # ``my_dst_rank == -1`` marks a dropped slot. Use rank 0
                # as a harmless pointer-table index for invalid lanes,
                # then fold the validity bit into the final address:
                # invalid lanes end up with ``my_row_base == 0`` and the
                # consumer side gates on ``addr != 0``. This avoids an
                # out-of-bounds ``output_ptrs[-1]`` load for dropped rows.
                # dropped 行（rank=-1）的处理：不能拿 -1 去查 output_ptrs（会读到 -8 字节处），所以先落到
                # rank 0 / slot 0，再把「是否有效」折进地址：无效行的 row_base 变成 0，真正的 store 在下面用
                # `bk_row_base != 0` 挡掉。空页不会被映射，所以 0 不可能是合法地址。
                # 【代价】每次 store 多一次 64 位比较；换来的是不做 per-lane 分支（避免 warp 内发散）。
                my_pred = (my_dst_rank + cutlass.Int32(1)) != cutlass.Int32(0)
                safe_dst_rank = cutlass.Int32(0)
                safe_dst_slot = cutlass.Int32(0)
                if my_pred:
                    safe_dst_rank = my_dst_rank
                    safe_dst_slot = my_dst_slot

                my_remote_base = output_ptrs[safe_dst_rank]
                my_row_base = ((my_remote_base + cutlass.Int64(safe_dst_slot) * cutlass.Int64(nbytes_per_row)) *
                               cutlass.Int64(my_pred))

                # Weight push: one scalar per row, gated on
                # ``n_tile_idx == 0`` so multi-N-tile epilogues emit
                # exactly one store per (lane, row). Placed before
                # ``consumer_wait`` so the scalar load co-schedules
                # with the acc-stage barrier stall.
                # 【side-load 的权重推送】每个 token 有一份 router 权重，down-projection 之后要乘上去。
                # 传统做法是 combine 完再单独跑一个 kernel 乘；这里顺手把「每行一个标量」推到对端的
                # (max_m, topk) 权重 buffer，槽位号与数据槽位一致 —— 下游 reduce 拿到的就是配好对的 (值, 权重)。
                # 【只在 n_tile_idx == 0 时发】一个 token 的 N 方向有多个 tile，每行在每 tile 都会被重算一次；
                # 不加这个门就会重复写 topk 次同一份权重（幂等，但白烧带宽）。
                if cutlass.const_expr(has_weight):
                    if n_tile_idx == cutlass.Int32(0) and my_pred:
                        w_val = dispatched_weights[my_global_row]
                        w_dst_addr = (weight_output_ptrs[safe_dst_rank] +
                                      cutlass.Int64(safe_dst_slot) * cutlass.Int64(weight_bytes))
                        w_dst = cute.make_tensor(
                            cute.make_ptr(
                                dispatched_weights.element_type,
                                w_dst_addr,
                                cute.AddressSpace.gmem,
                                assumed_align=weight_bytes,
                            ),
                            cute.make_layout(1),
                        )
                        w_dst[0] = w_val

                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                # 等到「这一级 acc stage 已经算完」才允许读 TMEM。这是 epi 侧唯一的等待点，
                # 也是整条流水的第二个背压位置。
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                # 【TMEM → RMEM → SMEM 三步，缺一不可】
                #   cute.copy(t2r)：tcgen05.ld，把 fp32 累加器搬进 RMEM（一条 warp 读自己那 32 行）；
                #   fence_view_async_tmem_load()：保证 ld 的结果对后续寄存器读可见 —— TMEM 的 load 是异步的，
                #     不 fence 可能读到旧值，这是最难查的一类错；
                #   转 bf16 后 r2s 写进 SMEM 单槽。
                # 之后才是 push：读 SMEM（行主序、16 lane 拼一行）→ 写对端 GMEM。
                for subtile_idx in cutlass.range_constexpr(subtile_cnt_const):
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc_mn,
                        tTR_rAcc,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc, ).load()
                    tRS_rC.store(acc_vec.to(self.c_dtype))

                    # 【提前释放 acc stage】只在最后一个 subtile、且 TMEM 数据已 fence 之后释放：此时本 tile 的
                    # TMEM 已全部读进寄存器，后面的 push 不再需要 TMEM。
                    # → 重叠了「epi 把最后一个 subtile 推出门」与「MMA 开始下一个 tile 的 UMMA」。
                    # 去掉会怎样：release 要等 push 完，MMA 白等一整段 store 时间；在 N=256、K 较大的 MFU-bound
                    # 配置下，这段等待直接把 GEMM 利用率打下来。
                    if cutlass.const_expr(subtile_idx == subtile_cnt_const - 1):
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state, )
                        acc_consumer_state.advance()

                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, 0)],
                    )

                    # 【为什么这里只需要 sync_warp】sC 只有 1 个槽，下一个 subtile 的 r2s 会覆写它；而「写哪些行」
                    # 与「读哪些行」是同一 warp 的同一批 lane（lane 即行），跨界只发生在 warp 内
                    # → 一条 sync_warp 就够，不需要 CTA 级 barrier。这是单槽方案能成立的全部前提。
                    # 【去掉会怎样】没有它，r2s 的写可能落在 push 的读之后（编译器/硬件重排），推出去半新半旧的
                    # 一行；换成 CTA barrier 也不会错，只是把 4 个 epi warp 互相绑死。
                    cute.arch.sync_warp()
                    # cute.arch.fence_proxy(
                    #     "async.shared",
                    #     space="cta",
                    # )

                    n_sub_offset = subtile_idx * epi_tile_n
                    cur_global_n_base = global_n_base + n_sub_offset
                    n_byte_offset = (cutlass.Int64(cur_global_n_base) * cutlass.Int64(dtype_bytes))
                    sC_subtile = sC[(None, None, 0)]
                    # 【16 lane 协作推一行，一拍两行】
                    #   leader_lane 是该行的「所有者」lane —— 行基址只存在它的寄存器里，这里用 shuffle 广播给
                    #     同组另外 15 条 lane；
                    #   sC_row 是本行在 SMEM 里的那一行，gOut_row 是对端 GMEM 里本行的起点（含 subtile 的 N 偏移）；
                    #   partition_S/D 用同一份 row_tiled_copy 切 SMEM 与 GMEM，保证两边「第几段 16B」对齐。
                    # bk_row_base == 0 只出现在被 drop 的行上（见 my_row_base 的构造），直接跳过。
                    for cycle in cutlass.range_constexpr(cycles_per_warp):
                        leader_lane = (cycle * row_groups_per_warp + group_idx)
                        bk_row_base = cute.arch.shuffle_sync(my_row_base, leader_lane)
                        cta_row_in_subtile = (epi_warp_idx * cpy_m_per_warp + cycle * row_groups_per_warp + group_idx)
                        sC_row = sC_subtile[(cta_row_in_subtile, None)]
                        gOut_row = cute.make_tensor(
                            cute.make_ptr(
                                self.c_dtype,
                                bk_row_base + n_byte_offset,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(epi_tile_n),
                        )
                        tCsT_row = row_thr_copy.partition_S(sC_row)
                        tCgT_row = row_thr_copy.partition_D(gOut_row)
                        # ``bk_row_base == 0`` only happens for lanes
                        # whose source row was dropped (see comment on
                        # ``my_row_base`` -- the null page is unmapped
                        # so no real allocation can land at 0).
                        if bk_row_base != cutlass.Int64(0):
                            cute.copy(store_atom, tCsT_row, tCgT_row)

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # 收尾：先交还分配许可（同 SM 上后续 CTA 才能申请 TMEM），再由 4 个 epi warp 汇合一次，
            # 最后 free。顺序不能反 —— free 之后还有 warp 在推数据，就会踩到刚分配出去的 TMEM。
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

    # 【epi 的三个 partition 辅助函数，各管一段】
    # 这个管 TMEM→RMEM：按 epi_tile 把累加器切成 subtile（flat_divide），再用 tcgen05 的
    # tmem_copy 生成 tiled copy；per_subtile_acc_shape 就是后面 RMEM 张量的形状。
    def _epilog_tmem_copy_and_partition(self, tidx, tAcc, tCgC, epi_tile, use_2cta_instrs):
        """``tCgC`` is the partitioned output tensor. Since C is 2D (no L),
        its rank is one less than the original MKL-style example (MMA atom
        modes + num_m_tiles + num_n_tiles, without a trailing num_l_tiles).
        """
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r,
            tAcc_epi[(None, None, 0, 0, 0)],
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # 2D C: tCgC has shape ((VEC_M, VEC_N), atom_m, atom_n, num_m, num_n).
        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None)],
            epi_tile,
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_epi)
        per_subtile_acc_shape = tTR_gC[(None, None, None, 0, 0, 0, 0)].shape
        tTR_rAcc = cute.make_rmem_tensor(
            per_subtile_acc_shape,
            self.acc_dtype,
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc, per_subtile_acc_shape

    # 这个管 RMEM→SMEM：用 make_tiled_copy_D 从 t2r 的 copy 反推 r2s，保证 RMEM 的寄存器布局与
    # SMEM 的写入布局一致（否则 .retile() 会对不上）。这就是「复用 t2r」而不是另造一个的理由。
    def _epilog_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rC, tidx, sC):
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    # 这个管 SMEM→GMEM 的 TMA 版本：保留但**不使用**（tma_atom_c 从未发射）。留着它的原因：它是
    # 「传统 GEMM 写本地 C」的另一半，删掉后本文件与标准 SM100 GEMM 例子的对照关系就断了。
    # 读代码时可以直接跳过。
    def _epilog_gmem_copy_and_partition(self, tma_atom_c, tCgC, epi_tile, sC):
        """2D C: tCgC rank is ((VEC_M, VEC_N), atom_m, atom_n, num_m, num_n)."""
        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None)],
            epi_tile,
        )
        sC_for_tma = cute.group_modes(sC, 0, 2)
        gC_for_tma = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma,
            gC_for_tma,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    # ---------------------------------------------------------------------------
    # 【SMEM 账本】公式一行，但每项都有讲究：
    #   num_ab_stage = (容量/occupancy - 1KB 保留 - epi 槽) / 每个 ab stage 的字节数
    #   * 1KB 保留：mbarrier 区 + 结构体对齐；
    #   * epi 槽 = cta_tile_m x epi_tile_n x 2B = 128x128x2 = 32KB（单槽）；
    #   * ab stage = A tile + B tile 的 SMEM 字节；
    #   * 整除后向下取整，余量浪费掉 —— 所以 epi 少留一个槽往往正好换来一级 ab。
    # 【num_acc_stage=2 与 SMEM 无关】它吃 TMEM，2 是 (128,256) fp32 累加器的上限（512 列）。
    # 【num_epi_stage 被硬钉成 1】注释里的 "Hard-pinned"：单槽 + 逐 subtile 交错是这套 epilogue 的
    # 既定设计，多槽在这里没有收益（epi 的瓶颈是 store 带宽，不是槽数）。
    # 【取舍】epi 槽改成 2 → ab 少一级 → TMA 预取深度下降，mainloop 更容易被 GMEM 延迟卡住；
    # 反过来 ab 也不是越多越好，超过「覆盖延迟」所需之后只是白占 SMEM（并挤掉 acc stage 的空间）。
    @staticmethod
    def _compute_stages(tiled_mma, mma_tiler_mnk, a_dtype, b_dtype, epi_tile, c_dtype, c_layout, smem_capacity,
                        occupancy):
        num_acc_stage = 2
        # Hard-pinned to a single epi slot.
        num_epi_stage = 1

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,
        )
        epi_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(
            b_dtype, b_smem_layout_staged_one)
        epi_bytes_per_stage = cute.size_in_bytes(
            c_dtype,
            epi_smem_layout_staged_one,
        )
        epi_bytes = epi_bytes_per_stage * num_epi_stage

        # ab-stages take all remaining SMEM
        num_ab_stage = (smem_capacity // occupancy - MegaMoEGroupGEMMCombine.reserved_smem_bytes -
                        epi_bytes) // ab_bytes_per_stage

        return num_acc_stage, num_ab_stage, num_epi_stage

    # 【grid 是「开满 SM」而不是「开满 tile」】problem_shape_ntile_mnl 的 z 维塞进
    # total_num_clusters，再由 get_grid_shape(max_active_clusters) 把 grid.z 截到并发上限；
    # 真正的推进靠 kernel 内的调度循环，越界序号由调度器返回 invalid tile 收尾。
    # 【本仓调用方的一个细节】_ops/group_gemm_combine.py 把 total_num_clusters 也传成
    # max_active_clusters，于是 params 里的「总 tile 数」并不等于真实 tile 总数；能正确收尾完全
    # 依赖 delinearize_z 用 found 覆盖了基类的 is_valid。改这块时要一起改。
    @staticmethod
    def _compute_grid(total_num_clusters, cluster_shape_mn, max_active_clusters):
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl,
            (*cluster_shape_mn, 1),
        )
        grid = utils.StaticPersistentGroupTileScheduler.get_grid_shape(
            tile_sched_params,
            max_active_clusters,
        )
        return tile_sched_params, grid

    # TMEM 列数由「累加器形状 + stage 数」算出，和 _setup_attributes 里那次调用是同一件事；
    # 抽成静态方法是为了让 __call__ 与 kernel 两侧在 JIT 期拿到同一个数。
    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma, mma_tiler, num_acc_stage):
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake)
