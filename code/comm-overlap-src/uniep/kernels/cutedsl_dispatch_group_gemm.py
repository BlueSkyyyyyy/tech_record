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
# 【中文批注副本】只插入了整行注释，没有改动、删除或重排任何一行代码；
# 可以用 tools/verify_annotations.py 校验（过滤注释行后与上游逐行一致）。
#
# 上游：ByteDance-Seed/Triton-distributed @ 7908e4e
#       FlashComm/python/flash_comm/ep_overlap/kernels/cutedsl_dispatch_group_gemm.py
# 对照副本（只读）：~/github/Triton-distributed/
# 背景：UniEP（ByteDance Seed + 清华）的通算重叠算子，用 CuTe DSL 写 Blackwell
#       SM100 kernel —— 所谓「Python 写 GPU kernel」，是 Python 在编译期生成
#       device 代码，不是 Python 跑在 GPU 上。
#
# 【一句话定位】MoE 前向的第一段 MegaKernel：把 pull 式 dispatch（按 token 到各
# rank 的对端显存把 A 的行拉回来）和按专家的 GroupGEMM（FC1）压进同一次 launch。
# 通信与计算不是两个 kernel 交替，而是同一批 CTA 上两套 warp 并行推进，用
# per-expert 的信号握手交接。
#
# 【本文件的注释只讲三件事】
#   1. 角色（roles）：8 个 warp 各干什么，为什么这样切，同步域怎么错开。
#   2. 不变量（invariants）：A 的行布局 = 专家连续打包；per-expert 信号的
#      release/acquire 顺序；tile scheduler 的组号单调性（信号只等一次的依据）；
#      SMEM 预算的分配次序。
#   3. 与概念的对应：mbarrier / TMA G2S·S2G / TMEM / persistent CTA /
#      warp specialization 这些名词落在这份代码的哪几行。
# 英文注释和 docstring 保持上游原样；中文只做补充，不逐句翻译。
#
# 【建议阅读顺序】（行号＝本副本行号，点进去即可）
#   ① 行 99 起的角色总表 → 行 172 起的路由表 + 两个 NamedBarrier
#      （同步域为什么必须错开）
#   ② 行 299 起  SMEM 预算：dispatch 先扣，剩下的才给 GEMM 流水
#   ③ 行 1043 起  dispatch 两个 warp：G2S 拉 → S2G 写 → 发 per-expert 信号
#   ④ 行 1369 起  TMA warp 为什么必须在 producer_acquire 之前等 signal
#   ⑤ 行 1458 起  MMA warp（tcgen05）与累加器流水
#   ⑥ 行 1556 起  epilogue 四个 warp（TMEM → 寄存器 → SMEM → gmem）
#   ⑦ 行 1769 起  _compute_stages：SMEM 的最后一张账单
# =============================================================================
#
# 【术语速查（正文首次出现处也会再提一次）】
#   dispatch 分发 / combine 合并 —— MoE 的两个 all-to-all 阶段。本文件只做 dispatch，
#     而且是 pull 式（本 rank 主动去对端显存读），combine 在姊妹文件里。
#   TMA（Tensor Memory Accelerator）—— 异步大块搬运引擎。G2S = global→shared，
#     S2G = shared→global。dispatch 用的是不带 tensor map 的 1D bulk 版本（只给地址
#     和字节数），GEMM 用的是带 tensor map 的 tiled 版本。
#   mbarrier（memory barrier）—— 放在（共享）内存里的 8B 屏障对象，可以按字节数
#     记账（tx_count）：TMA 搬完自动 arrive 到它。本文件用它的「full / empty」两种
#     语义来管 SMEM stage 的易手。
#   pipeline stage（流水级）—— 一组 SMEM buffer + 一对 full/empty mbarrier，实现
#     「生产者写第 i 级的同时消费者读第 i-1 级」。
#   TMEM（Tensor Memory）—— SM100 上 tensor core 的累加器存储，按列分配，读写都
#     由专门的指令完成，不占寄存器堆。
#   persistent CTA（常驻 CTA）—— grid 只起「一波」，CTA 跑完一个 tile 再领下一个，
#     省掉反复启动；本文件的 dispatch 条带划分依赖这个前提。
#   warp specialization（warp 特化）—— 一个 CTA 内不同 warp 跑不同代码路径，
#     用 mbarrier 而非 __syncthreads 协调，谁慢都不会被别人卡住。
from typing import Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cuda.bindings.driver as cuda
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from .cutedsl_utils import decode_token_src_rank_topk_and_indices
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cutlass_dsl import (
    Int32, )
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    create_initial_search_state, )

from .m_contig_group_tile_scheduler import MContiguousGroupTileScheduler


# =============================================================================
# 【角色分工】8 warps / 256 threads / 2 warpgroups，没有一个 warp 是闲的。
# warp specialization 的收益全在这张表里：
#
#   warp 0-3 (WG0) ─ epilogue：TMEM → 寄存器 → SMEM → TMA store 写出 C
#   warp 4         ─ dispatch 生产者：TMA G2S（对端 gmem → 本 CTA 的 smem 暂存行）
#   warp 5         ─ dispatch 消费者：TMA S2G（smem 暂存行 → 本地 A 的专家打包行）
#                    并且是该专家的「发信号的人」
#   warp 6         ─ MMA：发 tcgen05 指令驱动 tensor core，累加器在 TMEM
#   warp 7         ─ GEMM 的 TMA：把 A/B 从 gmem 搬进 SMEM 流水 stage
#
# 数据流（→ 表示方向，每条边都由不同的 warp 驱动）：
#   对端 gmem 的 A 行 ──G2S(w4)──> smem 暂存行 ──S2G(w5)──> 本地 gmem 的 A 专家行
#   本地 A/B ──TMA(w7)──> smem stage ──tcgen05(w6)──> TMEM ──(w0-3)──> smem ──TMA──> C
#
# 【关键】dispatch 链路与 GEMM 链路之间只有一个交集：per-expert 的
# expert_signals。除此之外两套 warp 不共享任何 barrier（见下面 NamedBarrier 段），
# 这正是「通信能真正跑在计算旁边」而不是「轮流跑」的机制来源。
# =============================================================================
class _CuTeDSLDispatchGroupGemmKernelImpl:
    """Persistent M-contiguous fused dispatch + group GEMM (Blackwell SM100).

    8 warps / 256 threads / 2 WGs, no idle warps:
      * WG0 (warps 0-3): epilogue.
      * WG1 (warps 4-5): pull dispatch (TMA G2S+S2G) and per-expert
        release-store.
      * WG1 warp 6:      MMA (tcgen05).
      * WG1 warp 7:      GEMM TMA loads; acquires on the per-expert signal
        before issuing the first A/B load.

    Per-expert (N, K) are identical so the scheduler runs off a compact 1D
    ``problem_sizes_m`` and uses a single TMA descriptor over monolithic
    A/B/C.
    """

    # 三个 SMEM 相关的常量，后面整份文件的空间账都建在它们上面：
    #   reserved_smem_bytes          —— 留给 mbarrier / 对齐等说不清的小开销的固定余量；
    #   dispatch_buffer_align_bytes  —— 暂存行缓冲的对齐粒度（TMA bulk 搬运要 128B 对齐）；
    #   dispatch_max_world_size      —— 对端指针表的上限：world_size 最多 128 个 rank。
    # 【注意】dispatch 的暂存 buffer 与 GEMM 的 A/B pipeline stage 抢的是同一块 SMEM
    # （SM100 每 CTA 约 227KB），所以 dispatch 多一行 → GEMM 少一级 stage。
    # 这个取舍在 _compute_stages 里结算。
    reserved_smem_bytes = 1024
    dispatch_buffer_align_bytes = 128

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple,
        cluster_shape_mn: tuple,
        dispatch_num_stages: int = 1,
    ):
        if dispatch_num_stages < 1 or dispatch_num_stages > 5:
            raise ValueError("dispatch_num_stages must be in [1, 5]")
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        # 【2-CTA MMA】use_2cta_instrs=True 时两个 CTA 结成一对，共享一条 tcgen05 指令：
        # 只有 leader CTA（v 坐标为 0）真正发射 MMA，但两边的 SMEM/TMEM 拼成一份大 tile
        # 一起用。后果会一路扩散：M 方向的 tile 由两个 CTA 分摊、barrier 要跨 CTA
        # （cluster 级）、multicast mask 要把 peer 算进来 —— 后面出现 CtaGroup 的地方
        # 都是在处理这件事。
        self.cta_group = (tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE)

        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False
        # occupancy=1：整个 SM 只放一个 CTA。persistent kernel 的常规选择 —— 独占全部
        # SMEM/TMEM 才能开深流水；也正因为如此，SMEM 的账必须在一个 CTA 内部算清。
        self.occupancy = 1

        # 【硬编码的角色路由表】后面每个 `if warp_idx == ...` 都从这里取值，
        # 改这几个数字等于重排 warp 分工：
        #   epilog_warp_id        = (0,1,2,3)  写回 C 的 4 个 warp（正好一个 warpgroup）
        #   dispatch_warp_id_base = 4          dispatch 从 warp 4 起，共 2 个
        #   mma_warp_id = 6 / tma_warp_id = 7
        # gemm_warp_id_end = cta_num_warps = 8，含义是「全部 warp」。它出现在
        # `if warp_idx < gemm_warp_id_end` 里，8-warp 配置下恒真 —— 结果是 dispatch warp
        # 也会构造 GEMM 的 pipeline 句柄、也参与 cluster 初始化同步。改 warp 数时注意：
        # 那些 cluster 级的 arrive/wait 需要 cluster 内所有线程参与，别把某个 warp 排掉。
        self.epilog_warp_id = (0, 1, 2, 3)
        self.dispatch_warp_id_base = 4
        self.num_dispatch_warps = 2
        self.mma_warp_id = 6
        self.tma_warp_id = 7
        self.cta_num_warps = 8
        self.gemm_warp_id_end = self.cta_num_warps
        # dispatch_num_stages 一个参数决定三样东西：暂存 buffer 的行数（buffer_rows）、
        # mbarrier 的对数（bar_count = 2×stages，一 full 一 empty）、以及 SMEM 里
        # dispatch 那笔支出的大小。默认 1 表示只保证「一进一出不打架」，通信侧不追求
        # 深流水 —— 深流水留给 GEMM 那一侧，通信靠的是所有 CTA 横向并行。
        self.dispatch_num_stages = dispatch_num_stages
        self.dispatch_max_world_size = 128
        self.dispatch_buffer_rows = self.dispatch_num_stages
        self.dispatch_bar_count = 2 * self.dispatch_num_stages
        self.threads_per_cta = 32 * self.cta_num_warps

        # 【两个 NamedBarrier 的同步域是刻意错开的】这是本 kernel 能重叠的机制之一。
        #   epilog_sync_barrier：只含 4 个 epilogue warp（128 线程）。epilogue 在 SMEM 里
        #     轮转 epi_buffer，「4 个 warp 一起写」与「warp0 发 TMA store」之间要问
        #     「这块 buffer 能不能覆写」。这个问题跟通信毫无关系，所以绝不能把 dispatch
        #     warp 拉进来 —— 否则 dispatch 每写一行都要陪等一次 epilogue 的节拍。
        #   tmem_alloc_barrier：MMA warp + 4 个 epilogue warp（160 线程）。TMEM 的申请与
        #     交付是这两拨人的事，同样不含 dispatch。
        # 【代价】barrier 的线程数多写一个 warp 或少写一个 warp 会直接挂死（而不是变慢），
        # 因为 NamedBarrier 的计数是硬编码的；这也是为什么它们必须在 __init__ 里算准。
        # Named barriers must NOT include dispatch warps: epilog_sync_barrier
        # is epilogue-local; tmem_alloc_barrier joins MMA + epilogue warps.
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        # 一个 CTA 可用的 SMEM 总量（SM100 约 227KB）。_compute_stages 拿它做减法，
        # 决定 A/B 能开几级流水、epilogue 能开几级写回缓冲。
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tma_load_bytes = 0

    # 【第一笔 SMEM 支出：dispatch 的暂存区】之所以单独成函数，是因为它必须在
    # GEMM 流水定深度之前就签到账上（调用顺序见 _setup_attributes）。拆开看：
    #   1. dispatch_buffer_rows(=num_stages) 行 × K 个元素 —— 每行正好一个 token 的
    #      hidden：G2S 进来、S2G 出去。stage 数 = 「在途 token 行」的上限。
    #   2. dispatch_max_world_size × 8B —— 对端 A 基址表（最多 128 个 rank）。
    #   3. weight_ptrs_bytes —— 只有顺带做 topk 权重搬运（has_weight）时才占，
    #      且只按真实 world_size 开（节点内 ≤8 rank，64B 量级，可忽略）。
    #   4. dispatch_bar_count × 8B —— dispatch pipeline 的 full/empty mbarrier。
    #   5. 再垫一个 128B 对齐余量。
    def _dispatch_smem_bytes(self, problem_shape_k: int, has_weight: bool = False, world_size: int = 0) -> int:
        # ``dispatch_input_weight_ptrs`` is only laid out for the
        # has_weight=True cache entry and is sized to ``world_size`` (not
        # the legacy ``dispatch_max_world_size`` upper bound) -- intranode
        # workloads have <= 8 ranks so the added SMEM is at most 64 B,
        # which stays well below the boundary that would force the GEMM
        # scheduler to drop an A/B stage. The weightless build keeps a
        # byte-identical SMEM layout and the same pipeline depth.
        dtype_bytes = self.a_dtype.width // 8
        weight_ptrs_bytes = world_size * 8 if has_weight else 0
        # 【这一行就是账本】rows*K*dtype + 128 个指针 + 可选权重指针 + 2*stages 个 8B
        # mbarrier + 128B 对齐。返回值在 _setup_attributes 里被直接从 SMEM 总量中扣掉，
        # 剩下的才轮到 GEMM 的 A/B stage 和 epilogue buffer —— 扣的顺序不能反。
        return (self.dispatch_buffer_rows * problem_shape_k * dtype_bytes + self.dispatch_max_world_size * 8 +
                weight_ptrs_bytes + self.dispatch_bar_count * 8 + self.dispatch_buffer_align_bytes)

    # 【第二笔账：GEMM 的流水深度】顺序是硬性的，先扣 dispatch 再算 stage：
    #     可供 GEMM 的 SMEM = 总 SMEM - dispatch_smem_bytes
    #     num_ab_stage      = (可供 - 1KB 余量 - epi buffer) / 每级 A+B 的字节数
    #     剩余零头再换成额外的 epilogue stage（见 _compute_stages）
    # 【去掉会怎样】如果把 dispatch 这笔支出挪到后面再算，num_ab_stage 会多算一到两级，
    # 运行时 SMEM 超限，kernel 直接起不来（launch 期硬错误，不是慢一点）。
    def _setup_attributes(self, has_weight: bool = False, world_size: int = 0):
        # make_trivial_tiled_mma 造的是一个「只含一条 MMA atom」的平凡 tiled MMA；
        # 真正的 tile 形状在下面手工拼：K 方向一次吃 mma_inst_shape_k * 4 个元素，
        # 即每个 k_tile 发 4 条 tcgen05 指令（mma_inst_tile_k = 4）。
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
        # 生成器（__init__）里 mma_tiler 的 K 位填了 1 当占位，这里才换成真实值
        # 「4 条 MMA 指令的 K」。接着 cta_tile = mma_tiler / (2-CTA 的 v 维)，再乘
        # cluster_shape 得到 cluster 级 tile —— 这三层形状后面会反复出现，
        # 记住「mma tile < cta tile < cluster tile」这个包含关系。
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

        self.epi_tile = utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # 【SMEM 预算的结算点，本文件最该看懂的一段】
        #   dispatch_smem_bytes 是通信侧的常驻开销：它是常量、与 GEMM 无关，而且必须在
        #   整个 kernel 生命周期里驻留（暂存 buffer 随时要用）。所以先把它从总 SMEM 里
        #   扣掉，再让 _compute_stages 用剩余空间算 A/B 能开几级流水。
        # → 通信占得越多，GEMM 的 A/B 流水越浅。这是本设计里唯一真实的资源冲突，
        #   也是「重叠」要付的价钱：不重叠的话 dispatch 的 buffer 本可以不常驻。
        # Reserve dispatch SMEM before sizing the GEMM pipeline stages so the

        self.dispatch_smem_bytes = self._dispatch_smem_bytes(
            self.problem_shape_k,
            has_weight=bool(has_weight),
            world_size=int(world_size),
        )
        gemm_smem_capacity = self.smem_capacity - self.dispatch_smem_bytes
        (self.num_acc_stage, self.num_ab_stage, self.num_epi_stage) = (self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            gemm_smem_capacity,
            self.occupancy,
        ))

        # 三份 staged（多级）SMEM layout：A/B 是 MMA 的输入（num_ab_stage 级），
        # epi 是 epilogue 写出的中转（num_epi_stage 级）。级数由上一段算出来。
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

        # 【TMEM 的列数】SM100 的累加器按「列」分配，num_acc_stage 级累加器要多少列在
        # 这里一次算死。后面由 epilogue warp0 一次性 tmem.allocate，MMA warp 只取指针。
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma,
            self.mma_tiler,
            self.num_acc_stage,
        )

    # =============================================================================
    # 下面 __call__ 是「主机侧构建」阶段（@cute.jit）：这里的 Python 全部在编译期
    # 执行，负责产出 TMA descriptor、各种 layout、grid 形状、SharedStorage 的字段
    # 顺序，最后 .launch() 才真正下发。device 逻辑从 kernel() 开始。
    # =============================================================================
    @cute.jit
    def __call__(
        self,
        input_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        recv_token_count: cute.Tensor,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_n: cutlass.Constexpr[int],
        problem_shape_k: cutlass.Constexpr[int],
        problem_sizes_m: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
        expert_signals: cute.Tensor,
        expert_signal_counters: cute.Tensor,
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        expert_alignment: cutlass.Constexpr[int],
        input_weight_ptrs: cute.Tensor,
        output_weight: cute.Tensor,
        topk: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
    ):
        """Launch fused pull dispatch + grouped GEMM.

        ``problem_sizes_m`` carries real per-expert token counts; the host
        pads A/C rows to ``cluster_tile_m``. ``expert_signals[i]`` is
        release-stored by dispatch when all rows for expert ``i`` are visible
        in A; the GEMM TMA warp acquires on it before its first load.

        When ``has_weight`` is set, the dispatch producer warp
        (``dispatch_warp_id_base``) issues a single 4 B
        ``ld.global / st.global`` per dispatched row in lockstep with the
        token-row G2S issue. Each of the 32 lanes serves its own row in
        parallel (no ``elect_one``) so the extra traffic overlaps the
        elected lane's TMA issue and stays off the critical path.
        ``output_weight`` is a local 1D tensor (one scalar per dispatched
        row, dtype = caller's weight dtype, 4 B). ``input_weight_ptrs`` is
        a 1D ``(world_size,)`` Int64 view of peer ``(num_tokens, topk)``
        symmetric weight matrices.
        """
        # 【JIT 期才知道的东西】dtype、主序、K 都从实参张量和 constexpr 里取。
        # 注意 problem_shape_k 到这一行才赋值，而 _dispatch_smem_bytes 依赖它 ——
        # 这就是为什么 SMEM 预算不能写在 __init__ 里，只能放在 __call__ / _setup_attributes。
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.c_dtype = mC.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(mC)
        self.problem_shape_k = problem_shape_k

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self.dispatch_pipe_cnt = 0
        self._setup_attributes(
            has_weight=bool(has_weight),
            world_size=int(world_size),
        )

        # 再构造一次 tiled MMA：tiled_mma 这个对象没有存到 self 上（只存了由它推出的
        # 形状），CuTe DSL 里构造本身只是编译期的元数据拼装，没有运行时开销。
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        # atom_thr_size = 一条 MMA 指令覆盖的 CTA 数（2-CTA 模式为 2）。下面用它把
        # 一份 A+B 的字节数放大成 num_tma_load_bytes：pipeline 的 tx_count 要按这条
        # 指令涉及的所有 CTA 的搬运量记账，否则 full mbarrier 的字节数凑不满，等待永不返回。
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn,
            tiled_mma.thr_id,
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
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

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # C 的 TMA 是 S2G 方向（smem → gmem）的 CopyBulkTensorTile，与 A/B 的 G2S 相反。
        # 这里用单级 layout 描述分区（多级只是同一 layout 的轮转，索引在 epilogue 里加）。
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mC,
            epi_smem_layout,
            self.epi_tile,
        )

        # 【persistent CTA 的 grid 是「一波」而不是「所有 tile」】grid 被
        # max_active_clusters 截断：总共只起这么多 cluster，每个 CTA 常驻、循环领 tile。
        # 这件事是后面 dispatch 敢用 num_blocks 做条带划分的前提 —— 只有网格里每个 CTA
        # 都从头活到尾，「第 linear_dispatch_block 份 token」才是一个稳定的划分。
        self.tile_sched_params, grid = self._compute_grid(
            total_num_clusters,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        # SMEM 里 C/A/B 三块大 buffer 的对齐粒度（1KB）；dispatch 那块的 128B 见前面
        # 的 dispatch_buffer_align_bytes。两者都在 SharedStorage 的字段声明里生效。
        self.buffer_align_bytes = 1024

        # Two SharedStorage layouts: the weight ptr table is only
        # included for the has_weight=True artifact AND is sized to
        # ``world_size`` (not the dispatch_max_world_size upper bound)
        # so it stays well below 1 KB even at world_size=128 and -- in
        # the practical intranode case (world_size <= 8) -- adds only
        # 64 B of SMEM, well below the threshold that would force the
        # GEMM scheduler to drop an A/B stage.
        # 【SharedStorage 有两个版本】用 @cute.struct 声明，字段顺序 = SMEM 物理偏移，
        # 编译期定死。之所以写两份而不是加一个可选字段，是为了让 has_weight=False 的
        # 版本与「不带权重搬运」的原版逐字节一致：多一个字段会把后面的 sA/sB 整体挪位，
        # SMEM 预算的账也跟着变。两份只差 dispatch_input_weight_ptrs 这一行。
        if cutlass.const_expr(has_weight):

            @cute.struct
            class SharedStorage:
                dispatch_pipeline_bars: cute.struct.MemRange[cutlass.Int64, self.dispatch_bar_count]
                dispatch_input_ptrs: cute.struct.MemRange[cutlass.Int64, self.dispatch_max_world_size]
                dispatch_input_weight_ptrs: cute.struct.MemRange[cutlass.Int64, world_size]
                dispatch_tma_buffer: cute.struct.Align[
                    cute.struct.MemRange[self.a_dtype, problem_shape_k * self.dispatch_buffer_rows],
                    self.dispatch_buffer_align_bytes,
                ]
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
        else:

            @cute.struct
            class SharedStorage:
                dispatch_pipeline_bars: cute.struct.MemRange[cutlass.Int64, self.dispatch_bar_count]
                dispatch_input_ptrs: cute.struct.MemRange[cutlass.Int64, self.dispatch_max_world_size]
                dispatch_tma_buffer: cute.struct.Align[
                    cute.struct.MemRange[self.a_dtype, problem_shape_k * self.dispatch_buffer_rows],
                    self.dispatch_buffer_align_bytes,
                ]
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

        # 字段顺序就是 SMEM 的物理布局：先小对象（mbarrier、指针表），再对齐的大 buffer
        # （dispatch 暂存 → sC → sA → sB）。dispatch 暂存排在最前且 128B 对齐，因为它是
        # TMA bulk 的直接落点。
        self.shared_storage = SharedStorage

        # 把 kernel() 需要的所有具柄（TMA atom 及其张量视图、layout、scheduler 参数、
        # 信号张量）一股脑传下去。.launch 的三个参数就是硬件视角的发射配置：
        # 256 线程/CTA、cluster 形状 (*cluster_shape_mn, 1)、min_blocks_per_mp=1（独占 SM）。
        self.kernel(
            input_ptrs,
            token_src_rank_topk_and_indices,
            recv_token_count,
            mA,
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
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
            expert_signals,
            expert_signal_counters,
            rank,
            world_size,
            expert_alignment,
            input_weight_ptrs,
            output_weight,
            topk,
            has_weight,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # =============================================================================
    # device 侧：以下每一行都跑在 GPU 上。段与段之间用 ===== 分隔，段首注明 warp 角色。
    # 先给一张时序心智模型（横轴是时间）：
    #
    #   warp4/5  [专家0 拉+写][专家1 拉+写][专家2 拉+写]        ← 通信只往前走
    #   warp7        [等信号0][load A/B 0][等信号1][load A/B 1]  ← 卡在信号上
    #   warp6                  [MMA 0][MMA 1][MMA 2]
    #   warp0-3                      [写回 0][写回 1][写回 2]
    #
    # 【重叠发生在「warp4/5 已经在搬专家 i」而「warp6/7 还在算专家 i-1」的那一刻】
    # =============================================================================
    @cute.kernel
    def kernel(
        self,
        input_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        recv_token_count: cute.Tensor,
        # 【同一块 A 的两个视图】mA_raw 是原始 (M,K) 张量 —— dispatch 的 S2G 按行往它上面
        # 写；mA_mk 是 make_tiled_tma_atom_A 返回的 TMA 视图 —— GEMM 用 tensor map 读。
        # 两者指向同一块 gmem，这就是「dispatch 写完，GEMM 直接读」的物理基础。
        mA_raw: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mn: cute.Tensor,
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
        expert_signals: cute.Tensor,
        expert_signal_counters: cute.Tensor,
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        expert_alignment: cutlass.Constexpr[int],
        input_weight_ptrs: cute.Tensor,
        output_weight: cute.Tensor,
        topk: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
    ):
        # warp_idx 是 warp 级标量（make_warp_uniform 声明「整个 warp 相同」），后面所有
        # 角色分支都基于它。uniform 化能让分支在编译期收敛，避免 warp 内出现分歧执行。
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # 【TMA descriptor 预取】descriptor 是放在内存里的一段描述符，第一次 issue TMA
        # 之前要先把它拉进 cache。prefetch_descriptor 无副作用，只是预热。放在最前面且
        # 只由 warp7 做：不预取的话第一批发 A/B 要白等一次内存往返，而这段时间正好是
        # 别的 warp 在跑 prologue —— 顺手就藏掉了。
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        # 又一次从 tiled_mma 反推 use_2cta_instrs（不用构造函数传进来的同名参数），
        # 因为 device 侧的 shape 才是真值来源；这样主机/设备两侧不可能不一致。
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # 【我是谁】bid[0] % 2 得到该 CTA 在 2-CTA MMA 对里的编号 v：v==0 是 leader，
        # 只有它发 tcgen05（见 MMA warp 的 is_leader_cta）。再由 cta_rank_in_cluster 和
        # cluster_layout_vmnk 换算出 cluster 内的 (v,m,n) 坐标 —— TMA multicast mask 和
        # A/B 的分区都依赖它。
        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        # 【dispatch 的搬运粒度 = 一个 token 的一整行 hidden】nbytes_per_token = K * dtype。
        # 记住这个数：G2S/S2G 的 tx_count、mbarrier 的字节数、对端地址的步长全都建立在
        # 「一次搬一整行」上。行内连续，所以不需要 2D tensor map，最便宜的 1D bulk copy 够用。
        dtype_bytes = cutlass.const_expr(self.a_dtype.width // 8)
        nbytes_per_token = cutlass.const_expr(problem_shape_k * dtype_bytes)
        nbytes_per_dispatch_g2s = nbytes_per_token

        # 两个「裸」bulk copy atom（不带 tensor map，只给源/目的地址和字节数）：
        #   CopyBulkG2SOp：对端 gmem → 本 CTA SMEM（dispatch 第一级）
        #   CopyBulkS2GOp：本 CTA SMEM → 本地 gmem（dispatch 第二级）
        # 【为什么必须两级】TMA 只有 G2S 和 S2G 两个方向，没有 gmem→gmem；跨 rank 的
        # pull 必须在本地 SMEM 中转一次。而这个「不得不用」的中转块，正好被打造成流水
        # 线的 stage —— 代价变成了机制。
        bulk_g2s_atom = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(),
            self.a_dtype,
            num_bits_per_copy=nbytes_per_dispatch_g2s * 8,
        )
        bulk_s2g_atom = cute.make_copy_atom(
            cpasync.CopyBulkS2GOp(),
            self.a_dtype,
            num_bits_per_copy=nbytes_per_token * 8,
        )
        # 一行 token 的 1D layout（长度 K，stride 1）。可以看出 dispatch 完全不关心 2D
        # 分块：它只按「行」搬，专家与行的映射已经在前面的专家前缀和里算好了。
        dispatch_row_layout = cute.make_layout(problem_shape_k)

        # 【SMEM 落地】@cute.struct 声明的 SharedStorage 在这里实例化，然后逐个字段
        # get_tensor 拿视图。分配只做一次 —— 地址在编译期就定了，这里只是取名字。
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        dispatch_sbuf = storage.dispatch_tma_buffer.get_tensor(
            cute.make_layout(
                (
                    self.dispatch_buffer_rows,
                    problem_shape_k,
                ),
                stride=(problem_shape_k, 1),
            ))
        dispatch_bar_ptr = storage.dispatch_pipeline_bars.data_ptr()
        dispatch_input_ptrs = storage.dispatch_input_ptrs.get_tensor(cute.make_layout(self.dispatch_max_world_size))
        # ``dispatch_input_weight_ptrs`` is only laid out in the
        # has_weight SharedStorage variant; the no-weight build doesn't
        # carry it so the SMEM layout matches the legacy kernel exactly.
        # Sized to ``world_size`` (the smallest constexpr that fits all
        # peers) so the SMEM cost stays in the noise.
        # 只有 has_weight 的 SharedStorage 版本才有这个字段（见上面两份声明）。
        # 用 const_expr 让编译期直接抹掉另一个分支，不留运行时判断，也不留死代码。
        if cutlass.const_expr(has_weight):
            dispatch_input_weight_ptrs = (storage.dispatch_input_weight_ptrs.get_tensor(cute.make_layout(world_size)))
        ab_full_mbar_ptr = storage.ab_full_mbar_ptr.data_ptr()
        acc_full_mbar_ptr = storage.acc_full_mbar_ptr.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf
        tmem_dealloc_mbar = storage.tmem_dealloc_mbar
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

        # ===== dispatch 的 pipeline（暂存行的 full / empty 握手）=====
        # 这条 pipeline 的生产者与消费者各只算 1 个线程（size=1），实际干活的是
        # warp4 的 elect_one 和 warp5 的 elect_one。于是通用的 PipelineTmaAsync 能被
        # 一个两-warp 流水复用：arrive 计数按 1 记，谁 acquire/commit 谁就是那个「1」。
        # tx_count 是一行的字节数，G2S 的 TMA 完成时按字节把它打满。
        # 【代价】num_stages=1 时同一时刻只有一行在途：G2S 必须等 S2G 把上一行读走才能
        # 发下一行。通信侧流水很浅，靠的是 num_blocks 个 CTA 的横向并行来补。
        dispatch_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.dispatch_num_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                size=1,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                size=1,
            ),
            tx_count=nbytes_per_dispatch_g2s,
            barrier_storage=dispatch_bar_ptr,
            tidx=lane_idx,
        )
        # 【对端指针表上 SMEM】把 world_size 个「对端 A 基址」从 gmem 抄进 SMEM，由前
        # world_size 个线程各抄一个（连续地址，coalesced）。之后每搬一行都要查这张表，
        # 查 SMEM 比查 gmem 便宜一个数量级。→ 表只有一份，全 CTA 共享。
        if tidx < cutlass.Int32(world_size):
            dispatch_input_ptrs[tidx] = input_ptrs[tidx]
            if cutlass.const_expr(has_weight):
                dispatch_input_weight_ptrs[tidx] = input_weight_ptrs[tidx]
        # 【全 CTA 唯一一次 __syncthreads】等指针表写完。这一行之后，所有 warp 就按角色
        # 分叉、再也不会有全体同步了 —— 它是整份文件同步语义的分水岭。
        # 这也是「慢 warp 拖所有人」的唯一位置；此时 dispatch 还没开始，无伤。
        cute.arch.barrier()
        dispatch_producer = dispatch_pipeline.make_producer()
        dispatch_consumer_read = dispatch_pipeline.make_consumer()
        dispatch_consumer_release = dispatch_consumer_read.clone()

        # 8-warp 配置下这个条件恒真（warp_idx ∈ [0,7] < 8），所以整块对所有人执行：
        # 每个 warp 都构造自己的 pipeline 句柄（指向同一块 mbarrier 存储），dispatch
        # warp 也不例外。改 warp 数时小心：下面的 pipeline_init_arrive/wait 是 cluster 级
        # 同步，需要 cluster 内所有线程参与，别把某个 warp 排在外面。
        # 【GEMM 的两条流水，方向相反】
        #   ab_pipeline（PipelineTmaUmma）：A/B 的 full/empty。生产者=warp7(TMA)，
        #       消费者=MMA。num_ab_stage 级 —— 深流水在这里。
        #   acc_pipeline（PipelineUmmaAsync）：累加器 tile 的 full/empty。生产者=MMA，
        #       消费者=4 个 epilogue warp。num_acc_stage=2 就够：MMA 算下一个 tile 时
        #       epilogue 还在写上一个。
        # defer_sync=True 表示初始化同步不由 pipeline 自己做，而由下面显式调用
        # pipeline_init_arrive/wait —— 这样能把 cluster 同步拆成两半，中间塞满活干。
        if warp_idx < self.gemm_warp_id_end:
            ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            # 这里的变量名值得停一下：num_tma_producer 才是 consumer_group 的 size ——
            # 它表示「有几个 TMA 生产者要往 empty barrier 上记账」（1-CTA 且无 multicast 时
            # 为 1）。empty 的到达数由生产者数量决定，不是由 MMA 的线程数决定。
            num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
            ab_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                num_tma_producer,
            )

            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=ab_full_mbar_ptr,
                num_stages=self.num_ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

            acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            # acc 的消费者按线程数记账：4 个 epilogue warp（2-CTA 时乘 2，因为一对 CTA 各有
            # 一份 epilogue）。记住这个数：后面 release 时每个 warp 只由 elect_one 发一次
            # arrive，4 次到达正好凑满它。
            num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
            acc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                num_acc_consumer_threads,
            )
            acc_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=acc_full_mbar_ptr,
                num_stages=self.num_acc_stage,
                producer_group=acc_pipeline_producer_group,
                consumer_group=acc_pipeline_consumer_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

            # TMEM 的分配者是 epilogue 的 warp0（allocator_warp_id = epilog_warp_id[0]）：
            # 谁消费谁申请。其它 warp（MMA）通过 barrier_for_retrieve 等它交付指针。
            tmem = utils.TmemAllocator(
                tmem_holding_buf,
                barrier_for_retrieve=self.tmem_alloc_barrier,
                allocator_warp_id=self.epilog_warp_id[0],
                is_two_cta=use_2cta_instrs,
                two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar,
            )

            # 【mbarrier 初始化的 fence + cluster 级 arrive】mbarrier_init_fence 保证 barrier
            # 的初始化对所有线程可见；cluster arrive 让同一个 cluster 的 CTA 互相知道
            # 「barrier 已经初始化完」。它必须早于任何 dispatch/GEMM 主循环里的 barrier wait，
            # 否则可能等到一个还没初始化的 barrier（表现为随机挂死）。
            # is_relaxed=True：只需要 barrier 初始化本身可见，没有数据依赖要排序，省一次强 fence。
            if warp_idx < self.gemm_warp_id_end:
                # mbarrier_init_fence + cluster arrive; must run before any
                # dispatch/GEMM main-loop barrier wait.
                pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

            # 【multicast mask：同一 cluster 里一份 A/B 只从 gmem 读一次，广播给多个 CTA】
            # 2-CTA MMA 时 A 的一份 tile 要同时出现在两个 CTA 的 SMEM（各喂同一条 MMA 指令的
            # 一半），B 同理。create_tma_multicast_mask 生成「这次 TMA 打给 cluster 里哪几个
            # CTA」的位图 → 省下一半 L2/DRAM 流量。
            # 【empty mask 是另一回事】ab_empty_mcast_mask 用于「释放 stage 时同时通知 cluster
            # 里的其它 CTA」（它们那份同样的 stage 也空了）。
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
            # 2-CTA 专用：peer 的 v 坐标 = v ^ 1（v 只有 1 bit，异或 1 就是同一条 MMA 指令的
            # 另一个 CTA）。它的 A/B mask 要或进 ab_empty_mcast_mask，否则对端那半 buffer
            # 永远等不到释放 → 死锁。
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

            # 【三张全局张量，三种 rank】
            #   A: (M,K) 单块 2D —— 所有专家的行按专家顺序拼成一条 M，专家边界只体现为 tile
            #      切分的起点，于是 GEMM 只需要一个 A descriptor；
            #   B: (N,K,L) 3D —— L = 专家数是「旁路维」，TMA 用 group_idx 直接索引，同样一个
            #      descriptor 覆盖所有专家；
            #   C: (M,N) 单块 2D。
            # 【前提】各专家的 (N,K) 必须完全相同（见文件头 docstring），否则不敢把 A/C 拍平成
            # 一个 M。local_tile 把三者切成 tile 网格，供后面的 tma_partition 使用。
            # A/C: monolithic 2D; B: 3D with L=num_experts as a bystander dim
            # indexed by ``group_idx``.
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

            # partition_A/B/C 按 MMA atom 的线程把全局 tile 网格切片；A/B 的切片随后交给
            # cpasync.tma_partition 变成「每级 stage 一个 TMA 落点」，C 的切片留给 epilogue。
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
            # tBgB 是 rank-4：(mma_tile, n_tile, k_tile, group)。多出来的 group 维就是
            # 「一个 descriptor 覆盖所有专家」的落点 —— 取切片时用 cur_group_idx 索引。
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

            # tCrA/tCrB 是「指向 SMEM stage 的 fragment」，不是数据拷贝：tcgen05 直接从 SMEM
            # 读 A/B，这里只声明每级 stage 的形状。tCtAcc_fake 是 TMEM 累加器的假 fragment，
            # 用来算 layout 和列数；真正的指针要等 epilogue allocate 之后才拿到。
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

            # 【init_arrive 与 init_wait 之间隔了约 80 行，这不是随手放的】中间这段
            # （local_tile / partition / tma_partition / fragment 构造）全是编译期的地址与
            # 布局计算，不需要等 barrier ready。把 cluster 同步的等待期用这些活填满，
            # 是「开局不空转」的一处小设计。
            if warp_idx < self.gemm_warp_id_end:
                pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            grid_dim = cute.arch.grid_dim()

            # 【M-contiguous tile scheduler（按 M 连续的分组调度器）】为什么不用 CUTLASS 自带
            # 的 GroupedGemm 调度器？因为这个变体里 per-expert 只有 M 是运行时变量，(N,K) 全同：
            #   * 每个专家的题目规模是一次 Int32 读（problem_sizes_m[g]），不是 vec-4；
            #   * N、K 折成 constexpr，省掉运行时的 ceil_div；
            #   * search_state 里留着 tile_count_prev_group，用来把「组内线性 tile 下标」
            #     还原成「全局 m 偏移」。
            # 【关键性质：组号单调不减】persistent CTA 按线性 tile 下标递增领活，而每个专家在
            # 全局 tile 序列里占一段连续区间 → 同一个 CTA 看到的 group_idx 只增不减。
            # 后面 TMA warp 的 last_waited_group 敢「每个专家只等一次」，靠的就是这条性质。
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
            # 每个 warp 角色各自持有一份 scheduler 状态（同一 CTA 内 4 套并行推进），它们必须
            # 走出完全相同的 tile 序列 —— 这是这套 warp specialization 的隐含约定：
            # 没有任何代码强制它，写错任何一处 advance 就会让角色错位。
            initial_work_tile_info = tile_sched.initial_work_tile_info()

            # 【组内下标 → 全局 m 下标】这一小段是「A/C 单块 2D」布局要付的代价：
            #   global_m_mma_tile = 该专家之前的 cluster-m-tile 数 * cluster_to_mma_m
            #                       + 组内 cta_tile_idx_m / cta_group_size
            # 其中「该专家之前的 cluster-m-tile 数」= tile_count_prev_group / ncluster_tile_n
            # （N 方向 tile 数恒定，所以把线性下标除回去就得到 m 方向的个数）。
            # 【成立条件】host 侧保证每个专家的 A slab 补齐到 cluster_tile_m 的整数倍
            # （expert_alignment 必须是 cluster_tile_m 的倍数，context.py 里有检查）。
            # 否则「前面专家的 tile 数」与实际行偏移对不上，取到的 A 就整块错位 ——
            # 这是 host 参数与 device 地址换算之间的一条隐式契约。
            # Scheduler returns group-local indices in cta-tile units; the
            # mma_tiler partitions need global mma-tile units. Conversion:
            #   global_m_mma_tile = (tile_count_prev_group / ncluster_tile_n)
            #                       * cluster_to_mma_m
            #                       + cta_tile_idx_m / cta_group_size
            # cluster_to_mma_m collapses to 1 for (2cta, Cm=2) but stays
            # explicit so other cluster shapes remain correct.
            cta_group_size = cute.size(tiled_mma.thr_id.shape)
            cluster_to_mma_m = self.cluster_tile_shape_mnk[0] // self.mma_tiler[0]
            ncluster_tile_n = cutlass.const_expr(
                (problem_shape_n + self.cluster_tile_shape_mnk[1] - 1) // self.cluster_tile_shape_mnk[1])
            cta_k_tile_cnt_constexpr = cutlass.const_expr(
                (problem_shape_k + self.cluster_tile_shape_mnk[2] - 1) // self.cluster_tile_shape_mnk[2])

            # =============================================================================
            # DISPATCH（warp 4/5）—— 把「本 CTA 负责的那些 token」从各 rank 的对端拉回来，
            # 落到本地 A 的专家打包行上，然后替这个专家发一个 release 信号。
            #
            # 【分工：按行条带（stripe）切给所有 CTA】
            #   专家 g 的行区间 = [expert_start, expert_start + expert_tokens)
            #   第 nb 个 CTA 只碰满足 (row - expert_start) % num_blocks == linear_dispatch_block
            #   的行；warp 内 32 个 lane 再按 stride num_blocks 各认一路。
            #   → 一个专家的行被 num_blocks 个 CTA 同时搬，谁搬完谁记账（见 expert_signals）。
            #
            #   warp4：取元数据 → 逐 lane 发 TMA G2S（对端 gmem → smem 暂存行）
            #   warp5：等 full → 发 TMA S2G（smem 暂存行 → 本地 A 第 row 行）→ 释放 stage
            #   两个 warp 都跑完这个专家后，warp5 的 elect_one 给该专家记账/发信号。
            # 【为什么拆两个 warp】G2S 之后必须等 TMA 完成才能 S2G；一个 warp 自己等自己，
            # 发射节奏会被气泡打断。拆开后 warp4 可以继续给下一行发 G2S，warp5 在等上一行
            # 的完成 —— 通信内部也有一层流水。
            # =============================================================================
            if (warp_idx >= self.dispatch_warp_id_base
                    and warp_idx < (self.dispatch_warp_id_base + self.num_dispatch_warps)):
                dispatch_grid_dim = cute.arch.grid_dim()
                dispatch_warp_rank = warp_idx - self.dispatch_warp_id_base
                # Persistent-GEMM grid is 3D (cluster_m, cluster_n, clusters);
                # flatten so each CTA owns a unique dispatch shard.
                # 【num_blocks = 整个 grid 的 CTA 总数】它是三样东西的公共分母：
                #   1. 行条带的宽度（每个 CTA 每隔 num_blocks 行拿一行）；
                #   2. 每轮 32 lane 之后的推进步长（batch_base += num_blocks * 32）；
                #   3. 后面「这个专家是否全落地」的计数阈值（expert_signal_counters 那段）。
                # grid 是 3D（cluster_m, cluster_n, clusters），所以先把 bid 拍平成一个唯一 id。
                num_blocks = cute.size(dispatch_grid_dim)
                linear_dispatch_block = (bid[2] * dispatch_grid_dim[0] * dispatch_grid_dim[1] +
                                         bid[1] * dispatch_grid_dim[0] + bid[0])
                pipe_cnt_rt = cutlass.Int32(self.dispatch_pipe_cnt)
                warp_batch = cutlass.Int32(32)
                # 这里读了一次「本 rank 收到的 token 总数」，但它没有被任何计算使用（下一行只是
                # 加上 world_size 再丢掉）。【结论】不影响功能：每个专家的行数完全由
                # problem_sizes_m 决定，行区间靠专家前缀和推出来。可以当作遗留的读/占位。
                num_recv_tokens = recv_token_count[rank]
                _ = num_recv_tokens + cutlass.Int32(world_size)

                # ---------------- warp 4：dispatch 生产者（TMA G2S） ----------------
                # 外层按专家走，内层按「一轮 32 个 token」走。整个循环里没有一次全局同步：
                # 专家与专家之间只靠 expert_signals 往下传。
                if dispatch_warp_rank == cutlass.Int32(0):

                    weight_bytes: cutlass.Constexpr[int] = (output_weight.element_type.width // 8)
                    expert_start = cutlass.Int32(0)
                    expert_idx = cutlass.Int32(0)
                    # 每个专家的行区间 = [expert_start, expert_end)。注意两者用的不是同一个数：
                    #   expert_end   = expert_start + expert_tokens   ← 真实行数
                    #   expert_start += aligned_tokens                ← 对齐后的行数
                    # → A 里每个专家占一块对齐的 slab，slab 尾部是对齐空洞（不被 dispatch 写，
                    #   但落在 GEMM 的 M 范围内，host 侧已按 padding 处理）。
                    while expert_idx < group_count:
                        expert_tokens = problem_sizes_m[expert_idx]
                        expert_end = expert_start + expert_tokens
                        batch_base = expert_start + linear_dispatch_block

                        while batch_base < expert_end:
                            lane_token_idx = batch_base + lane_idx * num_blocks
                            # 每个 lane 先给自己那一行取元数据（一个 Int64，32 个 lane 地址连续 = 一次 coalesced
                            # 256B load），编码是 (rank<<48)|(topk<<32)|token_idx，由
                            # cutedsl_utils.decode_token_src_rank_topk_and_indices 拆开。
                            # 下面再用 shuffle_sync 把任意 lane 的结果广播到全 warp —— 这样只由 elect_one 的
                            # 一个线程发 TMA，却能拿到 32 个 lane 各自的 (源 rank, 源 token 行)。
                            encoded = cutlass.Int64(-1)
                            if lane_token_idx < expert_end:
                                encoded = token_src_rank_topk_and_indices[lane_token_idx]
                            src_token_idx, src_rank, topk_idx = (decode_token_src_rank_topk_and_indices(encoded))

                            # 【顺手做的事：topk 权重的 side-load】dispatch 本来就知道每行的
                            # (src_rank, src_token, topk_idx)，于是 32 个 lane 各用一条 ld.global/st.global 把
                            # 对应权重从对端的权重矩阵搬进本地 output_weight[row]。
                            # 关键在「不用 elect_one」：32 个 lane 并行搬，和 elected lane 发 TMA 走的是不同的
                            # 执行单元，于是这点额外流量压在 TMA 发射的阴影里，不上关键路径。
                            # 代价：这 4B 的读写不在 TMA 的屏障保护之下，它的可见性要靠后面 per-expert 的
                            # release 原子操作来序化（release 会把之前的普通写一起排出去）。
                            # Side-load: each of the 32 lanes pulls its
                            # own (token, topk_idx) weight in parallel
                            # from the source rank's symmetric weight
                            # buffer into the local ``output_weight``
                            # row.
                            if cutlass.const_expr(has_weight):
                                if (lane_token_idx < expert_end and (src_rank + cutlass.Int32(1)) != cutlass.Int32(0)):
                                    w_src_addr = (dispatch_input_weight_ptrs[src_rank] +
                                                  cutlass.Int64(src_token_idx * cutlass.Int32(topk) + topk_idx) *
                                                  cutlass.Int64(weight_bytes))
                                    w_src_tensor = cute.make_tensor(
                                        cute.make_ptr(
                                            output_weight.element_type,
                                            w_src_addr,
                                            cute.AddressSpace.gmem,
                                            assumed_align=weight_bytes,
                                        ),
                                        cute.make_layout(1),
                                    )
                                    output_weight[lane_token_idx] = (w_src_tensor[0])

                            # 【本段最绕的地方：谁发 TMA】循环 32 次，每次处理「lane 那一行」，但真正发 TMA
                            # 的永远是被 elect_one 选中的那一个线程（TMA issue 必须单线程；32 个 lane 各发一次
                            # 会重复 32 倍）。所以先用 shuffle_sync 把第 lane 个 lane 解出的 (src_rank,
                            # src_token) 广播到全 warp，再由 elected 线程算地址、发 copy。
                            # → 一次 32-lane 的元数据 load，换来 32 次单线程 TMA 发射。
                            for lane in cutlass.range(32, unroll_full=True):
                                b_token_idx = (batch_base + cutlass.Int32(lane) * num_blocks)
                                b_src_token_idx = cute.arch.shuffle_sync(src_token_idx, lane)
                                b_src_rank = cute.arch.shuffle_sync(src_rank, lane)

                                if b_token_idx < expert_end:
                                    # acquire_and_advance blocks on the empty
                                    # mbarrier; the G2S TMA arrives the matching
                                    # full transaction barrier on completion.
                                    # 【流水线的空位检查】acquire_and_advance 阻塞在 empty mbarrier 上：只有 warp5
                                    # 把用过的 stage 释放了，这里才能拿到新的。num_stages=1 时，这个调用就是通信侧
                                    # 的实际节拍器 —— 它保证不会写坏正在被 S2G 读的那一行。
                                    # 配套的 full barrier 由 G2S 的 TMA 搬完时按字节自动打满（tx_count 记账），
                                    # 所以 copy 之后不需要任何显式 arrive。
                                    handle = (dispatch_producer.acquire_and_advance())
                                    # ``b_src_rank == -1`` should be excluded
                                    # by ``problem_sizes_m`` for valid rows,
                                    # but keep the ptr-table load in-bounds if
                                    # bad metadata reaches this path.
                                    # 【防御式判断】src_rank 编出来是 -1 表示「无效行」（元数据里的哨兵）。正常情况下
                                    # problem_sizes_m 已经把无效行排除在外、不会走到搬运；但指针表的下标不能为负，
                                    # 所以这里把 -1 夹到 0 号 rank，保证不越界读 SMEM。
                                    # 代价是两条 select，收益是脏元数据不会把指针表读飞。
                                    safe_b_src_rank = cutlass.Int32(0)
                                    safe_b_src_token_idx = cutlass.Int32(0)
                                    if (b_src_rank + cutlass.Int32(1)) != cutlass.Int32(0):
                                        safe_b_src_rank = b_src_rank
                                        safe_b_src_token_idx = b_src_token_idx
                                    remote_base_i64 = dispatch_input_ptrs[safe_b_src_rank]
                                    remote_row_i64 = remote_base_i64 + (cutlass.Int64(safe_b_src_token_idx) *
                                                                        cutlass.Int64(nbytes_per_token))
                                    # 单线程发一次 1D bulk TMA：远端行地址 → 暂存行 sDst（handle.index 决定第几级
                                    # stage）。注意 gSrc 的地址是现算的：对端 A 基址 + 源 token 行 * 行字节数 ——
                                    # 不需要 2D descriptor，也不需要知道对端张量的布局。
                                    with cute.arch.elect_one():
                                        sDst = cute.make_tensor(
                                            dispatch_sbuf.iterator + handle.index * problem_shape_k,
                                            dispatch_row_layout,
                                        )
                                        gSrc = cute.make_tensor(
                                            cute.make_ptr(
                                                self.a_dtype,
                                                remote_row_i64,
                                                cute.AddressSpace.gmem,
                                                assumed_align=16,
                                            ),
                                            dispatch_row_layout,
                                        )
                                        cute.copy(
                                            bulk_g2s_atom,
                                            gSrc,
                                            sDst,
                                            mbar_ptr=handle.barrier,
                                        )
                                    # commit：把「这次 G2S 的字节数已经挂到 full barrier 上」记账，warp5 的
                                    # wait_and_advance 才可能被放行。它必须在 cute.copy 之后 —— copy 是发起，
                                    # commit 才是「我这一轮提交完了」。
                                    handle.commit()
                            batch_base += num_blocks * warp_batch

                    # 专家收尾：expert_start 按对齐后的行数推进（A 里每个专家 slab 的物理步长）。
                    # 【注意这里没有任何同步】—— 同步在 warp5 那边，因为信号的语义是「A 里这个专家的
                    # 所有行都写完了」，只有负责写的那一方有资格宣布。
                        aligned_tokens = expert_tokens
                        if cutlass.const_expr(expert_alignment > 1):
                            aligned_tokens = (
                                (expert_tokens + expert_alignment - 1) // expert_alignment) * expert_alignment
                        expert_start += aligned_tokens
                        expert_idx += cutlass.Int32(1)

                # ---------------- warp 5：dispatch 消费者（TMA S2G）+ 发信号 ----------------
                # 它不需要元数据，只需要 (expert_start, 行号)：写入目标是本地第 b_token_idx 行，
                # 源就是 warp4 填好的第 handle.index 级 stage。
                # 【它才是「谁写 A」的那一方】所以 per-expert 的 release 信号必须由它发。
                elif dispatch_warp_rank == cutlass.Int32(1):
                    expert_start = cutlass.Int32(0)
                    expert_idx = cutlass.Int32(0)
                    while expert_idx < group_count:
                        cnt = cutlass.Int32(0)
                        expert_tokens = problem_sizes_m[expert_idx]
                        expert_end = expert_start + expert_tokens
                        batch_base = expert_start + linear_dispatch_block

                        while batch_base < expert_end:
                            for lane in cutlass.range(32, unroll_full=True):
                                b_token_idx = (batch_base + cutlass.Int32(lane) * num_blocks)

                                if b_token_idx < expert_end:
                                    # wait_and_advance blocks on the full mbarrier
                                    # set by the G2S TMA; the cloned consumer
                                    # handle releases the stage once S2G is done.
                                    # 阻塞在 full mbarrier 上：等 warp4 那次 G2S 的字节数全部到账。拿到 handle 就立刻
                                    # 发 S2G，把这一行搬到本地 A 的目标行；随后（同一个 elect_one 里）commit 并等
                                    # read 完成。注意 wait 用了 read=True：我们只需要「SMEM 那行已经被读走」，
                                    # 不需要等 gmem 那边的写落地 —— 这样 stage 能更早释放。
                                    handle = (dispatch_consumer_read.wait_and_advance())
                                    with cute.arch.elect_one():
                                        sSrc = cute.make_tensor(
                                            dispatch_sbuf.iterator + handle.index * problem_shape_k,
                                            dispatch_row_layout,
                                        )
                                        gDst = cute.make_tensor(
                                            cute.domain_offset(
                                                (b_token_idx, cutlass.Int32(0)),
                                                mA_raw,
                                            ).iterator,
                                            dispatch_row_layout,
                                        )
                                    # S2G：暂存行 → mA_raw 的第 b_token_idx 行（domain_offset 直接给出目标行首地址）。
                                    # 这就是「dispatch 写的就是 GEMM 的 A」这句话在代码里的样子：没有任何中间重排、
                                    # 没有第二个 kernel，A 的行号就是专家打包后的行号。
                                        cute.copy(bulk_s2g_atom, sSrc, gDst)
                                        cute.arch.cp_async_bulk_commit_group()
                                        cute.arch.cp_async_bulk_wait_group(
                                            self.dispatch_pipe_cnt,
                                            read=True,
                                        )
                                    # 释放 stage。dispatch_pipe_cnt 是构造时固定的「允许在途的组数」，本版本恒为 0：
                                    # 0 意味着上面的 wait 已经等到「全部读完」，于是这里每次都立刻释放。
                                    # 若把它调大，就变成延迟释放（攒几个再还）；下面那段收尾循环正是为那种配置准备的，
                                    # pipe_cnt=0 时它是空循环（range(0)）。
                                    if cnt >= pipe_cnt_rt:
                                        dispatch_consumer_release.release()
                                        dispatch_consumer_release.advance()

                                    cnt += cutlass.Int32(1)
                            batch_base += num_blocks * warp_batch

                        # 【专家级收尾：等 gmem 写落地】read=False 是「等写完成」而不是「等读完成」。
                        # 这一步是发信号之前必须做的：信号一旦置位，别的 CTA 的 GEMM 就会来读这些 A 行，
                        # 所以必须等本 CTA 最后几笔 S2G 真正写进了 gmem。
                        with cute.arch.elect_one():
                            cute.arch.cp_async_bulk_wait_group(0, read=False)

                        for _i in cutlass.range(self.dispatch_pipe_cnt, unroll_full=True):
                            if cnt > cutlass.Int32(0):
                                dispatch_consumer_release.release()
                                dispatch_consumer_release.advance()
                                cnt -= cutlass.Int32(1)

                        # 【per-expert 的 release/acquire 协议 —— 全文最关键的同步】
                        #   每条 A 行由某个 CTA 写入；一个专家的行被 num_blocks 个 CTA 分掉，所以
                        #   「专家 g 的 A 全部就绪」⇔「所有 CTA 都完成了自己那一份」。
                        #   做法：每个 CTA 完工后对 expert_signal_counters[g] 做一次 release 的 atomic_add；
                        #   谁的 add 返回 num_blocks-1（即我是最后一个）谁就把 expert_signals[g] 置 1。
                        #   → 一次加法 + 一次存储，替代了「每个 CTA 与所有人握手」的 O(n²) 同步。
                        # 【release / acquire 各管什么】
                        #   release 的 add  ：本 CTA 之前的 S2G 写在其它 CTA 看到 counter 之前可见；
                        #   release 的 store：把 counter 的可见性传递到 signal 上（链式 release）；
                        #   GEMM 侧的 acquire load：读到 signal==1 之后，那些 A 行的写一定可见。
                        # 【代价】粒度是「整个专家」：只要有一个 CTA 慢，所有想算这个专家的 cluster 都得等。
                        # 换来的是极便宜的同步 —— 没有 mbarrier、没有逐 tile 握手。
                        with cute.arch.elect_one():
                            # Per-expert readiness: each CTA bumps the counter;
                            # the last one release-stores the signal. Orders
                            # memory for GEMM acquire, no CTA-wide sync.
                            old = cute.arch.atomic_add(
                                (expert_signal_counters.iterator + expert_idx).llvm_ptr,
                                cutlass.Int32(1),
                                sem="release",
                                scope="gpu",
                            )
                            # 只有凑满 num_blocks 个 CTA 才置位 —— 这就是「所有 CTA 的该专家分片都落地」的
                            # 判定，而 num_blocks 与行条带的宽度是同一个数，阈值天然对齐。
                            # 【注意两个量都单调】counter 只增；signal 只会 0→1。所以 GEMM 侧可以放心用
                            # 「每个专家只等一次」的优化（见 TMA warp 的 last_waited_group）。
                            if old + cutlass.Int32(1) == num_blocks:
                                cute.arch.store(
                                    (expert_signals.iterator + expert_idx).llvm_ptr,
                                    cutlass.Int32(1),
                                    sem="release",
                                    scope="gpu",
                                )

                        aligned_tokens = expert_tokens
                        if cutlass.const_expr(expert_alignment > 1):
                            aligned_tokens = (
                                (expert_tokens + expert_alignment - 1) // expert_alignment) * expert_alignment
                        expert_start += aligned_tokens
                        expert_idx += cutlass.Int32(1)

            # =============================================================================
            # GEMM 的 TMA WARP（warp 7）—— A/B 的搬运工。它与 dispatch 之间只有一个契约：
            # expert_signals[g] == 1 之后，才允许发这个专家的第一笔 A/B load。
            #
            #   for 每个 tile（按 M-contiguous 顺序）:
            #       若 group 变了: acquire-load 自旋等 expert_signals[g]   ← 先等，再要 stage
            #       for k_tile: producer_acquire(stage) → TMA A → TMA B → advance
            # =============================================================================
            # ===== TMA WARP =====
            if warp_idx == self.tma_warp_id and initial_work_tile_info.is_valid_tile:
                work_tile = initial_work_tile_info
                # 生产者状态机：num_ab_stage 级轮转，count 记「本 tile 内第几个 k_tile」，
                # index 记「用第几级 SMEM stage」。所以每换一个 tile 要 reset_count，
                # 但 index 继续轮转、不回零（见下）。
                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.num_ab_stage,
                )
                last_waited_group = Int32(-1)

                # TMA warp 的 tile 循环。注意它和 MMA warp、epilogue 各自跑一份同样的循环，
                # 三者的推进必须严格对齐 —— 这是「用数据流（mbarrier）同步、不用控制流同步」
                # 的前提：控制流是各自算的，只要算得一样就等价。
                while work_tile.is_valid_tile:
                    grouped_gemm_cta_tile_info = work_tile.group_search_result
                    # K is constexpr in this variant, so cur_k_tile_cnt folds
                    # at JIT.
                    # K 在这个变体里是 constexpr（所有专家共享），所以 k_tile 数是编译期常量，
                    # 内层循环边界被 JIT 折成常数 —— 这是「M-contiguous、共享 N/K」的直接收益。
                    cur_k_tile_cnt = cta_k_tile_cnt_constexpr
                    cur_group_idx = grouped_gemm_cta_tile_info.group_idx

                    # 组内下标 → 全局 m 下标（与 dispatch 侧同一套换算）：先取「该专家之前的
                    # cluster-m-tile 数」，转成 mma-tile 单位，再加组内偏移。
                    m_cluster_tile_base = (tile_sched.search_state.tile_count_prev_group // ncluster_tile_n)
                    global_m_mma_tile_idx = (m_cluster_tile_base * cluster_to_mma_m +
                                             grouped_gemm_cta_tile_info.cta_tile_idx_m // cta_group_size)
                    n_tile_idx = grouped_gemm_cta_tile_info.cta_tile_idx_n

                    # 切出这个 tile 的 TMA 源：A 用全局 m 下标；B 多一个 group 维，用 cur_group_idx
                    # 索引 —— 「一个 B descriptor 覆盖所有专家」就落在这一个下标上。
                    tAgA_slice = tAgA[(None, global_m_mma_tile_idx, None)]
                    tBgB_slice = tBgB[(None, n_tile_idx, None, cur_group_idx)]

                    # 【本文件最容易被问「为什么」的一处】—— 等专家信号，放在 producer_acquire 之前。
                    #   若先 producer_acquire 再等信号：手里已经攥着一级空的 SMEM stage，而在等一个
                    #   「不知道多久才好」的通信事件。这段时间里这级 stage 谁也用不了，等于把 GEMM
                    #   的流水深度临时减 1；而深度本来就浅（SMEM 被 dispatch 占了一块），再减 1 可能
                    #   直接卡住 MMA 的供给。
                    #   先等信号、再要 stage：等待期间不占任何稀缺资源，MMA 可以继续吃前面已经装满的
                    #   stage，直到真的没得吃。
                    # 【代价】这个自旋是无界的（由通信进度决定），所以只能摆在「不持有资源」的位置。
                    # Acquire BEFORE producer_acquire so we never hold an SMEM
                    # stage idle while dispatch is still finishing this expert.
                    # 每个专家只等一次。合法性来自 M-contiguous 调度：同一 CTA 的 group_idx 单调不减
                    # （线性 tile 下标递增 + 每个专家占连续区间），所以等过 g 就不可能再回头遇到比 g
                    # 小的专家。→ 省掉每个 tile 一次 acquire load 的自旋。
                    # 【若调度器不是 M-contiguous】（比如按 (n,m) 交错或跨专家跳跃），group 会来回变，
                    # 这个缓存就会漏等 → 读到还没落地的 A：结果错，但不报错。
                    if cur_group_idx != last_waited_group:
                        sig_ptr = (expert_signals.iterator + cur_group_idx).llvm_ptr
                        # One lane spins on the GPU-scope acquire load;
                        # 只有 elect_one 的一个 lane 去自旋（省 warp 内 31 个 lane 的空转），sync_warp 之后
                        # 全 warp 才继续。acquire 语义在这里不可省：它把「看到 signal==1」与「之后 TMA
                        # 读到的 A 内容」建立 happens-before —— 换成 relaxed 读，编译器/硬件可以随意
                        # 提前 TMA 的读，信号就白等了。
                        with cute.arch.elect_one():
                            v = cute.arch.load(
                                sig_ptr,
                                cutlass.Int32,
                                sem="acquire",
                                scope="gpu",
                            )
                            while v == 0:
                                v = cute.arch.load(
                                    sig_ptr,
                                    cutlass.Int32,
                                    sem="acquire",
                                    scope="gpu",
                                )
                        cute.arch.sync_warp()
                        last_waited_group = cur_group_idx

                    # 换 tile 时把 k_tile 计数归零，但保留 index（stage 继续轮转，不回零）。
                    # 紧接着 try_acquire 先探一下第一级 stage 空不空 —— 「提前偷看」的写法：把等待
                    # 提前到现在，下面循环里第一次 producer_acquire 就能直接拿到结果，不再阻塞。
                    ab_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    # 只有后面还有 k_tile（count < cur_k_tile_cnt）才偷看：最后一块之后已经没有
                    # 下一次 acquire，省掉一次无谓的 mbarrier 试探。
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state, )

                    # 【A/B 的深流水】每一轮：acquire 一级 stage（可能阻塞在 empty 上）→ 发两笔 TMA
                    # （A、B 各一笔，各带 multicast mask）→ advance → 立刻偷看下一级空不空。
                    # 把「下次要不要等」的判断塞在两次 TMA 发射之后，等待就和 TMA 的飞行时间重叠了。
                    # stage 的另一半计数由 MMA warp 的 consumer_release 完成。
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

                    # 推进到下一个 tile。这个 advance 在 TMA warp / MMA warp / epilogue 三处各写一遍
                    # —— 每个角色都要走到同一个序列；写错任何一处，三个角色就会错位（表现为读到
                    # 别的 tile 的数据，而不是报错）。
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                # producer_tail：把自己的空/满记账收干净，避免 kernel 结束时留下未满足的 mbarrier
                # 期望，被之后复用这块 SMEM 的 CTA 继承成脏状态。
                ab_pipeline.producer_tail(ab_producer_state)

            # ===== MMA WARP =====
            # =============================================================================
            # MMA WARP（warp 6）—— 唯一发 tcgen05 指令的 warp，累加器住在 TMEM。
            #   * 先跟 epilogue 的分配者 rendezvous（wait_for_alloc）拿到 TMEM 指针；
            #   * 每个 tile：等 A/B stage full → 按 k-block 发 MMA（第一条 ACCUMULATE=False
            #     即覆盖写，其余累加）→ 释放 A/B stage → commit 累加器 tile 给 epilogue；
            #   * 2-CTA 模式下只有 leader CTA 真的发指令（is_leader_cta），非 leader 只跟着
            #     推进 scheduler —— 但两边的 barrier 是 cluster 级的，必须一起走。
            # =============================================================================
            if (warp_idx == self.mma_warp_id and initial_work_tile_info.is_valid_tile):
                # Rendezvous with the epilogue allocator before TMEM retrieve.
                # 与 epilogue 的 allocate 配对：TMEM 由 epilogue warp0 申请（它才是 allocator），
                # MMA warp 只是「等分配完成再取指针」。谁先到谁先等，不需要额外的约定。
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

                work_tile = initial_work_tile_info
                # 消费者状态机：和 TMA warp 的生产者状态机是同一个环的两半（同一个 num_ab_stage），
                # 一次 advance 就是一次 stage 易手。
                ab_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.num_ab_stage,
                )
                acc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.num_acc_stage,
                )
                # MMA 的 tile 循环。它与 TMA warp 之间没有任何显式同步语句，全靠 ab_pipeline 的
                # full/empty mbarrier 传递 —— 这就是「用数据流同步而不是控制流同步」。
                while work_tile.is_valid_tile:
                    cur_k_tile_cnt = cta_k_tile_cnt_constexpr

                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                    ab_consumer_state.reset_count()
                    peek_ab_full_status = cutlass.Boolean(1)
                    # 2-CTA：只有 leader CTA 发 MMA（一条 tcgen05 指令同时覆盖两个 CTA 的 SMEM/TMEM），
                    # 非 leader 在这里什么都不做。但它仍然要执行下面的 advance_to_next_work，
                    # 否则两个 CTA 的 tile 序列会分叉。
                    if is_leader_cta:
                        if ab_consumer_state.count < cur_k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state, )
                        acc_pipeline.producer_acquire(acc_producer_state)

                        # 【每个 tile 的第一条 MMA 是覆盖写】ACCUMULATE=False 让 D = A*B 直接写入累加器，
                        # 之后每个 k-block 都置 True 累加。少了这一句，tile 之间会串味（上一个 tile 的
                        # 结果被累加进来）。它只是 MMA 指令上的一个修饰位，不产生额外指令。
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                        # 内层 k 循环：先 consumer_wait（等 A/B stage 装满），再按 num_kblocks 发 MMA。
                        # num_kblocks 是本级 k_tile 里 MMA 指令的个数（由 mma_inst_shape_k * 4 决定），
                        # 用 unroll_full 完全展开 —— 这是热点中的热点，不能有任何循环开销。
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
                                # 一条 tcgen05 MMA：读 SMEM 里的 A/B（tCrA/tCrB 是 SMEM 上的 fragment），累加进
                                # TMEM（tCtAcc）。没有任何数据经过寄存器堆 —— 这是 warp6 一个 warp 就能驱动整个
                                # tensor core 的原因。
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[kblock_coord],
                                    tCrB[kblock_coord],
                                    tCtAcc,
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                            # 释放这一级 A/B stage：数据已经被 MMA 吃进 tensor core 的流水，TMA warp 可以往里
                            # 写下一个 k_tile 了。这一句就是 ab_pipeline 的「空」信号。
                            ab_pipeline.consumer_release(ab_consumer_state)
                            ab_consumer_state.advance()
                            peek_ab_full_status = cutlass.Boolean(1)
                            if ab_consumer_state.count < cur_k_tile_cnt:
                                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state, )

                        # 整个 tile 的 K 累加完 → commit 累加器，epilogue 的 consumer_wait 被唤醒。
                        # acc_pipeline 的级别数（num_acc_stage=2）与 num_ab_stage 无关：A/B 可以开很深，
                        # 累加器只要够「MMA 算下一个 tile 时 epilogue 还在写上一个」。
                        acc_pipeline.producer_commit(acc_producer_state)
                        acc_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                # 收尾：把累加器的记账清干净，别给 epilogue 留下未满足的期待。
                acc_pipeline.producer_tail(acc_producer_state)

            # ===== EPILOGUE WARPS =====
            # =============================================================================
            # EPILOGUE（warp 0-3，一个完整 warpgroup）—— TMEM → 寄存器 → SMEM → gmem。
            # 为什么是 4 个 warp：TMEM 的读（tcgen05.ld）与 SMEM 的写都按 warpgroup 组织，
            # 4 个 warp 才能把一份 tile 的数据并行摊开；而写 gmem 只用 TMA，warp0 一个 elect
            # 掉就够。
            #   tmem.allocate → t2r → 转 dtype → r2s → fence → (warp0) TMA store
            #   两个 epilogue-local barrier 保证「写完」与「buffer 可复用」的次序。
            # =============================================================================
            if (warp_idx >= self.epilog_warp_id[0] and warp_idx < self.epilog_warp_id[0] + len(self.epilog_warp_id)
                    and initial_work_tile_info.is_valid_tile):
                # TMEM 的申请就在这里：num_tmem_alloc_cols 列（对应 num_acc_stage 级累加器）。
                # 这是 CTA 级的一次性动作，之后 MMA warp 才拿得到指针。
                tmem.allocate(self.num_tmem_alloc_cols)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

                # epilogue 的线程号要在 4 个 warp 内重新编号（0..127）：TMEM/SMEM 的分区是按
                # 「epilogue warpgroup 内部线程号」切的，不是 CTA 全局 tidx。
                epi_tidx = tidx - cutlass.Int32(self.epilog_warp_id[0] * 32)

                (tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc) = (self._epilog_tmem_copy_and_partition(
                    epi_tidx,
                    tCtAcc_base,
                    tCgC,
                    epi_tile,
                    use_2cta_instrs,
                ))

                # 三份分区 = 三级搬运：
                #   tiled_copy_t2r / tTR_rAcc —— TMEM → 寄存器（累加器是 fp32）
                #   tiled_copy_r2s / tRS_rC   —— 寄存器 → SMEM（顺便转成输出 dtype）
                #   tma_atom_c / bSG_sC       —— SMEM → gmem（TMA store，异步）
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self._epilog_smem_copy_and_partition(
                    tiled_copy_t2r,
                    tTR_rC,
                    epi_tidx,
                    sC,
                )
                (tma_atom_c, bSG_sC,
                 bSG_gC_partitioned) = (self._epilog_gmem_copy_and_partition(tma_atom_c, tCgC, epi_tile, sC))

                # 开始消费累加器 tile：acc_consumer_state 与 MMA 的 acc_producer_state 是同一条
                # acc_pipeline 的两端。
                work_tile = initial_work_tile_info
                acc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.num_acc_stage,
                )
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilog_warp_id),
                )
                # 【epilogue 自己的流水：SMEM → gmem 的 TMA store】num_epi_stage 级 SMEM 缓冲
                # （_compute_stages 把 SMEM 的零头都给了它）。它让「TMA 还在写上一块」与
                # 「warpgroup 已经在往下一块 SMEM 里写」重叠；没有它，每次写回都要等 TMA 落地。
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_epi_stage,
                    producer_group=c_producer_group,
                )
                # 每个 tile：等 acc 就绪 → 按 epi subtile 逐块搬。subtile 数由 epi_tile 决定
                # （一份 C tile 会被拆成若干个能放进 SMEM 的小块）。
                while work_tile.is_valid_tile:
                    grouped_gemm_cta_tile_info = work_tile.group_search_result
                    cur_group_idx = grouped_gemm_cta_tile_info.group_idx

                    m_cluster_tile_base = (tile_sched.search_state.tile_count_prev_group // ncluster_tile_n)
                    global_m_mma_tile_idx = (m_cluster_tile_base * cluster_to_mma_m +
                                             grouped_gemm_cta_tile_info.cta_tile_idx_m // cta_group_size)
                    n_tile_idx = grouped_gemm_cta_tile_info.cta_tile_idx_n

                    bSG_gC = bSG_gC_partitioned[(None, None, None, global_m_mma_tile_idx, n_tile_idx)]
                    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                    # 等 MMA 把这一份累加器算完（acc_pipeline 的 full）。
                    acc_pipeline.consumer_wait(acc_consumer_state)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                    # num_prev_subtiles 让 epi_buffer 在 tile 之间也连续轮转（而不是每个 tile 从 0 开始）：
                    # 上一个 tile 最后几块的 TMA store 还在飞，本 tile 的头几块就能用别的 buffer，
                    # 不必等一轮「全部写完」。
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                    for subtile_idx in range(subtile_cnt):
                        epi_buffer = (num_prev_subtiles + subtile_idx) % self.num_epi_stage

                        # 三级搬运的主体：TMEM 读一块 → 转 dtype 写寄存器 → 写 SMEM。
                        # acc_vec.to(c_dtype) 是这里唯一的算术（fp32 累加器 → bf16/fp16 输出）。
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))

                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, epi_buffer)],
                        )
                        # Order epilogue SMEM writes before TMA-store reads;
                        # the named barrier is epilogue-warp scoped.
                        # 【fence + epilogue-local barrier】fence_proxy("async.shared") 让普通 SMEM 写对
                        # TMA（async proxy）可见；arrive_and_wait 保证 4 个 warp 都写完了，warp0 才发
                        # store。这两个 barrier 只含 4 个 warp（128 线程），dispatch warp 完全不参与 ——
                        # 所以 epilogue 的节拍再密，也拖不慢通信链路。
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                        # 只由 warp0 发 TMA store（另外 3 个 warp 只负责把数据写进 SMEM）：producer_commit
                        # 记账这次搬运，紧跟着的 producer_acquire 等最老的一笔 store 完成（即等它腾出
                        # SMEM buffer）。
                        # → 写回完全交给 TMA 异步做，warpgroup 发完就回去搬下一块。
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, epi_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                    # 释放累加器 stage。elect_one 让每个 warp 只 arrive 一次 → 4 次到达，正好等于
                    # acc consumer group 的线程数（2-CTA 时是 8，因为两个 CTA 各出 4 个 warp）。
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                # 收尾顺序是刻意排的：
                #   relinquish_alloc_permit —— 不再需要新的 TMEM；
                #   epilogue barrier        —— 确保没有 warp 还在读累加器；
                #   tmem.free               —— 2-CTA 时 free 内含一次 peer arrive/wait，所以必须放在
                #                              本地 barrier 之后，否则可能释放别人还在读的 TMEM；
                #   producer_tail           —— 把最后几笔 TMA store 收干净，kernel 才能退出。
                tmem.relinquish_alloc_permit()
                self.epilog_sync_barrier.arrive_and_wait()
                # In 2CTA mode, free() does a peer-CTA arrive/wait; keep it
                # after the epilogue-local barrier so no warp reads freed TMEM.
                tmem.free(tmem_ptr)
                c_pipeline.producer_tail()

    # 三个 epilogue 分区辅助函数是 CUTLASS SM100 例子的标准写法。值得注意的只有
    # 「rank 比 MKL 例子少一维」：这里的 C 是 (M,N)，没有 L（专家维）—— 专家已经拍进
    # M 里了，所以每个 tCgC 都是 2D。
    def _epilog_tmem_copy_and_partition(self, tidx, tAcc, tCgC, epi_tile, use_2cta_instrs):
        """Partition the per-tile C accumulator for TMEM->RF copy.

        ``tCgC`` is 2D (no L), one rank below the MKL example.
        """
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # flat_divide 把 (tile, stage) 再按 epi_tile 细分成 subtile，得到 (subtile, stage)
        # 的索引空间；后面那串 6 重下标里最后一位就是 acc stage。
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r,
            tAcc_epi[(None, None, 0, 0, 0)],
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None)],
            epi_tile,
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0)].shape,
            self.acc_dtype,
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

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

    def _epilog_gmem_copy_and_partition(self, tma_atom_c, tCgC, epi_tile, sC):
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

    # =============================================================================
    # 【SMEM 的最终账单】输入是「扣掉 dispatch 之后剩下的容量」，输出是三个 stage 数。
    #   1. num_acc_stage = 2 固定（累加器在 TMEM，不占 SMEM，这里只是沿用 CUTLASS 的命名）；
    #   2. num_epi_stage 从 2 起；
    #   3. num_ab_stage = (容量 - 1KB 余量 - epi buffer) / 每级 A+B 字节 ← 剩下的全给它；
    #   4. 再把除不尽的零头换成额外的 epilogue stage（写回缓冲越深，TMA store 越好藏）。
    # 【为什么 A/B 优先】A/B 的级数决定 MMA 的供给深度，直接决定吞吐；epilogue 只要够
    # 藏住一次 TMA store。所以先保 A/B，零头才给 epi。
    # 【reserved_smem_bytes 的作用】给 mbarrier / 对齐 / 编译器补的边角留 1KB 余量。
    # 少留这一手，一旦算多，launch 直接失败（是硬错误，不是变慢）。
    # =============================================================================
    @staticmethod
    def _compute_stages(tiled_mma, mma_tiler_mnk, a_dtype, b_dtype, epi_tile, c_dtype, c_layout, smem_capacity,
                        occupancy):
        num_acc_stage = 2
        num_epi_stage = 2

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

        # 每级 stage 的实际字节数：A 的一份 tile + B 的一份 tile。要按 swizzle 之后的
        # layout 算，不是 m*n*dtype 的朴素乘积 —— swizzle 会引入 padding。
        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(
            b_dtype, b_smem_layout_staged_one)
        epi_bytes_per_stage = cute.size_in_bytes(c_dtype, epi_smem_layout_staged_one)
        epi_bytes = epi_bytes_per_stage * num_epi_stage

        # 注意是 smem_capacity // occupancy 之后再减：occupancy 目前恒为 1，写成除法是为
        # 了保留「一个 SM 放多个 CTA 时按份均分」的通用性。
        num_ab_stage = (smem_capacity // occupancy - _CuTeDSLDispatchGroupGemmKernelImpl.reserved_smem_bytes -
                        epi_bytes) // ab_bytes_per_stage

        # 除不尽的零头（几 KB）转成额外的 epi stage。这解释了为什么 num_epi_stage 常常
        # 不是 2 —— 它是「把 SMEM 用满」的副产品，而不是调出来的参数。
        remaining_smem = (smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy *
                          (_CuTeDSLDispatchGroupGemmKernelImpl.reserved_smem_bytes + epi_bytes))
        num_epi_stage += remaining_smem // (occupancy * epi_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_epi_stage

    # 【grid 的形状】把 total_num_clusters 塞进 problem_shape_ntile_mnl 的 L 位，再用
    # max_active_clusters 截断 → 实际起的是「一波」persistent cluster。
    # 这个截断是 dispatch 条带划分（num_blocks）成立的前提：网格大小在 launch 时就定死，
    # 所有 CTA 从头活到尾，才谈得上「每个 CTA 认领 token 行的一份」。
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

    # TMEM 的列数由累加器 fragment 的 layout 决定（num_acc_stage 级一起算），
    # 一次性编译期计算。
    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma, mma_tiler, num_acc_stage):
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake)


# 生产入口类：只把 dispatch_num_stages 变成 keyword-only 参数，逻辑全在基类。
# docstring 已经点明它的定位 —— dispatch warp 发 per-expert 信号，GEMM 的 TMA warp
# 在发第一笔 A/B load 之前 acquire 这些信号，这就是完整的 producer/consumer 重叠路径。
class CuTeDSLDispatchGroupGemmKernel(_CuTeDSLDispatchGroupGemmKernelImpl):
    """Production fused dispatch + group-GEMM kernel.

    Dispatch warps emit per-expert release signals; the GEMM TMA warp
    acquires on them before issuing the first A/B load, yielding the full
    producer/consumer overlap path.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple,
        cluster_shape_mn: tuple,
        *,
        dispatch_num_stages: int = 1,
    ):
        super().__init__(
            acc_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            dispatch_num_stages=dispatch_num_stages,
        )
