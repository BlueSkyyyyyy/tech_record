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
# 【本文件】MContiguousGroupTileScheduler —— MoE grouped GEMM 的「M 连续」tile 调度器
#
# 上游：ByteDance-Seed/Triton-distributed，commit 7908e4e
#       路径 FlashComm/python/flash_comm/ep_overlap/kernels/m_contig_group_tile_scheduler.py
#       只读上游副本（对照用）：~/github/Triton-distributed/ 下同一相对路径
# 定位：UniEP（arXiv 2604.19241）第二个 MegaKernel「FC2 GroupGEMM + push-combine」的
#       确定性调度底座。它只干一件事：把扁平 tile 序号（linear tile index）翻译成
#       (group, m_tile, n_tile)，并保证同一个 expert 的 M tile 在扁平序号里连成一整段。
#
# 本文件的注释只讲三件事：
#   1. 相对 CUTLASS 基类 StaticPersistentGroupTileScheduler 改了哪 4 处，各自的「为什么」；
#   2. 「M 连续」如何同时换来 确定性累积、per-expert 一次等待、单张 TMA 描述符；
#   3. _group_search / delinearize_z 的搜索语义，以及 host 侧 padded_m 不变量的用途。
#
# 建议阅读顺序：类 docstring 的 4 条差异 → 下面的「M 连续买到什么」
#   → __init__ / create（调度器有哪些状态）→ delinearize_z（一次搜索定出所有坐标）
#   → 两个 _get_* 钩子（数据源换成 1D）→ 最后看 __extract_mlir_values__。
# =============================================================================
from cutlass.cutlass_dsl import (
    Int32,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    GroupSearchResult,
    GroupedGemmGroupSearchState,
    GroupedWorkTileInfo,
    StaticPersistentGroupTileScheduler,
)


# 【为什么用继承而不是重写】基类的 _group_search() 内部并不直接读 problem_shape_mnkl，
# 而是通过两个虚方法 _get_problem_for_group() / _get_cluster_tile_count_mn() 取「第 i 组有多大」。
# 于是「把 4 元组 (M,N,K,L) 换成 1D 的 problem_sizes_m」这件事，只需要覆写这两个钩子；
# 搜索算法本身（warp 窗口跳跃 + 前缀和 + ballot）一行都不用动。
#
# 【4 处改动总览：docstring 列的是 What，这里补 Why】
#   1) per-group 问题形状压成 1D problem_sizes_m，N/K/L 变成编译期常量
#      → EP 里每个 expert 的权重形状 (N,K) 完全一样，只有「收到多少 token」随流量抖动；
#        把不变的两维提到编译期，K 循环次数、N tile 数、grid 上限就都能在 JIT 期折叠。
#   2) _get_problem_for_group 从 vec-4 的 autovec_copy 降为一次 Int32 标量读
#      → 它在搜索热路径里被每条 lane 每轮调用一次，读量直接降到 1/4。
#   3) _get_cluster_tile_count_mn 的 N 因子变成 constexpr
#      → ceil_div(N, Cn) 在 host 侧算一次即可；留在循环里等于把常量折叠的机会丢掉。
#   4) delinearize_z 从搜索状态反推 cluster_count_m，不再二次加载该组的 (m,n,k,l)
#      → 省掉一次 GMEM 读和三次运行时整除，而这几件事本要由 TMA/MMA/epi 三类 warp
#        在每个 tile 上各重算一遍。
#
# 【版本耦合提醒】下面 __init__ 调基类时只传 9 个位置参数；本地 ~/github/cutlass 的基类
# 已多出 cached_problem_shape_0/1 两个必填参数，直接跑会 TypeError —— 说明这份 kernel
# 绑定在某个特定 CUTLASS 版本上，读的时候以「9 参版本」为准。
class MContiguousGroupTileScheduler(StaticPersistentGroupTileScheduler):
    """Grouped-GEMM tile scheduler for the M-contiguous (shared N/K) case.

    Differences from ``StaticPersistentGroupTileScheduler``:
      * Per-group problem is a 1D ``problem_sizes_m`` (G,) Int32 tensor of
        real (un-padded) token counts; N/K/L are compile-time constants.
      * ``_get_problem_for_group`` is a single Int32 load (vs vec-4 in base).
      * ``_get_cluster_tile_count_mn`` folds N into a constexpr.
      * ``delinearize_z`` recovers ``cluster_count_m`` from search state and
        keeps N/K constexpr (no runtime ceil_div on constants).

    Host invariant: ``padded_m`` is a multiple of ``Cm`` so
    ``ceil_div(real_m, Cm) == ceil_div(padded_m, Cm)``.
    """

    # ---------------------------------------------------------------------------
    # 【「M 连续」买到了什么 —— 本文件的中心思想】
    # 定义：同一个 group（expert）的 M tile 在扁平 tile 序号里连成一段，组内 M-major，
    #       组间按 expert 顺序首尾相接。三件事由它同时成立：
    #
    #   (a) 累积顺序确定（deterministic accumulation）
    #       扁平序号 → A 行号是单调一一映射：第 g 组的起点 = 「前 g 组 tile 数之和 / ncluster_n」。
    #       每个输出 tile 的 A 段由此完全确定，K 方向累加顺序又由 mainloop 固定
    #       → 结果只取决于 (group, m_tile, n_tile)，与「哪个 CTA、第几波、当时多少 token」无关，
    #         重跑 bit 级一致。RL 训练要求可复现，这是硬需求。
    #       去掉会怎样：tile 与 expert 段的对应关系变成运行时函数，必然引入跨 CTA 的部分和，
    #       累加顺序随调度抖动 → 结果不可复现。
    #
    #   (b) per-expert 信号可以「等一次」
    #       dispatch 侧的 GEMM 必须先等「这个 expert 的 A 收齐了」的 release 信号（GPU scope
    #       acquire 自旋）。因为组内 tile 连续，一个 CTA 进入某 expert 后会连续处理它的一串 tile
    #       （每次序号 +P 仍落在同组区间内），于是「换组才等」就够，一次等待摊到多个 tile 上。
    #       去掉会怎样：expert 交错排布 → 每个 tile 都可能换组 → 每 tile 一次 GPU-scope 自旋，
    #       这段延迟直接压在 TMA warp 的关键路径上，通信/计算的重叠被等待吃掉。
    #
    #   (c) A 的 TMA 描述符可以单张覆盖 monolithic A
    #       所有 expert 共享 (N,K)，且每段 A 都从 cluster tile 的整数倍处开始（见 host 不变量），
    #       于是 A 可以当成一块 (total_tokens_padded, K) 的大矩阵，一张 tiled TMA 描述符全覆盖，
    #       per-tile 只换 mode-1 上的全局 m tile 下标。
    #       去掉会怎样：段起点不对齐 → 要么 host 侧为每个 expert 建一张描述符并在设备侧选择，
    #       要么每 tile 重算描述符（TMA 描述符进 kernel 后不可改）。
    #
    # 【host 不变量】padded_m 是 Cm（cluster tile 的 M）的倍数
    #   => ceil_div(real_m, Cm) == ceil_div(padded_m, Cm)
    #   用途：调度器只拿 1D 的 real_m，就同时得到「该 expert 有几个 tile」和「它在 padded A 里的
    #   段起点」，host 不必再传一份 offsets 数组。
    #   崩法：若 host 按别的粒度 pad（如 128 行）而 Cm=256，累计 tile 数就会偏离 A 的真实偏移，
    #   tile 落到相邻 expert 的数据上 —— 不报错，只是算错。
    def __init__(
        self,
        params: utils.PersistentTileSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
        cluster_tile_shape_mnk: tuple,
        search_state: GroupedGemmGroupSearchState,
        group_count: int,
        problem_sizes_m: cute.Tensor,
        problem_shape_n: int,
        problem_shape_k: int,
    ):
        # 【__init__ 只做三件事】记下 1D 的 problem_sizes_m 与 constexpr 的 N/K，并把 N/K 折算成
        # ncluster_n/k（紧跟的两行）。调度游标本身是 super().__init__ 里那 5 个标量。
        StaticPersistentGroupTileScheduler.__init__(
            self,
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
            cluster_tile_shape_mnk,
            search_state,
            group_count,
            problem_sizes_m,
        )
        # 记下这三行之后，调度器状态就齐了 = super 那 5 个 + 这里 3 个：
        #   _current_work_linear_idx : 本 cluster 当前的扁平 tile 序号，初值 = bidz（见 create），
        #                              每前进一次加 num_persistent_clusters → 全网 round-robin
        #   num_persistent_clusters  : 并发 cluster 数 = grid 大小 / cluster 大小
        #   cta_id_in_cluster        : 本 CTA 在 cluster 内的 (x,y)，决定它负责 cluster tile 的哪一角
        #   search_state             : 上次搜索的游标（组号 + 已扫 tile 数），跨 tile 复用
        #   problem_sizes_m          : (G,) Int32 真实 token 数 —— 唯一随负载变化的输入
        #   problem_shape_n/k        : 编译期量，只用于折出 ncluster_n/k
        # 【为什么折成成员而不每 tile 现算】ncluster_n 是 _compute_cta_tile_coord 的入参，每个 tile
        # 每个 warp 都要用；它本来就是个常数（constexpr 的 N 除以 cluster tile 宽度）。
        self.problem_sizes_m = problem_sizes_m
        self.problem_shape_n = problem_shape_n
        self.problem_shape_k = problem_shape_k

        # 改动 3 的落点：N/K 是 constexpr，所以这两行算的是一个纯粹的主机侧常量，
        # 之后每 tile 的取用都只是读寄存器，不再有运行时除法。
        self.ncluster_n = (problem_shape_n + cluster_tile_shape_mnk[1] - 1) // cluster_tile_shape_mnk[1]
        self.ncluster_k = (problem_shape_k + cluster_tile_shape_mnk[2] - 1) // cluster_tile_shape_mnk[2]

    # ---------------------------------------------------------------------------
    # 【CuTe DSL 的对象跨 MLIR 边界机制】@cute.jit 的参数与返回值在 IR 里只能是 SSA value，
    # 普通 Python 对象过不去。于是这类对象必须实现「拆」(extract_mlir_values) 与「装」
    # (new_from_mlir_values)：拆成一个线性 value 列表，另一端按同样顺序重建。
    # 一句话：CuTe DSL 版的序列化/反序列化，而**顺序就是协议**。
    # 【顺序不能错】下面 7 段顺序与 new_from 里的 values[0] / [2:5] / [6:10] / [11:] 切片是硬编码
    # 契约；多一段少一段不会报错，只会把 tile 算到错误的位置上（静默错误）。
    def __extract_mlir_values__(self) -> list:
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        values.extend(extract_mlir_values(self.search_state))
        values.extend(extract_mlir_values(self.problem_sizes_m))
        values.extend(extract_mlir_values(self.params))
        return values

    # 重建端：values[11:] 整段交给 self.params（PersistentTileSchedulerParams，它自己也实现了
    # 同一对方法）。「参数对象内部含多少个 value」在这里被隐式依赖，改基类会连带改这里。
    def __new_from_mlir_values__(self, values: list) -> "MContiguousGroupTileScheduler":
        if len(values) < 11:
            raise ValueError("Length of mlir values extracted is incorrect.")
        new_num_persistent_clusters = new_from_mlir_values(self.num_persistent_clusters, [values[0]])
        new_current_work_linear_idx = new_from_mlir_values(self._current_work_linear_idx, [values[1]])
        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster, values[2:5])
        new_num_tiles_executed = new_from_mlir_values(self._num_tiles_executed, [values[5]])
        search_state = new_from_mlir_values(self.search_state, values[6:10])
        problem_sizes_m = new_from_mlir_values(self.problem_sizes_m, [values[10]])
        params = new_from_mlir_values(self.params, values[11:])

        return MContiguousGroupTileScheduler(
            params,
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
            self.cluster_tile_shape_mnk,
            search_state,
            self.group_count,
            problem_sizes_m,
            self.problem_shape_n,
            self.problem_shape_k,
        )

    # 【create 是唯一入口，kernel 里调用一次】注意初值取 bidz 而不是 0：cluster 在 grid 的 z 维
    # 上编号，每个 cluster 从自己的编号起步，之后每次前进 P（= num_persistent_clusters）步。
    # → 任意时刻整个 grid 覆盖的 tile 序号是连续、无重叠的一段，这就是静态持久化
    #    （static persistent）：网格一次开满 SM，tile 循环留在 kernel 内，省掉反复启动的开销。
    # 【代价】grid.z 的取值变成「SM 数」而不是「tile 数」，越界那部分必须由调度器自己判无效
    #   （下文 found/is_valid），kernel 侧不能靠循环次数收尾。
    @staticmethod
    @dsl_user_op
    def create(
        params: utils.PersistentTileSchedulerParams,
        block_idx: tuple,
        grid_dim: tuple,
        cluster_tile_shape_mnk: tuple,
        initial_search_state: GroupedGemmGroupSearchState,
        group_count: int,
        problem_sizes_m: cute.Tensor,
        problem_shape_n: int,
        problem_shape_k: int,
        *,
        loc=None,
        ip=None,
    ) -> "MContiguousGroupTileScheduler":
        """Build an M-contiguous scheduler with 1D ``problem_sizes_m`` and
        constexpr ``(N, K)``. See the base class for shared arguments."""
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(params.cluster_shape_mn, loc=loc,
                                                                                   ip=ip)

        bidx, bidy, bidz = block_idx
        current_work_linear_idx = Int32(bidz)
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        num_tiles_executed = Int32(0)
        return MContiguousGroupTileScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
            cluster_tile_shape_mnk,
            initial_search_state,
            group_count,
            problem_sizes_m,
            problem_shape_n,
            problem_shape_k,
        )

    # Overloaded helpers: the base class's ``_group_search`` / ``_prefix_sum``
    # call into these two methods, so overriding them redirects the scheduler
    # to the 1D ``problem_sizes_m`` tensor.

    # ---------------------------------------------------------------------------
    # 【改动 1/2/3 集中在这两个钩子上】基类的 _group_search / _prefix_sum 调用它们，覆写即把
    # 数据源从 (G,4) 的 problem_shape_mnkl 换成 (G,) 的 problem_sizes_m。
    #
    # 为什么可以只存 1D 的 M：MoE 每个 expert 是同一张大权重切出的一片，(N,K) 天然共享 → 提成
    # constexpr；每个 expert 的 A 段长度（收到的 token 数）不同，是唯一的运行时量。
    # 去掉会怎样：改回 4 元组后，基类的 cached 预取路径（use_cached_problem_shapes）会把 4 元组
    # 原样塞进 _get_cluster_tile_count_mn，与本类「标量 M」的覆写不兼容。
    # → 本类必须保持 use_cached_problem_shapes=False（create 没传，默认就是 False）。
    #
    # 【参数名陷阱】形参仍叫 problem_shape_mnkl（为了不改基类的调用点，duck typing），
    # 实际传进来的是 1D 的 problem_sizes_m。读代码时别被名字骗了。
    @dsl_user_op
    def _get_problem_for_group(self, problem_shape_mnkl: cute.Tensor, group_idx: Int32, *, loc=None, ip=None) -> Int32:
        """Single Int32 load of the real M for this group."""
        return Int32(problem_shape_mnkl[group_idx])

    @dsl_user_op
    def _get_cluster_tile_count_mn(self, problem_shape: Int32, *, loc=None, ip=None) -> Int32:
        """Cluster-tile count from real m; N factor is constexpr."""
        cur_ntile_m = (problem_shape + self.cluster_tile_shape_mnk[0] - 1) // self.cluster_tile_shape_mnk[0]
        return cur_ntile_m * Int32(self.ncluster_n)

    # ---------------------------------------------------------------------------
    # 【改动 4：文件的核心】输入扁平 tile 序号，输出 (group_idx, cta_tile_idx_m,
    # cta_tile_idx_n, is_valid)。一次搜索定出全部坐标，没有第二次数据加载。
    # 相比基类省两件事，都靠「M 连续」：
    #   (a) 不再二次加载该 group 的 (m,n,k,l) —— cluster_count_m 可由搜索结果反推（见下）；
    #   (b) 不再对运行时值做 ceil_div —— cluster_count_n / ncluster_k 都是 constexpr。
    # 【代价】每 tile 每类 warp 仍要跑一次 warp 级搜索（TMA/MMA/epi 各持一份调度器副本）；
    #   敢这么花，是因为「M 连续」让搜索游标几乎不移动，摊还后接近 O(1)。
    @dsl_user_op
    def delinearize_z(
        self,
        cta_tile_coord: tuple,
        *,
        loc=None,
        ip=None,
    ) -> GroupedWorkTileInfo:
        linear_idx = self._current_work_linear_idx

        # 第 1 步：在扁平 tile 序号上定位 group。_group_search 的实现在基类（~90 行），值得讲清：
        #   * 一个 warp 的 32 条 lane 各领一个 group（start + lane_idx），读该组的 mn tile 数；
        #   * _prefix_sum 用 shuffle_sync_up 做 warp 内前缀和 → 每 lane 得到「到本组为止（含）」的累计数；
        #   * ballot + popc 数出「累计数仍 <= linear_idx」的 lane 数 = 命中组在窗口内的偏移；
        #   * 命中组的起点累计数用 shuffle_sync 取前一条 lane 的值 → 就是 tile_count_prev_group；
        #   * 一窗 32 组都没命中就整窗跳过（start 推到窗口之后），轮数是 ceil(G/32)。
        #     → 接口语义是「在扁平序号上二分式定位 group」，实现是「warp 协作窗口前缀和 + 整窗跳跃」，
        #        不是逐 group 线性扫描（G=256 只要 8 轮，每轮每 lane 只读 1 个 int32）。
        # 起始游标来自上一次的 (start_group_idx, tile_count_prev_group)，不是从 0 重来 ——
        # 同一 cluster 的 linear_idx 单调递增，绝大多数情况下窗口根本不用动。
        self.search_state = self._group_search(
            linear_idx,
            self.problem_sizes_m,
            self.search_state.start_group_idx,
            self.search_state.tile_count_prev_group,
            loc=loc,
            ip=ip,
        )

        # found 即「本序号确实落在某个 group 的 tile 区间内」。越界时搜索会耗尽所有 group 仍找不到
        # → found=False → is_valid=False → 三类 warp 的 persistent 循环同时退出。
        # 【注意】终止条件不是「序号 < 总 tile 数」这种比较，而是搜索本身给不出结果 ——
        # 所以 host 可以把 grid.z 直接开成 max_active_clusters，不必精确等于真实 tile 总数。
        found = self.search_state.found
        is_valid = found
        group_idx = self.search_state.start_group_idx

        tile_count_prev_group = self.search_state.tile_count_prev_group
        cur_ntile_mn = self.search_state.tile_count_searched - tile_count_prev_group
        # 第 2 步（M 连续换来的省事）：搜索已算出「到本 group 为止的 mn tile 累计数」，减去「本 group
        # 之前的累计数」就是本 group 的 mn tile 总数；组内是 M-major，除以 constexpr 的 ncluster_n
        # 就还原出 M 方向的 cluster 数。
        # → 这就是不必再读 (m,n,k,l) 的原因：M 已经藏在搜索结果里了。
        cluster_count_m = cur_ntile_mn // Int32(self.ncluster_n)
        cluster_count_n = Int32(self.ncluster_n)

        # 第 3 步：组内扁平序号 → (m, n)。_compute_cta_tile_coord 按 M-major（AlongM）展开，即线性
        # 序号先把 M 走满再进 N —— 这就是「同一 expert 的 tile 在扁平空间里连成一段」的直接来源，也是上一步
        # 能反推出 m 数的前提。
        # 【去掉会怎样】改成 N-major 或交错排布后，同一 expert 的 tile 会散落在序号空间里，
        # 「按前缀和定位 expert 段」与「一次等待」两个性质会同时失效。
        cluster_tile_idx_in_current_group = linear_idx - tile_count_prev_group
        cta_tile_idx_m, cta_tile_idx_n = self._compute_cta_tile_coord(
            cluster_tile_idx_in_current_group,
            cta_tile_coord,
            cluster_count_m,
            cluster_count_n,
            loc=loc,
            ip=ip,
        )

        # 把 N/K/ncluster_k 以 Int32(...) 再包一层，只为让 GroupedWorkTileInfo 的 value 列表类型统一、
        # 便于跨 MLIR 边界传递；数值仍是编译期常量，不引入运行时依赖。
        group_search_result = GroupSearchResult(
            group_idx,
            cta_tile_idx_m,
            cta_tile_idx_n,
            cur_ntile_mn,
            Int32(self.problem_shape_n),
            Int32(self.problem_shape_k),
            Int32(self.ncluster_k),
        )
        return GroupedWorkTileInfo(cta_tile_coord, is_valid, group_search_result)
