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
#       FlashComm/python/flash_comm/ep_overlap/_ops/dispatch_group_gemm.py
# 对照副本（只读）：~/github/Triton-distributed/
# 背景：UniEP（ByteDance Seed + 清华，arXiv 2604.19241）MoE 前向的第一段
#       MegaKernel。本文件是它的 host 侧 API：device 逻辑全在
#       kernels/cutedsl_dispatch_group_gemm*.py，这里只决定「编哪个变体」
#       和「这一拍把哪些指针搬进去」。
#
# 【一句话定位】把 pull 式 dispatch 与 FC1 group-GEMM 融进一次 launch 的
# 编译/启动封装：编译期吃形状与 arch，运行期只搬指针。
#
# 【本文件的注释只讲三件事】
#   1. 编译期 vs 运行期：constexpr（进 cubin）与 tensor（每拍换指针）怎么分，
#      分错的代价是「重编译一次」还是「结果静默错」。
#   2. 最小 dummy tensor：JIT 只读 rank / dtype / 哪个 mode 的 stride==1，
#      不这么做要先占十几 GB —— 编译期与运行期解耦最典型的一处。
#   3. 布局契约：expert_alignment 为什么必须 ≥ GEMM_CLUSTER_TILE_M、
#      max_active_clusters 为什么就是 grid 的 cluster 数、无 weight 时为什么
#      还得传一个空张量。
# 英文注释和 docstring 保持上游原样；中文只做补充，不逐句翻译。
#
# 【建议阅读顺序】（行号＝本副本行号，点进去即可）
#   ① 行 96 起  三个构造期常量：什么会进变体键
#   ② 行 106 起  variant_args：改了什么必须重编译
#   ③ 行 157 起  arch → 几何：两套 tiling 与同一个 256
#   ④ 行 216 起  max_active_clusters = grid 的 cluster 数（persistent CTA）
#   ⑤ 行 239 起  expert_alignment 兜底：为什么不能比 cluster tile 更细
#   ⑥ 行 259 起  最小 dummy tensor：省掉的十几 GB
#   ⑦ 行 309 起  22 个实参：整份文件的契约清单
#   ⑧ 行 342 起  运行期：形状校验 + 只搬指针
# =============================================================================
from typing import Optional

import torch

# _base 收走了与具体 kernel 无关的公共设施，这里用到四样：
#   GEMM_CLUSTER_TILE_M —— GEMM 的 cluster 级 M tile（256），本文件用它给
#                          expert_alignment 兜底；它是 device 端写死几何的镜像，
#                          不是可调超参（cluster：线程块簇，几个 CTA 绑在一起
#                          共享多播与 barrier）；
#   make_moe_jit_dummies —— 造最小 JIT 占位张量（见 _build）；
#   mark_dynamic         —— 把 torch 张量包成「layout 动态」的 CuTe 视图，
#                          JIT 只从它取 rank / dtype / 哪个 mode 的 stride==1；
#   CuTeDSLEPOverlapOpBase —— 带全局 kernel cache 的基类，_compile 复用它。
from ._base import (
    CuTeDSLEPOverlapOpBase,
    GEMM_CLUSTER_TILE_M,
    cute_compile_options,
    make_moe_jit_dummies,
    mark_dynamic,
    resolve_nvcc_opt_level,
)


# 【三段式】_compile（决定是哪个变体）→ _build（真正 JIT 编译）→ run（只搬指针）。
# 切开不是为了好看：CuTe DSL 会把形状、dtype、arch、grid 全部烧进 cubin，
# 任何一项变了都必须重编译；反过来，只要这些不变，运行期就只剩一次 kernel
# launch 加参数封送，host 侧开销与普通 CUDA kernel 同量级。
# 【overlap 上的意义】host 侧不做 shape-dependent 决策、不插入同步，
# 通信与计算的交叠窗口不会被 host 分支打断；每个 step 调一次也不心疼。
class CuTeDSLDispatchGroupGemmOp(CuTeDSLEPOverlapOpBase):
    """Fused pull dispatch + FC1 group-GEMM op."""

    # 三个构造期常量：rank / world_size 进变体键也进 kernel 的 constexpr；
    # expert_alignment 由上层 owner 给（默认 1 = 不额外对齐），这里只存不改 ——
    # 「至少对齐到 cluster tile」这件事必须放到 _build 做，因为只有那里才知道
    # 当前 arch 的 cluster 几何是 (2,2) 还是 (2,1)。
    def __init__(self, *, rank: int, world_size: int, expert_alignment: int):
        super().__init__(rank=rank, world_size=world_size)
        self.expert_alignment = int(expert_alignment)

    # 【编译期入口】_compile 自己不做编译，只做两件事：
    #   (1) 把「会影响生成代码的一切」摊成一个可哈希的 variant 元组；
    #   (2) 交给基类的全局 kernel cache：命中就直接返回已编译 handle，
    #       未命中才调 factory。
    # 于是同进程内换 arch / 换 experts_per_rank / 换 has_weight 都不会串味，
    # 而重复调用只付一次 dict 查找。
    def _compile(self, dtype: torch.dtype, experts_per_rank: int, n_out: int, hidden_in: int, dispatch_num_stages: int,
                 num_sm: int, topk: int, weight_dtype: torch.dtype, has_weight: bool):
        # arch 从当前 device 读，而不是从配置读：variant 键必须反映「真正要跑的
        # 那张卡」，否则在混布集群里会拿到别的架构编出来的 cubin
        # （黑盒里表现为非法指令或静默错）。
        arch_major, arch_minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        # variant 元组 = 编译期常量清单，逐项都是「会进 cubin 的值」：
        #   dtype / weight_dtype → MMA 指令与指针解引用位宽；
        #   experts_per_rank     → group 数、problem_sizes_m 长度、信号数组长度；
        #   n_out / hidden_in    → MMA tiling、每 stage 的 SMEM 字节数、TMA box；
        #   rank / world_size    → 信号与指针表寻址、编译期循环上界；
        #   expert_alignment     → 每个 expert 的 M 补齐粒度（调度器的假设）；
        #   dispatch_num_stages  → dispatch 侧流水级数（吃 SMEM，改变 stage 划分）；
        #   num_sm               → persistent grid 的 cluster 数（见 _build 的
        #                          max_active_clusters）；
        #   topk / has_weight    → 对应分支能否被 constexpr DCE 掉。
        # 【一句话】这张表就是「改了什么必须重编译」的清单；表外的量才允许每拍变。
        variant_args = (
            dtype,
            int(experts_per_rank),
            int(n_out),
            int(hidden_in),
            int(self.rank),
            int(self.world_size),
            int(self.expert_alignment),
            int(dispatch_num_stages),
            int(num_sm),
            int(topk),
            weight_dtype,
            bool(has_weight),
            int(arch_major),
            int(arch_minor),
        )

        # factory 只在 cache miss 时被调用：命中路径上连 import cutlass 都不会
        # 发生 —— 这是热路径（每 step 一次 run）不付冷启动成本的关键。
        def factory():
            return self._build(dtype, experts_per_rank, n_out, hidden_in, dispatch_num_stages, num_sm, topk,
                               weight_dtype, has_weight, arch_major, arch_minor)

        # opt-level 按 CUDA 版本选（13.1+ 在这类 kernel 上 level 3 反而回归），
        # 它进 compile_options，也就隐含进了 cache 身份：不同 opt-level 是不同产物。
        opt_level = resolve_nvcc_opt_level()
        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
            compile_options=("--enable-tvm-ffi", f"--opt-level {opt_level}"),
        )

    # 【构建期】只有 cache miss 才走到这里。函数体把 cutlass 相关 import 全放在
    # 函数内部：只有冷启动路径需要付这些重量级 import 的时间。
    def _build(self, dtype: torch.dtype, experts_per_rank: int, n_out: int, hidden_in: int, dispatch_num_stages: int,
               num_sm: int, topk: int, weight_dtype: torch.dtype, has_weight: bool, arch_major: int, arch_minor: int):
        # 四个 import 各司其职：cutlass（dtype/常量）、cute（编译与 launch）、
        # cutlass_torch（torch dtype → cutlass dtype）、utils（HardwareInfo，
        # 算可驻留 cluster 数）。
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        # dtype 白名单在编译期就炸，而不是让 JIT 编到一半再报错：报错点离调用点
        # 越近，越不需要读 PTX 才知道哪里不对。
        self._assert_supported_dtype(dtype, "CuTeDSL dispatch+groupgemm")

        # 【arch → 几何】两套 kernel 源文件，几何写死，且必须与
        # GEMM_CLUSTER_TILE_M 自洽：
        #   sm100（Blackwell）：mma_tiler=(256,256)、cluster=(2,2)、2-CTA MMA；
        #   sm90 （Hopper）   ：mma_tiler=(128,256)、cluster=(2,1)、单 CTA MMA。
        # 两边「每 cluster 的 M tile」都恰好是 256：
        #   sm100：mma_M(256) / cta_group(2) * cluster_M(2) = 256
        #   sm90 ：mma_M(128) * cluster_M(2)               = 256
        # 【为什么必须相等】dispatch 侧按 256 行一块把收到的 token 打包进
        # A_padded，GEMM 侧按 cluster tile 遍历 group；一旦两边不等，一个 cluster
        # tile 会横跨两个 expert，problem_sizes_m 的边界就失效了 ——
        # 形状全对、结果全错。
        if arch_major >= 10:
            from ..kernels.cutedsl_dispatch_group_gemm import (
                CuTeDSLDispatchGroupGemmKernel, )

            mma_tiler = (256, 256)
            cluster = (2, 2)
            use_2cta = True
        elif arch_major == 9:
            from ..kernels.cutedsl_dispatch_group_gemm_sm90 import (
                CuTeDSLDispatchGroupGemmKernel, )

            mma_tiler = (128, 256)
            # Match GEMM_CLUSTER_TILE_M=256 used by dispatch packing.
            cluster = (2, 1)
            use_2cta = False
        # 低于 sm90 直接拒绝：kernel 依赖 TMA + cluster + 大 SMEM，没有降级实现。
        else:
            raise RuntimeError(
                f"CuTeDSL dispatch+groupgemm requires Hopper (sm90) or newer, got sm{arch_major}{arch_minor}")

        # torch dtype → cutlass dtype 的显式映射；白名单在 _assert_supported_dtype 里。
        ab_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
        # 【grid 不是问题规模，而是「能同时驻留多少 cluster」】
        # persistent CTA（持久化 CTA：block 数 ≈ 驻留上限，一个 block 占住 SM
        # 反复领任务，而不是「一 tile 一个 block」）要求 grid 在编译期就是常量：
        #   hw 上限：HardwareInfo.get_max_active_clusters(cluster_size) —— 受
        #            SMEM / register / cluster 调度能力约束；
        #   预算上限：num_sm // cluster_size —— 用户可能只把一部分 SM 给 GEMM
        #            （comm/comp SM 分区正是 overlap 的主旋钮）。
        # 取 min 再 max(1, ...)：num_sm 很小时也至少留一个 cluster，不会空 grid。
        # 结果同时喂给两个 constexpr：total_num_clusters（调度器认为问题最多有多少
        # 个 cluster tile）与 max_active_clusters（grid 形状 = 该数 × cluster 维度）。
        # 【代价】num_sm 因此是编译期量：改 SM 预算要重编译，不能运行期调。
        hardware_info = utils.HardwareInfo()
        max_active_clusters = min(
            hardware_info.get_max_active_clusters(cluster[0] * cluster[1]),
            max(1,
                int(num_sm) // (cluster[0] * cluster[1])),
        )

        # expert_alignment is supplied by the owner and may be the
        # alignment==1 default; GEMM still needs cluster-tile alignment.
        # 【对齐的真实理由】expert_alignment 表示「每个 expert 的 M 段在 A_padded
        # 里的补齐粒度」。GEMM 侧有两条硬约束：
        #   1) M-contiguous 调度器按 cluster tile（M=256）遍历 group，并假设
        #      ceil_div(real_m, Cm) == ceil_div(padded_m, Cm)，即每个 expert 的
        #      padded M 必须是 Cm 的整数倍（见 m_contig_group_tile_scheduler 的
        #      host invariant）；
        #   2) dispatch 的到达信号是 per-expert 的（expert_signals[i] 由 dispatch
        #      release-store，GEMM 的 TMA warp 在首次 load 前 acquire），信号覆盖
        #      的行域必须与 cluster tile 边界对齐，否则「信号已置位」并不等于
        #      「我这个 tile 要的行全都到了」。
        # 所以取 max(user, GEMM_CLUSTER_TILE_M)：允许用户要求更粗（例如 512，
        # 换取更稳的 problem_sizes_m），但绝不允许比 cluster tile 更细。
        # 【去掉会怎样】小 expert（token 数 < 256）会算出 0 个或半个 cluster tile：
        # 行被静默丢弃、或 tile 读到尚未到达的行，且信号与 tile 边界错位 ——
        # 偶发、难复现。
        expert_alignment = max(
            int(self.expert_alignment),
            GEMM_CLUSTER_TILE_M,
        )

        # 【最小 dummy tensor】CuTe DSL 的 JIT 在编译期只从张量里读三样东西：
        #   rank、element dtype、「哪个 mode 的 stride == 1」（leading dim）。
        # 数值不读；dynamic 轴的实际大小也不读（那是运行期 Int32 参数的事）。
        # 所以这里不需要真实 buffer，只需要「形状足以钉住 stride-1 不变式」的
        # 最小张量：
        #   A : (1, K)    行主序；
        #   B : (1, K, L) 且 mode-1 stride==1（用 permute 造出来，与生产视图同源）；
        #   C : (1, N)    行主序；
        #   psm: (L,)     Int32，对应运行期 recv_expert_counts 的 ABI。
        # 【不这么做的代价】生产形状下 A+C 要 M*(K+N)、B 要 L*N*K：光为了喂一次
        # cute.compile 就得先分配十几 GB BF16，编译一次还要等这些分配。
        # 换成 dummy 后总量降到几 MB、耗时到微秒级。
        # 【这就是编译期/运行期解耦的典型收益】编译期只吃「布局」，运行期才吃「数据」。
        # JIT only inspects dtype / rank / mode-1-stride-1 of these tensors;
        # values and dynamic-axis sizes are never read.
        dummy_A, dummy_B, dummy_C, dummy_psm = make_moe_jit_dummies(
            num_experts=experts_per_rank,
            n=n_out,
            hidden_in=hidden_in,
            ab_dtype=cutlass_torch.dtype(ab_dtype),
        )
        # 其余 dummy 都是 O(world_size) 或 O(experts_per_rank) 的一维表，本来就不大，
        # 但仍要真分配：它们的 dtype 与 rank 决定 kernel 里指针表如何被解引用。
        # 内容永远不被读 —— 只为让 mark_dynamic 能取到 descriptor。
        dummy_input_ptrs = torch.empty((self.world_size, ), dtype=torch.int64, device="cuda")
        dummy_tsri = torch.empty((1, ), dtype=torch.int64, device="cuda")
        dummy_recv_count = torch.empty((self.world_size, ), dtype=torch.int32, device="cuda")
        dummy_signals = torch.empty((experts_per_rank, ), dtype=torch.int32, device="cuda")
        dummy_counters = torch.empty((experts_per_rank, ), dtype=torch.int32, device="cuda")
        dummy_input_weight_ptrs = torch.empty(
            (self.world_size, ),
            dtype=torch.int64,
            device="cuda",
        )
        dummy_output_weight = torch.empty(
            (1, ),
            dtype=weight_dtype,
            device="cuda",
        )

        opt_level = resolve_nvcc_opt_level()
        # kernel 构造参数 = 架构级常量（accum dtype、是否 2-CTA、MMA tile、
        # cluster 形状、dispatch 流水级数）。它们不进 cute.compile 的实参表，
        # 而是成为 kernel 对象本身 —— 这也是为什么 dispatch_num_stages 只能靠
        # 「换一个编译产物」生效。
        kernel = CuTeDSLDispatchGroupGemmKernel(
            cutlass.Float32,
            use_2cta,
            mma_tiler,
            cluster,
            dispatch_num_stages=dispatch_num_stages,
        )
        # 【契约清单】下面 22 个实参严格对应 kernel.__call__ 的位置参数，顺序不能动。
        # 分三类看：
        #   ① 运行期张量（mark_dynamic 包过）：input_ptrs / tsri / recv_count /
        #      A / B / C / psm / expert_signals / counters / 两张 weight 表；
        #      JIT 只从它们取 rank+dtype+stride-1，运行期再换成真指针；
        #   ② 编译期 constexpr（裸 int/bool）：experts_per_rank、n_out、hidden_in、
        #      total_num_clusters、max_active_clusters、rank、world_size、
        #      expert_alignment、topk、has_weight —— 全部烧进 cubin；
        #   ③ stream：这里塞的是 fake stream（tvm-ffi env stream），真正的 stream
        #      在 launch 时从环境取，所以 run() 的实参表里看不到它。
        # 【要点】has_weight 是 constexpr → 有/无 weight 编出两个 cubin；无 weight
        # 版本里所有 weight 读写被 DCE（dead code elimination，死代码消除）掉，
        # 既不付寄存器也不付运行期分支。
        return cute.compile(
            kernel,
            mark_dynamic(dummy_input_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_tsri, enable_tvm_ffi=True),
            mark_dynamic(dummy_recv_count, enable_tvm_ffi=True),
            mark_dynamic(dummy_A, enable_tvm_ffi=True),
            mark_dynamic(dummy_B, enable_tvm_ffi=True),
            mark_dynamic(dummy_C, enable_tvm_ffi=True),
            int(experts_per_rank),
            int(n_out),
            int(hidden_in),
            mark_dynamic(dummy_psm, assumed_align=16, enable_tvm_ffi=True),
            int(max_active_clusters),
            int(max_active_clusters),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            mark_dynamic(dummy_signals, enable_tvm_ffi=True),
            mark_dynamic(dummy_counters, enable_tvm_ffi=True),
            int(self.rank),
            int(self.world_size),
            int(expert_alignment),
            mark_dynamic(dummy_input_weight_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_output_weight, enable_tvm_ffi=True),
            int(topk),
            int(bool(has_weight)),
            options=cute_compile_options(enable_tvm_ffi=True, opt_level=opt_level),
        )

    # 【运行期入口】run() 不碰 kernel 内容，只做三件事：
    #   1) 从张量形状反推 _compile 需要的变体参数，并做「低成本、错了会静默错」
    #      的校验；
    #   2) 走 cache 取已编译 handle（首次调用才真编译）；
    #   3) 把这一拍的指针封送进 kernel 并 launch。
    # 因此每 step 调用的 host 成本与一次普通 CUDA launch 同阶。
    def run(self, *, A_padded: torch.Tensor, B: torch.Tensor, token_src_rank_topk_and_indices: torch.Tensor,
            recv_count: torch.Tensor, recv_expert_counts: torch.Tensor, output: torch.Tensor,
            dispatch_input_ptrs: torch.Tensor, expert_signals: torch.Tensor, expert_signal_counters: torch.Tensor,
            dispatch_num_stages: int, num_sm: int, input_weight_ptrs: Optional[torch.Tensor] = None,
            output_weight: Optional[torch.Tensor] = None, topk: int, weight_dtype: torch.dtype) -> None:
        """Launch the compiled dispatch+groupgemm kernel.

        Caller must stage input into ``dispatch_input_buf`` and zero
        ``expert_signals`` / ``expert_signal_counters`` beforehand.

        Pass ``(input_weight_ptrs, output_weight, topk)`` together to
        enable the side-loaded weight dispatch path; ``output_weight``
        must be a 1D weight tensor sized to ``A_padded``.
        Leave them ``None`` for the weightless path (a transient empty
        scratch is fed to satisfy the JIT signature).
        """
        # 形状反推三个 constexpr：experts_per_rank（per-expert 计数向量的长度）、
        # hidden_in（A_padded 列数）、n_out（B 的第 0 维）。
        # 注意这里「反推即编译」：推错了不会报错，而是编出一个几何不对的 kernel。
        experts_per_rank = int(recv_expert_counts.shape[0])
        hidden_in = int(A_padded.shape[1])
        n_out = int(B.shape[0])

        # B 必须是 (N, K, L)，且 K == A_padded.shape[1]、L == experts_per_rank。
        # 为什么先查它：L 错了 → group 数与 problem_sizes_m 长度错位，kernel 会把
        # expert i 的行拿去做 expert i+1 的 GEMM，形状全对、结果全错（最难查的一类）。
        # K 错了 → TMA box 越界。
        # 【分工】这里不查 B 的 stride（K-major 视图由上层 _validate_gemm_B 保证），
        # 也不查 A 是否 contiguous —— 布局契约属于 dispatcher，本 op 只兜「静默错」。
        if B.shape != (n_out, hidden_in, experts_per_rank):
            raise ValueError(f"B must be (N, K={hidden_in}, L={experts_per_rank}); "
                             f"got {tuple(B.shape)}")
        # output 必须是 (M_padded, N)，且 M_padded 与 A_padded 行数严格相等。
        # 写成 (N, M) 或漏掉对齐 padding 都会让 GEMM epilogue 静默写越界 ——
        # 越界不一定崩，可能只是踩坏相邻 buffer，所以必须在这里拦。
        if output.shape != (A_padded.shape[0], n_out):
            raise ValueError(f"output must be (M={A_padded.shape[0]}, N={n_out}); "
                             f"got {tuple(output.shape)}")

        # 【两条路径】has_weight 不是优化开关，而是两个不同的 kernel 变体：
        #   False：dispatch 只搬 token 行，GEMM 输出直接落到 output；
        #   True ：dispatch 顺手把每个 token 的 top-k 标量搬到目标 rank 的
        #          (max_m, topk) 对称缓冲（input_weight_ptrs 是 world_size 长的
        #          指针表），并把本 rank 收到的每行 weight 写进 output_weight
        #          （每行 4 B）。
        # 为什么值得：combine 还原 y_t = Σ_j w_{t,j}·E_j(x_t) 要用这些标量；
        # 让它们搭 dispatch 已有的流水一起走，就不必为 weight 单开一次通信窗口 ——
        # 这正是 overlap 收益的一部分（上层 _prepare_dispatch_buffers 用同一个
        # ep_group_barrier 覆盖两块 staging，不增加跨 rank 同步）。
        has_weight = output_weight is not None
        if has_weight:
            # 以下每条校验都对应一个「编译期已经定死」的假设，必须在 launch 前对齐：
            #   input_weight_ptrs 非空 + int64 + 形状 (world_size,) → 指针表 ABI；
            #   topk > 0            → kernel 里按 topk 展开的常量循环不能为 0；
            #   output_weight.dtype → 它就是变体键里的 weight_dtype，错了会按错误
            #                         位宽解引用（静默读到垃圾）；
            #   1D 且长度 == M_padded → epilogue 按「M-contiguous 行号」逐行写标量，
            #                         短了就是越界写。
            # 成本：都是 O(1) 的 Python 检查，相对一次 launch 可忽略；
            # 收益：把「结果全错但不崩」变成「调用点立刻 TypeError / ValueError」。
            if input_weight_ptrs is None:
                raise ValueError("input_weight_ptrs must be supplied when "
                                 "output_weight is given")
            if int(topk) <= 0:
                raise ValueError(f"topk must be > 0 when has_weight=True; got {topk}")
            if output_weight.dtype != weight_dtype:
                raise TypeError(f"output_weight dtype must be {weight_dtype}; got {output_weight.dtype}")
            if output_weight.ndim != 1:
                raise ValueError("output_weight must be 1D; got shape "
                                 f"{tuple(output_weight.shape)}")
            if output_weight.shape[0] != A_padded.shape[0]:
                raise ValueError("output_weight first dim must equal A_padded.shape[0]"
                                 f" ({A_padded.shape[0]}); got {output_weight.shape[0]}")
            if input_weight_ptrs.dtype != torch.int64:
                raise TypeError("input_weight_ptrs must be int64; got "
                                f"{input_weight_ptrs.dtype}")
            if input_weight_ptrs.shape != (self.world_size, ):
                raise ValueError("input_weight_ptrs must be shape (world_size,)="
                                 f"({self.world_size},); got "
                                 f"{tuple(input_weight_ptrs.shape)}")
        # 【注意】dispatch_num_stages 与 num_sm 不出现在下面的 launch 实参里，
        # 但它们进变体键：它们改变的是流水级数与 grid 大小（编译期量），只能靠
        # 「换一个编译产物」生效。
        # 换句话说，调这两个旋钮的代价是一次 JIT 重编译 + 一次 cubin 加载，
        # 不是免费的。实践上应该先把 (num_sm, stages) 的取值域枚举好，
        # 别在训练中途反复重编。
        compiled = self._compile(
            dtype=A_padded.dtype,
            experts_per_rank=experts_per_rank,
            n_out=n_out,
            hidden_in=hidden_in,
            dispatch_num_stages=dispatch_num_stages,
            num_sm=num_sm,
            topk=int(topk),
            weight_dtype=weight_dtype,
            has_weight=has_weight,
        )

        # 【无 weight 也要传张量】has_weight=0 时 kernel 里所有 weight 读写都被
        # constexpr DCE 掉，但 JIT 签名仍要求这两个位置各有一个张量：
        # mark_dynamic 要从张量里取 descriptor（rank / dtype / stride）。
        # 传 None 不是「少一个参数」，而是「签名对不上」。
        # 于是现造两个空张量：O(world_size) int64 与 O(M) 标量。
        # 代价：每次调用两次 torch.empty（走 caching allocator，微秒级，不清零）；
        # 收益：weightless 路径不必在每个 step 之前维护两块「永远不读」的 buffer。
        if has_weight:
            launch_input_weight_ptrs = input_weight_ptrs
            launch_output_weight = output_weight
        else:
            # ``has_weight=0`` constexpr DCEs every read of these args,
            # but the JIT signature still needs a tensor of the right
            # shape/dtype for ``mark_dynamic`` to extract its descriptor.
            launch_input_weight_ptrs = torch.empty(
                (self.world_size, ),
                dtype=torch.int64,
                device=A_padded.device,
            )
            launch_output_weight = torch.empty(
                (A_padded.shape[0], ),
                dtype=weight_dtype,
                device=A_padded.device,
            )

        # 【真正 launch】这里只剩 11 个张量实参，全是指针/描述符：
        #   dispatch_input_ptrs → world_size 长的对称指针表（pull 的源头）；
        #   tsri                → 每个 A 行的 (src_rank, topk_idx, token_idx) 打包
        #                         编码：这一行来自谁、属于哪个 expert、是否为 padding；
        #   recv_count          → device 上的 (world_size,) 计数张量，kernel 据它决定
        #                         每个源 rank 要 pull 多少行（动态 M 轴的真实边界）；
        #   A_padded / B / output → 三块矩阵（A 的行数是本拍 host 侧容量，真实边界
        #                         由 recv_count 与 recv_expert_counts 给）；
        #   recv_expert_counts  → 每个 expert 的真实 token 数 = problem_sizes_m；
        #   expert_signals / counters → per-expert 到达信号：dispatch 侧 release、
        #                         GEMM 侧 acquire（acquire/release 与 C++ 内存序同义：
        #                         看到信号，就保证能看到它之前写的那些行）；
        #   weight 两张表       → 仅 has_weight=True 有意义。
        compiled(
            dispatch_input_ptrs,
            token_src_rank_topk_and_indices,
            recv_count,
            A_padded,
            B,
            output,
            recv_expert_counts,
            expert_signals,
            expert_signal_counters,
            launch_input_weight_ptrs,
            launch_output_weight,
        )
