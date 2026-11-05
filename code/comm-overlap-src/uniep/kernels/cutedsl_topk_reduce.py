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
#       FlashComm/python/flash_comm/ep_overlap/kernels/cutedsl_topk_reduce.py
# 对照副本（只读）：~/github/Triton-distributed/
# 背景：combine 之后，每个 token 的 topk 份输出散落在 staging 的连续 topk
#       行里；这个 kernel 是唯一把它们收回来的地方，也是整条链路数值语义的
#       落点 —— overlap 改了什么顺序都可以，这里必须与串行执行逐位一致。
#
# 【一句话定位】确定性 top-k 归约 kernel：一个 CTA 一个 token、k 从小到大、
# FP32 累加、128-bit 原子访存，全程没有一行原子操作。
#
# 【本文件的注释只讲三件事】
#   1. 并行怎么切：CTA↔token 的静态绑定、thread↔128-bit atom、
#      num_tiles 的循环上界从哪来。
#   2. 确定性从哪来：「固定求和顺序 + 唯一累加器 = 与串行逐位一致」这条链，
#      以及它与 atomicAdd 归约的本质区别（后者顺序由硬件到达顺序决定）。
#   3. 代价：topk 个寄存器块换 MLP、带宽贴着理论下界、不做跨 CTA 并行。
# 英文注释和 docstring 保持上游原样；中文只做补充，不逐句翻译。
#
# 【建议阅读顺序】（行号＝本副本行号，点进去即可）
#   ① 行 66 起  名字里的 BlockPerToken 就是确定性的一半
#   ② 行 86 起  grid / num_tiles / num_atoms 怎么算
#   ③ 行 135 起  一维取 id：bidx == token
#   ④ 行 160 起  128-bit 拷贝原子：一条 LDG.E.128 搬 8 个 half
#   ⑤ 行 192 起  zipped_divide 的分块方式
#   ⑥ 行 249 起  【确定性核心】两段循环：先全 load，再按 k 序 FP32 累加
# =============================================================================
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


# 【确定性的一半在名字里】BlockPerToken：一个 token 的全部 topk 行由同一个 CTA
# 读、同一个线程按固定顺序加。只要「一个输出元素只有一个 owner」，就不存在
# 归约顺序的竞争 —— 这是后面所有讨论的前提。
# non-persistent（非持久化）：grid = num_tokens，一个 CTA 干完一个 token 就退出，
# 由硬件调度器铺满 SM。这里不做 persistent 是因为负载天然均匀（每 token 固定
# topk 行），persistent 只会增加调度复杂度、拿不到收益。
class MoETopkReduceBlockPerToken:
    """Non-persistent topk reduce: one CTA per output token.
    """

    # 两个几何常量：
    #   num_threads = 256 → 每 CTA 256 个线程，一次覆盖 256 个 atom；
    #   atom_v      = 8   → 每线程一次搬 8 个元素（bf16/fp16 = 16 B = 128 bit）。
    # 于是 num_atoms = hidden/8，num_tiles = ceil(num_atoms/256)。
    # 以 hidden=4096 为例：512 atoms → 正好 2 个 tile，尾部无空转。
    num_threads: int = 256
    atom_v: int = 8

    # 【host 侧只算两件事】把 hidden/topk/num_experts 作为 constexpr 传下去（决定
    # 循环上界），并在这里把 num_tiles/num_atoms 也算好当 constexpr 传 —— 于是
    # kernel 里的 tile 循环是编译期展开的定长循环。
    # 签名分三类，读的时候别混：
    #   张量：staging / topk_indices / output；
    #   constexpr：hidden_size / topk / num_experts（进 cubin）；
    #   运行期标量：num_tokens（Int32）+ stream。
    @cute.jit
    def __call__(
        self,
        staging: cute.Tensor,  # (num_tokens * topk, hidden_size)
        topk_indices: cute.Tensor,  # (num_tokens, topk)
        output: cute.Tensor,  # (num_tokens, hidden_size)
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        num_experts: cutlass.Constexpr[int],
        num_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # hidden 必须能被 atom_v 整除：输出按 8 元素 atom 切块，最后一块若不足 8 个元素，
        # 128-bit 原子访问就会跨行 / 越界。host 侧看不到这个约束，只能在这里 assert。
        # 副作用：hidden % 8 == 0 同时保证行宽 hidden×2 B 是 16 B 的倍数，于是每行起点
        # 都 16 B 对齐（host 侧 mark_compact_dynamic 的 assumed_align=16 承诺的就是这件事）。
        assert hidden_size % self.atom_v == 0, (f"hidden_size={hidden_size} must be a multiple of "
                                                f"atom_v={self.atom_v}")
        # 【循环上界】num_atoms = hidden/8 是「一行有多少个 128-bit 块」；
        # num_tiles = ceil(num_atoms/256) 是「一个 CTA 要几轮才能盖住一行」。
        # 两者都作为 constexpr 进入 kernel：tile 循环是定长展开，没有运行期计数，
        # 也没有「每轮重算边界」的开销。
        # 代价：hidden 一变就重编译（所以 hidden 是 host 侧变体键的一部分）。
        num_atoms = hidden_size // self.atom_v
        num_tiles = (num_atoms + self.num_threads - 1) // self.num_threads
        # 【launch 形状】
        #   grid  = (num_tokens, 1, 1) —— 一个 CTA 一个 token；
        #   block = (256, 1, 1)。
        # 注意 grid 第一维是 token 数，而不是 token 数 × topk：topk 那一维被折进 CTA
        # 内部的循环里 —— 这正是「同一 token 的所有贡献落在同一 CTA」的实现方式。
        self.kernel(
            staging,
            topk_indices,
            output,
            hidden_size,
            topk,
            num_experts,
            num_tokens,
            num_tiles,
            num_atoms,
        ).launch(
            grid=(num_tokens, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    # 【kernel 本体】参数与 __call__ 同构，多了 num_tiles / num_atoms 两个 constexpr。
    # 整段没有 __syncthreads、没有跨 CTA 同步、没有任何 fence：每个线程只碰自己那
    # 8 个元素，线程之间零共享 —— 所以也就不存在「谁先谁后」的问题。
    @cute.kernel
    def kernel(
        self,
        staging: cute.Tensor,
        topk_indices: cute.Tensor,
        output: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        num_experts: cutlass.Constexpr[int],
        num_tokens: cutlass.Int32,
        num_tiles: cutlass.Constexpr[int],
        num_atoms: cutlass.Constexpr[int],
    ):
        # 一维化取 id：block 索引直接就是 token 索引。这个「bidx == token」的静态绑定
        # 是确定性的第二个来源：token 到 CTA 的映射不依赖任何运行期调度决策。
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        token_idx = bidx

        atom_v: cutlass.Constexpr[int] = self.atom_v
        num_threads: cutlass.Constexpr[int] = self.num_threads

        # 【访存形状】CopyUniversalOp + num_bits_per_copy=128 造拷贝原子：一次 cute.copy
        # 就是一条 LDG.E.128 / STG.E.128，正好覆盖一个线程的 8 个 half。
        # 换 32-bit 原子 → 同样的数据要 4 条指令；换 256-bit → 硬件没有这个宽度。
        # 128 b atoms map to one LDG.E.128 / STG.E.128 per thread.
        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            staging.element_type,
            num_bits_per_copy=128,
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            output.element_type,
            num_bits_per_copy=128,
        )

        # 【把索引提进寄存器】topk 是 constexpr，所以 my_topk 展开成 topk 个寄存器变量，
        # 不是一块内存。读取放在循环外：CTA 内 256 个线程读的是同一行 topk_indices，
        # 一次 L1 命中就够。若放进内层循环，就变成 topk 次重复读，还把索引依赖压进
        # 热点路径。
        # Cache topk indices in registers; all threads in the CTA share
        # one L1 line so the HBM cost is negligible.
        my_topk = cute.make_fragment(topk, cutlass.Int32)
        for k in cutlass.range_constexpr(topk):
            my_topk[k] = topk_indices[token_idx, k]

        # zipped_divide by (1, atom_v): mode-0 = atom shape (1, atom_v),
        # mode-1 = (num_rows, atoms_per_row). Slicing (0, None) collapses
        # the M-atom mode to a 1D (atom_v,) view consumed directly by
        # cute.copy.
        # 【分块方式】zipped_divide(tensor, (1, atom_v)) 把 (rows, hidden) 重排成：
        #   mode-0 = atom 形状 (1, 8)      —— 「一个块长什么样」
        #   mode-1 = (rows, atoms_per_row) —— 「块怎么排」
        # 再取 (0, None) 把 mode-0 塌成一维 (8,)，得到 cute.copy 直接能吃的 1D 视图。
        # 等价说法：把矩阵看成「rows × hidden/8 个 8 元素块」的数组，块内连续。
        # 这一步把「128-bit 对齐」从口头约定变成类型系统里的保证 —— 只要 hidden%8==0、
        # 行起点 16 B 对齐，每个块地址天然 16 B 对齐，不会跨行。
        g_out = cute.zipped_divide(output, (1, atom_v))
        g_stage = cute.zipped_divide(staging, (1, atom_v))

        # 寄存器预算（每线程）：
        #   tCrC / tCrAcc 各 8 个元素；
        #   tCrR_per_k 是 topk 个 8 元素块 —— topk=8 时就是 8×8×2 B = 128 B/线程。
        # 这块寄存器是「先全部 load、再累加」的代价，也是 MLP（memory-level parallelism，
        # 访存并行度）的来源：topk 条 LDG 同时在飞，HBM 延迟只付一次。
        atom_frag_shape = (atom_v, )
        tCrC = cute.make_fragment(atom_frag_shape, output.element_type)
        tCrAcc = cute.make_fragment(atom_frag_shape, cutlass.Float32)

        tCrR_per_k = [cute.make_fragment(atom_frag_shape, staging.element_type) for _ in range(topk)]

        # 【tile 循环】上界 num_tiles 是 constexpr → 整段编译期展开，没有循环开销。
        # 每轮：一个线程负责一个 atom（8 个元素），跨 topk 行做一次定序求和。
        for tile_idx in cutlass.range_constexpr(num_tiles):
            my_atom_idx = tile_idx * num_threads + tidx

            # 尾部保护：num_tiles 是向上取整来的，最后一个 tile 只有部分线程有活。
            # 去掉 → 越界读 staging、越界写 output，而且发生在最后一行附近，容易被忽略。
            if my_atom_idx < num_atoms:
                # 【先全 load，再累加】两个独立循环，不是一个融合循环：先把 topk 条 128-bit
                # load 全部发出去（topk 条 LDG 同时在飞，HBM 延迟被并行覆盖），再从寄存器按 k 序
                # 累加。若融合成一个循环，每次迭代都要等上一行 load 返回，延迟暴露 topk 倍。
                # 注意：这个重排只改变「加载时机」，不改变「求和顺序」—— 确定性不受影响。
                for k in cutlass.range_constexpr(topk):
                    expert_idx = my_topk[k]
                    # 【丢弃哨兵】topk_indices 里被丢掉的槽位写的是 num_experts（host 侧预置），
                    # 这里用它当哨兵：不 load、不累加。两个循环用同一个判据，于是
                    #   * 该槽的 tCrR_per_k[k] 从未被读（保持未初始化也不会进累加器）；
                    #   * 全部被丢的 token 得到 0 —— 与「权重为 0」的语义一致。
                    # 这也是 combine 侧敢于不清 staging 的原因：陈旧字节永远走不到累加器。
                    if expert_idx < num_experts:
                        # 行号 = token*topk + k：token-major、k 次之。这个布局由 combine epilogue 的
                        # dst_slot = src_token_idx*topk + src_topk_idx 决定，两侧必须逐字一致。
                        # 好处：同一 token 的 topk 行在 staging 里连续，本 CTA 读的是连续一段。
                        stage_row_idx = token_idx * topk + k
                        cute.copy(
                            load_atom,
                            g_stage[(0, None), (stage_row_idx, my_atom_idx)],
                            tCrR_per_k[k],
                        )

                # 【确定性核心】第二个循环只做加法：
                #   * 顺序固定：k = 0,1,...,topk-1，与串行参考实现同序；
                #   * 精度固定：中间结果全程 FP32（bf16 先 .to(Float32)），不提前降精度；
                #   * 归属固定：一个输出元素的全部 topk 个贡献由**同一个线程**累加，
                #     没有跨线程 / 跨 CTA 的 partial sum，也就没有「谁先加」的问题。
                # 于是「固定求和顺序 + 同一累加器」= 与串行执行逐位一致（bitwise identical）。
                # 【对比 atomicAdd 归约】用原子加把各 rank 的贡献打到 output 上：求和顺序由硬件
                # 到达顺序决定，每次跑都可能不同；FP32 加法不满足结合律 → 低位随机抖动。
                # 训练里 weight gradient 的累加顺序一变，两次「同样输入」的 run 就对不上 ——
                # 这才是生产框架不敢直接上 overlap 的真实原因。
                # 【代价】一个 token 只能一个 CTA 算，没有跨 CTA 并行；但本 kernel 本来就是带宽
                # 受限（每个输入元素只读一次、输出只写一次，已经贴着理论下界），并行度不足
                # 不构成瓶颈 —— 确定性在这里是「免费」的。
                tCrAcc.fill(0.0)
                for k in cutlass.range_constexpr(topk):
                    expert_idx = my_topk[k]
                    if expert_idx < num_experts:
                        tCrAcc.store(tCrAcc.load() + tCrR_per_k[k].load().to(cutlass.Float32))

                # 只在最后做一次 dtype 转换（FP32 → output dtype），整条链只有一次舍入。
                # 若每个 k 都转一次再累加，误差会随 topk 放大，而且不同实现之间无法对齐。
                tCrC.store(tCrAcc.load().to(output.element_type))
                cute.copy(
                    store_atom,
                    tCrC,
                    g_out[(0, None), (token_idx, my_atom_idx)],
                )
