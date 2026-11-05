//===- epilogue_evt_nvshmem.hpp ----------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
// Some code from cutlass/epilogue/threadblock/fusion/visitor_store.hpp
// in NVIDIA cutlass project
// Original license as follows
/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
  \brief Visitor tree store operations for the CUTLASS 2x epilogue
*/

// ---------------------------------------------------------------------------
// 【上游】bytedance/flux（comet 分支，commit 7f98c8a），路径 src/reduce_scatter/epilogue_evt_nvshmem.hpp
// 【一句话】GEMM+ReduceScatter 的跨节点 epilogue：同机先用 red.global 把各 rank 的部分和累加到 owner 的
//           fp32 buffer，owner 等齐 local_world_size 份后压回 bf16、再用 nvshmem put 跨机发出；
//           不另发一轮 RS kernel。
// 文件主体是一个 CUTLASS EVT（epilogue visitor tree，epilogue 访问者树）节点，形状和 CUTLASS 的
// VisitorAuxStore 一致：begin_step / visit / end_step，外加一个 end_epilogue。
// 本注释只讲三件事：
//   1) visit / end_step / end_epilogue 各在 tile 生命周期的哪个位置、各写哪块内存；
//   2) rank → buffer 指针的映射（scatter_ptr_aux / barrier_ptr_aux / local_rank_start / dst_rank）
//      以及 kMaxWorldSize 这类编译期上界的代价；
//   3) 为什么「边算边发」比「kernel 结束后再发一轮 RS」快，代价又是什么。
// 阅读顺序：Arguments/Params（指针怎么进来）→ end_step（本地怎么攒）→ end_epilogue（怎么发信号、谁等）
//           → get_callbacks（坐标和指针怎么算）。前两段不看也能读 end_epilogue，反过来不行。
// 术语：EVT = epilogue visitor tree；nbi = non-blocking immediate（发起即返回、不等完成）；
//       PE = NVSHMEM 的 processing element，编号即 rank；local_world_size = 一个节点内的卡数。
// ---------------------------------------------------------------------------
#pragma once

#include "cutlass/epilogue/threadblock/fusion/visitor_2x.hpp"
#include "cutlass/barrier.h"
#include "flux/cuda/memory_utils.hpp"
#ifdef FLUX_SHM_USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;
using X = Underscore;
using SystemBarrier = cutlass::detail::SystemBarrier;
using bytedance::flux::kMaxWorldSize;
// 【这里用的是 sys scope 版本的 barrier】flag 会被对端 GPU 上的原子加改写，只有 sys scope 的
// acquire/release 才能保证「看到计数就一定看得到数据」（见 include/flux/cuda/system_barrier.hpp）。
// cutlass::Barrier 是 gpu scope 的，只在「flag 的读写都发生在同一张卡」时够用 —— 那正是同目录下
// epilogue_evt.hpp（节点内版本）与本文件（跨节点版本）的分界线之一。

/////////////////////////////////////////////////////////////////////////////////////////////////

// 【它替换掉了 CUTLASS 的哪个 visitor】原本的 VisitorAuxStore 是「算完就把 fragment 写回我这段 gmem」；
// 这里多走一步：写回的目标是「这段输出的 owner」的累积 buffer，且写的方式是 red（归约加法）而不是 store。
// 于是 GEMM 的 epilogue 同时完成了 RS 的两个动作之一：
//   reduce  —— end_step 里靠跨卡 red 完成（节点内 8 张卡都往同一地址加）；
//   scatter —— end_epilogue 里靠一次 put 把归约好的整段送到远端节点。
// 【FuseReduction】true = epilogue 里直接做跨 rank 的原子累加（half/bf16 走 red）；
//   false = 只写自己的部分和，归约交给别的阶段。两种模式对 buffer 大小、类型、对齐的要求都不同。
// 【为什么要单独一版】节点内可以靠 P2P load/store 直接读写对端显存；跨节点只有 NVSHMEM 的 put/get，
//   nbi 的完成性、可见性语义和普通 store 完全不同 —— 这就是两个 epilogue 文件的分叉点。
template <
    class ThreadMap,
    class Element,
    FloatRoundStyle RoundStyle,
    class StrideMNL,
    class ThreadblockShape,
    bool FuseReduction>
struct VisitorAuxStoreScatterAccrossNode {
// 【Arguments vs Params】Arguments 是 host 侧塞进来的「指针的指针」（Element** / T**，每个 rank 一项），
// Params 是压平后的「指针数组 + 长度」。压平有两个理由：
//   1) kernel 参数里的数组长度必须是编译期常量 → 用 kMaxWorldSize（flux.h 里 = 32）开固定数组；
//   2) device 侧不用二次解引用，params.scatter_ptr_aux[i] 一次取到。
// 代价：数组只有前 world_size 项被赋值，其余是未初始化的野指针 —— 越界索引不会报错，只会写到飞。
//   【规则】所有以 rank 为下标的访问都必须 rank < world_size，且 rank 的含义要和 host 侧填表时一致。
// 【一个能说明代码在演进的细节】nnodes 一路传进了 Params，但下面 get_callbacks 里没人用它 ——
//   映射是自己按 world_size / local_world_size 现算的。读这套代码时以「实际用到的字段」为准。
  struct Arguments {
    Element **scatter_ptr_aux;
    StrideMNL dAux = {};
    int64_t rank = 0;
    int64_t world_size = 1;
    SystemBarrier::T **barrier_ptr_aux;
    Element *remote_reduce_buffer;
    int64_t nnodes = 1;
  };

  struct Params {
    Element *scatter_ptr_aux[kMaxWorldSize];
    StrideMNL dAux;
    int64_t rank;
    int64_t world_size;
    SystemBarrier::T *barrier_ptr_aux[kMaxWorldSize];
    Element *remote_reduce_buffer;
    int64_t nnodes;
  };

  template <class ProblemShape>
  static constexpr Params
// 【host → device 的唯一入口】逐项拷贝，循环上界是 args.world_size 而不是 kMaxWorldSize，
// 剩下的槽位保持未初始化。这就是 kMaxWorldSize 只能当「上界」、不能当「实际长度」用的原因。
  to_underlying_arguments(
      ProblemShape const &problem_shape, Arguments const &args, void *workspace) {
    Params params;
    params.dAux = args.dAux;
    params.rank = args.rank;
    params.world_size = args.world_size;
    params.remote_reduce_buffer = args.remote_reduce_buffer;
    params.nnodes = args.nnodes;
    for (int i = 0; i < args.world_size; i++) {
      params.scatter_ptr_aux[i] = args.scatter_ptr_aux[i];
      params.barrier_ptr_aux[i] = args.barrier_ptr_aux[i];
    }
    return params;
  }

  struct SharedStorage {};

// 【向量化宽度】vec_bits = 每线程每次 access 的元素数 × 元素位宽；VecType 取 min(128, vec_bits) 位。
// 【为什么 bf16 + FuseReduction 被摁到 vec_bits == 16】bf16 的归约要走 fp32 缓冲（end_step 里地址 ×2），
//   一次 access 若跨多个 bf16，「源地址 → fp32 地址」就不再是简单的 2 倍偏移，会错位；
//   所以这种组合被 static_assert 强行压到「一次一个 bf16」。fp16 归约、非归约都不受这个限制。
// 去掉这个断言的后果：能编译、能跑，只是数值悄悄错 —— 这类模板约束看着烦，删起来最贵。
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  static bool constexpr vec_bits_valid =
      (std::is_same_v<Element, cutlass::bfloat16_t> && FuseReduction && vec_bits == 16) or
      (!FuseReduction) or (!std::is_same_v<Element, cutlass::bfloat16_t>);
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static_assert(vec_bits_valid);
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
// 【visitor 本身几乎不占空间】只存一个 Params const*；Params 住在 kernel 参数区，生命周期覆盖整个 kernel，
// 所以这个指针永远有效。visitor 只当工厂用，每个 CTA 现场构造一份 Callbacks（见下）。
// SharedStorage 是空的：这个 visitor 不占 shared memory，等待和计数全走 gmem 里的 flag。
  VisitorAuxStoreScatterAccrossNode() {}

  CUTLASS_HOST_DEVICE
  VisitorAuxStoreScatterAccrossNode(Params const &params, SharedStorage const &shared_storage)
      : params_ptr(&params) {
    // if constexpr (std::is_same_v<Element, cutlass::bfloat16_t> and FuseReduction == true) {
    //   static_assert(vec_bits == 16);
    // }
  }

  Params const *params_ptr;

// 【workspace 大小】flag 数 = tile 总数（m、n 两个方向各自向上取整），一个 tile 一个 flag，再按 128B 对齐。
//   128B 对齐是为了让每个 tile 的 flag 独占一条 cacheline：相邻 tile 的到达互不打脏，
//   也不会出现「一个 CTA 在 spin 自己的 tile，把邻居那条线一直读成 invalid」的假共享。
// 这个函数是 host/device 共用的：host 按它分配，device 按同一公式索引 —— 两边必须用同一个
// ThreadblockShape，形状不一致就是越界写。
  /// Pad the given allocation size up to the nearest cache line
  static size_t
  cacheline_align_up(size_t size) {
    static const int CACHELINE_SIZE = 128;
    return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
  }

  /// Get the workspace size needed for barrier
  //   template <class ThreadblockShape>
  size_t
  get_barrier_workspace_size(int m, int n) const {
    // each threadblock tile needs a synchronization flag.
    int num_flags = ((m + ThreadblockShape::kM - 1) / ThreadblockShape::kM) *
                    ((n + ThreadblockShape::kN - 1) / ThreadblockShape::kN);

    return cacheline_align_up(sizeof(typename SystemBarrier::T) * num_flags);
  }

// 【Callbacks = 一个 CTA 的 tile 上下文】CUTLASS 在 epilogue 里按 tile 构造一次，这里把
// 「这个 tile 是谁、写到哪、等谁」全部提前算好存成成员：
//   tC_gAux   本地累积 buffer（这段输出 owner 的那块）的 gmem 视图
//   tC_rAux   寄存器里的暂存 fragment：visit 写它，end_step 读它
//   tR_gAux   远端 buffer 的 gmem 视图（跨节点那一跳的目的地）
//   tile_idx / remote_tile_idx  本地 / 远端的一维 tile 下标，用来索引 flag 和 buffer
//   local_rank / local_dst_rank / remote_dst_rank  rank 映射的三个结果（见 get_callbacks）
// 构造参数有 18 个，但都是「算一次、用很多次」的量：放进成员是为了在 visit/end_step 的热路径上
// 不再做除法和取模（32 位除法在 GPU 上很贵，尤其每个 fragment 都做一次的时候）。
  template <class GTensor, class RTensor, class GTensor_R, class ProblemShape, class BlockShapeMNK>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
        GTensor &&tC_gAux,
        RTensor &&tC_rAux,
        GTensor_R &&tR_gAux,
        ProblemShape problem_shape,
        BlockShapeMNK blk_shape_mnk,
        Params const *params_ptr,
        SystemBarrier::T *barrier_ptr,
        void *ptr_local_dst,
        void *ptr_R,
        int thread_idx,
        int tile_idx,
        int remote_tile_idx,
        int local_rank,
        int local_dst_rank,
        int remote_dst_rank,
        int local_world_size,
        int m_coord,
        int n_coord,
        int new_m_coord)
        : tC_gAux(cute::forward<GTensor>(tC_gAux)),
          tC_rAux(cute::forward<RTensor>(tC_rAux)),
          tR_gAux(cute::forward<GTensor_R>(tR_gAux)),
          problem_shape(problem_shape),
          blk_shape_mnk(blk_shape_mnk),
          params_ptr(params_ptr),
          barrier_ptr(barrier_ptr),
          ptr_local_dst(ptr_local_dst),
          ptr_R(ptr_R),
          thread_idx(thread_idx),
          tile_idx(tile_idx),
          remote_tile_idx(remote_tile_idx),
          local_rank(local_rank),
          local_dst_rank(local_dst_rank),
          remote_dst_rank(remote_dst_rank),
          local_world_size(local_world_size),
          m_coord(m_coord),
          n_coord(n_coord),
          new_m_coord(new_m_coord) {}

    GTensor tC_gAux;
    RTensor tC_rAux;
    GTensor_R tR_gAux;
    Params const *params_ptr;
    ProblemShape problem_shape;
    BlockShapeMNK blk_shape_mnk;
    SystemBarrier::T *barrier_ptr;
    int thread_idx;
    int tile_idx;
    int remote_tile_idx;
    int local_rank;
    int local_dst_rank;
    int remote_dst_rank;
    int local_world_size;
    int m_coord;
    int n_coord;
    int new_m_coord;
    void *ptr_R;
    void *ptr_local_dst;
// 【step 的生命周期】一个 step ≈ 输出的一片（K 循环之外的输出分片）：begin_step 清暂存，
// visit 逐 fragment 把 accum 转成输出类型塞进 tC_rAux，end_step 一次性写出去。
// 为什么要 clear：tC_rAux 复用同一块寄存器/局部空间，不清就会把上一个 step 的残值写进结果；
// 尾部 fragment 不满时尤其明显 —— 清空是「没算到的位置不脏」的唯一保证。
    CUTLASS_DEVICE void
    begin_step(int step_idx) {
      clear(tC_rAux);
    }

// 【visit：只搬进寄存器，不碰 gmem】把累加器 fragment 转成 Element 存进 tC_rAux 的对应位置，
// 返回值仍是 frg_input（CUTLASS 的链式约定，让后面的 visitor 继续用）。
// 为什么不在 visit 里直接写 gmem：一个 tile 的 epilogue 会被切成很多 fragment，
// 每个 fragment 写一次就是一次小事务；攒到 end_step 再按 128bit 向量整块写/整块 red，事务数差一个量级。
    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto  // returns an Array
    visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frg_idx,
        Array<ElementAccumulator, FragmentSize> const &frg_acc,
        Array<ElementInput, FragmentSize> const &frg_input) {
      using ConvertInput = NumericArrayConverter<Element, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux));
      tC_rAux_frg(frg_idx) = convert_input(frg_input);

      return frg_input;
    }
// 【fp32 缓冲 → bf16 的原地压缩】bf16 归约是在 fp32 缓冲上做的（见 end_step 的 ×2 偏移），
// 发出去之前要压回 bf16，而且只能原地压：输出只占输入的一半，读和写的区间重叠。
// 于是每一轮「读 float2 → 转 bf16 → __syncthreads() → 写 bf16」中间必须有同步：
//   先写的线程若不等，会踩掉后读线程还没读的输入。轮数 = count/2/128，128 个线程搬完整个 tile。
// 代价：每轮一次全 CTA 同步（tile 越大轮数越多）；收益：省掉一块和输出同样大的临时 buffer。
// assert(blockDim.x == n_threads) 把线程数写死成 128 —— 换 block 形状要同步改这里。
    template <const int count, const int n_threads>
    CUTLASS_DEVICE void
    recast_backto_bf16(void *ptr) {
      // printf("blockDimx:%d\n", blockDim.x);
      assert(blockDim.x == n_threads);
      constexpr int n_per_access = 2;
      const int tid = threadIdx.x;
      constexpr int n_iter = count / n_per_access / n_threads;
      float2 *p_f = (float2 *)ptr;
      __nv_bfloat162 *p_b = (__nv_bfloat162 *)ptr;
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < n_iter; iter++) {
        float2 val = *(p_f + iter * n_threads + tid);
        __nv_bfloat162 new_v = __float22bfloat162_rn(val);
        __syncthreads();
        *(p_b + iter * n_threads + tid) = new_v;
      }
    }
// 【end_step：把这一 step 的部分和直接累加进「这段输出 owner 的累积 buffer」】
//   注意目的地是 tC_gAux（local_dst_rank 的 buffer），不是自己的输出：节点内 8 张卡都往同一地址 red，
//   归约在「写」的过程中就完成了 —— 这是「把 RS 的 reduce 融进 epilogue」最直接的一处。
// 三个分支，语义完全不同：
//   half + FuseReduction → global_red：对 gmem 做「只加不读」的原子累加（VecType 宽度的 red 指令）。
//   bf16 + FuseReduction → 没有 bf16 的原子加，先转 fp32 再 red；目标 buffer 因此是 fp32 的 2 倍大，
//                          offset×2 就是「同一逻辑位置在 fp32 缓冲里的地址」。
//   其它                 → 普通 store（不融合归约时只是搬运）。
// 为什么用 red 而不是 load-add-store：不用把旧值读回来，省一半流量，也不会和别的 rank 的写互相打架；
//   代价是累加顺序不确定 —— 浮点加法不满足结合律，同样的输入两次跑出的末位可能不同。
// 【注意】跨节点的 put 要到 end_epilogue 才发，不是每个 step 发一次：一个 tile 的所有 step 都结束后
//   才值得发，发得早 = 发多次 = 多份流量。
    CUTLASS_DEVICE void
    end_step(int step_idx) {
      auto src_v = filter(tC_rAux);

      auto dst_v = filter(tC_gAux(_, _, _, step_idx));
      if constexpr (FuseReduction and std::is_same_v<Element, cutlass::half_t>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src_v); ++i) {
          cutlass::arch::global_red<VecType, sizeof(VecType), half_t>(
              reinterpret_cast<VecType const &>(src_v(i)), (void *)&dst_v(i), 1);
        }
      } else if constexpr (FuseReduction and std::is_same_v<Element, cutlass::bfloat16_t>) {
        // use float to perform the reduction
        // Note: this need to double the size of the output buffer
        using Red_Vec = uint_bit_t<32>;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src_v); ++i) {
          __nv_bfloat16 val = *reinterpret_cast<__nv_bfloat16 *>((void *)&src_v(i));
          // convert to the float32
          float float32 = __bfloat162float(val);

          int offset = (char *)&dst_v(i) - (char *)ptr_local_dst;
          char *target_ptr = (char *)ptr_local_dst + 2 * offset;
          cutlass::arch::global_red<Red_Vec, sizeof(Red_Vec), float>(
              reinterpret_cast<Red_Vec const &>(float32), (void *)target_ptr, 1);
        }
      } else {
        // just copy to the corresponding position
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src_v); ++i) {
          dst_v[i] = src_v[i];
          // cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(i), (void*)&dst_v(i), 1);
        }
      }
    }

// 【put 用 nbi】nvshmemx_putmem_nbi_block = non-blocking immediate，整 CTA 参与：
//   函数返回只代表「请求已发出」，不代表数据到了对端，更不代表对端可见。
// 完成性在这里不靠 nvshmem_quiet：接收侧靠 flag 计数判断到齐，源侧在 +1 之前插了 fence.acq_rel.sys
//   给 put 定序（见 end_epilogue），于是「计数到了」被当作「数据到了」。严格说 nbi 的完成属于
//   NVSHMEM 的语义（通常配 nvshmem_quiet / nvshmem_fence），这里没有显式调用 —— 把它当成
//   「靠 fence + 计数兜底」的实现细节，而不是可移植的保证。
// 为什么用 block 版本而不是 warp/thread 版本：整个 tile 是整 CTA 产出的，一次 put 更好合并、指令更少。
// 没有 FLUX_SHM_USE_NVSHMEM 时这个函数是空实现 —— 编译能过、跑起来不报错，只是跨节点那一跳静默消失；
//   排查「数值不对」时先确认这个宏。
    CUTLASS_DEVICE void
    copy_remote(void *dst, void *src, int size, int dst_rank) {
#ifdef FLUX_SHM_USE_NVSHMEM
// 四个参数：(目的地址, 源地址, 字节数, 目的 PE)。目的地址由调用方算好，这里不做地址翻译。
      nvshmemx_putmem_nbi_block(dst, src, size, dst_rank);
#endif
    }
// 【end_epilogue：这个 tile 的收尾，也是整个 overlap 的汇合点】三步，每一步的「为什么」都不一样：
//   1) 所有 CTA 都做：thread 0 先 fence.acq_rel.sys，再 atomicAdd_system(barrier_ptr + tile_idx, 1)。
//      fence 必须在原子之前：put/red 都是 relaxed 的，不加 fence 的话「计数 +1」可能先于「数据到达」
//      被对端看到，接收侧就会在数据没齐时开始读。
//   2) 只有 owner 才等：local_dst_rank == local_rank 的那个 CTA 调 wait_eq，等本节点 local_world_size 个到达。
//      非 owner 的 CTA 到这儿就返回，接着去算下一个 tile —— 【这就是 overlap】：
//      等待的成本只落在「这段输出的 owner」身上，而它等的时间被其它 CTA 的计算填满。
//   3) owner 把归约好的这段推给远端节点：bf16 先原地压回 bf16，再 put 到 remote_dst_rank 的
//      remote_reduce_buffer（按 remote_tile_idx 定位）。
    CUTLASS_DEVICE void
    end_epilogue() {
      if constexpr (FuseReduction) {
        // if fuse reduction, we use nvshmem directly send the reduced output
        using ReduceType =
            std::conditional_t<std::is_same_v<Element, cutlass::half_t>, half, float>;
        if (threadIdx.x == 0) {
          asm volatile(
              "fence.acq_rel.sys;\n");  // better performance compared to other kinds of fence
// 【after_inc 取了就丢】用 atomicAdd 而不是 red，意味着白等一次返回值；真不需要「我是不是最后一名」
// 的话 red 更便宜。读起来像是从「需要这个返回值」的版本改过来的。
// 【下面那条 __syncthreads】它在计数之后，只能保证「CTA 齐步进入 owner 分支」，
// 不构成「数据已写完」的保证 —— 那个保证得来自 fence 之前的同步；本文件里看不到这一步，
// 取决于 epilogue 框架在 end_step 与 end_epilogue 之间是否插过 __syncthreads，照抄时是最该补的地方。
          int after_inc = atomicAdd_system(barrier_ptr + tile_idx, 1);
        }
        __syncthreads();
        // SystemBarrier::arrive_inc(barrier_ptr_local_dst, threadIdx.x, tile_idx);
        auto BM = get<0>(blk_shape_mnk);
        auto BN = get<1>(blk_shape_mnk);
        // wait for all local reduce done

        if (local_dst_rank == local_rank) {
// 【整个 kernel 唯一被显式串行化的地方】owner 在这里等，等多久取决于「同一段的其他 7 张卡
// 多快算完同一个 tile」—— 这正是 Flux 要用 tile swizzling 去对齐「产出顺序」和「到达顺序」的原因。
          SystemBarrier::wait_eq(
              barrier_ptr,
              threadIdx.x,
              tile_idx,
              local_world_size);  // double check here
          void *src_ptr = static_cast<void *>(tile_idx * (BM * BN) + (ReduceType *)ptr_local_dst);
          void *dst_ptr = static_cast<void *>(remote_tile_idx * (BM * BN) + (half *)ptr_R);
          if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
            constexpr int N_COUNT = BM * BN;
            recast_backto_bf16<N_COUNT, 128>(src_ptr);
          }
          copy_remote(dst_ptr, src_ptr, BM * BN * 2, remote_dst_rank);
        }
      }
// 【为什么这样比「kernel 结束后再发一轮 RS」快】
//   传统两段式：GEMM kernel 全部 CTA 退出 → 全局同步 → 另一个 RS kernel 把 M×N 的部分和读一遍、归约、
//   写 M/world_size 行。这段时间 SM 全空，而且刚写出去的数据要从 HBM 再读回来一遍。
//   这里：tile 一算完，部分和还在寄存器/L2 里就被 red 进了 owner 的 buffer（省一次 HBM 往返）；
//   「归约」被摊到 GEMM 的整个生命周期，「散射」只剩 owner 的一次 put。通信不再是末尾一个串行段，
//   而是跟着 tile 的产出节奏走 —— GEMM 的 tile 产出本身就是天然的分片器，既不用额外切分，
//   也不用「留几个 SM 给通信」这种预算（Flux 的立场是「融」而不是「让」）。
//
// 【代价，别只看收益】
//   - 等待仍然是「整 CTA 停摆」：wait_eq 内部是 thread 0 自旋 + __syncthreads，其余线程被挂在 barrier 上，
//     不会去干别的活。能救的只有顺序（tile swizzling 让同一段先产出），不是让线程让路。
//   - 每个 tile 一次 sys-scope 原子 + 一次跨节点 put：tile 越细，overlap 越平滑，原子和 put 越多。
//   - flag 数组随 tile 数线性增长（每 tile 128B），host 要按 get_barrier_workspace_size 分配。
//   - 【这里的计数是 local_world_size（= 8），不是 world_size】上游自己写着 "double check here"：
//     节点内先凑 8 份，跨节点的 n_nodes 份由下一跳再叠。读这段代码时最该盯的就是这个数 ——
//     它定义了什么算「到齐」，写错不会崩，只会读到没归约完的数据。
    }
  };

// 【get_callbacks：把 (CTA 的 tile 坐标) 翻译成 (指针, 下标)】这是整个文件里唯一做「拓扑算术」的地方，
// 也是读 Flux 的 RS 最容易迷路的地方 —— 建议先把 rank 映射画出来再读代码：
//   全局 rank r → 节点 node = r / 8，节点内位置 local_rank = r % 8（local_world_size 在这里写死 8）。
//   输出沿 M 切成 world_size 段，第 k 段归 rank k；一个 tile 的 m 坐标决定它属于哪一段。
// 由此派生出三组下标：本地 owner（同节点）、远端 owner（跨节点）、以及本 rank 自己的位置。
// 【写死的三处】local_world_size = 8、BM = BN = 128、BlockShapeMNK = <128,128>：
//   它们都没走参数（注释里带着 should be passed / FIXME），换机器或换 tile 形状必须同时改这里和 host 侧，
//   否则就是「能跑、但算错」—— 这类硬编码是读这套代码时要先记下来的边界。
// 另外：world_size / rank 取自 nvshmem_n_pes() / nvshmem_my_pe()，不是 torch rank ——
//   NVSHMEM 的 PE 编号必须和 buffer 注册的顺序一致，接错一次后面所有映射全错。
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
      gemm::GemmCoord threadblock_tile_offset, int thread_idx, ProblemShape problem_shape) {
    // TODO: support RMPad for the accross node RS in the future
    constexpr int local_world_size = 8;  // should be passed from parameters
    // constexpr int tile_size = ThreadblockShape::kM;
    constexpr int tile_size [[maybe_unused]] = ThreadblockShape::kM;
#ifdef FLUX_SHM_USE_NVSHMEM
    const int world_size = nvshmem_n_pes();
    const int rank = nvshmem_my_pe();
#else
    // only support InterNode when nvshmem is enabled
// 【没开 NVSHMEM 的编译分支】world_size / rank 直接给 -1：编译能过，但下面所有映射都会算出负数下标。
// 这个文件本来就是为「跨节点」写的，没有 NVSHMEM 时不该被实例化。
    const int world_size = -1;
    const int rank = -1;
#endif
    const int local_rank = rank % local_world_size;
    auto M = get<0>(problem_shape);
    auto N = get<1>(problem_shape);
    constexpr int BM = _128{};  // FIXME
    constexpr int BN = _128{};
    assert(BM == tile_size);
    using BlockShapeMNK = cute::Shape<_128, _128>;
    BlockShapeMNK blk_mnk = cute::make_shape(_128{}, _128{});

// 【一个 tile 的三重身份】
//   它是哪一段输出的一部分 → local_m_block_idx（0..world_size-1，等于这段的 owner rank）
//   它在节点内的 owner 是谁 → local_dst_rank = local_m_block_idx % 8
//   它在远端的 owner 是谁   → remote_dst_node = local_m_block_idx / 8
// 地基是 coord_per_rank_global = M/BM/world_size，也就是「每个 rank 在 M 上负责几个 tile」；
// 用它做整除，就把 m 坐标映射成了 owner 段号。
// （同一行附近的 coord_per_rank_local 算了但没人用 —— 节点内版本的残留，别被它带偏。）
    const int m_coord = threadblock_tile_offset.m();
    const int n_coord = threadblock_tile_offset.n();
    const int coord_per_rank_local = (M / BM / local_world_size);
    const int coord_per_rank_global = (M / BM / world_size);
    const int local_m_block_idx = m_coord / coord_per_rank_global;
// 【节点基址】local_rank_start = 本节点第一个 PE 的全局编号；把「节点内位置」加上它就是全局 PE 号。
// barrier_ptr_aux / scatter_ptr_aux 都是按全局 PE 号排的数组，索引必须加这个偏移。
// 【关键性质】同一个 tile 在所有 rank 上算出的 local_dst_rank 相同 → 所有 rank 指向同一个 owner 的 buffer，
// 这就是「对称」的来源；前提是 host 侧给每个 rank 都填了一份「全体 rank 的指针表」。
    const int local_rank_start = rank / local_world_size * local_world_size;
    int local_dst_rank = local_m_block_idx % local_world_size;
    const int n_nodes = world_size / local_world_size;
    const int nodeid = rank / local_world_size;
    const int remote_dst_node = local_m_block_idx / local_world_size;
// 【远端 owner】remote_dst_node × 8 + local_rank：用 local_rank 而不是 local_dst_rank 是等价的 ——
// 因为真正执行转发的就是 owner，而 owner 的守卫就是 local_rank == local_dst_rank（见 end_epilogue）。
// 所以这里读作「远端节点里、段号对 8 取模等于我这张卡位置的那个 rank」，也就是该段在远端的 owner。
    const int remote_dst_rank = remote_dst_node * local_world_size + local_rank;

// 【全局一致的 tile 线性下标】按行优先把 (m,n) 压成一维：N/BN 是每行的 tile 数。
// 所有 rank 用同一套公式，所以任何一个 rank 都能用这个下标定位「这个 tile 在 owner 的 flag 数组里的槽位」——
// 它是跨 rank 的「共同语言」；写成别的顺序（比如列优先）同步照样能跑，只是和 host 侧的布局假设脱钩。
    const int tile_idx =
        get<1>(problem_shape) / BN * threadblock_tile_offset.m() + threadblock_tile_offset.n();

// 【barrier_ptr 的归属】barrier_ptr_aux[local_dst_rank + local_rank_start] 就是 owner 的 flag 数组。
// 每个 CTA 都往这块上对自己的 tile_idx 做 atomicAdd；owner 在 end_epilogue 里等它涨到 local_world_size。
// 【这个三元判断只防了一半】它挡的是数组指针本身为空（给个 nullptr），没有检查元素是否为 nullptr；
// end_epilogue 里会直接解引用 —— host 侧必须保证每个 rank 的 barrier buffer 都已分配。
// 热路径代码里这种「防了一半」的写法很常见，读的时候要看清楚它到底防住了什么。
    SystemBarrier::T *barrier_ptr =
        params_ptr->barrier_ptr_aux
            ? params_ptr->barrier_ptr_aux[local_dst_rank + local_rank_start]
            : nullptr;
    // assume we got a full dst tensor
    auto local_dst_offset = threadblock_tile_offset;
    auto remote_dst_offset = threadblock_tile_offset;
// 【远端的坐标】new_m_coord 把「全局 m 坐标」换成「远端 buffer 里的 m 坐标」：
//   nodeid 是源节点号，coord_per_rank_global 是每 rank 的 tile 数
//   → 远端 buffer 按「源节点」分块，每个源节点贡献自己那份部分和，目的节点收齐 n_nodes 份后再叠。
//   remote_tile_idx = new_m_coord * N/BN + n_coord 就是这个分块里的一维下标。
// 为什么不直接搬全局 m 坐标：那样不同节点的部分和会落到同一格，要么互相覆盖、要么得额外维护一张
// 「谁写过」的表；按源节点分块把这个维度变成了地址的一部分，代价是远端 buffer 要按 n_nodes 倍开。
    const int new_m_coord = nodeid * coord_per_rank_global + m_coord % coord_per_rank_global;
    const int remote_tile_idx = new_m_coord * get<1>(problem_shape) / BN + n_coord;
    remote_dst_offset.m() = new_m_coord;
    auto ptr_R = params_ptr->remote_reduce_buffer;
    auto ptr_local_dst = params_ptr->scatter_ptr_aux[local_dst_rank + local_rank_start];

// 【gmem 视图】mAux 是本地累积 buffer，rAux 是远端 buffer，两者共用 problem_shape / dAux ——
// 也就是缓冲区逻辑上仍是 (M,N,L)，但只有它 owner 的那段坐标会被命中。
// 真正的「哪一行落到 buffer 的哪个偏移」由 ThreadMap::partition 的 (rank, world_size, n_nodes) 参数完成：
// 它负责把「全局 tile 坐标」翻译成「owner buffer 内的偏移」。本文件只负责把 offset 设对：
// local_dst_offset 用原坐标，remote_dst_offset.m() 换成 new_m_coord。
    Tensor mAux = make_tensor(
        make_gmem_ptr(ptr_local_dst),
        problem_shape,
        params_ptr->dAux);  // (M,N,L)

    Tensor rAux = make_tensor(make_gmem_ptr(ptr_R), problem_shape,
                              params_ptr->dAux);  // (M,N,L)

    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    static_assert(is_static<BlockShapeMNK>::value, "BlockShape MNK not static");
    Tensor tC_gAux = recast<VecType>(group_modes<3, 6>(ThreadMap::partition(
        mAux,
        problem_shape,
        blk_mnk,
        thread_idx,
        local_dst_offset,
        rank,
        world_size,
        n_nodes)));  // layout need to be updated
    Tensor tR_gAux = recast<VecType>(group_modes<3, 6>(ThreadMap::partition(
        rAux, problem_shape, blk_mnk, thread_idx, remote_dst_offset, rank, world_size, n_nodes)));

// 【寄存器暂存与 gmem 视图同形】tC_rAux 只取前 3 个 mode（不含 step 维），因为一个 step 的输出
// 正好是 gmem 视图切掉 step 之后的那一片；这样 visit 里的 recast/索引能和 end_step 一一对应。
    Tensor tC_rAux = make_tensor_like(take<0, 3>(tC_gAux));

    return Callbacks<
        decltype(tC_gAux),
        decltype(tC_rAux),
        decltype(tR_gAux),
        ProblemShape,
        BlockShapeMNK>(
        cute::move(tC_gAux),
        cute::move(tC_rAux),
        cute::move(tR_gAux),
        problem_shape,
        blk_mnk,
        params_ptr,
        barrier_ptr,
        ptr_local_dst,
        ptr_R,
        thread_idx,
        tile_idx,
        remote_tile_idx,
        local_rank,
        local_dst_rank,
        remote_dst_rank,
        local_world_size,
        m_coord,
        n_coord,
        new_m_coord);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// 【一页纸总结】
//   数据流：GEMM 累加器 → visit（转 Element，进寄存器）→ end_step（red 进 owner 的累积 buffer，节点内归约）
//           → end_epilogue（atomicAdd_system 计数 → owner wait_eq 到 8 → bf16 压回 → put 到远端节点）
//           → 目标节点收齐 n_nodes 份后做最后一跳归约（不在本文件）。
//   被 overlap 的三件事：节点内归约（red）、跨节点 put、以及别的 CTA 的 GEMM。
//   唯一被串行化的是 owner 的那次 wait；它的长度由「同一段其他卡多快算完同一 tile」决定，
//   这也是 Flux 用 tile swizzling 对齐「产出顺序」和「到达顺序」的原因。
//   想改这段代码，先想清楚三件事：计数到几算齐、fence 放在哪、以及谁有权清零。
}  // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
