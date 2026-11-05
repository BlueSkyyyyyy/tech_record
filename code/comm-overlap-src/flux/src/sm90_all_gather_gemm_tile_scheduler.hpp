//===- sm90_all_gather_gemm_tile_scheduler.hpp -------------------- C++ ---===//
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

// =================================================================================================
// 上游源码带注释副本（只插入了 // 注释行，代码逐行未改）
//   仓库 / 分支  bytedance/flux, comet 分支
//   提交         7f98c8a
//   路径         src/all_gather/sm90_all_gather_gemm_tile_scheduler.hpp
//   配套         src/all_gather/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp
//
// 一句话定位：AG-GEMM 的 tile 调度器。只重写「M_idx 怎么编号」这一件事，让每个 rank
// 先算自己本地那份 A 所在的 tile，把远端数据的到达时间藏进计算里。
//
// 本注释只讲三件事：
//   ① M_idx 重映射的两个动作（段内旋转 + 段间按 node 置换），以及它为什么仍是双射；
//   ② 为什么从 M_start = M_per_rank * rank 开始算，而不是从 0 开始（overlap 的全部来源）；
//   ③ 映射的隐含前提（M 方向块数要能在 node 间均分）。
//
// 建议阅读顺序：
//   1) L156 起 4 行：M_per_rank / M_start / offset —— 三个数定下「先算哪里」
//   2) L200 起 30 行：get_current_work 的三个分支（单机 / 2~4 node 查表 / >4 node 公式）
//   3) 回头读 L81 起的 Arguments / Params：上面那几个数就装在这些字段里
// =================================================================================================
#pragma once

#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"

#include "cute/util/debug.hpp"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/swizzle_layout.hpp"

#include "cutlass/barrier.h"
#include "flux/flux.h"
#include "flux/cuda/memory_utils.hpp"

#include "all_gather/all_gather_swizzle.hpp"

namespace cutlass::gemm::kernel::detail {
using SystemBarrier = cutlass::detail::SystemBarrier;
///////////////////////////////////////////////////////////////////////////////

// Persistent Thread Block (TB) scheduler
// 只继承 PersistentTileSchedulerSm90，基类的 raster order / L2 swizzle / 线性编号全部沿用。
// 真正的改动只有两处：to_underlying_arguments（算偏移参数）和 get_current_work（改 M_idx）。
// 【为什么不改基类】：基类的 swizzle 决定「同一时刻哪些 CTA 算哪些 tile」（并发形状），
// 而这里要改的是「谁先算」（顺序语义），只能在取 tile 的入口改。
// 也正因为如此，这个重映射对 kernel 是透明的：kernel 只看到「M_idx 是重映射后的块号」。
class Sm90AGKernelTileScheduler : public PersistentTileSchedulerSm90 {
 public:
  using Base = PersistentTileSchedulerSm90;
  using WorkTileInfo = Base::WorkTileInfo;

  // Arguments 是 host 传进来的拓扑信息；Params 是换算好后 device 端只读的版本：
  //   nnodes / local_world_size：把 world 切成「node 段」，每段一段连续的 M 行；
  //   node_id = rank / local_world_size：本 rank 落在第几段；
  //   problem_blocks_m_offset：本 rank 本地数据在 M 方向的起始块号（L156 处算出）；
  //   problem_blocks_m：M 方向总块数（cluster 粒度，来自 get_tiled_cta_shape_mnl）。
  // WorkIdxMLayout 在这里只是声明、本文件没用到——从 reduce-scatter 那套调度器带过来的残留。
  struct Arguments : Base::Arguments {
    int nnodes = 1;
    int rank = 0;
    int world_size = 1;
    int local_world_size = 1;
    int local_rank = 0;
    SystemBarrier::T *ptr_barrier = nullptr;
  };

  struct Params : public Base::Params {
    using WorkIdxMLayout = cute::Layout<
        cute::Shape<int32_t, int32_t, cute::Int<2>>,
        cute::Stride<cute::Int<1>, int32_t, cute::Int<0>>>;

    SystemBarrier::T *ptr_barrier = nullptr;
    int problem_blocks_m_offset = 0;
    int problem_blocks_m = 0;

    int nnodes = 1;
    int node_id = 0;
  };

  // Sink scheduler params as a member
  Params scheduler_params;

  //
  // Methods
  //

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      [[maybe_unused]] KernelHardwareInfo const &hw_info,
      Arguments const &arguments,
      [[maybe_unused]] void *workspace = nullptr,
      [[maybe_unused]] int epilogue_subtile = 1) {
    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    Base::Params base_params;
    base_params.initialize(
        problem_blocks,
        to_gemm_coord(cluster_shape),
        hw_info,
        arguments.max_swizzle_size,
        arguments.raster_order);

    int nnodes = arguments.nnodes;
    int rank = arguments.rank;
    int world_size = arguments.world_size;
    int local_world_size = arguments.local_world_size;
    FLUX_CHECK(local_world_size * nnodes == world_size);
    // int local_rank = arguments.local_rank;
    int node_id = rank / local_world_size;
    // calculate offsets on M
    int problem_blocks_m = problem_blocks.x;
    FLUX_CHECK(local_world_size != 0);

    // int problem_blocks_m_offset = problem_blocks_m / local_world_size * local_rank;
    // ★ M 重映射的全部参数就这 4 行 ★
    //   M_per_rank = M / world_size         // 每个 rank 本地那份 A 的行数（gather 之前）
    //   M_start    = M_per_rank * rank      // 这份数据在 gathered A 里的起始行
    //   offset     = ceil(M_start / TILE)   // 第一个「完全落在本地段内」的 tile 号
    // 【为什么用 ceil 不用 floor】：floor 会取到「骑在本地段边界上」的那个 tile，
    // 它还依赖远端 chunk，先算它等于白等一轮；ceil 保证第一个 tile 的每一行都在本地。
    // 【offset 就是「先算本地」这件事的全部实现】——只有 3 行，运行时零开销。
    // rank → node 的分解（node_id）也只在这里做一次，device 端不再碰 tp_group。
    // 隐含前提：M 能被 world_size 均分（AG 的定义如此），否则 M_per_rank 只是平均值。
    int M_per_rank = static_cast<int>(get<0>(problem_shape_mnkl)) / world_size;
    int M_start = M_per_rank * rank;
    int problem_blocks_m_offset = (M_start + get<0>(tile_shape) - 1) / get<0>(tile_shape);

    return {
        base_params,
        arguments.ptr_barrier,
        problem_blocks_m_offset,
        problem_blocks_m,
        nnodes,
        node_id};
  }

  CUTLASS_HOST_DEVICE
  Sm90AGKernelTileScheduler() {}

  CUTLASS_DEVICE explicit Sm90AGKernelTileScheduler(Params const &params_)
      : Base(params_), scheduler_params(params_) {}

  // ★ get_current_work：每次取 tile 都把 M_idx 换一次序；整个映射是一个双射（不重不漏）★
  // 三个分支只差「段间怎么排」：
  //   nnodes == 1        ：段就是整块 M，只在段内旋转 → new = (M_idx + offset) % P
  //   2 <= nnodes <= 4   ：查 nodes_wizzle 表（all_gather_swizzle.hpp），按 node 置换段序；
  //                        表里只备了 2 和 4 两档（下标 nnodes/2），nnodes=3 会读到 -1 项，
  //                        属于未支持的配置
  //   nnodes > kNodes(4) ：表不够用，直接公式化 —— 段号按 (node_id - local_id) mod nnodes 旋转
  // 【4-node 那张表其实就是 (l - r) mod 4 的展开】（每行是上一行的循环移位），
  // 与 >4 node 的公式同构，只是把取模换成了一次查表。
  // 两个动作合起来看：
  //   local_id  = M_idx / local_block_m               // 这个 tile 落在第几段（node 段）
  //   m         = (M_idx + offset) % local_block_m    // 段内坐标先转 offset：本地 tile 排最前
  //   new_M_idx = (m + perm(local_id) * local_block_m) % P   // 再换段（node 之间错开）
  // 【为什么这样就能藏住通信】：kernel 里 chunk id = M 行 / m_per_data_chunk，
  // 而 M_start 那一段是本 rank 自己写的数据——host 的 copy_local 就在 GEMM 的同一条 stream
  // 上、launch 前还有一次跨 rank barrier，flag 必为 1；从 offset 开始算意味着前面若干 tile
  // 完全不等 flag，这段时间正好是 ring push 在搬远端数据。
  // 【代价】M 顺序被打乱、N 方向不动，L2 复用与 DRAM 行局部性略差（顺序读变成回绕读）；
  // 换来的是「等待」被换成「已经就绪的活」。这只在通信/计算比接近时净赚：
  // 通信太快，重映射白改顺序；通信太慢，本地 tile 算完照样要等（那时就退化成全等）。
  // 【隐含前提】P % nnodes == 0（每 node 的 M 段等长）。M 不能均分时 local_block_m 向下取整，
  // 尾部会多出一个不满的段，local_id 会越出置换表——读这段前先确认 M 的切分方式。
  // 另外：stream-K 的 continue_current_work 不经过这里（fetch_next_work 先问它），
  // 所以重映射对每个 tile 只施加一次，不会被重复叠加。
  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    auto work_tile_info = Base::get_current_work();
    if (work_tile_info.is_valid()) {
      int new_M_idx;
      auto M_idx = work_tile_info.M_idx;

      if (scheduler_params.nnodes > bytedance::flux::kNodes) {
        int local_block_m = scheduler_params.problem_blocks_m / scheduler_params.nnodes;
        int local_id = M_idx / local_block_m;

        auto m = (M_idx + scheduler_params.problem_blocks_m_offset) % local_block_m;
        new_M_idx = (m + (scheduler_params.node_id - local_id + scheduler_params.nnodes) %
                             scheduler_params.nnodes * local_block_m) %
                    scheduler_params.problem_blocks_m;
      } else if (scheduler_params.nnodes > 1) {
        int local_block_m = scheduler_params.problem_blocks_m / scheduler_params.nnodes;
        int local_id = M_idx / local_block_m;

        auto m = (M_idx + scheduler_params.problem_blocks_m_offset) % local_block_m;
        new_M_idx =
            (m + bytedance::flux::nodes_wizzle[scheduler_params.nnodes / 2]
                         .swizzle[scheduler_params.node_id * scheduler_params.nnodes + local_id] *
                     local_block_m) %
            scheduler_params.problem_blocks_m;
      } else {
        new_M_idx =
            (M_idx + scheduler_params.problem_blocks_m_offset) % scheduler_params.problem_blocks_m;
      }
      work_tile_info.M_idx = new_M_idx;
    }

    return work_tile_info;
  }
};

// =================================================================================================
// 收尾：这个调度器与 kernel 的耦合点只有一处语义——「M_idx 是重映射之后的块号」。
// kernel 拿它算 chunk 区间（data_chunk_id_start / data_chunk_id_end），
// 调度器负责让排在前面的 tile 只依赖本地 chunk。
// 【必须和 kernel 一起读】：单看调度器会以为只是「换个顺序算」，
// 单看 kernel 会以为只是「等 flag」；合起来才是「用 tile 顺序把通信时间藏起来」这一件事。
// =================================================================================================
}  // namespace cutlass::gemm::kernel::detail
