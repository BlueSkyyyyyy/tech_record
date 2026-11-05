//===- sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp --- C++ ---===//
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
// Some code from
// cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp
// in NVIDIA cutlass project
// Original license as follows
/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// =================================================================================================
// 上游源码带注释副本（只插入了 // 注释行，代码逐行未改）
//   仓库 / 分支  bytedance/flux, comet 分支
//   提交         7f98c8a
//   路径         src/all_gather/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp
//   配套         src/all_gather/sm90_all_gather_gemm_tile_scheduler.hpp（决定 tile 顺序）
//   骨架来源     cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp（license 见上）
//
// 一句话定位：通信在别的 stream 上做，这个 kernel 只做「按 chunk 等 flag」的 GEMM——
// 把「等一次完整 all-gather」拆成「每个 M tile 只等它自己覆盖的那几个数据 chunk 到齐」。
//
// 本注释只讲三件事：
//   ① 谁在等、等什么：flag 下标 ↔ chunk id ↔ 全局 M 行区间 的换算（L668 起）
//   ② 为什么等待只在 producer warp 里做、为什么轮内只能用 warp 级 barrier（L779 起）
//   ③ 三级 warp specialization 与三条 pipeline 的角色划分（L528 起）
//
// 建议阅读顺序（行号以本副本为准）：
//   1) L668 起 7 行：data_chunk_id_start / data_chunk_id_end —— 细粒度依赖的全部秘密
//   2) L716 起 7 行：入口处唯一一次块级等待
//   3) L779 起 15 行：换 tile 时重算区间 + WarpBarrier::wait_eq（真正的 overlap 点）
//   4) L642 起 2 行：scheduler.get_current_work()，M_idx 已被重映射
//      （重映射实现在配套文件 L200 起）
//   5) L727 起 3 行 与 L875 起 3 行：producer / consumer 的寄存器再分配
//   6) 最后看 LoadWarpOrderBarrier（L814）与文末的 fetch_next_work
// =================================================================================================
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"
#include "cutlass/barrier.h"
#include "flux/cuda/memory_utils.hpp"

#include "all_gather/sm90_all_gather_gemm_tile_scheduler.hpp"
// ------------------------------------------------------------------------------------------------
// 与上游 CUTLASS cooperative kernel 的差异只有四处，读的时候只盯这四处，其余都是原样：
//   1) Params 多出 ptr_barrier / rank / world_size / n_data_chunks / m_per_data_chunk，
//      chunk 怎么切由 host 侧算一次（本文件 L360 起），device 端只做整数除法；
//   2) 进 kernel 后按「首 tile 的 chunk 区间」等一次 flag（L716 起）；
//   3) producer 主循环换 tile 时重算区间、必要时再等（L779 起）；
//   4) tile 顺序换成 Sm90AGKernelTileScheduler（配套文件），让本地数据先算。
// 【注意】这个 kernel 完全不发起 all-gather：gathered buffer 由 host 侧另一条 stream
// 提前/并行填（ring push 或 all2all 的 cudaMemcpyAsync），这里只消费它的「完成度」。
// ------------------------------------------------------------------------------------------------

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {
///////////////////////////////////////////////////////////////////////////////
// 三个作用域递进的 barrier，本文件全用到了（实现都在 cutlass/barrier.h 与 flux 的扩展版里）：
//   WarpBarrier   = GenericBarrier<SyncwarpSync>          ：32 线程；轮内等 flag 用它，最便宜
//   Barrier       = GenericBarrier<SyncthreadsSync>       ：整个 CTA；这里声明了但没用到
//   SystemBarrier = GenericSystemBarrier<SyncthreadsSync> ：同一套语义，但读/写走
//                   ld.global.acquire.sys / fence.acq_rel.sys，保证跨 SM、跨设备的可见性——
//                   flag 是 host 用 cuStreamWriteValue32 写到显存里的（见 flux/include/system_barrier.hpp）。
// 【实现细节，读代码时最容易踩】：三者都是「thread_idx == 0 的那个线程自旋，
// 其余线程停在 Sync::sync()」，所以调用方传进来的 thread_idx 必须是组内 0 号线程的索引，
// 否则没有线程自旋、wait 直接返回。本文件两处调用传的都是全局 thread_idx：
// 入口那次由全员执行（thread 0 就是块内 0 号，成立）；轮内那次只由 producer mainloop
// warp 执行，而它正好是 warp 0（thread 0..31），也成立——这不是巧合，是分工决定的。
//
// SPLIT：把一个 rank 的本地 M 再切成几段搬（host 侧 transfers.hpp 里有同名宏，两处必须一致）。
// SPLIT=1 时一个 chunk 就是一个 rank 的全部本地 M，chunk 数 == world_size。
using Barrier = GenericBarrier<cutlass::detail::SyncthreadsSync>;
using SystemBarrier = cutlass::detail::SystemBarrier;
using WarpBarrier = GenericBarrier<cutlass::detail::SyncwarpSync>;

#define SPLIT 1

// 与上游同名 kernel 一样是 cooperative 变体：两个 consumer warp group 合算同一个 (M,N) tile，
// 一个 tile 的 MMA 由 256 个线程协作完成。
// Base 的 enable_if 把 Schedule 绑死为 KernelTmaWarpSpecializedCooperative——因为下面的
// smem 布局、pipeline 角色、tile 循环、load_order barrier 全是按 cooperative 写死的，
// 换 schedule 直接编不过。
template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class Sm90AGGemmUniversal {
 public:
  using Base = GemmUniversal<
      ProblemShape_,
      CollectiveMainloop_,
      CollectiveEpilogue_,
      TileScheduler_,
      cute::enable_if_t<cute::is_base_of_v<
          KernelTmaWarpSpecializedCooperative,
          typename CollectiveMainloop_::DispatchPolicy::Schedule>>>;
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(
      cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
      "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(ArchTag::kMinComputeCapability >= 90);

  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::
      TileSchedulerSelector<TileScheduler_, ArchTag, TileShape, ClusterShape>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  // 【384 线程与 64K 寄存器的账，两个魔数 40 / 232 就是这么来的】：
  //   NumLoadWarpGroups = 1        → 4 个 load warp（producer）
  //   NumMmaWarpGroups  = 256/128  → 2 个 MMA warp group（consumer）
  //   MaxThreadsPerBlock = 256 + 128 = 384，MinBlocksPerMultiprocessor = 1（一个 CTA 独占一个 SM）
  // 寄存器：256×232 + 128×40 = 59392 + 5120 = 64512 ≤ 65536，刚好塞满一个 SM 的寄存器文件。
  // 于是 producer 主动 dealloc 到 40、consumer alloc 到 232（见 L727 与 L875）。
  // 【warp specialization 的收益不只是「分工清楚」，更是把寄存器从搬运线程挪给 MMA 线程】：
  // 两个 WG 协作要 hold 的 accumulator 很大，232 是它能「2 个 WG 并行算同一 tile」的前提。
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups =
      CUTE_STATIC_V(size(TiledMma{})) / NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock =
      CUTE_STATIC_V(size(TiledMma{})) + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  /// Register requirement for Load and Math WGs
  static constexpr uint32_t LoadRegisterRequirement = 40;
  static constexpr uint32_t MmaRegisterRequirement = 232;

  // 1 stage ordered sequence between mainloop and epilogue producer load threads
  // LoadWarpOrderBarrier = OrderedSequenceBarrier<1, 2>：同一个 producer warp group 内，
  // 给两条「加载」流水线排序——group0 = Mainloop warp，group1 = Epilogue warp，SequenceDepth = 1。
  // 语义（cutlass/pipeline/sm90_pipeline.hpp）：group0 的 arrive 放行 group1 的 wait；
  // group1 从不 arrive，所以它只有第一次 wait 会真正阻塞，之后立即返回。
  // 【为什么需要】：两条 producer 流水线抢同一个 TMA 单元和 issue slot，让 A/B 的预取先上路，
  // 首 tile 的 K-loop 才不会被 epilogue 的 C/D 加载推后。
  // 【在 AG-GEMM 里这条排序常常整条不生效】：beta=0（D 直接覆盖写），
  // epilogue 的 is_producer_load_needed() 为 false，Epilogue producer 分支根本不进；
  // barrier 只留下 Mainloop 侧一次 arrive，几乎零成本。
  //
  // SharedStorage 分两块：tensors（mainloop 的 A/B stage + epilogue 的 C/D stage）与
  // pipelines（mainloop / epi_load / epi_store 三条 pipeline 的 mbarrier 数组 + load_order）。
  // 【全部 smem 只有一个 CTA 用，因为 MinBlocksPerMultiprocessor = 1】。
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1, 2>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
    void *ptr_barrier = nullptr;
    int rank = 0;
    int world_size = 0;
  };

  // Kernel entry point API
  // 【AG 多出来的接口全在 Params 里】：ptr_barrier 指向本 rank 的 flag 数组（在显存里，
  // host 用 cuStreamWriteValue32 写 1、device 用 ld.global.acquire.sys 读）；
  // rank / world_size 只用于调试断言；真正参与计算的是 n_data_chunks / m_per_data_chunk。
  // 注意 Arguments 里没有 chunk 相关字段——「怎么切 chunk」是 host 端 to_underlying_arguments
  // 的单点决策（L360 起），device 端只拿到两个整数。
  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info;
    TileSchedulerParams scheduler;
    void *ptr_barrier;
    int rank;
    int world_size;
    int TILE_SIZE_M;
    int n_data_chunks;
    int m_per_data_chunk;
    void *workspace;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static Params
  to_underlying_arguments(Arguments const &args, void *workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    auto problem_shape = args.problem_shape;
    if constexpr (detail::IF_SWAP_AB<CollectiveMainloop>::value) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST(
          "  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the "
          "SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST(
        "to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

    // Calculate workspace pointers
    uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(workspace);
    size_t workspace_offset = 0;

    void *scheduler_workspace = workspace_ptr;
    workspace_offset +=
        TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
            args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

    void *epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

    void *mainloop_workspace = nullptr;
    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will
    // be used in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is
    // 1, which means subtile will not be used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        problem_shape_MNKL,
        TileShape{},
        ClusterShape{},
        hw_info,
        args.scheduler,
        scheduler_workspace,
        NumEpilogueSubTiles);
    auto problem_blocks =
        TileScheduler::get_tiled_cta_shape_mnl(problem_shape_MNKL, TileShape{}, ClusterShape{});
    CUTLASS_ASSERT(args.world_size != 0);

    // ★ host 侧唯一一次「切 chunk」，之后 device 全靠这两个整数还原 ★
    //   n_data_chunks    = world_size * SPLIT    // 一次 all-gather 一共搬几块
    //   m_per_data_chunk = M / n_data_chunks     // = (M/world_size)/SPLIT，每块多少行
    // 对应的显存布局（host 侧 input_buffer，见 ths_op/all_gather_gemm_kernel.cc）：
    //   M 行 [0, m_per)        ← rank0 的第 0 段（SPLIT>1 时第 1 段紧接着排在后面）
    //   M 行 [m_per, 2*m_per)  ← rank0 的第 1 段 / 或 rank1 的第 0 段，取决于 SPLIT
    //   ...                    ← 依次排开：先按 rank 段、段内再按 split 段
    // 所以「M 行号 ÷ m_per_data_chunk」直接等于「这一行来自哪个 chunk」，chunk id 就是 flag 下标。
    // 【这一步把「通信进度」翻译成了「行号区间」】——kernel 里才能只等自己那几行。
    // 用除法而不是查表：tile 与 chunk 的边界不对齐是常态，除法天然支持任意不对齐。
    int TILE_SIZE_M = size<0>(TileShape{});
    int n_data_chunks = args.world_size * SPLIT;
    int m_per_data_chunk = get<0>(problem_shape_MNKL) / n_data_chunks;

    return {
        args.mode,
        problem_shape,
        CollectiveMainloop::to_underlying_arguments(
            args.problem_shape, args.mainloop, mainloop_workspace),
        CollectiveEpilogue::to_underlying_arguments(
            args.problem_shape, args.epilogue, epilogue_workspace),
        hw_info,
        scheduler,
        args.ptr_barrier,
        args.rank,
        args.world_size,
        TILE_SIZE_M,
        n_data_chunks,
        m_per_data_chunk,
        workspace};
  }

  CUTLASS_HOST_DEVICE static bool
  can_implement(Arguments const &args) {
    bool implementable =
        (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static int
  get_workspace_size(Arguments const &args) {
    size_t workspace_size = 0;
    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
        args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(
      Arguments const &args,
      void *workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter *cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(workspace);
    size_t workspace_offset = 0;
    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
        args.scheduler,
        workspace_ptr + workspace_offset,
        stream,
        args.problem_shape,
        args.hw_info,
        NumMmaWarpGroups,
        NumEpilogueSubTiles);
    workspace_offset +=
        TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
            args.scheduler,
            args.problem_shape,
            args.hw_info,
            NumMmaWarpGroups,
            NumEpilogueSubTiles);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = CollectiveEpilogue::initialize_workspace(
        args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // persistent：grid 只由 SM 数决定，与 tile 数无关（tile 数通常远大于 CTA 数）。
  // 【这是「按 tile 等 chunk」能成立的前提】：一个 CTA 常驻，算完一个 tile 立刻
  // fetch_next_work 取下一个，于是「等 flag」和「算别的 tile」落在同一条 CTA 时间线上，
  // 能互相遮盖；若按 tile 起 grid，等待只能靠 host 侧排队，遮盖不住。
  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const &params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can
    // run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN
                            ? TileScheduler::RasterOrderOptions::AlongN
                            : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(
        params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const &params, char *smem_buf) {
    using namespace cute;
    using X = Underscore;

// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if !defined(__CUDA_ARCH_FEAT_SM90_ALL)
    if constexpr (size<0>(typename TiledMma::AtomShape_MNK{}) == 64) {
      printf(
          "ERROR : Arch conditional MMA instruction used without targeting sm90a compute "
          "capability. Aborting.\n");
      return;
    }
#endif

    // Preconditions
    static_assert(
        size(TiledMma{}) == 256,
        "Cooperative kernel must have TiledMMA operating using 256 threads.");
    static_assert(
        size<0>(TileShape{}) >= 128,
        "Cooperative kernel requires Tile Size to be greater than or equal to 128 along "
        "the M-dimension.");
    static_assert(
        cute::rank(StrideA{}) == 3,
        "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(
        cute::rank(StrideB{}) == 3,
        "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(
        cute::rank(StrideC{}) == 3,
        "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(
        cute::rank(StrideD{}) == 3,
        "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    /* In the Cooperative kernel, Consumer0 and Consumer1 collaborate on the same tile */
    // 【三级 warp specialization：384 线程 = 3 个 warp group】
    //   WG0 Producer（4 warp，40 reg）：
    //       role 0 = Mainloop   → 发 A/B 的 TMA、等 chunk flag（本 kernel 的「等待点」）
    //       role 2 = Epilogue   → 发 C/D 的 TMA load（只在 beta≠0 时需要）
    //       role 1 / 3 空转，枚举里占位，保持「4 个 warp 编号」整齐
    //   WG1 / WG2 Consumer（各 4 warp，232 reg）：两个 WG 协作算同一个 tile 的 MMA
    // 【role 编号不是 warp 编号】：不同 warp group 里同一编号含义完全不同，
    // 下面所有分支都要 warp_group_role + producer_warp_role 两层判据，别混。
    // canonical_* 系列是 CUTLASS 在「编译期已知 warp 布局」下给的常量索引，别改成 threadIdx 推算。
    enum class WarpGroupRole { Producer = 0, Consumer0 = 1, Consumer1 = 2 };
    enum class ProducerWarpRole { Mainloop = 0, Warp1 = 1, Epilogue = 2, Warp3 = 3 };

    // Kernel level shared memory storage
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    int mma_thread_idx = thread_idx % size(TiledMma{});
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // 三条 pipeline，producer / consumer 角色按上面的分工设：
    //   mainloop_pipeline ：producer = 1 个 warp（不是 1 个 WG！），consumer = 256 线程（2 个 WG）
    //   epi_load_pipeline ：producer = Epilogue warp，consumer = 256 线程
    //   epi_store_pipeline：只有 producer（TMA store 靠 scoreboard 等，没有 consumer 侧 mbarrier）
    // 【num_consumers = size(TiledMma) = 256 是 cooperative 的标志】：一个 tile 由两个 WG 共同消费，
    // 每个 stage 必须等两个 WG 都 mma 完才释放（arrive 计数 256）——pipeline 深度因此直接决定
    // producer 能比 consumer 提前跑多远，也就是「等 flag 时手上还有多少活可干」。
    // producer 侧起点用 make_producer_start_state（故意用相反的 phase，跳过第一次 wait，
    // 因为 kernel 刚开始时 buffer 一定是空的）。
    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::Mainloop) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 ||
        warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = size(TiledMma{});
    mainloop_pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
    MainloopPipeline mainloop_pipeline(
        shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 ||
        warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = size(TiledMma{});
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(
        shared_storage.pipelines.load_order, params_load_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state =
        cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state =
        cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state =
        cutlass::make_producer_start_state<EpiStorePipeline>();

    // cluster 初始化可见性：smem 里的 mbarrier 必须被簇内所有 CTA 看到，才能开始 TMA。
    // cluster_arrive_relaxed + cluster_wait 是簇级同步；ClusterShape 只有 (1,..) 时退化成一条 __syncthreads。
    // 必须在任何 TMA 与 flag 等待之前——否则 TMA 可能写到还没 init 的 mbarrier 上。
    auto cluster_wait_fn = []() {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return []() { cute::cluster_wait(); };
      } else {
        __syncthreads();
        return []() {};  // do nothing
      }
    }();

    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    auto blk_shape = TileShape{};  // (BLK_M,BLK_N,BLK_K)

    TileScheduler scheduler{params.scheduler};
    auto work_tile_info = scheduler.get_current_work();

    // if (thread0() && params.rank == 0) {
    //   printf(
    //       "... m_per_data_chunk: %d, M: %d\n",
    //       m_per_data_chunk,
    //       static_cast<int>(get<0>(problem_shape_MNKL)));
    // }

    // ★★★ 细粒度依赖的全部秘密：4 行整数运算 ★★★
    //   M_idx                     ：本 tile 在 M 方向的块号——注意它已经被 scheduler 重映射过
    //                               （上一行 get_current_work，实现在配套文件 L200 起），
    //                               所以这里算的是「重映射之后」的区间，两个文件必须配套读
    //   M_idx * TILE_SIZE_M       ：该 tile 覆盖的全局 M 起始行
    //   ÷ m_per_data_chunk        ：起始行落在第几个 chunk  → 需要等的最小 chunk id
    //   (M_idx+1)*TILE_SIZE_M - 1 ：该 tile 的全局 M 结束行   → 需要等的最大 chunk id
    // 只要这 TILE_SIZE_M 行对应的 chunk 到齐就能开算，别的 chunk 还没到与它无关。
    // 【这就是「一次 all-gather 的等待」被拆成「每个 tile 只等自己那几个 chunk」的地方】。
    // 【TILE_SIZE_M 与 m_per_data_chunk 一般不整除】，一个 tile 可能横跨 2~3 个 chunk，
    // 所以区间要一头一尾都算；end 必须 clamp，否则最后一个 tile 的右端会超出行数。
    // 举个具体例子（world_size=8、每 rank 本地 M=512、TILE_SIZE_M=128、SPLIT=1）：
    //   m_per_data_chunk = 512 = 4 个 tile，于是每个 tile 恰好落在一个 chunk 内，start == end，
    //   等待退化成「等 1 个 flag」，且 tile j 等的就是 chunk j（即 rank j 的数据）。
    // 【代价】每 tile 两次整数除法，可忽略；真正的代价是「跨 chunk 边界的 tile 要等两个 chunk」——
    // TILE_SIZE_M 越大、chunk 越碎，边界 tile 占比越高（这正是 SPLIT 这根旋钮的副作用）。
    const int data_chunk_id_start =
        work_tile_info.M_idx * params.TILE_SIZE_M / params.m_per_data_chunk;
    int data_chunk_id_end = (work_tile_info.M_idx * params.TILE_SIZE_M + params.TILE_SIZE_M - 1) /
                            params.m_per_data_chunk;
    data_chunk_id_end = (data_chunk_id_end < params.n_data_chunks - 1)
                            ? data_chunk_id_end
                            : (params.n_data_chunks - 1);

    // In a warp specialized kernel, collectives expose data movement and compute operations
    // separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Prepare and partition the input tensors. Expects a tuple of tensors where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape
    // (BLK_M,BLK_K,m,k,l) get<1>(load_inputs) is the tma tensor B after local tiling so that it
    // has shape (BLK_N,BLK_K,n,k,l)
    // load_init 之后 gA_mkl 的 M 维已经是 world_size*local_M，即 gathered 后的完整 A：
    // TMA 直接按「全局 M 行号」去 input_buffer 取数，GEMM 完全不知道数据是从别的卡来的。
    // 【这是 Flux 的取舍】：不新增「分段 GEMM / 变长 M」，只把 A 的地址空间拼成全局布局，
    // 让 GEMM 代码原样按全局 M 索引；代价是必须有一块 world_size × 本地 A 大小的 gathered buffer。
    // cluster_wait_fn() 必须在这里：pipeline / mbarrier 的 init 先全簇可见，再谈数据到没到。
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(
        cute::tuple_size_v<decltype(load_inputs)> >= 2,
        "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = size<3>(gA_mkl);

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    // ★ 入口处的「首 tile 等待」——全 kernel 唯一一次块级等待 ★
    // SystemBarrier::wait_eq：thread 0 自旋 ld.global.acquire.sys 读 flag[id]==1，
    // 其余 383 个线程一起停在 __syncthreads()，所以是「三个 warp group 同时等到数据到齐」。
    // 为什么入口用块级：此刻没有任何计算能遮盖等待，块级最省心；同时给三个 WG 一个共同起点，
    // consumer 不会抢在 producer 前面空转（其实它们也只能等 pipeline，但省掉一次无谓的忙等）。
    // 为什么特判 start == end：区间退化成单个 chunk 时省一次循环判断，而这正是首 tile 的常见情形。
    // 【去掉会怎样】：主循环第一轮算出的 new_start / new_end 恰好等于这里的 start / end，
    // 于是循环内的等待被「相等则跳过」的逻辑略过——删掉这段等于第一个 tile 完全不等 flag，
    // 直接读还没搬完的远端数据。
    // 【代价】一次全 CTA 同步，频率 = 每 CTA 一次（不是每 tile 一次），可忽略。
    if (data_chunk_id_start == data_chunk_id_end) {
      SystemBarrier::wait_eq(params.ptr_barrier, thread_idx, data_chunk_id_end, 1);
    } else {
      for (int id = data_chunk_id_start; id <= data_chunk_id_end; ++id) {
        SystemBarrier::wait_eq(params.ptr_barrier, thread_idx, id, 1);
      }
    }

    if (warp_group_role == WarpGroupRole::Producer) {
      // producer 主动把寄存器让出去（40/线程），省下的全给两个 consumer WG 的 accumulator。
      // warpgroup_reg_dealloc 之后高编号寄存器不再归 producer 所有，MMA 侧才开得起 232。
      // 【没有这一句，232 就申请不到，两个 WG 协作的大 accumulator 落不了地】。
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        // 每个 work tile 一轮：
        //   valid_warpgroup_in_work_tile 为假 → 该 WG 不参与这个 tile（stream-K split 场景），
        //   直接取下一个 tile；注意这条路径也走 fetch_next_work，M 重映射一样生效。
        bool do_load_order_arrive = true;
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            work_tile_info = fetch_next_work(work_tile_info, scheduler);
            continue;
          }
          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          // auto offset = work_tile_info.M_idx / params.tiles_per_barrier;
          // WarpBarrier::wait_eq(params.ptr_barrier, thread_idx, offset, 1);
          // ★★ 换 tile 时重算一遍 chunk 区间：和入口那次不一样才等 ★★
          // 为什么「只和入口那次比」就够：flag 只会从 0 变 1（每次调用前 host 清零），
          // 重复等一个已置 1 的 chunk 只是一次 ld.global.acquire + 比较，立即返回；
          // 所以不必维护「等到哪了」的状态，只跟入口那次的区间比一次即可。
          // 注意这两个变量在循环里从不更新，比的是「首 tile 区间」，不是「上一个 tile」——
          // 相等（同一 chunk 内的后续 tile）就整段跳过，不等就整段重等一遍，代价可忽略。
          // 【为什么这里必须是 WarpBarrier，不能用入口那个 SystemBarrier】：
          //   1) producer 在 if 分支里，consumer 在另一个分支——这里调 __syncthreads 会直接死锁
          //      （两个分支永远不会汇合到同一个 barrier）；
          //   2) 只有 warp 内 32 个线程需要知道「数据到齐」，warp 级同步就够，比块级便宜一个量级；
          //   3) 数据可见性由 ld.global.acquire.sys 的 acquire 语义 + TMA 读 global 的顺序保证，
          //      不需要 CTA 内其它线程参与。
          // 【只有 producer 等，consumer 一个 flag 都不碰】：consumer 取数走 mainloop pipeline
          // 的 stage（mbarrier 的 transaction 计数），producer 没发 TMA 就没有 stage 可消费，
          // 依赖沿 pipeline 天然传递。flag 只需要挡住「TMA 发射」这一条边，链条就完整了。
          // 【上面被注释掉的 tiles_per_barrier + WarpBarrier 是早期粗粒度版本】：按固定 tile 数等一次；
          // 现在改成按 chunk 区间精确等，等待次数与数据到达粒度对齐（这是这份代码与论文的核心差异）。
          // 【代价】每 tile 两次除法 + 两次比较；风险是依赖「flag 单调」这个前提——
          // host 必须每次调用前清零（ths_op 里 barrier_buffer.zero_()），否则第二次调用会直接穿过去。
          const int new_data_chunk_id_start =
              work_tile_info.M_idx * params.TILE_SIZE_M / params.m_per_data_chunk;
          int new_data_chunk_id_end =
              (work_tile_info.M_idx * params.TILE_SIZE_M + params.TILE_SIZE_M - 1) /
              params.m_per_data_chunk;
          new_data_chunk_id_end = (new_data_chunk_id_end < params.n_data_chunks - 1)
                                      ? new_data_chunk_id_end
                                      : (params.n_data_chunks - 1);

          if (new_data_chunk_id_start != data_chunk_id_start ||
              new_data_chunk_id_end != data_chunk_id_end) {
            for (int id = new_data_chunk_id_start; id <= new_data_chunk_id_end; ++id) {
              WarpBarrier::wait_eq(params.ptr_barrier, thread_idx, id, 1);
            }
          }

          // Get the number of K tiles to compute for this work as well as the starting K tile
          // offset of the work.
          auto work_k_tile_count =
              TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
          auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
          auto k_tile_iter = cute::make_coord_iterator(
              idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

          // 一个 tile 的 K-loop 全部 TMA 在这里发出（内部按 stage 逐个 acquire，stage 满了就阻塞，
          // 等 consumer 释放）。注意 load 是「发完就走」：TMA 完成由 mbarrier 通知 consumer，
          // producer 不等数据落地。
          // advance(work_k_tile_count)：pipeline 状态跨 tile 连续滚动，不是每 tile 重置——
          // 这样 producer 在上一个 tile 的尾部就能提前把下一个 tile 的 stage 填上（软件流水）。
          collective_mainloop.load(
              params.mainloop,
              mainloop_pipeline,
              mainloop_pipe_producer_state,
              load_inputs,
              blk_coord,
              k_tile_iter,
              work_k_tile_count,
              lane_idx,
              block_rank_in_cluster,
              shared_storage.tensors.mainloop);
          // Update starting pipeline state for the next tile
          mainloop_pipe_producer_state.advance(work_k_tile_count);

          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            // 只做一次（do_load_order_arrive 守卫）：时机是「首个 tile 的 TMA 已发出」之后，
            // 相当于告诉 Epilogue producer「A/B 预取已上路，你可以开始发 C/D 了」。
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          // Get next work tile
          work_tile_info = fetch_next_work(work_tile_info, scheduler);
        }  // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        // fetch_next_work：stream-K 的 continue_current_work 优先（同一 tile 的剩余 K 段），
        // 否则 advance_to_next_work + get_current_work——后者会做 M 重映射（配套文件）。
        // load_tail 等所有 consumer 释放完 stage 才退出，保证 kernel 结束时 smem 不再被 TMA 写。
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
      }  // Mainloop Producer Warp End
      // Epilogue producer 分支：只有 epilogue 需要从 global 预取 C/D 时才存在
      // （AG-GEMM 的 beta=0 纯覆盖写场景 is_producer_load_needed() 为 false，整段跳过）。
      // load_order_barrier.wait()：等 Mainloop 的首批 TMA 已发出。注意它写在循环里，而
      // 上游 CUTLASS 写在循环外——语义等价（group1 从不 arrive，只有第一次会真阻塞），
      // 读的时候别以为它是「每 tile 排序一次」。

      // Epilogue Producer Warp
      else if (
          producer_warp_role == ProducerWarpRole::Epilogue &&
          collective_epilogue.is_producer_load_needed()) {
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::requires_separate_reduction(params.scheduler)) {
            load_order_barrier.wait();
          }
          if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            epi_load_pipe_producer_state = collective_epilogue.load(
                epi_load_pipeline,
                epi_load_pipe_producer_state,
                problem_shape_MNKL,
                blk_shape,
                blk_coord,
                tiled_mma,
                lane_idx,
                shared_storage.tensors.epilogue,
                work_tile_info.reduction_subtile_idx());
          }

          // Get next work tile
          work_tile_info = fetch_next_work(work_tile_info, scheduler);
        }  // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      }  // Epilogue Producer Warp End
    }  // Producer Warp Group End

    else if (
        warp_group_role == WarpGroupRole::Consumer0 ||
        warp_group_role == WarpGroupRole::Consumer1) {
      // consumer：寄存器要回 232（两个 WG 各 hold 一个 tile 的 accumulator 分片）。
      // 之后每个 tile：mma → mma_tail（等 MMA 落地并释放最后几个 stage）→ fixup → epilogue.store。
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      // Do we potentially issue tail arrives for TMA stores, if epilogue load is waiting for it
      bool do_store_tail = false;
      while (work_tile_info.is_valid()) {
        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        auto work_k_tile_count =
            TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);

        // Allocate the accumulators for the (M,N) blk_shape
        //
        // MSVC CTAD breaks if we say "Tensor" here, so we use "auto" instead.
        auto accumulators =
            partition_fragment_C(tiled_mma, take<0, 2>(blk_shape));  // (MMA,MMA_M,MMA_N)

        // Index of warp group within consumer warp groups
        int consumer_warp_group_idx = canonical_warp_group_idx() - NumLoadWarpGroups;

        if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
          collective_mainloop.mma(
              mainloop_pipeline,
              mainloop_pipe_consumer_state,
              accumulators,
              work_k_tile_count,
              mma_thread_idx,
              shared_storage.tensors.mainloop,
              params.mainloop);

          // Make sure the math instructions are done and free buffers before entering the epilogue
          collective_mainloop.mma_tail(
              mainloop_pipeline, mainloop_pipe_consumer_state, work_k_tile_count);

          // Update starting mainloop pipeline state for the next tile
          mainloop_pipe_consumer_state.advance(work_k_tile_count);
        }

        // Perform reduction across splits, if needed
        // consumer_warp_group_idx = 本 WG 在 consumer 里的序号（0/1），epilogue 与 stream-K 归约
        // 靠它把 accumulator 切片；fixup 只在 requires_separate_reduction（stream-K）时真做事，
        // 普通 persistent 调度下是空操作——AG 默认走 persistent，别在这里找等待逻辑。
        TileScheduler::fixup(
            params.scheduler,
            work_tile_info,
            accumulators,
            NumMmaWarpGroups,
            consumer_warp_group_idx);

        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
          // Epilogue and write to gD
          auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
              collective_epilogue.store(
                  epi_load_pipeline,
                  epi_load_pipe_consumer_state,
                  epi_store_pipeline,
                  epi_store_pipe_producer_state,
                  problem_shape_MNKL,
                  blk_shape,
                  blk_coord,
                  accumulators,
                  tiled_mma,
                  mma_thread_idx,
                  shared_storage.tensors.epilogue,
                  work_tile_info.reduction_subtile_idx());
          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
          epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;
          do_store_tail = true;
        }

        // Get next work tile
        work_tile_info = fetch_next_work(work_tile_info, scheduler);
      }  // Scheduler work fetch loop

      if (do_store_tail) {
        collective_epilogue.store_tail(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state);
      }
    }  // Consumer Warp Groups End
  }

  // =================================================================================================
  // 【全 kernel 的 overlap 结构，一句话收尾】
  //   host：在 cp_streams[0] 上跑 ring push / all2all 的 cudaMemcpyAsync，每搬完一个 chunk
  //         就用 cuStreamWriteValue32 把 flag[id] 写 1（「电平」语义，不是计数）；
  //         **不等它完成**，立刻在 current_stream 上 launch 本 kernel——
  //         两条 stream 之间没有 event 依赖，唯一的同步就是这排 flag。
  //   device：每个 CTA 先算「自己本地那段 M」（scheduler 重映射），算到哪个 chunk 就等哪个 flag；
  //         远端数据到达的时间被「已经在本地的那部分 tile」的计算盖住。
  //   → 通信时间被计算盖住，flag 只负责把「TMA 读哪一行」和「那一行到了没有」绑在一起。
  // 【代价与边界】gathered buffer 要占 world_size × 本地 A 的显存；等待粒度是 tile 而不是 cache line；
  // chunk 越碎等待越细、但要等的 flag 次数越多（SPLIT 就是这根旋钮）；
  // 全部收益的前提是「本地 tile 的计算量 ≥ 远端数据的搬运量」，否则重映射只是换个顺序等。
  // =================================================================================================
 private:
  // Kernel helper function to get next work unit
  CUTLASS_DEVICE
  typename TileScheduler::WorkTileInfo
  fetch_next_work(
      typename TileScheduler::WorkTileInfo &work_tile_info, TileScheduler &scheduler) const {
    // Check whether we should continue on with the current work unit. If this is the case,
    // the work unit will have been updated in continue_current_work to reflect the new
    // tile to be computed.
    if (scheduler.continue_current_work(work_tile_info)) {
      return work_tile_info;
    }

    // Get next work tile
    scheduler.advance_to_next_work();
    return scheduler.get_current_work();
  }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::kernel
