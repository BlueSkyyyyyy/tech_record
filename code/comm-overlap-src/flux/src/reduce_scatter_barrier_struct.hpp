//===- reduce_scatter_barrier_struct.hpp -------------------------- C++ ---===//
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

// ---------------------------------------------------------------------------
// 【上游】bytedance/flux（comet 分支，commit 7f98c8a），路径 src/reduce_scatter/reduce_scatter_barrier_struct.hpp
// 【一句话】GEMM+ReduceScatter 那套 flag 的「内存布局」：纯数据结构 + 几个零开销视图包装，
//           一行同步逻辑都没有 —— 「怎么等、怎么加」在 include/flux/cuda/system_barrier.hpp，
//           这里只决定 flag 摆在哪、占多大、按什么顺序排。
// 本注释只讲三件事：
//   1) 这个 barrier 服务谁：谁 arrive、谁 wait，为什么必须是「按 (tile, rank) 二维计数」；
//   2) 三个结构体（PerTileFlags / PerRankFlags / BarrierWorkQeueuFlags）的字段含义与 128B 对齐的用意；
//   3) 为什么需要 reset，以及这套布局里最容易踩的两个坑（SoA 的 sizeof、跨 iteration 的零点）。
// 阅读顺序：PerTileFlags（细粒度，一个 tile 一组）→ PerRankFlags（粗粒度，一段一组）
//           → DEFINE_AOS/SOA_MEMBER 两个宏（两种索引方式）→ WorkQueue（tile 的生产/消费队列）。
// 术语：tile = 一个 thread block 负责的输出块；flag = 显存里的一个 int，用原子加计数、用 acquire load 读；
//       segment（段）= 输出沿 M 切给某个 rank 的那一条。
// ---------------------------------------------------------------------------
#pragma once

#include <stdint.h>
#include <cstddef>
#include <type_traits>
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define FLUX_HOST_DEVICE __forceinline__ __device__ __host__
#define FLUX_DEVICE __forceinline__ __device__
#elif defined(__CUDACC_RTC__)
#define FLUX_HOST_DEVICE __forceinline__ __device__
#define FLUX_DEVICE __forceinline__ __device__
#else
#define FLUX_HOST_DEVICE inline
#define FLUX_DEVICE inline
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"

namespace bytedance::flux {

// 【它服务的那次集合通信】GEMM+RS 的形状：每个 rank 用自己那份 K 切片算出完整 M×N 的部分和，
// 最后要「沿 M 归约，并且只留下一段」—— rank r 只写第 r 段 M（M/world_size 行）。
// 于是每个 rank 的 epilogue 都在做两件事：
//   a) 把自己算出的 tile 部分和推进「这段输出的 owner」的累积 buffer（目的地由 tile 的 m 坐标决定）；
//   b) 自己是 owner 的那些 tile，要等所有 rank 的同一 tile 都到齐，才能做本地归约/写出。
// 谁 arrive：每个往 owner buffer 里写过部分和的 rank，每个 tile 一次。
// 谁 wait：owner 自己 —— 它等的是「这个 tile 上收到了 world_size 次 arrive」。
// 【为什么是二维计数】(tile_idx, 到达次数) 缺一不可：
//   tile_idx 决定「哪一块输出可以开始归约了」（细粒度，允许乱序完成）；
//   到达次数决定「这一块是否齐了」（每来一个 rank +1）。
//   若压成一维（全局一个总计数），就得等所有 tile、所有 rank 全到齐 —— 那就退回「kernel 末尾一轮 RS」了，
//   这个结构存在的全部意义（把等待打散到 tile 粒度）随之消失。
// 【为什么 flag 数组是每个 rank 一份】barrier_ptr[r] 是 rank r 的 flag 数组；所有 rank 都往 owner 的那份
//   上原子加，owner 只读自己那份 —— 读在本地、写在对端，跨卡只发生在「写」这一侧，
//   也避免所有 rank 争抢同一条 cacheline。
struct __attribute__((packed)) PerTileFlags {
// 【一个 tile 的字段】
//   epilogue        epilogue 写完这份部分和的计数（上游代码里最后等的是 == 2，即「自己算完 + 归约完」两笔账）
//   reduce          这个 tile 从 peer 拷回/拷出的计数
//   reduce_sub_node 跨节点那一跳（sub world = 节点内 8 卡）完成
//   epilogue_queue  / reduce_queue：只在用 work queue 时有效，存的是 tile_idx + 1（0 = 空槽），
//                   消费者按生产顺序直接取，不用扫全表找「哪个 tile 好了」
//   extra           语义不承诺的附加区，实际被当锚点用：指向紧跟其后的 WorkQueue flag 区
// 【padding_xxx[8]】不只是填充：它让相邻两个语义 flag 落在不同的 32B sector 里，避免「A 在写 epilogue」
//   把「B 正在读 reduce」所在的 sector 打脏（sector 级 false sharing）。
//   上游注释 "for m < kM" 指的是 m 不是 kM 整数倍时，尾块也走同一套偏移。
  // per tile structs
  int epilogue;                    // marked to 1 once epilogue done
  int padding_epilogue[8];         // don't delete. for m < kM
  int reduce;                      // marked to 1 once tiled copy done from/to peer in sub world
  int padding_reduce[8];           // don't delete. for m < kM
  int reduce_sub_node;             // marked to 1 once tile copy done from next sub world
  int padding_reduce_sub_node[8];  // don't delete. for m < kM
  int epilogue_queue;              // only valid for use_barrier = true
  int reduce_queue;
  int extra;
  struct __attribute__((packed)) ShapeOnly {
    int epilogue;                    // marked to 1 once epilogue done
    int padding_epilogue[8];         // don't delete. for m < kM
    int reduce;                      // marked to 1 once tiled copy done from/to peer in sub world
    int padding_reduce[8];           // don't delete. for m < kM
    int reduce_sub_node;             // marked to 1 once tile copy done from next sub world
    int padding_reduce_sub_node[8];  // don't delete. for m < kM
    int epilogue_queue;              // only valid for use_barrier = true
    int reduce_queue;
    // non per-tile structs. maybe per-sm or per-segment. use with caution
    int extra;
  };

// 【整块 128B】按 packed 布局：6 个 int（24B）+ 3 组 padding_xxx（3×32B）+ 8B 尾部 padding = 128B，
// 一个 tile 恰好吃一条 cacheline，相邻 tile 不会互相踩线。
// 注意行内注释里那个 256B 的算式是早期布局的残留，以 padding[128 - sizeof(ShapeOnly)] 为准。
// 【ShapeOnly 是干什么的】只为算「不含 padding 时有多大」：用 sizeof(ShapeOnly) 而不是 sizeof(PerTileFlags)，
// 避免结构体在自己的定义里引用自身大小。代价是字段抄了两遍 —— 加字段时必须两处一起改，
// 否则 padding 会算成负数或错位（packed 结构体不一定会报错，sizeof 会直接歪掉）。
  char padding[128 - sizeof(ShapeOnly)];  // 30 * 4 + 7 * 8 = 176; cacheline: 256 - 176 = 80
};

// 【粗粒度的一份】PerRankFlags 按「段(segment)」计数，不是按 tile：一段 = 某个 rank 负责的那 M/world_size 行。
//   gemm_done / copy_done / buffer_ready / remote_copy_done 是流水里的四个里程碑；
//   counter 是「这一段已有几个 tile 的 epilogue 做完」的累加器（tiled_n × 该段内的 m tile 数），
//   加满之后再给 gemm_done +1 —— 段级的到齐信号是由 tile 级计数汇总出来的。
// 什么时候用：不需要 tile 级精度、只要「这一段整体进度」的场合（buffer 复用、跨节点拷贝、GEMM-K 分块）。
// 【与 PerTileFlags 的分工】tile 级 flag 是 O(tiled_m × tiled_n) 个，段级只有 O(num_segments) 个；
//   段级省内存、省原子次数，代价是等的人只能等「一整段」。
struct __attribute__((packed)) PerRankFlags {
  int gemm_done;
  int buffer_ready;
  int copy_done;
  int counter;
  int remote_copy_done;
  struct __attribute__((packed)) ShapeOnly {
    int gemm_done;
    int buffer_ready;
    int copy_done;
    int counter;
    int remote_copy_done;
  };
  char padding[128 - sizeof(ShapeOnly)];  // 5 * 4 = 20; cacheline: 256 - 20 = 236
};

// 【WorkQueue 的两个计数器】放在 flag 区之后（通过 PerTileFlags::extra_ptr(0) 找到），每个段一对：
//   epilogue_done：这一段已经 push 了多少个 tile，atomicAdd 的返回值就是队列槽位 head_value；
//   reduce_done：  消费侧已经取走了多少个，决定下一个该读队列的哪个槽。
// 队列里存 tile_idx + 1，所以 0 天然表示「还没人写」—— 消费者看到 <= 0 就继续自旋。
struct __attribute__((packed)) BarrierWorkQeueuFlags {
  int epilogue_done;  // 8 for 8 gpus
  int reduce_done;
};

// 【END_OF_STRCTURE：这个数组之后的第一块空闲地址】ptr_ + size_ * sizeof(BaseType)。
// 用途：flag 区按 rank 分块分配，每块 = 主体数组 + 尾部追加区（WorkQueue 就住在追加区里），
// 所以需要知道主体数组的末尾在哪。
// 【只对 AoS 成立】sizeof(BaseType) 含 padding；用在 SoA 视图上会算过头（SoA 只按 offsetof 排，不带 padding）。
#define END_OF_STRCTURE \
  void *end_of_structure() { return (void *)((char *)ptr_ + size_ * sizeof(BaseType)); }

// 【SoA 视图的地址公式】第 index 个元素的 member 字段 = ptr + size_ * offsetof(member) + index。
//   也就是：同一个字段的所有元素排成一条连续数组，字段之间隔 size_ 个元素。
// 代价/坑：它只用 offsetof、完全不用 sizeof(BaseType)，所以结构体里的 padding 在 SoA 模式下不占空间；
//   host 侧必须按 size_ * sizeof(字段) 的量分配。谁要是按 size_ * sizeof(BaseType) 估大小，就会错位。
// 收益：同一个语义的 flag 排成数组，按段/按 rank 顺序扫描时步长固定、cache 友好。
#define DEFINE_SOA_MEMBER(member)                                                     \
  using member##_type = decltype(std::declval<BaseType>().member);                    \
  FLUX_HOST_DEVICE member##_type *member##_ptr(int index) {                           \
    return reinterpret_cast<member##_type *>(                                         \
               reinterpret_cast<char *>(ptr_) + size_ * offsetof(BaseType, member)) + \
           index;                                                                     \
  }                                                                                   \
  FLUX_HOST_DEVICE member##_type member(int index) { return *member##_ptr(index); }

// 【AoS 视图】ptr[index].member：第 index 个元素的整块结构体连续存放，128B padding 真实存在；
// 内存利用率低（120/128 是有效字段），收益是一个 tile 的所有 flag 共享一条 cacheline。
#define DEFINE_AOS_MEMBER(member)                                                   \
  using member##_type = decltype(std::declval<BaseType>().member);                  \
  FLUX_HOST_DEVICE member##_type member(int index) { return *member##_ptr(index); } \
  FLUX_HOST_DEVICE member##_type *member##_ptr(int index) { return &(ptr_[index].member); }

// 【字段清单为什么写成宏】同一份字段列表要生成两个 wrapper 的访问器；用 ADD_MEMBER_FN 把「字段名」
// 变成「访问器」，保证 AoS/SoA 两条路径的字段集合与顺序严格一致 —— 加字段只改这一处，不会漏掉另一套。
// 这些宏只生成 getter / 取指针函数，编译完就是几条地址运算，没有运行时代价。
//// Per-Tile-Flags ////
#define PerTileFlagsDefines(ADD_MEMBER_FN) \
  ADD_MEMBER_FN(epilogue)                  \
  ADD_MEMBER_FN(reduce)                    \
  ADD_MEMBER_FN(reduce_sub_node)           \
  ADD_MEMBER_FN(epilogue_queue)            \
  ADD_MEMBER_FN(reduce_queue)              \
  ADD_MEMBER_FN(extra)                     \
  END_OF_STRCTURE

// 【Wrapper = 无状态的地址计算器】只存 (ptr, size)，构造是两条赋值，没有分配、没有初始化；
// 用的时候现场造一个视图（见 reduce_scatter_kernel.hpp 的 flags(rank)），用完就丢。
// 也正因为无状态，它可以按 rank 换指针：barrier_ptr[r] 是 rank r 那份 flag 区在本进程地址空间里的地址
// （由 IPC / 对称堆映射而来，不是拷贝）。
class PerTileFlagsSoAWrapper {
 private:
  void *ptr_;
  int size_;

 public:
  using BaseType = PerTileFlags;
  FLUX_HOST_DEVICE
  PerTileFlagsSoAWrapper(void *ptr, int size) : ptr_(ptr), size_(size) {}
  PerTileFlagsDefines(DEFINE_SOA_MEMBER)
};

// 【tile 级选 AoS】一个 tile 的 epilogue / reduce / queue 通常被同一批 peer 在相近时间点访问，
// 128B 连着放，一次 cacheline 事务就能覆盖这个 tile 的全部 flag；
// 换成 SoA 的话，同一个 tile 的读会散到五六条线上（多几次 L2 访问，也多几处 false sharing）。
class PerTileFlagsAoSWrapper {
 private:
  PerTileFlags *ptr_;
  int size_;

 public:
  using BaseType = PerTileFlags;
  FLUX_HOST_DEVICE
  PerTileFlagsAoSWrapper(void *ptr, int size) : ptr_((PerTileFlags *)ptr), size_(size) {}

  PerTileFlagsDefines(DEFINE_AOS_MEMBER)
};
#undef PerTileFlagsDefines

// 【真正在用的是 AoS】这个别名是全仓库的入口。上面的 SoA 版本保留着但当前没有被使用；
// 要用它之前先确认 host 分配是按 sizeof(PerTileFlags) = 128B/条 算的 —— 两种视图的内存布局不通用。
using PerTileFlagsWrapper = PerTileFlagsAoSWrapper;

//// Per-Rank-Flags ////

#define PerRankFlagsDefines(ADD_MEMBER_FN) \
  ADD_MEMBER_FN(gemm_done)                 \
  ADD_MEMBER_FN(buffer_ready)              \
  ADD_MEMBER_FN(copy_done)                 \
  ADD_MEMBER_FN(counter)                   \
  ADD_MEMBER_FN(remote_copy_done)          \
  END_OF_STRCTURE

// 【段级选 SoA】段数少、访问模式是「按段号散着看」，SoA 让同一个字段排成数组，环形遍历时步长固定；
// 对应的 AoS 版本存在，但没有被选用。
class PerRankFlagsSoAWrapper {
 private:
  void *ptr_;
  int size_;

 public:
  using BaseType = PerRankFlags;
  FLUX_HOST_DEVICE
  PerRankFlagsSoAWrapper(void *ptr, int size) : ptr_(ptr), size_(size) {}
  PerRankFlagsDefines(DEFINE_SOA_MEMBER)
};

class PerRankFlagsAoSWrapper {
 private:
  PerRankFlags *ptr_;
  int size_;

 public:
  using BaseType = PerRankFlags;
  FLUX_HOST_DEVICE
  PerRankFlagsAoSWrapper(void *ptr, int size) : ptr_((PerRankFlags *)ptr), size_(size) {}

  PerRankFlagsDefines(DEFINE_AOS_MEMBER)
};

#undef PerRankFlagsDefines

// 【最容易踩的坑】SoA 下这条 buffer 的实际占用是「每字段 num_segments 个 int」，不是 num_segments×128B；
// 而 PerTileFlags 是 AoS、128B/条 —— 同一块 barrier workspace 里两种视图混用时，
// 行内注释里的「cacheline: 256 - 20」这类算式只能当历史记录看。
using PerRankFlagsWrapper = PerRankFlagsSoAWrapper;
// 【WorkQueue 的 flag 走同一套宏】两个字段 + AoS，size_ 传的是段数，
// 所以每个段一对计数器各占 128B —— 段数很少，这点浪费换的是「段与段之间绝不共享 cacheline」。
// 一个小疑点：PerTileFlags::extra 在 packed 布局里落在 116B、后面只剩 8B padding，
// 而 world_size 个 8B 计数器接在这里会越过 128B 边界 —— 这条 use_barrier_queue 路径建议连 host 侧分配一起看。
//// WorkQueue ///
#define BarrierWorkQeueuFlagsDefines(ADD_MEMBER_FN) \
  ADD_MEMBER_FN(epilogue_done)                      \
  ADD_MEMBER_FN(reduce_done)                        \
  END_OF_STRCTURE

class BarrierWorkQeueuFlagsSoAWrapper {
 private:
  void *ptr_;
  int size_;

 public:
  using BaseType = BarrierWorkQeueuFlags;
  FLUX_HOST_DEVICE
  BarrierWorkQeueuFlagsSoAWrapper(void *ptr, int size) : ptr_(ptr), size_(size) {}
  BarrierWorkQeueuFlagsDefines(DEFINE_SOA_MEMBER)
};

class BarrierWorkQeueuFlagsAoSWrapper {
 private:
  BarrierWorkQeueuFlags *ptr_;
  int size_;

 public:
  using BaseType = BarrierWorkQeueuFlags;
  FLUX_HOST_DEVICE
  BarrierWorkQeueuFlagsAoSWrapper(void *ptr, int size)
      : ptr_((BarrierWorkQeueuFlags *)ptr), size_(size) {}

  BarrierWorkQeueuFlagsDefines(DEFINE_AOS_MEMBER)
};

#undef BarrierWorkQeueuFlagsDefines

using BarrierWorkQeueuFlagsWrapper = BarrierWorkQeueuFlagsAoSWrapper;

#undef DEFINE_AOS_MEMBER
#undef DEFINE_SOA_MEMBER

// 【谁负责清零】这个文件不 reset 任何东西，布局只是布局。清零有三个地方可选：
//   1) host 侧每轮 memset（简单，多一次写全 buffer 的 kernel + 一次时序约束）；
//   2) 消费侧 wait_eq_reset（CAS 等到 val 顺手归零，见 system_barrier.hpp）；
//   3) 最后一个 arrive 的线程主动复位。
// 漏掉/选错的症状都一样：第二轮 launch 一看「计数已经够了」直接放行，读到上一轮的残值，静默算错。
// 【成本总结】这套结构的开销不在内存（一个 tile 128B），而在原子次数：
//   每个 tile 每次跨 rank 送达 = 一次 sys-scope 原子加。tile 越细，overlap 越平滑，原子也越多。
}  // namespace bytedance::flux

#pragma GCC diagnostic pop
