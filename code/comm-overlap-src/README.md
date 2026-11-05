# comm-overlap-src — UniEP / Flux 上游源码带注释副本

配套 [《通算并行研读》](../../content/posts/comm-overlap-00-index/index.md) 专题。
`code/comm-overlap/` 是**纯 Python 的性能账本**；这里是**真实 kernel 的逐段注释副本**——
账本里的每个符号（通信 SM 数、chunk 数、signal、stage）在这两个仓库里都能找到落点。

- 上游：`ByteDance-Seed/Triton-distributed` @ `7908e4e`（`FlashComm/python/flash_comm/ep_overlap/`）与
  `bytedance/flux` @ `7f98c8a`（`comet` 分支）。
- 规矩：**只插入整行注释，不改一行代码**；`tools/verify_annotations.py` 会滤掉注释后与上游逐行比对。
- 没有实测：本机没有 SM100 / 多卡 NVLink，UniEP 是 Blackwell(sm100)+Hopper(sm90)、Flux 是 Hopper(sm90) 代码，
  这里只做源码精读。论文数字见文章正文。

## 目录

```
comm-overlap-src/
  README.md                  # 本文件
  docs/                      # 精读文档（从这里开始读）
    00-overview.md           # 两个仓库的定位、文件地图、阅读顺序、记号约定
    01-uniep-dispatch-gemm.md
    02-uniep-gemm-combine.md
    03-uniep-determinism.md
    04-flux-ag-gemm.md
    05-flux-gemm-rs.md
    06-compare.md            # 对照表 + 「等待粒度」代价模型
  uniep/                     # 7 个带注释副本（kernels/ 6? 见 docs/00 的文件地图）
  flux/                      # 5 个带注释副本
  tools/
    MANIFEST.tsv             # 上游相对路径 → 副本路径
    verify_annotations.py    # 校验：注释副本与上游逐行等价
```

## 快速开始

```bash
# 1) 校验所有带注释副本与上游逐行一致（上游默认在 ~/github）
python3 tools/verify_annotations.py

# 2) 只看某一侧
python3 tools/verify_annotations.py --only uniep/
python3 tools/verify_annotations.py --list

# 3) 开始读
sed -n 1,60p docs/00-overview.md
```

## 为什么值得读这两份

同样是「通算并行」，两份代码给出的是**两种互不兼容的工程取舍**：

| | UniEP | Flux |
|---|---|---|
| 通信在哪 | 同一个 kernel、同一批 SM，靠 warp 角色分工 | 另一条 stream，拷贝引擎 / 别的 SM |
| 等多大一块 | 一个 expert | 一个 M-chunk（`M/(world_size×SPLIT)` 行） |
| 换来了什么 | 细粒度、无 kernel 边界、可做到 bitwise 确定 | 实现简单、对 GEMM 主体零侵入、拷贝引擎免费 |

读完 `docs/06-compare.md` 你会得到一个统一的判断标准：**这个方案的等待粒度是多少、为它多付了什么**。
之后再遇到 DualPipe / Weave / MegaFlux 这类调度工作，都只是在同一个坐标系里换参数。
