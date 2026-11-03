# 扩散模型精读配套代码

对应专题：[扩散模型精读](../../content/posts/diffusion-00-index/index.md)。

八个纯标准库 / NumPy / PyTorch CPU 脚本。它们不是模型实现，而是把文章里的账**算出来**：
噪声日程的有效窗口、ε 误差的放大、采样器的离散化误差、CFG 的锐化、latent/DiT/视频的 token 与 MACs、
FM 与扩散的步数差异、蒸馏的塌缩、FID 的有限样本偏差。

| 文件 | 依赖 | 对应文章 | 输出要点 |
|---|---|---|---|
| `schedules.py` | numpy | 一、二 | 线性日程只有 259/1000 步 SNR ≥ 0 dB；ᾱ_T=4.0e−5；cosine 有效步数 496 |
| `ddpm_toy.py` | torch | 一 | 8 高斯上训 DDPM；1000 步半径 2.031，20 步朴素跳采缩到 1.337；t=900 时 x0 误差放大 60× |
| `sampler_zoo.py` | numpy | 二、三 | 解析 score 下 DDIM/Euler/Heun 对比；DDIM 5 步 0.234、10 步 0.092，10 步后饱和 |
| `cfg_toy.py` | numpy | 四 | guidance scale 1→7.5，少数峰 39.8%→5.6%，std 0.495→0.277，均值从 1.40 掉到 0.96 |
| `latent_budget.py` | 标准库 | 五、六、九、十一 | DiT-XL/2 = 118.6 GMACs = 237 GFLOPs；720p/5s 视频 432K token、注意力矩阵 373 GB |
| `flow_vs_diffusion.py` | numpy | 七 | FM 4 步 0.106 vs DDIM 0.335；10 步打平；边缘轨迹长度比 FM 1.52 / DDIM 1.22 |
| `distill_toy.py` | torch | 八 | 独立配对一步半径 0.019（塌）；轨迹配对 1.985；速度蒸馏 1/4/10 步半径 0.099/1.698/1.892 |
| `fid_bias.py` | numpy | 十二 | 同分布 FID：n=100 时 22.1、n=10000 时 0.22；系数 c≈0.541 d²/n |

## 运行

```bash
cd code/diffusion
for f in *.py; do
  echo "== $f =="
  python3 "$f" --test || exit 1
done
```

`ddpm_toy.py` 和 `distill_toy.py` 需要 PyTorch（CPU 即可，约 5–25 秒）；其余只需 NumPy 或标准库。
所有脚本固定随机种子；`--test` 只断言结构性结论（如「L2 落在中间的比例 > 50%」），不断言具体小数。

## 几个可复现的数字

```text
# schedules.py
线性: SNR>=0dB 259/1000, alpha_bar<1e-3 175/1000, alpha_bar_T=4.036e-05
cosine: SNR>=0dB 496/1000, alpha_bar<1e-3 21/1000, alpha_bar_T=2.429e-09
t=500: linear alpha_bar=7.859e-02, cosine=4.938e-01

# ddpm_toy.py（训练 3000 步）
loss 0.2291
1000 步: 8/8 峰, 半径 2.031
 100 步: 8/8 峰, 半径 1.561
  20 步: 8/8 峰, 半径 1.337
t=900 x0 误差放大 60.27x；t=1000 157.41x

# sampler_zoo.py（sliced W2）
DDIM   5 步 0.234 | 10 步 0.092 | 20 步 0.091
Euler  5 步 0.197 | 10 步 0.094 | 20 步 0.082
Heun   5 步 0.156 | 10 步 0.095 | 20 步 0.080

# cfg_toy.py
s=1.0 少数峰 39.8% mean 1.397 std 0.495
s=7.5 少数峰  5.6% mean 0.959 std 0.277
s=15  少数峰  0.5% mean 0.872 std 0.087

# latent_budget.py
DiT-XL/2 @256: 118.6 GMACs (237 GFLOPs)
DiT @1024 p=2: 5.82 TFLOPs/step; 30 步+CFG 349 TFLOPs -> 1.01s @35% MFU
Wan-VAE 4x8x8/16ch: 432K token; full attention 373 GB (单头 bf16)

# flow_vs_diffusion.py
K=1 FM 1.416 / DDIM 1.404
K=4 FM 0.106 / DDIM 0.335
K=10 FM 0.087 / DDIM 0.092
路径长度/首尾距离: FM 1.524, DDIM 1.223

# distill_toy.py
教师 50 步: W2 0.124, 半径 1.995
独立配对 1 步: W2 1.406, 半径 0.019
轨迹配对 1 步: W2 0.120, 半径 1.985
速度蒸馏 1/2/4/10 步: 半径 0.099/1.081/1.698/1.892

# fid_bias.py
n=100 同分布 FID 22.096; n=10000 0.222
实测系数 c = 0.541（各向同性 d=64）
```

## 说明

- 所有脚本都是**玩具模型**，用来验证量级和趋势，不是论文复现。
- 不依赖 GPU、不下载数据、不访问网络。
- 文章里的论文数字来自 arXiv 原文（DDPM 2006.11239、DDIM 2010.02502、Score SDE 2011.13456、ADM 2105.05233、CFG 2207.12598、LDM 2112.10752、DiT 2212.09748、SD3 2403.03206、CM 2303.01469、LCM 2310.04378、DMD 2311.18828、Wan 2503.20314）。
- 代码里的 DiT MACs 用 fvcore 口径（一次乘加记 1），和论文的 118.6 Gflops 对齐；写成 FLOPs 时要乘 2。
