# 服务器预案（GPU 全流程）

这份文档是给"租一台 2 小时 GPU 机器"准备的操作手册 + 预算模型 + 质量闸门。
所有产生局面的步骤（自对弈、一层扫描、对局评估）都由同一个 C++/CUDA 二进制完成，
训练由 `training/train_resnet.py` 在同一块 GPU 上完成，**流程里没有 Python MCTS，也没有 CPU 推理**。

---

## 0. 先总结之前的错误（不要再犯）

| # | 错误 | 后果 | 现在的做法 |
|---|------|------|-----------|
| 1 | 引擎里根本没有 CUDA 后端，自对弈只能用 Python MCTS 顶 | 用 Python 的开销解释"GPU 慢"，架构性错误 | `engine/src/nn_cuda.cu` + `tools/selfplay/selfplay_gpu.cpp`：C++ 树 + 批量 CUDA 推理 |
| 2 | 用**时间**而不是**访问次数**做自对弈预算 | 10 ms/步 = 11 次模拟，目标熵 2.48≈均匀、最大概率 0.0865 **低于**均匀 0.090，两代数据不可学 | `--visits` 是唯一预算，`tools/selfplay/selfplay_gpu.cpp` 里没有 `--time` |
| 3 | 以为"GPU 推理慢"，其实是我 kernel 只有峰值 7% | 误判瓶颈，方向全错 | 换 im2col + cuBLAS GEMM：b6c64 从 1158 → 4813 evals/s（本机 MX350） |
| 4 | 在本机上把本机数字当项目数字，没有为租的机器准备 | 时间全花在错误的机器上 | 本文件 + `--plan`：先测，再按预算排生成数 |
| 5 | 从退化教师（b2c16，98.3% 概率压在一个数字上）蒸馏/截断先验 | 目标按扫描顺序排列，不可学（arena 0.308） | gen0 只从随机对局学价值，不复用任何教师 |
| 6 | 只看 loss 不看 arena / 不看目标分布 | 训练指标正常但棋力不涨 | 每代都有目标健康闸门 + 换色对局评估 |
| 7 | `play_batch` 的胜负符号写错（value ∈ {-1,0,0.5,1,2}） | 价值头学到垃圾 | 现在 value 只有 {0, 0.5, 1}，且有闸门检查分布 |

另外两个隐性问题也一并修了：37 平面布局里"走子方"平面写在 36（覆盖了黑方"9 号还剩几个"），
以及 `pipeline.py` 里 `--visits` 参数压根没定义。

---

## 1. 谁在算什么

| 步骤 | 程序 | 算力 |
|------|------|------|
| 随机对局（gen0，只学价值） | `shuyi_selfplay_gpu --mode random` | CPU，<5 s |
| 一层全宽价值扫描（gen1..K） | `--mode oneply` | GPU，每局面 729 次前向 |
| MCTS 自对弈（gen K+1..） | `--mode selfplay --visits N` | GPU，每次模拟一次前向 |
| 对局评估（换色） | `--mode match` | GPU |
| 训练 | `training/train_resnet.py` | GPU（PyTorch，AMP） |
| 吞吐基准 | `--mode bench` | GPU |

`--mode oneply` 的语义：把当前局面的**每个合法着法**都走一步，用价值网评估子局面，
取 `Q = 1 - V(子)`, 目标 `p ∝ exp((Q - maxQ) / T)`, `T = 0.5 × std(Q)`。
它是"深度 1、宽度全开"的搜索：不依赖先验，所以不会被先验的顺序污染。

---

## 2. 服务器准备

```bash
# 1) 传代码（不要传 build/ 和 runs/）
rsync -av --exclude build --exclude runs --exclude data ./ user@host:~/shuyi/

# 2) 一条命令：检查驱动/CUDA、编译、跑单测、CUDA 对比 CPU、校验平面布局、测吞吐
cd ~/shuyi && bash training/server_setup.sh
```

它会依次做：

1. `nvidia-smi`，确认驱动和算力；
2. `nvcc`（没有就 `apt install cuda-toolkit-12-4`）；
3. `cmake -DSHUYI_CUDA=ON`，`SHUYI_CUDA_ARCH=native` 只为本机 GPU 生成代码（编译快）；
4. `shuyi_tests`；
5. `shuyi_selfplay_gpu --check`：规则与 `Position` 逐手比对 + CUDA 与 CPU 前向对比（必须 <2e-3）；
6. `training/check_layout.py`：引擎喂给网络的平面 == 训练时的平面；
7. `--mode bench`：打印每个 batch 的 evals/s。

**第 5、6 步任一失败就不要开跑**，那是数值/特征错误，跑多久都是废数据。

---

## 3. 预算模型

设测得的前向吞吐为 `E`（evals/s，取 `--mode bench` 里你实际要用的 batch-games 对应的那一行），
一局 54 手，每代 `G` 局：

```
一层扫描：  T_oneply = G × 54 × 729 / E
MCTS：      T_mcts   = G × 54 × visits / (0.85 E)      # 0.85 是树/采样的开销
训练：      T_train  ≈ 步数 × 单步时间
            步数 = epochs × ceil(0.94 × G × 54 / batch)
```

本机 MX350（2 GB）实测，供量级参考：

| 网络 | batch=64 | batch=1024 |
|------|---------:|-----------:|
| b2c16 | 21,843 evals/s | 30,835 evals/s |
| b6c64 | 4,122 evals/s | 4,813 evals/s |

MX350 是 1.7 TFLOPS 的入门卡。4090（82 TFLOPS + TF32）同 kernel 大约 2 个数量级，
即 b6c64 在 10 万 evals/s 量级。**具体数字以 `--plan` 的输出为准，不要照抄本表。**

用 `--plan` 直接得到本机的时间表：

```bash
python3 training/pipeline.py --plan --channels 32 --blocks 4 \
        --games-per-gen 2000 --visits 2000 --oneply-gens 2
```

---

## 4. visits 到底要多少（400 不够）

开局分支是 **729**（81 格 × 9 个数字），中局仍有 300~400。所以：

- `visits = 400` ⇒ 每个着法平均 0.5~1.2 次访问。第一次访问按先验顺序展开一遍，
  之后几乎没有加深；目标分布基本由"先验顺序"决定，而不是由局面决定。
- 判据：**每个保留的根候选至少 20 次访问** ⇒ `visits ≥ 20 × rootK`。
  `rootK = 32` → 640 起；`rootK = 0`（全宽）时在中局要 7000 次，不现实。

实测（弱价值网 value0，b2c16，本机），`cell_ratio` 是"目标是否能区分格子"的指标
（1.0 = 格子完全无信息，正是之前废掉两代的那种数据）：

| 配置 | 熵 | 峰值 | 峰值/均匀 | 单数字质量 | cell_ratio |
|------|----|------|----------|-----------|-----------|
| MCTS 64, root-k 8 | 2.00 | 0.232 | 79× | 0.82 | 0.448 |
| MCTS 400, root-k 32 | 3.23 | 0.127 | 43× | 0.74 | 0.720 |
| MCTS 400, root-k 0 | 4.98 | 0.090 | 31× | 0.36 | **0.846** |
| MCTS 2000, root-k 0 | 4.40 | 0.212 | 73× | 0.48 | 0.682 |
| **oneply（729 全宽）** | 4.16 | 0.131 | 45× | 0.51 | **0.739** |

读法：

1. `400 + 全宽根` 的 `cell_ratio` 最差（0.846）——400 次访问摊到 340 个着法上什么也没搜到；
2. `2000 + 全宽根` 才接近一层扫描的质量，但贵 5 倍；
3. 一层扫描用 729 次**前向**就拿到更好的目标，比 MCTS 便宜得多。

**结论（写进默认值）：**

- gen0：随机对局；gen1..`--oneply-gens`（默认 2）：一层扫描；
- 之后 MCTS `visits ≥ max(2000, 20 × rootK)`，且先验够尖之前保持 `--root-k 0`；
  先验变尖（目标 `cell_ratio < 0.7`、峰值 > 40× 均匀）之后才允许 `--root-k 32 / --visits 1024`；
- 评估（arena）必须用和自对弈相同的 visits，否则测的是另一个棋手。

### 4.1 一层扫描的前提：价值网要"看得见格子"

一层扫描把 `V(子局面)` 变成 policy。如果价值网几乎不依赖落子位置，目标就会退化成
"选对数字、格子随机"——和之前废掉的两代一模一样。实测：

| gen0 数据量 / epoch | number_mass | cell_ratio | 闸门 |
|---------------------|------------:|-----------:|------|
| 648 局面 / 2 epoch | 0.69 | 0.99 | 拒绝 |
| 16.2 万局面 / 8 epoch（3000 局） | 0.51 | 0.74 | 通过 |
| 16.2 万局面 / 40 epoch | 更低 | 更低 | 通过 |

所以 **gen0 至少 2000~3000 局、20+ epoch**。如果 gen1 被闸门拒绝，要加大的是 gen0
（局数/epoch），不是 `visits`——错误信息也会直接这么说。

---

## 5. 一次 2 小时的排布（示例，用 10 万 evals/s 的卡）

| 阶段 | 内容 | 时间 |
|------|------|------|
| gen0 | 2000 局随机对局 + 训练 40 epoch | ~5 min |
| gen1 | oneply 2000 局（7870 万次前向）+ 训练 + arena 160 局 | ~15 min |
| gen2 | oneply（同上） | ~15 min |
| gen3 | MCTS 2000 visits（2.16 亿次模拟）+ 训练 + arena | ~45 min |
| gen4 | MCTS 同上 | ~45 min |

卡慢（3 万 evals/s）就把 `GAMES` 降到 1000，或者只跑 gen0..gen2；
**不要**为了多跑一代去降 `visits`，那正是之前浪费两代的做法。

```bash
GENERATIONS=5 ONEPLY_GENS=2 GAMES=2000 VISITS=2000 \
  bash training/run_generation.sh
tail -f runs/run_*/pipeline.log
```

---

## 5.5 顺次训练 + ELO 标定（2026-11-22 修正）

两件事必须同时成立，否则代际循环会空转：

1. **顺次训练**：第 N+1 代必须用第 N 代产出的模型下棋、并在此基础上训练。
   不能用"上一代赢了才换"的闸门把模型留在旧代——"某个模型在 match 里强"不等于
   "它产生的数据适合训练"，而停在旧代意味着后面每一代都在重放同一个分布
   （srv1 的 gen3..gen6 就是这样全部对着 gen2 的权重打的，白跑两代）。
   现在 `pipeline.py` 的链条**无条件前进**，arena 只作为记录。
2. **强弱用 ELO 标定**，不是 val loss，也不是"和上一代的对局得分"。
   `training/elo.py` 用 `--mode match` 在 GPU 上打循环赛（模型可以不同大小、
   不同输入平面），用带先验的最大似然拟合 Elo，并给出 bootstrap 90% 区间。

```bash
python training/elo.py --models runs/srv1/models/gen_00{0,1,2}.bin     --labels gen0 gen1 gen2 --games 40 --visits 128
python training/elo.py --report runs/elo/matrix.json      # 从已存对局重新拟合
```

部署模型取 ELO 最高者，而不是"最后一代"；`manifest.json` 里记 `elo` 与 `elo_best`。

## 6. 质量闸门

每代在**训练之前**检查目标分布，不达标直接失败（不浪费训练时间，也不会写出废模型）：

| 指标 | 阈值 | 含义 |
|------|------|------|
| `cell_ratio` | ≤ 0.90 | 目标能否区分格子；≈1 就是"只知道下哪个数字" |
| `max_prob × legal` | ≥ 2 | 峰值至少 2 倍于均匀 |
| `entropy / ln(legal)` | ≤ 0.97 | 别太接近均匀 |

失败信息会直接给出该调什么（提高 visits/root-k，或先继续用 oneply）。
阈值可用 `--max-cell-ratio` / `--min-max-prob-ratio` / `--max-entropy-ratio` 调。

另外两个硬性约束：

- **只在 gen0 允许蒸馏**（`--selfplay-model` 已被移除，管线里没有这个入口）；
- **训练窗口默认只含当代数据**（`--replay-gens 1`），要开窗自己加，但不要回放旧的退化数据。

---

## 7. 监控 / 恢复

- `runs/<name>/manifest.json`：每代的模式、目标统计、训练指标、arena 比分、模型名
  （`b{blocks}c{channels}-s{steps}-d{data}`）。
- `runs/<name>/gen_XXX/logs/*.log`：每个子进程的完整输出。
- 中断后 `python3 training/pipeline.py --run-name <name> --resume`：已完成的代直接跳过，
  当前代的训练有 `checkpoints/last.pt` 可续。
- 每个二进制都会打印 `sims/s`、`games/s`，和 `--plan` 的预估对不上就说明卡被抢占/降频了。

---

## 8. 已知欠账

1. **kernel 效率**：b6c64 在本机约 24% 峰值（4813 × 87 MFLOP ≈ 419 GFLOPS / 1.7 TFLOPS）。
   im2col 的显存流量对小网络（b2c16）仍是负担；大网络（b4c32 以上）才是目标。
   服务器上开 TF32（`compute_cap >= 8` 自动开）会再快一截，`--check` 会在 1e-3 内仍然通过。
2. **komi 已定标（2026-11-22 srv1 实跑）**：用训练后的网络自对弈测颜色平衡，
   komi 0 时红胜 52.4%，komi −5 时红胜 10.9% ⇒ 公平 komi ≈ **−0.3**，默认已改为 −0.5。
   之前 "−4.5…−6" 的结论是用退化网络（b2c16，64 visits）标的，已作废。
   注意价值头编码了 komi，换 komi 必须重训。详见 [SRV1_RUN.zh-CN.md](SRV1_RUN.zh-CN.md)。
3. **ELO 未接**：现在只用 arena 胜率（≥0.55 晋升）。CPU 引擎的锦标赛留作离线分析。
