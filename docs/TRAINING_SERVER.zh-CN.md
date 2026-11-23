# 数弈服务器训练（历史版本，已被 docs/SERVER_PLAN.zh-CN.md 取代）

> **注意**：这份文档描述的是旧流程（CPU 自对弈 + Python 并行多进程 + 时间预算）。
> 现在所有产生局面的步骤都在 `shuyi_selfplay_gpu`（C++ 树 + 批量 CUDA）里，
> 预算是访问次数而不是毫秒，生成分三阶段（随机 → 一层扫描 → MCTS）。
> 现行操作手册见 [SERVER_PLAN.zh-CN.md](SERVER_PLAN.zh-CN.md)，本文件仅作历史记录。

这份文档描述的是**替代 180 局基线**的正式训练流程。目标不是再手工跑一次
`selfplay + train`，而是让服务器在无人值守的情况下完成：

```text
MCTS/MCTS-NN 自对弈 → 追加回放数据 → GPU 训练 50+ epoch
      → 候选模型 vs 上一代 40 局 → 胜率达标才晋升
      → 下一代
```

默认 15 代，每代 800 局自对弈，约 **43,000 个局面/代**，60 个 epoch；
15 代累计生成约 65 万局面，回放窗口保留最近 4 代。

> “局面”指一个带 policy target / value target 的训练样本。每局通常 52–54 个
> 样本，所以 800 局 ≈ 4.3 万局面，远高于 10,000 的下限。

## 1. 机器要求

| 项目 | 建议 |
|---|---|
| 系统 | Ubuntu 22.04 / 24.04，x86_64，Python 3.10–3.12 |
| GPU | 任意支持 CUDA 12.4 的 NVIDIA GPU；8 GB 显存足够 32×4 模型 |
| 驱动 | `nvidia-smi` 正常，驱动支持 CUDA 12.4 |
| CUDA | 12.4 toolkit（`nvcc --version`）；只有 PyTorch 训练时也可用 wheel 自带运行时 |
| CPU | 16 核以上；自对弈是 CPU 并行的，核越多越快 |
| 内存 | 32 GB 以上；`--max-samples 400000` 时训练进程约 8–12 GB |
| 磁盘 | 20 GB 以上；每代数据约 130 MB（800 局 × 54 × 3 KB） |

服务器不需要浏览器、Node、Web runtime，也不需要 Windows/GDI。

## 2. 首次安装

把仓库放到服务器（路径不要含空格更省事），然后：

```bash
cd /path/to/shuyi
bash training/setup_ubuntu.sh
```

如果仓库还在本地，可以用 `scp`/`rsync` 上传；也可以用仓库里的可选助手
（只在本机安装 `paramiko`，密码从环境变量读取，不写进命令历史）：

```bash
python -m pip install paramiko
export SHUYI_SSH_PASSWORD='...'
python training/remote_deploy.py \
  --host 1.2.3.4 --user ubuntu --remote-dir /home/ubuntu/shuyi \
  --setup --start --workers 24
```

助手会排除 `build/`、`runs/`、`.venv/` 和 `data/*.bin`，在服务器上重新构建。

脚本会：

1. 检查 `nvidia-smi` / `nvcc` / `python3` / `cmake` / `ninja` / `g++`；
2. 创建 `.venv`；
3. 安装 **CUDA 12.4 的 PyTorch 2.4.1**：
   `--index-url https://download.pytorch.org/whl/cu124`；
4. 安装 numpy / tensorboard；
5. 用 Release + Ninja 构建 C++ 引擎、`shuyi_selfplay`、`shuyi_match`；
6. 跑 `ctest`；如果 `torch.cuda.is_available()` 为 `False`，脚本直接失败，
   避免在 CPU 上误跑 15 代。

如果系统没有基础包：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv cmake ninja-build g++
```

验证 GPU：

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

必须看到 `True` 和 GPU 型号。若为 `False`，不要开始 15 代训练，先修驱动/CUDA wheel。

## 3. 启动 15 代训练

```bash
source .venv/bin/activate
bash training/run_15gen.sh
```

默认参数：

| 参数 | 默认值 |
|---|---:|
| 代数 | 15 |
| 每代自对弈 | 800 局 |
| 每代最少局面 | 10,000（`--min-positions`，低于则报错） |
| 每手时间 | 120 ms |
| 自对弈并行进程 | CPU 核数 − 2 |
| 训练 epoch | 60 |
| 模型 | 32 通道 × 4 残差块 |
| 回放窗口 | 最近 4 代 |
| 训练样本上限 | 400,000 |
| 晋升对局 | 40 局，100 ms/手 |
| 晋升阈值 | 候选得分率 ≥ 0.55 |
| 阶段 ELO | 每 5 代一次，4 局/对，20 ms/手 |

也可以显式指定：

```bash
python training/pipeline.py \
  --run-name run_20261201 \
  --generations 15 \
  --games-per-gen 800 \
  --workers 24 \
  --time-ms 120 \
  --epochs 60 \
  --channels 32 --blocks 4 \
  --max-samples 400000 \
  --replay-gens 4 \
  --promotion-games 40 \
  --promotion-time 100 \
  --device cuda
```

### 第 0 代特殊在哪里

第 0 代没有上一代 NN，使用**普通 MCTS（无 NN）** 生成 bootstrap 数据，
然后从随机初始化训练第一个模型。第 1 代开始才用 `mcts_nn` 自对弈。

### 晋升规则

每代训练出 `candidate.bin`，与上一代 `models/gen_XXX.bin` 打 40 局、
先后手交替、2 手随机开局：

- 得分率 ≥ `--promotion-threshold`：候选晋升为 `models/gen_XXX.bin`；
- 否则：保留上一代模型作为本代最终模型，候选仍存档在 `gen_XXX/candidate.bin`。

这不是“训练一定变强”的假设，而是防止一代坏数据把主线模型带崩。

### 阶段性 ELO

`--elo-every N`（默认 5）会在第 0、N、2N… 代以及最后一代，用
`shuyi_tournament` 自动跑一次 `random / pruning / mcts / mcts_nn` 的小型
循环赛，随机基准仍固定 ELO 0：

```bash
python training/pipeline.py ... \
  --elo-every 3 --elo-games 4 --elo-time 20
```

每次结果写入 `runs/<run>/elo_gen_XXX/ELO_REPORT.md` 和 `elo_results.csv`，
`mcts_nn` 的 ELO/95% CI 同时写进 `manifest.json` 的 `elo` 字段。它用于看
**趋势**；每对 4 局的标准误很大，不能当最终强度结论。需要正式对比时，最后
单独跑更多局。

## 4. 目录与恢复

```text
runs/<run-name>/
  manifest.json                 运行配置、每代样本数、训练摘要、晋升结果
  gen_000/
    selfplay/shard_*.bin        自对弈数据（v2 格式）
    logs/selfplay_*.log
    logs/train.log
    candidate.bin
    checkpoints/last.pt         训练断点（含 optimizer）
    checkpoints/last.pt.best
    train_metrics.csv
    train_summary.json
    arena/match.json            与上一代的 40 局结果
    arena/games.csv
  gen_001/ ...
  models/gen_000.bin ...        每代最终模型（C++ 可直接加载）
```

断点续跑：

```bash
python training/pipeline.py --run-dir runs/run_20261201 --resume ...
```

- 已完成的代直接跳过；
- 如果某代训练到一半，`checkpoints/last.pt` 存在时会 `--resume` 继续优化器状态；
  该代自对弈会重新生成（当前实现不缓存“未完成代”的中间训练状态），最稳妥是
  让一代完整跑完再中断。

查看进度：

```bash
tail -f runs/<run-name>/gen_003/logs/train.log
tail -f runs/<run-name>/manifest.json
nvidia-smi
```

## 5. 时间估算

自对弈是 CPU 并行的，单局成本约：

```text
54 手 × 120 ms = 6.5 s/局（单核）
800 局 / 24 进程 ≈ 217 s ≈ 3.6 min
```

但 32×4 的 CPU 推理比 16×2 慢，实际以机器为准。训练 32×4、40 万样本、
60 epoch 在单张 8 GB GPU 上通常是几分钟到十几分钟。加上 40 局晋升对局，
一代大约 10–20 分钟；15 代约 **3–5 小时**。如果 CPU 核少，先跑：

```bash
bash training/run_15gen.sh --games-per-gen 300 --workers 8
```

`--min-positions` 默认 10,000；做更小的链路调试时显式加 `--min-positions 0`，
否则每代会被硬性拦下。

确认全链路后再放大。

本机做过一次单代规模烟测（仅证明吞吐/流程，不代表模型强度）：

```bash
python training/pipeline.py --generations 1 --games-per-gen 250 \
  --min-positions 10000 --workers 8 --time-ms 5 --epochs 60 \
  --channels 32 --blocks 4 --device cuda
```

结果：**13,000 个局面、250 局、60 epoch 完成**；32×4 训练在单张消费级
GPU 上约 1 分钟（首 epoch 有 CUDA 预热）。这次开局诊断 `corner=0.649`，
说明只训练一代 bootstrap 仍会偏角；正式 15 代要看曲线是否下降，而不是
用单代结果判断。

## 6. 训练脚本细节

`training/train_resnet.py` 相对第一版的关键变化：

- **数据分片**：`--data` 接受多个 `.bin`，按代拼成回放窗口；
- **合法着法掩码**：policy softmax 只在“空格 + 该数字仍有剩余”的 729 个
  动作上归一化，避免把概率浪费在非法动作；
- **8 重对称增强**：棋盘与 policy target 使用同一个旋转/翻转，手牌/行棋方不变；
- **按对局切分验证集**：v2 数据带 `game`，同一局的相邻局面不会同时进
  train/val；
- **AMP + cosine LR + warmup + grad clip**；
- **warm start**：`--init-model 上一代.bin`，不是每代从零开始；
- **checkpoint/resume**：保存 optimizer，支持中断续训；
- **指标**：policy loss、value MSE、合法 top-1/top-5、value MAE。

模型结构仍是 KataGo 风格 ResNet：

```text
19 输入平面（双方 1–9 石头 + 行棋方）
→ stem 3×3
→ N 个 residual block（conv3×3-ReLU-conv3×3 + skip）
→ policy 1×1（9 个数字 logits/格）+ 手牌条件 bias
→ value 1×1 池化 + 22 维全局特征 → FC → sigmoid
```

二进制格式 `SHUYINN2` 头部记录 `channels / blocks / inputs / globals`，
C++ `PolicyValueNet` 运行时按头部加载。因此服务器训练的 32×4 模型可以
直接被 Windows UI 的 `mcts_nn` 使用；旧的 16×2 `shuyi_resnet_v1.bin`
也仍能加载。

## 7. 自对弈数据格式（v2）

```text
magic "SYSELF2\0"
uint32 version = 2
uint32 samples
per sample:
  uint8  side
  int8   board[81]
  uint8  hand[18]
  float  policy[729]
  float  value
  uint32 game
  uint8  ply
```

`game` 是全局唯一的对局号（pipeline 用 `gen*1e6 + worker*1e4` 作为基址，
再由 selfplay 加局号），`ply` 用于全局特征。v1 文件仍可被训练脚本读取。

自对弈生成器新增：

- `--opening N`：随机开局手数，增加开局多样性；
- `--root-children N`：根候选宽度（默认 32；普通 MCTS / MCTS-NN 都可调）；
- `--game-offset N`：多进程 shard 的对局号基址；
- `--max-plies N`：调试用截断；
- 根节点 Dirichlet 噪声 α=0.3、权重 0.25；前 12 手温度采样，之后取访问最多。

## 8. 开局面诊断

每代训练后流水线会跑一次 `training/diagnose_openings.py`，把开局前 10 手的
policy 首选格分成：

- `corner`：四个角；
- `edge`：非角边界；
- `inner`：非中心的内部；
- `center`：中间 5×5。

结果写入 `gen_XXX/openings.json`，并在日志打印。若 `corner` 长期很高，说明
policy 仍偏好边角，应优先增加自对弈数据/时间、检查 policy target 或加入
开局课程，而不是只增加 epoch。

单独运行：

```bash
python training/diagnose_openings.py \
  --model runs/<run>/models/gen_014.bin \
  --data runs/<run>/gen_014/selfplay/shard_*.bin \
  --channels 32 --blocks 4 --ply 10
```

## 9. 评估与后续

- 每代晋升只看对上一代的 40 局，置信区间很宽；不要把单代 0.55 当作真实
  ELO 提升。正式结论用 `shuyi_tournament`/`shuyi_match` 跑更多局。
- 服务器训练完成后，把 `runs/<run>/models/gen_014.bin` 复制回
  `models/shuyi_resnet_v2.bin`，在 UI 里选择 MCTS+NN 即可；
- 如果 final ELO 仍不理想，优先增加 `--games-per-gen`、`--time-ms` 和
  `--promotion-games`，而不是只堆 epoch。自对弈数据质量比重复拟合旧数据更重要。

## 10. 当前限制（如实说明）

- 自对弈是 **C++ CPU 多进程**，不是 GPU batch MCTS；GPU 只用于训练。
  32×4 模型在 CPU 上每步评估比 16×2 慢，这是当前吞吐的瓶颈。
- 晋升对局只有 40 局，统计噪声大；它只用于防止明显退化，不是论文级评测。
- 第 0 代用普通 MCTS bootstrap，前期模型的 policy target 质量有限；
  后续代际会逐步改善。
- 这套流程不会自动上传模型到 Windows 机器；需要手动复制或用 `scp`。
