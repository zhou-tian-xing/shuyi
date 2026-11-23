# experimental/ — 自对弈、训练、评测

这里放**不进入 UI 发布**的东西：产数据的自对弈、训练流水线、横评与标定工具，以及几个
一次性的研究程序。它只依赖 `engine/` 的头文件，UI 不依赖它。

先说清楚一件事：**这不是"随便写写"的目录**。`selfplay/selfplay_gpu.cpp` 和
`training/pipeline.py` 是生产链路；`research/` 和若干 `training/*.py` 才是历史遗留。

```
selfplay/
  selfplay_gpu.cpp   生产自对弈：CUDA 批量前向，每 tick 每局下推一个叶子。
                     也提供 --mode match（模型对打，评分用）、--mode serve（给 arena 当玩家）、
                     --mode oneply、--mode bench、--check（CUDA 与 CPU 对拍）
  selfplay.cpp       CPU 版自对弈，慢；只用于无 GPU 的冒烟测试（ctest: shuyi_selfplay_smoke）
  match.cpp          两种引擎对打（早期工具，时间预算，已被 training/arena.py 取代）
  tournament.cpp     池内循环赛（早期工具，时间预算，已被 training/elo.py 取代）

research/
  a9_lemma.cpp       有限引理程序：黑 9 三枚的闭邻域损失下界（文档证明的一部分，ctest 会跑）
  komi_explore.cpp   搜索终局分数见证 → data/score_witnesses.txt
  nnbench.cpp        单模型前向基准

training/            训练与评测的全部脚本；**先读 training/README.md**（原则、闸门、改进记录）
  pipeline.py        一代一代：随机 → 单层扫描 → MCTS 自对弈 → 训练 → arena → ELO
  train_resnet.py    训练器（PyTorch）；模型格式的写出端
  arena.py           横评：random / pruning / mcts / mcts_nn / gpu 同池对战
  elo.py             Elo 拟合（Newton + bootstrap），可合并多份矩阵
  avalue.py          A值标定（模型自对弈扫 komi，logistic 求交点）
  cross_eval.py      线路横评 + 代数爬梯合并成一张表
  evalconfig.py      EVAL_VISITS：**所有评测唯一的预算常量**
  rule_probe.py      独立计分实现（横评判终局用，与引擎互校）
  target_stats.py    目标分布健康检查（闸门）
  model_migrate.py   旧模型头迁移（只改头，不动权重）
  summarize_run.py   manifest → markdown 表
  pull.py            断点续拉服务器产物
  ssh.py / remote*.py / *.sh   服务器部署与运维

tests/
  selfplay_smoke.py  CPU 自对弈冒烟（ctest 调用）
```

## 数据流（一张图）

```
pipeline.py ──▶ selfplay_gpu --mode selfplay ──▶ runs/<ruleset>/<run>/gen_XXX/selfplay/shard_000.bin
     │                                                       │
     │                                                       ▼
     │                                        train_resnet.py ──▶ candidate.bin
     ▼                                                       │
 arena.py（异构对手，评预算 EVAL_VISITS） ◀── gen_XXX vs gen_{XXX-1}（selfplay_gpu --mode match）
     │                                                       │
     ▼                                                       ▼
 elo.py（random=0 锚定，可合并多份矩阵）              models/b4c32-s..-d..-blk.bin
                                                             │
 avalue.py（同一模型自对弈扫 komi → 公平 A值）                 ▼
                                                     UI 的“MCTS + 网络”菜单 / arena 的 gpu: 玩家
```

## 三条硬规则（训练手册里有完整版）

1. **棋力只认 ELO，不认 val loss。** 历史上 val_policy 降了 1 分而 ELO 没涨。
2. **ELO 永远以 `random = 0` 为基准**，高胜率用相对 ELO 表达。
3. **评测预算固定为 visits**（`evalconfig.py` 的 `EVAL_VISITS = 512`），同一张表里
   所有参赛者必须是同一个数；`elo.py --report` 会拒绝合并预算不同的矩阵。

## 常见操作

```bash
# 训练（在本机或服务器上；服务器产物记得 pull.py 拉回本地）
python experimental/training/pipeline.py --run-name blk1 --ruleset blocking --generations 8

# 横评（先 export OMP_NUM_THREADS=1，否则多个 GPU 玩家会抢核假死）
export OMP_NUM_THREADS=1
python experimental/training/arena.py --players random pruning mcts \
    mcts_nn:runs/blocking/blk1/models/gen_005.bin \
    gpu:runs/blocking/blk1/models/gen_005.bin \
    --games 6 --visits 512 --ruleset blocking --out runs/arena/blocking512/table.json
python experimental/training/elo.py --report runs/arena/blocking512/table.json --anchor random

# A值标定
python experimental/training/avalue.py --model runs/blocking/blk1/models/gen_005.bin \
    --ruleset blocking --komis "-2,-1,0,1" --games 240

# 把服务器上的东西拉回来（断点续传，训练中就能跑）
SHUYI_PASS=... python experimental/training/pull.py \
    --remote /root/blk/runs/blocking/blk1 --local runs/blocking/blk1
```

## 历史遗留脚本

`training/` 里的 `gpu_selfplay.py`、`oneply_selfplay.py`、`komi_probe.py`、`diag_target.py`、
`diagnose_openings.py`、`fix_values.py`、`verify_rules.py`、`check_layout.py` 是当时的排查/参考实现，
Python 侧的主流程已经被 C++ 取代。**新代码不要再依赖它们。** 确认没用的就直接删。
