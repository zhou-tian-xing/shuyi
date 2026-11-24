# 训练手册（数弈）

这份文件是训练相关的**唯一入口**：原则、流程、评估口径、以及每次改进的记录。
改训练相关的东西时，先把改动写进第 6 节的记录，再改代码。

文件都在这个目录下（`training/`），需要完整文件地图看
[../../docs/MAINTAINERS.zh-CN.md](../../docs/MAINTAINERS.zh-CN.md)：

| 要做什么 | 用哪个文件 |
|---|---|
| 跑一代代训练 | `pipeline.py`（编排）、`train_resnet.py`（训练器） |
| 产自对弈数据 | `../tools/selfplay/selfplay_gpu.cpp`（C++/CUDA，不随产品发布） |
| 横评与 Elo | `arena.py`（异构玩家）、`elo.py`（拟合/合并）、`cross_eval.py`（合成一张表） |
| 标定 A值 | `avalue.py` |
| 评测预算常量 | `evalconfig.py`（`EVAL_VISITS`，**唯一**允许的预算来源） |
| 计分互校 | `rule_probe.py`（独立实现） |
| 目标分布闸门 | `target_stats.py` |
| 把服务器产物拉回本地 | `pull.py` |

---

## 1. 两条硬性原则（用户定，不要违背）

### 1.1 连续 10 代最高 ELO 没有提升 → 换更大的模型

单代噪声很大（160 局 arena 的 σ≈4%，200 局 head-to-head 的 σ≈3.5%），
所以"这代没涨"不能作为换模型的理由；但**连续 10 代**最高 ELO 都没超过历史最高
（超过阈值为 `--plateau-margin`，默认 +20 Elo），说明瓶颈在容量或数据分辨率，
而不是运气。

实现：`training/pipeline.py` 的 `--plateau-generations 10 --plateau-margin 20`，
触发后按 `--capacity-ladder "b4c32,b6c48,b6c64,b8c96"` 升一档，
manifest 里记 `capacity_history`。宽度变了不能热启动，所以升级是**开新 run、用已有数据冷启动**。

实测参考（srv1，2 小时 09 分，7 代）：gen2 之后 gen3..gen6 的 ELO 全部重叠在 +320…+375，
也就是**第 2 代之后就是平台**——按这条原则，那种情况该考虑换更大的模型，而不是继续刷代数。

### 1.2 ELO 永远以"随机落子 = 0"为基准，高胜率用相对 ELO 表达

评分不是"赢了上一代多少"，也不是 `val_loss`，而是同一池子里的 Elo 标定：

- `random`（均匀随机）**固定为 0**，不参与拟合；其他模型的分 = 相对随机落子的 Elo。
- 这样跨 run、跨模型大小、跨规则都能比较（只要锚点相同）。
- 接近 100% 胜率时不要写"100%"，写**相对 ELO 与下界**（如 `≥ +900`）：
  全胜样本没有失败可重采样，bootstrap 区间会退化，此时只有下界是诚实的。
- 实现：`training/elo.py --anchor random`；`training/arena.py` 自动加入 random 锚点。

绝对 Elo 依赖参赛池和先验，**只有同一池内的差距有意义**。

---

## 2. 数据从哪来

一个二进制，五种模式（`tools/selfplay/selfplay_gpu.cpp`，C++ 树 + 批量 CUDA）：

| 阶段 | 命令 | 作用 |
|------|------|------|
| 引导 | `--mode random` | 随机对局，只教价值头；policy 目标老实地均匀 |
| 引导 | `--mode oneply` | 一层**全宽**价值扫描：每个合法着法走一步、用价值网评估，`p ∝ exp((Q-max)/T)`，`T=0.5·std(Q)` |
| 主循环 | `--mode selfplay --visits N` | Dirichlet-MCTS 自对弈，根访问分布作 policy 目标 |
| 评估 | `--mode match` | 换色对局，GPU 上跑 |
| 评估 | `--mode serve` | 走行协议，和 CPU 引擎同池对战 |

为什么先 oneply：开局分支是 729（81 格 × 9 个数字），400 次访问摊下去只有 0.5~1.2 次/着法，
目标几乎等于先验顺序；实测 128 visits 在**阻挡规则**下 `cell_ratio` 已经 0.557，
而经典规则 400 visits 全宽根是 0.846（1.0 = 只区分"下哪个数字"）。

`visits` 的规则：**≥ 20 × root-k**；先验没变尖之前保持 `--root-k 0`。

---

## 3. 每代都过闸门（训练之前）

`training/pipeline.py` 在训练前检查目标分布，不达标直接失败（不浪费训练时间）：

| 指标 | 默认阈值 | 含义 |
|------|---------|------|
| `cell_ratio` | ≤ 0.90 | 目标能否区分格子；≈1 就是"只知道下哪个数字" |
| `max_prob × legal` | ≥ 2 | 峰值至少 2 倍于均匀 |
| `entropy / ln(legal)` | ≤ 0.97 | 别太接近均匀 |

被拒时错误信息会说明该调什么（提高 visits/root-k，或加大 gen0）。

---

## 3.4 按窗口定规模：`training/run_session.sh`

租来的卡有一个固定的钟，**窗口决定能跑几代，模型规模决定每代要多少数据和多少步**。
这两件事不该由手输的数字决定——375 局/代的教训就是手输。

```bash
HOURS=10 MODEL=runs/training/<run>/models/b6c96-s25920-d258451.bin \
RUN_NAME=b6c96-nopass-20261125a bash training/run_session.sh
```

| 步骤 | 做什么 |
| --- | --- |
| 探针 | 用真实模型、真实 visits 跑 24 局，测出这台机器的 `games/s` |
| 定规模 | `training/budget.py`：目标步数 = 10,000 × (M/1.03)，训练量按回放窗口 ×1.8 计，再算「几代 × 每代几局 × 几个 epoch」，写进 `plan.json` |
| 跑 | `pipeline.py ... --max-hours H`：只**在代与代之间**停，剩的时间不够下一代的 1.15 倍就收尾，manifest 里记 `stopped_early` |

`budget.py` 的算术有单测（`training/tests/test_budget.py`，进 ctest）：一代少于 800 局直接
报错，不会悄悄缩成"跑着像训练、其实是噪声"的小代。

---

## 3.5 排行榜：`training/leaderboard.py`

数据就是一张二维 CSV，`runs/leaderboard/matches_<visits>.csv`：

```
model,random,b4c32-s5960-d80938,b6c96-s13520-d183077
random,—,0-0-120,0-0-120
b4c32-s5960-d80938,120-0-0,—,87-0-33
```

格子是行方对列方的 `胜-和-负`，空格 = 没打过，对角线 `—`。想改结果就直接改这个
文件（改一个三角就够，另一半空着会自动镜像；两边矛盾会警告）。程序**只读表算分**，
不自己收集对局：

```bash
python training/leaderboard.py                 # 读所有 matches_*.csv，输出排行榜
python training/leaderboard.py --visits 128    # 只看一个预算
python training/leaderboard.py --anchor b4c32-s5960-d80938   # 换锚（没有 random 时）
python training/leaderboard.py --all           # 连只有引擎种类的表也显示
```

算法就是 `elo.py` 的同一套：逻辑回归最大似然 + 高斯先验，**锚钉在 0**（默认
`random`），附 200 次 bootstrap 的 95% 区间。每个 `visits` 一张表、各算各的，
**从不混预算**。

**新数据怎么进表**（跑一次预算 = 一批对局 → 填进那张预算的表）：

```bash
# 1) GPU 机器上：所有模型两两对局，一局一对、颜色交替，直接落 pair JSON
python training/roundrobin.py --visits 128 --games 20 --out runs/eval/v128
# 2) 拉回本地
python training/pull.py --remote /root/eval/v128 --local runs/eval/v128
# 3) 填表（已经数过的文件记在 imported_sources.json，重复跑不会重复加）
python training/leaderboard_import.py
# 4) 算分
python training/leaderboard.py --anchor b4c32-s5960-d80938
```

`roundrobin.py` 一个进程就能吃满整台机器的 CPU（实测 8 核全占），所以
`--workers` 默认 1；要并行必须给每个进程 `OMP_NUM_THREADS=1`，否则互相抢核，
总吞吐反而掉十几倍（64 visits 实测：单进程 7.9 局/秒，4 进程不断言 0.8 局/秒）。

`--anchor <模型>`：没有 `random` 的表，用池里最弱的那个当 0 分。**但只有 `random`
当锚才是绝对棋力**：它和搜索量无关，所以换预算不会把零点挪走。用别的锚，池子整体
变强时所有分数会一起"掉"，那是锚在动，不是模型变弱。

所以每张表都应该有 `random`：`roundrobin.py --random-anchor` 让每个模型都和均匀随机
`--random-anchor` 只给**梯子最下面那一档**加一场对随机（不算全场都打）：随机的分数
与搜索量无关，所以它是唯一不动的 0 分；其余各档的绝对分由梯子逐档累加出来。
40 局全胜只能读出下界（40-0 的 95% 下界约 +508），`leaderboard.py` 会把整张表按这个
下界平移，并在表头写明"这是保守下界"；某一档真的输过几局时，那就是点估计。

**表是追加的**：每个来源文件在 `imported_sources.json` 里记一次，重跑不会加两遍，新
对局往上加，旧对局不会被覆盖。来源之间打架时（同一对一次 80% 赢、另一次 80% 输）
照样累计，同时写进 `runs/leaderboard/conflicts.txt` 并把前几条打印出来——不静默丢。

`--compare` 那一节并排列出同一模型在各预算下的分数。**看变化只看这个**：算力变大时
同一模型的绝对分只应该涨或不动。

**同权重、不同预算**（判断"多给算力值不值"最干净的办法，不需要锚）：`--visits-b`
让两侧用不同的访问数，权重完全相同：

```bash
shuyi_selfplay_gpu --mode match --model models/b4c32-s42012-d595499.bin \\
    --model-b models/b4c32-s42012-d595499.bin --visits 64 --visits-b 256 \\
    --ruleset nopass --komi 0.5 --games 200
```

实测（MX350，200 局）：64 visits 对 256 visits 是 **76-124**，即 4 倍算力值约 +85 ELO。
注意预算是跟**玩家**走的，不是跟颜色：先把这条写错过一次，按颜色分会把两侧的
预算对调，结论正好反过来。

表本身会说它哪里不成立：

- 没有锚的表 → 分数只与池内均值可比，不是"比某物高多少"；
- 分成互不对局的几组 → 组间的分数是任意定的，只有组内比较有意义
  （补一场跨组对局就能连上，几十局就够）。

历史评测（32/96/512/800/1024）是一次性从旧 arena JSON 导入的，名字按
`rename_models.py` 之后的累计名对齐；**64 visits 那张因此被整张重建过**：旧格子
自相矛盾（同一对局两个三角给出相反结果），和 2026-11-24 新跑的数据也冲突
（旧表说 `b4c32-s5960` 13-7 赢 `b6c96-s2800`，新的 2100 局表说 0-20）。跨 run 的
旧 JSON 只能当历史参考，以 `runs/eval/` 里按文件名直接产出的数据为准。

## 3.6 改名字：`training/rename_models.py`

`s`/`d` 是**整条链**的累计值：接着 `b4c32-s34760-d470705` 训出来的 gen0 是
`-s36776-`，不是 `-s2016-`。`pipeline.py` 现在从 `--init-model` 的名字里读出起点
（读不出来就报错，而不是从 0 重新数），`rename_models.py` 负责把历史文件对齐：

```bash
python training/rename_models.py            # 只打印计划
python training/rename_models.py --apply    # 改名并同步更新 manifest 里的路径
```

## 3.7 编译与发包：`release/build_release.py`

一条命令做完"编译 → 测试 → 打包"，**包里不带模型**：

```bash
python release/build_release.py --cuda-build build-cuda     # 引擎与界面都取 CUDA 树
python release/build_release.py --cuda-build build-cuda --model models/b4c32-s42012-d595499.bin
```

- 权重是数据，不是产品：模型几乎每天都在变，二进制不是。没有模型也能启动，
  界面会直接说"把 b<blocks>c<channels>-s<steps>-d<data>.bin 放进 models/"。
- `ctest` 不通过就不发包；UI 和引擎必须同时存在（两个程序才是一个产品）。
- CUDA 树在 Windows 上要先 `vcvars64`，所以默认**不替它编译**，只取它的产物；
  `--build-cuda` 才会去编（Linux 上直接可用）。

## 4. 横评（所有线路、所有模型）

### 4.1 预算只有一个数字：visits

评估看的是**模型**，所以每一步的搜索量必须固定、与机器无关。**不看时间**：同一句
`--time 80` 在 3080 和 MX350 上是完全不同的访问数，那样测出来的是主机性能，不是棋力。
唯一允许的预算是 `training/evalconfig.py` 里的常量：

```python
EVAL_VISITS = 512      # 每一步、每个参赛者，都用这个数
```

一个"visit"是什么，按参赛者类型定义（这就是全部契约）：

| 参赛者 | 1 visit = |
|---|---|
| `gpu:<model>` | 一次树搜索模拟 = 一次批量 CUDA 前向 |
| `mcts_nn:<model>` | 一次树搜索模拟 = 一次 CPU 前向 |
| `mcts` | 一次树搜索模拟 = 一次 rollout |
| `pruning` | **一个节点 = 一次叶子评估**（alpha-beta 没有"模拟"，用节点数对齐） |

实施细节（都是为了让数字可复现）：

- `shuyi_engine` 的 `set_visits N` 现在对四种引擎都生效。剪枝引擎以前收到 `visits>0`
  会把 `timeLimitMs` 设成 0 然后**立刻中止**（等于下随机棋），所以旧脚本只能给它按时间跑——
  这是错的。现在 `SearchOptions::nodeLimit` 走节点预算，**完全不看墙钟**。
- 剪枝会搜完当前这一层才停（不能用一个半成品深度的着法），所以节点数会在预算之上溢出一点；
  这个溢出是**确定的**，两次相同输入永远给出相同的节点数与着法（`shuyi_visit_budget` 测试）。
- `arena.py` 里 `--time` 只保留为**挂死保护**（默认 1 小时，正常永远不会触发），
  `--hang-guard-ms` 可调。它必须永远晚于访问预算触发，否则表就变成主机性能表。
- `elo.py --report` **拒绝合并预算不同的矩阵**（旧表 96/128/192 混着合并过，那是错的，
  现在直接报错退出）。

### 4.2 跑法

```bash
export OMP_NUM_THREADS=1     # 必须：多个 serve 进程各吃满线程会假死
python training/arena.py --players random pruning mcts \
    mcts_nn:runs/training/b4c32-blocking-20261122/models/b4c32-s34760-d470705.bin \
    gpu:runs/training/b4c32-blocking-20261122/models/b4c32-s34760-d470705.bin \
    --games 6 --visits 512 --komi 1.26 --ruleset blocking --workers 2 \
    --out runs/probes/arena/blocking512/table.json
python training/elo.py --report runs/probes/arena/blocking512/table.json --anchor random
```

- 每局记录全部着法，终局分用 `training/rule_probe.py` 的计分函数算（和引擎同一公式）。
- 结果 `runs/probes/arena/<ruleset>.json` 可直接喂给 `elo.py --report`（可传多份，预算必须一致）。
- 样本量：50% 附近，20 局 ≈ ±20pp、60 局 ≈ ±12pp、200 局 ≈ ±7pp、400 局 ≈ ±5pp。
  **每对少于 60 局的表只用来排序，不要比小数点。**

### 4.3 512 visits 的实际成本（实测）

| 参赛者 | 每步 | 每局（~52 手） | 机器 |
|---|---|---|---|
| `gpu:*` | ~0.19 s | ~10 s | 本机 MX350 |
| `pruning` | ~2 ms | ~0.1 s | 本机 CPU |
| `mcts` | ~15 ms | ~0.8 s | 本机 CPU |
| `mcts_nn:*`（b4c32，CPU 前向 ~3.6 ms/次） | ~1.9 s | **~100 s** | 本机 CPU |

→ GPU 线路和剪枝/MCTS 线路在 512 visits 下都能跑完整表；CPU 版的 MCTS-NN 每局 ~100 s，
只够跑少数几对（GPU 上同样 512 visits 只要 10 s）。这不是预算问题，是 CPU 前向的固有限制——
所以**不要**为了让它跑快而给它更少的 visits，那样表就不可比了；要么接受它贵，要么上 GPU。

---

## 5. 规则隔离（两套规则并行）

- 模型 `SHUYINN3`、数据 `SYSELF3` 都带 `ruleset` 字节；**加载时双向校验**，
  阻挡模型配经典规则直接拒绝。
- 旧文件用 `training/model_migrate.py --ruleset classic` 显式升级（只改头，不动权重）。
- v2 的 37 平面文件（side 平面写在 36 的旧布局）会被拒绝——那批权重不能用。
- 目录 `runs/training/<arch>-<ruleset>-<YYYYMMDD>/`，模型名
  `b<blocks>c<channels>-s<steps>-d<data>.bin`（规则集在模型头里，不进名字）。

---

## 6. 改进记录

### 2026-11-24 · 第三套规则：无过手（nopass），并设为默认

规则：落子限制与阻挡完全相同，但**没有过手**——轮到一方而它无处可落，棋局立刻终局计分。
所以棋盘可能少于 54 枚收场，“把对手困死”变成真正的终局手段。

实现要点（细节见 `docs/RULES.zh-CN.md` 第 7 节）：

- `Ruleset::NoPass = 2`，`kDefaultRuleset` 改成它；`rulesetBlocks()` / `rulesetHasPass()`
  取代散落的 `== Ruleset::Blocking` 判断（core、引擎、selfplay 三处都有）。
- `Position::gameOver()` 在 NoPass 下多一个条件：`!hasPlacement()`。`hasPlacement()`
  只检查手里最大的数字（更大的都不行，更小的更不行），所以通常是几次比较就短路。
- `searchMoves()` 不再为 NoPass 造“强制过手”节点；`pass()` 直接返回 false。
  如果 selfplay 在非终局遇到无合法着法，直接 abort——那是生成器与规则不一致，不能继续跑。
- `rulesetCompatible(blocking, nopass) == true`：无过手的每一局都是阻挡的一局的前缀，
  所以阻挡模型可以直接下无过手，引擎在 `set_model` 应答里写明。反向不允许。

**踩到的坑（值得记住）**：协议层放行了“阻挡模型 + 无过手”，但 `mcts.cpp` 里判断是否
信任网络的代码还是一句 `nn_.ruleset() == position.ruleset()`。结果模型被悄悄丢掉，
MCTS 用均匀先验 + rollout 下棋，第一次测出来是“模型在无过手规则下 20 局全输给剪枝”，
看起来像“换规则把模型毁了”。实际是引擎没在用模型。修好后同一场比赛 20:0 反过来。
已加回归测试 `testNoPassUsesTheModel`：同一局面在 blocking / nopass 下的根值与最佳着法
必须完全一致（规则集不进网络输入），把无过手静默丢模型的复现路径钉死。

**实测影响**（这是用户的问题：换规则对现有模型影响多大？）：

| 数据 | 对局 | 在 54 手前结束（有强制过手） |
|---|---:|---:|
| 阻挡规则、训练好的模型自对弈（gen_005 分片） | 1500 | 5（0.3%） |
| 阻挡规则、随机落子（gen_000 分片） | 1500 | 20（1.3%） |
| 无过手、训练好的模型自对弈（实跑 200 局） | 200 | 1（0.5%） |

发生时的终局位置在 ply 50–52；分数平均只差 0.8 分，5 局里有 1 局胜负翻转。
也就是说换规则**不需要重训**：模型在无过手下的表现与阻挡下一致（对剪枝 256 与 512
visits 各 20:0）。下一轮训练会直接在无过手下产数据（`--ruleset nopass`），
komi 仍用 1.26——不过要重新标定，因为终局更早、颜色平衡可能变了
（200 局里红 120 / 黑 80，样本太小，不能直接下结论）。

### 2026-11-24 · 第一轮 v4 训练（3090，3 小时，3000 visits/步）

配置：从 `models/b4c32-s34760-d470705.bin`（v3，A 盲）热启动，`--a-cond`、
`--a-anchor 1.26`、`--lr 5e-4`、14 epoch、自对弈 3000 visits/步 × 800 局/代、
`--root-k 32 --top-k 16`、arena 120 局 @3000 visits、ELO 每代一次 @800 visits。

| 代 | 模式 | 局面 | 训练步 | arena 对上一代 | ELO(800 visits) |
|---|---|---:|---:|---|---:|
| 0 | warm（旧数据热启动） | 77,976 | 2,016 | **0.675** vs `s34760` | −140 |
| 1 | selfplay 3000v | 41,597 | 3,080 | 0.650 vs gen_000 | −91 |
| 2 | selfplay 3000v | 41,600 | 1,078 | 0.625 vs gen_001 | **+112** |
| 3 | selfplay 3000v | 41,597 | 1,078 | 0.617 vs gen_002 | +119 |

（ELO 是同池 4 个模型的循环赛，180 局、30 局/对、800 visits，池内均值为 0。）

结论：

1. **热启动第一代就涨**：同一份旧数据、只把价值目标换成 `sign(终局分 + A)` 并把 A 加进
   输入，对原模型 81:39（0.675，120 局 @3000 visits，≈+128 Elo）。所以 v4 不是"为了
   A 读数正确而牺牲棋力"，它同时提了棋力。
2. **每代都还在涨**：0.650 → 0.625 → 0.617，三代各 ≈ +90 Elo，**没有出现上一轮
   "第 2 代之后平台"的现象**。所以这一轮不是容量瓶颈，3000 visits 的搜索质量把
   b4c32 又往前推了一截。
3. **两个口径不一致，要分开读**：800 visits 的 ELO 说 gen_002 与 gen_003 打平
   （+112 / +119），3000 visits 的 arena 说 gen_003 赢 0.617。产品默认预算是 512~800，
   所以"对齐产品"用前者、判断"这一代有没有变强"用后者（后者才是搜索预算的匹配口径）。
4. **吞吐**：3090 上自对弈 0.437 games/s（≈64k sims/s），一代 800 局 × 3000 visits
   约 30 分钟；arena 120 局约 5 分钟；训练 14 epoch 约 2 分钟。一代端到端 ≈ 40 分钟。
   `--plan` 当初估 25.7 分钟/代，实际 30 分钟，误差来自它假设 sims/s ≈ 0.85×evals/s。

同一次中局（`protocol/tests/a_response.py`，512 visits）的 A 响应：

| A | −6 | −2 | 0 | +1.26 | +4 | +8 |
|---|---|---|---|---|---|---|
| v4 gen_003 | 0.057 | 0.301 | 0.544 | 0.699 | 0.903 | 0.979 |
| v3 `s34760` | 0.294 | 0.294 | 0.294 | 0.294 | 0.294 | 0.294 |

发布模型：`models/b4c32-s42012-d595499.bin`（步数/数据是跨两轮的累计值：
34,760+7,252，470,705+202,770−77,976）。

**租机器踩到的坑**（都已在仓库里修掉）：

- 默认 PyTorch 源在这个机房只有 ~260 KB/s，`torch+cu124` 要 40 分钟；换清华 PyPI 镜像
  后 797 MB 用了 2 分钟（镜像里是 cu121 轮子，3090 上照样跑）。不要用 `-q` 装 torch，
  看不到进度会误判为卡死。
- `git archive` 在 Windows 上按 `core.autocrlf` 写 CRLF，Linux 上 `set -euo pipefail
`
  直接报 "invalid option name"，Python 脚本也带 `
`。已加 `.gitattributes`。
- `cmake -S . -B build` 在 Linux 上会因为默认构建 Win32 界面而失败；现在 UI 的默认值
  跟随平台。
- 15 GB 内存的机器上，`--max-samples 600000` 的 37 平面输入要 ~7 GB；只放一代分片时
  峰值约 1 GB，够用。

### 2026-11-23 · A 值成为网络输入（模型格式 v4）

**问题**：`set_a 10` 与 `set_a 0` 在同一个中局给出的胜率一模一样（实测 0.5371 / 0.5371），
残局才有一点差别。原因不是缓存，是设计：A 只出现在终局判定
`sign(rawScore + A)` 里，非终局叶子的值是网络输出的，而 22 维全局特征里**没有 A**。
网络看不见的问题它答不了。

**改法**（三处必须同时改，否则等于没改）：

1. **模型格式 v4**（`SHUYINN4`）：全局特征 22 → 24，末尾两维 `A/10`、`|A|/10`
   （都截断到 ±2）。C++ 的 `buildGlobals(..., aValue, ...)` 与训练器
   `make_inputs(..., a_values)` 用同一个公式；v3 文件仍然能加载，但 `aInput()` 为 false，
   `capabilities` 和每条 `info` JSON 都会写 `a_input=0`，界面不会假装 A 在起作用。
2. **价值目标重标**：不再用数据里的 `value`（那是单一 A 下的一次搜索估计），
   改用**终局原始分**重算 `sign(score + side_sign·A)`。`SYSELF4` 每局都存了原始分，
   所以**不需要重新自对弈**就能得到任意 A 下的标签。
3. **训练时随机化 A**：每个 epoch 按样本重新抽 A，`--a-range 4`（均匀 ±4），
   其中 `--a-anchor-fraction 0.35` 的样本用引擎真正在用的 A（阻挡规则 +1.26）。
   验证集**固定**在锚点 A，这样 `val_loss` 跨 epoch 可比，且等于实际运行点。
   一个局面在不同 epoch 会以不同 A 出现，网络因此学到的是 `P(赢 | A)` 而不是单一 A 的记忆。

**为什么这样训练**：目标 `sign(S+A)` 里出现了 A，网络要拟合它，就必须先估计 S 的分布，
再把它平移 A。这正是我们想要的语义，而且不需要分数头。

**验证**（本机，b4c32 从 v3 权重热启动，只用 gen_005 的 6 万局面微调 3 个 epoch）：

| A | −6 | −2 | 0 | +1.26 | +4 | +8 |
|---|---|---|---|---|---|---|
| v4 胜率 | 0.253 | 0.495 | 0.620 | 0.655 | 0.789 | 0.910 |
| v3 胜率（对照） | 0.294 | 0.294 | 0.294 | 0.294 | 0.294 | 0.294 |

**用法**：`python training/train_resnet.py --a-cond ...`，流水线里是
`--a-cond [--a-anchor 1.26] [--a-range 4] [--a-anchor-fraction 0.35]`。
`--a-cond` 只能用于 `SYSELF4` 数据（需要每局的原始分）；旧分片会直接报错，而不是退回旧标签。
热启动 v3 → v4 是允许的：新增的两列权重补零，所以起点与旧模型完全等价，再靠微调把 A 学出来。

**顺带修掉的两个死路径**：`pipeline.py` / `arena.py` / `cross_eval.py` 里还有
`experimental/training/...` 的旧路径，`check_layout.py` 仍然指向 `experimental/`；
这些在重组后都是死引用，已全部改到 `training/`。

### 2026-11-23 · 链条可以从已有模型接着跑（`--init-model` / `--init-data`）

流水线原来只会从随机网络开链：gen0 随机走子、gen1/gen2 一层扫描，等于把上一轮
学到的东西全部丢掉。现在：

```bash
# 从现有模型继续：gen0 用已有数据微调，并与起点模型打一场 arena
INIT_MODEL=models/b4c32-s34760-d470705.bin INIT_DATA="runs/training/b4c32-blocking-20261122/gen_00*/selfplay" VISITS=3000 GAMES=1200 GENERATIONS=6 bash training/run_generation.sh
```

- gen0 的 `mode=warm`：不产新数据，`train_resnet --init-model <起点> --data <init-data>`，
  并且**必须**和起点模型打一场（arena 里叫 `warm_start`）——微调后比起点差，就是这次
  微调把旧模型扔了，要当场看见，而不是等三代之后从 ELO 曲线里猜。
- `--resume` 会校验 `init_model` 与 `a_cond`：换起点或换格式的续跑会被拒绝，
  否则 manifest 描述的就是一次从未发生过的运行。
- v3 → v4 的热启动是精确的（新增两列权重补零），所以 A 的权重从 0 开始学，
  起点强度不变。

### 2026-11-23 · `run_generation.sh` 的 komi 默认值会覆盖实测值

脚本原来写死 `KOMI="${KOMI:--0.5}"` 并把 `--komi` 无条件传给流水线，于是
"阻挡规则用 +1.26" 这个实测结论被脚本静默改回 −0.5。现在 `KOMI` 默认空，
空值就不传，由流水线按规则集取实测值；要覆盖再显式给。

### 2026-11-23 · 产品化拆分：ui / engine / core / protocol

1. **仓库按产品拆分**：`core/`（规则，两侧唯一共享）、`engine/`（无头引擎）、`ui/`（界面，
   不含搜索）、`protocol/`（唯一契约）、`tools/`（开发工具，不发布）、`training/`（本流水线）。
   引擎和界面各自有 CMake 工程，可以单独构建、单独打包、单独下载；界面链接 `core/`，
   不再链接 `shuyi_search`。
2. **协议 v2**：持久化进程 + 命令驱动 + 按 interval 流式推送 `info` JSON。命令可带 id，
   应答原样带回，乱序不会认错；`analyze` 自己结束时回 `= done`，被 `stop` 时回 `= stopped`。
3. **静默降级全部删除**：`cuda` 不可用直接报错；模型与规则集不符拒绝加载；`mcts_nn` 缺模型
   拒绝搜索；界面在没有引擎时不再自己走棋（内置 AI 与 `SHUYI_UI_BUILTIN_AI` 已删除）。
4. **树大小上限彻底移除**（CPU/CUDA 两侧）：它会在 3.5 万 visits 时把未终局叶子当终局，
   产生 100% 的假胜率。停止条件只剩预算、墙钟、`stop`。
5. **alpha 包**：`release/make_alpha.py` 生成 `dist/shuyi-alpha.zip`（界面 + CUDA 引擎 +
   最新阻挡模型 + 协议 + 说明，0.6 MB），包里没有训练脚本、工具或源码。

### 2026-11-22 · srv1 实跑后的修正

1. **顺次训练**：链条无条件前进（第 N+1 代打第 N 代的数据、从第 N 代权重续训），
   arena 只记录不设闸门。原因：srv1 的 gen3..gen6 全部对着 gen2 的权重打，白跑 50 分钟；
   而"某个模型 match 里强"不等于"它的数据适合训练"。
2. **ELO 标定取代 val_loss**：srv1 的 gen5/gen6 把 `val_policy` 从 4.40 降到 3.34、
   目标 `cell_ratio` 从 0.77 降到 0.60，但对 gen2 的 200 局复核是 0.495 / 0.5125 ——
   **指标降了、棋力没涨**。之后只认 ELO。
3. **komi 重新标定**：实测 gen2 在 komi 0 时红胜 52.4%、komi −5 时 10.9% ⇒ 公平 komi ≈ −0.3，
   默认从 −5 改为 −0.5；小数 komi 让和棋结构性消失（0.00%）。
4. **OpenMP 并行** descend/expand/back up（三棵树立，天然可并行）：本机 descend 0.29→0.14 ms、
   expand 1.65→0.55 ms；服务器上原来 GPU 只有 27% 利用率，CPU 那半 ~4.7 ms。
5. **每 tick 复用缓冲**，去掉每 tick 1.5 MB 的分配；`--stats` 打三段耗时。

### 2026-11-23 · 界面、横评、格式 v4、分数头

1. **棋子改方形**（圆角矩形，半径 = 格宽 40%），最后一手的金框也改成同形状；分析标记复用同一几何。
2. **MCTS 标记**：与棋子同尺寸的细边框 + 半透明染色（按棋盘色混合，避免用 alpha 通道）；
   **最优浅蓝**，其余按"与最优的胜率差"红→绿插值；
   差值 >20 个百分点或访问数 < 最优 5% 的格子**不画**。
   三栏依次是胜率 / 访问数（`820`、`1.2k`、`12k`、`120k`、`1.2M`，≤4 字符，有单测）/ 预期平均分。
   没有分数头时第三栏显示 `--`。
3. **横评工具**：`training/arena.py` 让 GPU 模型（`--mode serve`）与 CPU 引擎同池对战，
   任意两方配对、换色、记录着法、用 `rule_probe` 的计分函数算终局；
   `training/elo.py --anchor random` 把随机钉在 0。（预算规则已于 2026-11-24 改为统一 visits，见 4.1；此处旧的按时间预算说法作废。）
4. **数据 v4**：样本尾部加 `int8 score`（终局 raw 分，走子方视角），作为分数头目标；
   v1/v2/v3 仍可读。
5. **分数头框架**：`--score-weight`，模型 flags bit0 + 尾部权重（与价值头同形状），
   CPU 推理填 `NnOutput.expectedScore`（tanh × 30）。CUDA 路径暂不实现分数头（只用于产数据）。

### 2026-11-23 · 规则隔离打通

1. **一条命令跑通阻挡规则**：`python training/pipeline.py --ruleset blocking ...`
   → 数据（`SYSELF4`）、模型（`SHUYINN3`，ruleset 字节）、训练器（`--ruleset`）、
   arena/ELO（`--ruleset`）全部带标签，run 目录在
   `runs/training/b4c32-blocking-<date>/`，模型名 `b4c32-s<steps>-d<data>.bin`。
2. **UI 规则菜单**：`规则(&R)` → 现行规则 / 阻挡规则；开赛后不允许切换（会提示先开新局），
   切换时重启引擎进程并把 `ruleset` 下发给引擎。
3. `target_stats.py` 的"合法着法数"按规则算（阻挡下随机对局均值 308，经典是 390）。
4. 顺带修：训练器热启动时对缺失的分数头用 `strict=False`（其它键仍严格校验）。

### 2026-11-23 · 阻挡规则

1. `Ruleset{Classic,Blocking}` 作为运行时参数贯穿引擎/协议/数据/模型/UI；
   规则特有代码隔离在 `src/ai_blocking.cpp`。
2. 新剪枝：评估加"挤压"（双方可落子格数之差），排序加"封锁数"。
   同规则对打 **36 : 3** 胜过经典策略。
3. 强制过手 + 连续两次过手终局；过手位置不写训练样本（policy 没有"过手"这一维）。
4. 格式 v3（模型/数据带 ruleset）；`arena.py` + `elo.py --anchor random` 上线。

### 2026-11-23 · 阻挡规则首跑（服务器 3 小时）、A值标定、分数头真正接通

1. **blk1 运行**：RTX 3080，`--ruleset blocking --komi 0 --games-per-gen 1500 --visits 1024
   --root-k 16 --top-k 16 --batch-games 96 --arena-visits 256 --arena-games 120`，
   容量 b4c32（与 srv1 同容量，这样两套规则的差别不是容量差别），模型名
   `b4c32-s*-d*-blk.bin`。gen0 随机（3.5 s）→ gen1/gen2 单层扫描 → gen3 起 MCTS 自对弈。
   链无条件前进，arena 只记录（§1.2 的规则）。
2. **A值标定工具** `training/avalue.py`：同一个模型自己打自己（强弱对称，剩下的只有颜色偏差），
   逐 komi 跑 `--mode match`，直接读每局的 `a_is_red` + `result` 算红方得分，
   再用两参数 logistic 拟合交点 A\*（不局限于扫过的点，还有 delta-method 的 SE）。
3. **新规则默认 A=0**：先按 0 训练，标定后再决定是否改默认值（结果见 §8）。
4. **分数头从来没生效过（真 bug）**：`PolicyValueNet::load()` 里 `hasScoreHead_ = flags & 1`
   写在 `clear()` **之前**，而 `clear()` 会把它清零 —— 于是所有带 flags bit0 的模型读进来
   `hasScoreHead_` 都是 false，CPU 推理永远不填 `expectedScore`，`--score-weight` 训出来的头
   等于白训。修法：赋值挪到 `clear()/allocate()` 之后。单测覆盖（save→load 后 flags 仍在）。
5. **CUDA 前向补上分数头**：额外一次 1×1 空间归约（复用 `valueSpatialKernel`）+ 同形状 MLP，
   输出 `tanh(x) × kScoreScale`；`--check` 现在也对比 `score |max|`（本机 1e-6）。
6. **MCTS 真正用上 `expectedScore`**：叶子回传的分数改成"有分数头就用网络预测，
   没有才退回 value→分数 的线性映射"。单测：`enableScoreHead(1.0)` 的模型（输出恒为
   tanh(1)×30）在 visits=1 时，根节点 `averageScore` 正好是 −tanh(1)×30（根边一次符号翻转）。
   UI 第三行因此显示网络的判断，而不是从胜率猜的分数。
7. **UI 二级菜单**：`引擎(&E)` → `MCTS + 网络` 子菜单，列出 `models/*.bin` 和
   `runs/*/models/*.bin` 里的**全部**模型（名字后标 `[现行]`/`[阻挡]`）＋
   `重新扫描模型目录`；选中即换模型并重启引擎进程（`set_model` 只在启动时读）。
   `shuyi.exe --dump-models out.txt` 无界面导出菜单条目，回归检查用。
8. **横评/标定脚本**：`training/cross_eval.py` 把"线路横评（`arena.py`，异构对手）"与
   "代数爬梯（`elo.py --models`，批量 GPU）"合成**一张**以 random=0 为锚的评分表；
   `elo.py --report` 现在可以重复传多份矩阵合并（同一对按无序对相加）。
   `training/pull.py` 断点续拉服务器产物（模型/数据/日志/JSON），训练还在跑时就能同步。
9. **操作教训（server 横评）**：`arena.py` 并发拉起多个 `--mode serve` 的 GPU 进程时，
   每个进程默认用满 OpenMP 线程 —— 16 核上 7 个进程 × 8 线程，单步 `genmove` 退化成
   "看起来卡死"（实际是在抢核）。给 serve 进程加 `OMP_NUM_THREADS=1` 后，同样一局从
   十几分钟掉到 <11 s。以后跑 GPU 横评一律先设这个环境变量。

### 2026-11-23 · 界面：空格开关分析、超限继续、设置面板、口径统一、去掉硬上限

这一轮全是"用起来"的问题，逐条对应：

1. **空格开始/停止，停止不清屏。** 以前 `stopEngineAnalysis()` 会 `clearEngineAnalysis()`，
   一停就白屏。现在停止只发 `stop`，数字留在棋盘上（那是当前局面最好的信息），
   面板写"已暂停 · 空格继续（树保留）"。F3 保留为别名。
2. **到上限后按空格"这一步忽略上限"。** 引擎在分析自己结束时回
   `= done visits=N proven=0|1`（区别于被 `stop` 的 `= stopped`）；界面据此标
   "已达访问上限"，下次空格发 `analyze red 0` —— 不限量、继续**同一棵树**。
   为此给 `analyze`/`genmove` 加了可选的每命令预算参数。
3. **`set_visits` 不再重建引擎。** 它只改默认值。以前它 `makeCurrentEngine()`，
   前端一设预算就把树扔了——"超限继续"根本无从谈起。
4. **设置面板**（`引擎 → 设置…`）：分析访问上限（0=不限）、落子访问上限、A值。
   存进 `shuyi-ui.ini`（`analysis_visits=` / `play_visits=` / `a_value=`），启动时装载。
   改 A值仍会重建引擎（终局胜负随 A 变，旧树不能信）；只改访问上限不丢树。
5. **口径统一。** 右侧面板原来印 `analysis.winrate` 的原始平均值，左边标记印 Wilson
   下界，同一个局面两处数字不一样。现在两处都印：已解算 → 精确值，否则 Wilson 下界。
6. **不要硬性上限。** `mctsMaxTreeEdges` 整个删掉。它正是"35000 visits 后 100%"
   的根源（撞限时把未终局局面按静态分当硬胜负）。现在树一直长到预算用完，
   停下来的原因只有：预算、墙钟、`stop`。回归测试 `testNoHardValueWithoutProof`
   断言未证明局面永远不会出现 0.999+ 的值。

### 2026-11-23 · 对称性：等价点只搜一次、显示同一份结果

棋盘在正方形的 8 个等距变换（4 旋转 + 4 镜像）下不变，两套规则都只看"本身 + 上下左右
四邻"，所以变换前后是**同一个游戏**。但搜索一直把 a1 和 i1 当成两个不同的着法各自搜，
分析结果也各报各的，于是同一类点的胜率/访问数不一样，界面看起来像"引擎只认一个点"。

改动：

1. `Position::transformCell / symmetryMask / stabilizer / orbitSize`（`position.cpp`）：
   8 个变换的坐标映射、当前局面的对称群（8 位掩码）、固定某格的子群、某格轨道大小。
2. MCTS 每个节点带一个对称群掩码（`symmetry_`）：根本局面算一次，子节点是
   `父群 ∩ 固定着法子群`，不重复扫描棋盘。展开时把着法按轨道分组，**一个轨道只留一个
   可搜索的代表**（选策略先验最高的那个，因为网络的先验只是近似对称的），其余
   `searchable=0`、只记录 `representative` 指向代表。
3. 报告：`cells=` 里每个格子都按代表边的数字填，所以同一轨道的点显示**完全相同**的
   胜率/访问数/目差；`rootPolicy()` 也让等价着法共享访问数，否则自对弈的策略目标会
   教网络"两个等价着法里有一个更好"。
4. 证明（MCTS-Solver）只在代表边上归纳：等价着法带到的局面和代表一样，既不会多证也
   不会少证。
5. 顺手修掉一个**真 bug**：`rootKey_ = 0` 同时是**空盘的 Zobrist 键**，而它又被当作
   "还没有树"的哨兵。结果空盘的根节点一直用恒等掩码（没走对称合并）。现在用显式的
   `rootValid_` 标志，单测 `testSymmetryReport` 盯着这一点。

实测（空盘、mcts_nn、3000 visits，协议层 `cells=`）：

```
c1:3 0.607/135  g1:3 0.607/135  a3:3 0.607/135  i9:3 0.607/135   ← 8 个镜像点同一组数
b1:3 0.607/84   a1:3 0.603/13   （四个角 a1/i1/a9/i9 都是 0.603/13）
```

78 个空格里 52 个有数据（以前 12 个），因为一个代表的结果会覆盖它整个轨道。
`shuyi_tests` 新增 `testBoardSymmetry`（变换是双射、保持邻接、轨道大小、稳定子）和
`testSymmetryReport`（空盘四角数字完全一致；a1 有子时对角线转置对 b1/a2 一致）。

**同时修掉"只剩一个点"的两个来源：**

1. `EngineOptions::visits` 默认从 0 改成 **512**。以前 0 表示"按墙钟"，任何没显式设预算
   的前端（包括旧界面）在 analyze 模式下都会无限搜，访问数能涨到几千万，界面上就只剩
   一个被反复访问的点。现在默认就是固定预算，`set_visits 0` 才回到墙钟。
2. 界面标记：以前"访问数低于最优 5%"的点直接不画，搜索一旦收敛，棋盘上就剩一个标记。
   现在**排名前 8 的点无论如何都画**（`kMarkAlwaysShow = 8`），其余仍按窗口+门槛过滤。
   一个点上有 99% 访问是搜索收敛，不是引擎证明了唯一解；界面不该把这件事表现成"只剩一点"。

### 2026-11-23 · 那个"35000 visits 后左上角 100%"到底是什么

用户给的复现：开界面、双人对弈、阻挡、MCTS-NN + s34760、F3 分析，等到 **35000 visits**，
左上角立刻出现一个 100% 的点。

复现出来了，而且是**两个叠加的 bug**：

1. **树节点上限复用了"终局"标志。** `mctsMaxTreeEdges` 是 1.5M 条边，空盘每个节点展开
   要 48~729 条边，所以 **35000 visits 刚好把这个上限撞满**（`nodes=1500009` 那一刻）。
   旧代码在撞限时写的是 `terminal_[node] = 1`，于是 `simulate()` 走
   `outcomeFor(position)` —— 把一个**没结束的局面**按 `rawScore + A` 判成硬胜负。
   阻挡规则 A=+1.3、空盘 raw=0，所以每个被拒的叶子都算"红方硬胜"：值恒为 1.0，
   不是平均值，而是"确切 1.0"。这个子树因此被 PUCT 疯狂加访，界面上就是一个点 +
   100% + 访问暴涨。**这正是"35000 visits"这个数字的来源。**
   修法：`deadEnd_` 与 `terminal_` 彻底分开（撞限只标记"不再展开"）；撞限后**当次搜索
   直接结束**（`treeCapped_`），不再把剩下的预算悄悄换成 rollout；上限提到 4M 条边
   （约 200 MB），覆盖 10 万 visits 级的分析。回归测试 `testTreeCapIsNotTerminal`。
2. **用户看到的其实是旧的 CUDA 引擎。** `build-cuda` 是 `C:\shuyi_cuda_build` 的
   junction，编译自 `C:\shuyi_src`（→ 本仓库）的源码，但那个二进制停在 16:35，
   还没有这轮的修复。界面按 `cuda=1` 优先选它，所以测试跑的是旧代码。已重新编译；
   现在 `build-cuda/shuyi_engine.exe` 与 CPU 版同源、行为一致（实测 35000 visits、
   `nodes=1680729`、`proven=0`、没有任何点 ≥99.9%）。
   **以后改引擎必须重编 CUDA 树**，否则界面会挑到旧 GPU 引擎。

顺带修掉的两个"看起来像 bug"的地方：

3. **界面显示的是原始平均值。** 一次访问赢了就写 100%，这和"已解算"在视觉上无法区分。
   现在非 `proven` 的点显示 **Wilson 95% 下界**（和排名用的是同一个数），1 次访问的
   100% 显示成 21%；只有 `proven=1` 才显示精确值（那时 100% 是真的）。悬停面板同样处理，
   并标出"已解算"。
4. **引擎默认后端是 `cpu`。** 前端不发 `set_backend` 时，CUDA 构建也跑 CPU（9.6 ms/visit
   而不是 0.5 ms）。默认改成 `auto`：有 CUDA 就用 GPU。
5. **根宽度按预算定。** 以前 MCTS-NN 根宽度固定 16（`mctsNnRootChildren`），空盘只有
   12 个格子有数字。现在根宽度 = `max(32, visits/16)`：512 visits 开 32 个对称轨道，
   35000 visits 开满所有轨道。实测空盘阻挡规则：512 和 35000 visits 都是 **81/81 格有数据**，
   四角显示同一个值（`a1=i1=a9=i9=0.142/1`，Wilson 下界，不是 100%）。

### 2026-11-23 · 终局"硬结果"必须真的硬：MCTS 求解器 + 界面固定 visits

用户的原话是"如果搜索到终局，用硬性胜负、目差代替模型"。上一版只做了**叶子替换**：
终局叶子的值确实换成了规则算出的硬结果，但有两个问题。

1. **叶子值被平均掉了，看起来像"没强制胜利"。** 一个叶子是硬胜（1.0）时，它和一堆
   网络估值一起被平均；同时对手的反驳根本没被搜到，于是引擎在 6 步之内会说
   "99%，赢定了"，而精确求解那一步是 0.00。实测：ply=48 阻挡规则下 16 个候选点里有
   10 个报 >0.88 而精确值是 0.00。**这不是符号问题**——我用精确极小极大逐点验证过
   1 层和 2 层的传播符号（`shuyi_endgame_probe`，源码 `tools/research/endgame_probe.cpp`），
   终局胜负和目差的方向都是对的。
2. **界面里根本没有 visits 预算。** `engine_main.cpp` 在 analyze 模式下把
   `request.visits` 强制成 0，界面也从未发 `set_visits`，所以"分析"是按墙钟无限搜，
   同一个局面每次刷新数字都不一样、越看越"确定"。这与第 4.1 节的评估规则直接冲突。

改动：

1. `engine/src/mcts.cpp` 加 **MCTS-Solver**：节点新增 `provenValue_/provenScore_/scoreExact_`。
   终局叶子按规则立即 proven；一个着法让对手 proven 输 ⇒ 本节点 proven 胜（不必等
   全部着法）；全部着法都 proven 才能判负/和。已解算的节点不再被访问，
   **根一解算就提前结束搜索**（预算故意不花完）。
2. `complete_`：只有**没有被宽度剪枝**的节点才能做"所有着法都不行"的归纳，避免在
   被剪过的子树上伪造证明。
3. `deadEnd_` 取代原来的"终局"标志：树节点上限触发时以前会把**未终局的局面**按静态分
   当硬结果用，现在只当普通叶子。
4. `Analysis::proven` / `CellAnalysis::proven`，协议 `info ... proven=0|1`，
   `cells=` 第 6 个字段是每个点的证明标志；界面把已解算的点按**精确胜率**排名
   （Wilson 下界只用于估计值），状态栏显示"已解算"。
5. 目差只在**精确**时公布：根全解 → 全树的极大极小目差；根解但目差未定 → 报"引擎实际
   会走的那步"的精确目差；都没有就报 `--`。之前公布的是"已证明的子集里的最大值"，
   实测出现过 +3 而真实是 +1、+8 而真实是 +9。
6. `rootPolicy()` 在解算局面下只返回证明最优的那一手。否则自对弈的温度采样会从
   每点 1 次访问的均匀分布里挑一个**已证明的败着**。
7. 界面 `kUiVisits = 512`；`capabilities` 增加 `proto=1 visits=1`，界面只在引擎声明
   `visits=1` 时发 `set_visits`（老引擎退回 `set_time 650`）。analyze 模式同样遵守预算。

验证：

1. `engine/tests/test_engine.cpp::testTerminalProof`（进 ctest）：最后 1 手时引擎必须
   `proven`、胜率等于精确值、目差等于终局分、且 visits < 预算。
2. `shuyi_proof_check`（`tools/research/proof_check.cpp`）：随机 300 个阻挡规则
   残局，对**每一个** `proven` 用精确极小极大复核胜率和目差：
   `proven=127 checked=127 skipped=0 mismatches=0`。
3. 实测 ply=53：1 次模拟就解算（原来是 512 次还在报 0.96）；ply=52 阻挡：胜率精确、
   目差取"实际会走的那手"的精确值；ply=48：仍然解不动（16^6 的树 512 visits 不可能解），
   此时**不给**目差，只给估计胜率——这才是诚实的。

代价：解算局面常常提前停（最后 1 手 1~23 次访问就结束），`visits` 会小于预算；
目差未证明时不会为了"凑一个目差"把预算花完，而是照常搜到预算为止。评估脚本按预算跑，
不受影响。

### 2026-11-24 · 评估预算统一成 visits（这是评估的前提，不是可选项）

问题：评估要回答的是模型多强，所以每一步花多少搜索量必须**固定且与主机无关**。
原来的做法是 GPU 模型按 `--visits`、CPU 引擎按 `--time` 毫秒 —— 同一张表里既有 128 visits
又有 220 ms，CPU 行还随机器和负载漂移。3 小时那跑的三份矩阵（128 / 96 / 192）混着合并过，
那张表**作废**。

改动：

1. `training/evalconfig.py`：`EVAL_VISITS = 512` 是唯一允许的预算，所有评估脚本的默认值都来自它。
   命令行的 `--visits` 还在（调试用），但**一张表用多少就必须写进表里**，
   而且不同预算的表禁止合并（见第 4 点）。
   一个 visit 的定义：MCTS/MCTS-NN = 一次模拟（= 一次网络前向），剪枝 = 一个节点
   （= 一次叶子评估），纯 MCTS = 一次 rollout。
2. `SearchOptions::nodeLimit`：剪枝引擎现在有**节点预算**，且**完全不看墙钟**。
   以前给剪枝引擎 `set_visits` 会把它 `timeLimitMs` 设成 0 -> 立刻中止 -> 下随机棋，
   所以旧脚本只能按时间跑。这是这次才真正修掉的。
3. `arena.py`：四种引擎一律 `set_visits V`；`--time` 降级为**挂死保护**
   （`--hang-guard-ms`，默认 1 小时，必须晚于预算触发）。
4. `elo.py --report`：**拒绝合并预算不同的矩阵**（直接报错退出），防止再把两把尺子拼一起。
5. 回归测试：`engine/tests/visit_budget_smoke.py`（进 ctest）驱动协议读回 `visits`/`nodes`：
   MCTS/MCTS-NN 必须**正好**等于预算；剪枝的节点数可在预算之上溢出（它要搜完当前这一层），
   但**同一输入两次必须是同一个节点数和同一个着法**；C++ 侧 `testVisitBudget` 同样断言。
   同时把 MCTS 报的 `visits` 改成实际模拟次数（以前是根边访问数之和，永远少 1）。

实测（本机 MX350，512 visits/步）：GPU ~10 s/局、剪枝 ~0.1 s/局、纯 MCTS ~0.8 s/局、
CPU MCTS-NN ~100 s/局。所以 512 的表能完整跑前三种，CPU MCTS-NN 只能跑少数几对。

---

### 2026-11-23 · 引擎按能力挑选（UI/引擎分离的正确做法）

我之前把"想用 GPU"说成"要用 CUDA 构建的界面跑"，这是错的：界面和引擎是两个进程，
界面只管把**最快的那个引擎**拉起来，自己是什么构建无所谓。

1. 引擎新增协议命令 `capabilities` → `engine=shuyi_engine cuda=1|0 rulesets=...`。
2. 界面启动时按 `SHUYI_ENGINE` → exe 同级 → `../build-cuda` → `../build` → 当前目录 依次探测，
   **优先选回答 `cuda=1` 的**；再发 `set_backend auto`，标题显示 `(GPU)`/`(CPU)`。
3. `shuyi.exe --dump-engine out.txt` 无界面导出"试了哪些路径、各自回答什么、选了谁"，
   回归测试 `ui/tests/engine_choice_smoke.py`（ctest: `shuyi_engine_choice`）。
4. 本机把 `build-cuda` 做成指向 `C:\shuyi_cuda_build` 的 junction（nvcc 不能写中文路径），
   于是 MinGW 编的 CPU 界面会自动选中 MSVC/CUDA 编的引擎。
5. 顺带修：`GetModuleFileNameA`/`GetCurrentDirectoryA`/环境变量给的是 **ANSI**，
   之前用 `CP_UTF8` 去加宽，中文路径直接变乱码、候选全被判"不存在"。

---

### 2026-11-23 · 推理速度：CPU 前向重写 + 引擎可以走 CUDA

问题：界面里 MCTS+NN 几乎不动。测下来是 **CPU 前向 36 ms/次**（b4c32，86k 参数），
512 visits 一步要 18.6 s；而这一步的算术量只有 6.8 M MAC，属于**写得不对**。

1. `conv3x3` 重写：输入做 1 格 padding（去掉内层循环的 4 个边界判断）、
   按"9 个 tap × 81 个连续浮点"的 FMA 链、`restrict` 帮助别名分析、
   复用 scratch（原来每次卷积都分配三份 plane 大小的 vector）。
   **36.4 ms → 3.7~4.4 ms/次（约 10 倍）**，b2c16 从 4.9 ms → 0.2 ms。
2. `-march=native` 实测反而更慢（4.3 vs 3.7 ms，Ice Lake 的 AVX-512 降频），
   所以 `SHUYI_NATIVE_SIMD` 默认 **OFF**，要用自己开。
3. 引擎进程现在能走 CUDA：`-DSHUYI_CUDA=ON` 时 `shuyi_engine` 链接 `shuyi_cuda`，
   协议新增 `set_backend cpu|cuda|auto`，`info` 行回报 `backend=cuda|cpu`。
   界面启动时发 `set_backend auto`，标题显示 `(GPU)`/`(CPU)`。
   实测（b4c32，512 visits）：CPU 5.7~6.1 s → **GPU 0.39 s**，着法与节点数完全一致。
4. 教训（第二次踩）：改完编译选项后 `build/` 里的旧 `.o` 会让"优化没生效"——
   中间一度测出 14.5 ms 其实是陈旧对象。**改 flag 就清空构建目录**。

---

### 2026-11-23 · 终局精确值、删掉假的目差、容量结论

1. **终局用精确值代替模型**：`mctsFullWidthPlies = 6` —— 最后 6 手内不剪宽度，
   树才能真正走到终局；终局节点用 `raw + A` 的硬胜负、硬目差（`SimulationResult::hasScore`）。
   单测 `testEndgameExactScore`：两格空盘时，**没有分数头**的网络也必须报出真实分数。
2. **删掉 `(value*2-1)*73` 的目差映射**：这是错的（空盘会给出 −15.2 分）。现在分数只有两个
   来源——精确终局、或训练过的分数头；都没有时协议发 `hasscore=0`，界面第三行显示 `--`。
   `Edge` 增加 `scoreVisits`，平均分只在"真正带回分数"的访问上取平均。
3. **容量是主因**：b4c32 只有 **8.6 万参数**（86,010，实测文件 344,076 字节；v4 多 82 个 A 权重 = 86,092），
   b6c96 是 **103 万**（12.0×参数、12.2×MACs），b8c128 241 万（28×），b10c128 300 万（35×）。
   我们比 KataGo 的**起步档**还小 12 倍，且只训了 6 代 / 47 万局面 ⇒ 下一步直接换 b6c96，
   并把每代数据量提到 5 倍以上；详见 `docs/plans/NEXT_STEPS.zh-CN.md`。

---

### 2026-11-23 · UI 的最优标记、模型命名

1. **最优标记改成"置信下界 + 访问数"**：以前 `best` 取 `winrate` 最大的格子，于是一个
   只搜了 1 次的 100% 格子会被标成最优，每次刷新在两个格子之间来回跳（闪烁）。
   现在用 Wilson 95% 置信下界 `lb(p, n)`（n=1,p=1 → 0.21；n=200,p=0.55 → 0.49）作为
   "综合搜索量和胜率"的打分：`best = argmax lb`，颜色也按 `lb` 的差着色，
   平局时保留上一帧的最优格（`kMarkTieMargin`），所以标记稳定。
2. **第三行不再显示"猜出来的分数"**：协议新增 `info scorehead=0|1`，没有分数头时
   UI 显示 `--`。实测 blk1 的模型在开局位置会给出 `score=-15.2`（`(value*2-1)*73` 的
   线性映射），这种数字比空着更糟。
3. **模型命名统一为 KataGo 形式** `b{blocks}c{channels}-s{steps}-d{data}`：
   - 去掉 `-blk` / `-cls` 后缀 —— 规则已经在模型头的 ruleset 字节里，run 目录也分开；
   - 去掉 `gen_XXX.bin` 副本 —— 它不符合命名标准，还让 UI 菜单出现重复项；
     续跑与池内 Elo 现在都从 manifest 的 `named_model` 取路径；
   - 已有的 blk1 也按此整理（`b4c32-s34760-d470705-blk.bin` → `b4c32-s34760-d470705.bin`，
     `gen_00N.bin` 删除前逐个 SHA-256 校验过是同一份权重）。

---

### 2026-11-24 · 仓库按 ui / engine / experimental 重组

问题：根目录平铺着 `include/ src/ tools/ training/ tests/` 五套代码目录，加 16 篇文档，
新人和自己都找不到东西；而且 **UI 的模型菜单在一次从 `build/` 启动时只显示 2 个模型**。

改动：

1. 目录：`engine/`（include+src+tests）、`ui/`（src+include+tests）、
   `experimental/`（selfplay / research / training / tests）。`data/ models/ runs/` 是共享数据，
   留在根目录并写进文件地图；`docs/` 拆出 `plans/` 与 `reports/`。
2. 文档：新增 [../../docs/MAINTAINERS.zh-CN.md](../../docs/MAINTAINERS.zh-CN.md)（每个文件在哪、干什么、
   改它注意什么）、[../../docs/README.md](../../docs/README.md)（文档索引），
   以及 `engine/README.md`、`ui/README.md`、`tools/README.md`。
3. **模型菜单修好了**：以前按相对路径找 `models/` 和 `runs/*/models`，从 `build/shuyi.exe` 启动时
   工作目录是 `build/`，于是只剩两个夹具模型。现在扫描 exe 目录、其父目录（仓库根）、当前目录
   以及 `SHUYI_MODEL_DIRS`，每个根下递归找 `models/`；标签带运行名（`blk1/gen_005.bin`），
   规则 tag **从模型头的 ruleset 字节读**（`gen_000.bin` 这种名字猜不出来）。
   回归测试 `ui/tests/ui_menu_smoke.py` 会断言两种规则都在菜单里。
4. 顺带修了两个只在中文路径下暴露的 bug：`std::filesystem`/`fopen` 用 ANSI 编码打开
   UTF-8 路径会抛 "Illegal byte sequence"（现在有 `engine/include/shuyi/io.h`：`fopenUtf8`、
   `openOutUtf8`、`Utf8Args`），以及 `windows.h` 的 min/max 宏破坏 `std::max`。
   UI 侧因此把 `ModelEntry` 分成 ANSI `path` 和 UTF-8 `enginePath` 两份。

---

## 7. 横评结果（2026-11-23，经典规则，本机）

`training/arena.py --games 6 --time 80 --visits 96`，锚点 random = 0，168 局（**预算不统一，已被 9.2 的 512 visits 表取代，仅作历史记录**）：

| 模型 | Elo（相对随机） | ±90% | 战绩 |
|------|----------------:|------|------|
| `gen_006`（srv1 命名，旧格式） | **+1376** | [1294, 1576] | 40/0/2 |
| gpu:gen_004 | +1136 | [1098, 1229] | 33/0/9 |
| pruning（Alpha-Beta） | +1026 | [946, 1168] | 29/0/13 |
| gpu:gen_002 | +947 | [852, 1040] | 26/0/16 |
| gpu:gen_001 | +787 | [719, 866] | 20/0/22 |
| mcts（CPU） | +514 | [414, 650] | 11/0/31 |
| mcts_nn（旧部署 b2c16） | +444 | [347, 525] | 9/0/33 |
| random | 0 | [−40, 40] | 0/0/42 |

读法：

1. 新流程训出来的模型全部远高于旧部署模型（+787…+1376 对 +444）；
2. **96 visits 太少**：剪枝引擎（+1026）排在 gen_004 与 gen_002 之间。
   下一次横评必须把 visits 提到 512–1024，否则比的是"低预算下的鲁棒性"，不是棋力；
3. gen_001 < gen_002 < gen_004 < gen_006 的顺序与 srv1 池内 Elo 一致，说明两套测法互相印证；
4. random 的 CI 正好跨在 0 上（[−40, +40]），是锚点正确的自检。

复现：

```bash
python training/arena.py --games 6 --time 80 --visits 96 --workers 2 \
  --players random pruning mcts \
            gpu:runs/archive/b4c32-classic-20261122/models/b4c32-s7960-d108000.bin \
            gpu:runs/archive/b4c32-classic-20261122/models/b4c32-s15600-d212000.bin \
            gpu:runs/archive/b4c32-classic-20261122/models/b4c32-s23240-d316000.bin \
  --gpu-binary <build>/shuyi_selfplay_gpu
python training/elo.py --report runs/probes/arena/classic.json --anchor random
```

---

## 8. A值标定（第二套规则第一次定标）

规则里那个"平衡因子"就是本文档里的 A值：`T = P + A`，A 加在红方局面分上。A值公平不公平
不能靠猜，也不能用旧规则的数（两套规则的手数结构完全不同），必须用**该规则的模型自己打自己**
来量：同一权重两边对局，唯一的系统偏差就是颜色。

工具 `training/avalue.py`：

```bash
python training/avalue.py --model runs/training/b4c32-blocking-20261122/models/b4c32-s34760-d470705.bin \
    --ruleset blocking --komis "-2,-1,0,1" --games 240 --visits 192
```

每个 komi 跑 `--mode match`（换色、同模型两边），直接读每局 CSV 的 `a_is_red` + `result`
统计红方得分，再用两参数 logistic 拟合交点 A\*（带 delta-method 的 SE）。`--self-test`
用已知交点的合成曲线验证符号约定（komi 加给红方，斜率必须为正）。

blk1（阻挡规则，b4c32，gen_005，192 visits，每个 komi 240 局）：

| A | 红方得分 | 95% CI | 红/和/黑 |
|---:|---:|---|---|
| −2 | 0.158 | [0.118, 0.210] | 117/22/101 |
| −1 | 0.246 | [0.196, 0.304] | 119/20/101 |
| 0 | 0.344 | [0.287, 0.406] | 118/27/95 |
| +1 | 0.467 | [0.405, 0.530] | 121/28/91 |

**拟合 A\* = +1.26（SE 0.22，斜率 0.506 log-odds/点）。**

结论：

1. **阻挡规则下先手是劣势**：A=0 时红方只有 34.4%，与经典规则的 52.4% 方向相反。
   直觉上说得通——先落的子会先把邻居"让"给对手当封锁点。这条必须写进规则说明。
2. 所以第二套规则的公平 A≈**+1.3**，`pipeline.py` 的 `--komi` 默认值已按规则分开
   （classic −0.5 / blocking +1.3）。
3. 整数 A 下有 9~11% 的和棋；A 取小数（+1.26）结构上禁止平局 —— 想保留和棋就把 A 定成整数 1 或 2。
4. 这个 A 是用**192 visits 的 gen_005** 量的；模型更强或搜索更深时数字会移动，下一轮重测。

## 9. 阻挡规则横评

### 9.1 旧表作废（预算不统一）

2026-11-23 服务器那一跑里我先给过一张表：线路矩阵 128 visits、桥接 96 visits、爬梯 192 visits，
CPU 线路还是按 **220 ms** 跑的。**这张表不再有效** —— 同一张表里混了三种预算，
GPU 行和其它行根本不是一个单位，CPU 行还随主机快慢变。`elo.py` 现在会直接拒绝合并这种矩阵：

```
refusing to merge reports with different visit budgets
(lines_cpu.json: 128, gens.json: 192); every player in one rating must use the same fixed budget
```

旧数字只保留一个用途：说明"96~192 visits 的表会把剪枝插到两代网络之间"。要看棋力，看下面这张。

### 9.2 统一 512 visits 的表（本机，阻挡规则，komi +1.26）

`training/arena.py --games 6 --visits 512 --workers 2`，random 钉 0，共 115 局。
预算全部由 `training/evalconfig.py` 的 `EVAL_VISITS = 512` 决定，四种引擎都是 512。

| 参赛者 | 每步预算 | ELO（相对随机） | ±90% | 战绩 |
|--------|---------|----------------:|------|------|
| `b4c32-s34760-d470705`（第 6 代） | 512 visits | **+1520** | [1529, 1551] | 30/0/0 |
| `mcts_nn`（CPU 前向，同模型） | 512 visits | +1158 | [1137, 1213] | 9/0/0 |
| `b4c32-s17480-d236807`（第 3 代） | 512 visits | +1150 | [1158, 1185] | 24/0/6 |
| pruning（Alpha-Beta） | 512 nodes | +758 | [717, 837] | 27/0/19 |
| `b4c32-s5960-d80938`（第 1 代） | 512 visits | +630 | [545, 729] | 12/0/18 |
| mcts（CPU 无网络） | 512 visits | +498 | [423, 574] | 13/0/25 |
| random | — | 0 | [−27, 27] | 0/0/47 |

读法：

1. **顺序在 512 visits 下成立**：gen_005 > gen_002 > 剪枝 > gen_000 > 纯 MCTS > random。
   剪枝（每步 512 个叶子）仍然**夹在两代网络之间**——这是真的，不是预算造成的假象，
   因为现在所有参赛者都是同一个 512。
2. gen_005 一轮 30:0（对 gen_000/gen_002/pruning/mcts 全胜），说明 3 小时 6 代在阻挡规则下
   还远没到平台期。
3. **CPU 版 MCTS-NN 与 gen_002 打平（+1158 vs +1150）**，但只有 9 局且全胜 ——
   9:0 的成绩给出的 Elo 实际上只有下界意义，别把这两行当成"CPU 版和网络一样强"的证据。
   它每局 ~107 s，这就是为什么它只有 9 局。
4. 每对 4~6 局，区间只够排序。要达到 ±5pp 需要每对 400 局：GPU 一局 10 s，
   400 局 ≈ 1.1 小时/对 ⇒ 完整表必须放服务器（或只比"相邻两代"）。
5. 未败选手（30:0、9:0）的 ±90% 是自举分布被截断的一侧，区间会整体落在点估计**上方**，
   只能当成下界读 —— 这也是为什么 gen_005 与 mcts_nn 的区间看起来"偏"了。
6. 训练循环内部那张表（`pipeline.py --arena-visits`，默认等于 `--visits`）也是按 visits 的，
   但它只是"这一代是否变强"的判据，和这里的 512 标准表**不能混用**。

复现：

```bash
export OMP_NUM_THREADS=1
python training/arena.py --players random pruning mcts \
    gpu:runs/training/b4c32-blocking-20261122/models/b4c32-s5960-d80938.bin \
    gpu:runs/training/b4c32-blocking-20261122/models/b4c32-s17480-d236807.bin \
    gpu:runs/training/b4c32-blocking-20261122/models/b4c32-s34760-d470705.bin \
    --games 6 --visits 512 --komi 1.26 --ruleset blocking --workers 2 \
    --out runs/probes/arena/blocking512/table.json
python training/arena.py --players mcts_nn:runs/training/b4c32-blocking-20261122/models/b4c32-s34760-d470705.bin pruning \
    --games 4 --visits 512 --komi 1.26 --ruleset blocking \
    --out runs/probes/arena/blocking512/mctsnn_vs_pruning.json
python training/elo.py --report runs/probes/arena/blocking512/table.json \
    --report runs/probes/arena/blocking512/cpu_lines.json \
    --report runs/probes/arena/blocking512/mctsnn_vs_pruning.json \
    --report runs/probes/arena/blocking512/mctsnn_probe.json \
    --anchor random --bootstrap 200
```

## 10. blk1 训练曲线（阻挡规则，b4c32，1500 局/代，1024 visits）

| 代 | 方式 | 自对弈局面 | 模型 | 对上一代 | 池内 ELO |
|---:|---|---:|---|---|---|
| 0 | random | 80938 | b4c32-s5960-d80938-blk | 冷启动（无对比） | — |
| 1 | oneply | 77950 | b4c32-s11720-d158888-blk | 0.979 (116/3/1) | — |
| 2 | oneply | 77919 | b4c32-s17480-d236807-blk | 0.746 (88/3/29) | gen_002 +255 / gen_001 +121 / gen_000 −375 |
| 3 | selfplay | 77951 | b4c32-s23240-d314758-blk | 0.912 (105/9/6) | gen_003 +457 / gen_002 +127 / gen_001 −53 / gen_000 −531 |
| 4 | selfplay | 77971 | b4c32-s29000-d392729-blk | 0.754 (86/9/25) | gen_004 +483 / gen_003 +311 / gen_002 −30 / gen_001 −132 / gen_000 −632 |
| 5 | selfplay | 77976 | b4c32-s34760-d470705-blk | 0.633 (69/14/37) | 已到 3 小时窗口，池内 ELO 未跑完 |

要点：

1. **第一个 MCTS 代（gen3）跳跃最大**：对 gen2 是 0.912，池内从 +255 直接到 +457。
   一层扫描代（gen1/gen2）确实只值那 4~7 分钟，第 3 代必须切 MCTS —— 与 srv1 的结论一致。
2. **越往后收益越小**：gen4→gen5 的 head-to-head 只有 0.633，但池内 ELO 仍在涨（gen_004 +483）。
   这正是"要按 ELO 而不是按 val loss 部署"的原因：val_policy 每代都降，棋力两代才涨 10 分。
3. **阻挡规则自带和棋**（komi 0 时 9~14/120），因为 A 是整数；这是规则差异，不是 bug。
4. 时间账：自对弈 ~25 min/代（80938 局面、1024 visits、0.95~1.1 games/s），
   arena+池内 ELO ~6 min/代 ⇒ 3 小时只够 6 代。想跑 10 代（触发换大模型），要么 6 小时、
   要么把 `--games-per-gen` 降到 800。
