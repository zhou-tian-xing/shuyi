# 数弈（Shuyi）

9×9 的数字落子棋：**C++17 规则/搜索引擎 + 原生 Win32/GDI 界面**。界面与引擎是两个进程，
走 GTP 风格的行协议；不依赖浏览器或任何 Web 运行时。训练侧是 C++/CUDA 自对弈 + PyTorch 训练器。

![数弈原生界面与 MCTS 分析](docs/screenshot.png)

## 维护者从哪开始

- [docs/MAINTAINERS.zh-CN.md](docs/MAINTAINERS.zh-CN.md)：**文件地图**——每个文件在哪、干什么、改它要注意什么；
- [experimental/training/README.md](experimental/training/README.md)：训练手册（原则、闸门、评测口径、改进记录）；
- [docs/plans/NEXT_STEPS.zh-CN.md](docs/plans/NEXT_STEPS.zh-CN.md)：下一步思路（容量、终局精确值、训练量）。

## 当前状态（2026-11-23）

| 项目 | 状态 |
|---|---|
| 规则 | 两套并行：**现行规则**与**阻挡规则**（异色大子禁止相邻）。规则是运行时参数，数据/模型/界面全部隔离 |
| 引擎 | `random` / `pruning`(α-β) / `mcts` / `mcts_nn` 四种；`shuyi_engine` 为行协议进程 |
| 模型 | 阻挡规则 6 代（b4c32，每代约 7.8 万局面）；现行规则另有历史模型 |
| 评测 | 统一 **512 visits/步**，random 锚定 0；最强模型 `b4c32-s34760-d470705` = **+1520** |
| A值 | 现行规则约 −0.2；阻挡规则 **+1.26**（SE 0.22，模型自对弈标定） |
| 容量 | b4c32 ≈ **8.6 万参数**，比 KataGo 最小的 b6c96（103 万）小 12 倍——**当前第一瓶颈** |

## 快速开始

```bash
# 构建（MinGW / MSVC / Linux 均可）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
cd build && ctest --output-on-failure        # 8 项测试

# 开界面：引擎菜单里选 剪枝 / MCTS / MCTS+网络（子菜单列出扫描到的全部模型）
build/shuyi.exe

# 直接驱动引擎（行协议）
printf 'boardsize 9\nruleset blocking\nset_engine mcts_nn\nset_model runs/blocking/blk1/models/b4c32-s34760-d470705.bin\nset_visits 512\ngenmove red\n' | build/shuyi_engine
```

CUDA（自对弈产数据用）：`-DSHUYI_CUDA=ON -DCUDAToolkit_ROOT=<cuda>`。
Windows 上 nvcc 不能写中文路径，因此用 junction：`cmake -S C:/shuyi_src -B C:/shuyi_cuda_build ...`。

## 规则一句话

红先，双方轮流在空格放下数字 1–9 的棋子，每种数字各三枚；54 枚落完后按"每格附近红黑最大数字
的符号"逐格计分，加上 A值后为正红胜、为负黑胜、为零和棋。**阻挡规则**额外要求：落点四邻若有
异色且数字 ≥ 要下的数字则该着非法；无合法着法强制过手，连续两次过手终局。

完整规则见 [docs/RULES.zh-CN.md](docs/RULES.zh-CN.md)，A值的数学见
[docs/A_VALUE_ANALYSIS.zh-CN.md](docs/A_VALUE_ANALYSIS.zh-CN.md)。

## 当前强度（阻挡规则，komi +1.26，统一 512 visits/步）

`experimental/training/arena.py` + `elo.py`，random 锚定 0，115 局：

| 参赛者 | 每步预算 | ELO | ±90% | 战绩 |
|---|---|---:|---|---|
| `b4c32-s34760-d470705`（第 6 代） | 512 visits | **+1520** | [1529, 1551] | 30/0/0 |
| `mcts_nn`（CPU 前向，同模型） | 512 visits | +1158 | [1137, 1213] | 9/0/0 |
| `b4c32-s17480-d236807`（第 3 代） | 512 visits | +1150 | [1158, 1185] | 24/0/6 |
| `pruning`（α-β） | 512 nodes | +758 | [717, 837] | 27/0/19 |
| `b4c32-s5960-d80938`（第 1 代） | 512 visits | +630 | [545, 729] | 12/0/18 |
| `mcts`（无网络） | 512 visits | +498 | [423, 574] | 13/0/25 |
| `random` | — | 0 | [−27, 27] | 0/0/47 |

读法：搜索确实在涨（第 6 代对第 1、3 代全胜），但**剪枝引擎仍夹在两代之间**；
整条链只训了 6 代、47 万局面，离容量与数据的上限都很远。

## 训练记录（阻挡规则，b4c32，1500 局/代，1024 visits）

| 代 | 方式 | 自对弈局面 | 模型 | 对上一代 | 池内 ELO |
|---:|---|---:|---|---|---|
| 0 | 随机 | 80938 | `b4c32-s5960-d80938` | 冷启动 | — |
| 1 | 单层扫描 | 77950 | `b4c32-s11720-d158888` | 0.979 (116/3/1) | — |
| 2 | 单层扫描 | 77919 | `b4c32-s17480-d236807` | 0.746 (88/3/29) | gen2 +255 / gen1 +121 / gen0 −375 |
| 3 | 自对弈 | 77951 | `b4c32-s23240-d314758` | **0.912** (105/9/6) | gen3 +457 / gen2 +127 |
| 4 | 自对弈 | 77971 | `b4c32-s29000-d392729` | 0.754 (86/9/25) | gen4 +483 / gen3 +311 |
| 5 | 自对弈 | 77976 | `b4c32-s34760-d470705` | 0.633 (69/14/37) | 窗口到期，未跑完 |

A值标定（同一模型自对弈，每个 komi 240 局）：A=0 时红方只赢 **34.4%**，A=−2 时 15.8%，
A=+1 时 46.7% ⇒ **公平 A = +1.26**。也就是说**阻挡规则下先手是劣势**，与现行规则相反。

## 目录结构

```
engine/        规则 + 搜索 + 网络（静态库 + 行协议引擎）；只依赖自己
ui/            Win32 界面 + 引擎进程客户端；只依赖 engine 的头
experimental/  自对弈（C++/CUDA）、训练与评测脚本、一次性研究工具
data/          共享夹具：147 个终局分数见证
models/        固定模型夹具（v1/v2）+ 运行时默认模型
runs/          训练与评测输出：<ruleset>/<run>/gen_XXX/、models/、arena/、avalue/
docs/          规则 / 协议 / 架构 / 评测方法；plans/ 计划；reports/ 历次运行报告
```

每个文件在哪、干什么，看 [docs/MAINTAINERS.zh-CN.md](docs/MAINTAINERS.zh-CN.md)；
文档索引见 [docs/README.md](docs/README.md)。

## 训练与评测的三条硬规则

1. **棋力只认 ELO**，不看 val loss（历史上 val_policy 降了 1 分而棋力没涨）。
2. **ELO 永远以 `random = 0` 为基准**；高胜率用相对 ELO + 置信区间表达。
3. **评测预算固定为 visits**（`experimental/training/evalconfig.py` 的 `EVAL_VISITS = 512`），
   同一张表里所有参赛者必须是同一个数，**不许用时间预算**；`elo.py` 会拒绝合并不同预算的矩阵。

模型命名沿用 KataGo 形式：`b{blocks}c{channels}-s{steps}-d{data}`（步数 = 累计优化步，
数据 = 累计局面数）。规则不写进文件名——它是模型头里的一个字节，运行目录也已经分开。

## 已知限制（按影响排序）

1. **容量太小**：b4c32 只有 8.6 万参数，比 KataGo 最小的 b6c96（103 万）小 12 倍；
   25 分钟一代、6 代就撞到瓶颈。下一步先换 b6c96 并给足训练时间。
2. **训练量不足**：至今累计约 47 万局面，对 1M 参数级模型至少还差一个数量级。
3. **A值不是网络输入**：22 个全局特征里没有 komi，模型只在训练用的 A 下"知道"胜负，
   换 A 后胜率/目差会系统性偏移。
4. **没有分数头**：现行模型 flags bit0 = 0，界面第三行显示 `--`（**刻意**不显示用胜率
   线性映射猜出来的假分数）。
5. **CPU 版 MCTS-NN 慢**：b4c32 的 CPU 前向约 3.6 ms/次，512 visits 一步约 1.9 s（每局约 100 s）。

## 实现要点

- 棋盘坐标内部 `x,y = 0..8`，`a1` 在左下；81 位 bitboard 用两个 64 位字；Zobrist 含行棋方。
- 剪枝：迭代加深 + aspiration window + PVS + 置换表；预算按**节点数**（`nodeLimit`），不看墙钟。
- MCTS：SoA 节点/边、PUCT、强制过手节点、根/内部宽度上限；**最后一手附近不再剪宽度**。
- 终局是求解，不是估值：终局叶子按规则给出硬胜负/目差，一个着法让对手被证明输就是
  本方的**已证明必胜**（`proven=1`），根一解算就停搜。目差只在精确时公布，
  没证明就显示 `--`——不会把"已证明子集里的最大值"当目差（那会算出 +3 而真实是 +1）。
  实测最后 1 手 1~23 次模拟解算完成，ply=52（2 手）胜率精确；再往前 512 visits 解不动，
  引擎就只报估计胜率。
- 网络：ResNet（残差块 + 策略头 + 价值头 + 可选分数头），CPU 与 CUDA（im2col + cuBLAS）双后端，
  `--check` 逐位对拍。
- 界面：方形圆角棋子；分析标记与棋子同形，最优格按**置信下界**（Wilson）排序而不是裸胜率，
  所以不会在低访问的格子上乱跳；已解算的格子例外，按精确胜率（同胜负再比精确目差）；
  没有真实分数时不显示数字。界面只下一个预算：`set_visits 512`，从不按毫秒。
