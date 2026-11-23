# 数弈引擎架构

本文先冻结引擎主线的大结构，再进入实现。后续微优化不应改变这里的边界：
状态表示、置换表、MCTS 树、UI/引擎进程协议和 ELO 评测。

## 1. 进程与目录边界

```
┌─────────────────────────────┐      协议 v2：命令 + JSON 流    ┌──────────────────────────────┐
│ shuyi.exe                   │  stdin/stdout, 一行一条消息     │ shuyi_engine.exe             │
│ 原生 Win32 UI（ui/）        │ <────────────────────────────> │ 无头引擎（engine/）          │
│ - 棋盘/落子/悔棋：状态权威  │   = … / ? … / info {json}       │ - 搜索/网络/分析/走棋/bench  │
│ - 读线程 + 100ms 刷新 Timer │                                 │ - 单搜索线程，可被 stop 中断 │
│ - 只持有“展示用”分析快照    │                                 │ - 弱状态：只知道当前局面     │
└─────────────────────────────┘                                 └──────────────────────────────┘
              │                                                              │
              └────────────── 都只链接 core/（规则）────────────────────────┘
```

- 两个产品**分开编译、互不链接**：UI 链接 `core/`，引擎链接 `core/` + 自己的搜索。
  界面里没有任何搜索代码，引擎里没有任何 Win32。
- 契约只有 [protocol/ENGINE_PROTOCOL.md](../protocol/ENGINE_PROTOCOL.md)：命令表、
  `info` JSON 字段、流式语义、错误与版本规则。引擎可以被换成一个完全不同的实现。
- 引擎持久运行（一个进程服务整盘棋），不是每步新开进程/线程；分析按 interval
  （默认 100 ms）持续推送 JSON，UI 用 100 ms Timer 重绘，约 10 Hz。
- 对局记录归 UI；引擎只知道“当前局面 + 该局面的搜索树”。树的继承因此是同一个
  局面 key 的延续，`play` 把根推进到子节点，`set_visits` 不会丢弃它。
- 搜索只会因为三种原因停下：访问预算、墙钟（仅落子/显式请求）、`stop`。

## 2. 规则状态：`Position`

`Position` 是规则层唯一真源，同时服务 UI、GTP 引擎、循环赛和测试。

内部由三部分组成：

1. `board_[81]`：`+1..+9` 红子，`-1..-9` 黑子，`0` 空。用于快速单格读取与计分。
2. bitboard：
   - `empty_`：81 位空格；
   - `occupied_[2]`：红/黑占用；
   - `number_[2][10]`：每种颜色每个数字的棋子位图。
   bitboard 用于合法着法生成、终局判断、NN 特征和未来增量评估。
3. `zobrist_`：落子/悔棋时 XOR 增量维护的棋子哈希。`key()` 再混入行棋方，作为 TT 键。

不把 MCTS 的树节点和状态混在一起。树节点只存“从父节点走到这里的着法”和统计量；模拟时用可撤销的 `Position` 走路径。

## 3. 剪枝引擎与置换表

- 算法：迭代加深 Alpha-Beta；根节点保留最佳着法与 PV，并通过进度回调输出
  `depth/nodes/score/best`。叶节点在“黑方走完”的红方回合评估，评估纳入稳定格、
  边界格和双方剩余手牌。
- 深度只按“完整回合”递增：红方根搜索 2/4/6… 层，黑方根搜索 1/3/5… 层，
  避免在红方刚下完的奇数层评估。
- 由于每层约 700 个合法着法，纯全宽深搜在 650 ms 内只能到 2 层左右。
  当前使用选择性束搜索：根最多 32 个、深层最多 16 个候选（按 TT/PV/静态分排序），
  在 650 ms 下可到 4 层；`rootBranching/maxBranching=0` 可切回全宽。这是有意的
  强度/深度折中，不是“忘记加深”。
- 置换表 `TranspositionTable` 是独立类：
  - 固定大小、2 的幂、直接索引；
  - entry 保存 `key / depth / bound(EXACT/LOWER/UPPER) / value / best move / age`；
  - 每次新搜索递增 generation，深搜优先替换浅层旧项；
  - 值统一为“轮到走棋的一方”的视角。
- 根节点使用 **aspiration window（渴望窗口）**：第一次完整迭代后在上一轮分数
  周围开 ±24 分的窗口；若 fail-low/fail-high，则把窗口翻倍并把当前最佳着法
  钉到根着法序列首位重搜。只有窗口内精确完成的一轮才会替换上一轮结果，
  超时/中断不会把半轮搜索写回结果。
- 根和内部节点都使用 **PVS 零窗口**：第一个着法用完整窗口，后续着法先用
  `[alpha, alpha+1]` 侦察；只有超过 alpha 才用完整窗口重搜。`window/wres/pvs`
  会随 `info` 输出，便于确认窗口搜索实际生效。
- TT 由 `AlphaBetaEngine` 跨手持有；`think()` 临时搜索使用临时 TT。
- 大架构约束：TT 不放进 `Position`，也不把 PV 存在 TT 之外的单例里；搜索函数接收 `TranspositionTable&`。

## 4. MCTS：SoA 与树复用

MCTS 采用 **Structure of Arrays**，避免每个节点一个 `std::vector<Child>` 的对象开销：

```
nodeMove[]        每节点对应着法
nodeParent[]
nodeFirstEdge[]   该节点在全局 edge 数组中的起点
nodeEdgeCount[]
nodeVisits[]
nodeValueSum[]    从该节点行棋方视角的累计值
nodeTerminal[]

edgeMove[]        全局边数组（SoA）
edgeChild[]       -1 表示尚未展开
edgeVisits[]
edgeValueSum[]    从父节点行棋方视角的累计值
edgePrior[]       NN 先验 / 均匀先验
```

- 每个节点只展开一部分候选：根节点默认最多 `rootChildren=64`（MCTS-NN 为 16），
  深层节点最多 `maxChildren` 个。早期版本在根节点展开全部 729 个合法着法，
  在短时限下会把访问全部耗在根层；现已改为有限根宽度。
- 子节点按需创建：先建 edge，首次选中时再分配 child 节点。
- 树复用：`genmove/analyze` 结束后根节点保留。实际落子调用 `advance(move)`；若该着法在根边中，根移动到对应子节点，否则整棵树重置。
- 分析快照从根节点的边统计生成；除 `moves` 前几个候选外，还生成 81 格
  `cells` 推荐（推荐数字、胜率、访问数、平均终局分）。根节点保留全部着法边，
  但只有有限子集可被搜索选中，因此 UI 能展示所有点而搜索不被 729 宽根拖死。
- 值统一用“父节点行棋方视角”存边统计，回溯时逐层翻转，避免根/子视角混用。

## 5. MCTS + NN

- 策略价值网络是一个独立接口 `PolicyValueNet`，当前实现是 KataGo 风格 ResNet：
  - 输入：18 个石头平面 + 行棋方平面，共 19 个 9×9 平面；
  - 全局向量：18 个双方手牌计数 + 行棋方 + 当前局面分 + 手数 + 空格数；
  - 输出：729 个 `(格子,数字)` 的 policy logits + 一个 value（当前行棋方胜率）；
  - 通道数/残差块数不是编译期常量：模型头记录 shape，C++ 运行时按头加载。
    默认兼容 16×2 的 v1，服务器训练使用 32×4。
- 网络推理只发生在 MCTS 的扩展/叶评估点；当前 value 已不再与随机 rollout 平均，
  以缩短短时限下的 NN 评估成本。
- 权重从外部文件加载，缺失时引擎仍可启动并明确报告 `nn=none`，不会静默假装有 NN。
- `shuyi_selfplay` 用 MCTS 自对弈生成数据：根节点 Dirichlet 噪声、根访问分布作为
  policy target、终局胜负作为 value target；`training/train_resnet.py` 训练并导出 C++
  二进制权重。
- `info` 行携带 `nn=文件名`，模型版本/数据版本可随文件名区分。

## 6. GTP-like 协议

引擎进程从 stdin 读命令，stdout 输出。命令成功以 `=` 开头，错误以 `?` 开头；分析流是 `info ...` 行。

首批命令：

```
protocol_version
name
version
list_commands
boardsize 9
clear_board
set_engine random|pruning|mcts|mcts_nn
set_time 650
set_a 1
play red e5 5
genmove red
analyze red
stop
quit
```

`info` 行示例：

```
info engine=pruning depth=4 nodes=12000 window=24 wres=0 pvs=9 score=+5 winrate=0.68 best=e5:7 pv=e5:7,c4:3
info engine=mcts visits=20000 winrate=0.57 best=e5:7 moves=e5:7:0.57:9000,c4:3:0.21:3000
```

- `score` 是当前行棋方视角的分数（含 A值），用于剪枝展示。
- `winrate` 是当前行棋方的期望得分（胜 1、平 0.5、负 0）。
- `moves` 只带根节点前若干个候选，供 UI 画胜率分布。
- `genmove` 结束输出 `= e5 7`；`analyze` 持续输出 `info`，直到 `stop`。
- 所有搜索在独立工作线程；主线程继续读 `stop`，所以 UI 永远能取消。

## 7. ELO 评测

统一以 **随机走子引擎为 0 分基准**。

- 循环赛工具 `shuyi_tournament` 在进程内直接调用引擎接口，避免管道开销。
- 每对引擎按先后手交替对局，使用固定 seed 集合，结果可复现。
- 记录 `W/D/L`；ELO 用固定随机基准为 0 的 Bradley-Terry/Logistic 拟合。
- 置信区间用对局 bootstrap（重采样对局、重新拟合）得到 95% 区间。
- 输出机器可读 CSV 和 Markdown；时间控制、每对局数、引擎版本写入报告头。
- 因为对局数有限，ELO 区间可能很宽；报告中必须同时给出局数和 CI，禁止只报一个点估计。

## 8. 实施顺序

1. `Position` bitboard + Zobrist，保持现有规则测试全绿。
2. `TranspositionTable`、剪枝引擎接口和 GTP 进程骨架。
3. MCTS SoA + 树复用 + 分析快照。
4. NN 接口、模型加载、MCTS-NN 路径。
5. UI 引擎进程管理、10Hz 分析面板。
6. ELO 循环赛、文档和完整测试。

## 9. 当前实现状态

- `Position` 已有 81 位双字 bitboard、Zobrist 增量键、增量计分和 `validate()` 自检。
- `TranspositionTable` 是独立类，剪枝引擎跨手持有；`think()` 使用临时表。
- `shuyi_engine.exe` 已实现命令循环、搜索工作线程、`stop` 取消、`info` 100ms 节流。
- UI 通过 `CreateProcessW` + 匿名管道启动引擎；reader 线程入队，100ms Timer 消费并重绘。
- MCTS 节点/边使用 SoA；`onPlay()` 尝试把根推进到对应子节点，失败则重置。
- MCTS-NN 使用 KataGo 风格 ResNet：19 个输入平面、可配置通道/残差块，
  策略头输出 729 个 `(格子,数字)` logits，价值头融合 22 维全局特征
  （双方手牌、行棋方、当前局面分、手数、空格数）。仓库基线模型
  `models/shuyi_resnet_v1.bin` 是 16×2；服务器训练脚本默认 32×4，
  导出格式相同。
- 根节点支持 Dirichlet 噪声；`shuyi_selfplay` 以根访问分布为 policy target、
  终局胜负为 value target 生成 v2 数据，`training/train_resnet.py` 用 PyTorch
  训练并导出 C++ 可直接推理的二进制权重；`training/pipeline.py` 负责
  15 代自对弈、训练、双模型晋升的编排。
- 当前默认融合为 `0.5 * policy + 0.5 * uniform`、纯 value head
  （`nnValueBlend=1.0`）；UI 的 MCTS-NN 根候选宽度默认 16，自对弈可单独
  提高到 32，避免短时限下把访问全部花在根节点的数百个候选上。
- 剪枝叶节点在“黑方刚走完、轮到红方”的位置评估；评估函数区分已稳定格和边界格，
  并把双方手牌中的高数字潜力纳入 `handWeight` 项，避免只在红方刚落子后做静态判断。
- ELO 工具已实现随机基准、先后手交替、随机开局、Bradley-Terry 拟合和 200 次 bootstrap CI。

### 异步协议的一个细节

`genmove` 为了允许 `stop` 不被阻塞，采用两段响应：

```
=                      ← 命令已被接受
info ...               ← 搜索过程
= e5 7                 ← 最终着法
```

客户端不能把第一个空 `=` 当成最终着法。UI 侧用 `g_enginePending` 标记，只接受
带坐标/数字的最终行。`analyze` 同样先返回空 `=`，之后一直推 `info` 直到 `stop`。
