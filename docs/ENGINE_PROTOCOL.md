# 数弈引擎进程协议（GTP-like）

`shuyi_engine.exe` 是独立于 UI 的无窗口引擎进程。UI 通过 stdin/stdout 行协议访问它；
搜索在引擎工作线程中执行，主线程始终可以读取 `stop`。

## 1. 启动

```text
shuyi_engine.exe
```

stdout 只用于协议；诊断信息写 stderr（UI 客户端会重定向到 NUL）。

## 2. 基本响应

- 成功：`= ...`
- 错误：`? ...`
- 分析流：`info ...`（不带 `=` 前缀）
- 一个异步命令（`genmove`/`analyze`）会**立即**返回一个空的 `=`，随后推送
  `info`，`genmove` 最后再推一条 `= e5 7`。客户端必须按命令状态关联，不能把
  第一个 `=` 当成最终着法。

## 3. 命令

| 命令 | 说明 |
|---|---|
| `protocol_version` | 返回 `1` |
| `name` | `ShuyiEngine` |
| `version` | 引擎版本 |
| `list_commands` | 命令列表 |
| `capabilities` | `= engine=shuyi_engine cuda=0\|1 proto=1 visits=1 rulesets=classic,blocking` |
| `boardsize 9` | 只支持 9 |
| `clear_board` | 清空局面 |
| `set_engine random\|pruning\|mcts\|mcts_nn` | 切换搜索器 |
| `set_visits 512` | **固定模拟预算**（剪枝解释成节点数）；`>0` 时忽略 `set_time`，analyze 模式也遵守 |
| `set_time 650` | 每手搜索时间（ms）；没有 `set_visits` 时才是预算 |
| `set_a 1` | 设置 A值；会重建引擎 |
| `set_model models/shuyi_resnet_v1.bin` | 设置 ResNet 权重（文件头携带通道/块数）；空路径表示不使用 |
| `set_backend cpu\|cuda\|auto` | 选择前向后端；构建里没有 CUDA 时静默落到 CPU |
| `set_window 24` | 设置剪枝 aspiration 窗口半宽；0 表示完整窗口 |
| `set_pvs on\|off` | 开关剪枝 PVS 零窗口侦察 |
| `play red e5 5` | 红方在 `e5` 放数字 5 |
| `genmove red` | 先回空 `=` 应答，搜索结束后再回 `= e5 5` |
| `analyze red` | 持续分析，直到 `stop` 或预算用完 |
| `stop` | 停止当前搜索/分析 |
| `quit` | 退出 |

`play` 的颜色必须与当前行棋方一致；数字必须在 1–9 且该方仍有剩余。

`capabilities` 是界面挑选引擎的唯一依据：`cuda=1` 表示这个二进制能把前向放到 GPU，
`visits=1` 表示它接受 `set_visits`（界面据此用访问数而不是毫秒当预算）。

## 4. `info` 行

剪枝：

```text
info engine=pruning depth=4 nodes=12000 window=24 wres=0 pvs=9 score=+5 winrate=0.68 best=e5:7 pv=e5:7,c4:3
```

MCTS / MCTS-NN：

```text
info engine=mcts_nn nn=b4c32-s34760-d470705.bin backend=cuda hasscore=1 proven=0 visits=512 nodes=25305 \
  score=+3.20 winrate=0.57 best=e5:7 \
  moves=e5:7:0.57:9000:4.2,c4:3:0.21:3000:-1.1 \
  cells=a1:1:0.52:30:1.0:0,...,i9:4:0.48:20:-0.4:0
```

- `visits`：实际完成的模拟数。等于预算；**除非** `proven=1`（解算了就提前停）。
- `hasscore`：`score` 是否是真分数。为 1 时它只能来自两种来源：已解算的终局，
  或模型自带的分数头。为 0 时界面显示 `--`。**禁止**再用胜率线性映射出目差。
- `proven`：这个局面已被规则求解，`winrate`/`score`/`best` 是硬结果而不是估计。
- `score`：当前行棋方视角的分数/平均终局分（含 A值）。剪枝是搜索值；MCTS 是
  终局分（或分数头）的平均。`proven=1` 且目差已定时才是精确目差，否则可能不出现。
- `winrate`：当前行棋方期望得分（胜 1、平 0.5、负 0），不是红方胜率。
  `proven=0` 时它是搜索得到的均值（未证明），`proven=1` 时是精确值。
- `moves`：根节点前几个候选，字段为 `坐标:数字:胜率:访问数:平均终局分`。
- `cells`：**所有空格**的展示用推荐，字段为
  `坐标:推荐数字:胜率:访问数:平均终局分:已解算`。未搜索的格子访问数为 0。
  第 6 个字段为 1 表示该点已解算：胜率和目差是精确的，界面应当直接采信，
  而不是用"访问数不够"的置信下界把它压下去。老界面忽略多出来的字段即可。
- **对称点同值**：棋盘在 8 个等距变换下不变，一次搜索只做一个对称轨道。同一轨道上的
  格子会收到**完全相同**的推荐数字、胜率、访问数和平均终局分（形如
  `c1:3:0.607:135` 与 `g1:3:0.607:135`）。界面不需要自己算对称，直接用这些数字画。
- `window`：当前 aspiration 窗口半宽（0 表示完整窗口）；`wres` 是窗口
  fail-low/fail-high 后的重搜次数；`pvs` 是零窗口侦察后需要完整重搜的次数。
  这三个字段只在剪枝 `info` 中出现。
- `backend`：`cuda` 或 `cpu`，即这次前向实际落在哪。
- `nn=`：MCTS-NN 的模型文件名；没有模型时为 `none`。

## 5. UI 侧异步模型

UI 不做搜索，只在 100 ms Timer 中：

1. 从 reader 线程队列抽取 stdout 行；
2. 更新展示用 `EngineAnalysisView`；
3. 重绘约 10Hz。

`genmove` 的结果由 `g_enginePending` 标记；`analyze` 的流由 `g_analysisOn` 标记。
每次本地落子、悔棋、新局、A值修改或引擎切换都会先 `stop`，再同步坐标；
这样旧搜索的结果不会污染新局面。
