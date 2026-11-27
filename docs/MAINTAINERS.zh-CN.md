# 维护者文件地图（数弈）

这份文档只回答三个问题：**某个文件在哪、它是干什么的、改它要注意什么。**
改动代码之前先读这里，比顺着目录猜快得多。

---

## 1. 顶层：三个产品 + 两份开发资料

```
数弈/
├── core/            规则库：棋盘、着法合法性、计分、对称。两侧都只链接它，且只链接它
├── engine/          shuyi_engine.exe：无头引擎（搜索/分析/走棋/bench/自对弈）+ 行协议
├── ui/              shuyi.exe：Win32 界面。只管规则与调用，不含任何搜索
├── protocol/        引擎协议（唯一的契约）+ 一致性测试
├── tools/           开发工具：bench、证明检查、自对弈、锦标赛。不随发布
├── training/        Python 训练流水线。不随发布
├── release/         打包脚本（alpha 包由这里生成）
├── data/            共享测试夹具（终局见证等）
├── models/          随包发布的模型 + 测试夹具（shuyi_resnet_v1/v2）
├── runs/            训练与评测输出（不入库；布局与命名见 runs/README.md）
├── docs/            规则、架构、评测方法、计划、历次运行报告
├── build/           CPU 构建目录（MinGW）
└── build-cuda/      junction → 仓库外的 CUDA 构建树（MSVC + nvcc）
```

依赖关系是硬的，别混：

| 目录 | 可以依赖 | 绝不可以依赖 |
|---|---|---|
| `core/` | 标准库 | 搜索引擎、网络、Win32、任何 I/O 策略 |
| `engine/` | `core/` | `ui/`、`tools/`、`runs/`、Win32 |
| `ui/` | `core/` | `engine/` 的搜索、网络、CUDA——**只能**通过协议通信 |
| `tools/` | `core/`、`engine/` 的库 | 被产品反向依赖 |

`ui/` 链接 `core/` 是允许的（界面是规则与状态的权威）；`ui/` 链接 `shuyi_search` 是禁止的，
那等于把引擎塞进界面包。CMake 里没有这个 target 可以链，别加回去。

---

## 2. core/ —— 规则

```
core/include/shuyi/
├── rules.h          Ruleset 枚举（Classic / Blocking）、规则名与后缀、过手哨兵
├── position.h       棋盘状态：落子/撤销/合法着法/终局/哈希/对称（kSymmetryCount=8）
├── format.h         显示格式（访问数、胜率、Wilson 下界、平均分）
└── io.h             UTF-8 路径与 argv（仓库路径含中文，必须走这里）
core/src/position.cpp  落子/撤销/合法着法/计分/哈希/对称变换
```

改规则时：`position.cpp` 的合法着法与 `Position::play` 必须同时改，测试在
`engine/tests/test_rules.cpp`（含对称性与邻接关系）。计分只有一个实现，别在别处重写。

---

## 3. engine/ —— 搜索与网络

```
engine/include/shuyi/
├── engine.h         Engine 基类、EngineOptions、SearchRequest（visits/reportIntervalMs）、Analysis
├── endgame.h        终盘精确求解：节点预算 + 墙钟
├── mcts.h           （无独立头）MCTS 在 src/mcts.cpp，接口即 Engine
├── nn.h             策略价值网络：SHUYINN3/V4 格式、特征平面、CPU 前向、NnOutput
├── nn_cuda.h        CUDA 批量前向接口（nn_cuda.cu 实现）
└── protocol.h       行协议会话（命令解析、位置状态、单搜索线程）
```

```
engine/src/
├── endgame.cpp      终盘 negamax（play/undo），节点预算 + 墙钟
├── mcts.cpp         PUCT 树搜索（强制过手、证明回传、对称轨道、访问预算）
├── engine.cpp       三种 Engine 实现（random/mcts/mcts_nn）的适配层
├── nn.cpp           网络加载与前向
├── nn_cuda.cu       CUDA 批量前向
├── protocol.cpp     命令集、错误处理、启动握手、bench/selfplay
├── protocol_json.cpp  Analysis → JSON（协议第 4 节的唯一实现）
└── main.cpp         stdin 循环
```

改协议时：`protocol.cpp`（命令）和 `protocol_json.cpp`（字段）必须和
`protocol/ENGINE_PROTOCOL.md` 一起改，然后跑 `protocol/tests/`。

**不要**加回这些（都删除过，原因写在 git 历史里）：
- 树大小上限：撞上限时叶子会被当成终局，产生假的 100% 胜率；
- `SHUYI_UI_BUILTIN_AI`：界面内嵌搜索，等于界面包里带引擎；
- `set_backend` 的静默降级：`cuda` 失败必须 `?`，`auto` 必须在回复里报告实际后端；
- `mcts_nn` 没有模型还能跑：现在 `ready()` 会拒绝。

---

## 4. ui/ —— 界面

```
ui/include/
├── game_record.h    棋谱树：节点、变着、每个局面缓存的分析、保存/加载
├── engine_client.h  子进程 + 管道 + 读线程 + `command()` 同步握手
├── engine_view.h    分析视图结构 + `parseAnalysis()`（JSON → 视图）
├── json.h           极简 JSON 解析器
└── discovery.h      引擎候选路径与 `capabilities` 探测
ui/src/
├── ui_win32.cpp     窗口、菜单、布局、绘制、按键与鼠标
├── game_record.cpp  棋谱树、PGN 式文本、与上一手预测的差值
├── engine_client.cpp 进程/管道/读线程
├── engine_protocol.cpp  解析 info JSON、着法 token
├── json.cpp         JSON 解析
└── discovery.cpp    引擎发现
```

界面里的三条硬规矩：

1. **不计算**。没有引擎就显示错误并提示去“引擎 → 选择引擎程序…”，绝不自己走一步。
2. **不猜**。引擎报错就显示错误；解析不了的分析数据也显示错误，不保留成空白棋盘。
3. **协议版本硬校验**。`proto != 2` 直接拒绝并弹窗说明，不做“可能兼容”的尝试。

启动顺序（`startEngineProcess`）：`boardsize` → `ruleset` → `set_a` → `set_model` →
`set_engine` → `set_backend auto` → `set_visits`，每条都用 `EngineClient::command()`
同步等回复，任何 `?` 都会中止启动并弹窗。

---

## 5. protocol/、tools/、training/

```
protocol/ENGINE_PROTOCOL.md   协议全文（命令表、JSON 字段表、错误与版本规则）
protocol/tests/engine_client.py  Python 客户端（也供训练脚本使用）
protocol/tests/protocol_smoke.py 命令/流式/预算/错误行为的一致性测试
protocol/tests/visit_budget_smoke.py  预算与确定性
tools/research/               证明检查、终局探针、bench、一次性研究
tools/selfplay/               自对弈、锦标赛、对局工具
training/                     训练流水线（pipeline.py 是入口）、评测（arena/elo）、文档
```

`tools/` 与 `training/` 永远不进发布包；`release/make_alpha.py` 只拷 `ui/`、`engine/`、
`models/`、`protocol/` 的产物。

---

## 6. 构建与打包

```bash
# 开发者：核心 + 引擎 + 界面 + 测试
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 8
cd build && ctest --output-on-failure

# 只有引擎 / 只有界面（两个产品各自独立）
cmake -S engine -B build-engine -DSHUYI_CUDA=ON
cmake -S ui     -B build-ui

# 开发工具（默认不编）
cmake -S .    -B build -DSHUYI_BUILD_TOOLS=ON
cmake -S tools -B build-tools

# alpha 发布包
python release/make_alpha.py --engine build-cuda/shuyi_engine.exe \
    --ui build-cuda/shuyi.exe --model models/b4c32-s34760-d470705.bin
```

CUDA 构建走 `tools\build_cuda.bat`：nvcc 不接受中文路径，所以源码经一个 ASCII 镜像目录进去，
构建树也放在仓库外（默认 `F:\projects\shuyi_cuda\{src,build}`，`SHUYI_CUDA_ROOT` 可改）。
仓库里的 `build-cuda` 是指向构建树的 junction，不要把它当真目录提交。

---

## 7. 改什么去哪里找

| 想改 | 文件 |
|---|---|
| 规则（合法着法、计分、终局） | `core/src/position.cpp`、`engine/tests/test_rules.cpp` |
| 搜索行为（PUCT、预算、证明、对称） | `engine/src/mcts.cpp` |
| 终盘精确求解 | `engine/src/endgame.cpp`、`engine/include/shuyi/endgame.h` |
| 网络格式/特征 | `engine/src/nn.cpp`、`training/train_resnet.py`（必须同步） |
| 协议命令或字段 | `engine/src/protocol.cpp`、`engine/src/protocol_json.cpp`、`protocol/ENGINE_PROTOCOL.md` |
| 界面绘制/交互 | `ui/src/ui_win32.cpp` |
| 界面如何解析分析 | `ui/src/engine_protocol.cpp`、`protocol/tests/protocol_smoke.py` |
| 引擎发现/启动 | `ui/src/discovery.cpp`、`startEngineProcess()` |
| 发布包内容 | `release/make_alpha.py` |
