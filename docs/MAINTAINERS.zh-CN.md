# 维护者文件地图（数弈）

这份文档只回答三个问题：**某个文件在哪、它是干什么的、改它要注意什么。**
改动代码之前先读这里，比顺着目录猜快得多。

---

## 1. 顶层：三块代码 + 三块数据

```
数弈/
├── engine/          C++ 引擎：规则、搜索、网络。可单独编译成静态库 + 行协议引擎
├── ui/              Win32 前端（GDI），以及和引擎进程通信的客户端
├── experimental/    自对弈、训练、评测、一次性研究工具（不参与 UI 发布）
├── data/            共享测试夹具（终局见证等），引擎测试和实验工具都读它
├── models/          手写/迁移的固定模型夹具 + 运行时默认模型
├── runs/            训练与评测的输出（每代数据、模型、日志、JSON）
├── docs/            文档：规则、协议、架构、评测方法、计划、历次运行报告
├── build/           CMake 构建目录（MinGW，CPU）
├── build-cuda/      CUDA 构建目录（MSVC + nvcc，自对弈用）
└── CMakeLists.txt   唯一的构建入口
```

三块代码的边界是硬的，别混：

| 目录 | 可以依赖 | 不可以依赖 |
|---|---|---|
| `engine/` | 只依赖自己 | 不许 `#include` UI 或 experimental 的东西；不许读 `runs/` |
| `ui/` | `engine/` 的头（`shuyi/…`） | 不许直接调 experimental 的工具 |
| `experimental/` | `engine/` 的头 | UI 不许依赖它；它的失败不能影响引擎测试 |

`models/`、`data/`、`runs/` 是三方共享的数据，不属于任何一块代码。

---

## 2. engine/ —— 规则与搜索

```
engine/include/shuyi/
├── rules.h          Ruleset 枚举（Classic / Blocking）、规则名与后缀、过手哨兵
├── position.h       棋盘状态：落子、撤销、合法着法、终局、哈希、阻挡判定（kBlockingReach）
├── ai.h             剪枝引擎：SearchOptions（含 nodeLimit）、SearchResult、think/chooseMove
├── ai_policy.h      "评估 + 排序"策略接口，规则特异的实现挂在这里
├── mcts.h           （无）MCTS 在 src/mcts.cpp 里，接口在 engine.h 的 Engine
├── engine.h         Engine 基类、EngineOptions、SearchRequest（visits 预算）、Analysis
├── nn.h             策略价值网络：文件格式 SHUYINN3、特征平面、CPU 前向、NnOutput
├── nn_cuda.h        CUDA 批量前向的接口（nn_cuda.cu 实现，也被 shuyi_engine 用）
├── tt.h             置换表
├── format.h         显示用的数字格式（访问数、胜率、平均分）
└── io.h             UTF-8 路径与 argv 处理（仓库路径含中文，必须走这里）
```

```
engine/src/
├── position.cpp     落子/撤销/合法着法/计分/哈希
├── ai.cpp           迭代加深 + aspiration + PVS 剪枝搜索（nodeLimit 在这里生效）
├── ai_blocking.cpp  阻挡规则的评估与排序（挤压项、封锁数）——规则特异代码只放这里
├── mcts.cpp         PUCT 树搜索（含强制过手节点、分数头回传、访问预算）
├── engine.cpp       Engine 四种实现（random/pruning/mcts/mcts_nn）的适配层
├── engine_main.cpp  行协议引擎（stdin/stdout）；`set_backend auto` 时挂上 CUDA 前向
├── nn.cpp           模型文件读写 + CPU 前向
├── nn_cuda.cu       im2col + cuBLAS 的 CUDA 前向（含分数头）
└── tt.cpp           置换表
```

```
engine/tests/
├── test_position.cpp  规则与局面基本行为
├── test_rules.cpp     阻挡规则与强制过手（12 个用例）
├── test_komi.cpp      A值/平局可达性（读 data/score_witnesses.txt 的 147 个终局）
├── test_engine.cpp    搜索、置换表、网络、访问预算（testVisitBudget）
├── protocol_smoke.py  行协议冒烟
└── visit_budget_smoke.py  四种引擎都必须正好用完给定 visits（机器无关性）
```

改引擎时的自查清单：

1. `cmake --build build && cd build && ctest` 全绿；
2. 改了模型/数据格式 → 同步 `nn.cpp` 的版本号、`experimental/training/train_resnet.py` 的写出端，
   并且**清空 CUDA 构建目录**（旧 `.o` 会按旧布局读文件而 `--check` 照样通过）；
3. 改了搜索预算语义 → 跑 `engine/tests/visit_budget_smoke.py`，评分表的前提就是它。

---

## 3. ui/ —— Win32 前端

```
ui/include/engine_client.h   子进程 + 行协议的客户端（启动、发送、非阻塞读）
ui/src/engine_client.cpp     同上实现
ui/src/ui_win32.cpp          全部界面：窗口、棋盘绘制、面板、菜单、AI 标记
ui/tests/ui_menu_smoke.py    无界面回归：`shuyi.exe --dump-models` 导出的模型菜单
```

`ui_win32.cpp` 一个文件装了整个窗口，按下面顺序找东西：

| 想改什么 | 找什么 |
|---|---|
| 棋盘/棋子/标记的画法 | `drawBoard`、`drawStone`、`drawAnalysisMarks` |
| 右侧面板的三栏（胜率/访问数/平均分） | `drawPanel`、`formatVisits` 等（在 engine/format.h） |
| 菜单与快捷键 | `createMenuBar`、`WM_COMMAND` |
| 模型二级菜单 | `scanModels`、`populateModelMenu`、`selectModel` |
| 规则切换、A值 | `setRuleset`、`setAValue` |
| 引擎进程与协议 | `startEngineProcess`、`sendEngine`、`handleEngineLine` |
| 推理后端（CPU/GPU） | `set_backend auto`（`engine_main.cpp` 的 `attachEvaluator`）、`EngineOptions::nnOverride` |
| 选哪个引擎进程 | `engineCandidates` / `chooseEnginePath` / `probeEngine`（协议 `capabilities`），`--dump-engine` 可无界面诊断 |

两条硬规则：

1. **模型目录要能在任意工作目录下找到**：菜单扫描 exe 所在目录、它的父目录（仓库根）、
   当前目录，以及 `SHUYI_MODEL_DIRS`（分号分隔）。只按相对路径找会在从 `build/` 启动时漏掉
   `runs/*/models`。
2. **引擎进程按能力挑，不按构建挑**：UI 与引擎是分离进程，界面只负责"拉起最快的引擎"。
   候选顺序见 `engineCandidates()`，用 `capabilities` 探测 `cuda=1`；`SHUYI_ENGINE` 可覆盖。
2. **路径编码**：Win32 的 `*A` API 给的是 ANSI，引擎进程要 UTF-8（`Utf8Args` 会转）。
   `ModelEntry` 因此同时存 `path`（ANSI）和 `enginePath`（UTF-8），别混用。

---

## 4. experimental/ —— 自对弈、训练、评测

```
experimental/selfplay/
├── selfplay_gpu.cpp   生产用自对弈：CUDA 批量前向 + 每 tick 每局一个叶子；也提供
│                      --mode match（模型对打）、--mode serve（给 arena 当玩家）、--check
├── selfplay.cpp       CPU 版自对弈（慢，用于无 GPU 的冒烟测试）
├── match.cpp          两种引擎对打的轻量工具（早期，时间预算，已被 arena.py 取代）
└── tournament.cpp     池内循环赛（早期，时间预算，已被 arena.py/elo.py 取代）

experimental/research/
├── a9_lemma.cpp       文档里的有限引理程序（黑 9 三枚 → 损失下界），测试会跑
├── komi_explore.cpp   寻找终局分数见证（data/score_witnesses.txt 的来源）
└── nnbench.cpp        单模型前向基准

experimental/training/            （见该目录 README.md，那里是训练手册）
├── pipeline.py        一代一代跑：随机→单层→MCTS→训练→arena→ELO
├── train_resnet.py    训练器（PyTorch），模型格式的写出端
├── arena.py           横评：异构玩家（random/pruning/mcts/mcts_nn/gpu）同池
├── elo.py             Elo 拟合（Newton + bootstrap），可合并多份矩阵
├── avalue.py          A值标定（模型自对弈扫 komi，logistic 求交点）
├── cross_eval.py      线路横评 + 代数爬梯合并成一张表
├── evalconfig.py      EVAL_VISITS：所有评测唯一的预算常量
├── rule_probe.py      独立的计分实现（横评用它判终局，和引擎互校）
├── target_stats.py    目标分布健康检查（闸门）
├── model_migrate.py   旧模型头迁移（只改头，不动权重）
├── summarize_run.py   manifest → markdown 表
├── pull.py            断点续拉服务器产物
├── ssh.py / remote*.py / *.sh   服务器部署与运维
└── README.md          训练手册：原则、数据来源、闸门、横评、规则隔离、改进记录

experimental/tests/selfplay_smoke.py   CPU 自对弈冒烟（ctest 调用）
```

`experimental/` 里有些脚本是**历史遗留**（`gpu_selfplay.py`、`oneply_selfplay.py`、
`komi_probe.py`、`diag_target.py`、`diagnose_openings.py`、`fix_values.py`、
`verify_rules.py`、`check_layout.py`）：它们记录当时的排查过程，C++ 版本已经取代了其中的主流程。
新写的东西不要再依赖它们；哪天确认没用了就删，别让它们继续散落在目录里。

---

## 5. 数据、模型、输出

| 路径 | 是什么 | 谁读它 |
|---|---|---|
| `data/score_witnesses.txt` | 147 个终局见证（分数 −73…+73），SHA-256 记在规则文档里 | `test_komi.cpp`、`komi_explore.cpp` |
| `models/shuyi_resnet_v1.bin` / `_v2.bin` | 手工构造/迁移的固定夹具 | 单测、协议冒烟、默认模型 |
| `models/experimental/` | 早期实验模型 | 手工 |
| `runs/<ruleset>/<run>/gen_XXX/` | 每代：`selfplay/shard_000.bin`、训练日志、`candidate.bin` | `pipeline.py` 续跑、人看 |
| `runs/<ruleset>/<run>/models/` | 每代命名模型 `b4c32-s..-d..-blk.bin` | arena、ELO、UI 菜单 |
| `runs/arena/`、`runs/avalue/`、`runs/elo/` | 评测矩阵、A值标定、对局记录 | `elo.py`、文档里的表 |
| `docs/reports/*_run/` | 历史运行留下的模型与曲线（只读参考） | 文档、旧表复现 |

**`runs/` 不是代码**：它随时可以被重新生成，但服务器是按小时计费的，所以在租用机器上跑出来的
东西一定要先 `pull.py` 拉回本地（默认路径就是 `runs/`）。

---

## 6. 构建与测试

```bash
# CPU（MinGW / Linux 均可）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
cd build && ctest --output-on-failure          # 8 项

# CUDA（Windows + MSVC；nvcc 不能写中文路径，所以用 junction C:\shuyi_src）
cmake -S C:/shuyi_src -B C:/shuyi_cuda_build -G Ninja -DSHUYI_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=E:/CUDA/v12.4
```

测试清单（`ctest` 名称 → 断言什么）：

| 名称 | 断言 |
|---|---|
| `shuyi_tests` | 规则、局面、A值、搜索、网络、**访问预算** |
| `shuyi_match_smoke` | 两种引擎能打完一局 |
| `shuyi_tournament_smoke` | 循环赛能产出 Elo 表 |
| `shuyi_a9_lemma` | 文档里的有限引理（Windows） |
| `shuyi_protocol` | 行协议握手与非法输入 |
| `shuyi_selfplay_smoke` | CPU 自对弈能产出数据 |
| `shuyi_visit_budget` | 四种引擎都按 visits 停，且两次运行一致 |
| `shuyi_ui_menu` | 模型二级菜单列出了所有模型、tag 与 id 正确 |

---

## 7. 改任何东西之前

1. **先看 `experimental/training/README.md` 的原则**：ELO 是唯一标准（不是 val loss）、
   random 钉 0、顺次训练、评测预算固定为 `EVAL_VISITS` 且不许用时间。
2. 规则特异的代码只放 `ai_blocking.cpp` / `rules.h` / `position.cpp` 的阻挡分支，
   不要在搜索骨架里加 `if (ruleset == Blocking)`。
3. 改了格式（模型/数据/协议）三处一起改：写出端、读入端、测试；并清空 CUDA 构建目录。
4. 文档跟着改：本文件的表格 + 根 `README.md` 的目录结构 + 训练手册的改进记录。
