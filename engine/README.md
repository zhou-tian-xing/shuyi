# engine/ — 规则、搜索、网络

这是引擎库（`shuyi_core`）和行协议引擎进程（`shuyi_engine`）。**它不认识 UI，也不认识训练脚本。**

```
include/shuyi/
  rules.h         规则枚举 Classic / Blocking、规则名与模型后缀、过手哨兵（kPassCell/kPassNumber）
  position.h      棋盘状态：落子、撤销、合法着法、searchMoves()、终局、哈希、阻挡判定
  ai.h            剪枝搜索：SearchOptions（nodeLimit 在这里）、SearchResult、think / chooseMove
  ai_policy.h     "评估 + 排序"策略接口；规则特异的实现由 ai_blocking.cpp 提供
  engine.h        Engine 基类、EngineOptions、SearchRequest（visits 预算）、Analysis
  nn.h            策略价值网络：SHUYINN3 文件格式、特征平面、CPU 前向、分数头
  nn_cuda.h       CUDA 批量前向接口
  tt.h            置换表
  format.h        显示格式（访问数 1.2k/12k、胜率、平均分）
  io.h            UTF-8 路径与 argv（仓库路径含中文，所有文件读写都走这里）

src/
  position.cpp    落子/撤销/合法着法/计分/Zobrist
  ai.cpp          迭代加深 + aspiration window + PVS；nodeLimit 在这里生效
  ai_blocking.cpp 阻挡规则的评估（挤压项）与排序（封锁数）——规则特异代码只放这里
  mcts.cpp        PUCT 树搜索：强制过手节点、分数头回传、固定 visits 预算
  engine.cpp      四种 Engine 实现（random / pruning / mcts / mcts_nn）
  engine_main.cpp 行协议进程：UI 和 arena 都通过 stdin/stdout 驱动它
  nn.cpp          模型读写 + CPU 前向
  nn_cuda.cu      im2col + cuBLAS 的 CUDA 前向（含分数头）
  tt.cpp          置换表

tests/
  test_position.cpp / test_rules.cpp / test_komi.cpp / test_engine.cpp
  protocol_smoke.py        行协议冒烟
  visit_budget_smoke.py    四种引擎都必须正好用完 visits（评分的前提）
```

## 三条不能破的规则

1. **评估预算只能按 visits。** `EngineOptions::visits`（`SearchRequest::visits`）是唯一预算；
   剪枝引擎把它解释成**节点数**（`SearchOptions::nodeLimit`，完全不看墙钟），
   MCTS 解释成模拟次数。默认值就是 **512**，因为 0 会被前端当成"按墙钟"，
   而 analyze 模式的墙钟是 1 小时——访问数能涨到几千万。要回到墙钟必须显式 `set_visits 0`。
   回归测试：`tests/visit_budget_smoke.py`。
2. **对称性：等价着法只搜一次。** 棋盘在 8 个等距变换下不变，两套规则都只看"本身 + 四邻"，
   所以 `Position::symmetryMask()` 非恒等时，展开只给每个轨道留一个可搜索代表
   （先验最高的那个），其余边 `searchable=0` 并把 `representative` 指向代表。
   报告时等价格子用代表边的数字，`rootPolicy()` 让它们共享访问数。
   节点自带对称群掩码，子节点是 `父群 ∩ 固定该着法的子群`，不重复扫棋盘。
3. **规则要隔离。** 搜索骨架里没有 `if (ruleset == Blocking)`；阻挡规则只在
   `ai_blocking.cpp` + `position.cpp` 的阻挡分支 + `rules.h` 里出现。
   看走眼的最快办法：`grep -rn "Blocking" src/`，应该只命中 position.cpp 和 ai_blocking.cpp。
4. **格式变化要三处同时改**：写出端（`nn.cpp` / `experimental/training/train_resnet.py`）、
   读入端（`nn.cpp` / 数据读取）、测试。改完**清空 CUDA 构建目录**——旧的 `.o` 会按旧布局
   读文件，而 `--check` 依然"通过"（踩过三次）。

## 构建与测试

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # 在仓库根
cmake --build build -j 8
cd build && ctest --output-on-failure
```

只跑引擎相关的测试：`ctest -R "shuyi_tests|protocol|visit_budget"`。

## 行协议（给 UI 和 arena 用的最小集合）

```
boardsize 9
ruleset classic|blocking        # 只在第一步之前
set_engine random|pruning|mcts|mcts_nn
set_model <path>                # 仅 mcts_nn
set_visits N                    # 固定预算；N>0 时忽略 set_time（analyze 模式也遵守）
set_time MS                     # 仅交互用（不用于评测）
set_a <komi>
play red|black <vertex> <n> | play <side> pass
genmove red|black               # 先回 "=" 应答，再回 "= <vertex> <n>" 或 "= pass"
analyze red|black               # 异步 info 行
capabilities                    # engine=... cuda=0|1 proto=1 visits=1 rulesets=...
```

`info` 行的 MCTS 部分：`visits=` `nodes=` `hasscore=` `proven=` `score=` `winrate=`
`best=` `moves=` `cells=`。两个字段容易误读：

1. `hasscore=1` 才表示 `score=` 是真的：它只能来自**已解算的终局**（或模型的分数头）。
   没有分数头又没搜到终局时是 0，界面显示 `--`。不要再用胜率线性映射出目差，
   空盘会算出 −15 目。
2. `proven=1` 表示这个局面**已被求解**：`winrate`/`score`/`best` 是规则的硬结果，
   不是估计；此时 `visits` 会小于预算（搜到答案就停）。`cells=` 的第 6 个字段
   （`cell:n:winrate:visits:avgscore:proven`）是逐点的证明标志，界面据此用精确值
   而不是置信下界排序——1 次访问的已解算必胜点不该被 40 次访问的 55% 压下去。

终局处理（`mcts.cpp` 的 `updateProof`）：终局叶子按规则 proven；一手让对手 proven 输
就是本方的 proven 胜；只有**未被宽度剪枝**（`complete_`）且全部着法 proven 才能判负/和。
目差只在精确时公布：根全解 → 极小极大目差；根解但目差未定 → 引擎实际会走的那手的精确
目差；否则 `--`。

完整协议见 [../docs/ENGINE_PROTOCOL.md](../docs/ENGINE_PROTOCOL.md)，
实现说明见 [../docs/ENGINE_ARCHITECTURE.zh-CN.md](../docs/ENGINE_ARCHITECTURE.zh-CN.md)。
