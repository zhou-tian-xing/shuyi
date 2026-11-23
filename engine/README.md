# engine/ — 无头引擎（一个独立产品）

这个目录就是引擎产品：搜索、网络、行协议进程。**它不认识 UI，也不认识训练脚本。**
它把规则当作依赖（`../core`），除此之外没有任何内部依赖。

```
../core/include/shuyi/
  rules.h         规则枚举 Classic / Blocking、规则名与模型后缀、过手哨兵
  position.h      棋盘状态：落子、撤销、合法着法、终局、哈希、对称
  format.h        显示格式（访问数、胜率、Wilson 下界、平均分）
  io.h            UTF-8 路径与 argv

include/shuyi/
  engine.h        Engine 基类、EngineOptions、SearchRequest（visits/interval）、Analysis
  ai.h            剪枝搜索：SearchOptions（nodeLimit）、SearchResult、searchAlphaBeta
  ai_policy.h     "评估 + 排序"策略接口；规则特异的实现由 ai_blocking.cpp 提供
  nn.h            策略价值网络：SHUYINN3/SHUYINN4 文件格式、特征平面、CPU 前向、分数头
  nn_cuda.h       CUDA 批量前向接口
  tt.h            置换表
  protocol.h      行协议会话：命令解析、位置状态、单搜索线程

src/
  ai.cpp          迭代加深 + aspiration window + PVS；nodeLimit 在这里生效
  ai_blocking.cpp 阻挡规则的评估与排序——规则特异代码只放这里
  mcts.cpp        PUCT 树搜索：强制过手、证明回传、对称轨道、访问预算
  engine.cpp      四种 Engine 实现（random / pruning / mcts / mcts_nn）
  nn.cpp          模型读写 + CPU 前向
  nn_cuda.cu      im2col + cuBLAS 的 CUDA 前向（含分数头）
  tt.cpp          置换表
  protocol.cpp    命令集、严格错误、启动握手、bench / selfplay
  protocol_json.cpp  Analysis → JSON
  main.cpp        stdin 循环
tests/            规则/搜索的 C++ 测试
```

## 构建（不需要界面）

```bash
cmake -S engine -B build-engine -DCMAKE_BUILD_TYPE=Release
cmake --build build-engine
# CUDA 12.4：
cmake -S engine -B build-engine-cuda -DSHUYI_CUDA=ON -DCUDAToolkit_ROOT=<cuda>
```

产物是一个 `shuyi_engine.exe`。把它和任意前端放在一起、或在界面里指定路径，就得到完整功能。

## 三条不能破的规则

1. **评估预算只能按 visits。** `SearchRequest::visits` 是唯一预算；剪枝引擎把它解释成**节点数**
   （完全不看墙钟），MCTS 解释成模拟次数。默认 512。`visits 0` 表示分析不限量（直到 `stop`），
   落子则退回墙钟——因为落子必须回来。
2. **没有隐藏的停止条件。** 搜索只会因为预算、墙钟或 `stop` 结束。**不要**加树大小上限：
   曾经的上限在 3.5 万 visits 时把未终局叶子当终局，产生了假的 100% 胜率。
3. **无法执行就报错。** `cuda` 不可用、模型缺失、规则集不符、着法非法，一律 `?`；
   不允许静默降级成 CPU、随机走子或旧分析。`auto` 是显式选项，并且必须报告实际后端。

## 协议

[`../protocol/ENGINE_PROTOCOL.md`](../protocol/ENGINE_PROTOCOL.md) 是唯一契约：命令表、
`info` JSON 字段表、流式语义、错误规则、版本规则。改命令或字段时三处一起改：
`src/protocol.cpp`、`src/protocol_json.cpp`、协议文档；然后跑 `protocol/tests/`。
