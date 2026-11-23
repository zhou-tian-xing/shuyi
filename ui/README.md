# ui/ — Win32 前端（一个独立产品）

原生 GDI 窗口，**不依赖浏览器，也不含任何搜索**。界面是规则与状态的权威：它维护棋局、
校验着法、绘制棋盘；一切计算都发给 `shuyi_engine.exe` 子进程，通过行协议（stdin/stdout）
接收 100 ms 一次的分析流。

```
include/engine_client.h    子进程 + 管道 + 读线程 + 同步握手 command()
include/engine_view.h      分析视图 + parseAnalysis()（JSON → 视图）
include/json.h             极简 JSON 解析器
include/discovery.h        引擎候选路径 + capabilities 探测
src/ui_win32.cpp           窗口、菜单、绘制、规则状态、分析状态机
src/engine_client.cpp      进程/管道/读线程
src/engine_protocol.cpp    解析 info JSON、着法 token
src/json.cpp               JSON 解析
src/discovery.cpp          引擎发现
tests/ui_menu_smoke.py     无界面检查模型菜单（ctest: shuyi_ui_menu）
tests/engine_choice_smoke.py 无界面检查“用户选的引擎优先”（ctest: shuyi_engine_choice）
```

## 构建（不需要引擎源码）

```bash
cmake -S ui -B build-ui -DCMAKE_BUILD_TYPE=Release
cmake --build build-ui
```

界面只链接 `../core`（规则）。它**不认识** `shuyi_search`：CMake 里没有这个依赖，
加回去就等于把引擎塞进界面包。

## 引擎从哪来

界面只是启动一个子进程，所以重要的是那个程序说不说协议，而不是它是谁编译的。顺序：

1. `SHUYI_ENGINE` 环境变量；
2. `shuyi-ui.ini` 里的 `engine=`（相对路径按本 exe 所在目录解析）；
3. 自动探测：exe 同目录 → `../build-cuda` → `../build` → 当前目录，取第一个
   `capabilities` 回 `proto=2` 的程序；多个可用时优先 `cuda=1`。

探测到之后用 `capabilities` 做**硬校验**：`proto != 2` 直接拒绝并弹窗说明，不做“可能兼容”。
启动时每条命令都同步等回复（`EngineClient::command`），任何 `?` 都会中止启动并弹窗，
不会带着半个配置继续跑。

## 三条硬规矩

1. **不计算**：没有引擎就提示去“引擎 → 选择引擎程序…”，绝不自己走一步。
2. **不猜**：引擎报错、分析 JSON 解析失败，都显示错误，不拿旧数据或空白冒充结果。
3. **口径一致**：棋盘上的标记、右侧摘要、悬停面板都用“已解算则精确值，否则 Wilson 95% 下界”；
   目差只在引擎确实有分数时显示，没有就 `--`。

## 快捷键与设置

* **空格**：开始分析 / 停止分析（不清屏）。达到访问上限后自动停下，再按空格不计上限、
  继续同一棵树；F3 是别名。
* **引擎 → 设置…**：分析访问上限、落子访问上限、A值（存 `shuyi-ui.ini`，并下发引擎）。
* **引擎 → 引擎信息…**：当前引擎程序、协议版本、后端（CUDA/CPU）、模型。
* `--dump-engine` / `--dump-models` 是测试用的无界面探针（`ui/tests/` 使用）。
