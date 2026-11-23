# ui/ — Win32 前端

原生 GDI 窗口，**不依赖浏览器**。UI 自己不算棋：它启动 `shuyi_engine.exe` 子进程，
通过行协议（stdin/stdout）驱动，异步读取 `info` 行。

```
include/engine_client.h   子进程 + 管道客户端：start / send / drain / stop
src/engine_client.cpp     同上实现（reader 线程 → 队列，UI 100 ms 定时器消费）
src/ui_win32.cpp          整个窗口：绘制、面板、菜单、AI 标记、规则与 A值
tests/ui_menu_smoke.py    无界面检查模型二级菜单（ctest: shuyi_ui_menu）
```

## 怎么找东西（`ui_win32.cpp` 一个文件装完）

| 想改什么 | 函数 |
|---|---|
| 棋盘格线、坐标、棋子（方形圆角） | `drawBoard`、`drawStone` |
| 最后一手金框、分析标记 | `drawAnalysisMarks`（细边 + 按棋盘色混合的染色） |
| 右侧面板三栏（胜率 / 访问数 / 预期平均分） | `drawPanel`、`formatVisits`（在 engine/format.h） |
| 菜单、快捷键 | `createMenuBar`、`WM_COMMAND` |
| 模型二级菜单 | `scanModels`、`populateModelMenu`、`selectModel`、`populateModelMenu` |
| 规则切换 / A值 | `setRuleset`、`setAValue`、`syncRulesetChecks` |
| 引擎进程与协议 | `startEngineProcess`、`sendEngine`、`handleEngineLine`、`syncEnginePosition` |

## 引擎进程：UI 自己找最快的那个

界面和引擎是两个进程，只靠行协议说话，所以**界面是哪份构建无所谓**——重要的是它拉起来的
那个 `shuyi_engine.exe` 是不是快的。启动时按下面的顺序找引擎，并用 `capabilities` 探测：

```
SHUYI_ENGINE                       # 显式指定，优先
<exe 目录>/shuyi_engine.exe
<exe 目录>/../build-cuda/shuyi_engine.exe   # 以及 /Release
<exe 目录>/../build/shuyi_engine.exe       # 以及 /Release
<当前目录>/{,build-cuda,build}/shuyi_engine.exe
```

探测用协议命令 `capabilities`，引擎回答 `engine=shuyi_engine cuda=1|0 rulesets=...`：

- **有任何一个候选回答 `cuda=1`，就用它**（这就是"引擎得是快的"）；
- 否则用第一个能正常回答的（CPU）；
- 启动后再发 `set_backend auto`，`info` 行回报 `backend=cuda|cpu`，窗口标题显示 `(GPU)`/`(CPU)`。

所以：**MinGW 编的 CPU 版界面 + MSVC/CUDA 编的引擎** 是正常组合，不需要"用 CUDA 版界面"。
把 CUDA 构建放在 `<repo>/build-cuda`（本机是指向 `C:\shuyi_cuda_build` 的 junction），
界面就会自动挑到它；也可以 `set SHUYI_ENGINE=...` 指定任意引擎（包括远程挂载的）。

`shuyi.exe --dump-engine out.txt` 会把"试了哪些路径、每个候选回答什么、最后选了谁"
写出来，不需要开窗口就能诊断：

```
try 1 F:\...\build-cuda\shuyi_engine.exe
cuda  F:\...\build-cuda\shuyi_engine.exe
cpu   F:\...\build\shuyi_engine.exe
chosen=F:\...\build-cuda\shuyi_engine.exe
```

回归测试：`ui/tests/engine_choice_smoke.py`（ctest: `shuyi_engine_choice`）。

### 实测（b4c32，512 visits 一步 genmove）

| 引擎 | 耗时 | 备注 |
|---|---:|---|
| CPU，旧卷积实现 | 18.6 s | 36 ms/次前向，已修 |
| CPU，现实现（MinGW） | 2.1 s | 4.2 ms/次 |
| CPU，现实现（MSVC + `/arch:AVX2`） | 4.0 s | 不开 AVX2 是 5.7~6.1 s |
| **CUDA（自动选中）** | **0.26~0.41 s** | 与 CPU 版本同一着法、同一节点数 |

MSVC 构建建议加 `-DSHUYI_NATIVE_SIMD=ON`（给 `/arch:AVX2`）；MinGW 上实测反而略慢，
所以这个开关默认是关的，按机器实测决定。

## 分析标记：最优点按"置信下界"排，不按裸胜率

棋盘上的标记（胜率 / 访问数 / 预期平均分）排序用 `winRateLowerBound(winrate, visits)`
（Wilson 95% 下界，在 `engine/include/shuyi/format.h`，有单测）：

- 搜了 1 次、胜率 100% 的格子 → 0.21；搜了 200 次、胜率 55% → 0.49；
- 所以 `best` = **下界最大的空格**，颜色也按下界的差着色（红→绿），访问数低于最优 5%
  的格子不画；
- 平局（差 < `kMarkTieMargin`）保留上一帧的最优格。

这样既"综合了搜索量和胜率"，也不会出现"两个格子来回闪"的现象——访问数只会增长，
排名因此单调稳定。

**例外：已解算的格子用精确胜率。** 引擎在 `cells=` 里给每个点带一个证明标志
（第 6 个字段）。`proven=1` 的点是规则的硬结果，1 次访问也是真的，所以它按精确胜率
排序、不受 5% 访问数门槛限制；同样是已解算且胜负相同，则比精确目差（输也要少输）。
状态栏会显示"已解算"，此时 `访问` 数小于预算属于**正常**（搜到答案就停了）。

**排名前 8 的点永远画出来**（`kMarkAlwaysShow`）。搜索收敛时一个点可能吃掉九成访问，
只按"5% 门槛"过滤会让棋盘上只剩一个标记——那看起来像引擎证明了唯一解，其实只是
把预算花在了一个点上。等价点（同一对称轨道）由引擎直接给出相同数字，所以棋盘的
对称位置会显示完全一样的胜率/访问数/目差。

界面对引擎只下一个预算：`kUiVisits = 512`（`set_visits`），从不按毫秒。
`capabilities` 里没有 `visits=1` 的老引擎才退回 `set_time 650`。

## 模型菜单：必须能在任何工作目录下找到所有模型

`引擎(&E) → MCTS + 网络` 子菜单会列出**扫描到的全部模型**，条目形如
`blk1/gen_005.bin  [阻挡]`（`[现行]`/`[阻挡]` 从模型文件头里的 ruleset 字节读，不靠文件名猜）。

扫描根目录依次是：**exe 所在目录**、**它的父目录（仓库根）**、**当前工作目录**，以及
`SHUYI_MODEL_DIRS`（分号分隔）里给的目录。每个根下**递归**找名为 `models` 的目录，
收里面的 `*.bin`（跳过 `build*`、`.*`、`__pycache__`）。

> 这条是修过的 bug：以前只按相对路径找 `models/` 和 `runs/*/models`，而从 `build/`
> 启动时工作目录是 `build/`，于是只显示 `build/models` 的两个夹具模型，训练出来的网络全都不见了。
> 回归测试 `ui/tests/ui_menu_smoke.py` 会检查菜单里同时出现两种规则的模型。

菜单底部有 `重新扫描模型目录`，跑完新的训练不用重启 UI。

## 路径编码（中文仓库）

仓库叫"数弈"，所以路径里带中文。Win32 的 `*A` API 返回的是 **ANSI**，
而引擎进程收到的参数要 **UTF-8**（`shuyi::Utf8Args` 在进程启动时转）。因此：

- `ModelEntry` 同时存 `path`（ANSI，给 Win32 / `fopen`）和 `enginePath`（UTF-8，给 `set_model`）；
- 写文件走 `shuyi::fopenUtf8`（见 `engine/include/shuyi/io.h`）；
- `wWinMain` 的命令行是 UTF-16，转 UTF-8 再解析（`--dump-models` 的路径就这么来的）。

## 无界面自测

```bash
build/shuyi.exe --dump-models out.txt   # 把菜单条目写出来，不需要开窗口
```

输出格式：第一行 `models=N`，之后每行 `<菜单 id>\t<标签>`，最后是 `重新扫描模型目录` 的 id。
`ui/tests/ui_menu_smoke.py` 就是靠它做回归的。

## 操作

数字键 1–9 选数字；点击棋盘落子；F2 新局；Ctrl+Z 悔棋；F3 分析；`[` `]` 调 A值。
完整说明见 [../docs/RULES.zh-CN.md](../docs/RULES.zh-CN.md)。
