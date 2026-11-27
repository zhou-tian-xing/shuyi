# 数弈 1.0

9×9 的数字落子棋，第一个正式版本。两个互不链接的程序：**原生 Win32 界面**（管规则、状态、显示）
和**无头 NN-MCTS 引擎**（管搜索、网络、计算），两者只通过一条文本行协议通信。

## 下载哪个

| 你的机器 | 下载 | 里面有什么 |
| --- | --- | --- |
| 任何 x64 Windows | **`shuyi-1.0-ui-cpu.zip`** | 界面 + CPU 引擎 + 最强模型（4.7 MB） |
| 有 NVIDIA 显卡 | `shuyi-1.0-ui-cuda.zip` | 同上，引擎换成 CUDA 版本（需要 CUDA 12 运行时，见下） |
| 只要引擎 | `shuyi-1.0-engine-cpu.zip` / `-engine-cuda.zip` | 只有 `shuyi_engine.exe` + 协议文档 |

**解压 → 双击 `shuyi.exe` → 按空格分析，或直接落子。** 包里自带权重，不用另外下载。

> **CUDA 包需要先装 CUDA 12 运行时。** NVIDIA 驱动本身**不包含**它，缺了会表现为
> "双击没反应"（Windows 报缺少 DLL）。需要这三个文件：
> `cudart64_12.dll`、`cublas64_12.dll`、`cublasLt64_12.dll`——装 CUDA Toolkit 12.x 即可，
> 或把它们放进 `shuyi.exe` 所在目录。CPU 包不需要任何额外组件；不确定就用 CPU 包。

## 怎么玩

- 点棋盘落子；空格让引擎分析当前局面，再按一次停止。
- 左侧是分析的产物（行棋方、胜率、目差、选点、当前分值、走势折线）；右侧是棋谱树：
  ↑/↓ 回溯主干，点节点回到那一手，Delete 删掉这一手和它的后续。
- 新局 / 保存 / 载入在左上角；棋谱存成文本，每一手写作 `a4-2`。
- 引擎菜单 → 设置… 可改分析/落子访问上限和 A 值（默认 A=+0.5）。
- 引擎菜单 → MCTS + 网络 选模型；把任意 `.bin` 丢进 `models/` 就会出现在菜单里。
- 数字怎么看：点上的数字是推荐的棋子数字，颜色是胜率（已解算用精确值，否则用 Wilson 95%
  置信下界）；已解算的点直接显示目差（+3 / −12），因为胜率只有 0% 和 100%，分不出输 2 目和输 18 目。

规则全文在包里的 `RULES.zh-CN.md`，协议在 `ENGINE_PROTOCOL.md`。

## 这个版本有什么

- **三套规则**：`classic`、`blocking`（阻挡）、`nopass`（默认，无过手，一方无子可落即终局）。
- **三种引擎**：`random`、`mcts`、`mcts_nn`（网络引导），全部按 **visits** 而不是
  墙钟计时——同一个数字在笔记本和 GPU 上含义相同。
- **CUDA 后端**：同一套界面，同样的访问量快一个数量级（本机构建在 RTX 3080 上验证；
  `capabilities` 会报 `cuda=1`，分析流里 `backend` 显示 `cuda`）。
- **A 值作为输入**：网络把 A 当输入特征，换 A 不需要换模型。
- **分数头**：已解算的局面直接给目差，而不是把 100% 胜率摊成无数种赢法。
- **引擎可替换**：界面只认协议，任何实现了 `ENGINE_PROTOCOL.md` 的程序都能插进来。

## 最强权重

`b6c96-s305983-d3226791.bin` — 1.03 M 参数，4.1 MB，nopass 规则。

| 项 | 值 |
| --- | --- |
| 64 visits 榜 ELO | **+2708**（`random = 0`，44 个模型、19,000 局的同一张表） |
| sha256 | `ec29256d9608f0d5f59e7cb3be68584b8575b226115646acc10ca9fd3b4a63df` |

同一份文件也在仓库的 [`models/`](https://github.com/zhou-tian-xing/shuyi/tree/main/models) 目录里，
命名、兼容性与其余权重的说明见 [models/README.md](https://github.com/zhou-tian-xing/shuyi/blob/main/models/README.md)。
ELO 的口径（64 visits、锚在 random、不同榜之间不可比）见
[ELO 方法](https://github.com/zhou-tian-xing/shuyi/blob/main/docs/ELO_METHODOLOGY.zh-CN.md)。

## 已知限制

- **只有 Windows x64**：界面是 Win32/GDI，没有别的移植。
- **CUDA 包需要 NVIDIA 驱动 + CUDA 12 运行时**（`cudart64_12.dll`、`cublas64_12.dll`、
  `cublasLt64_12.dll`），且构建时的 GPU 架构必须在目标机上能跑。缺任何一项请用 CPU 包——
  它不需要任何额外组件，这也是它是默认推荐的原因。
- **ELO 只在同一张表内可比**：换参赛池、换 visits 都会让绝对分平移。
- **权重是数据**：模型比程序更新得快，所以它单独发布、单独下载，不进安装包逻辑。
- 训练侧的变慢还没有归因完（见
  [缩放分析](https://github.com/zhou-tian-xing/shuyi/blob/main/docs/reports/SCALING_ANALYSIS.zh-CN.md)）；
  这不影响 1.0 的可玩性，但影响下一次训练该改什么。

## 从源码构建

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSHUYI_BUILD_UI=ON
cmake --build build -j 8
ctest --test-dir build
```

CUDA：加 `-DSHUYI_CUDA=ON`（Linux 见 `training/setup_ubuntu.sh`，Windows 需要先跑 `vcvars64`）。

## 校验

每个包里都有 `BUILD.txt`，列出构建 commit、时间以及包内每个文件的 sha256。
源码包对应本 tag，也可以直接用 GitHub 自动生成的 Source code 链接。
