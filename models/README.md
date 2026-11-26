# 模型文件

引擎和界面是两个程序，**权重是数据**：把任意 `b<blocks>c<channels>-s<steps>-d<data>.bin`
放进这个目录（或包里的 `models/`），界面菜单「引擎 → MCTS + 网络」就会列出它。

文件名就是它的来历：`b6c96` = 6 个残差块 × 96 通道；`s305983` = 累计优化步数；
`d3226791` = 累计自对弈局面数。名字是**累加**的，所以同一架构的两份权重可以直接比大小。

## 随仓库发布的权重

| 文件 | 规则集 | 64 visits ELO | 说明 |
| --- | --- | ---: | --- |
| `b6c96-s305983-d3226791.bin` | nopass | **+2708** | 目前最强。1.03 M 参数，4.1 MB |
| `b4c32-s42012-d595499.bin` | blocking | 见旧榜 | v4 包用的小网络，344 KB |
| `b4c32-s34760-d470705.bin` | blocking | 见旧榜 | v3 包用的小网络，344 KB |
| `shuyi_resnet_v1.bin` / `_v2.bin` | classic | — | 早期小网络，只作兼容/测试夹具 |

ELO 的含义和口径见 [docs/ELO_METHODOLOGY.zh-CN.md](../docs/ELO_METHODOLOGY.zh-CN.md)：
64 visits、`random = 0`、同一张 44 个模型的榜。**不同榜之间的绝对分不可比**。

`b6c96-s305983-d3226791.bin` 的 sha256：

```
ec29256d9608f0d5f59e7cb3be68584b8575b226115646acc10ca9fd3b4a63df
```

## 训练过程中产生的其余权重

`runs/training/<run>/models/` 下有几十份中间代（Git 忽略，不入库），它们是同一张榜上的
其他点。想看全部点及其数据量，运行 `python training/plot_elo_scaling.py`，
图落在 `docs/figures/elo_vs_data.png`。

## 兼容性

* 规则集必须匹配：nopass 权重可用于 nopass；blocking 权重不能用于 nopass（反之不行）。
  引擎会拒绝不匹配的组合并在状态栏说明原因。
* 架构必须匹配引擎支持的形状（16 块 × 128 通道以内）；输入平面 19/37、全局量 22/24 由文件头决定。
* 换模型不需要重启界面：菜单里选另一份即可。
