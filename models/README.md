# 模型文件

把任意 `b<blocks>c<channels>-s<steps>-d<data>.bin` 放进这个目录（或安装包同名的 `models/`），
界面菜单「引擎 → MCTS + 网络」会列出它，不需要重启。

文件名就是来历：`b6c96` = 6 个残差块 × 96 通道，`s305983` = 累计优化步数，
`d3226791` = 累计自对弈局面数。同一架构的两份权重可以直接比这些数字。

## 从哪拿权重

- 安装包里带一份当时最强的权重；
- [Releases](https://github.com/zhou-tian-xing/shuyi/releases) 的资产里有历史权重和单独的最强模型。

仓库里只放了几份小文件和格式夹具（`b4c32` ×2、`shuyi_resnet_v1/v2`、外加当前最强的一份），
避免 clone 变慢。其余权重留在训练产物 `runs/training/<run>/models/` 里，不入库。

## 兼容性

- 规则集写在模型头里，不在文件名里。nopass 权重可用于 nopass，blocking 权重不能用于 nopass，
  反之亦然；选错时引擎会拒绝并说明原因。
- 架构由文件头决定（16 块 × 128 通道以内，输入平面 19/37，全局量 22/24）。
- CPU 与 CUDA 引擎都能读同一份文件。

ELO 的口径见 [docs/ELO_METHODOLOGY.zh-CN.md](../docs/ELO_METHODOLOGY.zh-CN.md)。
