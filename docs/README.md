# docs/ — 文档索引

维护者先看 [MAINTAINERS.zh-CN.md](MAINTAINERS.zh-CN.md)（文件地图：每个文件在哪、干什么）。

## 规则与数学

| 文档 | 内容 |
|---|---|
| [RULES.zh-CN.md](RULES.zh-CN.md) | 棋盘、落子、计分、A值、三套规则（现行/阻挡/无过手）的完整定义 |
| [A_VALUE_ANALYSIS.zh-CN.md](A_VALUE_ANALYSIS.zh-CN.md) | A值的平局可达性：`P ∈ [-73, 73]` 的证明与 147 个见证 |
| [plans/PLAN_BLOCK_RULE.zh-CN.md](plans/PLAN_BLOCK_RULE.zh-CN.md) | 阻挡规则的设计、实测数据、A值标定结果 |

## 引擎

| 文档 | 内容 |
|---|---|
| [../protocol/ENGINE_PROTOCOL.md](../protocol/ENGINE_PROTOCOL.md) | 协议 v2：命令、JSON 流式分析、错误与版本规则 |
| [ENGINE_ARCHITECTURE.zh-CN.md](ENGINE_ARCHITECTURE.zh-CN.md) | 数据结构、搜索流程、MCTS、网络与 CUDA 后端；§10 为 5000v 分支因子实测 |
| [ELO_METHODOLOGY.zh-CN.md](ELO_METHODOLOGY.zh-CN.md) | 评测协议与 Elo 拟合（含"预算必须按 visits"的硬规则） |

## 训练与评测

训练手册在代码旁边：[../training/README.md](../training/README.md)
（原则、数据来源、闸门、横评、规则隔离、改进记录）。这里只放计划和历史报告。

| 文档 | 内容 |
|---|---|
| [plans/SERVER_PLAN.zh-CN.md](plans/SERVER_PLAN.zh-CN.md) | 服务器预案：预算模型、visits 取值、质量闸门 |
| [plans/NEXT_STEPS.zh-CN.md](plans/NEXT_STEPS.zh-CN.md) | 当前棋力提升计划、固定参照、停滞诊断 |
| [reports/RUN_20261125C.zh-CN.md](reports/RUN_20261125C.zh-CN.md) | 八小时 9 代：规模、逐代竞技场、64 visits 同表增量 +140 Elo、三个观察 |
| [TRAINING_SERVER.zh-CN.md](TRAINING_SERVER.zh-CN.md) | 服务器部署与运维 |
| [reports/](reports/) | 历次运行报告（srv1、one-hour、b10c128、v3fix、诊断、ELO 报表）+ 当次模型 |

## 历史运行报告

| 报告 | 是什么 |
|---|---|
| [reports/SRV1_RUN.zh-CN.md](reports/SRV1_RUN.zh-CN.md) | 3080 上 2h09m、7 代、b4c32（第一个有意义的长跑） |
| [reports/ONE_HOUR_RUN.zh-CN.md](reports/ONE_HOUR_RUN.zh-CN.md) | 1 小时 15 代（暴露了"20 局晋升门噪声太大"） |
| [reports/B10C128_3H_RUN.zh-CN.md](reports/B10C128_3H_RUN.zh-CN.md) | b10c128 容量实验 |
| [reports/V3FIX_RUN.zh-CN.md](reports/V3FIX_RUN.zh-CN.md) | 特征平面 v3 修复后的复跑 |
| [reports/DIAGNOSIS_2026-11-22.zh-CN.md](reports/DIAGNOSIS_2026-11-22.zh-CN.md) | 一次系统性诊断 |
| [reports/ELO_REPORT.md](reports/ELO_REPORT.md)、[reports/elo_results.csv](reports/elo_results.csv) | 早期循环赛的 Elo 报表 |

> 报告里的命令路径可能还停留在旧目录结构（`tools/`、`training/`），
> 现在的对应位置见 [MAINTAINERS.zh-CN.md](MAINTAINERS.zh-CN.md)。
