# runs/ — 训练与评测的全部输出

这个目录**不入库**（见 `.gitignore`）：里面是二进制权重、自对弈数据、日志和评测
结果，都可以由 `training/` 下的脚本重新生成。人写的说明在 `docs/`。

## 命名约定

1. **一个 run 一个目录**，目录名固定三段：

   ```
   <arch>-<ruleset>-<YYYYMMDD>        例：b4c32-nopass-20261124
   ```

   架构和规则集说明这个 run **是什么**，日期把两次同形状的 run 分开。别的东西
   不进名字——尤其不要 `run_20261122_182132` 这种时间戳、也不要 `blk1`、`smoke4`
   这种只有当时看得懂的代号。

2. **模型文件一律**：

   ```
   b<blocks>c<channels>-s<optimizer steps>-d<training samples>.bin
   例：b4c32-s34760-d470705.bin
   ```

   规则集**不进文件名**（它是模型头第 33 字节）。一代只写一个文件：
   2026-11-24 起 `pipeline.py` 把训练出的 `candidate.bin` 直接 **move** 成上面的
   名字，不再留 `gen_XXX/candidate.bin` 这种第二种写法。唯一的例外是发布副本
   （见第 4 条）。

3. **数据分片**保持 `gen_XXX/selfplay/shard_XXX.bin`：它属于哪一代写在路径里，
   再在名字里重复一遍没有意义。

4. **同一份权重只出现一次**：`models/` 里的发布副本和 `runs/training/<run>/models/`
   里产出它的副本字节相同，界面菜单按内容去重、只显示路径更浅的那个
   （`models/b4c32-s42012-d595499.bin` 与那个 run 的 `b4c32-s7252-d202770.bin`
   就是这样一对：前者是累计口径，后者是 run 内口径）。

## 布局

```
runs/
├── training/                       真正用来出模型的 run
│   ├── b4c32-blocking-20261122/    blk1：阻挡规则 6 代，b4c32 第一代基线
│   └── b4c32-blocking-20261123/    acon：A 进网络（v4）+ 热启动，4 代
├── smoke/                          pipeline 冒烟测试，几秒一轮，不产出可用模型
│   ├── acond-20261123/             A 条件化管线（dry/dry2/dry3/e2e）
│   ├── b2c16-20261122/             小模型全流程（3 代 × 8 局）
│   └── b2c16-oneply-20261122/      单步价值自对弈探针
├── probes/                         一次性测量输出，纯记录，不参与训练
│   ├── arena/                      横向对局矩阵（blocking / blocking512 / probe）
│   ├── avalue/                     A 值标定（komi 扫描）
│   ├── branching/                  分支因子测量
│   └── elo/                        早期 elo 合并结果
└── archive/                        已结束的历史 run，不再追加
    ├── b2c16-classic-20261122/     onehour：16×2，15 代
    ├── b10c128-classic-20261122/   b10c128 三小时（结论：容量不是瓶颈）
    ├── b4c32-classic-20261122/     srv1：b4c32 经典规则，7 代
    ├── b6c64-classic-20261122/     v3fix：目标里加位置信息后的修复 run
    └── …
```

## 训练 run 一览

| 目录 | 规则 | 时间 | 代 | 模型 | 结论 / 备注 |
| --- | --- | --- | ---: | --- | --- |
| `training/b4c32-blocking-20261122` | blocking | 11-22 23:30 → 11-23 00:49 | 6 | `b4c32-s5960…` … `b4c32-s34760-d470705` | 第一次 b4c32 链式训练；`s34760` 是随后 3 小时 run 的热启动模型。manifest 只记录了 5 代（第 6 代训练完、模型已产出，进程在写 manifest 前被停掉）。 |
| `training/b4c32-blocking-20261123` | blocking | 11-23 23:00 → 11-24 01:13 | 4 | `b4c32-s7252-d202770`（累计口径 = `models/b4c32-s42012-d595499`） | A 进网络（`SHUYINN4`）第一次真训练：热启动 5e-4，gen0 对 `s34760` 0.675。ELO 池内 −140/−91/+112/+119，即没有稳定爬升。 |

`runs/training/*/elo/` 是每一代的池内 ELO；`gen_XXX/arena/match.json` 是与上一代的
对局记录；`manifest.json` 是唯一的权威索引（--resume 依赖它）。

## 归档 run 一览

| 目录 | 当时名字 | 规则 | 模型 | 为什么在这 |
| --- | --- | --- | --- | --- |
| `archive/b2c16-classic-20261122` | onehour | classic | `models/gen_000.bin`（无 s/d 可考） | 2026-11-22 01:42 的一小时 run，15 代。当时代码不记录 steps/data，所以名字无法规范化，保留原文件名并在此说明。 |
| `archive/b10c128-classic-20261122` | b10c128_3h | classic | `b10c128-s105750-d2340000` | 10 块 128 通道，3 小时；蒸馏固定 teacher，结论是“目标不可学”而非容量不足。 |
| `archive/b4c32-classic-20261122` | srv1 | classic | `b4c32-s7960…` `s15600…` `s23240…` | 7 代，但 gen2–gen6 的权重字节完全相同（同数据同种子重复训练），只留三份不同的。 |
| `archive/b6c64-classic-20261122` | v3fix | classic | `b6c64-s1120-d15600` | 价值目标加入位置信息后的验证 run。 |

## 2026-11-24 整理记录

**做了什么**

- 旧布局 `runs/<ruleset>/<name>/`（blocking/blk1、classic/ 空目录）改成
  `runs/training/<arch>-<ruleset>-<date>/`，`pipeline.py` 与
  `run_generation.sh` 的默认 run 目录同步改成这个形状。
- 从服务器拉回来时按 `gen_000.bin` 落地的四个模型，改成 manifest 里的规范名
  `b4c32-s2016-d77976 / s5096-d119573 / s6174-d161173 / s7252-d202770`。
- `docs/reports/` 里的 run 数据（`b10c128_run/`、`onehour_run/`、`srv1_run/`、
  `v3fix_run/`、`elo_results.csv`）搬到 `runs/archive/`；`docs/reports/` 现在只
  放人写的报告。顺带发现并合并了一份 16 MB 的重复副本（`runs/archive/docs_reports/`）。
- `models/` 只留当前发的两个模型 + 两个测试夹具
  （`shuyi_resnet_v1/v2.bin`：`SHUYINN2` + 19 平面，`engine/tests` 和
  `protocol/tests` 直接用它们当 A-blind / classic 的样本，不能删）。

**删掉了什么（都是可证明的重复或坏文件）**

| 删除 | 理由 |
| --- | --- |
| 14 个 `candidate.bin` | 与 `models/b4c32-s*.bin` / `b2c16-s*.bin` 字节相同（md5 逐一核对） |
| `smoke4/models/gen_00X.bin`、`acon/gen_00X.bin`（重命名前的副本） | 同上，规范名已存在 |
| `srv1/models/gen_003…006.bin` | 与 `b4c32-s23240-d316000` 字节相同 |
| `runs/archive/docs_reports/`（16 MB） | 与 `docs/reports` + 模型重复，内容已全部合并 |
| `acon_server/artifacts.tgz` | 里面的每个文件都已在同目录解包 |
| `acon/a.bin b.bin c.bin init.bin`、`a_probe.py` | 一次性 A 条件化探针的权重，没有任何记录说明它们是什么；能力已由 `training/avalue.py` 覆盖 |
| `smoke2.log`、`runs/classic/`、`gens.json.bak` | 对应数据目录已不存在 / 空目录 / 备份文件 |
| `archive/b2c16-classic-20261122/models/gen_000.bin` | 与 `models/shuyi_resnet_v2.bin` 字节相同 |
| `runs/smoke/b2c16-20261122/models/*.bin`、`runs/smoke/…/probe.bin` | v2 + 37 平面，引擎按“退役布局”拒绝加载，留着只会在模型菜单里给出必然失败的选项 |

**留下的坑（明确记下来，不假装没有）**

- `archive/b2c16-classic-20261122` 里的 16×2 权重无法规范化命名：当时的 manifest
  不记录 steps/data。
- `training/b4c32-blocking-20261122/manifest.json` 少最后一代的记录（模型文件在，
  记录不在）；用 `--resume` 会重跑那一代。
