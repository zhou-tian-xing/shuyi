# 补对局数据：一次上机跑什么

目标：**每一对 500 局**，让每一档的分差区间收到 ±30 ELO 左右（500 局在 50% 附近是
±4.4%）。梯子 16 档 = 15 场，另加最弱档对随机一场。每个预算 15 × 500 = 7500 局。

## 先看成本（租机器前跑这条，不占用 GPU）

```bash
BUDGETS="64 128 256" GAMES=500 bash training/eval_session.sh --plan
```

实测吞吐（V100，单进程，8 核全占，梯子里混着 b4c32/b6c96）：

| 预算 | 15 场 × 500 局 | 估计用时 |
| ---: | ---: | ---: |
| 64 | 7500 | ~21 min |
| 128 | 7500 | ~41 min |
| 256 | 7500 | ~96 min |
| 512 | 7500 | ~156 min |

所以一小时只能装下 64 + 128（加编译传输约 10 分钟，正好一小时）；两小时能到 128 跑完
再开 256 的一半；**256 以上建议直接租 3 小时**。

## 上机步骤

```bash
# 1) 传代码（不含 runs/）与 16 个模型
export MSYS_NO_PATHCONV=1 SHUYI_PASS=...
python training/remote_deploy.py --host HOST --port PORT --user root \
    --remote-dir /root/shuyi --password-env SHUYI_PASS
#    模型不在仓库里，要单独传（每个 ~0.3–4 MB）：
python training/ssh.py put models/*.bin /root/shuyi/models/   # 见下方说明

# 2) 编译（CUDA）
python training/ssh.py run "cd /root/shuyi && export PATH=/usr/local/cuda-12.4/bin:\$PATH && \
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DSHUYI_BUILD_TOOLS=ON -DSHUYI_CUDA=ON && cmake --build build -j8"

# 3) 跑（nohup 挂后台，日志在 logs/）
python training/ssh.py run "cd /root/shuyi && BUDGETS='64 128' GAMES=500 \
  nohup bash training/eval_session.sh > /root/session.log 2>&1 &"

# 4) 拉回来、填表、算榜
python training/pull.py --remote /root/shuyi/runs/eval/ladder128 --local runs/eval/ladder128
python training/leaderboard_import.py
python training/leaderboard.py --write-ladder runs/eval/ladder_order.txt
```

模型上传要逐个点名：`python training/ssh.py put` 只接受成对的「本地 远端」，例如

```bash
python training/ssh.py put \
  runs/training/b6c96-nopass-20261124/models/b6c96-s14680-d198656.bin \
  /root/shuyi/models/b6c96-s14680-d198656.bin
```

（一次可以给多对。`models/` 目录里只有随包的两个模型，其余在各 run 目录下。）

## 这一轮要盯的三件事

1. **`b6c96-s14680-d198656` 还没被评过**，它排在梯子里 `b6c96-s13520` 旁边；跑完看它
   挪到哪一档。如果它和相邻档打出一边倒，说明当初的排序位置错了，重排后补那一场。
2. **每一场都要能读出分差**：跑完先看 `runs/eval/ladder<B>/results.csv`，任何一个
   500-0 都说明这一档挨得太远，需要把中间那一档插进来，而不是硬算。
3. **最弱档对随机**：16 档时它是第 16 场。40 局 35-5 只给点估计 ±；500 局会把这一格
   压到 ±2%，整张表的绝对分才站得住。

## 不要做的事

- 不要跑全量循环赛：105 对里有 44 对是一边倒，那些格子只含「强很多」，白花时间。
- 不要给每个模型都挂随机：13 个模型对随机都是 40-0，那 13 场没有任何信息。
- 不要为了省时间把 500 局改成 100 局：±9.8% 折成 ELO 是 ±70，仍然读不出「换预算
  值不值」，等于没测。
