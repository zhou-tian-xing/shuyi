# 数弈 engine protocol — version 2

This file is the **only** contract between a front end and the engine. They are
separate programs, built separately, released separately, and may live in
different projects or be written in different languages. Nothing is linked.

```
front end (shuyi.exe)                engine (shuyi_engine.exe)
   owns the game state   -- commands -->   owns the computation
   draws the board       <-- info / =  --   keeps a search tree for one position
```

The engine is a headless service: it answers “what are the candidate moves and
what are they worth for this position”. It never owns a game record.

## 1. Transport

* One process, started once, kept alive for the whole session.
* Commands go to **stdin**, one per line, UTF-8, `\n` terminated.
* The engine writes to **stdout**, one message per line, flushed immediately.
* Blank lines and lines starting with `#` are ignored.
* Concurrent searches are not part of the protocol: sending a command that needs
  the engine stops the running search first.

## 2. Framing

Three kinds of line leave the engine:

| Line | Meaning |
|---|---|
| `= <text>` | the command succeeded (the final reply) |
| `? <text>` | the command failed; `<text>` says why |
| `info <json>` | an asynchronous analysis snapshot |

`analyze` starts a search and streams `info` lines at a fixed interval; its
**final** reply is `= done …` when the visit budget ran out, `= stopped` when a
`stop` ended it. Exactly one final reply is sent per command.

`genmove` also streams `info` while it thinks, and its final reply is the move:
`= <cell> <number>` or `= pass`.

### Optional command ids

A command may start with an integer id; every line belonging to it repeats the
id:

```
7 analyze red 512
7 info {"visits":64,...}
7 = done visits=512 proven=0 elapsed_ms=2103
```

Without an id nothing is prefixed (the simplest front end). With ids a front end
can pipeline: replies that arrive out of order stay attributable. A late
`= stopped` from an interrupted analysis is then unambiguous.

## 3. Commands

| Command | Reply | Notes |
|---|---|---|
| `protocol_version` | `2` | |
| `name` | `ShuyiEngine` | |
| `version` | e.g. `0.3.0` | |
| `list_commands` | space separated names | |
| `capabilities` | `engine=shuyi_engine proto=2 cuda=0\|1 backend=cpu\|cuda rulesets=classic,blocking,nopass visits=1 stream=1 json=1 model=0\|1 a_input=0\|1` | probe before choosing a binary |
| `boardsize <n>` | empty | only 9 is accepted; anything else is an error |
| `ruleset [classic\|blocking\|nopass]` | the active rule set | only before the first move; `nopass` is the product default |
| `clear_board` | empty | resets the position and the tree |
| `play <red\|black> <a1..i9> <1..9>` | empty | the front end owns the game; an illegal move is an error |
| `play <red\|black> pass` | empty | only legal when the side has no placement |
| `genmove <red\|black> [visits]` | `a1 5` or `pass` | `visits` is that command's budget |
| `analyze <red\|black> [visits] [interval <ms>]` | `done …` / `stopped` | streams `info`; `visits 0` = no limit |
| `stop` | `stopped` | ends a running search; the tree is kept |
| `set_engine <random\|pruning\|mcts\|mcts_nn>` | the name | rebuilds the search, keeps the position |
| `set_backend <cpu\|cuda\|auto>` | the backend actually used | `cuda` fails on a CPU build |
| `set_model <path>` | the path, plus `(no A input: v3 model)` when the file predates A conditioning | validated and rule-set checked before use |
| `set_visits <n>` | `n` or `off` | default budget only; does not touch the tree |
| `set_time <ms>` | `ms` | clock budget, used only when no visit budget applies |
| `set_a <value>` | the value | score offset added to Red's raw score. With a v4 model it is an input to the network; with a v3 model it only shifts the terminal result (`a_input=0`) |
| `set_window <points>` | the value | aspiration window (alpha-beta) |
| `set_pvs <on\|off>` | `on`/`off` | principal-variation search (alpha-beta) |
| `bench [visits]` | `visits=… time_ms=… visits_per_second=…` | throughput probe |
| `selfplay <games> [visits]` | `games=… red=… black=… draw=…` | sanity check of a model |
| `quit` | empty, then exit | |

### Visit budgets

A visit is the unit that is comparable across machines:

* MCTS / MCTS+NN: one visit is one simulation (one network evaluation).
* alpha-beta: one visit is one node (one position evaluation).

`set_visits N` sets the default. `analyze`/`genmove` may carry their own budget.
`visits 0` means “no budget”: for an analysis the search runs until `stop`, for a
move the clock applies instead (a move must come back eventually). There is no
tree-size cap and no hidden stopping rule: a search ends on its budget, its
clock, or `stop`.

### Streaming

`analyze … interval 100` publishes a snapshot at most every 100 ms (default
100 ms, allowed 1–60000). The interval is a reporting rate, **not** a limit: a
slow machine streams fewer updates, it does not search less. Always one final
snapshot is sent, followed by the final reply.

Extending an analysis: the tree belongs to the position, so a second `analyze`
on the same position continues it. `set_visits` does not reset it; `play`,
`clear_board`, `ruleset`, `set_a`, `set_engine`, `set_backend` and `set_model`
rebuild it because the search itself changed.

## 4. The analysis object

`info` carries one JSON object per line. Every field is optional except
`engine`, `visits` and `best`; a front end must ignore fields it does not know.

```json
{
  "engine": "mcts_nn",
  "turn": "red",
  "backend": "cuda",
  "nn": "b4c32-s34760-d470705.bin",
  "a_input": 1,
  "visits": 512,
  "nodes": 984,
  "depth": 0,
  "proven": false,
  "winrate": 0.5312,
  "has_score": true,
  "score": 1.234,
  "best": "e5:3",
  "elapsed_ms": 412,
  "pv": ["e5:3", "d4:1"],
  "moves": [
    {"move": "e5:3", "winrate": 0.5312, "visits": 210, "score": 1.234}
  ],
  "cells": [
    {"cell": "e5", "number": 3, "winrate": 0.5312, "visits": 210, "score": 1.234,
     "has_score": true, "proven": false},
    {"cell": "a1", "number": 1, "winrate": 0.0, "visits": 7, "score": -12.0,
     "has_score": true, "proven": true}
  ]
}
```

| Field | Meaning |
|---|---|
| `engine` | `random`, `pruning`, `mcts` or `mcts_nn` |
| `turn` | side to move for this analysis (`red`/`black`); all values are from its point of view |
| `backend` | which forward actually ran: `cpu` or `cuda` |
| `nn` | network name, empty without one |
| `a_input` | `1` when the loaded network takes A as an input, `0` when it cannot see it: a front end that offers an A control must say which |
| `visits`, `nodes` | search effort |
| `proven` | the position is solved: win rate, score and best move are game-theoretic values |
| `winrate` | win probability for the side to move, `[0,1]` |
| `has_score` | `score` is real (terminal result or a trained score head) |
| `score` | expected final score, points, side-to-move perspective, includes the A value |
| `best` | `cell:number`, or `none` |
| `pv` | principal variation, compact move tokens |
| `moves` | candidate moves, best first |
| `cells` | one entry per board cell with data; the UI draws these |
| `cells[].proven` | this move's subtree is solved: `winrate` and `score` are exact |
| `cells[].has_score` | this cell's `score` is real. For a solved cell it is the **exact margin**; for an estimate it is an average over the visits that carried a score. `false` means "no score for this cell", which a front end must render as `--`, never as `0` |

A solved cell's `score` is the exact margin even when the network has no score
head: the line ends in a terminal value, so a move that loses by 12 reports
`-12`, not the sampled `0` that the same position used to report. A front end
that sorts solved moves by win rate alone cannot tell `-2` from `-12`; the
margin is the number to show.

A cell that is not in `cells` has no analysis. Nothing in this object is derived
by the front end: if `has_score` is false there is no score, and the front end
shows nothing rather than inventing one.

**Presence is the flag, `visits` is not.** The exact endgame solver answers a
position in the last few plies with a full minimax and runs no simulation at all,
so its analysis has `visits: 0` and every cell in it has `visits: 0` — while
carrying `proven: true` and an exact `score`. A front end that reads "no visits"
as "no data" throws the solved position away and leaves the board blank in exactly
the phase where the numbers are certain; the window did that until it was fixed.
Rank and draw a cell by `proven`/`has_score`, and treat a *missing* cell — not a
zero visit count — as "not analysed".

## 5. Errors and the no-fallback rule

Every command either does what it says or fails with `?`.

* A wrong board size, an unknown command, a bad argument: `?`.
* `set_backend cuda` on a CPU build: `?`. With `auto` the engine uses the GPU when
  it works and reports `cpu` in the reply and in every analysis object.
* `set_model` refuses a network trained for an incompatible rule set.  A
  blocking-trained model is accepted under `nopass` (every nopass game is a
  blocking game up to its earlier end) and the reply says so; the reverse is
  refused.  It also refuses a
  model file whose shape it cannot read (a v4 file in a v3-only engine is an
  error, not a silently truncated load).
* `genmove`/`analyze` refuse to run when the selected engine cannot run (for
  example `mcts_nn` without a model).
* If the search cannot produce a move, the reply is `?`, never a made-up move.

A front end may not substitute a local search, a heuristic move or an old
analysis for a failed command. It shows the error.

## 6. Versioning

The major version is in `capabilities` (`proto=2`) and `protocol_version`.
A front end that needs version 2 must refuse an engine that reports anything
else, with a message naming both versions. Additive JSON fields do not change the
version; a changed meaning, framing or command does.

## 7. Conformance tests

`protocol/tests/` contains a dependency-free client and two tests that only use
this document:

```
python protocol/tests/protocol_smoke.py  <engine.exe> [model.bin]
python protocol/tests/visit_budget_smoke.py <engine.exe> [model.bin]
python protocol/tests/a_response.py      <engine.exe> <model.bin>   # manual
```

They run as part of `ctest` and are the acceptance test for any engine — this one
or one written elsewhere. `a_response.py` is manual (it needs a trained v4
network): it asserts that a model reporting `a_input=1` moves its win rate with A,
and that a v3 model (`a_input=0`) does not.
