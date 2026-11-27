#!/usr/bin/env python3
"""Cross-line arena: every player, one rating list.

All players are processes speaking the same line protocol, so a CPU MCTS, a CPU
MCTS+NN and a GPU model sit in the same tournament:

    random                       shuyi_engine.exe, uniform random (the Elo anchor)
    mcts                         shuyi_engine.exe, CPU MCTS
    mcts_nn:<model>              shuyi_engine.exe, CPU MCTS + network
    gpu:<model>                  shuyi_selfplay_gpu.exe --mode serve (batched CUDA)

`random` is always included and elo.py pins it at 0, so every rating reads as
"points above uniform-random play" and stays comparable across runs.

The arena records the move list of each game and scores the final board with
training/rule_probe.py's scorer, which is the same formula both engines use; no
third-party result reporting is involved.

    python training/arena.py --games 16 --visits 128 \
        --players random mcts mcts_nn:models/shuyi_resnet_v2.bin \
                  gpu:docs/srv1_run/models/gen_002.bin
    python training/elo.py --report runs/probes/arena/classic.json --anchor random
"""
import argparse
import importlib.util
import itertools
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalconfig import EVAL_VISITS, KOMI_BY_RULESET  # noqa: E402

REPO = Path(__file__).resolve().parents[1]  # repo root (training/..)
DEFAULT_ENGINE = REPO / "build" / "shuyi_engine"
DEFAULT_GPU = REPO / "build" / "shuyi_selfplay_gpu"


def _load_rule_probe():
    spec = importlib.util.spec_from_file_location("rule_probe", str(REPO / "training" / "rule_probe.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RULE_PROBE = _load_rule_probe()


def resolve_binary(value):
    path = Path(value)
    if not path.exists() and __import__("os").name == "nt" and not path.suffix:
        candidate = Path(str(path) + ".exe")
        if candidate.exists():
            return candidate
    return path


class PlayerDead(RuntimeError):
    pass


class Process:
    def __init__(self, argv, name):
        self.name = name
        self.proc = subprocess.Popen(
            [str(a) for a in argv], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            bufsize=1,
        )

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_result(self, need_payload=False):
        """Reads one response.  Commands that carry no payload answer a bare `=`;
        `genmove` answers `=`, then `info ...` lines, then `= a1 5`, so it has to
        wait for a non-empty payload."""
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line or line.startswith("info"):
                continue
            if line.startswith("?"):
                return "?"
            if line.startswith("="):
                payload = line[1:].strip()
                if payload or not need_payload:
                    return payload

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=3)
        except Exception:  # noqa: BLE001 - never mask a game result
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass


class EnginePlayer(Process):
    def __init__(self, kind, model, binary, args):
        super().__init__([binary], f"engine:{kind}")
        for command in (f"ruleset {args.ruleset}", f"set_engine {kind}"):
            self.send(command)
            self.read_result()
        if kind == "mcts_nn":
            if not model:
                raise SystemExit("mcts_nn needs a model path: mcts_nn:<file>")
            self.send(f"set_model {model}")
            self.read_result()
        # Every player is budgeted in visits, never in milliseconds: a clock
        # budget turns a rating into a benchmark of the host.  One visit is one
        # tree simulation (one network evaluation or one rollout) -- see
        # training/evalconfig.py.
        for command in (f"set_visits {args.visits}", f"set_time {args.hang_guard_ms}",
                        f"set_a {args.komi}"):
            self.send(command)
            self.read_result()

    def genmove(self, side):
        self.send("genmove " + side)
        return self.read_result(need_payload=True)

    def play(self, side, move):
        self.send(f"play {side} {move}")
        return self.read_result()


class GpuPlayer(Process):
    def __init__(self, model, binary, args):
        super().__init__(
            [binary, "--mode", "serve", "--model", str(model), "--ruleset", args.ruleset,
             "--visits", str(args.visits), "--komi", str(args.komi),
             "--root-k", str(args.root_k)], f"gpu:{Path(model).stem}")
        # The serve mode greets with two non-protocol lines; consume them
        # explicitly (read_result would skip them and then block).
        self.proc.stdout.readline()  # device banner
        self.proc.stdout.readline()  # "ready ruleset=... visits=..."

    def genmove(self, side):
        self.send("genmove " + side)
        return self.read_result(need_payload=True)

    def play(self, side, move):
        self.send(f"play {side} {move}")
        return self.read_result()


def parse_spec(spec):
    if spec == "random":
        return ("engine", "random", None, "random")
    if spec.startswith("gpu:"):
        return ("gpu", None, spec[4:], f"gpu:{Path(spec[4:]).stem}")
    if spec.startswith("mcts_nn:"):
        return ("engine", "mcts_nn", spec[len("mcts_nn:"):], f"mcts_nn:{Path(spec[len('mcts_nn:'):]).stem}")
    return ("engine", spec, None, spec)


def make_player(spec, binaries, args):
    kind, engine_kind, model, _label = spec
    if kind == "gpu":
        return GpuPlayer(model, binaries["gpu"], args)
    return EnginePlayer(engine_kind, model, binaries["engine"], args)


def score_moves(moves):
    """Final raw score from the move list; moves are (side, payload)."""
    state = RULE_PROBE.State()
    for side, payload in moves:
        if payload in ("pass", "?"):
            state.ply += 1
            state.passes += 1
            continue
        vertex, number = payload.split()
        x = ord(vertex[0]) - ord("a")
        y = int(vertex[1:]) - 1
        cell = y * RULE_PROBE.SIZE + x
        if state.board[cell] != 0:
            raise ValueError(f"duplicate move {payload}")
        state.board[cell] = int(number) if state.side == 0 else -int(number)
        state.rem[state.side][int(number) - 1] -= 1
        state.ply += 1
        state.passes = 0
    return RULE_PROBE.raw_score(state.board), state.ply


def play_one(spec_a, spec_b, a_is_red, binaries, args):
    """Returns (a_wins, draws, a_losses) contributions and the move log."""
    a = make_player(spec_a, binaries, args)
    b = make_player(spec_b, binaries, args)
    moves = []
    try:
        done = False
        for _ in range(80):
            for side in ("red", "black"):
                mover = a if (side == "red") == a_is_red else b
                tag = "A" if mover is a else "B"
                payload = mover.genmove(side)
                if payload is None:
                    return None, moves, tag
                if payload == "?":
                    # "game is over" is the normal answer once 54 stones are down.
                    if not moves:
                        return None, moves, tag
                    done = True
                    break
                moves.append((side, payload))
                # Both players must see every move: each keeps its own board.
                for player in (a, b):
                    reply = player.play(side, payload)
                    if reply == "?":
                        return None, moves, ("A" if player is a else "B")
            if done:
                break
            if len(moves) >= 2 and all(m[1] == "pass" for m in moves[-2:]):
                break
        total, _ply = score_moves(moves)
        red_win = 1 if total + args.komi > 0 else (-1 if total + args.komi < 0 else 0)
        if red_win == 0:
            return 0, moves, None
        a_is_red_winner = (red_win > 0) == a_is_red
        return (1 if a_is_red_winner else -1), moves, None
    finally:
        a.close()
        b.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=16, help="games per pair")
    ap.add_argument("--visits", type=int, default=128)
    ap.add_argument("--hang-guard-ms", type=int, default=3600000,
                    help="only a liveness guard: a CPU player that has not answered "
                         "in this long forfeits. It must never bind before the visit "
                         "budget does, or the table becomes a benchmark of the host.")
    ap.add_argument("--time", type=int, default=0,
                    help="deprecated, ignored; kept so old command lines still parse")
    ap.add_argument("--komi", type=float)
    ap.add_argument("--ruleset", choices=["classic", "blocking", "nopass"], default="nopass")
    ap.add_argument("--root-k", type=int, default=32)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--engine-binary", default=str(DEFAULT_ENGINE))
    ap.add_argument("--gpu-binary", default=str(DEFAULT_GPU))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.komi is None:
        args.komi = KOMI_BY_RULESET[args.ruleset]
    binaries = {"engine": str(resolve_binary(args.engine_binary)),
                "gpu": str(resolve_binary(args.gpu_binary))}

    specs = [parse_spec(p) for p in args.players]
    if not any(s[1] == "random" for s in specs):
        print("note: adding the 'random' anchor (it defines Elo 0)", flush=True)
        specs.insert(0, parse_spec("random"))
    names = [s[3] for s in specs]
    pairs = list(itertools.combinations(range(len(specs)), 2))
    print(f"{len(specs)} players, {len(pairs)} pairs, {args.games} games/pair "
          f"({len(pairs) * args.games} games), ruleset={args.ruleset}, "
          f"komi={args.komi}, visits={args.visits} (fixed for every player)", flush=True)
    for i, j in pairs:
        print(f"  {names[i]:>28} vs {names[j]:<28}")
    if args.dry_run:
        return 0
    # Only require the binaries the chosen players actually use.
    needed = {spec[0] for spec in specs}
    for key in sorted(needed):
        if not Path(binaries[key]).exists():
            raise SystemExit(f"missing {key} binary: {binaries[key]}")

    out_path = Path(args.out) if args.out else REPO / "runs" / "arena" / f"{args.ruleset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    pair_results = {}

    def run_pair(pair):
        i, j = pair
        wins = draws = losses = 0
        for game in range(args.games):
            a_is_red = (game % 2 == 0)
            try:
                result, moves, error = play_one(specs[i], specs[j], a_is_red, binaries, args)
            except Exception as exc:  # noqa: BLE001 - a crashed player must not stop the arena
                print(f"  !! {names[i]} vs {names[j]}: {exc}", flush=True)
                result, moves, error = None, [], "?"
            if error is not None:
                # The side that failed to produce or accept a move forfeits.
                a_forfeited = (error == "A")
                result = -1 if a_forfeited else 1
            if result is None:
                draws += 1
            elif result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1
        with lock:
            pair_results[pair] = (wins, draws, losses)
            print(f"  {names[i]:>28} vs {names[j]:<28} {wins:>3}/{draws:>2}/{losses:<3}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(run_pair, pairs))

    matrix = {
        "names": names,
        "pairs": [[names[i], names[j], *pair_results[(i, j)], 1.0] for i, j in pairs],
        "visits": args.visits,
        "komi": args.komi,
        "ruleset": args.ruleset,
        "games_per_pair": args.games,
        "hang_guard_ms": args.hang_guard_ms,
    }
    out_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"saved {out_path}")
    print(f"next: python training/elo.py --report {out_path} --anchor random")
    return 0


if __name__ == "__main__":
    sys.exit(main())
