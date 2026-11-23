#!/usr/bin/env python3
"""Shuyi generation pipeline: one GPU binary does every position-generating step.

Design (and why it looks like this)
-----------------------------------
Branching factor at the start of a game is 729 (81 cells x 9 numbers).  A tree
search with a few hundred visits therefore visits most moves once and produces a
policy target whose support is an artefact of the prior's ordering, not a
function of the position -- two generations were lost to exactly that, with
target entropy equal to uniform entropy and max probability *below* uniform.

So the pipeline bootstraps the prior before it asks the tree to do anything:

  generation 0     uniform-random games -> train   (value head only; teaches the
                                                    score function, policy target
                                                    is honestly uniform)
  generation 1..K  one-ply full-width value scan   (every legal move evaluated,
                                                    policy = softmax(Q/T) with T
                                                    scaled to the Q spread)
  generation K+1.. MCTS self-play at --visits      (only now is a prior worth
                                                    truncating)

Every position comes from `shuyi_selfplay_gpu` (C++ tree, batched CUDA
evaluation); training is `training/train_resnet.py` on CUDA.  There is no Python
MCTS in this loop and no CPU inference anywhere.

Each generation is gated on the *health of its targets* before any training
happens, and promotion is decided by a GPU arena with alternating colours.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root (training/..)
DEFAULT_SELFPLAY_GPU = REPO / "build" / "shuyi_selfplay_gpu"
DEFAULT_TRAINER = REPO / "training" / "train_resnet.py"
TOTAL_PLIES = 54
MOVES = 729


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    print(f"[{now()}] {message}", flush=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_manifest(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"generations": []}


def resolve_binary(value):
    path = Path(value)
    if not path.exists() and os.name == "nt" and not path.suffix:
        candidate = Path(str(path) + ".exe")
        if candidate.exists():
            return candidate
    return path


def shard_stats(path):
    try:
        with open(path, "rb") as f:
            magic = f.read(8)
            version, count = struct.unpack("<II", f.read(8))
        if magic not in (b"SYSELF1\x00", b"SYSELF2\x00", b"SYSELF3\x00", b"SYSELF4\x00"):
            return 0
        return int(count)
    except OSError:
        return 0


def run_command(cmd, log_path, dry_run=False, capture=None):
    """Run a child process, stream its output, and return the lines it printed."""
    printable = " ".join(str(part) for part in cmd)
    log(f"$ {printable}")
    if dry_run:
        return []
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(printable + "\n")
        log_file.flush()
        process = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                lines.append(line.rstrip("\n"))
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        code = process.wait()
    if code != 0:
        tail = Path(log_path).read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"command failed ({code}): {printable}\n{tail}")
    return lines


def parse_json_line(lines, prefix):
    for line in reversed(lines):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):].strip())
    return {}


class Plan:
    """Measured throughput, used to size a generation before spending the GPU."""

    def __init__(self, evals_per_sec):
        self.evals_per_sec = max(1.0, float(evals_per_sec))

    def selfplay_seconds(self, games, visits):
        # One evaluation per simulation; ~15% is tree bookkeeping and sampling.
        return games * TOTAL_PLIES * visits / (self.evals_per_sec * 0.85)

    def oneply_seconds(self, games):
        return games * TOTAL_PLIES * MOVES / self.evals_per_sec

    def games_for_budget(self, budget_seconds, visits, share):
        per_game = TOTAL_PLIES * visits / (self.evals_per_sec * 0.85)
        return int(max(1, budget_seconds * share / max(1e-9, per_game)))

    def describe(self, args):
        table = [
            ("random gen0", f"{args.games_per_gen} games", "~0 (no network)"),
            (
                f"oneply (gens 1..{args.oneply_gens})",
                f"{args.games_per_gen} games x {MOVES} evals/ply",
                f"{self.oneply_seconds(args.games_per_gen) / 60:.1f} min",
            ),
            (
                f"mcts (gens {args.oneply_gens + 1}..)",
                f"{args.games_per_gen} games x {args.visits} visits",
                f"{self.selfplay_seconds(args.games_per_gen, args.visits) / 60:.1f} min",
            ),
            (
                "arena",
                f"{args.arena_games} games x {self.arena_visits(args)} visits",
                f"{self.selfplay_seconds(args.arena_games, self.arena_visits(args)) / 60:.1f} min",
            ),
        ]
        return table

    def arena_visits(self, args):
        return args.arena_visits or args.visits


def bench(plan_args, selfplay_bin, model, chunk=1024):
    """Measure forward-pass throughput on this GPU; returns evals/s at the batch we use."""
    lines = run_command(
        [str(selfplay_bin), "--model", str(model), "--mode", "bench", "--chunk", str(chunk)],
        plan_args / "logs" / "bench.log",
    )
    best = 0.0
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit() and parts[1].replace(".", "", 1).isdigit():
            best = max(best, float(parts[2]))
    if best <= 0.0:
        raise RuntimeError("bench produced no throughput numbers")
    return best


def generate(args, gen_dir, gen, mode, model):
    shard = gen_dir / "selfplay" / "shard_000.bin"
    shard.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.selfplay_gpu),
        "--mode", mode,
        "--games", str(args.games_per_gen),
        "--batch-games", str(min(args.games_per_gen, args.batch_games)),
        "--ruleset", args.ruleset,
        "--opening", str(args.opening),
        "--komi", str(args.komi),
        "--temp-moves", str(args.temp_moves),
        "--temperature", str(args.temperature),
        "--seed", str(args.seed + gen * 100000),
        "--chunk", str(args.chunk),
        "--out", str(shard),
    ]
    if mode != "random":
        cmd += ["--model", str(model)]
    if mode == "oneply":
        cmd += ["--oneply-temp-scale", str(args.oneply_temp_scale)]
    if mode == "selfplay":
        cmd += [
            "--visits", str(args.visits),
            "--root-k", str(args.root_k),
            "--top-k", str(args.top_k),
            "--noise", str(args.noise),
            "--dirichlet", str(args.dirichlet),
        ]
    started = time.time()
    lines = run_command(cmd, gen_dir / "logs" / f"{mode}.log", args.dry_run)
    total = 0 if args.dry_run else shard_stats(shard)
    if not args.dry_run and total <= 0:
        raise RuntimeError(f"generation {gen}: {mode} produced no samples")
    stats = parse_json_line(lines, "TARGET_STATS")
    log(f"generation {gen}: {mode} produced {total} positions in {time.time() - started:.1f}s")
    return shard, total, stats


def check_targets(args, gen, mode, total, stats):
    """Refuse to train on a target distribution that cannot teach a policy."""
    if args.dry_run or not stats:
        return None
    if mode == "random":
        # Generation 0 is deliberate ignorance: the policy target is uniform over
        # the legal moves and only the value head is trained. Gating it on policy
        # information would reject the bootstrap itself.
        log(f"generation {gen}: random-play bootstrap, policy targets are uniform by design")
        return {"mode": mode, "positions": total, "note": "uniform policy by design"}
    support = float(stats.get("support", 0.0))
    entropy = float(stats.get("entropy", 0.0))
    max_prob = float(stats.get("max_prob", 0.0))
    legal = float(stats.get("legal", 0.0))
    cell_ratio = float(stats.get("cell_ratio", 1.0))
    number_mass = float(stats.get("number_mass", 0.0))
    if legal <= 1.0:
        raise RuntimeError(f"generation {gen}: target legal count is {legal:.1f}")
    uniform_entropy = __import__("math").log(legal)
    uniform_prob = 1.0 / legal
    ratio = entropy / uniform_entropy if uniform_entropy > 0 else 1.0
    record = {
        "mode": mode,
        "positions": total,
        "legal": legal,
        "support": support,
        "entropy": entropy,
        "entropy_ratio": ratio,
        "max_prob": max_prob,
        "max_prob_vs_uniform": max_prob / uniform_prob,
        "number_mass": number_mass,
        "cell_ratio": cell_ratio,
    }
    log(
        f"generation {gen}: targets legal={legal:.1f} support={support:.1f} "
        f"entropy={entropy:.3f} ({ratio:.3f} of uniform) "
        f"peak={max_prob:.4f} ({max_prob / uniform_prob:.2f}x uniform) "
        f"number_mass={number_mass:.3f} cell_ratio={cell_ratio:.3f}"
    )
    # cell_ratio near 1.0 is the exact signature of the wasted generations: all of
    # the mass on one number, cells arbitrary, so the policy head could only learn
    # the number.
    if cell_ratio > args.max_cell_ratio:
        raise RuntimeError(
            f"generation {gen}: the target does not distinguish cells "
            f"(cell_ratio {cell_ratio:.3f} > {args.max_cell_ratio}). "
            f"{number_mass:.1%} of the mass sits on a single number. "
            + (
                "The value network barely depends on where a stone goes, so this is a "
                "generation-0 sizing problem, not a search problem: train gen0 on more "
                "games and epochs so the value head sees the spatial structure."
                if mode == "oneply"
                else "Increase --visits (rule of thumb: >= 20 x root-k) or lower --root-k."
            )
        )
    if max_prob < args.min_max_prob_ratio * uniform_prob:
        raise RuntimeError(
            f"generation {gen}: policy targets peak at {max_prob:.4f}, only "
            f"{max_prob / uniform_prob:.2f}x uniform ({uniform_prob:.4f}); nothing to learn. "
            f"Increase --visits/--root-k or use oneply."
        )
    if ratio > args.max_entropy_ratio:
        raise RuntimeError(
            f"generation {gen}: policy targets are too close to uniform "
            f"(entropy ratio {ratio:.3f} > {args.max_entropy_ratio}). "
            f"The search added no information; increase --visits/--root-k or use oneply."
        )
    return record


def train(args, gen_dir, gen, shards, init_model):
    candidate = gen_dir / "candidate.bin"
    checkpoint = gen_dir / "checkpoints" / "last.pt"
    summary = gen_dir / "train_summary.json"
    cmd = [
        args.python,
        str(args.trainer),
        "--data", *[str(path) for path in shards],
        "--out", str(candidate),
        "--checkpoint", str(checkpoint),
        "--summary", str(summary),
        "--log-csv", str(gen_dir / "train_metrics.csv"),
        "--channels", str(args.channels),
        "--blocks", str(args.blocks),
        "--input-planes", str(args.input_planes),
        "--ruleset", args.ruleset,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--max-samples", str(args.max_samples),
        "--lr", str(args.lr),
        "--value-weight", str(args.value_weight),
        "--val", str(args.val),
        "--seed", str(args.seed + gen * 101),
        "--device", args.device,
    ]
    if args.a_cond:
        cmd += [
            "--a-cond",
            "--a-anchor", str(args.a_anchor),
            "--a-range", str(args.a_range),
            "--a-anchor-fraction", str(args.a_anchor_fraction),
        ]
    if args.amp:
        cmd.append("--amp")
    else:
        cmd.append("--no-amp")
    if init_model is not None and Path(init_model).exists() and not args.train_from_scratch:
        cmd += ["--init-model", str(init_model)]
    if args.resume and checkpoint.exists():
        cmd += ["--resume", str(checkpoint)]
    run_command(cmd, gen_dir / "logs" / "train.log", args.dry_run)
    if args.dry_run or not summary.exists():
        return candidate, {}
    return candidate, json.loads(summary.read_text(encoding="utf-8"))


def arena(args, gen_dir, gen, candidate, previous):
    out_dir = gen_dir / "arena"
    out_dir.mkdir(parents=True, exist_ok=True)
    match_json = out_dir / "match.json"
    cmd = [
        str(args.selfplay_gpu),
        "--mode", "match",
        "--model", str(candidate),
        "--model-b", str(previous),
        "--name-a", f"gen_{gen:03d}",
        "--name-b", f"gen_{gen - 1:03d}",
        "--games", str(args.arena_games),
        "--batch-games", str(min(args.arena_games, args.batch_games)),
        "--visits", str(args.arena_visits or args.visits),
        "--ruleset", args.ruleset,
        "--opening", str(args.opening),
        "--komi", str(args.komi),
        "--chunk", str(args.chunk),
        "--seed", str(args.seed + gen * 31337),
        "--json", str(match_json),
        "--csv", str(out_dir / "games.csv"),
    ]
    lines = run_command(cmd, gen_dir / "logs" / "arena.log", args.dry_run)
    if args.dry_run:
        return {"score_a": 0.5, "promoted": False, "dry_run": True}
    result = parse_json_line(lines, "MATCH_RESULT")
    if not result:
        raise RuntimeError(f"generation {gen}: arena produced no result")
    # Measurement only: the chain always advances to this generation's model.  A
    # gate that could revert would mean later generations keep generating data
    # with an older net -- and "stronger in a match" does not imply "its data is
    # the right thing to train on".  Strength ranking is the ELO stage's job.
    result["above_threshold"] = float(result["score_a"]) >= args.promotion_threshold
    log(
        f"generation {gen}: arena {result['a_wins']}/{result['draws']}/{result['b_wins']} "
        f"score={result['score_a']:.3f} (chain advances; Elo decides deployment)"
    )
    return result


def stage_elo(args, run_dir, gen, models, labels):
    """Round robin over the recent model pool; ratings, not a pairwise score."""
    out_dir = run_dir / "elo"
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = out_dir / f"ratings_gen_{gen:03d}.json"
    cmd = [
        args.python, str(args.elo),
        "--models", *[str(m) for m in models],
        "--labels", *labels,
        "--games", str(args.elo_games),
        "--visits", str(args.elo_visits or args.visits),
        "--komi", str(args.komi),
        "--opening", str(args.opening),
        "--ruleset", args.ruleset,
        "--bootstrap", str(args.elo_bootstrap),
        "--json", str(out_dir / f"matrix_gen_{gen:03d}.json"),
        "--ratings-json", str(ratings_path),
        "--selfplay-gpu", str(args.selfplay_gpu),
    ]
    run_command(cmd, gen_dir_log := (run_dir / f"gen_{gen:03d}" / "logs" / "elo.log"), args.dry_run)
    if args.dry_run or not ratings_path.exists():
        return {}
    data = json.loads(ratings_path.read_text(encoding="utf-8"))
    ranking = sorted(data["elo"].items(), key=lambda kv: -kv[1])
    log(
        f"generation {gen}: elo " +
        " ".join(f"{name}={rating:+.0f}" for name, rating in ranking)
    )
    return data


def parse_capacity(name):
    """'b6c64' -> (6, 64)."""
    text = name.strip().lower()
    if not text.startswith("b") or "c" not in text:
        raise SystemExit(f"bad capacity rung {name!r}, expected like b6c64")
    blocks, channels = text[1:].split("c", 1)
    return int(blocks), int(channels)


def next_rung(ladder, blocks, channels):
    rungs = [parse_capacity(item) for item in ladder.split(",") if item.strip()]
    for index, rung in enumerate(rungs):
        if rung == (blocks, channels):
            return rungs[index + 1] if index + 1 < len(rungs) else None
    for rung in rungs:
        if rung > (blocks, channels):
            return rung
    return None


def plateau_report(manifest, args, gen):
    """The 10-generation rule: no new Elo high for N generations means the model
    has run out of capacity (or the data resolution has), so the next run should
    use a bigger network.  See training/README.md 1.1."""
    best = None
    best_gen = None
    for record in manifest["generations"]:
        elo = record.get("elo") or {}
        ratings = elo.get("elo") or {}
        if not ratings:
            continue
        top = max(ratings.values())
        if best is None or top > best:
            best = top
            best_gen = record["gen"]
    if best is None or best_gen is None:
        return None
    stagnation = gen - best_gen
    status = {
        "best_elo": best,
        "best_gen": best_gen,
        "generations_without_progress": stagnation,
        "limit": args.plateau_generations,
        "escalate": bool(args.plateau_generations and stagnation >= args.plateau_generations),
    }
    if status["escalate"]:
        rung = next_rung(args.capacity_ladder, args.blocks, args.channels)
        status["next_rung"] = f"b{rung[0]}c{rung[1]}" if rung else "none (ladder exhausted)"
        log(
            f"generation {gen}: {stagnation} generations without a new Elo high "
            f"(best {best:.0f} at gen {best_gen}) -> capacity escalation suggested: "
            f"{status['next_rung']}"
        )
    return status


def replay_shards(run_dir, gen, count):
    shards = []
    for index in range(max(0, gen - count + 1), gen + 1):
        shards.extend(sorted((run_dir / f"gen_{index:03d}" / "selfplay").glob("shard_*.bin")))
    return shards


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan", action="store_true",
                        help="benchmark the GPU and print the cost table, then exit")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--oneply-gens", type=int, default=2,
                        help="generations that use the one-ply value scan (gen 0 is random play)")
    parser.add_argument("--games-per-gen", type=int, default=2000)
    parser.add_argument("--visits", type=int, default=1024, help="MCTS visits per move")
    parser.add_argument("--arena-visits", type=int, default=0, help="0 = same as --visits")
    parser.add_argument("--arena-games", type=int, default=120)
    parser.add_argument("--root-k", type=int, default=32, help="root children kept, 0 = all")
    parser.add_argument("--top-k", type=int, default=16, help="interior children kept, 0 = all")
    parser.add_argument("--batch-games", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--ruleset", choices=["classic", "blocking"], default="classic",
                        help="rule set: data, models and run directory are tagged with it")
    parser.add_argument("--komi", type=float, default=None,
                        help="points added to Red; default is per rule set: -0.5 classic "
                             "(Red 52%% at 0 in the 3080 run) and +1.26 blocking "
                             "(training/avalue.py measured A* = +1.26 at 192 visits)")
    parser.add_argument("--opening", type=int, default=2)
    parser.add_argument("--temp-moves", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--dirichlet", type=float, default=0.3)
    parser.add_argument("--oneply-temp-scale", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--input-planes", type=int, default=37, choices=[19, 37])
    parser.add_argument("--a-cond", action="store_true",
                        help="train a v4 model that takes the A value as an input "
                             "(see training/README.md, 'A as an input')")
    parser.add_argument("--a-anchor", type=float, default=1.26,
                        help="A the engine plays with; part of each epoch trains at it")
    parser.add_argument("--a-range", type=float, default=12.0,
                        help="explore A uniformly in [-range, range]")
    parser.add_argument("--a-anchor-fraction", type=float, default=0.35)
    parser.add_argument("--max-samples", type=int, default=600000)
    parser.add_argument("--val", type=float, default=0.06)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--replay-gens", type=int, default=1)
    parser.add_argument("--train-from-scratch", action="store_true",
                        help="ignore the previous generation's weights (gen 0 always does)")
    parser.add_argument("--elo", default=str(REPO / "training" / "elo.py"),
                        help="ELO rating script (GPU round robin)")
    parser.add_argument("--elo-every", type=int, default=1,
                        help="rate the model pool every N generations (0 = never)")
    parser.add_argument("--elo-games", type=int, default=30, help="games per pair")
    parser.add_argument("--elo-visits", type=int, default=0, help="0 = same as --visits")
    parser.add_argument("--elo-pool", type=int, default=5, help="models in the rating pool")
    parser.add_argument("--elo-bootstrap", type=int, default=120)
    parser.add_argument("--plateau-generations", type=int, default=10,
                        help="0 disables; see training/README.md 1.1")
    parser.add_argument("--plateau-margin", type=float, default=20.0,
                        help="Elo a generation must beat the best so far to count as progress")
    parser.add_argument("--capacity-ladder", default="b4c32,b6c48,b6c64,b8c96")
    parser.add_argument("--promotion-games", type=int, default=0,
                        help="deprecated alias for --arena-games")
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--max-cell-ratio", type=float, default=0.90,
                        help="fail if the target's cells are as uninformative as uniform")
    parser.add_argument("--max-entropy-ratio", type=float, default=0.97,
                        help="fail a generation if target entropy exceeds this fraction of uniform")
    parser.add_argument("--min-max-prob-ratio", type=float, default=2.0,
                        help="fail a generation if the target peak is below this multiple of uniform")
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--selfplay-gpu", default=str(DEFAULT_SELFPLAY_GPU))
    parser.add_argument("--trainer", default=str(DEFAULT_TRAINER))
    parser.add_argument("--python", default=sys.executable)
    # KataGo convention: b{blocks}c{channels}-s{steps}-d{data}.  Steps are the
    # cumulative optimizer steps, data the cumulative self-play positions.  The
    # rule set is NOT in the file name: it is a byte in the model header and the
    # run directory already separates the two rule sets.
    parser.add_argument("--name-template",
                        default="b{blocks}c{channels}-s{steps}-d{data}")
    args = parser.parse_args(argv)
    args.selfplay_gpu = str(resolve_binary(args.selfplay_gpu))
    args.trainer = str(resolve_binary(args.trainer))
    if args.komi is None:
        # Measured, not guessed: classic -0.5 (Red 52% at 0 on the 3080 run),
        # blocking +1.26 (training/avalue.py: A* = +1.26, SE 0.22, 240 games per
        # point at 192 visits -- going first is a liability under this rule).
        args.komi = -0.5 if args.ruleset == "classic" else 1.26
    if args.promotion_games:
        args.arena_games = args.promotion_games

    if args.generations < 1:
        raise SystemExit("--generations must be >= 1")
    if args.oneply_gens < 0:
        raise SystemExit("--oneply-gens must be >= 0")
    if not Path(args.selfplay_gpu).exists() and not args.dry_run:
        raise SystemExit(f"missing self-play binary: {args.selfplay_gpu}")

    run_dir = (
        Path(args.run_dir) if args.run_dir
        else REPO / "runs" / args.ruleset / args.run_name
    )
    models_dir = run_dir / "models"
    if args.plan:
        models_dir.mkdir(parents=True, exist_ok=True)
        # A random net is enough to measure the forward pass.
        probe = models_dir / "probe.bin"
        if not probe.exists():
            probe_cmd = [
                args.python, str(args.trainer), "--init-only", "--out", str(probe),
                "--channels", str(args.channels), "--blocks", str(args.blocks),
                "--input-planes", str(args.input_planes),
            ]
            if args.a_cond:
                probe_cmd += ["--a-cond", "--a-anchor", str(args.a_anchor)]
            run_command(probe_cmd, run_dir / "logs" / "probe.log")
        evals = bench(run_dir, args.selfplay_gpu, probe, args.chunk)
        plan = Plan(evals)
        print(f"\nmeasured forward throughput: {evals:,.0f} evals/s "
              f"({args.blocks} blocks x {args.channels} channels)\n")
        print(f"{'stage':28s} {'work':38s} {'cost':>10s}")
        for name, work, cost in plan.describe(args):
            print(f"{name:28s} {work:38s} {cost:>10s}")
        print()
        return 0

    if run_dir.exists() and not (args.resume or args.force):
        raise SystemExit(f"run directory exists: {run_dir} (use --resume or --force)")
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = read_manifest(manifest_path) if args.resume else {"generations": []}
    if args.resume and manifest.get("config"):
        old = manifest["config"]
        for key in ("channels", "blocks", "input_planes"):
            if int(old.get(key, -1)) != int(getattr(args, key)):
                raise SystemExit(
                    f"resume shape mismatch: run used {key}={old.get(key)}, "
                    f"current {key}={getattr(args, key)}"
                )
        # A v3 model and a v4 model are different formats: resuming a run with a
        # different --a-cond would warm start a 22-wide head from a 24-wide file.
        if bool(old.get("a_cond", False)) != bool(args.a_cond):
            raise SystemExit(
                f"resume mismatch: run used a_cond={old.get('a_cond')}, "
                f"current a_cond={args.a_cond}"
            )
    manifest["config"] = vars(args)
    manifest["started"] = manifest.get("started", now())
    manifest["updated"] = now()
    write_json(manifest_path, manifest)

    completed = {int(item["gen"]): item for item in manifest["generations"] if "gen" in item}
    cumulative_steps = sum(int(item.get("optimizer_steps", 0)) for item in manifest["generations"])
    cumulative_data = sum(int(item.get("selfplay_samples", 0)) for item in manifest["generations"])
    previous_model = None
    for gen in range(args.generations):
        gen_dir = run_dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        if args.resume and gen in completed:
            # The manifest records the canonical file name, so resuming does not
            # depend on a second (gen_XXX) copy existing.
            record = completed[gen]
            previous_model = record.get("named_model") or record.get("final_model")
            log(f"generation {gen}: already complete, skipping")
            continue

        if gen == 0:
            mode = "random"
        elif gen <= args.oneply_gens:
            mode = "oneply"
        else:
            mode = "selfplay"
        if mode != "random" and (previous_model is None or not Path(previous_model).exists()):
            raise RuntimeError(f"generation {gen}: no model to generate {mode} data with")

        log(f"===== generation {gen}/{args.generations - 1} ({mode}) =====")
        shard, samples, stats = generate(args, gen_dir, gen, mode, previous_model)
        target_record = check_targets(args, gen, mode, samples, stats)

        shards = replay_shards(run_dir, gen, args.replay_gens) or [shard]
        if args.dry_run:
            shards = [shard]
        log(f"generation {gen}: training on {len(shards)} shard(s)")

        candidate, summary = train(args, gen_dir, gen, shards, None if gen == 0 else previous_model)
        train_samples = sum(shard_stats(path) for path in shards)
        steps_per_epoch = max(
            1, (int(train_samples * (1.0 - args.val)) + args.batch - 1) // args.batch
        )
        optimizer_steps = steps_per_epoch * args.epochs
        cumulative_steps += optimizer_steps
        cumulative_data += samples

        record = {
            "gen": gen,
            "mode": mode,
            "play_model": str(previous_model) if previous_model else "",
            "selfplay_samples": samples,
            "target_stats": target_record,
            "optimizer_steps": optimizer_steps,
            "cumulative_steps": cumulative_steps,
            "cumulative_data": cumulative_data,
            "train_summary": summary.get("last", {}),
            "candidate": str(candidate),
            "finished": now(),
        }

        # The chain always advances to this generation's model: generation N+1
        # must play with generation N and train on that data (sequential
        # training).  Keeping an older "best" model would freeze the data
        # distribution and waste generations -- which is exactly what happened in
        # the 3080 run, where gen3..gen6 all played with gen2's weights.
        record["chained_from"] = str(previous_model) if previous_model else ""
        if gen == 0 or previous_model is None:
            record["arena"] = {"note": "bootstrap generation; nothing to compare against"}
        else:
            record["arena"] = arena(args, gen_dir, gen, candidate, previous_model)
        record["promoted"] = True
        model_name = args.name_template.format(
            blocks=args.blocks, channels=args.channels,
            steps=cumulative_steps, data=cumulative_data,
        )
        record["model_name"] = model_name
        final_model = models_dir / f"{model_name}.bin"
        record["final_model"] = str(final_model)
        if not args.dry_run:
            shutil.copyfile(candidate, final_model)
            log(f"generation {gen}: model {final_model.name}")

        if args.elo_every and gen >= 1 and not args.dry_run and (
            gen % args.elo_every == 0 or gen == args.generations - 1
        ):
            pool = []
            for g in range(gen, -1, -1):
                if g == gen:
                    pool.append(final_model)
                    continue
                earlier = completed.get(g)
                if not earlier:
                    continue
                named = earlier.get("named_model") or earlier.get("final_model")
                if named:
                    pool.append(Path(named))
            pool = [p for p in pool if p.exists()][: max(2, args.elo_pool)]
            pool.reverse()
            labels = [f"gen_{g:03d}" for g in range(gen - len(pool) + 1, gen + 1)]
            elo = stage_elo(args, run_dir, gen, pool, labels)
            if elo:
                record["elo"] = elo
                best = max(elo["elo"].items(), key=lambda kv: kv[1])
                record["elo_best"] = best[0]
                manifest["elo_best"] = best[0]

        plateau = plateau_report(manifest, args, gen)
        if plateau:
            record["plateau"] = plateau
            if plateau.get("escalate"):
                manifest["capacity_escalation"] = {
                    "gen": gen,
                    "reason": f"{plateau['generations_without_progress']} generations without an Elo high",
                    "best_elo": plateau["best_elo"],
                    "best_gen": plateau["best_gen"],
                    "next_rung": plateau.get("next_rung"),
                }

        manifest["generations"] = [item for item in manifest["generations"] if item.get("gen") != gen]
        manifest["generations"].append(record)
        manifest["generations"].sort(key=lambda item: item.get("gen", -1))
        manifest["updated"] = now()
        write_json(manifest_path, manifest)
        previous_model = final_model

        if summary:
            last = summary.get("last", {})
            log(
                f"generation {gen}: val={summary.get('best_val_loss')} "
                f"policy={last.get('val_policy')} value={last.get('val_value')} "
                f"top1={last.get('top1')} mae={last.get('value_mae')}"
            )

    manifest["finished"] = now()
    manifest["updated"] = now()
    write_json(manifest_path, manifest)
    log(f"done: {run_dir}")
    log(f"final model: {models_dir / f'gen_{args.generations - 1:03d}.bin'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
