#!/usr/bin/env python3
"""Size a generation run to a wall-clock window.

The failure this exists to prevent: a run that is sized by "how many games did I
feel like" instead of by the box, so ten rented hours end up training a 1M
parameter network on a few hundred games.  Data per generation and optimizer
steps per generation scale with the model; the window decides how many
generations fit, never the other way round.

Inputs are measured, not assumed: `run_session.sh` probes the self-play rate with
the real model and visits, and passes the training rate it observes on the same
box.  Everything here is arithmetic plus a printout, so it can be checked without
a GPU (`--self-test`).
"""
import argparse
import json
import math

POSITIONS_PER_GAME = 52      # measured: 36,399 positions / 700 games
SELF_PLAY_SHARE = 1.0        # games/s already covers the whole self-play pass
ARENA_SEARCH_FACTOR = 0.8    # a match searches both sides: a bit slower per game


def steps_per_generation(positions, epochs, batch=512, train_share=0.94):
    return int(positions * train_share / batch * epochs)


def epochs_for_steps(positions, target_steps, batch=512, train_share=0.94,
                     low=12, high=60):
    per_epoch = max(1, positions * train_share / batch)
    epochs = math.ceil(target_steps / per_epoch)
    return max(low, min(high, epochs))


def deadline_exceeded(remaining_seconds, gen_seconds, slack=1.15):
    """True when the next generation no longer fits in what is left of the window.

    The generation in progress always finishes; this only decides whether to
    *start* another one.  Without an observed generation there is no estimate, so
    the answer is "keep going" -- a run must always produce at least one model.
    """
    if not gen_seconds:
        return False
    return remaining_seconds < max(gen_seconds) * slack


def plan(rate_games_per_s, hours, params_millions=1.03, visits=1024,
         arena_games=100, setup_minutes=15, tail_minutes=12,
         min_gens=4, max_gens=16, min_games=2000, max_games=4000,
         train_rate=20000.0, batch=512, target_steps_base=10000,
         replay_gens=2):
    """Return a plan dict; raises ValueError when the window fits nothing."""
    if rate_games_per_s <= 0:
        raise ValueError("self-play rate must be positive")
    usable = hours * 3600.0 - setup_minutes * 60.0 - tail_minutes * 60.0
    if usable <= 600.0:
        raise ValueError("the window has under ten minutes of usable time")

    arena_rate = rate_games_per_s * ARENA_SEARCH_FACTOR
    arena_seconds = arena_games / arena_rate
    # Steps and data both grow with the network: this is the "a bigger model needs
    # more gradient descent" rule, written down instead of remembered.
    target_steps = int(target_steps_base * (params_millions / 1.03))

    def size_for(gens):
        budget = usable / gens
        # A generation trains on the shards still inside the replay window, not
        # only on the one it just produced.  Ignoring that understates the
        # training pass by up to `replay_gens` times and puts the run over the
        # clock; the first generation has nothing to replay, so the average over
        # the run is used.
        train_factor = 1.0 + (replay_gens - 1) * max(0, gens - 1) / max(1, gens)
        epochs = 24
        games = 0.0
        for _ in range(6):
            seconds_per_game = 1.0 / rate_games_per_s
            train_per_game = POSITIONS_PER_GAME * epochs * train_factor / train_rate
            games = (budget - arena_seconds) / (seconds_per_game + train_per_game)
            positions = games * POSITIONS_PER_GAME
            epochs = epochs_for_steps(positions * train_factor, target_steps, batch=batch)
        games = int(max(0, games))
        positions = games * POSITIONS_PER_GAME
        epochs = epochs_for_steps(positions * train_factor, target_steps, batch=batch)
        steps = steps_per_generation(positions * train_factor, epochs, batch=batch)
        train_seconds = positions * train_factor * epochs / train_rate
        gen_seconds = games / rate_games_per_s + train_seconds + arena_seconds
        return {
            "generations": gens,
            "games_per_generation": games,
            "positions_per_generation": positions,
            "epochs": epochs,
            "optimizer_steps_per_generation": steps,
            "self_play_seconds": int(games / rate_games_per_s),
            "train_seconds": int(train_seconds),
            "train_factor": train_factor,
            "arena_seconds": int(arena_seconds),
            "generation_minutes": gen_seconds / 60.0,
            "total_hours": (setup_minutes * 60.0 + tail_minutes * 60.0
                            + gens * gen_seconds) / 3600.0,
        }

    # Generations are the unit of learning: each one replaces the data with a
    # search run by a better network and adds a step of the ladder.  So spend the
    # window on as many generations as it fits, subject to each one still holding
    # at least `min_games` games -- few hundred games teach nothing, and 2,000 is
    # the floor that has been holding up (the 2026-11-25 runs: 2,177 and 4,194
    # games both improved every generation, 4,194 just bought fewer of them).
    # The old rule was "one generation per 90 minutes", which is a description of
    # the last run rather than a reason, and it silently halved the number of
    # model updates when self-play got faster.
    best = size_for(max_gens)
    found = False
    for gens in range(max_gens, min_gens - 1, -1):
        candidate = size_for(gens)
        if candidate["total_hours"] > hours:
            continue
        if candidate["games_per_generation"] < min_games:
            continue
        best = candidate
        found = True
        break
    if not found:
        best = size_for(min_gens)
        raise ValueError(
            f"even {min_gens} generations give only "
            f"{best['games_per_generation']} games each; the window is too short "
            f"for {params_millions:.2f}M parameters at this rate")
    # More than the box can use in one generation: split it into more generations.
    while best["games_per_generation"] > max_games and best["generations"] < max_gens:
        best = size_for(best["generations"] + 1)
    if best["total_hours"] > hours:
        raise ValueError(f"plan needs {best['total_hours']:.2f} h, window is {hours:.2f} h")

    best.update({
        "rate_games_per_s": rate_games_per_s,
        "hours": hours,
        "visits": visits,
        "params_millions": params_millions,
        "arena_games": arena_games,
        "setup_minutes": setup_minutes,
        "replay_gens": replay_gens,
        "tail_minutes": tail_minutes,
        "total_games": best["games_per_generation"] * best["generations"],
        "total_positions": best["positions_per_generation"] * best["generations"],
        "total_optimizer_steps": (best["optimizer_steps_per_generation"]
                                  * best["generations"]),
    })
    return best


def render(p):
    lines = [
        f"window            {p['hours']:.2f} h  (setup {p['setup_minutes']} min, "
        f"tail {p['tail_minutes']} min)",
        f"self-play rate    {p['rate_games_per_s']:.2f} games/s  ({p['visits']} visits)",
        f"model             {p['params_millions']:.2f} M parameters",
        "",
        f"generations       {p['generations']}",
        f"games / gen       {p['games_per_generation']}",
        f"positions / gen   {p['positions_per_generation']:,}",
        f"epochs            {p['epochs']}  (x{p['train_factor']:.1f} replay shards)",
        f"steps / gen       {p['optimizer_steps_per_generation']:,}",
        f"time / gen        {p['generation_minutes']:.0f} min "
        f"(self-play {p['self_play_seconds'] // 60} min, "
        f"train {p['train_seconds'] // 60} min, arena {p['arena_seconds'] // 60} min)",
        "",
        f"total games       {p['total_games']:,}",
        f"total positions   {p['total_positions']:,}",
        f"total steps       {p['total_optimizer_steps']:,}",
        f"total wall time   {p['total_hours']:.2f} h",
    ]
    return "\n".join(lines)


def self_test():
    p = plan(0.58, 10.0, 1.03, 1024)
    assert p["generations"] >= 4
    assert p["games_per_generation"] >= 2000
    assert p["total_hours"] <= 10.0
    assert p["optimizer_steps_per_generation"] >= 3000
    # A slower box must buy fewer, smaller generations -- never a plan that
    # overruns the window, and never one that drops under the 2,000-game floor:
    # below that it has to refuse instead of quietly producing toy generations.
    slow = plan(0.35, 10.0, 1.03, 1024)
    assert slow["total_hours"] <= 10.0, slow
    assert slow["total_positions"] < p["total_positions"]
    assert slow["games_per_generation"] >= 2000
    try:
        plan(0.25, 10.0, 1.03, 1024)
    except ValueError:
        pass
    else:
        raise AssertionError("a box this slow cannot hold the 2,000-game floor in ten hours")
    # A bigger network must ask for more gradient steps per position.
    small = plan(0.58, 10.0, 1.03, 1024)
    big = plan(0.58, 30.0, 3.00, 1024)
    assert big["optimizer_steps_per_generation"] > small["optimizer_steps_per_generation"]
    # A window too short to do anything useful must fail loudly, not silently
    # produce a 40-game generation.
    try:
        plan(0.58, 1.0, 1.03, 1024)
    except ValueError:
        pass
    else:
        raise AssertionError("a one-hour window should be refused, not fudged")
    print("budget self-test passed")
    print(render(p))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, help="measured self-play games/s")
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--params", type=float, default=1.03, help="millions of parameters")
    parser.add_argument("--visits", type=int, default=1024)
    parser.add_argument("--arena-games", type=int, default=100)
    parser.add_argument("--setup-minutes", type=float, default=15)
    parser.add_argument("--tail-minutes", type=float, default=12)
    parser.add_argument("--min-gens", type=int, default=4)
    parser.add_argument("--max-gens", type=int, default=16)
    parser.add_argument("--min-games", type=int, default=2000,
                        help="floor on games per generation; a few hundred teach nothing")
    parser.add_argument("--max-games", type=int, default=4000,
                        help="ceiling on games per generation; above it, add a generation")
    parser.add_argument("--train-rate", type=float, default=20000.0)
    parser.add_argument("--replay-gens", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.rate is None:
        parser.error("--rate is required (or use --self-test)")
    p = plan(args.rate, args.hours, args.params, args.visits, args.arena_games,
             args.setup_minutes, args.tail_minutes, args.min_gens, args.max_gens,
             args.min_games, args.max_games, args.train_rate,
             replay_gens=args.replay_gens)
    if args.json:
        print(json.dumps(p, indent=1))
    else:
        print(render(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
