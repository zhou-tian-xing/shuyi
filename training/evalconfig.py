#!/usr/bin/env python3
"""The one evaluation budget, in the one place it is allowed to live.

Why a budget and not a clock: a rating must measure the model, not the host.
Two runs on different machines, or two runs on the same machine with a different
background load, must spend the same search effort.  Wall-clock budgets fail
that test (an 80 ms budget is a different number of visits on every machine), so
every evaluation path in this repo is budgeted in **visits** and nothing else.

What one visit means, per player type (this is the whole contract):

    gpu:<model>        one tree simulation = one batched CUDA network evaluation
    mcts_nn:<model>    one tree simulation = one CPU network evaluation
    mcts               one tree simulation = one rollout playout

So a table is only meaningful if every player in it used the same number.  The
number is ``EVAL_VISITS`` below; change it here, not at the call site, or the
tables stop being comparable.

Budget history (why it is what it is):

    96    too small -- the hand-written alpha-beta ranked *between* two network
          generations, i.e. the table measured low-budget robustness
    128   used for the first blocking-rule table; still noisy
    192   ladder/bridge rows of that table
    512   current standard: standard error on a 50% pair is about 1.6 pp per
          400 games, and one game costs 0.5-3 s on a 3080

Measured cost of one move at 512 (blocking rule, b4c32, ~52 plies, this laptop's
MX350 plus a desktop CPU; a 3080 is several times faster per visit):

    gpu          512 visits/move ~= 0.19 s/move ~= 10 s/game
    mcts         512 rollouts    ~= 15 ms/move  ~= 0.8 s/game
    mcts_nn      512 sims        ~= 2 s/move    ~= 107 s/game  (CPU forward)
"""

import math

# The standard default.  Everything that rates a model uses this.
EVAL_VISITS = 512

# How many games a pair needs before the result means anything.  50% +/-:
#   games  95% half-width
#      20          20 pp
#      60          12 pp
#     200           7 pp
#     400           5 pp
EVAL_GAMES_PER_PAIR = 60


KOMI_BY_RULESET = {"classic": -0.5, "blocking": 1.26, "nopass": 0.5}


def match_config(report, *, allow_untagged=False):
    """Validate the conditions shared by every game in a rateable report."""
    ruleset = report.get("ruleset")
    if not isinstance(ruleset, str) or ruleset not in KOMI_BY_RULESET:
        if ruleset is not None or not allow_untagged:
            raise ValueError("missing or unknown ruleset")
    komi = report.get("komi")
    if type(komi) not in (int, float) or not math.isfinite(komi):
        raise ValueError("missing or invalid komi")
    visits = report.get("visits")
    if type(visits) is not int or visits <= 0:
        raise ValueError("missing or invalid visits")
    visits_b = report.get("visits_b", None if "a_wins" in report and "pairs" not in report else visits)
    if type(visits_b) is not int or visits_b != visits:
        raise ValueError("the two players used different visit budgets")
    return ruleset, float(komi) or 0.0, visits


def board_directory(root, ruleset, komi):
    return root / ruleset / f"komi_{float(komi) or 0.0}"
