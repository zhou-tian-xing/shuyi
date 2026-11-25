"""Tests for the session budget: sizing maths and the deadline rule.

Both are the difference between a ten-hour rental that trains a model and one
that trains a few hundred games.  They are pure arithmetic, so they are tested
here instead of by watching a GPU for ten hours.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import budget  # noqa: E402

failures = 0


def check(condition, what):
    global failures
    if condition:
        return
    print(f"FAIL {what}")
    failures += 1


def test_sizing():
    p = budget.plan(0.58, 10.0, 1.03, 1024)
    check(p["generations"] >= 4, "ten hours must fit at least four generations")
    check(p["games_per_generation"] >= 2000,
          f"375-game generations are the failure this exists to stop, got {p['games_per_generation']}")
    check(p["games_per_generation"] <= 4000,
          f"above 4,000 games a generation should be split, got {p['games_per_generation']}")
    check(p["optimizer_steps_per_generation"] >= 3000,
          "a generation needs enough gradient steps to move the weights")
    check(p["total_hours"] <= 10.0, "the plan must fit the window")
    check(p["total_positions"] > 500_000, "ten hours must buy real data")

    # The window is spent on generations, not on thicker generations: a faster
    # box must get more of them, not just bigger ones.
    fast = budget.plan(1.20, 10.0, 1.03, 1024)
    check(fast["generations"] > p["generations"], "a faster box buys more generations")
    check(fast["games_per_generation"] >= 2000, "and still at least 2,000 games each")
    check(fast["total_positions"] >= p["total_positions"] * 0.8,
          "more generations must not cost most of the data")

    # The plan is a function of the measured rate: half the speed, roughly half
    # the data, and still inside the window.
    slow = budget.plan(0.29, 10.0, 1.03, 1024)
    check(slow["total_hours"] <= 10.0, "a slow box still must not overrun")
    check(slow["total_positions"] < p["total_positions"], "a slow box buys less data")
    check(slow["generations"] <= p["generations"], "a slow box cannot have more generations")

    # Bigger model, more gradient steps per generation.
    big = budget.plan(0.58, 10.0, 3.0, 1024)
    check(big["optimizer_steps_per_generation"] > p["optimizer_steps_per_generation"],
          "a bigger network needs more gradient descent")

    # A window that cannot buy one honest generation must fail, not fudge.
    try:
        budget.plan(0.58, 1.0, 1.03, 1024)
    except ValueError:
        pass
    else:
        check(False, "a one-hour window must be refused")


def test_deadline():
    check(not budget.deadline_exceeded(7200, []),
          "before the first generation there is nothing to estimate from")
    check(budget.deadline_exceeded(60, [600]),
          "a minute left, the last generation took ten: stop")
    check(not budget.deadline_exceeded(1000, [600]),
          "1000 s left, 600 s + 15% slack needed: start another")
    check(budget.deadline_exceeded(680, [600]),
          "680 s left is inside the 15% slack: stop before overrunning")
    check(budget.deadline_exceeded(750, [200, 700]),
          "the estimate uses the slowest generation observed, not the last one")


if __name__ == "__main__":
    test_sizing()
    test_deadline()
    if failures == 0:
        print("budget tests passed")
    sys.exit(0 if failures == 0 else 1)
