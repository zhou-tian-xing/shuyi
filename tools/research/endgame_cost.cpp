// How much does the exact endgame solver cost as the remaining plies grow?
//
// The display only turns "exact" on when the root is solved.  `exactEndgamePlies`
// decides when the solver is tried at all, so this prints, for a few random
// legal positions with 5..9 plies left: whether the solver finished inside a
// budget, how many nodes it needed, and how long it took.  It is the measurement
// behind the threshold, not a rule of thumb.
#include "shuyi/endgame.h"
#include "shuyi/position.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace shuyi;

namespace {

std::uint64_t g_seed = 20261125;

std::uint64_t next() {
    g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return g_seed;
}

Position latePosition(int pliesLeft, std::uint64_t seedStart) {
    g_seed = seedStart;
    Position position;
    position.setRuleset(Ruleset::NoPass);
    while (kTotalMoves - position.stones() > pliesLeft) {
        const std::vector<Move> moves = position.legalMoves();
        if (moves.empty()) break;
        position.play(moves[next() % moves.size()]);
    }
    return position;
}

} // namespace

int main() {
    const long long budgets[] = {250000LL, 2000000LL, 10000000LL};
    std::printf("%5s %5s %10s %8s %s\n", "plies", "which", "budget", "nodes", "result");
    for (int plies = 5; plies <= 9; ++plies) {
        for (int which = 0; which < 4; ++which) {
            Position position = latePosition(plies, 1000 + static_cast<std::uint64_t>(which) * 7919);
            if (kTotalMoves - position.stones() != plies) {
                std::printf("%5d %5d   (skipped: game ended earlier)\n", plies, which);
                continue;
            }
            for (long long budget : budgets) {
                const auto start = std::chrono::steady_clock::now();
                const EndgameSolution solved = solveEndgame(position, 0.5, budget);
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
                std::printf("%5d %5d %10lld %8lld %s (%.1f ms)\n", plies, which, budget,
                            solved.nodes, solved.solved ? "solved" : "GAVE UP", ms);
                if (solved.solved) break;   // a bigger budget adds nothing
            }
        }
    }

    // The same positions under the limits the engine actually uses: a node
    // budget plus a wall-clock deadline.  What matters for a front end is that
    // the worst case is bounded, not that every position is solved.
    std::printf("\nwith deadlines (node budget + wall clock)\n");
    struct Limit { long long nodes; int ms; };
    const Limit limits[] = {{250000LL, 120}, {1500000LL, 400}, {3000000LL, 800}};
    for (int plies = 5; plies <= 9; ++plies) {
        for (int which = 0; which < 4; ++which) {
            Position position = latePosition(plies, 1000 + static_cast<std::uint64_t>(which) * 7919);
            if (kTotalMoves - position.stones() != plies) continue;
            for (const Limit& limit : limits) {
                const auto start = std::chrono::steady_clock::now();
                const EndgameSolution solved = solveEndgame(position, 0.5, limit.nodes, limit.ms);
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
                std::printf("%5d %5d %8lld/%-4d %8lld %s (%.1f ms)\n", plies, which, limit.nodes,
                            limit.ms, solved.nodes, solved.solved ? "solved" : "GAVE UP", ms);
            }
        }
    }
    return 0;
}
