// Exact play for the last few plies.
//
// The MCTS proof machinery publishes a margin only when every reply is solved,
// and it stops collecting replies once a win is found.  In the last three or four
// plies that leaves the most useful number -- "by how many points" -- unpublished
// even though the answer is a finite computation: at most 36 x 27 x 18 = 17,496
// leaves with three stones left, and alpha-beta cuts that hard.
//
// So once few stones remain the engine does not search, it solves.  The result is
// exact, deterministic (the same position always gives the same number) and
// complete (every legal move gets its own margin).
#pragma once

#include <vector>

#include "shuyi/position.h"

namespace shuyi {

struct EndgameMove {
    Move move{};
    double score = 0.0;  // side-to-move perspective, includes the A value
};

struct EndgameSolution {
    bool solved = false;
    double score = 0.0;  // best achievable, side-to-move perspective
    Move best{};
    // Every legal move with its exact score.  Empty when !solved.
    std::vector<EndgameMove> moves;
    long long nodes = 0;
};

// Stones still in hand on both sides: the game cannot last longer than this many
// plies (a pass would, which is why the solver also stops on gameOver()).
int pliesRemaining(const Position& position);

// Solve `position` exactly.  Returns solved=false when the node budget runs out
// (too many plies left), so the caller can fall back to a search.
EndgameSolution solveEndgame(const Position& position, double aValue,
                             long long nodeBudget = 250000);

} // namespace shuyi
