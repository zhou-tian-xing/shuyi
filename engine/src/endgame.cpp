#include "shuyi/endgame.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace shuyi {
namespace {

using Clock = std::chrono::steady_clock;

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Score of the finished game from the side to move.  Same convention as the
// search: `finalScore` is Red-positive, so Black negates it.
double terminalScore(const Position& position, double aValue) {
    const double total = position.finalScore(aValue);
    return position.sideToMove() == Player::Red ? total : -total;
}

struct Budget {
    long long nodes = 0;
    long long limit = 0;
    // 0 = no clock.  Checked every kClockEvery nodes: a steady_clock read is
    // cheap but not free, and the solve is the only thing on this thread.
    Clock::time_point deadline{};
    bool hasDeadline = false;

    bool exhausted() const { return nodes > limit; }
    bool expired() const { return hasDeadline && Clock::now() >= deadline; }
    bool stop() const { return exhausted() || expired(); }
};

constexpr long long kClockEvery = 2048;

// Negamax on the score, which is what the game maximises: a win is a positive
// score, a draw is zero, and among wins the larger score is the better one.  That
// makes the value ordering total, so alpha-beta prunes correctly without a
// separate win/draw/loss layer.
//
// Play/undo, not "copy the position": a Position carries its history vector, so a
// copy allocates on every node.  Measured on the laptop, that alone was ~600 k
// nodes/s; the same tree walked with play/undo runs several times faster, which is
// the difference between a 0.4 s stall and a 0.05 s one.
double negamax(Position& position, double aValue, double alpha, double beta, Budget& budget) {
    if (position.gameOver()) return terminalScore(position, aValue);
    if (++budget.nodes > budget.limit) return 0.0;
    if (budget.hasDeadline && (budget.nodes % kClockEvery) == 0 && budget.expired()) return 0.0;
    const std::vector<Move> moves = position.searchMoves();
    if (moves.empty()) return terminalScore(position, aValue);
    double best = -kInfinity;
    for (const Move& move : moves) {
        position.play(move);
        const double value = -negamax(position, aValue, -beta, -alpha, budget);
        position.undo();
        if (budget.stop()) return 0.0;
        best = std::max(best, value);
        alpha = std::max(alpha, best);
        if (alpha >= beta) break;  // the opponent has a better option elsewhere
    }
    return best;
}

} // namespace

int pliesRemaining(const Position& position) {
    return std::max(0, kTotalMoves - position.stones());
}

EndgameSolution solveEndgame(const Position& position, double aValue, long long nodeBudget,
                             int timeLimitMs) {
    EndgameSolution solution;
    Budget budget;
    budget.limit = nodeBudget;
    if (timeLimitMs > 0) {
        budget.hasDeadline = true;
        budget.deadline = Clock::now() + std::chrono::milliseconds(timeLimitMs);
    }
    if (position.gameOver()) {
        solution.solved = true;
        solution.score = terminalScore(position, aValue);
        solution.moves.clear();
        return solution;
    }

    Position work = position;
    const std::vector<Move> moves = work.searchMoves();
    if (moves.empty()) {
        solution.solved = true;
        solution.score = terminalScore(work, aValue);
        return solution;
    }

    double best = -kInfinity;
    Move bestMove = moves.front();
    solution.moves.reserve(moves.size());
    for (const Move& move : moves) {
        work.play(move);
        // Full window per root move: every root move needs its own exact score,
        // not just the best one, because the board shows them all.
        const double value = -negamax(work, aValue, -kInfinity, kInfinity, budget);
        work.undo();
        if (budget.stop()) {
            solution.moves.clear();
            // Still report the work spent: a caller comparing budgets needs it.
            solution.nodes = budget.nodes;
            return solution;  // too many plies left: let the caller search
        }
        solution.moves.push_back(EndgameMove{move, value});
        if (value > best) {
            best = value;
            bestMove = move;
        }
    }
    solution.solved = true;
    solution.score = best;
    solution.best = bestMove;
    solution.nodes = budget.nodes;
    return solution;
}

} // namespace shuyi
