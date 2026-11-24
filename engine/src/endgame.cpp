#include "shuyi/endgame.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace shuyi {
namespace {

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
    bool exhausted() const { return nodes > limit; }
};

// Negamax on the score, which is what the game maximises: a win is a positive
// score, a draw is zero, and among wins the larger score is the better one.  That
// makes the value ordering total, so alpha-beta prunes correctly without a
// separate win/draw/loss layer.
double negamax(Position& position, double aValue, double alpha, double beta,
               Budget& budget) {
    if (position.gameOver()) return terminalScore(position, aValue);
    if (++budget.nodes > budget.limit) return 0.0;
    const std::vector<Move> moves = position.searchMoves();
    if (moves.empty()) return terminalScore(position, aValue);
    double best = -kInfinity;
    for (const Move& move : moves) {
        Position child = position;
        child.play(move);
        const double value = -negamax(child, aValue, -beta, -alpha, budget);
        if (budget.exhausted()) return 0.0;
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

EndgameSolution solveEndgame(const Position& position, double aValue,
                             long long nodeBudget) {
    EndgameSolution solution;
    if (position.gameOver()) {
        solution.solved = true;
        solution.score = terminalScore(position, aValue);
        solution.moves.clear();
        return solution;
    }
    const std::vector<Move> moves = position.searchMoves();
    if (moves.empty()) {
        solution.solved = true;
        solution.score = terminalScore(position, aValue);
        return solution;
    }

    Budget budget;
    budget.limit = nodeBudget;
    double best = -kInfinity;
    Move bestMove = moves.front();
    solution.moves.reserve(moves.size());
    for (const Move& move : moves) {
        Position child = position;
        child.play(move);
        // Full window per root move: every root move needs its own exact score,
        // not just the best one, because the board shows them all.
        const double value = -negamax(child, aValue, -kInfinity, kInfinity, budget);
        if (budget.exhausted()) {
            solution.moves.clear();
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
