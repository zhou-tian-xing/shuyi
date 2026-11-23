#include "shuyi/ai.h"
#include "shuyi/ai_policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace shuyi {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kInfinity = 1000000;

int sideValue(const Position& position) {
    const int raw = position.rawScore();
    return position.sideToMove() == Player::Red ? raw : -raw;
}

int handAwareEvaluationImpl(const Position& position, double handWeight) {
    int stable = 0;
    int unstable = 0;
    for (int cell = 0; cell < kCells; ++cell) {
        const int x = xOf(cell);
        const int y = yOf(cell);
        bool settled = !position.emptyMask().test(cell);
        auto check = [&](int xx, int yy) {
            if (!onBoard(xx, yy)) return;
            if (position.emptyMask().test(index(xx, yy))) settled = false;
        };
        check(x - 1, y);
        check(x + 1, y);
        check(x, y - 1);
        check(x, y + 1);
        if (settled) stable += position.scoreCell(cell);
        else unstable += position.scoreCell(cell);
    }

    double hand = 0.0;
    for (int number = 1; number <= 9; ++number) {
        const int diff = position.remaining(Player::Red, number) -
                         position.remaining(Player::Black, number);
        hand += static_cast<double>(diff) * (static_cast<double>(number) - 4.5);
    }
    int empty = 0;
    for (int cell = 0; cell < kCells; ++cell) empty += position.emptyMask().test(cell) ? 1 : 0;
    const double open = 0.5 + static_cast<double>(empty) / 81.0;
    const int handTerm = static_cast<int>(std::lround(hand * handWeight * open));
    return stable + static_cast<int>(std::lround(unstable * 0.55)) + handTerm;
}

} // namespace

int evaluateHandAware(const Position& position, double handWeight) {
    return handAwareEvaluationImpl(position, handWeight);
}

int classicOrderBonus(const Position& before, Move move, int immediateDelta) {
    (void)before;
    (void)move;
    return immediateDelta;
}

const SearchPolicy kClassicSearchPolicy{"classic", &evaluateHandAware, &classicOrderBonus};

const SearchPolicy& policyFor(Ruleset ruleset) {
    return ruleset == Ruleset::Blocking ? kBlockingSearchPolicy : kClassicSearchPolicy;
}

namespace {

struct RankedMove {
    Move move;
    int order;
    int tie;
};

std::string moveText(Move move) {
    if (!move.valid()) return "?";
    return squareName(move.cell) + ":" + std::to_string(move.number);
}

class Searcher {
public:
    Searcher(const SearchOptions& options, TranspositionTable& table)
        : options_(options), table_(table), rng_(options.seed), started_(Clock::now()) {}

    SearchResult run(const Position& root) {
        SearchResult result;
        policy_ = &policyFor(root.ruleset());
        root_ = &root;
        if (root.gameOver()) return result;

        const std::vector<Move> legal = root.searchMoves();
        if (legal.empty()) return result;
        result.move = legal.front();
        result.completed = false;
        if (options_.useTt) table_.newSearch();

        // A cheap one-ply answer is always available, even with a tiny time
        // budget or an immediate abort.
        Position scratch = root;
        Position orderRoot = root;
        int bestQuick = std::numeric_limits<int>::min();
        for (Move move : orderMoves(orderRoot)) {
            if (outOfTime()) break;
            scratch.play(move);
            const int value = -sideValue(scratch);
            scratch.undo();
            if (value > bestQuick) {
                bestQuick = value;
                result.move = move;
                result.value = value;
            }
        }
        currentBest_ = result.move;
        currentValue_ = result.value;

        const int remainingMoves = kTotalMoves - root.ply();
        const int hardDepth = std::min(options_.maxDepth, remainingMoves);
        Move previousBest = result.move;
        int lastCompletedScore = result.value;
        bool haveCompleted = false;

        // Align leaves with red-to-move positions (black has just finished a
        // round).  Root red searches 2,4,6... plies; root black searches
        // 1,3,5... plies.  This is the cheap alternative to expanding every
        // black reply in a quiescence search.
        const int startDepth = options_.extendToBlack ? 1 : (root.sideToMove() == Player::Red ? 2 : 1);
        const int depthStep = options_.extendToBlack ? 1 : 2;
        for (int depth = startDepth; depth <= hardDepth; depth += depthStep) {
            if (outOfTime()) break;
            Position position = root;
            std::vector<Move> rootMoves = orderMoves(position, previousBest);
            if (options_.rootBranching > 0 &&
                static_cast<int>(rootMoves.size()) > options_.rootBranching) {
                rootMoves.resize(options_.rootBranching);
            }

            // The first completed iteration gives us a score to centre the
            // next aspiration window on.  A window of 0 means a full window.
            int window = (haveCompleted && options_.aspirationWindow > 0)
                             ? options_.aspirationWindow
                             : 0;
            bool accepted = false;
            for (int attempt = 0; attempt < 12; ++attempt) {
                const int alpha = window > 0 ? std::max(-kInfinity, lastCompletedScore - window)
                                             : -kInfinity;
                const int beta = window > 0 ? std::min(kInfinity, lastCompletedScore + window)
                                            : kInfinity;
                currentDepth_ = depth;
                currentWindow_ = window;
                Move iterationBest{};
                int iterationValue = -kInfinity;
                bool complete = true;
                bool failLow = false;
                bool failHigh = false;
                searchRoot(position, rootMoves, depth, alpha, beta, iterationBest, iterationValue,
                           complete, failLow, failHigh);
                if (!complete || aborted_) break;
                if (window > 0 && (failLow || failHigh)) {
                    ++windowResearches_;
                    // A fail-low/fail-high means the true score is outside the
                    // window.  Widen both sides and search again; pin the best
                    // move found so far at the front of the next attempt.
                    if (iterationBest.valid()) {
                        previousBest = iterationBest;
                        const auto found = std::find(rootMoves.begin(), rootMoves.end(), iterationBest);
                        if (found != rootMoves.end()) {
                            std::rotate(rootMoves.begin(), found, found + 1);
                        }
                    }
                    window = std::min(512, window * 2);
                    if (outOfTime()) break;
                    continue;
                }

                // Only an exact, fully searched iteration may replace the
                // previous depth's result.
                result.move = iterationBest;
                result.value = iterationValue;
                result.depth = depth;
                result.window = window;
                result.completed = true;
                result.pv = extractPv(root, iterationBest);
                currentBest_ = iterationBest;
                currentValue_ = iterationValue;
                previousBest = iterationBest;
                lastCompletedScore = iterationValue;
                haveCompleted = true;
                accepted = true;
                maybeReport(true);
                break;
            }
            if (!accepted) break;
            if (depth >= remainingMoves) break;
        }
        result.nodes = nodes_;
        result.windowResearches = windowResearches_;
        result.pvsResearches = pvsResearches_;
        if (result.pv.empty()) result.pv = extractPv(root, result.move);
        return result;
    }

private:
    void searchRoot(Position& position, const std::vector<Move>& moves, int depth, int alpha,
                    int beta, Move& bestMove, int& bestValue, bool& complete, bool& failLow,
                    bool& failHigh) {
        bestValue = -kInfinity;
        const int alphaStart = alpha;
        int rootAlpha = alpha;
        failLow = false;
        failHigh = false;
        for (std::size_t i = 0; i < moves.size(); ++i) {
            if (outOfTime()) {
                complete = false;
                return;
            }
            const Move move = moves[i];
            position.play(move);
            ++nodes_;
            int value = 0;
            if (i == 0) {
                value = -negamax(position, depth - 1, -beta, -rootAlpha);
            } else if (options_.usePvs && rootAlpha > -kInfinity && beta > rootAlpha + 1) {
                // Scout with a null window first; re-search with the real
                // window only when the move actually improves alpha.
                value = -negamax(position, depth - 1, -rootAlpha - 1, -rootAlpha);
                if (!aborted_ && value > rootAlpha && value < beta) {
                    ++pvsResearches_;
                    value = -negamax(position, depth - 1, -beta, -rootAlpha);
                }
            } else {
                value = -negamax(position, depth - 1, -beta, -rootAlpha);
            }
            position.undo();
            if (aborted_) {
                complete = false;
                return;
            }
            if (value > bestValue) {
                bestValue = value;
                bestMove = move;
                currentBest_ = move;
                currentValue_ = value;
            }
            if (value > rootAlpha) rootAlpha = value;
            if (rootAlpha >= beta) {
                failHigh = true;
                break;
            }
            maybeReport(false);
        }
        failLow = bestValue <= alphaStart;
        if (bestValue == -kInfinity) complete = false;
    }
    // The game has an even number of plies and black makes the last move.
    // Evaluating immediately after a red move would compare a "red just
    // moved" position with a "black just moved" one.  At a depth-zero node
    // with black to move we therefore play one black move (quiescence) and
    // evaluate only the resulting red-to-move positions.
    int evaluateForSide(const Position& position) const {
        const int redPerspective = policy_->evaluate(position, options_.handWeight);
        return position.sideToMove() == Player::Red ? redPerspective : -redPerspective;
    }

    int blackQuiescence(Position& position, int alpha, int beta) {
        int best = -kInfinity;
        for (Move move : orderMoves(position)) {
            if (outOfTime()) return 0;
            position.play(move);
            ++nodes_;
            int value;
            if (position.gameOver()) {
                value = -sideValue(position);
            } else {
                value = -evaluateForSide(position);
            }
            position.undo();
            if (aborted_) return 0;
            best = std::max(best, value);
            alpha = std::max(alpha, value);
            if (alpha >= beta) break;
        }
        return best == -kInfinity ? evaluateForSide(position) : best;
    }

    int negamax(Position& position, int depth, int alpha, int beta) {
        if (position.gameOver()) return sideValue(position);
        if (depth <= 0) {
            if (options_.extendToBlack && position.sideToMove() == Player::Black) {
                return blackQuiescence(position, alpha, beta);
            }
            return evaluateForSide(position);
        }
        if (outOfTime()) return 0;

        const int alphaOriginal = alpha;
        const int betaOriginal = beta;
        std::uint64_t key = 0;
        Move ttMove{};
        if (options_.useTt) {
            key = position.key();
            int ttValue = 0;
            if (table_.probe(key, depth, alpha, beta, ttValue, ttMove)) return ttValue;
        }

        int best = -kInfinity;
        Move bestMove{};
        std::vector<Move> moves = orderMoves(position, ttMove);
        if (options_.maxBranching > 0 && static_cast<int>(moves.size()) > options_.maxBranching) {
            moves.resize(options_.maxBranching);
        }
        for (std::size_t i = 0; i < moves.size(); ++i) {
            const Move move = moves[i];
            position.play(move);
            ++nodes_;
            int value = 0;
            if (i == 0) {
                value = -negamax(position, depth - 1, -beta, -alpha);
            } else if (options_.usePvs && alpha > -kInfinity && beta > alpha + 1) {
                value = -negamax(position, depth - 1, -alpha - 1, -alpha);
                if (!aborted_ && value > alpha && value < beta) {
                    ++pvsResearches_;
                    value = -negamax(position, depth - 1, -beta, -alpha);
                }
            } else {
                value = -negamax(position, depth - 1, -beta, -alpha);
            }
            position.undo();
            if (aborted_) return 0;
            if (value > best) {
                best = value;
                bestMove = move;
            }
            alpha = std::max(alpha, value);
            if (alpha >= beta) break;
        }

        if (options_.useTt) {
            TTFlag flag = TTFlag::Exact;
            if (best <= alphaOriginal) flag = TTFlag::Upper;
            else if (best >= betaOriginal) flag = TTFlag::Lower;
            table_.store(key, depth, best, flag, bestMove);
        }
        return best;
    }

    // Most promising moves first: the immediate change in the mover's score,
    // with a TT/previous-iteration move pinned to the front.
    std::vector<Move> orderMoves(Position& position, Move preferred = {}) const {
        std::vector<Move> moves = position.searchMoves();
        const Player side = position.sideToMove();
        const int before = position.rawScore();
        std::vector<RankedMove> ranked;
        ranked.reserve(moves.size());
        for (Move move : moves) {
            position.play(move);
            const int delta = position.rawScore() - before;
            position.undo();
            const int fromMover = side == Player::Red ? delta : -delta;
            const int bonus = policy_ ? policy_->orderBonus(position, move, fromMover) : fromMover;
            const int tie = static_cast<int>(rng_() % 7);
            ranked.push_back({move, bonus, tie});
        }
        // A comparator must be a strict weak ordering: the earlier version
        // returned true for comp(x, x) when x was the preferred move, which is
        // undefined behaviour.  libstdc++ tolerated it, MSVC's stable_sort walked
        // off the end and killed the process (STATUS_STACK_BUFFER_OVERRUN).
        std::stable_sort(ranked.begin(), ranked.end(), [&](const RankedMove& a, const RankedMove& b) {
            const bool aPreferred = preferred.valid() && a.move == preferred;
            const bool bPreferred = preferred.valid() && b.move == preferred;
            if (aPreferred != bPreferred) return aPreferred;
            if (a.order != b.order) return a.order > b.order;
            return a.tie < b.tie;
        });
        moves.clear();
        moves.reserve(ranked.size());
        for (const auto& r : ranked) moves.push_back(r.move);
        return moves;
    }

    std::string extractPv(const Position& root, Move first) const {
        if (!first.valid()) return {};
        std::string pv;
        Position position = root;
        Move move = first;
        for (int i = 0; i < 10 && move.valid(); ++i) {
            if (!pv.empty()) pv += ",";
            pv += moveText(move);
            if (!position.play(move)) break;
            int value = 0;
            Move next{};
            if (!options_.useTt ||
                !table_.probe(position.key(), 0, -kInfinity, kInfinity, value, next)) {
                // probe() fills next on a key hit even when the bound is not
                // usable; a miss leaves it invalid.
            }
            move = next;
        }
        return pv;
    }

    void maybeReport(bool completed) {
        if (!options_.onProgress) return;
        if (!completed) {
            const auto now = Clock::now();
            const auto since = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - lastReport_).count();
            if (since < std::max(10, options_.reportIntervalMs)) return;
        }
        lastReport_ = Clock::now();
        SearchProgress progress;
        progress.depth = currentDepth_;
        progress.nodes = nodes_;
        progress.score = currentValue_;
        progress.best = currentBest_;
        progress.pv = root_ ? extractPv(*root_, currentBest_) : std::string{};
        progress.window = currentWindow_;
        progress.windowResearches = windowResearches_;
        progress.pvsResearches = pvsResearches_;
        progress.completed = completed;
        options_.onProgress(progress);
    }

    bool outOfTime() {
        if (aborted_) return true;
        if (options_.stop && options_.stop->load(std::memory_order_relaxed)) {
            aborted_ = true;
            return true;
        }
        if (options_.nodeLimit > 0) {
            // A node budget is machine independent: the search expands the same
            // nodes in the same order everywhere, so the move does not depend on
            // how fast the host is.
            if (nodes_ >= options_.nodeLimit) aborted_ = true;
            return aborted_;
        }
        if ((nodes_ & 63) != 0) return false;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - started_).count();
        if (options_.timeLimitMs > 0 && elapsed >= options_.timeLimitMs) aborted_ = true;
        return aborted_;
    }

    SearchOptions options_;
    TranspositionTable& table_;
    const SearchPolicy* policy_ = &kClassicSearchPolicy;
    const Position* root_ = nullptr;
    mutable std::mt19937_64 rng_;
    Clock::time_point started_;
    Clock::time_point lastReport_;
    long long nodes_ = 0;
    int currentDepth_ = 0;
    int currentValue_ = 0;
    int currentWindow_ = 0;
    int windowResearches_ = 0;
    int pvsResearches_ = 0;
    Move currentBest_{};
    bool aborted_ = false;
};

} // namespace

SearchResult searchAlphaBeta(const Position& position, const SearchOptions& options,
                             TranspositionTable& table) {
    Searcher searcher(options, table);
    return searcher.run(position);
}

SearchResult think(const Position& position, const SearchOptions& options) {
    TranspositionTable table(static_cast<std::size_t>(std::max(1, options.ttSizeMb)));
    return searchAlphaBeta(position, options, table);
}

Move chooseMove(const Position& position, const SearchOptions& options) {
    return think(position, options).move;
}

} // namespace shuyi
