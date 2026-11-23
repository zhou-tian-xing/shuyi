// Soundness check for the MCTS solver: whenever the engine claims a position is
// "proven", an independent exact search must agree on the result and the score.
#include "shuyi/engine.h"
#include "shuyi/position.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

using namespace shuyi;

static long long nodes = 0;
static bool aborted = false;
static const long long kNodeCap = 3000000LL;

// Exact (win value, score) for the side to move at `position`, from the mover's
// point of view.  Aborts past kNodeCap and sets `aborted`.
static std::pair<double, double> solve(const Position& position) {
    if (++nodes > kNodeCap) {
        aborted = true;
        return {0.5, 0.0};
    }
    const Player mover = position.sideToMove();
    if (position.gameOver()) {
        const double total = position.finalScore(1.0);
        double value = 0.5;
        if (total > 0.0) value = mover == Player::Red ? 1.0 : 0.0;
        else if (total < 0.0) value = mover == Player::Black ? 1.0 : 0.0;
        const double score = mover == Player::Red ? total : -total;
        return {value, score};
    }
    double bestValue = -1.0;
    double bestScore = 0.0;
    for (Move move : position.searchMoves()) {
        Position next = position;
        next.play(move);
        const std::pair<double, double> child = solve(next);
        if (aborted) return {0.5, 0.0};
        const double value = 1.0 - child.first;
        const double score = -child.second;
        if (value > bestValue + 1e-9 || (std::fabs(value - bestValue) <= 1e-9 && score > bestScore)) {
            bestValue = value;
            bestScore = score;
        }
    }
    return {bestValue, bestScore};
}

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "models/shuyi_resnet_v1.bin";
    const int games = argc > 2 ? std::atoi(argv[2]) : 60;
    const int visits = argc > 3 ? std::atoi(argv[3]) : 64;

    int proven = 0;
    int checked = 0;
    int skipped = 0;
    int mismatches = 0;

    for (int game = 0; game < games; ++game) {
        Position position;
        position.setRuleset(Ruleset::Blocking);
        std::uint64_t seed = 2026 + static_cast<std::uint64_t>(game) * 7717;
        const int targetPly = 49 + (game % 5);
        while (position.ply() < targetPly && !position.gameOver()) {
            std::vector<Move> moves = position.searchMoves();
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            position.play(moves[seed % moves.size()]);
        }
        if (position.gameOver()) continue;

        EngineOptions options;
        options.nnModelPath = model;
        options.nnValueBlend = 1.0;
        options.aValue = 1.0;
        options.seed = 500 + static_cast<std::uint64_t>(game);
        auto engine = makeEngine(EngineKind::MctsNn, options);
        SearchRequest request;
        request.timeLimitMs = 0;
        request.visits = visits;
        const Analysis analysis = engine->search(position, request);
        if (!analysis.proven) continue;
        ++proven;

        nodes = 0;
        aborted = false;
        const std::pair<double, double> exact = solve(position);
        if (aborted) {
            ++skipped;
            continue;
        }
        ++checked;
        // The win rate must be the game-theoretic value of the position.  The
        // reported margin belongs to the move the engine would play, so it is
        // compared against that move's exact margin.
        Position chosen = position;
        chosen.play(analysis.best);
        const std::pair<double, double> chosenExact = solve(chosen);
        const double chosenMargin = -chosenExact.second;
        const double valueDelta = std::fabs(analysis.winrate - exact.first);
        const double scoreDelta = (analysis.proven && analysis.hasScore)
                                      ? std::fabs(analysis.averageScore - chosenMargin)
                                      : 0.0;
        if (valueDelta > 1e-9 || scoreDelta > 1e-6) {
            ++mismatches;
            std::printf("MISMATCH game=%d ply=%d engine=%.3f/%+.2f exact=%.3f/%+.2f best=%s%d\n",
                        game, position.ply(), analysis.winrate, analysis.averageScore, exact.first,
                        exact.second, squareName(analysis.best.cell).c_str(),
                        static_cast<int>(analysis.best.number));
            // What does the engine believe per move, and what is the truth?
            for (const AnalysisCandidate& candidate : analysis.candidates) {
                Position next = position;
                next.play(candidate.move);
                nodes = 0;
                aborted = false;
                const std::pair<double, double> child = solve(next);
                std::printf("    move %s%d engine=%.3f/%+.2f exact=%.3f/%+.2f visits=%d%s\n",
                            squareName(candidate.move.cell).c_str(),
                            static_cast<int>(candidate.move.number), candidate.winrate,
                            candidate.averageScore, 1.0 - child.first, -child.second,
                            candidate.visits, aborted ? " (exact aborted)" : "");
            }
        }
    }
    std::printf("proven=%d checked=%d skipped(too big)=%d mismatches=%d\n", proven, checked, skipped,
                mismatches);
    return mismatches == 0 ? 0 : 1;
}
