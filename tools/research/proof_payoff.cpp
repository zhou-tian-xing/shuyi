// Does adaptive proving pay?  Takes a position with N plies left, runs the MCTS
// engine with a visit budget, and prints whether the root came back proven, how
// long it took, and how the nodes were spent.  Run with and without the prover to
// see what the extra seconds buy.
#include "shuyi/engine.h"
#include "shuyi/position.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using namespace shuyi;

namespace {

std::uint64_t g_seed = 1;

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

int main(int argc, char** argv) {
    const int plies = argc > 1 ? std::atoi(argv[1]) : 8;
    const int visits = argc > 2 ? std::atoi(argv[2]) : 1024;
    const long long proofStart = argc > 3 ? std::atoll(argv[3]) : 100000;
    const int topMoves = argc > 4 ? std::atoi(argv[4]) : 3;
    for (int game = 0; game < 3; ++game) {
        Position position = latePosition(plies, 1000 + static_cast<std::uint64_t>(game) * 7919);
        const int left = kTotalMoves - position.stones();
        if (left != plies) continue;

        EngineOptions options;
        options.seed = 7 + static_cast<std::uint64_t>(game);
        options.proofStartNodes = proofStart;
        options.proofTopMoves = topMoves;
        auto engine = makeEngine(EngineKind::Mcts, options);
        SearchRequest request;
        request.visits = visits;
        const auto start = std::chrono::steady_clock::now();
        const Analysis analysis = engine->search(position, request);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count();
        std::printf(
            "plies=%d game=%d proof=%lld top=%d -> valid=%d proven=%d visits=%d "
            "has_score=%d score=%.1f cells=%zu  %.0f ms\n",
            left, game, proofStart, topMoves, analysis.valid ? 1 : 0,
            analysis.proven ? 1 : 0, analysis.visits, analysis.hasScore ? 1 : 0,
            analysis.averageScore, analysis.cells.size(), ms);
    }
    return 0;
}
