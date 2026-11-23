// Diagnostic: does an exact terminal result reach the root with the right sign?
// Solves the remaining game exactly (small endgames only) and compares every
// root move against what MCTS reports.
#include "shuyi/engine.h"
#include "shuyi/position.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace shuyi;

static long long nodes = 0;
static const long long kNodeCap = 40000000LL;

// Exact value for the side to move at `position`.
static double solve(const Position& position) {
    if (++nodes > kNodeCap) return 0.5;
    if (position.gameOver()) {
        const double total = position.finalScore(1.0);
        if (total > 0.0) return position.sideToMove() == Player::Red ? 1.0 : 0.0;
        if (total < 0.0) return position.sideToMove() == Player::Black ? 1.0 : 0.0;
        return 0.5;
    }
    double best = -1.0;
    for (Move move : position.searchMoves()) {
        Position next = position;
        next.play(move);
        const double value = 1.0 - solve(next);
        if (value > best) best = value;
        if (best >= 1.0) break;
    }
    return best;
}

static double solveMove(const Position& position, Move move) {
    Position next = position;
    next.play(move);
    return 1.0 - solve(next);
}

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "models/shuyi_resnet_v1.bin";
    const int targetPly = argc > 2 ? std::atoi(argv[2]) : 52;
    const int seedStart = argc > 3 ? std::atoi(argv[3]) : 0;
    const int rulesetArg = argc > 4 ? std::atoi(argv[4]) : 0;
    const int visits = argc > 5 ? std::atoi(argv[5]) : 512;
    const int kindArg = argc > 6 ? std::atoi(argv[6]) : 3;

    Position position;
    position.setRuleset(rulesetArg == 1 ? Ruleset::Blocking : Ruleset::Classic);
    std::uint64_t seed = 12345 + seedStart;
    std::vector<Move> played;
    while (position.ply() < targetPly && !position.gameOver()) {
        std::vector<Move> moves = position.searchMoves();
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        const Move move = moves[seed % moves.size()];
        position.play(move);
        played.push_back(move);
    }
    std::printf("MOVES");
    for (const Move& move : played) {
        std::printf(" %s:%d", squareName(move.cell).c_str(), static_cast<int>(move.number));
    }
    std::printf("\n");

    const int legal = static_cast<int>(position.searchMoves().size());
    std::printf("ply=%d stones=%d passes=%d side=%s raw=%d moves=%d\n", position.ply(),
                position.stones(), position.passes(), playerName(position.sideToMove()),
                position.rawScore(), legal);

    EngineOptions options;
    options.nnModelPath = model;
    options.nnValueBlend = 1.0;
    options.aValue = 1.0;
    options.seed = 777;
    auto engine = makeEngine(static_cast<EngineKind>(kindArg), options);
    SearchRequest request;
    request.timeLimitMs = 0;
    request.visits = visits;
    const Analysis analysis = engine->search(position, request);
    std::printf("best=%s%d winrate=%.3f avgscore=%.1f visits=%d pv=%s\n",
                squareName(analysis.best.cell).c_str(), analysis.best.number, analysis.winrate,
                analysis.averageScore, analysis.visits, analysis.pv.c_str());

    std::vector<double> engineWin(kCells, -1.0);
    std::vector<int> engineVisits(kCells, 0);
    std::vector<int> engineNumber(kCells, 0);
    for (const CellAnalysis& cell : analysis.cells) {
        engineWin[cell.cell] = cell.winrate;
        engineVisits[cell.cell] = cell.visits;
        engineNumber[cell.cell] = cell.move.number;
    }

    // Exact value per cell, and per (cell, number) for the engine's pick.
    double bestExact = -1.0;
    Move bestMove{};
    Position exactRoot = position;
    nodes = 0;
    std::vector<std::pair<Move, double>> exact;
    for (Move move : exactRoot.searchMoves()) {
        exact.push_back({move, solveMove(exactRoot, move)});
    }
    std::printf("exact solve nodes=%lld time ok\n", nodes);
    for (const auto& entry : exact) {
        if (entry.second > bestExact) {
            bestExact = entry.second;
            bestMove = entry.first;
        }
    }

    int tooHigh = 0;
    int tooLow = 0;
    std::printf("%-5s %-4s %-7s %-7s %-7s %-6s\n", "cell", "num", "exact", "engine", "visits", "note");
    for (const CellAnalysis& cell : analysis.cells) {
        if (cell.visits == 0) continue;
        double cellExact = -2.0;
        double bestAtCell = -2.0;
        for (const auto& entry : exact) {
            if (entry.first.cell != cell.cell) continue;
            if (entry.first.cell == cell.cell) bestAtCell = std::max(bestAtCell, entry.second);
            if (entry.first.number == cell.move.number) cellExact = entry.second;
        }
        if (cellExact < -1.5) cellExact = bestAtCell; // number not enumerated (pass)
        const double diff = cell.winrate - cellExact;
        const char* note = "";
        if (diff > 0.26) {
            note = "TOO HIGH";
            ++tooHigh;
        } else if (diff < -0.26) {
            note = "too low";
            ++tooLow;
        }
        const bool bestLine = cellExact >= bestExact - 1e-9 || cell.cell == analysis.best.cell;
        if (true) {
            std::printf("%-5s %-4d %-7.2f %-7.2f %-7d %s\n", squareName(cell.cell).c_str(),
                        cell.move.number, cellExact, cell.winrate, cell.visits, note);
        }
    }
    std::printf("exact best: %s%d value=%.2f\n", squareName(bestMove.cell).c_str(), bestMove.number,
                bestExact);
    std::printf("engine move value=%.2f %s ; tooHigh=%d tooLow=%d\n",
                solveMove(exactRoot, analysis.best),
                solveMove(exactRoot, analysis.best) >= bestExact - 1e-9 ? "OPTIMAL" : "SUBOPTIMAL",
                tooHigh, tooLow);
    return 0;
}
