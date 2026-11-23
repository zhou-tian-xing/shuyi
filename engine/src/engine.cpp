#include "shuyi/engine.h"

#include "shuyi/ai.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace shuyi {
namespace {

int sideTotal(const Position& position, double aValue) {
    const double total = position.rawScore() + (position.sideToMove() == Player::Red ? aValue : -aValue);
    return static_cast<int>(std::lround(total));
}

double scoreWinrate(int score) {
    const double x = static_cast<double>(score) / 4.0;
    return 1.0 / (1.0 + std::exp(-x));
}

class RandomEngine final : public Engine {
public:
    explicit RandomEngine(const EngineOptions& options) : options_(options), rng_(options.seed) {}

    EngineKind kind() const override { return EngineKind::Random; }
    void reset() override {}

    Analysis search(const Position& position, const SearchRequest& request) override {
        Analysis analysis;
        analysis.kind = EngineKind::Random;
        const auto moves = position.searchMoves();
        if (moves.empty()) return analysis;
        std::uniform_int_distribution<std::size_t> pick(0, moves.size() - 1);
        analysis.best = moves[pick(rng_)];
        analysis.valid = true;
        analysis.nodes = 1;
        analysis.score = sideTotal(position, options_.aValue);
        analysis.winrate = scoreWinrate(analysis.score);
        analysis.pv = squareName(analysis.best.cell) + ":" + std::to_string(analysis.best.number);
        AnalysisCandidate candidate;
        candidate.move = analysis.best;
        candidate.winrate = analysis.winrate;
        candidate.score = analysis.score;
        candidate.visits = 1;
        analysis.candidates.push_back(candidate);
        if (request.onUpdate) request.onUpdate(analysis);
        return analysis;
    }

private:
    EngineOptions options_;
    std::mt19937_64 rng_;
};

class PruningEngine final : public Engine {
public:
    explicit PruningEngine(const EngineOptions& options)
        : options_(options), table_(static_cast<std::size_t>(std::max(1, options.ttSizeMb))) {}

    EngineKind kind() const override { return EngineKind::Pruning; }

    Analysis search(const Position& position, const SearchRequest& request) override {
        SearchOptions search;
        // A visit budget means "N position evaluations" for alpha-beta too: the
        // node limit is machine independent, the wall clock is not.
        if (request.visits > 0) {
            search.timeLimitMs = 0;
            search.nodeLimit = request.visits;
        } else {
            search.timeLimitMs = request.timeLimitMs;
        }
        search.maxDepth = options_.maxDepth;
        search.seed = options_.seed;
        search.useTt = true;
        search.ttSizeMb = options_.ttSizeMb;
        search.stop = request.stop;
        search.reportIntervalMs = request.reportIntervalMs;
        search.aspirationWindow = options_.aspirationWindow;
        search.usePvs = options_.usePvs;

        const auto convert = [&](const SearchProgress& progress) {
            Analysis analysis;
            analysis.kind = EngineKind::Pruning;
            // Alpha-beta scores are static evaluations in points, so the score
            // is a real number rather than a win-rate guess.
            analysis.hasScore = true;
            analysis.valid = progress.best.valid();
            analysis.best = progress.best;
            analysis.depth = progress.depth;
            analysis.nodes = progress.nodes;
            analysis.window = progress.window;
            analysis.windowResearches = progress.windowResearches;
            analysis.pvsResearches = progress.pvsResearches;
            analysis.pv = progress.pv;
            analysis.score = progress.score + static_cast<int>(std::lround(
                                                 position.sideToMove() == Player::Red
                                                     ? options_.aValue
                                                     : -options_.aValue));
            analysis.winrate = scoreWinrate(analysis.score);
            if (analysis.valid) {
                AnalysisCandidate candidate;
                candidate.move = analysis.best;
                candidate.winrate = analysis.winrate;
                candidate.score = analysis.score;
                candidate.averageScore = analysis.score;
                candidate.visits = 1;
                analysis.candidates.push_back(candidate);
                CellAnalysis cell;
                cell.cell = analysis.best.cell;
                cell.move = analysis.best;
                cell.winrate = analysis.winrate;
                cell.averageScore = analysis.score;
                cell.visits = 1;
                cell.hasData = true;
                analysis.cells.push_back(cell);
            }
            if (request.onUpdate) request.onUpdate(analysis);
            return analysis;
        };

        search.onProgress = [&](const SearchProgress& progress) { convert(progress); };
        const SearchResult result = searchAlphaBeta(position, search, table_);
        Analysis analysis;
        analysis.kind = EngineKind::Pruning;
        analysis.valid = result.move.valid();
        analysis.best = result.move;
        analysis.depth = result.depth;
        analysis.nodes = result.nodes;
        analysis.window = result.window;
        analysis.windowResearches = result.windowResearches;
        analysis.pvsResearches = result.pvsResearches;
        analysis.pv = result.pv;
        analysis.score = result.value + static_cast<int>(std::lround(
                                          position.sideToMove() == Player::Red ? options_.aValue
                                                                               : -options_.aValue));
        analysis.winrate = scoreWinrate(analysis.score);
        if (analysis.valid) {
            AnalysisCandidate candidate;
            candidate.move = analysis.best;
            candidate.winrate = analysis.winrate;
            candidate.score = analysis.score;
            candidate.averageScore = analysis.score;
            candidate.visits = 1;
            analysis.candidates.push_back(candidate);
            CellAnalysis cell;
            cell.cell = analysis.best.cell;
            cell.move = analysis.best;
            cell.winrate = analysis.winrate;
            cell.averageScore = analysis.score;
            cell.visits = 1;
            cell.hasData = true;
            analysis.cells.push_back(cell);
        }
        if (request.onUpdate) request.onUpdate(analysis);
        return analysis;
    }

private:
    EngineOptions options_;
    TranspositionTable table_;
};

} // namespace

const char* engineName(EngineKind kind) {
    switch (kind) {
    case EngineKind::Random: return "random";
    case EngineKind::Pruning: return "pruning";
    case EngineKind::Mcts: return "mcts";
    case EngineKind::MctsNn: return "mcts_nn";
    }
    return "unknown";
}

bool parseEngineName(const std::string& name, EngineKind& kind) {
    if (name == "random") kind = EngineKind::Random;
    else if (name == "pruning" || name == "alphabeta") kind = EngineKind::Pruning;
    else if (name == "mcts") kind = EngineKind::Mcts;
    else if (name == "mcts_nn" || name == "mcts-nn") kind = EngineKind::MctsNn;
    else return false;
    return true;
}

Move Engine::genmove(const Position& position, const SearchRequest& request) {
    const Analysis analysis = search(position, request);
    return analysis.best;
}

std::unique_ptr<Engine> makeMctsEngine(const EngineOptions& options, bool useNn);
std::unique_ptr<Engine> makeMctsNnEngine(const EngineOptions& options);

std::unique_ptr<Engine> makeEngine(EngineKind kind, const EngineOptions& options) {
    switch (kind) {
    case EngineKind::Random: return std::make_unique<RandomEngine>(options);
    case EngineKind::Pruning: return std::make_unique<PruningEngine>(options);
    case EngineKind::Mcts: return makeMctsEngine(options, false);
    case EngineKind::MctsNn: return makeMctsNnEngine(options);
    }
    return nullptr;
}

} // namespace shuyi
