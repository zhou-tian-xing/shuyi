#include "shuyi/engine.h"

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

} // namespace

const char* engineName(EngineKind kind) {
    switch (kind) {
    case EngineKind::Random: return "random";
    case EngineKind::Mcts: return "mcts";
    case EngineKind::MctsNn: return "mcts_nn";
    }
    return "unknown";
}

bool parseEngineName(const std::string& name, EngineKind& kind) {
    if (name == "random") kind = EngineKind::Random;
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
    case EngineKind::Mcts: return makeMctsEngine(options, false);
    case EngineKind::MctsNn: return makeMctsNnEngine(options);
    }
    return nullptr;
}

} // namespace shuyi
