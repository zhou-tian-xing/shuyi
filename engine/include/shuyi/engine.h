#pragma once

#include "shuyi/nn.h"
#include "shuyi/position.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace shuyi {

enum class EngineKind : std::uint8_t { Random = 0, Pruning = 1, Mcts = 2, MctsNn = 3 };

const char* engineName(EngineKind kind);
bool parseEngineName(const std::string& name, EngineKind& kind);

// Optional replacement for the CPU network forward.  The line-protocol engine
// hands in a CUDA evaluator when the build has one, so the UI's MCTS+NN can run
// on the GPU instead of one CPU core.  Left empty -> PolicyValueNet on the CPU.
struct NnOverride {
    std::function<NnOutput(const Position&)> evaluate;
    std::function<bool()> hasScoreHead;
};

struct EngineOptions {
    std::uint64_t seed = 0x5EED5EED1234ULL;
    int timeLimitMs = 800;
    // Simulations per search, and the default for a front end that never asks:
    // a clock budget makes the same position mean different things on different
    // machines, and an unbounded analysis drifted into millions of visits.  Set
    // 0 explicitly (protocol: `set_visits 0`) to fall back to the clock.
    int visits = 512;
    int ttSizeMb = 16;
    int maxDepth = 12;
    int mctsMaxChildren = 48;
    int mctsRootChildren = 64;
    // With this many plies left the tree stops pruning its width: exact terminal
    // results are worth more than any network estimate, and a pruned tree may
    // never reach them.  0 disables.
    int mctsFullWidthPlies = 6;
    int mctsNnRootChildren = 16; // MCTS-NN root width cap (CPU inference)
    int aspirationWindow = 24;
    bool usePvs = true;
    std::size_t mctsMaxTreeEdges = 1500000;
    double mctsC = 1.2;
    double mctsPuct = 1.7;
    double nnPriorBlend = 0.5; // 1 = pure policy, 0 = uniform
    double nnValueBlend = 1.0; // 1 = pure value head, 0 = rollout
    double rootDirichletAlpha = 0.0; // >0 enables Dirichlet noise at the root
    double rootNoiseWeight = 0.25;
    double aValue = 1.0;
    std::string nnModelPath;
    std::shared_ptr<NnOverride> nnOverride;
};

struct AnalysisCandidate {
    Move move{};
    double winrate = 0.5;
    int visits = 0;
    int score = 0;
    double averageScore = 0.0; // side-to-move perspective, terminal points
};

struct CellAnalysis {
    int cell = 0;
    Move move{};         // recommended number at this cell
    double winrate = 0.5;
    int visits = 0;
    double averageScore = 0.0;
    bool hasData = false;
    // This move's subtree is solved, so winrate and averageScore are exact
    // rather than sampled.  The UI trusts proven cells even with a low visit
    // count, which a confidence bound could never do.
    bool proven = false;
};

struct PolicyVisit {
    Move move{};
    int visits = 0;
    double prior = 0.0;
};

struct Analysis {
    EngineKind kind = EngineKind::Random;
    // True only when averageScore is a real number: an exact terminal score or a
    // trained score head.  There is no "score from win rate" fallback any more,
    // so when this is false the UI shows "--" and no score is transmitted.
    bool hasScore = false;
    bool valid = false;
    // The search solved the position: win rate, score and best move are exact
    // game-theoretic values rather than estimates, and the visit budget was
    // deliberately left partly unspent.
    bool proven = false;
    Move best{};
    int depth = 0;
    long long nodes = 0;
    int window = 0;
    int windowResearches = 0;
    int pvsResearches = 0;
    int visits = 0;
    double winrate = 0.5;
    int score = 0; // side-to-move perspective, includes the A value
    double averageScore = 0.0;
    std::string pv;
    std::vector<AnalysisCandidate> candidates;
    std::vector<CellAnalysis> cells;
};

struct SearchRequest {
    int timeLimitMs = 800;
    // When visits > 0 the search stops after that many simulations and the time
    // limit is ignored.  A visit budget is machine independent, so a "10 ms"
    // run can no longer silently turn into an 11-simulation run on a slower CPU.
    int visits = 0;
    const std::atomic<bool>* stop = nullptr;
    std::function<void(const Analysis&)> onUpdate;
};

class Engine {
public:
    virtual ~Engine() = default;
    virtual EngineKind kind() const = 0;
    virtual void reset() {}
    virtual void onPlay(const Position&, Move) {}
    virtual std::string infoTag() const { return {}; }
    virtual std::vector<PolicyVisit> rootPolicy() const { return {}; }
    virtual Analysis search(const Position& position, const SearchRequest& request) = 0;

    Move genmove(const Position& position, const SearchRequest& request);
};

std::unique_ptr<Engine> makeEngine(EngineKind kind, const EngineOptions& options);

} // namespace shuyi
