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
// A is passed through: a model that cannot see it says so via `aInput`.
struct NnOverride {
    std::function<NnOutput(const Position&, double aValue)> evaluate;
    std::function<bool()> hasScoreHead;
    std::function<bool()> aInput;
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
    // Root width ceiling when searching against a clock (a visit budget sizes the
    // root itself: one network forward per root move, so a 512-visit search opens
    // 32 of them and a 35k-visit analysis opens every symmetry orbit).
    int mctsRootChildren = 64;
    // With this many plies left the tree stops pruning its width: exact terminal
    // results are worth more than any network estimate, and a pruned tree may
    // never reach them.  0 disables.
    int mctsFullWidthPlies = 6;
    // With this many plies left the last plies are solved exactly instead of
    // searched.  A proof publishes a margin only when every reply is solved, and
    // it stops at the first winning line, so a search would report "solved, no
    // margin" in exactly the phase where the margin is the interesting number.
    // Measured on a 3080 laptop CPU: 3-6 stones left costs 1-34 ms (the hands are
    // nearly empty, so the branching is small); 7 costs seconds.  0 disables.
    int exactEndgamePlies = 6;
    // After the search, one more ply is still worth solving outright.  Measured
    // with tools/research/endgame_cost.cpp: a 7-ply position needs 0.5-1.8 M nodes
    // (1-4 s) with alpha-beta, so a few thousand visits can never prove it, while
    // the solver usually can; 8 plies already needs 10 M nodes and 15-25 s, so the
    // second gate stops at 7.  0 disables the second-chance solve.
    int exactEndgameAfterPlies = 7;
    long long exactEndgameAfterNodes = 2000000;
    int aspirationWindow = 24;
    bool usePvs = true;
    // There is deliberately no tree-size cap: a cap silently changed how leaves
    // were scored once it was reached (a stale build reported a hard 100% win at
    // ~35k visits that way).  The tree grows until the visit budget is spent, and
    // the only stop conditions are the budget, the clock and `stop`.
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
    // averageScore is a real number: the exact margin of a solved line, or an
    // average over visits that carried a terminal/net score.  A false here means
    // "this cell has no score", which the front end shows as "--".  It is not the
    // same as 0.0: a position lost by 18 points used to report 0.0 because no
    // visit carried a score, and the board then read like an even game.
    bool hasScore = false;
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
    // How often the search publishes a snapshot, in wall-clock milliseconds.
    // This is a reporting rate, never a stopping condition: the search ends on
    // the visit budget or on `stop`, so a slow machine streams fewer updates
    // instead of searching less.
    int reportIntervalMs = 100;
    const std::atomic<bool>* stop = nullptr;
    std::function<void(const Analysis&)> onUpdate;
};

class Engine {
public:
    virtual ~Engine() = default;
    virtual EngineKind kind() const = 0;
    // False with a reason when this engine cannot run as configured (a network
    // engine without a loadable model).  Callers must refuse the command instead
    // of silently searching with an empty prior.
    virtual bool ready(std::string* error) const {
        if (error) error->clear();
        return true;
    }
    virtual void reset() {}
    virtual void onPlay(const Position&, Move) {}
    // Name of the loaded network, empty for an engine without one.
    virtual std::string modelName() const { return {}; }
    virtual std::vector<PolicyVisit> rootPolicy() const { return {}; }
    virtual Analysis search(const Position& position, const SearchRequest& request) = 0;

    Move genmove(const Position& position, const SearchRequest& request);
};

std::unique_ptr<Engine> makeEngine(EngineKind kind, const EngineOptions& options);

} // namespace shuyi
