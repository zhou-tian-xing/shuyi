#pragma once

#include "shuyi/position.h"
#include "shuyi/tt.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace shuyi {

struct SearchOptions {
    int timeLimitMs = 800;
    // Node budget for alpha-beta: one node is one position evaluation.  When it
    // is set the wall clock is ignored, so the same position produces the same
    // move (and the same node count) on any machine.  Ratings must never be a
    // benchmark of the host, so evaluation uses this, not timeLimitMs.
    int nodeLimit = 0;
    int maxDepth = 12;
    std::uint64_t seed = 0x5EED1234ULL;
    bool useTt = true;
    int ttSizeMb = 16;
    int reportIntervalMs = 100;
    bool extendToBlack = false; // optional extra black quiescence at odd leaves
    double handWeight = 0.35;  // weight of remaining-piece potential
    int rootBranching = 32;    // selective width; 0 = unlimited
    int maxBranching = 16;     // selective width below the root; 0 = unlimited
    // Iterative-deepening aspiration window in points.  The root is searched
    // around the previous iteration's score; fail-low/fail-high widens the
    // window and re-searches.  0 disables aspiration (full window).
    int aspirationWindow = 24;
    // Principal-variation search: later moves get a null window first and are
    // re-searched only when they beat alpha.
    bool usePvs = true;
    const std::atomic<bool>* stop = nullptr;
    std::function<void(const struct SearchProgress&)> onProgress;
};

struct SearchProgress {
    int depth = 0;
    long long nodes = 0;
    int score = 0; // side-to-move perspective, points
    Move best{};
    std::string pv;
    int window = 0;            // current aspiration half-width, 0 = full
    int windowResearches = 0;  // failed aspiration windows re-searched
    int pvsResearches = 0;     // null-window PVS re-searches
    bool completed = false;
};

// Static red-perspective evaluation used at depth-zero leaves.  It combines
// settled/frontier board scores with the potential of both remaining hands.
int evaluateHandAware(const Position& position, double handWeight = 0.35);

struct SearchResult {
    Move move{};
    int value = 0; // side-to-move perspective
    int depth = 0;
    long long nodes = 0;
    int window = 0;
    int windowResearches = 0;
    int pvsResearches = 0;
    bool completed = false;
    std::string pv;
};

// Time-limited iterative-deepening Alpha-Beta.  The TT is caller-owned so the
// engine can keep it across moves while think() can use a temporary one.
SearchResult searchAlphaBeta(const Position& position, const SearchOptions& options,
                             TranspositionTable& table);

// Compatibility helpers used by the existing UI/tests.  These use a temporary
// transposition table.
SearchResult think(const Position& position, const SearchOptions& options = {});
Move chooseMove(const Position& position, const SearchOptions& options = {});

} // namespace shuyi
