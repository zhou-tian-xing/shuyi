// Rule-specific evaluation and move ordering for the Alpha-Beta search.
//
// The search skeleton (iterative deepening, aspiration window, PVS, TT, selective
// width) is identical for both rule sets; what changes is how a leaf is scored and
// how moves are ordered.  Keeping those two functions behind a struct means the
// blocking rule gets its own engine without forking the search, and every
// rule-specific line lives in src/ai_blocking.cpp.
#pragma once

#include "shuyi/position.h"
#include "shuyi/rules.h"

namespace shuyi {

struct SearchPolicy {
    const char* name = "classic";
    // Red-perspective score of a non-terminal position.
    int (*evaluate)(const Position& position, double handWeight) = nullptr;
    // Ordering bonus (mover perspective) for a move that produced
    // `immediateDelta` raw points.  Purely a performance knob: it must never
    // change which moves are legal.
    int (*orderBonus)(const Position& before, Move move, int immediateDelta) = nullptr;
};

const SearchPolicy& policyFor(Ruleset ruleset);

extern const SearchPolicy kClassicSearchPolicy;
extern const SearchPolicy kBlockingSearchPolicy;

} // namespace shuyi
