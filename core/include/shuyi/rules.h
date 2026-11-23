// Rule sets.  Two rules share one board, one score function and one move
// encoding; they differ only in which moves are legal.
//
//   Classic   every empty cell accepts every number still in hand.
//   Blocking  a cell rejects number n when one of its four orthogonal neighbours
//             holds an *opposing* stone of number >= n.  "Nearby" is the same
//             neighbourhood the score uses (the cell plus its four orthogonal
//             neighbours), so there is exactly one geometric primitive in the
//             game.
//
// The blocking rule makes a stone a claim: a red 9 permanently denies Black the
// five cells of its cross, so the order in which numbers are spent becomes part
// of the strategy.  It also cuts the branching factor by ~30% mid-game and ~70%
// late (measured by training/rule_probe.py), which is why it is worth a second
// search engine rather than a flag inside the old one.
//
// Everything rule-specific must stay on one side of this header: the engine
// dispatches through Ruleset, it never forks on a free-form bool.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace shuyi {

enum class Ruleset : std::uint8_t {
    Classic = 0,
    Blocking = 1,
};

constexpr Ruleset kDefaultRuleset = Ruleset::Classic;
constexpr int kRulesetCount = 2;

inline const char* rulesetName(Ruleset ruleset) {
    return ruleset == Ruleset::Blocking ? "blocking" : "classic";
}

// Short tag used in file names: b4c32-...-blk.bin vs ...-cls.bin.
inline const char* rulesetTag(Ruleset ruleset) {
    return ruleset == Ruleset::Blocking ? "blk" : "cls";
}

inline bool parseRuleset(std::string_view text, Ruleset& out) {
    if (text == "classic" || text == "cls" || text == "0") {
        out = Ruleset::Classic;
        return true;
    }
    if (text == "blocking" || text == "blk" || text == "1") {
        out = Ruleset::Blocking;
        return true;
    }
    return false;
}

// A pass is a forced turn when the side to move has no legal placement.  It is
// encoded as a Move with number 0 so it can live in the same history list; no
// real move can collide with it.
// number 10 is outside the legal range 1..9 and, unlike 0, is not what a
// default-constructed Move holds (unset moves must stay distinguishable).
constexpr std::uint8_t kPassCell = 0;
constexpr std::uint8_t kPassNumber = 10;

} // namespace shuyi
