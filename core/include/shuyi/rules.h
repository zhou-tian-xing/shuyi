// Rule sets.  Three rules share one board, one score function and one move
// encoding; they differ only in which moves are legal and in when the game ends.
//
//   Classic   every empty cell accepts every number still in hand.
//   Blocking  a cell rejects number n when one of its four orthogonal neighbours
//             holds an *opposing* stone of number >= n.  "Nearby" is the same
//             neighbourhood the score uses (the cell plus its four orthogonal
//             neighbours), so there is exactly one geometric primitive in the
//             game.  A player with no legal placement is forced to pass; if both
//             sides pass in a row the game ends early.
//   NoPass    Blocking's placement rule with no pass at all: the moment the side
//             to move has no legal placement, the game ends and is scored.  The
//             board can therefore finish with fewer than 54 stones, and a
//             player who starves the opponent has ended the game rather than
//             bought a tempo.
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
    NoPass = 2,
};

constexpr Ruleset kDefaultRuleset = Ruleset::NoPass;
constexpr int kRulesetCount = 3;

inline const char* rulesetName(Ruleset ruleset) {
    switch (ruleset) {
    case Ruleset::Blocking: return "blocking";
    case Ruleset::NoPass: return "nopass";
    default: return "classic";
    }
}

// Short tag used in file names: b4c32-...-blk.bin vs ...-cls.bin vs ...-nps.bin.
inline const char* rulesetTag(Ruleset ruleset) {
    switch (ruleset) {
    case Ruleset::Blocking: return "blk";
    case Ruleset::NoPass: return "nps";
    default: return "cls";
    }
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
    if (text == "nopass" || text == "nps" || text == "2") {
        out = Ruleset::NoPass;
        return true;
    }
    return false;
}

// Does this rule set block a number by the neighbouring enemy?  Classic is the
// only free-placement rule; both later rule sets share the blocking geometry.
inline bool rulesetBlocks(Ruleset ruleset) { return ruleset != Ruleset::Classic; }

// Can the side to move be forced to pass?  Only NoPass removes the pass.
inline bool rulesetHasPass(Ruleset ruleset) { return ruleset != Ruleset::NoPass; }

// May a model trained for `model` be used to play `game`?
//
// Equal rule sets always match.  On top of that, a blocking-trained model may
// play NoPass: every NoPass game is a legal blocking game up to its earlier end,
// so the value head has seen those positions.  The reverse is not true -- a
// NoPass model has never learned the continuation after a pass -- and Classic
// models share nothing with the blocking geometry, so both are refused.
inline bool rulesetCompatible(Ruleset model, Ruleset game) {
    if (model == game) return true;
    return model == Ruleset::Blocking && game == Ruleset::NoPass;
}

// A pass is a forced turn when the side to move has no legal placement.  It is
// encoded as a Move with number 10 so it can live in the same history list; no
// real move can collide with it (number 10 is outside 1..9 and, unlike 0, is not
// what a default-constructed Move holds, so unset moves stay distinguishable).
constexpr std::uint8_t kPassCell = 0;
constexpr std::uint8_t kPassNumber = 10;

} // namespace shuyi
