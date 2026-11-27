// The game record: a tree of moves, with the analysis of every position kept.
//
// The window shows one position at a time, so a plain move list would be enough
// to *play*.  What it cannot do is keep what the engine said about a position
// after the user moves on -- and that is the point of an analysis front end: go
// back two moves, look at the numbers, play something else, come back and still
// find them.
//
// So the record is a tree:
//
//   * a node is one position, reached by `move` from its parent;
//   * `children[0]` is the trunk -- the line that was played first, and therefore
//     the one the charts follow and the Down key descends;
//   * replaying a move that is already in the tree reuses that node, so its
//     cached analysis survives a detour;
//   * the analysis of a position is stored *on* the node, never in one global
//     "last result": that global is what made the board show another position's
//     numbers after a navigation.
//
// The text form is PGN-like (`a4-2 b2-5 (c3-1) d4-7`) because a human can read it
// and the parser for it is ten lines.
#pragma once

#include "engine_view.h"
#include "shuyi/position.h"

#include <string>
#include <vector>

namespace shuyi {

struct RecordNode {
    int parent = -1;
    std::vector<int> children;  // children[0] is the trunk
    Move move{};                // the move that reaches this node; invalid at the root
    // What the engine said about *this* position, in the side-to-move
    // perspective it was sent in.  Invalid until an analysis arrived.
    EngineAnalysisView analysis;
    // Which analysis run produced `analysis` (-1: not from a run of this session,
    // i.e. nothing known yet or loaded from a file).  Numbers from two runs are
    // not comparable: a run that starts after the search tree was rebuilt begins
    // again from zero visits.
    int run = -1;
};

// The move as the record and the window spell it: "a4-2", "pass".
std::wstring moveLabel(Move move);
// The same move in the protocol's spelling ("a4:2"), which is what the engine's
// `play` and `analyze` commands take.
std::string moveProtocolText(Move move);
bool parseMoveLabel(const std::string& text, Move& move);

class GameRecord {
public:
    GameRecord();

    int root() const { return 0; }
    int current() const { return current_; }
    int count() const { return static_cast<int>(nodes_.size()); }
    bool contains(int id) const { return id >= 0 && id < count(); }

    const RecordNode& node(int id) const { return nodes_[static_cast<std::size_t>(id)]; }
    RecordNode& node(int id) { return nodes_[static_cast<std::size_t>(id)]; }
    int parentOf(int id) const { return contains(id) ? nodes_[static_cast<std::size_t>(id)].parent : -1; }
    const std::vector<int>& childrenOf(int id) const {
        return nodes_[static_cast<std::size_t>(id)].children;
    }
    // Plies from the root: the root is 0, and the side to move at depth d is Red
    // for even d (a pass also flips the side, so depth and parity agree).
    int depth(int id) const;
    // Root -> id, excluding the root itself.
    std::vector<int> line(int id) const;

    // Plays `move` under `parent` and makes the result current.  An existing edge
    // is reused, which is what keeps a revisited variation's analysis.
    int play(int parent, Move move);
    // Makes `id` current; returns the id (unchanged when out of range).
    int select(int id);
    int up();    // parent, or -1 at the root
    int down();  // first child (the trunk), or -1 at a leaf
    // Deletes `id` and its subtree, makes the parent current.  The root cannot be
    // deleted.  Returns false when nothing was removed.
    bool erase(int id);

    void setAnalysis(int id, EngineAnalysisView analysis, int run = -1);

    // PGN-like text: a header line, the rule set, then the moves.  Variations are
    // written inline in parentheses, right after the move they replace.  The rule
    // set is part of the record: replaying it under another rule set would produce
    // a different game from the same moves.
    //
    // With an analysis on any node the file grows a block of `#`-prefixed lines
    // (see the .cpp for the format), which a reader that only knows the moves
    // skips as comments.  That is the point of the prefix: a 1.1 build opens a
    // record saved with analysis and simply does not show the numbers.
    std::string save(Ruleset ruleset) const;
    bool load(const std::string& text, Ruleset* ruleset, std::string* error);

private:
    int addChild(int parent, Move move);
    void writeLineText(std::string& out, int node) const;
    void writeAnalysis(std::string& out, int node, const std::string& path) const;
    bool loadAnalysisLine(const std::string& line, std::string* error);

    std::vector<RecordNode> nodes_;
    int current_ = 0;
};

// What a move did to the number the previous position's analysis gave it, in the
// mover's own win rate.  `delta` is positive when the move turned out better than
// that number; when the engine had never scored this particular move, the number
// it is compared with is the verdict on the position it was played from.
struct MoveVerdict {
    double delta = 0.0;
    bool scored = false;  // the engine had a value for this move, not just the position
};
bool moveVerdict(const RecordNode& parent, const RecordNode& child, MoveVerdict* verdict);

} // namespace shuyi
