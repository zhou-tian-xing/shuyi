// Shuyi core rules.  Header-only enough for the small board, but the
// implementation lives in src/position.cpp so tests and the UI share one copy.
#pragma once

#include "shuyi/rules.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace shuyi {

// 81-bit board mask.  Two words keep the hot representation compact while
// remaining fully portable (no __int128 / compiler builtins).
struct Bitboard {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    constexpr Bitboard() = default;
    constexpr Bitboard(std::uint64_t low, std::uint64_t high) : lo(low), hi(high) {}

    constexpr bool any() const { return lo != 0 || hi != 0; }
    constexpr bool none() const { return !any(); }
    constexpr bool test(int cell) const {
        return cell < 64 ? ((lo >> cell) & 1ULL) != 0 : ((hi >> (cell - 64)) & 1ULL) != 0;
    }
    constexpr void set(int cell) {
        if (cell < 64) lo |= (1ULL << cell);
        else hi |= (1ULL << (cell - 64));
    }
    constexpr void reset(int cell) {
        if (cell < 64) lo &= ~(1ULL << cell);
        else hi &= ~(1ULL << (cell - 64));
    }
    friend constexpr Bitboard operator&(Bitboard a, Bitboard b) { return {a.lo & b.lo, a.hi & b.hi}; }
    friend constexpr Bitboard operator|(Bitboard a, Bitboard b) { return {a.lo | b.lo, a.hi | b.hi}; }
    friend constexpr Bitboard operator~(Bitboard a) {
        constexpr std::uint64_t kMask = (1ULL << 17) - 1;
        return {~a.lo, (~a.hi) & kMask};
    }
    friend constexpr bool operator==(Bitboard a, Bitboard b) { return a.lo == b.lo && a.hi == b.hi; }
    friend constexpr bool operator!=(Bitboard a, Bitboard b) { return !(a == b); }
    Bitboard& operator&=(Bitboard b) { *this = *this & b; return *this; }
    Bitboard& operator|=(Bitboard b) { *this = *this | b; return *this; }
};

constexpr int kSize = 9;
constexpr int kCells = kSize * kSize;
constexpr int kPiecesPerNumber = 3;
constexpr int kPiecesPerSide = 27;
constexpr int kTotalMoves = 54;

enum class Player : std::uint8_t { Red = 0, Black = 1 };
enum class Outcome : std::int8_t { Ongoing = 0, RedWin = 1, Draw = 2, BlackWin = 3 };

constexpr Player opponent(Player p) {
    return p == Player::Red ? Player::Black : Player::Red;
}

const char* playerName(Player p);

inline int index(int x, int y) { return y * kSize + x; }
inline int xOf(int cell) { return cell % kSize; }
inline int yOf(int cell) { return cell / kSize; }
inline bool onBoard(int x, int y) { return x >= 0 && x < kSize && y >= 0 && y < kSize; }

// "a1" ... "i9"; a1 is the lower-left corner in the drawing, but the board is
// symmetric and the coordinate mapping is simply x=0..8 / y=0..8.
std::string squareName(int cell);
bool parseSquare(const std::string& text, int& cell);

struct Move {
    std::uint8_t cell = 0;
    std::uint8_t number = 0;

    constexpr Move() = default;
    constexpr Move(int c, int n) : cell(static_cast<std::uint8_t>(c)), number(static_cast<std::uint8_t>(n)) {}
    constexpr bool valid() const { return number >= 1 && number <= 9 && cell < kCells; }
    friend constexpr bool operator==(Move a, Move b) { return a.cell == b.cell && a.number == b.number; }
    friend constexpr bool operator!=(Move a, Move b) { return !(a == b); }
};

inline constexpr bool isPass(Move move) { return move.number == kPassNumber; }

// A complete final placement (54 stones, 27 per side, three of each number).
struct Position {
    Position();

    // Cell encoding: +1..+9 = red stone with that number, -1..-9 = black
    // stone, 0 = empty.  Nearby means the cell itself plus the four
    // orthogonal neighbours (Manhattan distance <= 1), clipped at the border.
    int stoneAt(int cell) const { return board_[cell]; }
    int stoneAt(int x, int y) const { return board_[index(x, y)]; }
    Player sideToMove() const { return (ply_ & 1) ? Player::Black : Player::Red; }
    int ply() const { return ply_; }          // turns played, passes included
    int stones() const { return stones_; }    // stones placed (never decreases)
    int passes() const { return passes_; }
    Ruleset ruleset() const { return ruleset_; }
    void setRuleset(Ruleset ruleset) { ruleset_ = ruleset; }
    // The game ends when all 54 stones are placed; under the blocking rule also
    // when both sides pass in a row; under NoPass the moment the side to move
    // has no legal placement (there is no pass to fall back on).
    bool gameOver() const {
        return stones_ >= kTotalMoves || passes_ >= 2 ||
               (!rulesetHasPass(ruleset_) && !hasPlacement());
    }
    // Can the side to move place anything?  Under the blocking rule it can be
    // starved, in which case the rules force a pass; under NoPass that same
    // situation is already terminal, so this stays false.
    bool mustPass() const { return !gameOver() && !hasPlacement(); }
    int rawScore() const { return rawScore_; }
    double finalScore(double aValue) const { return static_cast<double>(rawScore_) + aValue; }
    Outcome outcome(double aValue = 1.0) const;

    int remaining(Player player, int number) const { return remaining_[static_cast<int>(player)][number]; }
    int remainingPieces(Player player) const;
    int scoreCell(int cell) const { return cellScore_[cell]; }

    // Symmetry of the board.  Scoring reads a cell plus its four orthogonal
    // neighbours, and the blocking rule reads the same neighbourhood, so every
    // isometry of the square maps a legal position to an equally legal one with
    // the same score.  Two moves related by a symmetry of the current position
    // are therefore exactly equivalent and must not be searched twice.
    static constexpr int kSymmetryCount = 8;
    static int transformCell(int cell, int symmetry);
    // Bit `s` set when symmetry `s` maps this position onto itself (bit 0 is the
    // identity and is always set).
    std::uint8_t symmetryMask() const;
    // Number of cells in the orbit of `cell` under `mask`.
    static int orbitSize(int cell, std::uint8_t mask);

    // Bitboard views are part of the engine interface, not just an
    // optimization: move generation, TT keys and NN features all use them.
    static constexpr Bitboard bit(int cell) {
        return cell < 64 ? Bitboard{1ULL << cell, 0} : Bitboard{0, 1ULL << (cell - 64)};
    }
    Bitboard emptyMask() const { return empty_; }
    Bitboard occupiedMask(Player player) const { return occupied_[static_cast<int>(player)]; }
    Bitboard numberMask(Player player, int number) const {
        return number_[static_cast<int>(player)][number];
    }
    // Zobrist key including the side to move.  Kept incrementally in play/undo.
    std::uint64_t key() const { return zobrist_; }

    // One move per empty cell and unused number.  The three copies of a number
    // are indistinguishable, so duplicates are never generated.
    std::vector<Move> legalMoves() const;
    // Placements plus, when the side to move is starved, the single forced pass.
    // Search code should use this so a blocked position is not mistaken for a
    // terminal one; interfaces that talk about cells (policy head, UI hints) keep
    // using legalMoves().
    std::vector<Move> searchMoves() const;
    bool isLegal(Move move) const;

    bool play(Move move);
    // Forced pass: only legal when the side to move has no placement.
    bool pass();
    bool undo();
    void reset();

    // Blocking-rule helper: the largest opposing number adjacent to `cell`
    // (0 when there is none).  A number n is placeable at `cell` only if
    // n > blockingReach(cell).  Always 0 for the classic rule.
    int blockingReach(int cell) const;

    const std::vector<Move>& history() const { return history_; }
    Move lastMove() const { return history_.empty() ? Move{} : history_.back(); }

    // Used by tests and the analysis tools.
    static int sign(int value) { return (value > 0) - (value < 0); }
    int recomputeScoreFromScratch() const;
    std::uint64_t recomputeKeyFromScratch() const;
    bool validate(std::string* error = nullptr) const;

private:
    int evaluateCell(int cell) const;
    void refreshAround(int cell);
    // Is any (cell, number) placement legal right now?  Early-outs on the first
    // one, so the terminal check is cheap: gameOver() asks it for NoPass.
    bool hasPlacement() const;

    std::array<std::int8_t, kCells> board_{};
    Ruleset ruleset_ = kDefaultRuleset;
    int stones_ = 0;
    int passes_ = 0;
    std::array<std::array<std::uint8_t, 10>, 2> remaining_{};
    std::array<std::int8_t, kCells> cellScore_{};
    std::array<Bitboard, 2> occupied_{};
    std::array<std::array<Bitboard, 10>, 2> number_{};
    Bitboard empty_{};
    std::uint64_t zobrist_ = 0;
    std::vector<Move> history_;
    int ply_ = 0;
    int rawScore_ = 0;
};

} // namespace shuyi
