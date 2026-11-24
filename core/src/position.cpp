#include "shuyi/position.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>

namespace shuyi {
namespace {
int clampScore(int v) { return (v > 0) - (v < 0); }

std::uint64_t splitmix64(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct ZobristTable {
    std::uint64_t piece[2][10][kCells]{};
    std::uint64_t side = 0;

    ZobristTable() {
        std::uint64_t state = 0x5EED5EED12345678ULL;
        for (int sideIndex = 0; sideIndex < 2; ++sideIndex)
            for (int number = 1; number <= 9; ++number)
                for (int cell = 0; cell < kCells; ++cell)
                    piece[sideIndex][number][cell] = splitmix64(state);
        side = splitmix64(state);
    }
};

const ZobristTable& zobristTable() {
    static const ZobristTable table;
    return table;
}

std::uint64_t pieceKey(Player side, int number, int cell) {
    return zobristTable().piece[static_cast<int>(side)][number][cell];
}
} // namespace

const char* playerName(Player p) { return p == Player::Red ? "红" : "黑"; }

std::string squareName(int cell) {
    if (cell < 0 || cell >= kCells) return "??";
    std::string s;
    s.push_back(static_cast<char>('a' + xOf(cell)));
    s.push_back(static_cast<char>('1' + yOf(cell)));
    return s;
}

bool parseSquare(const std::string& text, int& cell) {
    if (text.size() != 2) return false;
    char xc = static_cast<char>(std::tolower(static_cast<unsigned char>(text[0])));
    char yc = text[1];
    if (xc < 'a' || xc > 'i' || yc < '1' || yc > '9') return false;
    cell = index(xc - 'a', yc - '1');
    return true;
}

Position::Position() { reset(); }

void Position::reset() {
    board_.fill(0);
    for (auto& side : remaining_) {
        side.fill(0);
        for (int n = 1; n <= 9; ++n) side[n] = kPiecesPerNumber;
    }
    cellScore_.fill(0);
    occupied_.fill(Bitboard{});
    for (auto& side : number_) side.fill(Bitboard{});
    empty_ = Bitboard{~0ULL, (1ULL << (kCells - 64)) - 1};
    zobrist_ = 0;
    history_.clear();
    ply_ = 0;
    stones_ = 0;
    passes_ = 0;
    rawScore_ = 0;
}

int Position::evaluateCell(int cell) const {
    const int x = xOf(cell);
    const int y = yOf(cell);
    int redMax = 0;
    int blackMax = 0;

    auto add = [&](int xx, int yy) {
        if (!onBoard(xx, yy)) return;
        const int v = board_[index(xx, yy)];
        if (v > 0) redMax = std::max(redMax, static_cast<int>(v));
        else if (v < 0) blackMax = std::max(blackMax, -static_cast<int>(v));
    };

    add(x, y);
    add(x - 1, y);
    add(x + 1, y);
    add(x, y - 1);
    add(x, y + 1);
    return clampScore(redMax - blackMax);
}

void Position::refreshAround(int cell) {
    const int x = xOf(cell);
    const int y = yOf(cell);
    auto refresh = [&](int xx, int yy) {
        if (!onBoard(xx, yy)) return;
        const int c = index(xx, yy);
        const int next = evaluateCell(c);
        rawScore_ += next - static_cast<int>(cellScore_[c]);
        cellScore_[c] = static_cast<std::int8_t>(next);
    };
    refresh(x, y);
    refresh(x - 1, y);
    refresh(x + 1, y);
    refresh(x, y - 1);
    refresh(x, y + 1);
}

int Position::transformCell(int cell, int symmetry) {
    const int x = xOf(cell);
    const int y = yOf(cell);
    const int n = kSize - 1;
    switch (symmetry & 7) {
    case 0: return index(x, y);                 // identity
    case 1: return index(n - x, y);             // mirror in x
    case 2: return index(x, n - y);             // mirror in y
    case 3: return index(n - x, n - y);         // 180 degrees
    case 4: return index(y, x);                 // main diagonal
    case 5: return index(y, n - x);             // rotate 90 degrees
    case 6: return index(n - y, x);             // rotate 270 degrees
    default: return index(n - y, n - x);        // anti diagonal
    }
}

std::uint8_t Position::symmetryMask() const {
    std::uint8_t mask = 0;
    for (int symmetry = 0; symmetry < kSymmetryCount; ++symmetry) {
        bool same = true;
        for (int cell = 0; cell < kCells && same; ++cell) {
            if (board_[cell] != board_[transformCell(cell, symmetry)]) same = false;
        }
        if (same) mask = static_cast<std::uint8_t>(mask | (1u << symmetry));
    }
    return mask;
}

int Position::orbitSize(int cell, std::uint8_t mask) {    int size = 0;
    for (int symmetry = 0; symmetry < kSymmetryCount; ++symmetry) {
        if (((mask >> symmetry) & 1u) == 0) continue;
        bool seen = false;
        for (int earlier = 0; earlier < symmetry; ++earlier) {
            if (((mask >> earlier) & 1u) && transformCell(cell, earlier) == transformCell(cell, symmetry)) {
                seen = true;
                break;
            }
        }
        if (!seen) ++size;
    }
    return size;
}

int Position::remainingPieces(Player player) const {
    int total = 0;
    for (int n = 1; n <= 9; ++n) total += remaining_[static_cast<int>(player)][n];
    return total;
}

int Position::blockingReach(int cell) const {
    // Largest opposing number in the four orthogonal neighbours.  A placement of
    // n is legal only when n > this value, i.e. a red 9 denies Black every
    // number on those cells because nothing is > 9.
    const int x = xOf(cell);
    const int y = yOf(cell);
    const int redSign = sideToMove() == Player::Red ? 1 : -1; // my stones
    int reach = 0;
    const auto consider = [&](int xx, int yy) {
        if (!onBoard(xx, yy)) return;
        const int v = board_[index(xx, yy)];
        if (v * redSign >= 0) return; // my own stone or empty
        reach = std::max(reach, v > 0 ? v : -v);
    };
    consider(x - 1, y);
    consider(x + 1, y);
    consider(x, y - 1);
    consider(x, y + 1);
    return reach;
}

bool Position::hasPlacement() const {
    if (stones_ >= kTotalMoves) return false;
    const auto& left = remaining_[static_cast<int>(sideToMove())];
    int largest = 0;
    for (int number = 9; number >= 1; --number) {
        if (left[number] > 0) {
            largest = number;
            break;
        }
    }
    if (largest == 0) return false;
    for (int cell = 0; cell < kCells; ++cell) {
        if (!empty_.test(cell)) continue;
        // The strongest number in hand is the one that beats the highest reach,
        // so a single compare settles most cells.
        if (!rulesetBlocks(ruleset_) || largest > blockingReach(cell)) return true;
    }
    return false;
}

bool Position::isLegal(Move move) const {
    if (gameOver() || !move.valid()) return false;
    if (board_[move.cell] != 0) return false;
    const Player side = sideToMove();
    if (remaining_[static_cast<int>(side)][move.number] <= 0) return false;
    if (rulesetBlocks(ruleset_) && move.number <= blockingReach(move.cell)) return false;
    return true;
}

std::vector<Move> Position::legalMoves() const {
    std::vector<Move> moves;
    if (gameOver()) return moves;
    moves.reserve(static_cast<std::size_t>(remainingPieces(sideToMove())) * 16);
    const Player side = sideToMove();
    const auto& left = remaining_[static_cast<int>(side)];
    if (!rulesetBlocks(ruleset_)) {
        for (int number = 1; number <= 9; ++number) {
            if (left[number] == 0) continue;
            for (int cell = 0; cell < kCells; ++cell) {
                if (empty_.test(cell)) moves.emplace_back(cell, number);
            }
        }
        return moves;
    }
    // Blocking: walk cells so the four neighbours are read once per cell, then
    // emit only the numbers that beat the local reach.
    for (int cell = 0; cell < kCells; ++cell) {
        if (!empty_.test(cell)) continue;
        const int floorNumber = blockingReach(cell) + 1;
        for (int number = floorNumber; number <= 9; ++number) {
            if (left[number] == 0) continue;
            moves.emplace_back(cell, number);
        }
    }
    return moves;
}

std::vector<Move> Position::searchMoves() const {
    std::vector<Move> moves = legalMoves();
    // A forced pass exists only where the rules allow a pass at all.  Under
    // NoPass the same position is already terminal, so gameOver() is true and
    // no pass is emitted.
    if (moves.empty() && !gameOver() && rulesetHasPass(ruleset_)) {
        moves.emplace_back(kPassCell, kPassNumber);
    }
    return moves;
}

bool Position::play(Move move) {
    if (move.number == kPassNumber) return pass();
    if (!isLegal(move)) return false;
    const Player side = sideToMove();
    const int sideIndex = static_cast<int>(side);
    const int signedNumber = side == Player::Red ? static_cast<int>(move.number)
                                                  : -static_cast<int>(move.number);
    board_[move.cell] = static_cast<std::int8_t>(signedNumber);
    --remaining_[sideIndex][move.number];
    const Bitboard placed = bit(move.cell);
    empty_ &= ~placed;
    occupied_[sideIndex] |= placed;
    number_[sideIndex][move.number] |= placed;
    zobrist_ ^= pieceKey(side, move.number, move.cell) ^ zobristTable().side;
    history_.push_back(move);
    refreshAround(move.cell);
    ++ply_;
    ++stones_;
    return true;
}

bool Position::pass() {
    // NoPass has no pass move at all: being unable to place ends the game.
    if (!rulesetHasPass(ruleset_)) return false;
    if (gameOver()) return false;
    if (!legalMoves().empty()) return false; // a pass is only ever forced
    history_.push_back(Move{kPassCell, kPassNumber});
    zobrist_ ^= zobristTable().side;
    ++ply_;
    ++passes_;
    return true;
}

bool Position::undo() {
    if (history_.empty()) return false;
    const Move move = history_.back();
    history_.pop_back();
    if (move.number == kPassNumber) {
        zobrist_ ^= zobristTable().side;
        --ply_;
        --passes_;
        return true;
    }
    board_[move.cell] = 0;
    --ply_;
    const Player mover = sideToMove();
    const int sideIndex = static_cast<int>(mover);
    ++remaining_[sideIndex][move.number];
    const Bitboard removed = bit(move.cell);
    empty_ |= removed;
    occupied_[sideIndex] &= ~removed;
    number_[sideIndex][move.number] &= ~removed;
    zobrist_ ^= pieceKey(mover, move.number, move.cell) ^ zobristTable().side;
    refreshAround(move.cell);
    --stones_;
    return true;
}

Outcome Position::outcome(double aValue) const {
    if (!gameOver()) return Outcome::Ongoing;
    const double value = finalScore(aValue);
    if (value > 0.0) return Outcome::RedWin;
    if (value < 0.0) return Outcome::BlackWin;
    return Outcome::Draw;
}

int Position::recomputeScoreFromScratch() const {
    int sum = 0;
    for (int cell = 0; cell < kCells; ++cell) sum += evaluateCell(cell);
    return sum;
}

std::uint64_t Position::recomputeKeyFromScratch() const {
    std::uint64_t key = sideToMove() == Player::Black ? zobristTable().side : 0;
    for (int cell = 0; cell < kCells; ++cell) {
        const int value = board_[cell];
        if (value == 0) continue;
        const Player side = value > 0 ? Player::Red : Player::Black;
        const int number = value > 0 ? value : -value;
        key ^= pieceKey(side, number, cell);
    }
    return key;
}

bool Position::validate(std::string* error) const {
    std::array<std::array<int, 10>, 2> counts{};
    int stones = 0;
    for (int cell = 0; cell < kCells; ++cell) {
        const int v = board_[cell];
        if (v == 0) continue;
        ++stones;
        const int side = v > 0 ? 0 : 1;
        const int number = v > 0 ? v : -v;
        if (number < 1 || number > 9) {
            if (error) *error = "stone number out of range at " + squareName(cell);
            return false;
        }
        if (++counts[side][number] > kPiecesPerNumber) {
            if (error) *error = "too many stones of one number at " + squareName(cell);
            return false;
        }
    }
    if (stones != stones_) {
        if (error) *error = "stone count does not match the incremental counter";
        return false;
    }
    if (stones + passes_ != ply_) {
        if (error) *error = "stone/pass count does not match ply";
        return false;
    }
    {
        int historyStones = 0;
        int historyPasses = 0;
        for (const Move& move : history_) {
            if (move.number == kPassNumber) ++historyPasses;
            else ++historyStones;
        }
        if (historyStones != stones_ || historyPasses != passes_) {
            if (error) *error = "history does not match stones/passes";
            return false;
        }
    }
    if (history_.size() != static_cast<std::size_t>(ply_)) {
        if (error) *error = "history size does not match ply";
        return false;
    }
    // NoPass cannot represent a pass at all, so a position that carries one is
    // not a legal NoPass position (it came from another rule set).
    if (!rulesetHasPass(ruleset_) && passes_ != 0) {
        if (error) *error = "this rule set has no pass, but the position has one";
        return false;
    }
    for (int side = 0; side < 2; ++side) {
        for (int n = 1; n <= 9; ++n) {
            if (remaining_[side][n] != kPiecesPerNumber - counts[side][n]) {
                if (error) *error = "remaining count inconsistent";
                return false;
            }
        }
    }
    for (int cell = 0; cell < kCells; ++cell) {
        const int value = board_[cell];
        const int side = value > 0 ? 0 : (value < 0 ? 1 : -1);
        const bool occupied = side >= 0;
        const bool emptyBit = empty_.test(cell);
        const bool occupiedBit = (occupied_[0] | occupied_[1]).test(cell);
        if (emptyBit == occupied || occupiedBit != occupied) {
            if (error) *error = "bitboard occupancy is inconsistent at " + squareName(cell);
            return false;
        }
        if (occupied) {
            const int number = value > 0 ? value : -value;
            if (!number_[side][number].test(cell)) {
                if (error) *error = "number bitboard is inconsistent at " + squareName(cell);
                return false;
            }
        }
    }
    for (int side = 0; side < 2; ++side) {
        for (int number = 1; number <= 9; ++number) {
            for (int cell = 0; cell < kCells; ++cell) {
                if (!number_[side][number].test(cell)) continue;
                const int expected = side == 0 ? number : -number;
                if (board_[cell] != expected) {
                    if (error) *error = "number bitboard points at the wrong stone";
                    return false;
                }
            }
        }
    }
    if (zobrist_ != recomputeKeyFromScratch()) {
        if (error) *error = "incremental Zobrist key is inconsistent";
        return false;
    }
    if (rawScore_ != recomputeScoreFromScratch()) {
        if (error) *error = "incremental score is inconsistent";
        return false;
    }
    for (int cell = 0; cell < kCells; ++cell) {
        if (cellScore_[cell] != evaluateCell(cell)) {
            if (error) *error = "cached cell score is inconsistent at " + squareName(cell);
            return false;
        }
    }
    if (error) error->clear();
    return true;
}

} // namespace shuyi
