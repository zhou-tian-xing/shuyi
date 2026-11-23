#include "shuyi/tt.h"

#include <algorithm>
#include <limits>

namespace shuyi {

namespace {
std::size_t floorPowerOfTwo(std::size_t value) {
    std::size_t result = 1;
    while (result <= value / 2) result <<= 1;
    return result;
}
} // namespace

TranspositionTable::TranspositionTable(std::size_t megabytes) {
    megabytes = std::max<std::size_t>(1, std::min<std::size_t>(megabytes, 1024));
    const std::size_t bytes = megabytes * 1024 * 1024;
    std::size_t count = std::max<std::size_t>(1024, bytes / sizeof(TTEntry));
    count = floorPowerOfTwo(count);
    table_.assign(count, TTEntry{});
    mask_ = count - 1;
}

void TranspositionTable::clear() {
    std::fill(table_.begin(), table_.end(), TTEntry{});
    generation_ = 0;
}

void TranspositionTable::newSearch() {
    ++generation_;
    if (generation_ == std::numeric_limits<std::uint16_t>::max()) {
        clear();
    }
}

std::uint16_t TranspositionTable::encodeMove(Move move) {
    if (!move.valid()) return 0;
    return static_cast<std::uint16_t>(move.cell | (move.number << 7));
}

Move TranspositionTable::decodeMove(std::uint16_t encoded) {
    if (encoded == 0) return Move{};
    Move move;
    move.cell = static_cast<std::uint8_t>(encoded & 0x7F);
    move.number = static_cast<std::uint8_t>(encoded >> 7);
    return move.valid() ? move : Move{};
}

bool TranspositionTable::probe(std::uint64_t key, int depth, int alpha, int beta, int& value,
                               Move& bestMove) const {
    const TTEntry& entry = table_[key & mask_];
    if (entry.key != key || entry.flag == static_cast<std::uint8_t>(TTFlag::None)) return false;
    bestMove = decodeMove(entry.move);
    if (entry.depth < depth) return false;
    if (entry.flag == static_cast<std::uint8_t>(TTFlag::Exact)) {
        value = entry.value;
        return true;
    }
    if (entry.flag == static_cast<std::uint8_t>(TTFlag::Lower) && entry.value >= beta) {
        value = entry.value;
        return true;
    }
    if (entry.flag == static_cast<std::uint8_t>(TTFlag::Upper) && entry.value <= alpha) {
        value = entry.value;
        return true;
    }
    return false;
}

void TranspositionTable::store(std::uint64_t key, int depth, int value, TTFlag flag, Move bestMove) {
    TTEntry& entry = table_[key & mask_];
    const bool replace = entry.key == key || entry.flag == static_cast<std::uint8_t>(TTFlag::None) ||
                         depth >= entry.depth || entry.age != generation_;
    if (!replace) return;
    entry.key = key;
    entry.value = static_cast<std::int16_t>(value);
    entry.depth = static_cast<std::int8_t>(depth);
    entry.flag = static_cast<std::uint8_t>(flag);
    entry.move = encodeMove(bestMove);
    entry.age = generation_;
}

std::size_t TranspositionTable::usedPermille() const {
    if (table_.empty()) return 0;
    const std::size_t sample = std::min<std::size_t>(table_.size(), 4096);
    std::size_t used = 0;
    for (std::size_t i = 0; i < sample; ++i) used += table_[i].flag != 0 ? 1 : 0;
    return sample == 0 ? 0 : used * 1000 / sample;
}

} // namespace shuyi
