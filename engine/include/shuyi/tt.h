#pragma once

#include "shuyi/position.h"

#include <cstdint>
#include <vector>

namespace shuyi {

enum class TTFlag : std::uint8_t { None = 0, Exact = 1, Lower = 2, Upper = 3 };

struct TTEntry {
    std::uint64_t key = 0;
    std::int16_t value = 0;
    std::int8_t depth = -1;
    std::uint8_t flag = 0;
    std::uint16_t move = 0; // cell | number << 7
    std::uint16_t age = 0;
};

class TranspositionTable {
public:
    explicit TranspositionTable(std::size_t megabytes = 16);
    void clear();
    void newSearch();

    // Returns true if the stored bound is usable for the current window.
    // `bestMove` is always filled when the key matches, even if the bound is
    // not strong enough to cut off.
    bool probe(std::uint64_t key, int depth, int alpha, int beta, int& value, Move& bestMove) const;
    void store(std::uint64_t key, int depth, int value, TTFlag flag, Move bestMove);

    std::size_t entries() const { return table_.size(); }
    std::size_t usedPermille() const;
    std::uint16_t generation() const { return generation_; }

    static std::uint16_t encodeMove(Move move);
    static Move decodeMove(std::uint16_t encoded);

private:
    std::vector<TTEntry> table_;
    std::size_t mask_ = 0;
    std::uint16_t generation_ = 0;
};

} // namespace shuyi
