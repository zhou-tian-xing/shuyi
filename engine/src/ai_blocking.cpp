// Blocking-rule evaluation and move ordering.
//
// The blocking rule turns a stone into a claim: a red stone of number n denies the
// opponent every number <= n on the four orthogonal neighbours, permanently.  That
// changes what a static evaluation should look at:
//
//   * material is unchanged (both sides always place 27 stones, three of each
//     number), so the raw score is still the backbone;
//   * what the rule adds is *squeeze*: a side can be starved of legal placements,
//     lose tempo, and never place its expensive stones.  Mobility is therefore a
//     first-class evaluation term, not a tie-break;
//   * the remaining hand is worth more when it is the opponent who is short of
//     space, and less when high numbers can no longer be placed anywhere.
//
// Move ordering rewards denial: how many opponent placements a move removes from
// the neighbourhood.  A 9 next to an enemy cluster can delete dozens of candidate
// moves, which is exactly what makes the PVS re-searches cheap.
#include "shuyi/ai_policy.h"

#include <algorithm>
#include <cmath>

namespace shuyi {
namespace {

// Cheap mobility proxy: for each empty cell, can the side place anything at all?
// This is one four-neighbour scan per cell, i.e. the same order of cost as the
// existing score scan, and it captures "squeezed" without generating moves.
int placementCells(const Position& position, Player side) {
    bool anyNumber[10] = {false};
    for (int number = 1; number <= 9; ++number) {
        anyNumber[number] = position.remaining(side, number) > 0;
    }
    const int sign = side == Player::Red ? 1 : -1;
    int count = 0;
    for (int cell = 0; cell < kCells; ++cell) {
        if (!position.emptyMask().test(cell)) continue;
        const int x = xOf(cell);
        const int y = yOf(cell);
        int reach = 0;
        const auto consider = [&](int xx, int yy) {
            if (!onBoard(xx, yy)) return;
            const int v = position.stoneAt(index(xx, yy));
            if (v == 0 || v * sign > 0) return; // empty or my own stone
            reach = std::max(reach, v > 0 ? v : -v);
        };
        consider(x - 1, y);
        consider(x + 1, y);
        consider(x, y - 1);
        consider(x, y + 1);
        for (int number = reach + 1; number <= 9; ++number) {
            if (anyNumber[number]) {
                ++count;
                break;
            }
        }
    }
    return count;
}

// Red-perspective remaining-hand potential, as in the classic evaluator, but
// scaled down as the board fills: a 9 held with three empty cells left is worth
// much less than a 9 held with forty.
int handPotential(const Position& position, double handWeight) {
    double hand = 0.0;
    for (int number = 1; number <= 9; ++number) {
        const int diff = position.remaining(Player::Red, number) -
                         position.remaining(Player::Black, number);
        hand += static_cast<double>(diff) * (static_cast<double>(number) - 4.5);
    }
    int empty = 0;
    for (int cell = 0; cell < kCells; ++cell) {
        if (position.emptyMask().test(cell)) ++empty;
    }
    if (empty == 0) return 0;
    const double open = static_cast<double>(empty) / 81.0;
    return static_cast<int>(std::lround(hand * handWeight * open));
}

int blockingEvaluate(const Position& position, double handWeight) {
    const int raw = position.rawScore();
    const int mine = placementCells(position, Player::Red);
    const int theirs = placementCells(position, Player::Black);
    // Being squeezed costs real tempo, so mobility is weighted above a pure
    // tie-break but below a full point of score (which is a whole cell flip).
    const double squeeze = static_cast<double>(mine - theirs) * 0.5;
    const int hand = handPotential(position, handWeight);
    return raw + hand + static_cast<int>(std::lround(squeeze));
}

int blockingOrderBonus(const Position& before, Move move, int immediateDelta) {
    // Denial: how many opponent numbers this placement deletes from the four
    // neighbours.  The opponent's reach at a neighbour cell becomes at least
    // move.number, so the numbers in (oldReach, move.number] stop being playable
    // there.
    const int x = xOf(move.cell);
    const int y = yOf(move.cell);
    const Player me = before.sideToMove();
    const int mySign = me == Player::Red ? 1 : -1;
    const auto neighbourReach = [&](int xx, int yy) {
        // Largest *my* number adjacent to (xx, yy), counting the stone being placed.
        int reach = 0;
        const int nx[4] = {xx - 1, xx + 1, xx, xx};
        const int ny[4] = {yy, yy, yy - 1, yy + 1};
        for (int i = 0; i < 4; ++i) {
            if (!onBoard(nx[i], ny[i])) continue;
            const int cell = index(nx[i], ny[i]);
            if (cell == move.cell) {
                reach = std::max(reach, static_cast<int>(move.number));
                continue;
            }
            const int v = before.stoneAt(cell);
            if (v == 0 || v * mySign < 0) continue; // empty or enemy
            reach = std::max(reach, v > 0 ? v : -v);
        }
        return reach;
    };
    int denial = 0;
    const auto consider = [&](int xx, int yy) {
        if (!onBoard(xx, yy)) return;
        if (before.stoneAt(index(xx, yy)) != 0) return;
        denial += std::max(0, static_cast<int>(move.number) - neighbourReach(xx, yy));
    };
    consider(x - 1, y);
    consider(x + 1, y);
    consider(x, y - 1);
    consider(x, y + 1);
    return 2 * immediateDelta + denial;
}

} // namespace

const SearchPolicy kBlockingSearchPolicy{"blocking", &blockingEvaluate, &blockingOrderBonus};

} // namespace shuyi
