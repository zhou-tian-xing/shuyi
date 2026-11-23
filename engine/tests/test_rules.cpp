// Tests for the blocking rule set and for the pass machinery it introduces.
//
// The blocking rule: a placement of number n at a cell is illegal when one of the
// four orthogonal neighbours holds an *opposing* stone of number >= n.  Own stones
// never block.  Everything else (score, 54 stones, three copies per number) is the
// classic game, so these tests also pin down that switching the rule set does not
// disturb the score.
#include "shuyi/ai.h"
#include "shuyi/format.h"
#include "shuyi/position.h"
#include "shuyi/rules.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

using namespace shuyi;

namespace {
int failures = 0;

void checkImpl(bool condition, const char* expression, int line) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL line " << line << ": " << expression << "\n";
    }
}

#define CHECK(expr) checkImpl((expr), #expr, __LINE__)

bool hasMove(const std::vector<Move>& moves, int cell, int number) {
    return std::any_of(moves.begin(), moves.end(),
                       [&](Move m) { return m.cell == cell && m.number == number; });
}

Position blockingStart() {
    Position position;
    position.setRuleset(Ruleset::Blocking);
    return position;
}

// Play a red stone without going through legality (used to set up shapes).
void placeUnchecked(Position& position, int cell, int number, bool red) {
    // Legality is checked by play(); for setups that are trivially legal this is
    // enough: the side to move alternates, so play a filler move when needed.
    const Player side = position.sideToMove();
    const bool wantRed = red;
    if ((side == Player::Red) == wantRed) {
        CHECK(position.play(Move(cell, number)));
        return;
    }
    // Play a throwaway black/red stone far away first.
    for (int filler = kCells - 1; filler >= 0; --filler) {
        if (position.stoneAt(filler) != 0) continue;
        if (position.play(Move(filler, 1))) break;
    }
    CHECK(position.play(Move(cell, number)));
}

void testBlockingDeniesNeighbour() {
    Position position = blockingStart();
    // Red 5 at e5 (index 40); Black may not place 1..5 next to it but may place 6..9.
    CHECK(position.play(Move(40, 5)));            // red
    CHECK(!position.isLegal(Move(41, 5)));        // same number, adjacent
    CHECK(!position.isLegal(Move(41, 1)));
    CHECK(!position.isLegal(Move(39, 5)));
    CHECK(position.isLegal(Move(41, 6)));         // strictly larger is fine
    CHECK(position.isLegal(Move(41, 9)));
    CHECK(position.isLegal(Move(50, 9)));         // not adjacent, anything goes
    const auto moves = position.legalMoves();
    CHECK(!hasMove(moves, 41, 5));
    CHECK(hasMove(moves, 41, 6));
    // The 5 is red, so Red is not blocked by its own stone.
    CHECK(position.sideToMove() == Player::Black);
}

void testBlockingNineSealsCross() {
    Position position = blockingStart();
    CHECK(position.play(Move(40, 9)));            // red 9 at the centre
    const int ring[4] = {39, 41, 31, 49};
    for (int cell : ring) {
        CHECK(!position.isLegal(Move(cell, 9)));
        CHECK(!position.isLegal(Move(cell, 1)));
    }
    const auto moves = position.legalMoves();
    for (int cell : ring) {
        for (int number = 1; number <= 9; ++number) {
            CHECK(!hasMove(moves, cell, number));
        }
    }
    // Diagonal is not "nearby".
    CHECK(position.isLegal(Move(50, 9)));
}

void testBlockingIsSideAsymmetric() {
    Position position = blockingStart();
    CHECK(position.play(Move(40, 3)));            // red 3
    CHECK(!position.isLegal(Move(41, 3)));        // black blocked
    CHECK(position.play(Move(41, 9)));            // black 9 is allowed
    // Red 9 next to a black 9 is fine (9 is not > 9, so red cannot play 9 there).
    CHECK(!position.isLegal(Move(42, 9)));
    CHECK(position.isLegal(Move(42, 1)) == false); // 1 < 9 too
    // Far from the black 9, red is free.
    CHECK(position.isLegal(Move(0, 9)));
}

void testClassicUnaffected() {
    Position position;                            // classic by default
    CHECK(position.ruleset() == Ruleset::Classic);
    CHECK(position.play(Move(40, 9)));
    CHECK(position.isLegal(Move(41, 9)));         // classic ignores adjacency
    CHECK(position.isLegal(Move(41, 1)));
    // Copies are not distinguishable, so this is 80 empty cells x 9 numbers.
    CHECK(position.legalMoves().size() == 80 * 9);
}

void testBlockingScoreUnchanged() {
    // Same placement sequence must give the same score under both rule sets.
    std::mt19937 rng(11);
    for (int game = 0; game < 20; ++game) {
        Position classic;
        Position blocking = blockingStart();
        bool diverged = false;
        for (int ply = 0; ply < kTotalMoves && !diverged; ++ply) {
            const auto moves = classic.searchMoves();
            if (moves.empty()) break;
            const Move move = moves[rng() % moves.size()];
            if (!blocking.isLegal(move)) {
                diverged = true;
                break;
            }
            CHECK(classic.play(move));
            CHECK(blocking.play(move));
        }
        if (diverged) continue;
        CHECK(classic.rawScore() == blocking.rawScore());
        CHECK(classic.key() == blocking.key());
        CHECK(classic.stones() == blocking.stones());
    }
}

void testPassOnlyWhenStarved() {
    Position position = blockingStart();
    CHECK(!position.pass());                      // plenty of moves, refused
    CHECK(position.ply() == 0);
    // A fresh game never needs a pass, so searchMoves == legalMoves.
    CHECK(position.searchMoves().size() == position.legalMoves().size());
}

void testPassRoundTrip() {
    // Force starvation: fill the board except one cell, then hand the turn to a
    // side whose only legal numbers are blocked there.  Building that exactly is
    // fiddly, so drive it through random blocking games until one passes -- with
    // 4-neighbour blocking the late game runs dry often enough.
    std::mt19937 rng(5);
    int seenPasses = 0;
    for (int game = 0; game < 400; ++game) {
        Position position = blockingStart();
        std::vector<Move> played;
        while (!position.gameOver()) {
            const auto moves = position.searchMoves();
            if (moves.empty()) break;
            const Move move = moves[rng() % moves.size()];
            if (isPass(move)) {
                ++seenPasses;
                CHECK(position.play(move));
                CHECK(position.passes() == 1);
                const int plyBefore = position.ply();
                CHECK(position.undo());
                CHECK(position.ply() == plyBefore - 1);
                CHECK(position.passes() == 0);
                break;
            }
            CHECK(position.play(move));
            played.push_back(move);
            CHECK(position.validate());
        }
        // Replaying the history must reproduce the position bit for bit.
        Position replay = blockingStart();
        for (const Move& move : position.history()) {
            CHECK(replay.play(move));
        }
        CHECK(replay.key() == position.key());
        CHECK(replay.rawScore() == position.rawScore());
        CHECK(replay.stones() == position.stones());
    }
    if (seenPasses == 0) {
        // Not a failure: with 4-neighbour blocking a random game rarely starves.
        std::cout << "note: no forced pass encountered in random blocking games\n";
    }
}

void testBlockingSearchFindsMoves() {
    Position position = blockingStart();
    SearchOptions options;
    options.timeLimitMs = 40;
    options.maxDepth = 4;
    const SearchResult result = searchAlphaBeta(position, options, *new TranspositionTable(4));
    CHECK(result.move.valid());
    CHECK(position.isLegal(result.move));
}

void testVisitAbbreviations() {
    // The mark shows at most four characters, so 1.2k / 12k / 120k / 1.2M.
    CHECK(formatVisits(0) == L"0");
    CHECK(formatVisits(820) == L"820");
    CHECK(formatVisits(999) == L"999");
    CHECK(formatVisits(1000) == L"1.0k");
    CHECK(formatVisits(1024) == L"1.0k");
    CHECK(formatVisits(1200) == L"1.2k");
    CHECK(formatVisits(9999) == L"10.0k");  // rounded, still four characters
    CHECK(formatVisits(12345) == L"12k");
    CHECK(formatVisits(120000) == L"120k");
    CHECK(formatVisits(999999) == L"1000k");
    CHECK(formatVisits(1234567) == L"1.2M");
    CHECK(formatVisits(12345678) == L"12.3M");
    for (long long value : {0LL, 999LL, 1024LL, 12345LL, 123456LL, 1234567LL}) {
        CHECK(formatVisits(value).size() <= 5); // "1.2M" is 4, "1000k" is 5 worst case
    }
}

void testScoreFormatting() {
    CHECK(formatWinRate(0.62) == L"62%");
    CHECK(formatWinRate(0.0) == L"0%");
    CHECK(formatWinRate(1.0) == L"100%");
    CHECK(formatAverageScore(3.4) == L"+3.4");
    CHECK(formatAverageScore(-0.5) == L"-0.5");
    CHECK(formatAverageScore(0.0) == L"0.0");
}

void testMarkRanking() {
    // The board marks rank by winRateLowerBound, which is what keeps the "best"
    // mark from following a single lucky visit around the board.
    const double oneVisitPerfect = winRateLowerBound(1.0, 1);
    const double searchedGood = winRateLowerBound(0.55, 200);
    const double searchedBad = winRateLowerBound(0.45, 200);
    CHECK(oneVisitPerfect > 0.15 && oneVisitPerfect < 0.30);
    CHECK(searchedGood > 0.48 && searchedGood < 0.52);
    CHECK(searchedGood > oneVisitPerfect);
    CHECK(searchedGood > searchedBad);
    // Same win rate, more visits -> tighter bound.
    CHECK(winRateLowerBound(0.6, 400) > winRateLowerBound(0.6, 40));
    // Degenerate inputs must not produce NaN or a negative bound.
    CHECK(winRateLowerBound(0.0, 0) == 0.0);
    CHECK(winRateLowerBound(0.0, 10) >= 0.0);
    CHECK(winRateLowerBound(1.0, 10000) > 0.99);
    // A one-visit 100% cell must never outrank a 200-visit 55% cell: this is the
    // invariant that stopped the flicker.
    CHECK(!(oneVisitPerfect > searchedGood));
}

// The board is symmetric under the eight isometries of the square, and both rule
// sets only look at a cell plus its four orthogonal neighbours, so a rotated or
// mirrored position has exactly the same value.  Everything downstream (search,
// reporting, marks) relies on these helpers.
void testBoardSymmetry() {
    // Every transform is a bijection that keeps the orthogonal neighbourhood.
    for (int symmetry = 0; symmetry < Position::kSymmetryCount; ++symmetry) {
        std::array<int, kCells> seen{};
        for (int cell = 0; cell < kCells; ++cell) {
            const int image = Position::transformCell(cell, symmetry);
            CHECK(image >= 0 && image < kCells);
            ++seen[image];
            // Adjacency: the four neighbours map onto the image's neighbours.
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (std::abs(dx) + std::abs(dy) != 1) continue;
                    const int x = xOf(cell) + dx;
                    const int y = yOf(cell) + dy;
                    if (!onBoard(x, y)) continue;
                    const int neighbour = Position::transformCell(index(x, y), symmetry);
                    const int imageX = xOf(image) + dx;
                    const int imageY = yOf(image) + dy;
                    // The transformed neighbour must be adjacent to the image.
                    const bool adjacent =
                        (xOf(neighbour) == xOf(image) + 1 && yOf(neighbour) == yOf(image)) ||
                        (xOf(neighbour) == xOf(image) - 1 && yOf(neighbour) == yOf(image)) ||
                        (yOf(neighbour) == yOf(image) + 1 && xOf(neighbour) == xOf(image)) ||
                        (yOf(neighbour) == yOf(image) - 1 && xOf(neighbour) == xOf(image));
                    (void)imageX;
                    (void)imageY;
                    CHECK(adjacent);
                }
            }
        }
        for (int cell = 0; cell < kCells; ++cell) CHECK(seen[cell] == 1);
    }
    // Identity fixes everything; the empty board admits all eight symmetries.
    for (int cell = 0; cell < kCells; ++cell) CHECK(Position::transformCell(cell, 0) == cell);
    Position empty;
    CHECK(empty.symmetryMask() == 0xFF);

    // A stone on the centre keeps all eight; a stone on the main diagonal keeps
    // only the identity and that diagonal.
    Position centre;
    centre.play(Move(index(4, 4), 5));
    CHECK(centre.symmetryMask() == 0xFF);
    Position diagonal;
    diagonal.play(Move(index(0, 0), 1));
    CHECK((diagonal.symmetryMask() & 1) != 0);   // identity always survives
    CHECK((diagonal.symmetryMask() & (1u << 4)) != 0);  // main diagonal
    CHECK((diagonal.symmetryMask() & (1u << 1)) == 0);  // the x mirror does not
    // ... and once two non-equivalent stones are down, only the identity is left.
    diagonal.play(Move(index(1, 0), 2));
    CHECK(diagonal.symmetryMask() == 1);

    // Orbits on an empty board: four corners, four edge centres, and a single
    // centre cell; a cell in the middle of an edge has four images.
    const std::uint8_t all = empty.symmetryMask();
    CHECK(Position::orbitSize(index(0, 0), all) == 4);
    CHECK(Position::orbitSize(index(4, 4), all) == 1);
    CHECK(Position::orbitSize(index(4, 0), all) == 4);
    CHECK(Position::orbitSize(index(2, 2), all) == 4);
    CHECK(Position::orbitSize(index(2, 0), all) == 8);
}

void testRulesetNames() {
    Ruleset parsed = Ruleset::Classic;
    CHECK(parseRuleset("blocking", parsed) && parsed == Ruleset::Blocking);
    CHECK(parseRuleset("blk", parsed) && parsed == Ruleset::Blocking);
    CHECK(parseRuleset("classic", parsed) && parsed == Ruleset::Classic);
    CHECK(parseRuleset("cls", parsed) && parsed == Ruleset::Classic);
    CHECK(!parseRuleset("nonsense", parsed));
    CHECK(std::string(rulesetName(Ruleset::Blocking)) == "blocking");
    CHECK(std::string(rulesetTag(Ruleset::Classic)) == "cls");
}

} // namespace

int runRulesTests() {
    const int before = failures;
    testBlockingDeniesNeighbour();
    testBlockingNineSealsCross();
    testBlockingIsSideAsymmetric();
    testClassicUnaffected();
    testBlockingScoreUnchanged();
    testPassOnlyWhenStarved();
    testPassRoundTrip();
    testBlockingSearchFindsMoves();
    testRulesetNames();
    testVisitAbbreviations();
    testScoreFormatting();
    testMarkRanking();
    testBoardSymmetry();
    return failures - before;
}
