#include "shuyi/position.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <cstdlib>

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

// The base-game tests below (scoring, hashing, undo) are rule-set independent,
// but which placements are *legal* is not.  They run under Classic so every
// empty cell accepts every number; the product default is NoPass now.
inline Position classicStart() {
    Position position;
    position.setRuleset(Ruleset::Classic);
    return position;
}

void testGeometry() {
    CHECK(kCells == 81);
    CHECK(index(0, 0) == 0);
    CHECK(index(8, 8) == 80);
    CHECK(squareName(0) == "a1");
    CHECK(squareName(8) == "i1");
    CHECK(squareName(80) == "i9");
    int cell = -1;
    CHECK(parseSquare("a1", cell) && cell == 0);
    CHECK(parseSquare("i9", cell) && cell == 80);
    CHECK(parseSquare("E5", cell) && cell == index(4, 4));
    CHECK(!parseSquare("j1", cell));
    CHECK(!parseSquare("a0", cell));
    CHECK(!parseSquare("a10", cell));
}

void testInitialPosition() {
    Position position = classicStart();
    CHECK(position.ply() == 0);
    CHECK(position.sideToMove() == Player::Red);
    CHECK(!position.gameOver());
    CHECK(position.rawScore() == 0);
    CHECK(position.remainingPieces(Player::Red) == 27);
    CHECK(position.remainingPieces(Player::Black) == 27);
    for (int number = 1; number <= 9; ++number) {
        CHECK(position.remaining(Player::Red, number) == 3);
        CHECK(position.remaining(Player::Black, number) == 3);
    }
    CHECK(position.legalMoves().size() == 81u * 9u);
    CHECK(position.validate());
}

void testSingleStoneScores() {
    {
        Position position = classicStart();
        CHECK(position.play(Move(index(4, 4), 5)));
        CHECK(position.rawScore() == 5); // centre + four orthogonal neighbours
        CHECK(position.sideToMove() == Player::Black);
        CHECK(position.remaining(Player::Red, 5) == 2);
        CHECK(position.stoneAt(index(4, 4)) == 5);
        CHECK(position.validate());
    }
    {
        Position position = classicStart();
        CHECK(position.play(Move(index(0, 0), 1)));
        CHECK(position.rawScore() == 3); // corner
        CHECK(position.scoreCell(index(0, 0)) == 1);
        CHECK(position.scoreCell(index(1, 0)) == 1);
        CHECK(position.scoreCell(index(0, 1)) == 1);
        CHECK(position.scoreCell(index(1, 1)) == 0);
        CHECK(position.validate());
    }
    {
        Position position = classicStart();
        CHECK(position.play(Move(index(8, 4), 9)));
        CHECK(position.rawScore() == 4); // edge
        CHECK(position.scoreCell(index(0, 0)) == 0);
        CHECK(position.validate());
    }
}

void testNoWrapAndSign() {
    Position position = classicStart();
    CHECK(position.play(Move(index(8, 0), 1))); // red i1
    CHECK(position.rawScore() == 3);
    CHECK(position.scoreCell(index(0, 0)) == 0); // a1 must not see i1
    CHECK(position.scoreCell(index(8, 8)) == 0); // i9 must not see i1
    CHECK(position.play(Move(index(0, 0), 1)));   // black a1
    CHECK(position.rawScore() == 0);
    CHECK(position.scoreCell(index(0, 0)) == -1); // sees only the black a1
    CHECK(position.scoreCell(index(1, 0)) == -1);
    CHECK(position.scoreCell(index(8, 0)) == 1);
    CHECK(position.validate());
}

void testEqualMaxima() {
    Position position = classicStart();
    CHECK(position.play(Move(index(4, 4), 5))); // red 5 e5
    CHECK(position.play(Move(index(4, 3), 5))); // black 5 e4
    CHECK(position.scoreCell(index(4, 4)) == 0);
    CHECK(position.scoreCell(index(4, 3)) == 0);
    CHECK(position.scoreCell(index(4, 5)) == 1);
    CHECK(position.scoreCell(index(3, 4)) == 1);
    CHECK(position.scoreCell(index(5, 4)) == 1);
    CHECK(position.scoreCell(index(3, 3)) == -1);
    CHECK(position.scoreCell(index(5, 3)) == -1);
    CHECK(position.rawScore() == 0);
    CHECK(position.validate());
}

void testBitboardsAndHash() {
    Position position = classicStart();
    int empty = 0;
    for (int cell = 0; cell < kCells; ++cell) empty += position.emptyMask().test(cell) ? 1 : 0;
    CHECK(empty == 81);
    const std::uint64_t initialKey = position.key();
    CHECK(position.play(Move(index(4, 4), 7)));
    CHECK(!position.emptyMask().test(index(4, 4)));
    CHECK(position.occupiedMask(Player::Red).test(index(4, 4)));
    CHECK(position.numberMask(Player::Red, 7).test(index(4, 4)));
    CHECK(position.numberMask(Player::Black, 7).none());
    CHECK(position.key() != initialKey);
    const std::uint64_t afterPlay = position.key();
    CHECK(position.undo());
    CHECK(position.key() == initialKey);
    CHECK(position.emptyMask().test(index(4, 4)));
    CHECK(afterPlay != initialKey);
    CHECK(position.validate());
}

void testLegalityAndDuplicates() {
    Position position = classicStart();
    const auto initial = position.legalMoves();
    CHECK(std::count(initial.begin(), initial.end(), Move(index(0, 0), 1)) == 1);
    CHECK(!position.play(Move(index(0, 0), 0)));
    CHECK(!position.play(Move(index(0, 0), 10)));
    CHECK(position.play(Move(index(0, 0), 1)));
    CHECK(!position.play(Move(index(0, 0), 1))); // occupied
    CHECK(position.play(Move(index(1, 0), 1)));   // black 1
    CHECK(position.play(Move(index(1, 1), 1)));   // second red 1
    CHECK(position.play(Move(index(2, 0), 1)));
    CHECK(position.play(Move(index(2, 1), 1)));   // third red 1
    CHECK(position.play(Move(index(3, 0), 1)));   // third black 1
    CHECK(position.sideToMove() == Player::Red);
    CHECK(position.remaining(Player::Red, 1) == 0);
    CHECK(position.remaining(Player::Black, 1) == 0);
    for (Move move : position.legalMoves()) CHECK(move.number != 1);
    CHECK(position.legalMoves().size() == 8u * 75u);
    CHECK(position.validate());
}

void testUndo() {
    Position position = classicStart();
    CHECK(position.play(Move(index(4, 4), 7)));
    CHECK(position.play(Move(index(0, 0), 1)));
    CHECK(position.play(Move(index(8, 8), 9)));
    CHECK(position.rawScore() == 5); // +5 centre, -3 corner, +3 far corner
    CHECK(position.undo());
    CHECK(position.rawScore() == 2); // red 7 at centre, black 1 at a1
    CHECK(position.sideToMove() == Player::Red);
    CHECK(position.undo());
    CHECK(position.rawScore() == 5); // only red 7 at centre
    CHECK(position.sideToMove() == Player::Black);
    CHECK(position.undo());
    CHECK(position.rawScore() == 0);
    CHECK(position.ply() == 0);
    CHECK(position.history().empty());
    CHECK(!position.undo());
    CHECK(position.validate());
}

void testRandomGamesAndUndo() {
    std::mt19937 rng(20261121);
    for (int game = 0; game < 8; ++game) {
        Position position = classicStart();
        while (!position.gameOver()) {
            auto moves = position.legalMoves();
            CHECK(!moves.empty());
            std::uniform_int_distribution<std::size_t> pick(0, moves.size() - 1);
            CHECK(position.play(moves[pick(rng)]));
            CHECK(position.validate());
        }
        CHECK(position.ply() == kTotalMoves);
        CHECK(position.remainingPieces(Player::Red) == 0);
        CHECK(position.remainingPieces(Player::Black) == 0);
        CHECK(position.rawScore() >= -73 && position.rawScore() <= 73);
        CHECK(position.outcome(1.0) != Outcome::Ongoing);

        // Undo the whole game and check that the initial position is restored.
        while (position.undo()) {
            CHECK(position.validate());
        }
        CHECK(position.ply() == 0);
        CHECK(position.rawScore() == 0);
        CHECK(position.sideToMove() == Player::Red);
        CHECK(position.remainingPieces(Player::Red) == 27);
        CHECK(position.remainingPieces(Player::Black) == 27);
    }
}

void testAValue() {
    Position position = classicStart();
    CHECK(position.play(Move(index(4, 4), 5)));
    CHECK(position.finalScore(1.0) == 6.0);
    CHECK(position.finalScore(-1.5) == 3.5);
    CHECK(position.outcome(1.0) == Outcome::Ongoing);
    while (!position.gameOver()) {
        const auto moves = position.legalMoves();
        CHECK(position.play(moves.front()));
    }
    CHECK(position.finalScore(1.0) == position.rawScore() + 1.0);
    CHECK(position.finalScore(0.0) == static_cast<double>(position.rawScore()));
    const double raw = position.finalScore(0.0);
    const Outcome expected = raw > 0 ? Outcome::RedWin : (raw < 0 ? Outcome::BlackWin : Outcome::Draw);
    CHECK(position.outcome(0.0) == expected);
}


} // namespace

int runKomiTests();
int runEngineTests();
int runRulesTests();

int main() {
    // A one-line marker per test so a crash points at the test that caused it;
    // stderr is unbuffered enough for this to survive an abnormal exit.
    const bool trace = std::getenv("SHUYI_TEST_TRACE") != nullptr;
    const auto step = [&](const char* name) {
        if (trace) std::cerr << "[test] " << name << std::endl;
    };
    step("geometry");
    testGeometry();
    step("InitialPosition");
    testInitialPosition();
    step("SingleStoneScores");
    testSingleStoneScores();
    step("NoWrapAndSign");
    testNoWrapAndSign();
    step("EqualMaxima");
    testEqualMaxima();
    step("LegalityAndDuplicates");
    testLegalityAndDuplicates();
    step("Undo");
    testUndo();
    step("RandomGamesAndUndo");
    testRandomGamesAndUndo();
    step("AValue");
    testAValue();
    step("Ai");
    step("rules");
    failures += runRulesTests();
    step("komi");
    const int komiFailures = runKomiTests();
    failures += komiFailures;
    step("engine");
    failures += runEngineTests();
    if (failures == 0) {
        std::cout << "All tests passed.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " test(s) failed.\n";
    return EXIT_FAILURE;
}
