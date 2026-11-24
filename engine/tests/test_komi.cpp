// Verifies the computer-generated score witnesses used by the A-value analysis.
#include "shuyi/position.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace shuyi;

namespace {
int failures = 0;

// These tests exercise the base game (scoring, witnesses), which is rule-set
// independent; the placements they use are only legal under Classic.
inline Position classicStart() {
    Position position;
    position.setRuleset(Ruleset::Classic);
    return position;
}

void checkImpl(bool condition, const char* expression, int line) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL line " << line << ": " << expression << "\n";
    }
}

#define CHECK(expr) checkImpl((expr), #expr, __LINE__)

struct Witness {
    int score = 0;
    std::array<std::string, 9> rows{};
};

bool parseWitnesses(const std::string& path, std::vector<Witness>& out, std::string& error) {
    std::ifstream in{std::filesystem::path(path)};
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        lines.push_back(line);
    }
    std::size_t at = 0;
    while (at < lines.size()) {
        std::istringstream scoreStream(lines[at]);
        Witness witness;
        if (!(scoreStream >> witness.score)) {
            error = "expected score at witness line " + std::to_string(at);
            return false;
        }
        ++at;
        if (at + 9 > lines.size()) {
            error = "truncated witness board";
            return false;
        }
        for (int y = 0; y < 9; ++y) witness.rows[y] = lines[at++];
        out.push_back(witness);
    }
    return true;
}

bool buildPosition(const Witness& witness, Position& position, std::string& error) {
    std::array<std::vector<Move>, 2> moves;
    std::array<std::array<int, 10>, 2> counts{};
    for (int y = 0; y < 9; ++y) {
        std::istringstream row(witness.rows[y]);
        std::string token;
        for (int x = 0; x < 9; ++x) {
            if (!(row >> token)) {
                error = "short row in witness " + std::to_string(witness.score);
                return false;
            }
            if (token == ".") continue;
            if (token.size() != 2 || (token[0] != 'R' && token[0] != 'B') ||
                token[1] < '1' || token[1] > '9') {
                error = "bad token " + token;
                return false;
            }
            const int side = token[0] == 'R' ? 0 : 1;
            const int number = token[1] - '0';
            ++counts[side][number];
            moves[side].emplace_back(index(x, y), number);
        }
        std::string extra;
        if (row >> extra) {
            error = "long row in witness " + std::to_string(witness.score);
            return false;
        }
    }
    for (int side = 0; side < 2; ++side) {
        if (moves[side].size() != 27) {
            error = "witness " + std::to_string(witness.score) + " has wrong stone count";
            return false;
        }
        for (int n = 1; n <= 9; ++n) {
            if (counts[side][n] != 3) {
                error = "witness " + std::to_string(witness.score) + " has wrong label count";
                return false;
            }
        }
    }

    // Any complete final placement can be reached: red and black simply play
    // their lists in any order.  This also exercises real alternating play.
    position.reset();
    // These witnesses are unordered final placements, so they are only reachable
    // when every empty cell accepts every number (Classic).  Under a blocking
    // rule the placement order would have to be legal too, which is a different
    // question the witnesses do not answer.
    position.setRuleset(Ruleset::Classic);
    for (int i = 0; i < 27; ++i) {
        if (!position.play(moves[0][i]) || !position.play(moves[1][i])) {
            error = "witness is not reachable by alternating legal moves";
            return false;
        }
    }
    if (!position.gameOver()) {
        error = "witness does not finish the game";
        return false;
    }
    return true;
}

} // namespace

int runKomiTests() {
    const std::string path = std::string(SHUYI_SOURCE_DIR) + "/data/score_witnesses.txt";
    std::vector<Witness> witnesses;
    std::string error;
    CHECK(parseWitnesses(path, witnesses, error));
    if (!error.empty()) std::cerr << error << "\n";
    CHECK(witnesses.size() == 147);

    std::array<bool, 147> seen{};
    std::set<int> drawableA;
    bool sawMinusOne = false;
    for (const Witness& witness : witnesses) {
        if (witness.score < -73 || witness.score > 73) {
            CHECK(false);
            continue;
        }
        seen[witness.score + 73] = true;
        Position position = classicStart();
        if (!buildPosition(witness, position, error)) {
            std::cerr << error << "\n";
            CHECK(false);
            continue;
        }
        CHECK(position.validate());
        CHECK(position.rawScore() == witness.score);
        CHECK(position.finalScore(0.0) == static_cast<double>(witness.score));

        const double drawingA = -static_cast<double>(witness.score);
        drawableA.insert(witness.score * -1);
        CHECK(position.finalScore(drawingA) == 0.0);
        CHECK(position.outcome(drawingA) == Outcome::Draw);
        if (witness.score == -1) {
            sawMinusOne = true;
            CHECK(position.outcome(1.0) == Outcome::Draw);
        }

        // A non-integral A can never cancel an integral score.
        CHECK(position.finalScore(0.5) != 0.0);
        CHECK(position.outcome(0.5) != Outcome::Draw);
    }

    for (int score = -73; score <= 73; ++score) CHECK(seen[score + 73]);
    for (int a = -73; a <= 73; ++a) CHECK(drawableA.count(a) == 1);
    CHECK(sawMinusOne);

    // Outside the achievable score interval no integer A can draw.
    Position any;
    for (const Witness& witness : witnesses) {
        Position position = classicStart();
        if (buildPosition(witness, position, error)) {
            any = position;
            break;
        }
    }
    CHECK(any.finalScore(74.0) != 0.0);
    CHECK(any.outcome(74.0) != Outcome::Draw);
    CHECK(any.finalScore(-74.0) != 0.0);
    CHECK(any.outcome(-74.0) != Outcome::Draw);

    // A compact hand-checkable construction for P=-1, used in the rules/A值
    // documentation: red 1 at the isolated corner a1 (+3), black 1 at the
    // isolated edge i2 (-4), and 26 colour-swapped mirror pairs contributing 0.
    Witness explicitMinusOne;
    explicitMinusOne.score = -1;
    // buildPosition() consumes rows from y=0 (rank 1) upward; reverse the
    // visual top-to-bottom diagram used in the documentation.
    explicitMinusOne.rows = {
        "R1 . . R1 R1 R2 R2 . .",
        ". . R2 R3 R3 R3 . . B1",
        ". R4 R4 R4 R5 R5 R5 . .",
        "R6 R6 R6 R7 R7 R7 R8 R8 .",
        "R8 R9 R9 R9 . B9 B9 B9 B8",
        ". B8 B8 B7 B7 B7 B6 B6 B6",
        ". . B5 B5 B5 B4 B4 B4 .",
        ". . . B3 B3 B3 B2 . .",
        ". . B2 B2 B1 B1 . . .",
    };
    Position explicitPosition;
    if (buildPosition(explicitMinusOne, explicitPosition, error)) {
        CHECK(explicitPosition.rawScore() == -1);
        CHECK(explicitPosition.outcome(1.0) == Outcome::Draw);
        CHECK(explicitPosition.finalScore(1.0) == 0.0);
    } else {
        std::cerr << error << "\n";
        CHECK(false);
    }

    return failures;
}
