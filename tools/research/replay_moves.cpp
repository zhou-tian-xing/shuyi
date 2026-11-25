// Replay a move list from stdin ("play red a1 3" lines) and report, after every
// ply, whether the position is over and how many legal moves the side to move
// has.  Used to explain a front-end/engine disagreement about "search is not
// available in this position".
#include "shuyi/position.h"

#include <iostream>
#include <string>

using namespace shuyi;

int main() {
    Position position;
    position.setRuleset(Ruleset::NoPass);
    std::string line;
    int ply = 0;
    while (std::getline(std::cin, line)) {
        if (line.rfind("play ", 0) != 0) continue;
        char colour[16] = {0};
        char cellName[8] = {0};
        int number = 0;
        if (std::sscanf(line.c_str(), "play %15s %7s %d", colour, cellName, &number) != 3) continue;
        int cell = -1;
        parseSquare(cellName, cell);
        const Move move(cell, static_cast<std::uint8_t>(number));
        if (!position.isLegal(move)) {
            std::cout << "ply " << (ply + 1) << " ILLEGAL " << cellName << " " << number
                      << " (side " << (position.sideToMove() == Player::Red ? "red" : "black") << ")\n";
            return 1;
        }
        position.play(move);
        ++ply;
        std::cout << "ply " << ply << " stones " << position.stones() << " over "
                  << (position.gameOver() ? 1 : 0) << " legal "
                  << position.legalMoves().size() << "\n";
    }
    std::cout << "final: over " << (position.gameOver() ? 1 : 0) << " side "
              << (position.sideToMove() == Player::Red ? "red" : "black") << "\n";
    return 0;
}
