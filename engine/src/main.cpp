// shuyi_engine — the headless 数弈 engine process.
//
// A front end starts this once and then talks to it: commands in, analysis
// lines out.  It has no UI dependency and no notion of a saved game; it only
// knows the position it was last told about and the search tree for it.
#include "shuyi/protocol.h"

#include <iostream>
#include <string>

int main() {
    std::ios::sync_with_stdio(false);
    shuyi::ProtocolSession session;
    std::string line;
    while (std::getline(std::cin, line)) {
        session.handle(line);
        if (session.finished()) break;
    }
    return 0;
}
