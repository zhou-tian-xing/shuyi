// Unit tests for the analysis parser: the engine's JSON, as the window reads it.
//
// The fixture is a real `info` line (trimmed to two cells) from shuyi_engine.
// If this test fails, the window would show an empty board even though the
// engine answered correctly, which is exactly the failure mode worth a test.
#include "engine_view.h"

#include <cstdio>
#include <string>

using namespace shuyi;

namespace {

int sq(const char* name) {
    int cell = -1;
    parseSquare(name, cell);
    return cell;
}

int failures = 0;

void check(bool condition, const char* what) {
    if (condition) return;
    std::printf("FAIL %s\n", what);
    ++failures;
}

const char* kAnalysis =
    "{\"engine\":\"mcts_nn\",\"turn\":\"red\",\"backend\":\"cuda\","
    "\"visits\":512,\"nodes\":984,\"depth\":0,\"proven\":false,"
    "\"winrate\":0.5312,\"has_score\":true,\"score\":1.234,"
    "\"best\":\"e5:3\",\"elapsed_ms\":412,\"nn\":\"b4c32-s34760-d470705.bin\","
    "\"pv\":[\"e5:3\",\"d4:1\"],"
    "\"moves\":[{\"move\":\"e5:3\",\"winrate\":0.5312,\"visits\":210,\"score\":1.234}],"
    "\"cells\":[{\"cell\":\"e5\",\"number\":3,\"winrate\":0.5312,\"visits\":210,"
    "\"score\":1.234,\"proven\":false},"
    "{\"cell\":\"a1\",\"number\":1,\"winrate\":0.42,\"visits\":7,\"score\":0.5,"
    "\"proven\":true}]}";

void testAnalysis() {
    EngineAnalysisView view;
    std::string error;
    check(parseAnalysis(kAnalysis, view, &error), "a real info line must parse");
    check(view.valid, "analysis is valid");
    check(view.kind == "mcts_nn", "engine name");
    check(view.turn == "red", "turn");
    check(view.backend == "cuda", "backend");
    check(view.visits == 512 && view.nodes == 984, "visit counts");
    check(view.proven == false, "not proven");
    check(view.hasScore, "score head present");
    check(view.best.valid() && view.best.cell == sq("e5") && view.best.number == 3,
          "best move");
    check(view.pv == "e5:3,d4:1", "principal variation");
    check(view.candidates.size() == 1 && view.candidates[0].visits == 210, "candidates");
    check(view.hasCell[sq("e5")] && view.cells[sq("e5")].visits == 210,
          "cell e5");
    check(view.hasCell[sq("a1")] && view.cells[sq("a1")].proven,
          "proven cell");
    check(view.cells[sq("e5")].hasScore, "cells inherit has_score");
    check(!view.hasCell[sq("b2")], "a cell without data stays empty");
}

void testBrokenInput() {
    EngineAnalysisView view;
    std::string error;
    check(!parseAnalysis("{\"engine\":", view, &error), "truncated JSON must fail");
    check(!error.empty(), "the failure has a reason");
    check(!parseAnalysis("not json", view, &error), "garbage must fail");
    check(!parseAnalysis("[]", view, &error), "an array is not an analysis");
}

void testMoveTokens() {
    check(parseMoveToken("e5:3").cell == sq("e5"), "compact move");
    check(parseMoveToken("e5 3").cell == sq("e5"), "protocol move");
    check(!parseMoveToken("pass").valid(), "pass is not a placement");
    check(!parseMoveToken("none").valid(), "none is not a move");
    check(!parseMoveToken("e5:10").valid(), "number out of range");
    check(!parseMoveToken("z9:1").valid(), "cell out of board");
}

} // namespace

int main() {
    testAnalysis();
    testBrokenInput();
    testMoveTokens();
    if (failures == 0) std::printf("engine view tests passed\n");
    return failures == 0 ? 0 : 1;
}
