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
    // Per-cell has_score: a solved cell (a1) has an exact margin, a sampled one
    // without a score head (c3) has none, and e5's average came from a score head.
    "\"cells\":[{\"cell\":\"e5\",\"number\":3,\"winrate\":0.5312,\"visits\":210,"
    "\"score\":1.234,\"has_score\":true,\"proven\":false},"
    "{\"cell\":\"a1\",\"number\":1,\"winrate\":0.0,\"visits\":7,\"score\":-12.0,"
    "\"has_score\":true,\"proven\":true},"
    "{\"cell\":\"c3\",\"number\":2,\"winrate\":0.48,\"visits\":3,\"score\":0.0,"
    "\"has_score\":false,\"proven\":false}]}";

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
    check(view.cells[sq("e5")].hasScore, "a score-head cell carries its average");
    check(view.cells[sq("a1")].proven && view.cells[sq("a1")].hasScore &&
              view.cells[sq("a1")].averageScore == -12.0,
          "a solved cell carries its exact margin");
    check(!view.cells[sq("c3")].hasScore, "a cell without a score source has none");
    check(!view.hasCell[sq("b2")], "a cell without data stays empty");
}

// The exact endgame solver answers with `visits: 0` and a proven, exact margin on
// every legal move: no simulation ran, the answer is a full minimax.  The board
// must still draw those cells -- requiring a visit count here threw the whole
// solved position away and left the board blank at the end of the game.
const char* kExactAnalysis =
    "{\"engine\":\"mcts_nn\",\"turn\":\"red\",\"backend\":\"cpu\","
    "\"visits\":0,\"nodes\":41,\"depth\":1,\"proven\":true,"
    "\"winrate\":1.0,\"has_score\":true,\"score\":3.0,"
    "\"best\":\"e5:3\",\"elapsed_ms\":1,\"nn\":\"b4c32-s42012-d595499.bin\","
    "\"a_input\":1,"
    "\"cells\":[{\"cell\":\"e5\",\"number\":3,\"winrate\":1.0,\"visits\":0,"
    "\"score\":3.0,\"has_score\":true,\"proven\":true},"
    "{\"cell\":\"a1\",\"number\":1,\"winrate\":0.0,\"visits\":0,"
    "\"score\":-12.0,\"has_score\":true,\"proven\":true},"
    "{\"cell\":\"c3\",\"number\":2,\"winrate\":0.5,\"visits\":0,"
    "\"score\":0.0,\"has_score\":true,\"proven\":true}]}";

void testExactAnalysisCells() {
    EngineAnalysisView view;
    std::string error;
    check(parseAnalysis(kExactAnalysis, view, &error), "an exact solve must parse");
    check(view.valid && view.proven, "the position is proven");
    check(view.visits == 0, "an exact solve carries no visits");
    check(view.best.valid() && view.best.cell == sq("e5"), "best move survives");
    check(view.hasCell[sq("e5")] && view.hasCell[sq("a1")] && view.hasCell[sq("c3")],
          "every solved cell is present");
    check(view.cells[sq("e5")].hasData, "a solved cell with zero visits is data");
    check(view.cells[sq("a1")].hasData, "a losing solved cell is data too");
    check(view.cells[sq("a1")].proven && view.cells[sq("a1")].averageScore == -12.0,
          "the losing cell keeps its exact margin");
    check(view.cells[sq("c3")].winrate == 0.5 && view.cells[sq("c3")].hasScore,
          "a drawn cell is 50% with an exact zero margin");
}

void testRankAnalysisCells() {
    const char* payload =
        R"({"engine":"mcts_nn","best":"b1:1","visits":27382,"cells":[)"
        R"({"cell":"b1","number":1,"winrate":0.18,"visits":166},)"
        R"({"cell":"c7","number":1,"winrate":0.25,"visits":5900},)"
        R"({"cell":"f5","number":1,"winrate":0.30,"visits":4500},)"
        R"({"cell":"a2","number":1,"winrate":1.0,"visits":1},)"
        R"({"cell":"d1","number":1,"winrate":1.0,"visits":0},)"
        R"({"cell":"e3","number":1,"winrate":1.0,"visits":5000}]})";
    EngineAnalysisView view;
    Position position;
    check(parseAnalysis(payload, view, nullptr), "mixed search snapshot parses");
    check(position.play(Move(sq("e3"), 2)), "occupy the stale suggestion's cell");
    check(rankAnalysisCells(view, position) ==
              std::vector<int>{sq("f5"), sq("c7"), sq("a2"), sq("b1")},
          "Wilson lower bound beats the engine's best, more visits alone and a lucky sample");

    view.cells[sq("d1")].proven = true;
    const auto solvedOrder = rankAnalysisCells(view, position);
    check(!solvedOrder.empty() && solvedOrder.front() == sq("d1"),
          "a proven win with zero visits uses its exact win rate");

    // A proof outranks a guess even when the proof is a loss: the panel must show
    // the fact, not hide it behind a sampled estimate.  Within the solved cells the
    // ordering is still by rate and then by margin.
    view.cells[sq("b1")].proven = true;
    view.cells[sq("b1")].winrate = 0.0;
    const auto losingOrder = rankAnalysisCells(view, position);
    check(losingOrder.size() >= 2 && losingOrder[0] == sq("d1") && losingOrder[1] == sq("b1"),
          "a proven loss ranks above every estimate, below a proven win");

    // A solved cell with no visits and no hasData flag is still a row: the endgame
    // solver answers without simulating anything.
    view.hasCell[sq("i9")] = true;
    view.cells[sq("i9")] = CellAnalysisView{};
    view.cells[sq("i9")].move = Move(sq("i9"), 1);
    view.cells[sq("i9")].proven = true;
    view.cells[sq("i9")].winrate = 1.0;
    const auto noDataOrder = rankAnalysisCells(view, position);
    check(noDataOrder.size() == losingOrder.size() + 1 && noDataOrder[1] == sq("i9"),
          "a proven cell without visits is still shown, and ranks with the solved moves");
}

void testRankTiedAnalysisCells() {
    EngineAnalysisView view;
    check(parseAnalysis(kExactAnalysis, view, nullptr), "exact search snapshot parses");
    auto& other = view.cells[sq("a1")];
    other.winrate = 1.0;
    other.averageScore = 4.0;
    check(rankAnalysisCells(view, Position{}) ==
              std::vector<int>{sq("a1"), sq("e5"), sq("c3")},
          "proven wins are ranked by margin when their rates match");
    other.averageScore = 3.0;
    check(rankAnalysisCells(view, Position{}) ==
              std::vector<int>{sq("e5"), sq("a1"), sq("c3")},
          "an exact tie keeps the engine's representative");
}

void testDistinctSymmetricCells() {
    // An empty board is symmetric under all eight isometries, so the four
    // corners holding the same number are one move.  A number used on a
    // different orbit is a different move even when the cells are symmetric.
    const char* payload =
        R"({"engine":"mcts_nn","turn":"red","visits":400,"winrate":0.5,"best":"a1:4",)"
        R"("cells":[{"cell":"a1","number":4,"winrate":0.52,"visits":100},)"
        R"({"cell":"i1","number":4,"winrate":0.52,"visits":100},)"
        R"({"cell":"a9","number":4,"winrate":0.52,"visits":100},)"
        R"({"cell":"i9","number":4,"winrate":0.52,"visits":100},)"
        R"({"cell":"b1","number":5,"winrate":0.48,"visits":90}]})";
    EngineAnalysisView view;
    check(parseAnalysis(payload, view, nullptr), "symmetric fixture parses");
    const Position empty;
    const auto ranked = rankAnalysisCells(view, empty);
    check(ranked.size() == 5, "the ranking keeps every sampled cell for the board");
    const auto rows = distinctSymmetricCells(ranked, view, empty);
    check(rows.size() == 2, "one row per move, not per symmetric cell");
    check(view.cells[rows[0]].move.number == 4 && view.cells[rows[1]].move.number == 5,
          "the surviving rows are the two distinct moves");

    // One stone breaks the symmetry: only the identity is left, and the cells
    // are genuinely different moves again.  The stone has to sit off every axis
    // -- a centre or a corner stone still fixes part of the board.
    Position played;
    check(played.play(Move(sq("d3"), 9)), "occupy an off-axis cell");
    check(distinctSymmetricCells(ranked, view, played).size() == 5,
          "an asymmetric position keeps every cell");
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
    testExactAnalysisCells();
    testRankAnalysisCells();
    testRankTiedAnalysisCells();
    testDistinctSymmetricCells();
    testBrokenInput();
    testMoveTokens();
    if (failures == 0) std::printf("engine view tests passed\n");
    return failures == 0 ? 0 : 1;
}
