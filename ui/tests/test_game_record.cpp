// Unit tests for the record tree: navigation, variation reuse, deletion, the
// saved text, and the colour delta the tree view draws from.
//
// These are the rules the window depends on and that nothing else checks: a tree
// that quietly loses a branch, or loses the analysis of one, cannot be seen in a
// screenshot.
#include "game_record.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

using namespace shuyi;

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (condition) return;
    std::printf("FAIL %s\n", what);
    ++failures;
}

Move moveAt(const char* label) {
    Move move;
    parseMoveLabel(label, move);
    return move;
}

EngineAnalysisView evaluated(double winrate, double score = 0.0) {
    EngineAnalysisView view;
    view.valid = true;
    view.winrate = winrate;
    view.averageScore = score;
    view.hasScore = true;
    view.visits = 100;
    return view;
}

void testNavigation() {
    GameRecord record;
    check(record.count() == 1, "a fresh record is just the root");
    check(record.depth(0) == 0, "the root is depth 0");
    check(record.up() == -1, "there is nothing above the root");
    check(record.down() == -1, "there is nothing below the root");

    const int a = record.play(0, moveAt("a4-2"));
    const int b = record.play(a, moveAt("b2-5"));
    check(record.count() == 3, "two moves make three nodes");
    check(record.current() == b, "the last move is current");
    check(record.depth(b) == 2, "depth counts the plies");
    check(record.line(b) == std::vector<int>({a, b}), "the line is root -> node");
    check(record.up() == a, "up goes to the parent");
    check(record.up() == 0, "up again reaches the root");
    check(record.down() == a, "down follows the trunk");
    check(record.select(b) == b, "select moves to a node");
    check(record.select(99) == b, "select ignores an unknown node");
}

void testVariationReuse() {
    GameRecord record;
    const int a = record.play(0, moveAt("a4-2"));
    const int b = record.play(a, moveAt("b2-5"));
    record.setAnalysis(b, evaluated(0.75));
    // A different move at the same node is a new branch, not a replacement.
    const int c = record.play(a, moveAt("c3-1"));
    check(record.childrenOf(a).size() == 2, "two children at the branch point");
    check(record.childrenOf(a).front() == b, "the trunk stays first");
    check(record.current() == c, "the new move becomes current");

    // Replaying the trunk reaches the *same* node again: the cached analysis is
    // the reason the tree exists, so it must survive the detour.
    check(record.play(a, moveAt("b2-5")) == b, "an existing move reuses its node");
    check(record.count() == 4, "no duplicate node was created");
    check(record.node(b).analysis.valid && record.node(b).analysis.winrate == 0.75,
          "the reused node keeps its analysis");

    // The trunk is the line played first, and Down follows it.
    record.select(a);
    check(record.down() == b, "down follows the first child");
    check(record.line(b) == std::vector<int>({a, b}), "the trunk's line");
}

void testErase() {
    GameRecord record;
    const int a = record.play(0, moveAt("a4-2"));
    const int b = record.play(a, moveAt("b2-5"));
    record.play(b, moveAt("c3-1"));
    const int other = record.play(a, moveAt("d4-7"));
    record.setAnalysis(other, evaluated(0.9));
    // Ids are remapped by the rebuild, so the assertions go through the tree, not
    // through an id captured before the delete.
    check(!record.erase(0), "the root cannot be deleted");
    check(record.erase(b), "a branch can be deleted");
    check(record.count() == 3, "the whole subtree went with it");  // root, a4-2, d4-7
    const int parent = record.current();
    check(record.node(parent).move == moveAt("a4-2"), "the parent becomes current");
    check(record.depth(parent) == 1, "and it is the move above the deleted one");
    check(record.childrenOf(parent).size() == 1, "the surviving branch keeps its place");
    const int survivor = record.childrenOf(parent).front();
    check(record.node(survivor).move == moveAt("d4-7"), "the surviving node is unchanged");
    check(record.node(survivor).analysis.valid, "and keeps its analysis");
}

void testSaveLoad() {
    GameRecord record;
    const int a = record.play(0, moveAt("a4-2"));
    const int b = record.play(a, moveAt("b2-5"));
    // An analysis rich enough to exercise every field the file carries: a proven
    // verdict, a score, a cell with its own margin, and a candidate.
    EngineAnalysisView view = evaluated(0.62, 1.5);
    view.visits = 1200;
    view.nodes = 2400000;
    view.elapsedMs = 950;
    view.backend = "cuda";
    view.best = moveAt("b2-5");
    CellAnalysisView cell;
    cell.move = moveAt("b2-5");
    cell.winrate = 0.62;
    cell.visits = 1200;
    cell.averageScore = 1.5;
    cell.hasData = true;
    cell.hasScore = true;
    view.cells[cell.move.cell] = cell;
    view.hasCell[cell.move.cell] = true;
    CellAnalysisView solved;
    solved.move = moveAt("c3-4");
    solved.winrate = 1.0;
    solved.proven = true;
    solved.hasData = true;
    solved.hasScore = true;
    solved.averageScore = 7.0;
    view.cells[solved.move.cell] = solved;
    view.hasCell[solved.move.cell] = true;
    AnalysisCandidateView candidate;
    candidate.move = moveAt("b2-5");
    candidate.winrate = 0.62;
    candidate.visits = 1200;
    candidate.averageScore = 1.5;
    view.candidates.push_back(candidate);
    record.setAnalysis(a, view);
    record.setAnalysis(b, evaluated(0.6, 2.0));
    record.play(a, moveAt("c3-1"));            // a variation beside the trunk
    record.play(b, moveAt("d4-7"));
    const std::string text = record.save(Ruleset::Blocking);
    check(text.find("#node a4:2 wr 0.6200") != std::string::npos,
          "the analysis is written with its move path");
    check(text.find("#cell a4:2 b2:5 0.6200 1200") != std::string::npos, "cells are written");

    GameRecord loaded;
    Ruleset ruleset = Ruleset::Classic;
    std::string error;
    check(loaded.load(text, &ruleset, &error), "a saved record loads back");
    check(ruleset == Ruleset::Blocking, "the rule set travels with the record");
    check(loaded.count() == record.count(), "every node came back");
    // Structure: root -> a4-2, then two children (b2-5 trunk, c3-1 variation),
    // and b2-5 has d4-7 under it.
    const int root = loaded.root();
    check(loaded.childrenOf(root).size() == 1, "one first move");
    const int first = loaded.childrenOf(root).front();
    check(loaded.node(first).move == moveAt("a4-2"), "the first move survives");
    check(loaded.childrenOf(first).size() == 2, "the variation survives");
    const int trunk = loaded.childrenOf(first).front();
    check(loaded.node(trunk).move == moveAt("b2-5"), "the trunk is still first");
    check(loaded.childrenOf(trunk).size() == 1 &&
              loaded.node(loaded.childrenOf(trunk).front()).move == moveAt("d4-7"),
          "the continuation under the trunk survives");
    check(loaded.node(loaded.childrenOf(first).back()).move == moveAt("c3-1"),
          "the variation is still a sibling");
    // The analysis came back with it, on the right nodes.
    const EngineAnalysisView& back = loaded.node(first).analysis;
    check(back.valid, "the analysis survives");
    check(back.visits == 1200 && back.nodes == 2400000 && back.elapsedMs == 950,
          "effort survives");
    check(std::fabs(back.winrate - 0.62) < 1e-9 && back.hasScore &&
              std::fabs(back.averageScore - 1.5) < 1e-9,
          "the verdict survives");
    check(back.backend == "cuda" && back.best == moveAt("b2-5"), "backend and best move survive");
    check(back.hasCell[cell.move.cell] && back.cells[cell.move.cell].visits == 1200,
          "a cell survives");
    check(back.hasCell[solved.move.cell] && back.cells[solved.move.cell].proven &&
              std::fabs(back.cells[solved.move.cell].averageScore - 7.0) < 1e-9,
          "a solved cell keeps its exact margin");
    check(back.candidates.size() == 1 && back.candidates.front().move == moveAt("b2-5"),
          "candidates survive");
    check(loaded.node(trunk).analysis.valid, "the other analysed node came back too");
    // The text is canonical: saving the loaded tree gives the same bytes.
    check(loaded.save(Ruleset::Blocking) == text, "save/load round trips to the same text");
    // Loading starts at the end of the main line, not at the last token read.
    check(loaded.current() == loaded.childrenOf(trunk).front(),
          "the current node is the end of the trunk");

    // The file is still readable by a reader that knows only the moves: the
    // analysis lines are comments, so dropping them gives the old record.
    std::string plain;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.rfind("#node ", 0) == 0 || line.rfind("#cell ", 0) == 0 ||
            line.rfind("#cand ", 0) == 0) {
            continue;
        }
        plain += line;
        plain += "\n";
    }
    GameRecord noAnalysis;
    check(noAnalysis.load(plain, nullptr, &error), "the same file loads without its analysis");
    check(noAnalysis.count() == record.count() && !noAnalysis.node(first).analysis.valid,
          "the moves are all there and no analysis is invented");

    GameRecord broken;
    check(!broken.load("a4-2 (b2-5", nullptr, &error), "an unbalanced '(' is refused");
    check(!error.empty(), "the refusal has a reason");
    check(!broken.load("a4-2 zz-1", nullptr, &error), "a bad move is refused");
    check(!broken.load("# nothing here\n", nullptr, &error), "an empty record is refused");
    check(broken.load("#node a4:2 wr 0.5\nmoves a4:2\n", nullptr, &error),
          "an analysis line for a move in the record is accepted");
    check(!broken.load("moves a4:2\n#node b2:5 wr 0.5\n", nullptr, &error),
          "an analysis line for a move that is not there is refused");

    // A bare move list without a header is accepted.
    GameRecord bare;
    check(bare.load("a4-2 b2-5", nullptr, &error), "a bare move list loads");
    check(bare.count() == 3, "three nodes from two moves");
}

void testPasses() {
    GameRecord record;
    const int a = record.play(0, moveAt("pass"));
    check(record.node(a).move == moveAt("pass"), "a pass is a move");
    check(record.save(Ruleset::Blocking).find("pass") != std::string::npos,
          "a pass is written out");
    GameRecord loaded;
    check(loaded.load(record.save(Ruleset::Blocking), nullptr, nullptr), "a pass loads back");
    check(loaded.node(loaded.childrenOf(0).front()).move == moveAt("pass"), "and stays a pass");
}

void testMoveVerdict() {
    GameRecord record;
    const int a = record.play(0, moveAt("a4-2"));
    const int b = record.play(a, moveAt("b2-5"));
    // The parent predicted 0.6 for that move (from the mover's point of view) and
    // 0.7 for the position itself.
    EngineAnalysisView parentView = evaluated(0.7);
    AnalysisCandidateView candidate;
    candidate.move = moveAt("b2-5");
    candidate.winrate = 0.6;
    parentView.candidates.push_back(candidate);
    record.setAnalysis(a, parentView);

    MoveVerdict verdict;
    record.setAnalysis(b, evaluated(0.25));
    check(moveVerdict(record.node(a), record.node(b), &verdict), "both sides have numbers");
    // The child is the opponent's view: 1 - 0.25 = 0.75 for the mover, i.e. 0.15
    // better than the 0.6 the parent predicted.
    check(std::fabs(verdict.delta - 0.15) < 1e-9, "a better-than-predicted move is positive");
    check(verdict.scored, "the engine had a number for this move");

    record.setAnalysis(b, evaluated(0.9));
    check(moveVerdict(record.node(a), record.node(b), &verdict), "still computable");
    check(std::fabs(verdict.delta + 0.5) < 1e-9, "a worse-than-predicted move is negative");

    // A move the engine never scored is compared with the position it was played
    // from: 1 - 0.9 = 0.1 against the parent's own 0.7.
    const int other = record.play(a, moveAt("c3-1"));
    record.setAnalysis(other, evaluated(0.9));
    check(moveVerdict(record.node(a), record.node(other), &verdict),
          "an unscored move still has a baseline");
    check(!verdict.scored, "the engine had no number for this move");
    check(std::fabs(verdict.delta + 0.6) < 1e-9, "the position's own verdict is the baseline");

    record.setAnalysis(other, EngineAnalysisView{});
    check(!moveVerdict(record.node(a), record.node(other), &verdict), "no child analysis, no verdict");
    record.setAnalysis(b, evaluated(0.25));
    record.setAnalysis(a, EngineAnalysisView{});
    check(!moveVerdict(record.node(a), record.node(b), &verdict), "no parent analysis, no verdict");
}

void testLabels() {
    check(moveLabel(moveAt("a4-2")) == L"a4-2", "the record spells a move a4-2");
    check(moveProtocolText(moveAt("a4-2")) == "a4:2", "the protocol spells it a4:2");
    check(moveLabel(moveAt("pass")) == L"pass", "a pass is spelled pass");
    Move move;
    check(parseMoveLabel("a4:2", move) && move == moveAt("a4-2"), "the protocol form parses");
    check(!parseMoveLabel("a4", move), "a move without a number is refused");
    check(!parseMoveLabel("a4-0", move), "number 0 is refused");
}

} // namespace

int main() {
    // `SHUYI_TEST_TRACE=1` names each test before it runs, so a crash points at
    // the test that caused it.
    const bool trace = std::getenv("SHUYI_TEST_TRACE") != nullptr;
    const auto step = [&](const char* name) {
        if (trace) std::fprintf(stderr, "[test] %s\n", name);
    };
    step("navigation");
    testNavigation();
    step("variation reuse");
    testVariationReuse();
    step("erase");
    testErase();
    step("save/load");
    testSaveLoad();
    step("passes");
    testPasses();
    step("move verdict");
    testMoveVerdict();
    step("labels");
    testLabels();
    if (failures == 0) std::printf("game record tests passed\n");
    return failures == 0 ? 0 : 1;
}
