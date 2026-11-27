#include "shuyi/endgame.h"
#include "shuyi/engine.h"
#include "shuyi/nn.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

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

EngineOptions optionsFor(std::uint64_t seed) {
    EngineOptions options;
    options.seed = seed;
    options.mctsMaxChildren = 24;
    options.aValue = 1.0;
    return options;
}

// A rating must spend the same search effort on every machine, so every engine
// kind has to honour a visit budget instead of a clock.
// A finished game has an exact result, and near the end the search must reach
// those results instead of trusting the network.  Two things are asserted:
//   * early positions with a head-less network report "no score" (the old code
//     invented one with a +-73 point linear map and the UI showed -15 on an
//     empty board);
//   * with two plies left the search reaches terminals, so a score exists even
//     without a score head.
void testEndgameExactScore() {
    const std::string headless = std::string(SHUYI_SOURCE_DIR) + "/models/shuyi_resnet_v1.bin";
    Position opening = classicStart();

    EngineOptions options = optionsFor(4242);
    options.nnModelPath = headless;
    options.nnValueBlend = 1.0;
    auto engine = makeEngine(EngineKind::MctsNn, options);
    SearchRequest request;
    request.visits = 128;
    const Analysis early = engine->search(opening, request);
    CHECK(early.valid);
    CHECK(!early.hasScore);           // no score head, no terminal: no score at all
    CHECK(early.averageScore == 0.0); // and certainly not a fabricated one

    // Walk a legal game down to two plies left.
    Position late = classicStart();
    std::uint64_t seed = 99;
    while (kTotalMoves - late.ply() > 2) {
        const std::vector<Move> moves = late.legalMoves();
        CHECK(!moves.empty());
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        late.play(moves[seed % moves.size()]);
    }
    CHECK(kTotalMoves - late.ply() == 2);

    EngineOptions lateOptions = optionsFor(4243);
    lateOptions.nnModelPath = headless;
    lateOptions.nnValueBlend = 1.0;
    auto lateEngine = makeEngine(EngineKind::MctsNn, lateOptions);
    SearchRequest lateRequest;
    lateRequest.visits = 300;
    const Analysis finish = lateEngine->search(late, lateRequest);
    CHECK(finish.valid);
    CHECK(late.isLegal(finish.best));
    CHECK(finish.hasScore);
}

// A search must never report a hard 1.0/0.0 value for an unsolved position.  The
// old tree-size cap did exactly that: once the cap was reached it scored the leaf
// with rawScore() as if the game were over, which produced a solid 100% win at
// ~35k visits on an empty board.  There is no cap any more, so this pins the
// invariant instead: without a proof, values stay samples.
void testNoHardValueWithoutProof() {
    const std::string headless = std::string(SHUYI_SOURCE_DIR) + "/models/shuyi_resnet_v1.bin";
    EngineOptions options = optionsFor(24680);
    options.nnModelPath = headless;
    options.nnValueBlend = 1.0;  // value head only: it cannot output exactly 1.0 here
    auto engine = makeEngine(EngineKind::MctsNn, options);
    // classicStart(), not a default Position: the fixture is a classic model and
    // the product default is nopass, where the search would ignore the model and
    // fall back to rollouts -- which still has a score, so this test would pass
    // while measuring nothing it says it measures.
    Position opening = classicStart();
    SearchRequest request;
    request.timeLimitMs = 0;
    request.visits = 600;
    const Analysis analysis = engine->search(opening, request);
    CHECK(analysis.valid);
    CHECK(opening.isLegal(analysis.best));
    CHECK(!analysis.proven);
    CHECK(analysis.winrate < 0.999);
    CHECK(analysis.winrate > 0.001);
    for (const CellAnalysis& info : analysis.cells) {
        CHECK(!info.proven);
        if (info.visits == 0) continue;
        CHECK(info.winrate < 0.999);
        CHECK(info.winrate > 0.001);
        // This network has no score head and nothing here is solved, so there is
        // no score.  Reporting 0.0 would look like an even position (the old
        // behaviour: every cell of a lost position read "0").
        CHECK(!info.hasScore);
    }
}

// A solved position must return the exact result with the right sign, and it
// must stop spending visits once it is solved.  This is the regression test for
// the sign of the terminal value as it travels back up the tree: a loss that
// propagates as a win shows up here as a win rate of 1.0 on a lost position.
// Plain minimax, written here without alpha-beta so the solver's pruning and its
// sign convention are checked against something that shares no code with it.
double bruteForceScore(Position& position, double aValue) {
    Position terminal = position;
    double total = 0.0;
    if (position.gameOver()) {
        total = terminal.finalScore(aValue);
        return position.sideToMove() == Player::Red ? total : -total;
    }
    const std::vector<Move> moves = position.searchMoves();
    if (moves.empty()) {
        total = terminal.finalScore(aValue);
        return position.sideToMove() == Player::Red ? total : -total;
    }
    double best = -1e18;
    for (const Move& move : moves) {
        Position child = position;
        child.play(move);
        best = std::max(best, -bruteForceScore(child, aValue));
    }
    return best;
}

// The last plies are solved exactly: every legal move gets its own margin, and the
// number the board shows is the game-theoretic score, not a sample.
void testExactEndgameSolver() {
    int checked = 0;
    for (int game = 0; game < 5; ++game) {
        Position late;
        late.setRuleset(Ruleset::NoPass);
        std::uint64_t seed = 11 + static_cast<std::uint64_t>(game) * 977;
        while (kTotalMoves - late.stones() > 3) {
            const std::vector<Move> moves = late.legalMoves();
            if (moves.empty()) break;
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            late.play(moves[seed % moves.size()]);
        }
        if (kTotalMoves - late.stones() > 3) continue;

        const double aValue = 1.0;
        const EndgameSolution solved = solveEndgame(late, aValue);
        CHECK(solved.solved);
        CHECK(!solved.moves.empty());
        CHECK(late.isLegal(solved.best));

        const double expected = bruteForceScore(late, aValue);
        CHECK(std::fabs(solved.score - expected) < 1e-9);
        // Every move's score must match the plain minimax too, not only the best.
        double bestSeen = -1e18;
        for (const EndgameMove& entry : solved.moves) {
            CHECK(late.isLegal(entry.move));
            Position child = late;
            child.play(entry.move);
            const double value = -bruteForceScore(child, aValue);
            CHECK(std::fabs(entry.score - value) < 1e-9);
            bestSeen = std::max(bestSeen, value);
        }
        CHECK(std::fabs(bestSeen - solved.score) < 1e-9);
        // A win is a positive score: if the sign convention were flipped, this
        // would fail on the games that are decided.
        for (const EndgameMove& entry : solved.moves) {
            Position child = late;
            child.play(entry.move);
            if (!child.gameOver()) continue;
            const double total = child.finalScore(aValue);
            const double fromMover = late.sideToMove() == Player::Red ? total : -total;
            CHECK(std::fabs(entry.score - fromMover) < 1e-9);
        }
        ++checked;
    }
    CHECK(checked > 0);
}

// The engine publishes those exact numbers: no visits, every cell solved, and the
// margin on the board equals the minimax score of that move.
void testExactEndgamePublished() {
    Position late;
    late.setRuleset(Ruleset::NoPass);
    std::uint64_t seed = 4242;
    while (kTotalMoves - late.stones() > 3) {
        const std::vector<Move> moves = late.legalMoves();
        if (moves.empty()) break;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        late.play(moves[seed % moves.size()]);
    }
    CHECK(kTotalMoves - late.stones() <= 3);

    EngineOptions options = optionsFor(7);
    auto engine = makeEngine(EngineKind::Mcts, options);  // no model needed
    SearchRequest request;
    request.visits = 512;
    const Analysis analysis = engine->search(late, request);
    CHECK(analysis.valid);
    CHECK(analysis.proven);
    CHECK(analysis.hasScore);
    CHECK(analysis.visits == 0);                        // solved, not sampled
    CHECK(late.isLegal(analysis.best));
    CHECK(!analysis.cells.empty());
    for (const CellAnalysis& info : analysis.cells) {
        CHECK(info.hasData);
        CHECK(info.proven);
        CHECK(info.hasScore);
        CHECK(late.isLegal(info.move));
        Position child = late;
        child.play(info.move);
        const double value = -bruteForceScore(child, options.aValue);
        CHECK(std::fabs(info.averageScore - value) < 1e-9);
    }
}

// One ply past the exact gate.  A 7-ply position is ~50^7 nodes wide, so a few
// hundred visits cannot prove it: the tree never gets the complete move lists a
// loss proof needs, and the margin stays a sample.  The solver can finish it
// (0.5-1.8 M nodes), so the search hands the position over instead of reporting
// an estimate that will be replaced one move later, when the position is 6 plies
// deep and the cheap gate applies.
void testExactEndgameAfterSearch() {
    int checked = 0;
    for (int game = 0; game < 4; ++game) {
        Position late;
        late.setRuleset(Ruleset::NoPass);
        std::uint64_t seed = 4242 + static_cast<std::uint64_t>(game) * 977;
        while (kTotalMoves - late.stones() > 7) {
            const std::vector<Move> moves = late.legalMoves();
            if (moves.empty()) break;
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            late.play(moves[seed % moves.size()]);
        }
        if (kTotalMoves - late.stones() != 7) continue;

        EngineOptions options = optionsFor(51 + game);
        // The default clock (400 ms) is a *policy*: it keeps the stream alive when
        // the solve turns out expensive.  This test is about the mechanism, so it
        // asks for a deterministic budget instead of whatever the machine manages.
        options.exactEndgameAfterNodes = 2000000;
        options.exactEndgameAfterTimeMs = 0;
        auto engine = makeEngine(EngineKind::Mcts, options);  // no model needed
        SearchRequest request;
        request.visits = 128;
        const Analysis analysis = engine->search(late, request);
        CHECK(analysis.valid);
        // Either an exact solve, or an honest search: a proven result with zero
        // visits is the only shape that may claim certainty, and every other
        // answer must carry the visits it was sampled with.
        if (!analysis.proven) {
            CHECK(analysis.visits > 0);
            continue;
        }
        CHECK(analysis.hasScore);
        CHECK(late.isLegal(analysis.best));
        CHECK(!analysis.cells.empty());
        // A proven root has two honest shapes: the solver solved the whole
        // position (no visits, every cell exact), or one move was solved and it
        // wins by force (visits > 0, and only that cell is exact).  The rule this
        // pins is per cell: a cell that says "solved" carries the exact margin,
        // and a cell that is not solved carries the visits it was sampled with.
        int provenCells = 0;
        for (const CellAnalysis& info : analysis.cells) {
            CHECK(late.isLegal(info.move));
            if (!info.proven) {
                CHECK(info.visits > 0);
                continue;
            }
            ++provenCells;
            CHECK(info.hasScore);
            Position child = late;
            child.play(info.move);
            CHECK(std::fabs(info.averageScore + bruteForceScore(child, options.aValue)) < 1e-9);
        }
        CHECK(provenCells > 0);
        // A solved move outranks every sample: the panel must be showing one.
        bool bestIsProven = false;
        for (const CellAnalysis& info : analysis.cells) {
            if (info.move == analysis.best) bestIsProven = info.proven;
        }
        CHECK(bestIsProven);
        // The cached answer must be returned again, without re-solving: an
        // extending analysis calls search() repeatedly on the same position.
        const Analysis again = engine->search(late, request);
        CHECK(again.proven && again.hasScore);
        CHECK(std::fabs(again.averageScore - analysis.averageScore) < 1e-9);
        ++checked;
    }
    CHECK(checked > 0);
}

// One ply further the second gate must stay shut: 8 plies measured 10 M nodes and
// 15-25 s, which is not an answer a front end can wait for.  The search must come
// back as a search (visits > 0), never as a zero-visit "exact" result.
void testExactGateStopsAtSeven() {
    Position late;
    late.setRuleset(Ruleset::NoPass);
    std::uint64_t seed = 777;
    while (kTotalMoves - late.stones() > 8) {
        const std::vector<Move> moves = late.legalMoves();
        if (moves.empty()) break;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        late.play(moves[seed % moves.size()]);
    }
    if (kTotalMoves - late.stones() != 8) return;
    EngineOptions options = optionsFor(61);
    // 8 plies is 10 M nodes and 15-25 s: the answer must come back from the
    // search, never from a zero-visit "exact" claim.
    auto engine = makeEngine(EngineKind::Mcts, options);
    SearchRequest request;
    request.visits = 128;
    const Analysis analysis = engine->search(late, request);
    CHECK(analysis.valid);
    CHECK(analysis.visits > 0);   // it searched
    CHECK(!(analysis.proven && analysis.visits == 0));
}

// The solver must stop on the clock, not on luck.  A 9-ply position needs seconds
// (measured 4-6 s, and some never finish inside 10 M nodes); with a deadline the
// caller gets "not solved" back in the time it asked for and can publish the
// search result instead.
void testSolverRespectsTheClock() {
    Position late;
    late.setRuleset(Ruleset::NoPass);
    std::uint64_t seed = 900;
    while (kTotalMoves - late.stones() > 9) {
        const std::vector<Move> moves = late.legalMoves();
        if (moves.empty()) break;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        late.play(moves[seed % moves.size()]);
    }
    if (kTotalMoves - late.stones() != 9) return;
    const auto start = std::chrono::steady_clock::now();
    const EndgameSolution solved = solveEndgame(late, 0.5, 10000000, 80);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    CHECK(!solved.solved);   // 9 plies is not an answer this budget can buy
    CHECK(ms < 400.0);       // and it must not overrun the 80 ms it was given
    // Without a clock the same call is free to run to its node budget; that is
    // the mode the tests above use for exact answers.
    const EndgameSolution unbounded = solveEndgame(late, 0.5, 4000000, 0);
    CHECK(unbounded.solved || unbounded.nodes >= 4000000);
}

// Adaptive proving: the search spends proof budget on the moves it likes, and a
// solved move is marked in the tree.  At 8 plies the tree alone never proves
// anything (the subtree is 50^8), so this is the only way the position gets an
// exact answer before it becomes 6-7 plies deep.
void testAdaptiveProof() {
    Position late;
    late.setRuleset(Ruleset::NoPass);
    std::uint64_t seed = 1000;   // same walk as tools/research/proof_payoff.cpp
    while (kTotalMoves - late.stones() > 8) {
        const std::vector<Move> moves = late.legalMoves();
        if (moves.empty()) break;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        late.play(moves[seed % moves.size()]);
    }
    if (kTotalMoves - late.stones() != 8) return;

    EngineOptions options = optionsFor(71);
    options.proofStartNodes = 100000;
    options.proofTopMoves = 3;
    auto engine = makeEngine(EngineKind::Mcts, options);
    SearchRequest request;
    request.visits = 1024;
    const Analysis analysis = engine->search(late, request);
    CHECK(analysis.valid);
    CHECK(analysis.proven);
    CHECK(analysis.hasScore);
    // Some cell must carry the proof, and its margin must match the solver.
    int provenCells = 0;
    for (const CellAnalysis& info : analysis.cells) {
        if (!info.proven) continue;
        ++provenCells;
        Position child = late;
        child.play(info.move);
        const EndgameSolution solved = solveEndgame(child, options.aValue, 2000000);
        CHECK(solved.solved);
        // The cell's score is from the root's side to move; the solver's is from
        // the child's, so they differ by a sign.
        CHECK(std::fabs(info.averageScore + solved.score) < 1e-9);
    }
    CHECK(provenCells > 0);
}

void testTerminalProof() {
    const std::string headless = std::string(SHUYI_SOURCE_DIR) + "/models/shuyi_resnet_v1.bin";

    auto exactValueForMover = [](const Position& position, Move move) {
        // The move ends the game here, so the result is a hard fact.
        Position after = position;
        const Player mover = after.sideToMove();
        after.play(move);
        const double total = after.finalScore(1.0);
        if (total > 0.0) return mover == Player::Red ? 1.0 : 0.0;
        if (total < 0.0) return mover == Player::Black ? 1.0 : 0.0;
        return 0.5;
    };

    int provenPositions = 0;
    for (int game = 0; game < 6; ++game) {
        // One stone left: every legal move finishes the game, so the search can
        // always solve the position inside a tiny budget.
        Position late;
        std::uint64_t seed = 7 + static_cast<std::uint64_t>(game) * 131;
        while (kTotalMoves - late.stones() > 1) {
            const std::vector<Move> moves = late.legalMoves();
            CHECK(!moves.empty());
            if (moves.empty()) break;
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            late.play(moves[seed % moves.size()]);
        }
        if (late.stones() != kTotalMoves - 1) continue;

        EngineOptions options = optionsFor(900 + static_cast<std::uint64_t>(game));
        options.nnModelPath = headless;
        options.nnValueBlend = 1.0;
        auto engine = makeEngine(EngineKind::MctsNn, options);
        SearchRequest request;
        request.visits = 64;
        const Analysis analysis = engine->search(late, request);
        CHECK(analysis.valid);
        CHECK(analysis.proven);            // solved, not estimated
        CHECK(analysis.visits <= 64);
        CHECK(late.isLegal(analysis.best));
        const double expected = exactValueForMover(late, analysis.best);
        CHECK(std::fabs(analysis.winrate - expected) < 1e-9);
        // The reported score must be the real final score, not a network guess.
        Position after = late;
        after.play(analysis.best);
        CHECK(after.gameOver());
        const double total = after.finalScore(1.0);
        const double expectedScore = late.sideToMove() == Player::Red ? total : -total;
        CHECK(std::fabs(analysis.averageScore - expectedScore) < 1e-6);
        // Every legal move ends the game, so every cell is solved and has to carry
        // its own exact margin: "loses by 2" and "loses by 12" must not both come
        // out as the sampled 0.0 the scoreless network leaves behind.
        int cellsWithScore = 0;
        for (const CellAnalysis& info : analysis.cells) {
            if (!info.hasData) continue;
            CHECK(info.proven);
            CHECK(info.hasScore);
            Position child = late;
            CHECK(late.isLegal(info.move));
            if (!late.isLegal(info.move)) continue;
            child.play(info.move);
            CHECK(child.gameOver());
            const double cellTotal = child.finalScore(1.0);
            const double expectedCell =
                late.sideToMove() == Player::Red ? cellTotal : -cellTotal;
            CHECK(std::fabs(info.averageScore - expectedCell) < 1e-6);
            ++cellsWithScore;
        }
        CHECK(cellsWithScore > 0);
        ++provenPositions;
    }
    CHECK(provenPositions > 0);
}

// Symmetry: equivalent points are one search, and they must be reported with the
// same numbers.  The empty board is a regression case in two ways -- all eight
// symmetries apply, and its Zobrist key is 0, which used to collide with the
// "no tree yet" sentinel and left the root with the identity mask only.
void testSymmetryReport() {
    const std::string headless = std::string(SHUYI_SOURCE_DIR) + "/models/shuyi_resnet_v1.bin";
    EngineOptions options = optionsFor(31337);
    options.nnModelPath = headless;
    options.nnValueBlend = 1.0;
    auto engine = makeEngine(EngineKind::MctsNn, options);
    Position empty = classicStart();  // see testNoHardValueWithoutProof
    SearchRequest request;
    request.timeLimitMs = 0;
    request.visits = 256;
    const Analysis analysis = engine->search(empty, request);
    CHECK(analysis.valid);

    CellAnalysis corner[4];
    int found = 0;
    for (int side = 0; side < 4; ++side) {
        const int cell = index(side & 1 ? 8 : 0, side & 2 ? 8 : 0);
        for (const CellAnalysis& info : analysis.cells) {
            if (info.cell == cell) {
                corner[found++] = info;
                break;
            }
        }
    }
    CHECK(found == 4);
    if (found == 4) {
        // The four corners are one orbit on an empty board.
        CHECK(corner[0].visits > 0);  // searched at all
        for (int i = 1; i < 4; ++i) {
            CHECK(corner[i].visits == corner[0].visits);
            CHECK(std::fabs(corner[i].winrate - corner[0].winrate) < 1e-12);
        }
    }

    // After a stone on a1 the surviving symmetry is the main-diagonal transpose,
    // which swaps b1 with a2: those two must still agree.
    Position diagonal = classicStart();
    diagonal.play(Move(index(0, 0), 1));
    engine->reset();
    const Analysis after = engine->search(diagonal, request);
    CHECK(after.valid);
    const auto cellOf = [&](int cell) {
        CellAnalysis out;
        out.cell = -1;
        for (const CellAnalysis& info : after.cells) {
            if (info.cell == cell) out = info;
        }
        return out;
    };
    const CellAnalysis b1 = cellOf(index(1, 0));
    const CellAnalysis a2 = cellOf(index(0, 1));
    if (b1.cell >= 0 && a2.cell >= 0) {
        CHECK(b1.visits == a2.visits);
        CHECK(std::fabs(b1.winrate - a2.winrate) < 1e-12);
    }
}

void testVisitBudget() {
    Position position = classicStart();
    const int budget = 200;
    EngineOptions options = optionsFor(991);
    auto engine = makeEngine(EngineKind::Mcts, options);
    SearchRequest request;
    request.timeLimitMs = 0; // no clock at all
    request.visits = budget;
    const Analysis first = engine->search(position, request);
    CHECK(first.valid);
    CHECK(first.visits == budget);
}



void testEngineKinds() {
    Position position = classicStart();
    SearchRequest request;
    request.timeLimitMs = 35;
    for (EngineKind kind : {EngineKind::Random, EngineKind::Mcts, EngineKind::MctsNn}) {
        EngineOptions options = optionsFor(1234 + static_cast<int>(kind));
        options.nnModelPath = std::string(SHUYI_SOURCE_DIR) + "/models/shuyi_resnet_v1.bin";
        auto engine = makeEngine(kind, options);
        CHECK(engine != nullptr);
        const Analysis analysis = engine->search(position, request);
        CHECK(analysis.valid);
        CHECK(analysis.best.valid());
        CHECK(position.isLegal(analysis.best));
        if (kind == EngineKind::Mcts || kind == EngineKind::MctsNn) {
            CHECK(analysis.visits > 0);
            CHECK(!analysis.cells.empty());
            for (const CellAnalysis& cell : analysis.cells) {
                CHECK(cell.move.valid());
                CHECK(cell.visits >= 0);
            }
        }
        if (kind == EngineKind::MctsNn) {
            CHECK(!engine->modelName().empty());
            CHECK(engine->modelName() != "none");
        }

        Position next = position;
        CHECK(next.play(analysis.best));
        engine->onPlay(next, analysis.best);
        request.timeLimitMs = 20;
        const Analysis reused = engine->search(next, request);
        CHECK(reused.valid);
        CHECK(next.isLegal(reused.best));
    }
}



void testDirichletRoot() {
    Position position = classicStart();
    EngineOptions options = optionsFor(77);
    options.rootDirichletAlpha = 0.5;
    options.rootNoiseWeight = 1.0;
    auto engine = makeEngine(EngineKind::Mcts, options);
    SearchRequest request;
    request.timeLimitMs = 30;
    engine->search(position, request);
    const auto policy = engine->rootPolicy();
    CHECK(!policy.empty());
    bool varies = false;
    for (std::size_t i = 1; i < policy.size(); ++i) {
        if (std::abs(policy[i].prior - policy[0].prior) > 1e-6) varies = true;
    }
    CHECK(varies);
}

void testNnModel() {
    PolicyValueNet net;
    std::string error;
    const std::string path = std::string(SHUYI_SOURCE_DIR) + "/models/shuyi_resnet_v1.bin";
    CHECK(net.load(path, &error));
    if (!error.empty()) std::cerr << error << "\n";
    Position position = classicStart();
    const NnOutput output = net.evaluate(position, 1.0);
    CHECK(output.value >= 0.0f && output.value <= 1.0f);
    const Move move(index(4, 4), 7);
    CHECK(move.valid());
    CHECK(net.moveFromIndex(net.moveIndex(move)) == move);

    // The binary format is shape-tagged: a wider server model must round-trip
    // through the same C++ loader.
    PolicyValueNet wider = PolicyValueNet::random(42, "shape-test", 8, 1);
    CHECK(wider.channels() == 8);
    CHECK(wider.blocks() == 1);
    // Relative on purpose: ctest runs in the build directory, and a system temp
    // directory is not always writable under the test harness.
    const std::string temp = "shuyi_shape_test.bin";
    CHECK(wider.save(temp, &error));
    PolicyValueNet reloaded;
    CHECK(reloaded.load(temp, &error));
    CHECK(reloaded.channels() == 8);
    CHECK(reloaded.blocks() == 1);
    const NnOutput reloadedOutput = reloaded.evaluate(position, 1.0);
    CHECK(reloadedOutput.value >= 0.0f && reloadedOutput.value <= 1.0f);
    std::filesystem::remove(temp);

    // v3: 18 stone + 18 explicit hand-count planes + side = 37 inputs.
    PolicyValueNet v3 = PolicyValueNet::random(43, "v3-shape-test", 8, 1, kResInputPlanesV3);
    CHECK(v3.modelInputPlanes() == kResInputPlanesV3);
    CHECK(v3.save(temp, &error));
    PolicyValueNet v3reloaded;
    CHECK(v3reloaded.load(temp, &error));
    CHECK(v3reloaded.modelInputPlanes() == kResInputPlanesV3);
    const NnOutput v3out = v3reloaded.evaluate(position, 1.0);
    CHECK(v3out.value >= 0.0f && v3out.value <= 1.0f);
    std::filesystem::remove(temp);

    // v4: A is an input.  A v4 model must answer differently for different A; a
    // v3 model cannot see A and answers identically -- that asymmetry is the
    // whole point, and it is what made "set_a 10" leave a midgame win rate
    // unchanged.  Both directions are asserted, so dropping A from the input or
    // misreading an old file as A-aware fails here.
    PolicyValueNet aAware = PolicyValueNet::random(45, "a-input-test", 8, 1);
    CHECK(aAware.aInput());
    CHECK(aAware.globalFeatures() == kResGlobalFeatures);
    const NnOutput aLow = aAware.evaluate(position, -6.0);
    const NnOutput aHigh = aAware.evaluate(position, 6.0);
    CHECK(std::fabs(aLow.value - aHigh.value) > 1e-6f);
    CHECK(aAware.save(temp, &error));
    PolicyValueNet aReloaded;
    CHECK(aReloaded.load(temp, &error));
    CHECK(aReloaded.aInput());
    CHECK(std::fabs(aReloaded.evaluate(position, 6.0).value - aHigh.value) < 1e-6f);
    std::filesystem::remove(temp);

    PolicyValueNet aBlind;
    CHECK(aBlind.load(std::string(SHUYI_SOURCE_DIR) + "/models/shuyi_resnet_v1.bin", &error));
    CHECK(!aBlind.aInput());
    const float blindLow = aBlind.evaluate(position, -6.0).value;
    const float blindHigh = aBlind.evaluate(position, 6.0).value;
    CHECK(blindLow == blindHigh);

    // Score head flag must survive clear()/allocate() inside load(); it is a
    // separate bit in the v3 header and used to be wiped on the way in.
    PolicyValueNet scorer = PolicyValueNet::random(44, "score-test", 8, 1);
    // The search compares the model's rule set with the position's before it
    // trusts the value head; this fixture is played on a Classic position.
    scorer.setRuleset(Ruleset::Classic);
    scorer.enableScoreHead(1.0f);
    CHECK(scorer.hasScoreHead());
    CHECK(scorer.save(temp, &error));
    PolicyValueNet scoreLoaded;
    CHECK(scoreLoaded.load(temp, &error));
    CHECK(scoreLoaded.hasScoreHead());
    const NnOutput scored = scoreLoaded.evaluate(position, 1.0);
    // Output bias 1.0 through tanh: the head is untrained but not silent.
    CHECK(std::isfinite(scored.expectedScore));
    CHECK(std::fabs(scored.expectedScore - std::tanh(1.0f) * kScoreScale) < 1e-3f);

    // The search must report the score head's prediction as the average score
    // at full network blend, instead of the value-to-score heuristic.
    // The search must back up the score head's prediction instead of the
    // value-to-score heuristic.  Two visits with width 1 keep the tree at
    // root + one child, so the root's average is exactly one sign flip of it.
    EngineOptions nnOptions = optionsFor(77);
    nnOptions.nnModelPath = temp;
    nnOptions.visits = 1;
    nnOptions.mctsRootChildren = 1;
    nnOptions.mctsMaxChildren = 1;
    nnOptions.nnValueBlend = 1.0;
    auto scoreEngine = makeEngine(EngineKind::MctsNn, nnOptions);
    CHECK(scoreEngine != nullptr);
    SearchRequest scoreRequest;
    scoreRequest.timeLimitMs = 500;
    scoreRequest.visits = 1;
    const Analysis scoreAnalysis = scoreEngine->search(position, scoreRequest);
    CHECK(scoreAnalysis.valid);
    if (std::getenv("SHUYI_TEST_TRACE")) {
        std::fprintf(stderr, "[score] avg=%.6f winrate=%.6f visits=%d want=%.3f\n",
                     scoreAnalysis.averageScore, scoreAnalysis.winrate, scoreAnalysis.visits,
                     -std::tanh(1.0f) * kScoreScale);
    }
    CHECK(std::fabs(scoreAnalysis.averageScore + std::tanh(1.0f) * kScoreScale) < 1.0);
    std::filesystem::remove(temp);
}

// A blocking-trained model must be *used* under NoPass, not quietly dropped.
//
// The search compares the model's rule set with the position's before it trusts
// the value head, and that comparison used to be equality.  The protocol gate
// accepted the model (rulesetCompatible) and the search then ignored it, so the
// engine played NoPass with a uniform prior and rollout values -- which looks
// exactly like "the network got much worse under the new rule".
//
// The rule set is not part of the network input, so the same board must give the
// same root value under both rule sets.  If the model were dropped under NoPass
// the two numbers would differ.
void testNoPassUsesTheModel() {
    const std::string model =
        std::string(SHUYI_SOURCE_DIR) + "/models/b4c32-s42012-d595499.bin";
    if (!std::filesystem::exists(model)) {
        CHECK(false);  // the shipped model is part of the repository
        return;
    }
    auto evaluate = [&](Ruleset ruleset) {
        Position position = classicStart();
        position.reset();
        position.setRuleset(ruleset);
        EngineOptions options = optionsFor(31);
        options.nnModelPath = model;
        options.nnValueBlend = 1.0;
        options.visits = 1;
        options.mctsRootChildren = 1;
        options.mctsMaxChildren = 1;
        auto engine = makeEngine(EngineKind::MctsNn, options);
        SearchRequest request;
        request.visits = 1;
        request.timeLimitMs = 2000;
        return engine->search(position, request);
    };
    const Analysis blocking = evaluate(Ruleset::Blocking);
    const Analysis nopass = evaluate(Ruleset::NoPass);
    CHECK(blocking.valid);
    CHECK(nopass.valid);
    CHECK(std::fabs(blocking.winrate - nopass.winrate) < 1e-6f);
    CHECK(blocking.best.cell == nopass.best.cell);
    CHECK(blocking.best.number == nopass.best.number);
}

} // namespace

int runEngineTests() {
    testEngineKinds();
    testDirichletRoot();
    testNnModel();
    testEndgameExactScore();
    testExactEndgameAfterSearch();
    testExactGateStopsAtSeven();
    testSolverRespectsTheClock();
    testAdaptiveProof();
    testNoPassUsesTheModel();
    testNoHardValueWithoutProof();
    testTerminalProof();
    testExactEndgameSolver();
    testExactEndgamePublished();
    testSymmetryReport();
    testVisitBudget();
    return failures;
}
