#include "shuyi/ai.h"
#include "shuyi/engine.h"
#include "shuyi/nn.h"
#include "shuyi/tt.h"

#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
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

EngineOptions optionsFor(std::uint64_t seed) {
    EngineOptions options;
    options.seed = seed;
    options.ttSizeMb = 4;
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
    Position opening;

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
    Position late;
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
    Position opening;
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
    }
}

// A solved position must return the exact result with the right sign, and it
// must stop spending visits once it is solved.  This is the regression test for
// the sign of the terminal value as it travels back up the tree: a loss that
// propagates as a win shows up here as a win rate of 1.0 on a lost position.
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
    Position empty;
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
    Position diagonal;
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
    Position position;
    const int budget = 200;

    for (EngineKind kind : {EngineKind::Mcts}) {
        EngineOptions options = optionsFor(991);
        auto engine = makeEngine(kind, options);
        SearchRequest request;
        request.timeLimitMs = 0; // no clock at all
        request.visits = budget;
        const Analysis first = engine->search(position, request);
        CHECK(first.valid);
        CHECK(first.visits == budget);
    }

    // Alpha-beta: one visit is one node.  It finishes the iteration it is in, so
    // the count can overshoot; that overshoot must be identical run to run.
    const auto runPruning = [&](long long* nodes, Move* move) {
        EngineOptions options = optionsFor(992);
        auto engine = makeEngine(EngineKind::Pruning, options);
        SearchRequest request;
        request.timeLimitMs = 0;
        request.visits = budget * 10;
        const Analysis analysis = engine->search(position, request);
        CHECK(analysis.valid);
        if (nodes) *nodes = analysis.nodes;
        if (move) *move = analysis.best;
    };
    long long nodesA = 0, nodesB = 0;
    Move moveA{}, moveB{};
    runPruning(&nodesA, &moveA);
    runPruning(&nodesB, &moveB);
    CHECK(nodesA >= budget * 10);
    CHECK(nodesA <= budget * 10 * 4);
    CHECK(nodesA == nodesB);
    CHECK(moveA == moveB);
}

void testTt() {
    TranspositionTable table(1);
    table.newSearch();
    Move move(index(4, 4), 7);
    table.store(0x123456789ABCDEF0ULL, 4, 12, TTFlag::Exact, move);
    int value = 0;
    Move best;
    CHECK(table.probe(0x123456789ABCDEF0ULL, 4, -100, 100, value, best));
    CHECK(value == 12);
    CHECK(best == move);
    CHECK(!table.probe(0x123456789ABCDEF1ULL, 4, -100, 100, value, best));
}

void testSearchProgress() {
    Position position;
    SearchOptions search;
    search.timeLimitMs = 40;
    search.maxDepth = 3;
    int reports = 0;
    search.onProgress = [&](const SearchProgress& progress) {
        ++reports;
        CHECK(progress.nodes >= 0);
    };
    TranspositionTable table(2);
    const SearchResult result = searchAlphaBeta(position, search, table);
    CHECK(result.move.valid());
    CHECK(position.isLegal(result.move));
    CHECK(reports > 0);
}

void testEngineKinds() {
    Position position;
    SearchRequest request;
    request.timeLimitMs = 35;
    for (EngineKind kind : {EngineKind::Random, EngineKind::Pruning, EngineKind::Mcts,
                            EngineKind::MctsNn}) {
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
            CHECK(engine->infoTag().rfind("nn=", 0) == 0);
            CHECK(engine->infoTag() != "nn=none");
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

void testAspirationPvs() {
    Position position;
    SearchOptions base;
    base.timeLimitMs = 5000;
    base.maxDepth = 4;
    base.rootBranching = 10;
    base.maxBranching = 6;
    base.seed = 0xA5A5A5A5ULL;
    base.aspirationWindow = 1;
    base.usePvs = true;
    // Selective beam + TT can legitimately change which moves survive between
    // re-searches, so compare the window algorithm with a deterministic tree.
    base.useTt = false;

    TranspositionTable tableA(2);
    const SearchResult aspiration = searchAlphaBeta(position, base, tableA);
    CHECK(aspiration.completed);
    CHECK(aspiration.depth >= 4);
    CHECK(aspiration.windowResearches > 0);
    CHECK(aspiration.pvsResearches > 0);

    SearchOptions full = base;
    full.aspirationWindow = 0;
    full.usePvs = false;
    TranspositionTable tableB(2);
    const SearchResult reference = searchAlphaBeta(position, full, tableB);
    CHECK(reference.completed);
    CHECK(reference.depth == aspiration.depth);
    if (reference.value != aspiration.value) {
        std::cerr << "aspiration value=" << aspiration.value << " depth=" << aspiration.depth
                  << " windowResearches=" << aspiration.windowResearches
                  << " pvs=" << aspiration.pvsResearches
                  << " reference value=" << reference.value << " depth=" << reference.depth
                  << "\\n";
    }
    CHECK(reference.value == aspiration.value);
    CHECK(position.isLegal(aspiration.move));
    CHECK(position.isLegal(reference.move));

    SearchOptions fullPvs = base;
    fullPvs.aspirationWindow = 0;
    fullPvs.usePvs = true;
    TranspositionTable tableD(2);
    const SearchResult pvsOnly = searchAlphaBeta(position, fullPvs, tableD);
    CHECK(pvsOnly.completed);
    if (pvsOnly.value != reference.value) {
        std::cerr << "pvsOnly value=" << pvsOnly.value << " depth=" << pvsOnly.depth
                  << " pvs=" << pvsOnly.pvsResearches << " reference=" << reference.value << "\\n";
    }
    CHECK(pvsOnly.value == reference.value);

    SearchOptions noPvs = base;
    noPvs.aspirationWindow = 0;
    noPvs.usePvs = false;
    TranspositionTable tableC(2);
    const SearchResult plain = searchAlphaBeta(position, noPvs, tableC);
    CHECK(plain.completed);
    if (plain.value != reference.value) {
        std::cerr << "noPvs value=" << plain.value << " depth=" << plain.depth
                  << " reference=" << reference.value << "\\n";
    }
    CHECK(plain.value == reference.value);
}

void testHandAwareEvaluation() {
    Position empty;
    CHECK(evaluateHandAware(empty) == 0);
    CHECK(evaluateHandAware(empty, 0.0) == 0);

    Position position;
    CHECK(position.play(Move(index(4, 4), 9))); // red spends a high number
    CHECK(position.play(Move(index(0, 0), 1)));  // black spends a low number
    const int noHand = evaluateHandAware(position, 0.0);
    const int withHand = evaluateHandAware(position, 1.0);
    CHECK(noHand != withHand);
}

void testDirichletRoot() {
    Position position;
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
    Position position;
    const NnOutput output = net.evaluate(position);
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
    const NnOutput reloadedOutput = reloaded.evaluate(position);
    CHECK(reloadedOutput.value >= 0.0f && reloadedOutput.value <= 1.0f);
    std::filesystem::remove(temp);

    // v3: 18 stone + 18 explicit hand-count planes + side = 37 inputs.
    PolicyValueNet v3 = PolicyValueNet::random(43, "v3-shape-test", 8, 1, kResInputPlanesV3);
    CHECK(v3.modelInputPlanes() == kResInputPlanesV3);
    CHECK(v3.save(temp, &error));
    PolicyValueNet v3reloaded;
    CHECK(v3reloaded.load(temp, &error));
    CHECK(v3reloaded.modelInputPlanes() == kResInputPlanesV3);
    const NnOutput v3out = v3reloaded.evaluate(position);
    CHECK(v3out.value >= 0.0f && v3out.value <= 1.0f);
    std::filesystem::remove(temp);

    // Score head flag must survive clear()/allocate() inside load(); it is a
    // separate bit in the v3 header and used to be wiped on the way in.
    PolicyValueNet scorer = PolicyValueNet::random(44, "score-test", 8, 1);
    scorer.enableScoreHead(1.0f);
    CHECK(scorer.hasScoreHead());
    CHECK(scorer.save(temp, &error));
    PolicyValueNet scoreLoaded;
    CHECK(scoreLoaded.load(temp, &error));
    CHECK(scoreLoaded.hasScoreHead());
    const NnOutput scored = scoreLoaded.evaluate(position);
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

} // namespace

int runEngineTests() {
    testTt();
    testSearchProgress();
    testAspirationPvs();
    testHandAwareEvaluation();
    testEngineKinds();
    testDirichletRoot();
    testNnModel();
    testEndgameExactScore();
    testNoHardValueWithoutProof();
    testTerminalProof();
    testSymmetryReport();
    testVisitBudget();
    return failures;
}
