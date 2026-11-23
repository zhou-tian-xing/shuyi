// Two-engine match used by the generation trainer for promotion tests.
//
// Unlike shuyi_tournament (which fits a full round-robin ELO table), this tool
// answers one question: did candidate A beat baseline B?  It alternates
// colours, uses random openings, and writes a small JSON/CSV summary that the
// Python pipeline can consume.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "shuyi/engine.h"
#include "shuyi/position.h"
#include "shuyi/io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace shuyi;

namespace {

struct Options {
    std::string aName = "candidate";
    std::string bName = "baseline";
    EngineKind aEngine = EngineKind::MctsNn;
    EngineKind bEngine = EngineKind::MctsNn;
    std::string aModel;
    std::string bModel;
    int games = 40;
    int timeMs = 100;
    int visits = 0;
    int openingPlies = 2;
    int rootChildren = 16;
    std::uint64_t seed = 20261201;
    std::string outPath = "match.json";
    std::string csvPath;
};

struct Game {
    int index = 0;
    bool aIsRed = true;
    double scoreA = 0.5;
    double marginA = 0.0;
    int plies = 0;
};

void usage() {
    std::cout
        << "usage: shuyi_match [options]\n"
        << "  --a-engine NAME  random|pruning|mcts|mcts_nn (default mcts_nn)\n"
        << "  --b-engine NAME  random|pruning|mcts|mcts_nn (default mcts_nn)\n"
        << "  --a-model PATH   model for A when A is mcts_nn\n"
        << "  --b-model PATH   model for B when B is mcts_nn\n"
        << "  --a-name NAME    label in the report\n"
        << "  --b-name NAME    label in the report\n"
        << "  --games N        total games, colours alternate (default 40)\n"
        << "  --time MS        time per move (default 100)\n"
        << "  --opening N      random opening plies (default 2)\n"
        << "  --root-children N  MCTS-NN root width cap (default 16)\n"
        << "  --seed N         deterministic seed\n"
        << "  --out PATH       JSON output (default match.json)\n"
        << "  --csv PATH       optional per-game CSV\n";
}

bool parseArgs(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return {};
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            usage();
            return false;
        } else if (arg == "--a-engine") {
            if (!parseEngineName(need("--a-engine"), options.aEngine)) return false;
        } else if (arg == "--b-engine") {
            if (!parseEngineName(need("--b-engine"), options.bEngine)) return false;
        } else if (arg == "--a-model") {
            options.aModel = need("--a-model");
        } else if (arg == "--b-model") {
            options.bModel = need("--b-model");
        } else if (arg == "--a-name") {
            options.aName = need("--a-name");
        } else if (arg == "--b-name") {
            options.bName = need("--b-name");
        } else if (arg == "--games") {
            options.games = std::max(2, std::stoi(need("--games")));
        } else if (arg == "--time") {
            options.timeMs = std::max(1, std::stoi(need("--time")));
        } else if (arg == "--visits") {
            options.visits = std::max(1, std::stoi(need("--visits")));
        } else if (arg == "--opening") {
            options.openingPlies = std::max(0, std::stoi(need("--opening")));
        } else if (arg == "--root-children") {
            options.rootChildren = std::max(1, std::stoi(need("--root-children")));
        } else if (arg == "--seed") {
            options.seed = std::stoull(need("--seed"));
        } else if (arg == "--out") {
            options.outPath = need("--out");
        } else if (arg == "--csv") {
            options.csvPath = need("--csv");
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            usage();
            return false;
        }
    }
    return true;
}

std::unique_ptr<Engine> create(EngineKind kind, const Options& options, const std::string& model,
                               std::uint64_t seed) {
    EngineOptions engineOptions;
    engineOptions.seed = seed;
    engineOptions.timeLimitMs = options.visits > 0 ? 0 : options.timeMs;
    engineOptions.visits = options.visits;
    engineOptions.maxDepth = 10;
    engineOptions.ttSizeMb = 16;
    engineOptions.mctsMaxChildren = 32;
    engineOptions.mctsRootChildren = 64;
    engineOptions.mctsNnRootChildren = options.rootChildren;
    engineOptions.nnModelPath = model;
    return makeEngine(kind, engineOptions);
}

Game playGame(const Options& options, int index, std::uint64_t seed, bool aIsRed) {
    auto engineA = create(options.aEngine, options, options.aModel, seed ^ 0xA5A5A5A5ULL);
    auto engineB = create(options.bEngine, options, options.bModel, seed ^ 0x5A5A5A5AULL);
    if (!engineA || !engineB) return {index, aIsRed, 0.5, 0.0, 0};

    Position position;
    std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ULL);
    for (int i = 0; i < options.openingPlies && !position.gameOver(); ++i) {
        const auto moves = position.legalMoves();
        if (moves.empty()) break;
        std::uniform_int_distribution<std::size_t> pick(0, moves.size() - 1);
        const Move move = moves[pick(rng)];
        position.play(move);
        engineA->onPlay(position, move);
        engineB->onPlay(position, move);
    }

    while (!position.gameOver()) {
        Engine* engine = (position.sideToMove() == Player::Red) == aIsRed ? engineA.get()
                                                                         : engineB.get();
        SearchRequest request;
        request.timeLimitMs = options.visits > 0 ? 0 : options.timeMs;
        request.visits = options.visits;
        Analysis analysis = engine->search(position, request);
        if (!analysis.valid || !position.isLegal(analysis.best)) {
            const auto moves = position.legalMoves();
            if (moves.empty()) break;
            std::uniform_int_distribution<std::size_t> pick(0, moves.size() - 1);
            analysis.best = moves[pick(rng)];
        }
        const Move move = analysis.best;
        position.play(move);
        engineA->onPlay(position, move);
        engineB->onPlay(position, move);
    }

    const double total = position.finalScore(1.0);
    Game game;
    game.index = index;
    game.aIsRed = aIsRed;
    game.plies = position.ply();
    game.scoreA = aIsRed ? (total > 0.0 ? 1.0 : (total < 0.0 ? 0.0 : 0.5))
                         : (total < 0.0 ? 1.0 : (total > 0.0 ? 0.0 : 0.5));
    game.marginA = aIsRed ? total : -total;
    return game;
}


std::string jsonQuote(const std::string& text) {
    std::string out = "\"";
    for (char c : text) {
        if (c == '\\' || c == '"') out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

double wilsonLow(double p, int n, double z = 1.96) {
    if (n <= 0) return 0.0;
    const double denom = 1.0 + z * z / n;
    const double centre = p + z * z / (2.0 * n);
    const double spread = z * std::sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n));
    return std::max(0.0, (centre - spread) / denom);
}

double wilsonHigh(double p, int n, double z = 1.96) {
    if (n <= 0) return 1.0;
    const double denom = 1.0 + z * z / n;
    const double centre = p + z * z / (2.0 * n);
    const double spread = z * std::sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n));
    return std::min(1.0, (centre + spread) / denom);
}

} // namespace

int main(int argc, char** argv) {
    shuyi::Utf8Args utf8Args(argc, argv);
    argv = utf8Args.pointers();
    Options options;
    if (!parseArgs(argc, argv, options)) return 1;

    std::vector<Game> games;
    games.reserve(static_cast<std::size_t>(options.games));
    std::uint64_t gameSeed = options.seed;
    int wins = 0, draws = 0, losses = 0;
    double score = 0.0;
    double margin = 0.0;
    for (int i = 0; i < options.games; ++i) {
        const bool aIsRed = (i % 2) == 0;
        const Game game = playGame(options, i, gameSeed++, aIsRed);
        games.push_back(game);
        score += game.scoreA;
        margin += game.marginA;
        if (game.scoreA > 0.75) ++wins;
        else if (game.scoreA < 0.25) ++losses;
        else ++draws;
        std::cerr << "game " << (i + 1) << "/" << options.games << " A="
                  << (aIsRed ? "red" : "black") << " score=" << game.scoreA
                  << " margin=" << game.marginA << "\n";
    }

    const double rate = score / std::max(1, options.games);
    double elo = 0.0;
    if (rate > 0.0 && rate < 1.0) {
        elo = 400.0 * std::log10(rate / (1.0 - rate));
    } else if (rate >= 1.0) {
        elo = 800.0;
    } else {
        elo = -800.0;
    }

    std::ostringstream json;
    json << std::fixed << std::setprecision(4);
    json << "{\n";
    json << "  \"a_name\": " << jsonQuote(options.aName) << ",\n";
    json << "  \"b_name\": " << jsonQuote(options.bName) << ",\n";
    json << "  \"a_engine\": \"" << engineName(options.aEngine) << "\",\n";
    json << "  \"b_engine\": \"" << engineName(options.bEngine) << "\",\n";
    json << "  \"a_model\": " << jsonQuote(options.aModel) << ",\n";
    json << "  \"b_model\": " << jsonQuote(options.bModel) << ",\n";
    json << "  \"games\": " << options.games << ",\n";
    json << "  \"wins\": " << wins << ",\n";
    json << "  \"draws\": " << draws << ",\n";
    json << "  \"losses\": " << losses << ",\n";
    json << "  \"score_a\": " << rate << ",\n";
    json << "  \"score_ci_low\": " << wilsonLow(rate, options.games) << ",\n";
    json << "  \"score_ci_high\": " << wilsonHigh(rate, options.games) << ",\n";
    json << "  \"elo_diff\": " << elo << ",\n";
    json << "  \"mean_margin_a\": " << margin / std::max(1, options.games) << ",\n";
    json << "  \"time_ms\": " << options.timeMs << ",\n";
    json << "  \"opening_plies\": " << options.openingPlies << ",\n";
    json << "  \"seed\": " << options.seed << "\n";
    json << "}\n";

    const auto outPath = std::filesystem::path(options.outPath);
    if (!outPath.parent_path().empty()) std::filesystem::create_directories(outPath.parent_path());
    std::ofstream out = openOutUtf8(outPath.string(), std::ios::out);
    out << json.str();
    out.close();

    if (!options.csvPath.empty()) {
        const auto csvPath = std::filesystem::path(options.csvPath);
        if (!csvPath.parent_path().empty()) {
            std::filesystem::create_directories(csvPath.parent_path());
        }
        std::ofstream csv = openOutUtf8(csvPath.string(), std::ios::out);
        csv << "game,a_is_red,score_a,margin_a,plies\n";
        for (const Game& game : games) {
            csv << game.index << "," << (game.aIsRed ? 1 : 0) << "," << game.scoreA << ","
                << game.marginA << "," << game.plies << "\n";
        }
    }

    std::cout << json.str();
    return 0;
}
