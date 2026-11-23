// Round-robin ELO tournament for the Shuyi engines.
//
// The anchor is the uniform-random engine, whose rating is fixed at 0.  All
// other ratings are fitted with a Bradley-Terry/logistic model and confidence
// intervals are obtained by bootstrapping the game list.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "shuyi/engine.h"
#include "shuyi/position.h"
#include "shuyi/io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace shuyi;

namespace {

struct Options {
    int gamesPerPair = 20;
    int timeMs = 30;
    int visits = 0;
    int openingPlies = 2;
    std::uint64_t seed = 20261121;
    std::vector<EngineKind> engines{EngineKind::Random, EngineKind::Pruning, EngineKind::Mcts,
                                    EngineKind::MctsNn};
    std::string nnModel;
    std::string outPath = "docs/ELO_REPORT.md";
    std::string csvPath = "docs/elo_results.csv";
};

struct GameRecord {
    int a = 0;
    int b = 0;
    double scoreA = 0.5;
    std::uint64_t seed = 0;
};

struct Stats {
    int games = 0;
    int wins = 0;
    int draws = 0;
    int losses = 0;
    double score = 0.0;
};

void usage() {
    std::cout
        << "usage: shuyi_tournament [options]\n"
        << "  --games N       games per pair (default 20)\n"
        << "  --time MS       time per move in ms (default 30)\n"
        << "  --opening N     random opening plies (default 2)\n"
        << "  --seed N        deterministic tournament seed\n"
        << "  --engines LIST  random,pruning,mcts,mcts_nn\n"
        << "  --nn PATH       model for mcts_nn\n"
        << "  --out PATH      markdown report path\n"
        << "  --csv PATH      csv path\n";
}

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    std::string current;
    for (char c : text) {
        if (c == separator) {
            if (!current.empty()) parts.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) parts.push_back(current);
    return parts;
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
        if (arg == "--help" || arg == "-h") { usage(); return false; }
        if (arg == "--games") options.gamesPerPair = std::max(2, std::stoi(need("--games")));
        else if (arg == "--time") options.timeMs = std::max(1, std::stoi(need("--time")));
        else if (arg == "--visits") options.visits = std::max(1, std::stoi(need("--visits")));
        else if (arg == "--opening") options.openingPlies = std::max(0, std::stoi(need("--opening")));
        else if (arg == "--seed") options.seed = std::stoull(need("--seed"));
        else if (arg == "--nn") options.nnModel = need("--nn");
        else if (arg == "--out") options.outPath = need("--out");
        else if (arg == "--csv") options.csvPath = need("--csv");
        else if (arg == "--engines") {
            options.engines.clear();
            for (const std::string& name : split(need("--engines"), ',')) {
                EngineKind kind;
                if (!parseEngineName(name, kind)) {
                    std::cerr << "unknown engine: " << name << "\n";
                    return false;
                }
                options.engines.push_back(kind);
            }
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            usage();
            return false;
        }
    }
    if (options.engines.size() < 2) {
        std::cerr << "at least two engines are required (random plus a candidate)\n";
        return false;
    }
    const bool hasRandom = std::find(options.engines.begin(), options.engines.end(), EngineKind::Random) !=
                           options.engines.end();
    if (!hasRandom) {
        std::cerr << "the random engine is required as the ELO 0 anchor\n";
        return false;
    }
    return true;
}

std::unique_ptr<Engine> create(EngineKind kind, const Options& options, std::uint64_t seed) {
    EngineOptions engineOptions;
    engineOptions.seed = seed;
    engineOptions.timeLimitMs = options.visits > 0 ? 0 : options.timeMs;
    engineOptions.visits = options.visits;
    engineOptions.maxDepth = 10;
    engineOptions.ttSizeMb = 16;
    engineOptions.mctsMaxChildren = 32;
    engineOptions.nnModelPath = options.nnModel;
    return makeEngine(kind, engineOptions);
}

double outcomeScore(Outcome outcome, bool aIsRed) {
    if (outcome == Outcome::Draw) return 0.5;
    if (outcome == Outcome::RedWin) return aIsRed ? 1.0 : 0.0;
    if (outcome == Outcome::BlackWin) return aIsRed ? 0.0 : 1.0;
    return 0.5;
}

GameRecord playGame(EngineKind a, EngineKind b, const Options& options, std::uint64_t seed, bool aIsRed) {
    auto engineA = create(a, options, seed ^ 0xA5A5A5A5ULL);
    auto engineB = create(b, options, seed ^ 0x5A5A5A5AULL);
    Position position;
    std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ULL);

    for (int i = 0; i < options.openingPlies && !position.gameOver(); ++i) {
        const auto moves = position.legalMoves();
        if (moves.empty()) break;
        std::uniform_int_distribution<std::size_t> pick(0, moves.size() - 1);
        position.play(moves[pick(rng)]);
    }

    while (!position.gameOver()) {
        Engine* engine = (position.sideToMove() == Player::Red) == aIsRed ? engineA.get() : engineB.get();
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

    GameRecord record;
    record.a = -1;
    record.b = -1;
    record.seed = seed;
    record.scoreA = outcomeScore(position.outcome(1.0), aIsRed);
    return record;
}

double eloLoss(const std::vector<GameRecord>& games, const std::vector<double>& rating,
               double lambda) {
    constexpr double k = 0.005756462732485114; // ln(10)/400
    double loss = 0.0;
    for (const GameRecord& game : games) {
        const double p = 1.0 / (1.0 + std::exp(-(rating[game.a] - rating[game.b]) * k));
        const double q = std::clamp(p, 1e-12, 1.0 - 1e-12);
        loss -= game.scoreA * std::log(q) + (1.0 - game.scoreA) * std::log(1.0 - q);
    }
    for (double r : rating) loss += 0.5 * lambda * r * r;
    return loss;
}

bool solveLinear(std::vector<std::vector<double>> matrix, std::vector<double> rhs,
                 std::vector<double>& solution) {
    const int n = static_cast<int>(rhs.size());
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(matrix[row][col]) > std::abs(matrix[pivot][col])) pivot = row;
        }
        if (std::abs(matrix[pivot][col]) < 1e-12) return false;
        std::swap(matrix[pivot], matrix[col]);
        std::swap(rhs[pivot], rhs[col]);
        const double div = matrix[col][col];
        for (int j = col; j < n; ++j) matrix[col][j] /= div;
        rhs[col] /= div;
        for (int row = 0; row < n; ++row) {
            if (row == col || matrix[row][col] == 0.0) continue;
            const double factor = matrix[row][col];
            for (int j = col; j < n; ++j) matrix[row][j] -= factor * matrix[col][j];
            rhs[row] -= factor * rhs[col];
        }
    }
    solution = std::move(rhs);
    return true;
}

std::vector<double> fitElo(const std::vector<GameRecord>& games, int count, int anchor,
                           const std::vector<double>& initial, int iterations = 60) {
    constexpr double k = 0.005756462732485114; // ln(10)/400
    constexpr double lambda = 1e-6;
    std::vector<double> rating = initial;
    if (static_cast<int>(rating.size()) != count) rating.assign(count, 0.0);
    if (!rating.empty()) rating[anchor] = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<std::vector<double>> hessian(count, std::vector<double>(count, 0.0));
        std::vector<double> grad(count, 0.0);
        for (const GameRecord& game : games) {
            const double p = 1.0 / (1.0 + std::exp(-(rating[game.a] - rating[game.b]) * k));
            const double w = k * k * p * (1.0 - p);
            hessian[game.a][game.a] += w;
            hessian[game.b][game.b] += w;
            hessian[game.a][game.b] -= w;
            hessian[game.b][game.a] -= w;
            const double g = (game.scoreA - p) * k;
            grad[game.a] -= g;
            grad[game.b] += g;
        }
        for (int i = 0; i < count; ++i) {
            hessian[i][i] += lambda;
            grad[i] += lambda * rating[i];
        }
        // Hard anchor: random is exactly 0.
        for (int i = 0; i < count; ++i) {
            hessian[anchor][i] = 0.0;
            hessian[i][anchor] = 0.0;
        }
        hessian[anchor][anchor] = 1.0;
        grad[anchor] = rating[anchor];

        std::vector<double> delta;
        if (!solveLinear(hessian, grad, delta)) break;
        double maxDelta = 0.0;
        for (double d : delta) maxDelta = std::max(maxDelta, std::abs(d));

        // Damped Newton step.
        double step = 1.0;
        const double currentLoss = eloLoss(games, rating, lambda);
        std::vector<double> candidate = rating;
        for (int line = 0; line < 20; ++line) {
            for (int i = 0; i < count; ++i) candidate[i] = rating[i] - step * delta[i];
            candidate[anchor] = 0.0;
            if (eloLoss(games, candidate, lambda) <= currentLoss) break;
            step *= 0.5;
        }
        rating = candidate;
        if (maxDelta * step < 1e-4) break;
    }
    rating[anchor] = 0.0;
    return rating;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = p * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(values.size() - 1, lo + 1);
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

void writeReport(const Options& options, const std::vector<EngineKind>& engines,
                 const std::vector<double>& elo,
                 const std::vector<std::array<double, 2>>& ci, const std::vector<Stats>& stats,
                 const std::string& nnInfo) {
    std::ostringstream markdown;
    markdown << "# 数弈引擎 ELO 报告\n\n";
    markdown << "- 基准：`random` 固定为 0；其余为相对 ELO。\n";
    markdown << "- 时间控制：每手 `" << options.timeMs << " ms`；每对局数：`"
             << options.gamesPerPair << "`；随机开局手数：`" << options.openingPlies << "`。\n";
    markdown << "- 种子：" << options.seed << "；A值：+1。\n";
    if (!nnInfo.empty()) markdown << "- MCTS-NN 模型：" << nnInfo << "。\n";
    markdown << "- 置信区间：对局 bootstrap 95%；对局数有限，区间不可省略。\n\n";
    markdown << "| 引擎 | 对局 | 胜/平/负 | 得分率 | ELO | 95% CI |\n";
    markdown << "|---|---:|---:|---:|---:|---:|\n";
    for (std::size_t i = 0; i < engines.size(); ++i) {
        markdown << "| `" << engineName(engines[i]) << "` | " << stats[i].games << " | "
                 << stats[i].wins << "/" << stats[i].draws << "/" << stats[i].losses << " | ";
        if (stats[i].games) markdown << std::fixed << std::setprecision(3)
                                    << stats[i].score / stats[i].games;
        else markdown << "-";
        markdown << " | " << std::fixed << std::setprecision(1) << elo[i] << " | ["
                 << ci[i][0] << ", " << ci[i][1] << "] |\n";
    }
    markdown << "\n> ELO 会随时间控制、对局数、开局随机化和 NN 模型版本变化；"
             << "不可把本表当作与外部引擎可比的绝对强度。\n";

    const auto outPath = std::filesystem::path(options.outPath);
    const auto csvPath = std::filesystem::path(options.csvPath);
    if (!outPath.parent_path().empty()) std::filesystem::create_directories(outPath.parent_path());
    std::ofstream out = openOutUtf8(outPath.string(), std::ios::out);
    out << markdown.str();

    if (!csvPath.parent_path().empty()) std::filesystem::create_directories(csvPath.parent_path());
    std::ofstream csv = openOutUtf8(csvPath.string(), std::ios::out);
    csv << "engine,games,wins,draws,losses,score_rate,elo,ci_low,ci_high\n";
    for (std::size_t i = 0; i < engines.size(); ++i) {
        csv << engineName(engines[i]) << "," << stats[i].games << "," << stats[i].wins << ","
            << stats[i].draws << "," << stats[i].losses << ","
            << (stats[i].games ? stats[i].score / stats[i].games : 0.0) << "," << elo[i] << ","
            << ci[i][0] << "," << ci[i][1] << "\n";
    }
    std::cout << markdown.str();
}

} // namespace

int main(int argc, char** argv) {
    shuyi::Utf8Args utf8Args(argc, argv);
    argv = utf8Args.pointers();
    Options options;
    if (!parseArgs(argc, argv, options)) return 1;

    const int count = static_cast<int>(options.engines.size());
    const int anchor = static_cast<int>(std::find(options.engines.begin(), options.engines.end(),
                                                  EngineKind::Random) -
                                        options.engines.begin());
    std::vector<GameRecord> games;
    std::vector<Stats> stats(count);

    std::uint64_t gameSeed = options.seed;
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            for (int g = 0; g < options.gamesPerPair; ++g) {
                const bool aIsRed = (g % 2) == 0;
                GameRecord record = playGame(options.engines[i], options.engines[j], options, gameSeed++, aIsRed);
                record.a = i;
                record.b = j;
                games.push_back(record);
            }
            std::cerr << "pair " << engineName(options.engines[i]) << " vs "
                      << engineName(options.engines[j]) << " done\n";
        }
    }

    for (const GameRecord& game : games) {
        Stats& a = stats[game.a];
        Stats& b = stats[game.b];
        ++a.games;
        ++b.games;
        a.score += game.scoreA;
        b.score += 1.0 - game.scoreA;
        if (game.scoreA > 0.75) { ++a.wins; ++b.losses; }
        else if (game.scoreA < 0.25) { ++a.losses; ++b.wins; }
        else { ++a.draws; ++b.draws; }
    }

    std::vector<double> initial(count, 0.0);
    std::vector<double> elo = fitElo(games, count, anchor, initial);

    constexpr int kBootstrap = 200;
    std::vector<std::vector<double>> samples(count);
    std::mt19937_64 rng(options.seed ^ 0xB007E1A0ULL);
    for (int b = 0; b < kBootstrap; ++b) {
        std::vector<GameRecord> resampled;
        resampled.reserve(games.size());
        for (std::size_t i = 0; i < games.size(); ++i) {
            std::uniform_int_distribution<std::size_t> pick(0, games.size() - 1);
            resampled.push_back(games[pick(rng)]);
        }
        const std::vector<double> fitted = fitElo(resampled, count, anchor, elo, 40);
        for (int i = 0; i < count; ++i) samples[i].push_back(fitted[i]);
    }
    std::vector<std::array<double, 2>> ci(count);
    for (int i = 0; i < count; ++i) {
        ci[i][0] = percentile(samples[i], 0.025);
        ci[i][1] = percentile(samples[i], 0.975);
    }

    std::string nnInfo = options.nnModel.empty() ? "未指定（mcts_nn 使用无模型回退）" : options.nnModel;
    writeReport(options, options.engines, elo, ci, stats, nnInfo);
    return 0;
}
