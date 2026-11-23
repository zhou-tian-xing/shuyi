// MCTS self-play data generator for the ResNet trainer.
//
// Each sample stores the board, all remaining hand counts, side to move, the
// root MCTS visit distribution (policy target), the eventual game result
// (value target), plus a game id and ply.  The metadata lets the trainer split
// train/validation by game instead of leaking near-identical positions from the
// same game into both sides.
//
// The search budget is a *visit count*, not a wall-clock budget.  A time limit
// makes the target quality depend on the machine: at 10 ms on a slow CPU the
// search produced 11 simulations against ~350 legal moves, i.e. no search at
// all.  --visits makes the data reproducible and comparable across machines.
// --threads runs that many games concurrently, each with its own engine.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "shuyi/engine.h"
#include "shuyi/nn.h"
#include "shuyi/io.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace shuyi;

namespace {
constexpr char kMagic[8] = {'S', 'Y', 'S', 'E', 'L', 'F', '2', '\0'};
constexpr std::uint32_t kVersion = 2;

struct Sample {
    std::uint8_t side = 0;
    std::array<std::int8_t, kCells> board{};
    std::array<std::uint8_t, 18> hand{};
    std::array<float, kNnPolicy> policy{};
    float value = 0.5f;
    std::uint32_t game = 0;
    std::uint8_t ply = 0;
};

struct Options {
    int games = 100;
    int timeMs = 0;
    int visits = 0;
    int threads = 1;
    int tempMoves = 12;
    int openingPlies = 2;
    int rootChildren = 32;
    int maxPlies = kTotalMoves;
    std::uint32_t gameOffset = 0;
    double temperature = 1.0;
    double dirichletAlpha = 0.3;
    double noiseWeight = 0.25;
    std::uint64_t seed = 20261122;
    EngineKind kind = EngineKind::Mcts;
    std::string modelPath;
    std::string output = "data/selfplay.bin";
};

void usage() {
    std::cout
        << "usage: shuyi_selfplay [--games N] [--visits N | --time MS] [--threads N]\n"
        << "       [--engine mcts|mcts_nn] [--model PATH] [--out PATH] [--seed N]\n"
        << "       [--temp T] [--temp-moves N] [--opening N] [--root-children N]\n"
        << "       [--max-plies N] [--game-offset N] [--dirichlet A] [--noise W]\n"
        << "\n"
        << "  --visits N   simulations per move; overrides --time (recommended)\n";
}

bool parse(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return {};
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            usage();
            return false;
        } else if (arg == "--games") {
            options.games = std::max(1, std::stoi(next("--games")));
        } else if (arg == "--time") {
            options.timeMs = std::max(1, std::stoi(next("--time")));
        } else if (arg == "--visits") {
            options.visits = std::max(1, std::stoi(next("--visits")));
        } else if (arg == "--threads") {
            options.threads = std::clamp(std::stoi(next("--threads")), 1, 256);
        } else if (arg == "--temp") {
            options.temperature = std::stod(next("--temp"));
        } else if (arg == "--temp-moves") {
            options.tempMoves = std::max(0, std::stoi(next("--temp-moves")));
        } else if (arg == "--opening") {
            options.openingPlies = std::max(0, std::stoi(next("--opening")));
        } else if (arg == "--root-children") {
            options.rootChildren = std::max(1, std::stoi(next("--root-children")));
        } else if (arg == "--max-plies") {
            options.maxPlies = std::clamp(std::stoi(next("--max-plies")), 2, kTotalMoves);
        } else if (arg == "--game-offset") {
            options.gameOffset = static_cast<std::uint32_t>(std::stoul(next("--game-offset")));
        } else if (arg == "--dirichlet") {
            options.dirichletAlpha = std::stod(next("--dirichlet"));
        } else if (arg == "--noise") {
            options.noiseWeight = std::stod(next("--noise"));
        } else if (arg == "--seed") {
            options.seed = std::stoull(next("--seed"));
        } else if (arg == "--model") {
            options.modelPath = next("--model");
        } else if (arg == "--out") {
            options.output = next("--output");
        } else if (arg == "--engine") {
            const std::string name = next("--engine");
            if (!parseEngineName(name, options.kind)) {
                std::cerr << "unknown engine: " << name << "\n";
                return false;
            }
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            usage();
            return false;
        }
    }
    if (options.visits <= 0 && options.timeMs <= 0) {
        std::cerr << "give a search budget: --visits N (preferred) or --time MS\n";
        return false;
    }
    return true;
}

std::vector<int> policyCounts(const std::vector<PolicyVisit>& visits, const Position& position) {
    std::vector<int> counts(kNnPolicy, 0);
    for (const PolicyVisit& visit : visits) {
        if (position.isLegal(visit.move)) {
            counts[visit.move.cell * 9 + (visit.move.number - 1)] += visit.visits;
        }
    }
    bool any = false;
    for (int count : counts) any |= count > 0;
    if (!any) {
        const auto legal = position.legalMoves();
        for (const Move& move : legal) counts[move.cell * 9 + (move.number - 1)] = 1;
    }
    return counts;
}

void normalizePolicy(std::array<float, kNnPolicy>& policy, const std::vector<int>& counts) {
    double sum = 0.0;
    for (int count : counts) sum += count;
    if (sum <= 0.0) return;
    for (int i = 0; i < kNnPolicy; ++i) policy[i] = static_cast<float>(counts[i] / sum);
}

Move chooseMove(const std::vector<int>& counts, const Position& position, double temperature,
                bool explore, std::mt19937_64& rng) {
    std::vector<Move> legal = position.legalMoves();
    std::vector<double> weights;
    weights.reserve(legal.size());
    double best = -1.0;
    Move bestMove = legal.front();
    for (Move move : legal) {
        const double count = counts[move.cell * 9 + (move.number - 1)];
        if (count > best) {
            best = count;
            bestMove = move;
        }
        weights.push_back(explore ? std::pow(count + 1e-9, 1.0 / std::max(0.05, temperature))
                                  : count);
    }
    if (!explore) return bestMove;
    std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
    return legal[dist(rng)];
}

double outcomeValue(const Position& position, Player side) {
    if (!position.gameOver()) return 0.5;
    const double total = position.finalScore(1.0);
    if (total > 0.0) return side == Player::Red ? 1.0 : 0.0;
    if (total < 0.0) return side == Player::Black ? 1.0 : 0.0;
    return 0.5;
}

bool writeDataset(const std::string& path, const std::vector<Sample>& samples) {
    std::ofstream out = openOutUtf8(path);
    if (!out) return false;
    out.write(kMagic, sizeof(kMagic));
    const std::uint32_t version = kVersion;
    const std::uint32_t count = static_cast<std::uint32_t>(samples.size());
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const Sample& sample : samples) {
        out.write(reinterpret_cast<const char*>(&sample.side), sizeof(sample.side));
        out.write(reinterpret_cast<const char*>(sample.board.data()), sample.board.size());
        out.write(reinterpret_cast<const char*>(sample.hand.data()), sample.hand.size());
        out.write(reinterpret_cast<const char*>(sample.policy.data()),
                  static_cast<std::streamsize>(sample.policy.size() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(&sample.value), sizeof(sample.value));
        out.write(reinterpret_cast<const char*>(&sample.game), sizeof(sample.game));
        out.write(reinterpret_cast<const char*>(&sample.ply), sizeof(sample.ply));
    }
    return static_cast<bool>(out);
}

std::mutex g_logMutex;

} // namespace

int main(int argc, char** argv) {
    shuyi::Utf8Args utf8Args(argc, argv);
    argv = utf8Args.pointers();
    Options options;
    if (!parse(argc, argv, options)) return 1;

    const auto started = std::chrono::steady_clock::now();
    std::atomic<int> nextGame{0};
    std::vector<std::vector<Sample>> perThread(options.threads);
    std::vector<std::pair<int, int>> winsPerThread(options.threads, {0, 0});
    std::vector<std::thread> workers;
    workers.reserve(options.threads);

    for (int tid = 0; tid < options.threads; ++tid) {
        workers.emplace_back([&, tid]() {
            EngineOptions engineOptions;
            engineOptions.seed = options.seed + 7919ULL * static_cast<std::uint64_t>(tid);
            engineOptions.timeLimitMs = options.visits > 0 ? 0 : options.timeMs;
            engineOptions.visits = options.visits;
            engineOptions.aValue = 1.0;
            engineOptions.mctsRootChildren = options.rootChildren;
            engineOptions.rootDirichletAlpha = options.dirichletAlpha;
            engineOptions.rootNoiseWeight = options.noiseWeight;
            engineOptions.nnModelPath = options.modelPath;
            auto engine = makeEngine(options.kind, engineOptions);
            if (!engine) {
                std::lock_guard<std::mutex> lock(g_logMutex);
                std::cerr << "cannot create engine\n";
                return;
            }
            std::mt19937_64 rng(options.seed ^ (0xD1A1C4E5ULL + 0x9E37ULL * tid));
            auto& samples = perThread[tid];
            int red = 0, black = 0;

            while (true) {
                const int game = nextGame.fetch_add(1, std::memory_order_relaxed);
                if (game >= options.games) break;

                Position position;
                engine->reset();
                const int opening = std::min(options.openingPlies, options.maxPlies - 1);
                for (int i = 0; i < opening && !position.gameOver(); ++i) {
                    const auto moves = position.legalMoves();
                    if (moves.empty()) break;
                    std::uniform_int_distribution<std::size_t> pick(0, moves.size() - 1);
                    const Move move = moves[pick(rng)];
                    position.play(move);
                    engine->onPlay(position, move);
                }

                std::vector<Sample> gameSamples;
                while (!position.gameOver() && position.ply() < options.maxPlies) {
                    SearchRequest request;
                    request.visits = options.visits;
                    request.timeLimitMs = options.visits > 0 ? 0 : options.timeMs;
                    const Analysis analysis = engine->search(position, request);
                    const auto visits = engine->rootPolicy();
                    const std::vector<int> counts = policyCounts(visits, position);

                    Sample sample;
                    sample.side = position.sideToMove() == Player::Red ? 0 : 1;
                    for (int cell = 0; cell < kCells; ++cell) {
                        sample.board[cell] = static_cast<std::int8_t>(position.stoneAt(cell));
                    }
                    for (int side = 0; side < 2; ++side) {
                        const Player player = side == 0 ? Player::Red : Player::Black;
                        for (int number = 1; number <= 9; ++number) {
                            sample.hand[side * 9 + (number - 1)] =
                                static_cast<std::uint8_t>(position.remaining(player, number));
                        }
                    }
                    normalizePolicy(sample.policy, counts);
                    sample.game = options.gameOffset + static_cast<std::uint32_t>(game);
                    sample.ply = static_cast<std::uint8_t>(position.ply());

                    const bool explore = static_cast<int>(gameSamples.size()) < options.tempMoves;
                    Move move = chooseMove(counts, position, options.temperature, explore, rng);
                    if (!analysis.valid || !position.isLegal(move)) {
                        const auto legal = position.legalMoves();
                        if (legal.empty()) break;
                        move = legal.front();
                    }
                    gameSamples.push_back(sample);
                    position.play(move);
                    engine->onPlay(position, move);
                }

                const double redValue = position.gameOver() ? outcomeValue(position, Player::Red) : 0.5;
                for (Sample& sample : gameSamples) {
                    sample.value = static_cast<float>(sample.side == 0 ? redValue : 1.0 - redValue);
                    samples.push_back(std::move(sample));
                }
                if (redValue > 0.5) ++red;
                else if (redValue < 0.5) ++black;
                {
                    std::lock_guard<std::mutex> lock(g_logMutex);
                    const double elapsed =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                    std::cerr << "game " << nextGame.load() << "/" << options.games
                              << " samples=" << samples.size() << " threads=" << options.threads
                              << " elapsed=" << elapsed << "s\n";
                }
            }
            winsPerThread[tid] = {red, black};
        });
    }
    for (auto& worker : workers) worker.join();

    std::vector<Sample> allSamples;
    std::size_t total = 0;
    for (const auto& samples : perThread) total += samples.size();
    allSamples.reserve(total);
    for (const auto& samples : perThread) {
        allSamples.insert(allSamples.end(), samples.begin(), samples.end());
    }
    std::sort(allSamples.begin(), allSamples.end(), [](const Sample& a, const Sample& b) {
        if (a.game != b.game) return a.game < b.game;
        return a.ply < b.ply;
    });

    int red = 0, black = 0;
    for (const auto& pair : winsPerThread) {
        red += pair.first;
        black += pair.second;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    if (!writeDataset(options.output, allSamples)) {
        std::cerr << "cannot write dataset: " << options.output << "\n";
        return 1;
    }
    const double positions = static_cast<double>(allSamples.size());
    const double simulations = positions * (options.visits > 0 ? options.visits : 0);
    std::cout << "wrote " << allSamples.size() << " samples to " << options.output << "\n";
    std::cout << "games=" << options.games << " threads=" << options.threads
              << " visits=" << options.visits << " seconds=" << seconds
              << " games/s=" << (options.games / seconds)
              << " positions/s=" << (positions / seconds)
              << " sims/s=" << (simulations / seconds) << "\n";
    std::cout << "red=" << red << " black=" << black << "\n";
    return 0;
}
