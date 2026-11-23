// Native C++/CUDA search driver: self-play, one-ply value scan, GPU arena,
// and a forward-throughput benchmark.
//
// Why this file exists
// --------------------
// The engine's tree lives in C++ and every simulation needs exactly one neural
// evaluation, so the only sane design is: descend one leaf per game, evaluate
// all leaves of the tick in ONE batched CUDA call, expand and back up.  There is
// no Python and no per-simulation host round trip anywhere in here.
//
// The search budget is always a visit count (--visits), never a time limit: a
// time limit silently degrades into "11 simulations on a slow machine", which
// produces policy targets that are pure noise (target entropy == uniform
// entropy, max probability below uniform).  That is how two generations were
// wasted, so time is not an option here at all.
//
// Modes
// -----
//   selfplay  MCTS self-play with the given model, writes a SYSELF2 shard
//   oneply    full-width one-ply value scan: every legal move is evaluated and
//             the policy target is softmax((Q - max Q) / T).  Branching at the
//             start of the game is 729 (81 cells x 9 numbers), so a tree search
//             with a few hundred visits cannot produce a learnable target until
//             the prior is already informative.  This mode bootstraps it.
//   match     A vs B arena, colours alternating, argmax moves, GPU evaluation
//   bench     forward throughput sweep, used to size --batch-games
#include "shuyi/engine.h"
#include "shuyi/nn.h"
#include "shuyi/nn_cuda.h"
#include "shuyi/io.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace shuyi;

namespace {

// v3 added the rule-set byte; v4 appends the game's final score to every sample,
// which is the target of the score head.
constexpr char kMagic[8] = {'S', 'Y', 'S', 'E', 'L', 'F', '4', '\0'};
constexpr std::uint32_t kVersion = 4;
constexpr int kMaxPlies = 54;
constexpr int kNumbers = 9;
constexpr int kMoves = kCells * kNumbers; // 729

struct Node {
    std::array<std::int8_t, kCells> board{};
    std::array<std::uint8_t, 18> rem{};
    std::int16_t raw = 0;
    std::uint8_t ply = 0;    // turns played, passes included
    std::uint8_t side = 0;
    std::uint8_t stones = 0; // stones placed (0..54)
    std::uint8_t passes = 0; // consecutive passes, 2 ends the game
};

// A game ends when all 54 stones are placed or both sides passed in a row.
bool terminalState(const Node& node) {
    return node.stones >= kMaxPlies || node.passes >= 2;
}

struct Edge {
    float prior = 0.0f;
    float visits = 0.0f;
    float value = 0.0f;
    int child = -1;
};

struct Tree {
    std::vector<Node> nodes;
    std::vector<int> firstEdge;
    std::vector<int> edgeCount;
    std::vector<int> nodeVisits;
    std::vector<Edge> edges;
    std::vector<Move> move;
    std::vector<std::uint8_t> terminal;

    void reset(const Node& root) {
        nodes.assign(1, root);
        firstEdge.assign(1, 0);
        edgeCount.assign(1, 0);
        nodeVisits.assign(1, 0);
        terminal.assign(1, 0);
        edges.clear();
        move.clear();
    }

    int newNode(const Node& node) {
        const int index = static_cast<int>(nodes.size());
        nodes.push_back(node);
        firstEdge.push_back(0);
        edgeCount.push_back(0);
        nodeVisits.push_back(0);
        terminal.push_back(0);
        return index;
    }
};

struct Options {
    std::string mode = "selfplay";
    int games = 100;
    int batchGames = 128;
    int visits = 1024;
    int rootK = 32;   // 0 = no truncation at the root (full width)
    int topK = 16;    // 0 = no truncation inside the tree
    int opening = 2;
    int tempMoves = 12;
    int chunk = 1024;
    int threads = 0; // 0 = all hardware threads
    bool stats = false;      // per-phase timing of a tick
    Ruleset ruleset = kDefaultRuleset;
    double temperature = 1.0;
    double oneplyTemperature = 0.05; // used when --oneply-temp-scale is 0
    double oneplyTempScale = 0.5;    // T = scale * std(Q) over the legal moves
    double noise = 0.25;
    double dirichlet = 0.3;
    double cpuct = 1.5;
    double komi = 1.0;
    std::uint64_t seed = 20261122;
    std::string model;
    std::string modelB;
    std::string nameA = "A";
    std::string nameB = "B";
    std::string out = "data/selfplay.bin";
    std::string jsonOut;
    std::string csvOut;
    std::string dumpFeatures;
    bool check = false;
};

void usage() {
    std::cout
        << "usage: shuyi_selfplay_gpu --model PATH [options]\n"
        << "\n"
        << "  --mode selfplay|oneply|match|bench   (default selfplay)\n"
        << "  --model PATH         model for selfplay/oneply, side A for match\n"
        << "  --model-b PATH       side B model (match)\n"
        << "  --games N            number of games\n"
        << "  --batch-games N      games evaluated per CUDA call\n"
        << "  --visits N           simulations per move (selfplay/match)\n"
        << "  --root-k N           root candidates kept, 0 = all legal moves\n"
        << "  --top-k N            interior candidates kept, 0 = all legal moves\n"
        << "  --komi X             points added to Red's raw score\n"
        << "  --opening N          random opening plies\n"
        << "  --temp T / --temp-moves N   self-play temperature schedule\n"
        << "  --noise W / --dirichlet A   Dirichlet root noise\n"
        << "  --cpuct C            PUCT constant\n"
        << "  --oneply-temp T      softmax temperature for oneply targets\n"
        << "  --chunk N            positions per CUDA call inside one batch\n"
        << "  --out PATH           shard path (selfplay/oneply)\n"
        << "  --json PATH / --csv PATH   match results\n"
        << "  --seed N\n"
        << "  --check              rule parity + CUDA vs CPU reference\n"
        << "  --dump-features PATH write feature planes for the layout test\n";
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&](const char* name) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") {
            usage();
            return false;
        } else if (arg == "--mode") {
            o.mode = next("--mode");
        } else if (arg == "--model") {
            o.model = next("--model");
        } else if (arg == "--model-b") {
            o.modelB = next("--model-b");
        } else if (arg == "--name-a") {
            o.nameA = next("--name-a");
        } else if (arg == "--name-b") {
            o.nameB = next("--name-b");
        } else if (arg == "--games") {
            o.games = std::max(1, std::stoi(next("--games")));
        } else if (arg == "--batch-games") {
            o.batchGames = std::clamp(std::stoi(next("--batch-games")), 1, 8192);
        } else if (arg == "--visits") {
            o.visits = std::max(1, std::stoi(next("--visits")));
        } else if (arg == "--root-k") {
            o.rootK = std::max(0, std::stoi(next("--root-k")));
        } else if (arg == "--top-k") {
            o.topK = std::max(0, std::stoi(next("--top-k")));
        } else if (arg == "--opening") {
            o.opening = std::max(0, std::stoi(next("--opening")));
        } else if (arg == "--temp-moves") {
            o.tempMoves = std::max(0, std::stoi(next("--temp-moves")));
        } else if (arg == "--ruleset") {
            if (!parseRuleset(next("--ruleset"), o.ruleset)) {
                std::cerr << "--ruleset must be classic or blocking" << std::endl;
                return false;
            }
        } else if (arg == "--stats") {
            o.stats = true;
        } else if (arg == "--threads") {
            o.threads = std::max(1, std::stoi(next("--threads")));
        } else if (arg == "--chunk") {
            o.chunk = std::clamp(std::stoi(next("--chunk")), 64, 65536);
        } else if (arg == "--temperature" || arg == "--temp") {
            o.temperature = std::stod(next("--temperature"));
        } else if (arg == "--oneply-temp") {
            o.oneplyTemperature = std::stod(next("--oneply-temp"));
            o.oneplyTempScale = 0.0;
        } else if (arg == "--oneply-temp-scale") {
            o.oneplyTempScale = std::stod(next("--oneply-temp-scale"));
        } else if (arg == "--noise") {
            o.noise = std::stod(next("--noise"));
        } else if (arg == "--dirichlet") {
            o.dirichlet = std::stod(next("--dirichlet"));
        } else if (arg == "--cpuct") {
            o.cpuct = std::stod(next("--cpuct"));
        } else if (arg == "--komi") {
            o.komi = std::stod(next("--komi"));
        } else if (arg == "--seed") {
            o.seed = std::stoull(next("--seed"));
        } else if (arg == "--out") {
            o.out = next("--out");
        } else if (arg == "--json") {
            o.jsonOut = next("--json");
        } else if (arg == "--csv") {
            o.csvOut = next("--csv");
        } else if (arg == "--dump-features") {
            o.dumpFeatures = next("--dump-features");
        } else if (arg == "--check") {
            o.check = true;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    if (o.model.empty() && o.mode != "random") {
        std::cerr << "--model is required\n";
        return false;
    }
    if (o.mode == "match" && o.modelB.empty()) {
        std::cerr << "--model-b is required for --mode match\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- rules ---

std::array<int, 5> crossOf(int cell) {
    const int x = cell % 9;
    const int y = cell / 9;
    std::array<int, 5> cells{};
    int n = 0;
    cells[n++] = cell;
    if (x > 0) cells[n++] = cell - 1;
    if (x < 8) cells[n++] = cell + 1;
    if (y > 0) cells[n++] = cell - 9;
    if (y < 8) cells[n++] = cell + 9;
    while (n < 5) cells[n++] = -1; // unique cells only: duplicates would double count
    return cells;
}

int scoreSign(const std::array<std::int8_t, kCells>& board, int cell) {
    if (cell < 0) return 0;
    const int x = cell % 9;
    const int y = cell / 9;
    int redMax = 0;
    int blackMax = 0;
    const auto add = [&](int xx, int yy) {
        if (xx < 0 || xx > 8 || yy < 0 || yy > 8) return;
        const int v = board[yy * 9 + xx];
        if (v > 0) redMax = std::max<int>(redMax, v);
        else if (v < 0) blackMax = std::max<int>(blackMax, -v);
    };
    add(x, y);
    add(x - 1, y);
    add(x + 1, y);
    add(x, y - 1);
    add(x, y + 1);
    return (redMax > blackMax) - (redMax < blackMax);
}

Node makeStart() {
    Node node;
    node.rem.fill(3);
    node.stones = 0;
    node.passes = 0;
    return node;
}

// Largest opposing number in the four orthogonal neighbours of `cell`.
int blockingReach(const Node& node, int cell) {
    const int x = cell % 9;
    const int y = cell / 9;
    const int mySign = (node.side == 0) ? 1 : -1;
    int reach = 0;
    const auto consider = [&](int xx, int yy) {
        if (xx < 0 || xx > 8 || yy < 0 || yy > 8) return;
        const int v = node.board[yy * 9 + xx];
        if (v == 0 || v * mySign > 0) return; // empty or my own
        reach = std::max(reach, v > 0 ? v : -v);
    };
    consider(x - 1, y);
    consider(x + 1, y);
    consider(x, y - 1);
    consider(x, y + 1);
    return reach;
}

bool play(Node& node, Move mv, Ruleset ruleset);

// Forced pass: the side to move has no placement but the game goes on.
bool passTurn(Node& node) {
    ++node.ply;
    ++node.passes;
    node.side = static_cast<std::uint8_t>(node.ply & 1);
    return true;
}

bool applyMove(Node& node, Move mv, Ruleset ruleset) {
    if (mv.number == kPassNumber) return passTurn(node);
    return play(node, mv, ruleset);
}

bool play(Node& node, Move mv, Ruleset ruleset) {
    if (mv.cell >= kCells || mv.number < 1 || mv.number > 9) return false;
    if (node.board[mv.cell] != 0) return false;
    const int side = node.ply & 1;
    if (node.rem[side * 9 + (mv.number - 1)] == 0) return false;
    if (ruleset == Ruleset::Blocking && mv.number <= blockingReach(node, mv.cell)) return false;
    const auto cross = crossOf(mv.cell);
    int before = 0;
    for (int i = 0; i < 5; ++i) before += scoreSign(node.board, cross[i]);
    node.board[mv.cell] = static_cast<std::int8_t>(side == 0 ? mv.number : -mv.number);
    int after = 0;
    for (int i = 0; i < 5; ++i) after += scoreSign(node.board, cross[i]);
    node.raw = static_cast<std::int16_t>(node.raw + after - before);
    --node.rem[side * 9 + (mv.number - 1)];
    ++node.ply;
    ++node.stones;
    node.passes = 0;
    node.side = static_cast<std::uint8_t>(node.ply & 1);
    return true;
}

double terminalValue(const Node& node, double komi) {
    const double total = static_cast<double>(node.raw) + komi;
    if (total > 0.0) return node.side == 0 ? 1.0 : 0.0;
    if (total < 0.0) return node.side == 1 ? 1.0 : 0.0;
    return 0.5;
}

void legalMoves(const Node& node, std::vector<Move>& out, Ruleset ruleset) {
    out.clear();
    if (terminalState(node)) return;
    const int side = node.side;
    for (int number = 1; number <= kNumbers; ++number) {
        if (node.rem[side * 9 + (number - 1)] == 0) continue;
        for (int cell = 0; cell < kCells; ++cell) {
            if (node.board[cell] != 0) continue;
            if (ruleset == Ruleset::Blocking && number <= blockingReach(node, cell)) continue;
            out.emplace_back(cell, number);
        }
    }
}

// ------------------------------------------------------------ evaluation ---

class Evaluator {
public:
    bool init(const std::vector<std::string>& paths, int chunk, std::string* error) {
        chunk_ = std::max(64, chunk);
        for (const std::string& path : paths) {
            auto cpu = std::make_unique<PolicyValueNet>();
            std::string err;
            if (!cpu->load(path, &err)) {
                if (error) *error = path + ": " + err;
                return false;
            }
            // Models may use different feature layouts (19-plane legacy vs
            // 37-plane hand planes) or different widths; each one builds its own
            // planes.  Refusing to mix them would make a single rating pool
            // impossible.
            auto gpu = std::make_unique<CudaPolicyValueNet>();
            if (!gpu->load(*cpu, &err)) {
                if (error) *error = path + ": " + err;
                return false;
            }
            cpus_.push_back(std::move(cpu));
            gpus_.push_back(std::move(gpu));
            names_.push_back(path);
        }
        return true;
    }

    int count() const { return static_cast<int>(gpus_.size()); }
    int planes() const { return cpus_[0]->modelInputPlanes(); }
    const PolicyValueNet& cpu(int index) const { return *cpus_[index]; }
    const std::string& name(int index) const { return names_[index]; }
    int channels() const { return cpus_[0]->channels(); }
    Ruleset rulesetOf(int index) const { return cpus_[index]->ruleset(); }
    int blocks() const { return cpus_[0]->blocks(); }
    const std::string& device() const { return gpus_[0]->deviceName(); }

    int planes(int index) const { return cpus_[index]->modelInputPlanes(); }

    // Evaluates nodes[0..count) with net `index`.  logits: [count][kNnPolicy].
    // scores is optional and only filled for models with a score head.
    void eval(int index, const Node* nodes, int count, float* logits, float* values,
              float* scores = nullptr) {
        if (count <= 0) return;
        const int planes = cpus_[index]->modelInputPlanes();
        const int chunk = std::min<int>(chunk_, count);
        features_.resize(static_cast<std::size_t>(chunk) * planes * kCells);
        globals_.resize(static_cast<std::size_t>(chunk) * kResGlobalFeatures);
        for (int base = 0; base < count; base += chunk) {
            const int m = std::min(chunk, count - base);
            for (int i = 0; i < m; ++i) {
                const Node& node = nodes[base + i];
                cpus_[index]->buildFeaturePlanes(
                    node.board, node.ply,
                    features_.data() + static_cast<std::size_t>(i) * planes * kCells);
                cpus_[index]->buildGlobals(node.board, node.ply, node.raw,
                                           globals_.data() + static_cast<std::size_t>(i) *
                                                                 kResGlobalFeatures);
            }
            gpus_[index]->evaluate(features_.data(), globals_.data(), m,
                                   logits + static_cast<std::size_t>(base) * kNnPolicy,
                                   values + base, scores ? scores + base : nullptr);
        }
    }

private:
    std::vector<std::unique_ptr<PolicyValueNet>> cpus_;
    std::vector<std::unique_ptr<CudaPolicyValueNet>> gpus_;
    std::vector<std::string> names_;
    std::vector<float> features_;
    std::vector<float> globals_;
    int chunk_ = 1024;
};

// --------------------------------------------------------------- samples ---

struct Sample {
    std::uint8_t side = 0;
    std::array<std::int8_t, kCells> board{};
    std::array<std::uint8_t, 18> hand{};
    std::array<float, kNnPolicy> policy{};
    float value = 0.5f;
    std::uint32_t game = 0;
    std::uint8_t ply = 0;
    // Final raw score of the game from the mover's perspective, clamped to the
    // int8 range.  It rides in the shard's reserved byte so the format stays
    // compatible; it is the target of the score head.
    std::int8_t finalScore = 0;
};

std::int8_t clampScoreByte(int score) {
    if (score > 127) return 127;
    if (score < -128) return -128;
    return static_cast<std::int8_t>(score);
}

bool writeShard(const std::string& path, const std::vector<Sample>& samples, Ruleset ruleset) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    std::ofstream out = openOutUtf8(path);
    if (!out) return false;
    out.write(kMagic, sizeof(kMagic));
    const std::uint32_t version = kVersion;
    const std::uint32_t count = static_cast<std::uint32_t>(samples.size());
    const std::uint8_t rulesetByte = static_cast<std::uint8_t>(ruleset);
    const std::uint8_t reserved[3] = {0, 0, 0};
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(&rulesetByte), sizeof(rulesetByte));
    out.write(reinterpret_cast<const char*>(reserved), sizeof(reserved));
    for (const Sample& s : samples) {
        out.write(reinterpret_cast<const char*>(&s.side), sizeof(s.side));
        out.write(reinterpret_cast<const char*>(s.board.data()), s.board.size());
        out.write(reinterpret_cast<const char*>(s.hand.data()), s.hand.size());
        out.write(reinterpret_cast<const char*>(s.policy.data()),
                  static_cast<std::streamsize>(s.policy.size() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(&s.value), sizeof(s.value));
        out.write(reinterpret_cast<const char*>(&s.game), sizeof(s.game));
        out.write(reinterpret_cast<const char*>(&s.ply), sizeof(s.ply));
        out.write(reinterpret_cast<const char*>(&s.finalScore), sizeof(s.finalScore));
    }
    return static_cast<bool>(out);
}

// Entropy of the policy target is the single number that says whether a shard
// can teach anything at all.  The failure this reports on: with a flat prior and
// a small visit budget the target put 98% of its mass on one *number* and spread
// the rest arbitrarily over cells, so the policy head learned "which number"
// (2.7% top-1) and nothing about the board.  `cell_ratio` is that diagnosis as a
// number: entropy over the cells of the most likely number, divided by uniform.
// Near 1.0 means the cells carry no information.
void reportTargets(const std::vector<Sample>& samples, Ruleset ruleset) {
    if (samples.empty()) return;
    double entropy = 0.0;
    double maxProb = 0.0;
    double top5 = 0.0;
    double legalTotal = 0.0;
    double valueMean = 0.0;
    double support = 0.0;
    double numberMass = 0.0;
    double cellRatio = 0.0;
    std::array<double, 9> massByNumber{};
    for (const Sample& s : samples) {
        std::array<int, 9> legalCells{};
        int legal = 0;
        for (int cell = 0; cell < kCells; ++cell) {
            if (s.board[cell] != 0) continue;
            int reach = 0;
            if (ruleset == Ruleset::Blocking) {
                // Same four-neighbour reach the move generator uses.
                const int x = cell % 9;
                const int y = cell / 9;
                const int mySign = (s.side == 0) ? 1 : -1;
                const auto consider = [&](int xx, int yy) {
                    if (xx < 0 || xx > 8 || yy < 0 || yy > 8) return;
                    const int v = s.board[yy * 9 + xx];
                    if (v == 0 || v * mySign > 0) return;
                    reach = std::max(reach, v > 0 ? v : -v);
                };
                consider(x - 1, y);
                consider(x + 1, y);
                consider(x, y - 1);
                consider(x, y + 1);
            }
            for (int number = 0; number < 9; ++number) {
                if (number + 1 <= reach) continue;
                if (s.hand[s.side * 9 + number] == 0) continue;
                ++legal;
                ++legalCells[number];
            }
        }
        double sum = 0.0;
        double ent = 0.0;
        double top1 = 0.0;
        int nonZero = 0;
        std::array<double, 5> best{};
        std::array<double, 9> byNumber{};
        for (int cell = 0; cell < kCells; ++cell) {
            for (int number = 0; number < 9; ++number) {
                const double p = s.policy[cell * 9 + number];
                if (p <= 0.0) continue;
                ++nonZero;
                sum += p;
                ent -= p * std::log(p);
                byNumber[number] += p;
                if (p > top1) top1 = p;
                for (int k = 0; k < 5; ++k) {
                    if (p > best[k]) {
                        for (int j = 4; j > k; --j) best[j] = best[j - 1];
                        best[k] = p;
                        break;
                    }
                }
            }
        }
        if (sum <= 0.0) continue;
        int topNumber = 0;
        for (int number = 1; number < 9; ++number) {
            if (byNumber[number] > byNumber[topNumber]) topNumber = number;
        }
        // Cell distribution conditioned on the most likely number.
        double cellEnt = 0.0;
        const double cellMass = byNumber[topNumber];
        const int cellsForNumber = legalCells[topNumber];
        for (int cell = 0; cell < kCells; ++cell) {
            const double p = s.policy[cell * 9 + topNumber];
            if (p <= 0.0) continue;
            cellEnt -= (p / cellMass) * std::log(p / cellMass);
        }
        const double cellUniform =
            cellsForNumber > 1 ? std::log(static_cast<double>(cellsForNumber)) : 1.0;
        cellRatio += cellEnt / cellUniform;
        numberMass += cellMass;
        for (int number = 0; number < 9; ++number) massByNumber[number] += byNumber[number];
        entropy += ent;
        maxProb += top1;
        top5 += best[0] + best[1] + best[2] + best[3] + best[4];
        legalTotal += legal;
        support += nonZero;
        valueMean += s.value;
    }
    const double n = static_cast<double>(samples.size());
    std::printf(
        "TARGET_STATS {\"positions\": %.0f, \"entropy\": %.4f, \"max_prob\": %.4f, "
        "\"top5\": %.4f, \"support\": %.2f, \"legal\": %.2f, \"value_mean\": %.4f, "
        "\"number_mass\": %.4f, \"cell_ratio\": %.4f}\n",
        n, entropy / n, maxProb / n, top5 / n, support / n, legalTotal / n, valueMean / n,
        numberMass / n, cellRatio / n);
    std::printf("number mass:");
    for (int number = 0; number < 9; ++number) {
        std::printf(" %d=%.3f", number + 1, massByNumber[number] / n);
    }
    std::printf("\n");
    std::fflush(stdout);
}

// -------------------------------------------------------------- searching ---

struct Search {
    Tree tree;
    std::mt19937_64 rng;
    std::vector<Sample> samples;
    int modelFor[2] = {0, 0}; // model index playing Red (0) and Black (1)
    int net = 0;
};

struct Scratch {
    std::vector<std::vector<std::pair<int, int>>> paths;
    std::vector<int> leaf;
    std::vector<float> logits;
    std::vector<float> values;
    // Persistent per-tick buffers: allocating ~1.5 MB per tick at batch 512 is
    // pure allocator churn and it showed up as the GPU sitting idle 40% of the
    // time during self-play.
    std::vector<Node> group;
    std::vector<int> slot;
    std::vector<float> groupLogits;
    std::vector<float> groupValues;
};

void descend(Tree& tree, double cpuct, std::vector<std::pair<int, int>>& path, int& leaf,
             Ruleset ruleset) {
    int node = 0;
    path.clear();
    while (true) {
        if (terminalState(tree.nodes[node]) || tree.terminal[node]) {
            leaf = node;
            return;
        }
        if (tree.edgeCount[node] == 0) {
            leaf = node;
            return;
        }
        const int begin = tree.firstEdge[node];
        const int end = begin + tree.edgeCount[node];
        const double sqrtN = std::sqrt(static_cast<double>(tree.nodeVisits[node]) + 1.0);
        int bestEdge = -1;
        double bestScore = -1e300;
        for (int e = begin; e < end; ++e) {
            const Edge& edge = tree.edges[e];
            double score;
            if (edge.child < 0) {
                score = 1e9 + edge.prior; // first-visit urgency, in prior order
            } else {
                const double q = edge.visits > 0.0f ? edge.value / edge.visits : 0.0;
                score = q + cpuct * edge.prior * sqrtN / (1.0 + edge.visits);
            }
            if (score > bestScore) {
                bestScore = score;
                bestEdge = e;
            }
        }
        if (bestEdge < 0) {
            leaf = node;
            return;
        }
        path.push_back({node, bestEdge});
        if (tree.edges[bestEdge].child < 0) {
            Node child = tree.nodes[node];
            applyMove(child, tree.move[bestEdge], ruleset);
            const int index = tree.newNode(child);
            tree.edges[bestEdge].child = index;
            leaf = index;
            return;
        }
        node = tree.edges[bestEdge].child;
    }
}

// One simulation per game: descend, one batched CUDA call for the whole tick,
// then expand and back up.
struct TickTiming {
    double descend = 0.0;
    double eval = 0.0;
    double expand = 0.0;
    long long ticks = 0;
};

void tick(Evaluator& evaluator, std::vector<Search>& games, const std::vector<int>& active,
          Scratch& s, const Options& o, TickTiming* timing = nullptr) {
    const int n = static_cast<int>(active.size());
    if (n == 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    s.paths.assign(n, {});
    s.leaf.assign(n, -1);
    // The three per-game phases (descend / expand / back up) touch independent
    // trees, so they parallelise perfectly.  Single-threaded they cost ~4 ms per
    // tick at 512 games while the CUDA call takes ~5 ms, which left the GPU at
    // 27% utilisation; this is what closes that gap.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int i = 0; i < n; ++i) {
        descend(games[active[i]].tree, o.cpuct, s.paths[i], s.leaf[i], o.ruleset);
    }
    const auto t1 = std::chrono::steady_clock::now();
    s.logits.assign(static_cast<std::size_t>(n) * kNnPolicy, 0.0f);
    s.values.assign(n, 0.5f);

    for (int net = 0; net < evaluator.count(); ++net) {
        s.group.clear();
        s.slot.clear();
        for (int i = 0; i < n; ++i) {
            const Search& g = games[active[i]];
            if (g.net != net) continue;
            const Node& node = g.tree.nodes[s.leaf[i]];
            if (terminalState(node)) continue;
            s.group.push_back(node);
            s.slot.push_back(i);
        }
        if (s.group.empty()) continue;
        s.groupLogits.resize(s.group.size() * kNnPolicy);
        s.groupValues.resize(s.group.size());
        evaluator.eval(net, s.group.data(), static_cast<int>(s.group.size()),
                       s.groupLogits.data(), s.groupValues.data());
        for (std::size_t k = 0; k < s.slot.size(); ++k) {
            std::memcpy(s.logits.data() + static_cast<std::size_t>(s.slot[k]) * kNnPolicy,
                        s.groupLogits.data() + k * kNnPolicy, kNnPolicy * sizeof(float));
            s.values[s.slot[k]] = s.groupValues[k];
        }
    }

    const auto t2 = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int i = 0; i < n; ++i) {
        // Per-thread scratch: the expansion sort/softmax buffers must not be
        // shared between games running in parallel.
        static thread_local std::vector<Move> tlsMoves;
        static thread_local std::vector<std::pair<float, int>> tlsRanked;
        static thread_local std::vector<float> tlsPriors;
        static thread_local std::vector<double> tlsNz;
        Search& g = games[active[i]];
        Tree& tree = g.tree;
        const int leaf = s.leaf[i];
        Node& state = tree.nodes[leaf];
        double leafValue;
        if (terminalState(state)) {
            leafValue = terminalValue(state, o.komi);
        } else {
            leafValue = s.values[i];
            if (tree.edgeCount[leaf] == 0) {
                legalMoves(state, tlsMoves, o.ruleset);
                if (tlsMoves.empty()) {
                    // Blocked: the rules force a pass, which is the only move.
                    tlsMoves.emplace_back(kPassCell, kPassNumber);
                }
                {
                    const int limit = leaf == 0 ? o.rootK : o.topK;
                    const int keep =
                        limit <= 0 ? static_cast<int>(tlsMoves.size())
                                   : std::min<int>(limit, static_cast<int>(tlsMoves.size()));
                    tlsRanked.clear();
                    tlsRanked.reserve(tlsMoves.size());
                    for (std::size_t m = 0; m < tlsMoves.size(); ++m) {
                        const int index = tlsMoves[m].cell * kNumbers + (tlsMoves[m].number - 1);
                        tlsRanked.push_back(
                            {s.logits[static_cast<std::size_t>(i) * kNnPolicy + index],
                             static_cast<int>(m)});
                    }
                    if (keep < static_cast<int>(tlsRanked.size())) {
                        std::partial_sort(tlsRanked.begin(), tlsRanked.begin() + keep, tlsRanked.end(),
                                          [](const auto& a, const auto& b) {
                                              return a.first > b.first;
                                          });
                        tlsRanked.resize(keep);
                    }
                    tlsPriors.assign(tlsRanked.size(), 0.0f);
                    float maxLogit = -1e30f;
                    for (const auto& item : tlsRanked) maxLogit = std::max(maxLogit, item.first);
                    double sum = 0.0;
                    for (std::size_t k = 0; k < tlsRanked.size(); ++k) {
                        tlsPriors[k] = std::exp(tlsRanked[k].first - maxLogit);
                        sum += tlsPriors[k];
                    }
                    for (float& p : tlsPriors) p = static_cast<float>(p / sum);
                    if (leaf == 0 && o.noise > 0.0 && tlsPriors.size() > 1) {
                        std::gamma_distribution<double> gamma(o.dirichlet, 1.0);
                        tlsNz.assign(tlsPriors.size(), 0.0);
                        double total = 0.0;
                        for (double& z : tlsNz) {
                            z = gamma(g.rng);
                            total += z;
                        }
                        for (std::size_t k = 0; k < tlsPriors.size(); ++k) {
                            tlsPriors[k] = static_cast<float>((1.0 - o.noise) * tlsPriors[k] +
                                                             o.noise * (tlsNz[k] / total));
                        }
                    }
                    tree.firstEdge[leaf] = static_cast<int>(tree.edges.size());
                    tree.edgeCount[leaf] = static_cast<int>(tlsRanked.size());
                    for (std::size_t k = 0; k < tlsRanked.size(); ++k) {
                        Edge edge;
                        edge.prior = tlsPriors[k];
                        tree.edges.push_back(edge);
                        tree.move.push_back(tlsMoves[tlsRanked[k].second]);
                    }
                }
            }
        }
        double v = leafValue;
        for (auto it = s.paths[i].rbegin(); it != s.paths[i].rend(); ++it) {
            v = 1.0 - v; // a child's win probability is the parent's loss probability
            Edge& edge = tree.edges[it->second];
            edge.visits += 1.0f;
            edge.value += static_cast<float>(v);
            tree.nodeVisits[it->first] += 1;
        }
        tree.nodeVisits[leaf] += 1;
    }
    if (timing) {
        const auto t3 = std::chrono::steady_clock::now();
        timing->descend += std::chrono::duration<double>(t1 - t0).count();
        timing->eval += std::chrono::duration<double>(t2 - t1).count();
        timing->expand += std::chrono::duration<double>(t3 - t2).count();
        timing->ticks += 1;
    }
}

// Store the root sample, pick a move, and restart the tree at the child.  The
// tree is not reused across moves: the values would come from the same net, so
// reuse is only a speed optimisation, and a fresh tree keeps the bookkeeping
// honest (and makes colour-alternating matches correct).
bool advance(Search& g, const Options& o, int moveNo, bool sample, bool temperature) {
    Tree& tree = g.tree;
    const Node& root = tree.nodes[0];
    if (terminalState(root)) return false;
    const int begin = tree.firstEdge[0];
    const int end = begin + tree.edgeCount[0];
    double total = 0.0;
    int bestEdge = -1;
    for (int e = begin; e < end; ++e) {
        total += tree.edges[e].visits;
        if (bestEdge < 0 || tree.edges[e].visits > tree.edges[bestEdge].visits) bestEdge = e;
    }
    if (bestEdge < 0 || total <= 0.0) return false;

    // A forced pass has no policy entry, so it is not a training position.
    const bool passOnly = end - begin == 1 && isPass(tree.move[begin]);
    if (sample && !passOnly) {
        Sample out;
        out.side = root.side;
        out.board = root.board;
        out.hand = root.rem;
        out.ply = root.ply;
        out.game = 0;
        for (int e = begin; e < end; ++e) {
            out.policy[tree.move[e].cell * kNumbers + (tree.move[e].number - 1)] =
                static_cast<float>(tree.edges[e].visits / total);
        }
        g.samples.push_back(out);
    }

    int chosen = bestEdge;
    if (temperature && moveNo < o.tempMoves && o.temperature > 1e-3) {
        std::vector<double> weights;
        weights.reserve(end - begin);
        for (int e = begin; e < end; ++e) {
            weights.push_back(std::pow(static_cast<double>(tree.edges[e].visits),
                                       1.0 / std::max(0.05, o.temperature)));
        }
        std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
        chosen = begin + static_cast<int>(dist(g.rng));
    }
    const int child = tree.edges[chosen].child;
    Node next;
    if (child >= 0) {
        next = tree.nodes[child];
    } else {
        next = tree.nodes[0];
        applyMove(next, tree.move[chosen], o.ruleset);
    }
    tree.reset(next);
    g.net = g.modelFor[next.side];
    return true;
}

Node nodeFromPosition(const Position& position) {
    Node node;
    node.rem.fill(0);
    for (int side = 0; side < 2; ++side) {
        for (int number = 1; number <= kNumbers; ++number) {
            node.rem[side * 9 + (number - 1)] =
                static_cast<std::uint8_t>(position.remaining(side == 0 ? Player::Red : Player::Black,
                                                             number));
        }
    }
    for (int cell = 0; cell < kCells; ++cell) {
        node.board[cell] = static_cast<std::int8_t>(position.stoneAt(cell));
    }
    node.raw = static_cast<std::int16_t>(position.rawScore());
    node.ply = static_cast<std::uint8_t>(position.ply());
    node.side = position.sideToMove() == Player::Red ? 0 : 1;
    node.stones = static_cast<std::uint8_t>(position.stones());
    node.passes = static_cast<std::uint8_t>(position.passes());
    return node;
}

void seedOpenings(Search& g, const Options& o, std::uint64_t seed) {
    Node node = makeStart();
    std::mt19937_64 opener(seed);
    std::vector<Move> moves;
    for (int i = 0; i < o.opening; ++i) {
        legalMoves(node, moves, o.ruleset);
        if (moves.empty()) break;
        applyMove(node, moves[opener() % moves.size()], o.ruleset);
    }
    g.tree.reset(node);
    g.net = g.modelFor[node.side];
    g.rng.seed(seed + 104729ULL);
}

// ----------------------------------------------------------------- random ---

// Uniform-random play.  The policy target is uniform over the legal moves, so
// it teaches the network nothing about policy -- that is deliberate.  What it
// gives is the value head: a first estimate of P(Red wins) that is good enough
// for the one-ply scan to rank moves, which is how the first useful policy
// target is bootstrapped without any teacher.
int runRandom(const Options& o) {
    const auto started = std::chrono::steady_clock::now();
    std::vector<Sample> all;
    int redWins = 0;
    int blackWins = 0;
    int draws = 0;
    std::vector<Move> moves;

    for (int start = 0; start < o.games; start += o.batchGames) {
        const int count = std::min(o.batchGames, o.games - start);
        std::vector<Node> states(count);
        std::vector<std::vector<Sample>> samples(count);
        std::vector<std::mt19937_64> rngs(count);
        for (int g = 0; g < count; ++g) {
            states[g] = makeStart();
            rngs[g] = std::mt19937_64(o.seed + 104729ULL * static_cast<std::uint64_t>(start + g));
        }
        for (int moveNo = 0; moveNo < kMaxPlies + 4; ++moveNo) {
            for (int g = 0; g < count; ++g) {
                if (terminalState(states[g])) continue;
                legalMoves(states[g], moves, o.ruleset);
                if (moves.empty()) {
                    passTurn(states[g]);
                    continue;
                }
                Sample sample;
                sample.side = states[g].side;
                sample.board = states[g].board;
                sample.hand = states[g].rem;
                sample.ply = states[g].ply;
                sample.game = static_cast<std::uint32_t>(start + g);
                const float uniform = 1.0f / static_cast<float>(moves.size());
                for (const Move& mv : moves) {
                    sample.policy[mv.cell * kNumbers + (mv.number - 1)] = uniform;
                }
                samples[g].push_back(sample);
                applyMove(states[g], moves[rngs[g]() % moves.size()], o.ruleset);
            }
        }
        for (int g = 0; g < count; ++g) {
            const Node& final = states[g];
            const double total = static_cast<double>(final.raw) + o.komi;
            const double red = total > 0.0 ? 1.0 : (total < 0.0 ? 0.0 : 0.5);
            if (red > 0.5) ++redWins;
            else if (red < 0.5) ++blackWins;
            else ++draws;
            for (Sample& sample : samples[g]) {
                sample.value = static_cast<float>(sample.side == 0 ? red : 1.0 - red);
                const int rawScore = static_cast<int>(final.raw);
                sample.finalScore = clampScoreByte(sample.side == 0 ? rawScore : -rawScore);
                all.push_back(sample);
            }
        }
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const int done = std::min(start + count, o.games);
        std::printf("batch games=%d/%d positions=%zu %.1fs\n", done, o.games, all.size(), elapsed);
        std::fflush(stdout);
    }
    if (!writeShard(o.out, all, o.ruleset)) {
        std::cerr << "cannot write " << o.out << "\n";
        return 1;
    }
    std::printf("wrote %zu samples to %s\n", all.size(), o.out.c_str());
    std::printf("games=%d seconds=%.1f games/s=%.3f\n", o.games,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
                o.games /
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                        .count());
    std::printf("red=%d black=%d draw=%d\n", redWins, blackWins, draws);
    reportTargets(all, o.ruleset);
    return 0;
}

// -------------------------------------------------------------- self-play ---

int runSelfplay(Evaluator& evaluator, const Options& o) {
    const auto started = std::chrono::steady_clock::now();
    std::vector<Sample> all;
    int redWins = 0;
    int blackWins = 0;
    int draws = 0;
    Scratch scratch;
    TickTiming timing;

    for (int start = 0; start < o.games; start += o.batchGames) {
        const int count = std::min(o.batchGames, o.games - start);
        std::vector<Search> games(count);
        std::vector<int> active;
        for (int g = 0; g < count; ++g) {
            seedOpenings(games[g], o,
                         o.seed + 7919ULL * static_cast<std::uint64_t>(start + g));
        }
        for (int moveNo = 0; moveNo < kMaxPlies + 4; ++moveNo) {
            active.clear();
            for (int g = 0; g < count; ++g) {
                if (!terminalState(games[g].tree.nodes[0])) active.push_back(g);
            }
            if (active.empty()) break;
            for (int visit = 0; visit < o.visits; ++visit) {
                tick(evaluator, games, active, scratch, o, o.stats ? &timing : nullptr);
            }
            for (int g : active) {
                if (!advance(games[g], o, moveNo, true, true)) {
                    games[g].tree.terminal[0] = 1;
                }
            }
        }
        for (int g = 0; g < count; ++g) {
            const Node& final = games[g].tree.nodes[0];
            const double total = static_cast<double>(final.raw) + o.komi;
            const double red = total > 0.0 ? 1.0 : (total < 0.0 ? 0.0 : 0.5);
            if (terminalState(final)) {
                if (red > 0.5) ++redWins;
                else if (red < 0.5) ++blackWins;
                else ++draws;
            } else {
                ++draws;
            }
            for (Sample& sample : games[g].samples) {
                sample.value = static_cast<float>(sample.side == 0 ? red : 1.0 - red);
                sample.game = static_cast<std::uint32_t>(start + g);
                const int rawScore = static_cast<int>(final.raw);
                sample.finalScore = clampScoreByte(sample.side == 0 ? rawScore : -rawScore);
                all.push_back(sample);
            }
        }
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const int done = std::min(start + count, o.games);
        std::printf("batch games=%d/%d positions=%zu %.1fs %.3f games/s\n", done, o.games,
                    all.size(), elapsed, done / elapsed);
        std::fflush(stdout);
    }

    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (!writeShard(o.out, all, o.ruleset)) {
        std::cerr << "cannot write " << o.out << "\n";
        return 1;
    }
    std::printf("wrote %zu samples to %s\n", all.size(), o.out.c_str());
    std::printf(
        "games=%d batch=%d visits=%d seconds=%.1f games/s=%.3f positions/s=%.1f sims/s=%.0f\n",
        o.games, o.batchGames, o.visits, seconds, o.games / seconds, all.size() / seconds,
        static_cast<double>(all.size()) * o.visits / seconds);
    std::printf("red=%d black=%d draw=%d\n", redWins, blackWins, draws);
    if (timing.ticks > 0) {
        const double per = 1000.0 / static_cast<double>(timing.ticks);
        std::printf(
            "TICK_MS descend=%.2f eval=%.2f expand=%.2f (batch=%d threads=%d ticks=%lld)\n",
            timing.descend * per, timing.eval * per, timing.expand * per, o.batchGames, o.threads,
            timing.ticks);
    }
    reportTargets(all, o.ruleset);
    return 0;
}

// ------------------------------------------------------- one-ply full width ---

// Policy from a full-width one-ply value scan.  Every legal move is evaluated,
// so the support of the target is the whole legal set and the supervision is a
// deterministic function of the position -- something a policy head can learn
// before a search prior exists.
int runOneply(Evaluator& evaluator, const Options& o) {
    const auto started = std::chrono::steady_clock::now();
    std::vector<Sample> all;
    int redWins = 0;
    int blackWins = 0;
    int draws = 0;

    std::vector<Move> moves;
    std::vector<Node> children;
    std::vector<Move> ownerMove;
    std::vector<std::size_t> offsets;

    for (int start = 0; start < o.games; start += o.batchGames) {
        const int count = std::min(o.batchGames, o.games - start);
        std::vector<Node> states(count);
        std::vector<std::vector<Sample>> samples(count);
        std::vector<std::mt19937_64> rngs(count);
        for (int g = 0; g < count; ++g) {
            std::mt19937_64 opener(o.seed + 7919ULL * static_cast<std::uint64_t>(start + g));
            Node node = makeStart();
            for (int i = 0; i < o.opening; ++i) {
                legalMoves(node, moves, o.ruleset);
                if (moves.empty()) break;
                applyMove(node, moves[opener() % moves.size()], o.ruleset);
            }
            states[g] = node;
            rngs[g] = std::mt19937_64(o.seed + 104729ULL * static_cast<std::uint64_t>(start + g));
        }

        for (int moveNo = 0; moveNo < kMaxPlies + 4; ++moveNo) {
            children.clear();
            ownerMove.clear();
            offsets.assign(count + 1, 0);
            for (int g = 0; g < count; ++g) {
                offsets[g] = children.size();
                if (terminalState(states[g])) continue;
                legalMoves(states[g], moves, o.ruleset);
                for (const Move& mv : moves) {
                    Node child = states[g];
                    play(child, mv, o.ruleset);
                    children.push_back(child);
                    ownerMove.push_back(mv);
                }
            }
            offsets[count] = children.size();
            bool any = false;
            for (int g = 0; g < count; ++g) {
                if (offsets[g + 1] > offsets[g]) any = true;
            }
            if (!any) break;

            const std::size_t slice = 4096;
            std::vector<float> logits(slice * kNnPolicy);
            std::vector<float> values(slice);
            std::vector<double> q(children.size(), 0.0);
            for (std::size_t base = 0; base < children.size(); base += slice) {
                const int m = static_cast<int>(std::min(slice, children.size() - base));
                evaluator.eval(0, children.data() + base, m, logits.data(), values.data());
                for (int k = 0; k < m; ++k) {
                    // The child is the opponent to move, so its win probability is
                    // the mover's loss probability.
                    q[base + k] = 1.0 - static_cast<double>(values[k]);
                }
            }

            for (int g = 0; g < count; ++g) {
                if (terminalState(states[g])) continue;
                const std::size_t begin = offsets[g];
                const std::size_t end = offsets[g + 1];
                if (begin == end) {
                    passTurn(states[g]); // blocked: forced pass, no training sample
                    continue;
                }
                double best = -1e30;
                double sum = 0.0;
                for (std::size_t k = begin; k < end; ++k) {
                    best = std::max(best, q[k]);
                    sum += q[k];
                }
                // Temperature is scaled to the spread of this position's Q values.
                // A fixed temperature makes the target either flat (weak value net,
                // unlearnable) or a one-hot spike (noisy value net, teaches noise);
                // scaling by the standard deviation keeps the target's entropy tied
                // to how much the evaluation actually distinguishes the moves.
                double spread = 0.0;
                if (end - begin > 1) {
                    const double mean = sum / static_cast<double>(end - begin);
                    double var = 0.0;
                    for (std::size_t k = begin; k < end; ++k) {
                        const double d = q[k] - mean;
                        var += d * d;
                    }
                    spread = std::sqrt(var / static_cast<double>(end - begin));
                }
                const double temperature =
                    o.oneplyTempScale > 0.0
                        ? std::max(1e-4, o.oneplyTempScale * spread)
                        : std::max(1e-4, o.oneplyTemperature);
                std::vector<double> p(end - begin);
                double total = 0.0;
                for (std::size_t k = begin; k < end; ++k) {
                    p[k - begin] = std::exp((q[k] - best) / temperature);
                    total += p[k - begin];
                }
                Sample sample;
                sample.side = states[g].side;
                sample.board = states[g].board;
                sample.hand = states[g].rem;
                sample.ply = states[g].ply;
                sample.game = static_cast<std::uint32_t>(start + g);
                for (std::size_t k = begin; k < end; ++k) {
                    const Move& mv = ownerMove[k];
                    sample.policy[mv.cell * kNumbers + (mv.number - 1)] =
                        static_cast<float>(p[k - begin] / total);
                }
                samples[g].push_back(sample);

                std::size_t chosen = begin;
                if (moveNo < o.tempMoves) {
                    std::discrete_distribution<std::size_t> dist(p.begin(), p.end());
                    chosen = begin + dist(rngs[g]);
                } else {
                    for (std::size_t k = begin; k < end; ++k) {
                        if (p[k - begin] > p[chosen - begin]) chosen = k;
                    }
                }
                states[g] = children[chosen];
            }
        }

        for (int g = 0; g < count; ++g) {
            const Node& final = states[g];
            const double total = static_cast<double>(final.raw) + o.komi;
            const double red = total > 0.0 ? 1.0 : (total < 0.0 ? 0.0 : 0.5);
            if (terminalState(final)) {
                if (red > 0.5) ++redWins;
                else if (red < 0.5) ++blackWins;
                else ++draws;
            } else {
                ++draws;
            }
            for (Sample& sample : samples[g]) {
                sample.value = static_cast<float>(sample.side == 0 ? red : 1.0 - red);
                const int rawScore = static_cast<int>(final.raw);
                sample.finalScore = clampScoreByte(sample.side == 0 ? rawScore : -rawScore);
                all.push_back(sample);
            }
        }
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const int done = std::min(start + count, o.games);
        std::printf("batch games=%d/%d positions=%zu %.1fs %.3f games/s\n", done, o.games,
                    all.size(), elapsed, done / elapsed);
        std::fflush(stdout);
    }

    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (!writeShard(o.out, all, o.ruleset)) {
        std::cerr << "cannot write " << o.out << "\n";
        return 1;
    }
    std::printf("wrote %zu samples to %s\n", all.size(), o.out.c_str());
    std::printf("games=%d batch=%d seconds=%.1f games/s=%.3f positions/s=%.1f evals/s=%.0f\n",
                o.games, o.batchGames, seconds, o.games / seconds, all.size() / seconds,
                static_cast<double>(all.size()) * kMoves / seconds);
    std::printf("red=%d black=%d draw=%d\n", redWins, blackWins, draws);
    reportTargets(all, o.ruleset);
    return 0;
}

// ------------------------------------------------------------------ match ---

int runMatch(Evaluator& evaluator, const Options& o) {
    const auto started = std::chrono::steady_clock::now();
    Scratch scratch;
    int aWins = 0;
    int bWins = 0;
    int draws = 0;
    std::vector<int> results(o.games, -1); // 1 = A won, 0 = draw, -1 = A lost

    for (int start = 0; start < o.games; start += o.batchGames) {
        const int count = std::min(o.batchGames, o.games - start);
        std::vector<Search> games(count);
        std::vector<int> aIsRed(count, 0);
        for (int g = 0; g < count; ++g) {
            aIsRed[g] = ((start + g) % 2 == 0) ? 1 : 0; // alternate colours
            games[g].modelFor[0] = aIsRed[g] ? 0 : 1;
            games[g].modelFor[1] = aIsRed[g] ? 1 : 0;
            seedOpenings(games[g], o, o.seed + 7919ULL * static_cast<std::uint64_t>(start + g));
        }
        std::vector<int> active;
        for (int moveNo = 0; moveNo < kMaxPlies + 4; ++moveNo) {
            active.clear();
            for (int g = 0; g < count; ++g) {
                if (!terminalState(games[g].tree.nodes[0])) active.push_back(g);
            }
            if (active.empty()) break;
            for (int visit = 0; visit < o.visits; ++visit) {
                tick(evaluator, games, active, scratch, o);
            }
            for (int g : active) {
                if (!advance(games[g], o, moveNo, false, false)) {
                    games[g].tree.terminal[0] = 1;
                }
            }
        }
        for (int g = 0; g < count; ++g) {
            const Node& final = games[g].tree.nodes[0];
            const double total = static_cast<double>(final.raw) + o.komi;
            const int redResult = total > 0.0 ? 1 : (total < 0.0 ? -1 : 0);
            int a;
            if (!terminalState(final) || redResult == 0) a = 0;
            else a = (redResult > 0) == (aIsRed[g] == 1) ? 1 : -1;
            results[start + g] = a;
            if (a > 0) ++aWins;
            else if (a < 0) ++bWins;
            else ++draws;
        }
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const int done = std::min(start + count, o.games);
        std::printf("match %d/%d A=%d B=%d D=%d %.1fs %.2f games/s\n", done, o.games, aWins,
                    bWins, draws, elapsed, done / elapsed);
        std::fflush(stdout);
    }

    const double games = static_cast<double>(o.games);
    const double score = (static_cast<double>(aWins) + 0.5 * static_cast<double>(draws)) / games;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::printf("MATCH_RESULT {\"a\": \"%s\", \"b\": \"%s\", \"games\": %d, \"a_wins\": %d, "
                "\"b_wins\": %d, \"draws\": %d, \"score_a\": %.4f, \"komi\": %.2f, "
                "\"visits\": %d}\n",
                o.nameA.c_str(), o.nameB.c_str(), o.games, aWins, bWins, draws, score, o.komi,
                o.visits);
    std::printf("seconds=%.1f games/s=%.3f sims/s=%.0f\n", seconds, games / seconds,
                games * kMaxPlies * o.visits / seconds);
    if (!o.jsonOut.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(
            std::filesystem::path(o.jsonOut).parent_path(), ec);
        std::FILE* f = fopenUtf8(o.jsonOut, "w");
        if (f) {
            std::fprintf(f,
                         "{\"a\": \"%s\", \"b\": \"%s\", \"games\": %d, \"a_wins\": %d, "
                         "\"b_wins\": %d, \"draws\": %d, \"score_a\": %.6f, \"komi\": %.2f, "
                         "\"visits\": %d, \"seconds\": %.1f}\n",
                         o.nameA.c_str(), o.nameB.c_str(), o.games, aWins, bWins, draws, score,
                         o.komi, o.visits, seconds);
            std::fclose(f);
        }
    }
    if (!o.csvOut.empty()) {
        std::FILE* f = fopenUtf8(o.csvOut, "w");
        if (f) {
            std::fprintf(f, "game,a_is_red,result\n");
            for (int g = 0; g < o.games; ++g) {
                std::fprintf(f, "%d,%d,%d\n", g, (g % 2 == 0) ? 1 : 0, results[g]);
            }
            std::fclose(f);
        }
    }
    std::fflush(stdout);
    return 0;
}


// ---------------------------------------------------------------- serve ---

// A line protocol so a GPU model can sit in the same tournament as the CPU
// engines (training/arena.py).  One game per process:
//
//   ruleset                      -> = classic|blocking
//   set_visits N                 -> = N
//   play red|black a1 5|pass     -> =
//   genmove red|black            -> = a1 5   (or = pass)
//   quit
int runServe(Evaluator& evaluator, const Options& o) {
    Position position;
    position.setRuleset(o.ruleset);
    int visits = o.visits;
    Scratch scratch;
    std::printf("ready ruleset=%s visits=%d\n", rulesetName(o.ruleset), visits);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string command;
        in >> command;
        if (command.empty()) continue;
        if (command == "quit") {
            std::printf("=\n");
            std::fflush(stdout);
            return 0;
        }
        if (command == "ruleset") {
            std::printf("= %s\n", rulesetName(o.ruleset));
            std::fflush(stdout);
            continue;
        }
        if (command == "set_visits") {
            int value = 0;
            in >> value;
            if (value > 0) visits = value;
            std::printf("= %d\n", visits);
            std::fflush(stdout);
            continue;
        }
        if (command == "play") {
            std::string color, vertex;
            int number = 0;
            in >> color >> vertex;
            if (vertex == "pass") {
                if (!position.pass()) {
                    std::printf("? pass not forced\n");
                    std::fflush(stdout);
                    continue;
                }
                std::printf("=\n");
                std::fflush(stdout);
                continue;
            }
            in >> number;
            int cell = 0;
            if (!parseSquare(vertex, cell) || number < 1 || number > 9 ||
                !position.play(Move(cell, number))) {
                std::printf("? illegal move\n");
                std::fflush(stdout);
                continue;
            }
            std::printf("=\n");
            std::fflush(stdout);
            continue;
        }
        if (command == "genmove") {
            std::string color;
            in >> color;
            if (position.gameOver()) {
                std::printf("= pass\n");
                std::fflush(stdout);
                continue;
            }
            const auto moves = position.searchMoves();
            if (moves.empty()) {
                std::printf("= pass\n");
                std::fflush(stdout);
                continue;
            }
            if (moves.size() == 1 && isPass(moves[0])) {
                std::printf("= pass\n");
                std::fflush(stdout);
                continue;
            }
            Search game;
            game.tree.reset(nodeFromPosition(position));
            game.rng.seed(static_cast<std::uint64_t>(position.key()));
            game.net = 0;
            game.modelFor[0] = 0;
            game.modelFor[1] = 0;
            std::vector<Search> games;
            games.push_back(std::move(game));
            std::vector<int> active{0};
            for (int visit = 0; visit < visits; ++visit) {
                tick(evaluator, games, active, scratch, o);
            }
            game = std::move(games[0]);
            // Best root child by visits.
            const int begin = game.tree.firstEdge[0];
            const int end = begin + game.tree.edgeCount[0];
            int best = -1;
            float bestVisits = -1.0f;
            for (int e = begin; e < end; ++e) {
                if (game.tree.edges[e].visits > bestVisits) {
                    bestVisits = game.tree.edges[e].visits;
                    best = e;
                }
            }
            if (best < 0) {
                std::printf("= pass\n");
                std::fflush(stdout);
                continue;
            }
            const Move move = game.tree.move[best];
            if (isPass(move)) {
                std::printf("= pass\n");
            } else {
                std::printf("= %s %d\n", squareName(move.cell).c_str(),
                            static_cast<int>(move.number));
            }
            std::fflush(stdout);
            continue;
        }
        std::printf("? unknown command\n");
        std::fflush(stdout);
    }
    return 0;
}

// ------------------------------------------------------------------ bench ---

int runBench(Evaluator& evaluator, const Options& o) {
    std::vector<Node> pool;
    std::mt19937_64 rng(o.seed);
    for (int g = 0; g < 64; ++g) {
        Node node = makeStart();
        std::vector<Move> moves;
        const int plies = static_cast<int>(rng() % 50);
        for (int i = 0; i < plies; ++i) {
            legalMoves(node, moves, o.ruleset);
            if (moves.empty()) {
                passTurn(node);
                continue;
            }
            applyMove(node, moves[rng() % moves.size()], o.ruleset);
        }
        pool.push_back(node);
    }
    std::printf("device=%s planes=%d channels=%d blocks=%d\n", evaluator.device().c_str(),
                evaluator.planes(), evaluator.channels(), evaluator.blocks());
    std::printf("%10s %12s %12s\n", "batch", "ms/call", "evals/s");
    for (int batch : {1, 8, 32, 64, 128, 256, 512, 1024}) {
        std::vector<Node> nodes;
        for (int i = 0; i < batch; ++i) nodes.push_back(pool[i % pool.size()]);
        std::vector<float> logits(static_cast<std::size_t>(batch) * kNnPolicy);
        std::vector<float> values(batch);
        for (int warm = 0; warm < 3; ++warm) {
            evaluator.eval(0, nodes.data(), batch, logits.data(), values.data());
        }
        const int reps = batch >= 256 ? 5 : 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            evaluator.eval(0, nodes.data(), batch, logits.data(), values.data());
        }
        const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                              .count() * 1000.0 / reps;
        std::printf("%10d %12.3f %12.0f\n", batch, ms, batch * 1000.0 / ms);
        std::fflush(stdout);
    }
    return 0;
}

// ------------------------------------------------------------------ check ---

int runCheck(Evaluator& evaluator, const Options& o) {
    std::mt19937_64 rng(7);
    std::vector<Node> nodes;
    std::vector<Position> positions;
    for (int attempt = 0; attempt < 64; ++attempt) {
        Node node = makeStart();
        Position position;
        position.setRuleset(o.ruleset);
        std::vector<Move> moves;
        const int plies = static_cast<int>(rng() % 40);
        for (int i = 0; i < plies; ++i) {
            legalMoves(node, moves, o.ruleset);
            if (moves.empty()) break;
            const Move mv = moves[rng() % moves.size()];
            if (!applyMove(node, mv, o.ruleset) || !position.play(mv)) {
                std::printf("rule mismatch at ply %d (move %d,%d)\n", i, mv.cell, mv.number);
                return 1;
            }
            for (int c = 0; c < kCells; ++c) {
                if (node.board[c] != position.stoneAt(c)) {
                    std::printf("board differs at ply %d cell %d\n", i + 1, c);
                    return 1;
                }
            }
            if (node.raw != position.rawScore()) {
                std::printf("score differs at ply %d: node=%d position=%d\n", i + 1,
                            static_cast<int>(node.raw), position.rawScore());
                return 1;
            }
        }
        bool same = node.ply == position.ply() && node.raw == position.rawScore();
        for (int cell = 0; cell < kCells && same; ++cell) {
            same = node.board[cell] == position.stoneAt(cell);
        }
        if (!same) {
            std::printf("board/score mismatch after %d plies\n", plies);
            return 1;
        }
        nodes.push_back(node);
        positions.push_back(position);
    }

    std::vector<float> logits(nodes.size() * kNnPolicy);
    std::vector<float> values(nodes.size());
    std::vector<float> scores(nodes.size(), 0.0f);
    const bool scoreHead = evaluator.cpu(0).hasScoreHead();
    evaluator.eval(0, nodes.data(), static_cast<int>(nodes.size()), logits.data(), values.data(),
                   scoreHead ? scores.data() : nullptr);
    double maxLogit = 0.0;
    double maxValue = 0.0;
    double maxScore = 0.0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const NnOutput reference = evaluator.cpu(0).evaluate(positions[i]);
        for (int k = 0; k < kNnPolicy; ++k) {
            maxLogit = std::max(maxLogit, std::fabs(static_cast<double>(
                                              logits[i * kNnPolicy + k] - reference.policy[k])));
        }
        maxValue = std::max(maxValue, std::fabs(static_cast<double>(values[i] - reference.value)));
        if (scoreHead) {
            maxScore = std::max(maxScore,
                                std::fabs(static_cast<double>(scores[i] - reference.expectedScore)));
        }
    }
    std::printf("rules parity OK (%zu positions); cuda vs cpu: logits |max|=%.6f value |max|=%.6f",
                nodes.size(), maxLogit, maxValue);
    if (scoreHead) std::printf(" score |max|=%.6f", maxScore);
    std::printf("%s\n", scoreHead ? "" : " (no score head)");
    if (maxLogit >= 5e-3 || maxValue >= 5e-3 || (scoreHead && maxScore >= 1e-2)) {
        // Where do the logits differ?  A cluster at the board edge means the
        // zero padding of the convolution is wrong; a uniform spread means the
        // GEMM indexes are.
        std::vector<std::pair<double, int>> diffs;
        const NnOutput reference = evaluator.cpu(0).evaluate(positions[0]);
        for (int k = 0; k < kNnPolicy; ++k) {
            diffs.push_back({std::fabs(static_cast<double>(logits[k] - reference.policy[k])), k});
        }
        std::partial_sort(diffs.begin(), diffs.begin() + 8, diffs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int i = 0; i < 8; ++i) {
            const int k = diffs[i].second;
            std::printf("  diff %.4f at cell=(%d,%d) number=%d\n", diffs[i].first, k / 9 % 9,
                        (k / 9) / 9, k % 9 + 1);
        }
    }

    if (!o.dumpFeatures.empty()) {
        const int planes = evaluator.planes();
        std::FILE* f = fopenUtf8(o.dumpFeatures, "wb");
        if (!f) {
            std::cerr << "cannot write " << o.dumpFeatures << "\n";
            return 1;
        }
        const std::uint32_t count = static_cast<std::uint32_t>(nodes.size());
        const std::uint32_t planeCount = static_cast<std::uint32_t>(planes);
        std::fwrite(&count, sizeof(count), 1, f);
        std::fwrite(&planeCount, sizeof(planeCount), 1, f);
        std::vector<float> features(static_cast<std::size_t>(planes) * kCells);
        for (const Node& node : nodes) {
            const auto ply = static_cast<std::uint8_t>(node.ply);
            std::fwrite(&ply, sizeof(ply), 1, f);
            std::fwrite(node.board.data(), node.board.size(), 1, f);
            std::fwrite(node.rem.data(), node.rem.size(), 1, f);
            evaluator.cpu(0).buildFeaturePlanes(node.board, node.ply, features.data());
            std::fwrite(features.data(), sizeof(float), features.size(), f);
        }
        std::fclose(f);
        std::printf("wrote feature dump %s (%u positions, %u planes)\n", o.dumpFeatures.c_str(),
                    count, planeCount);
    }
    // 5e-3 leaves room for TF32 tensor-core GEMMs on Ampere and newer, where the
    // library is allowed to round mantissas; a real indexing bug is off by ~1.
    return (maxLogit < 5e-3 && maxValue < 5e-3) ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    shuyi::Utf8Args utf8Args(argc, argv);
    argv = utf8Args.pointers();
    Options options;
    if (!parse(argc, argv, options)) return 1;
#ifdef _OPENMP
    // Descend / expand / back up run on the CPU and used to be the reason the
    // GPU sat idle two thirds of the time.
    if (options.threads > 0) omp_set_num_threads(options.threads);
#endif

    if (options.mode == "random") return runRandom(options);

    std::string reason;
    if (!CudaPolicyValueNet::available(&reason)) {
        std::cerr << "cuda unavailable: " << reason << "\n";
        return 1;
    }
    std::vector<std::string> models{options.model};
    if (options.mode == "match") models.push_back(options.modelB);

    Evaluator evaluator;
    std::string error;
    if (!evaluator.init(models, options.chunk, &error)) {
        std::cerr << "cannot load model: " << error << "\n";
        return 1;
    }
    // A model trained for one rule set has the wrong value semantics for the
    // other, so mixing them is refused rather than silently trusted.
    for (int i = 0; i < evaluator.count(); ++i) {
        if (evaluator.rulesetOf(i) != options.ruleset) {
            std::cerr << "model " << evaluator.name(i) << " was trained for the "
                      << rulesetName(evaluator.rulesetOf(i)) << " rule set but --ruleset is "
                      << rulesetName(options.ruleset) << "\n";
            return 1;
        }
    }
    std::printf("device=%s planes=%d channels=%d blocks=%d mode=%s ruleset=%s\n",
                evaluator.device().c_str(), evaluator.planes(), evaluator.channels(),
                evaluator.blocks(), options.mode.c_str(), rulesetName(options.ruleset));

    if (options.check) return runCheck(evaluator, options);
    if (options.mode == "serve") return runServe(evaluator, options);
    if (options.mode == "bench") return runBench(evaluator, options);
    if (options.mode == "match") return runMatch(evaluator, options);
    if (options.mode == "oneply") return runOneply(evaluator, options);
    if (options.mode == "selfplay") return runSelfplay(evaluator, options);
    std::cerr << "unknown mode: " << options.mode << "\n";
    return 1;
}
