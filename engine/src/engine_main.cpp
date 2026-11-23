// shuyi_engine: headless GTP-like engine process.  It deliberately has no
// Win32/UI dependency so the UI can be replaced or run headless.
#include "shuyi/engine.h"

#ifdef SHUYI_CUDA_ENGINE
#include "shuyi/nn_cuda.h"
#endif
#include "shuyi/position.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace shuyi;

namespace {

#ifdef SHUYI_CUDA_ENGINE
// Wraps the batched CUDA forward for the single position an MCTS simulation
// needs.  The CPU model is still loaded: it owns the feature layout, the move
// index mapping and the rule set, and it is what the fallback uses.
class CudaEvaluator {
public:
    bool load(const std::string& path, std::string* error) {
        if (!net_.load(path, error)) return false;
        if (!gpu_.load(net_, error)) return false;
        planes_ = net_.modelInputPlanes();
        features_.assign(static_cast<std::size_t>(planes_) * kCells, 0.0f);
        globals_.assign(kResGlobalFeatures, 0.0f);
        logits_.assign(kNnPolicy, 0.0f);
        return true;
    }

    bool ready() const { return gpu_.loaded(); }
    bool hasScoreHead() const { return net_.hasScoreHead(); }
    const PolicyValueNet& model() const { return net_; }

    NnOutput evaluate(const Position& position) {
        NnOutput out;
        std::array<std::int8_t, kCells> board{};
        for (int cell = 0; cell < kCells; ++cell) {
            board[cell] = static_cast<std::int8_t>(position.stoneAt(cell));
        }
        net_.buildFeaturePlanes(board, position.ply(), features_.data());
        net_.buildGlobals(board, position.ply(), position.rawScore(), globals_.data());
        float value = 0.5f;
        float score = 0.0f;
        gpu_.evaluate(features_.data(), globals_.data(), 1, logits_.data(), &value,
                      net_.hasScoreHead() ? &score : nullptr);
        for (int i = 0; i < kNnPolicy; ++i) out.policy[i] = logits_[i];
        out.value = value;
        out.expectedScore = net_.hasScoreHead() ? score : 0.0f;
        return out;
    }

private:
    PolicyValueNet net_;
    CudaPolicyValueNet gpu_;
    int planes_ = kResInputPlanes;
    mutable std::vector<float> features_;
    mutable std::vector<float> globals_;
    mutable std::vector<float> logits_;
};
#endif



std::string trim(std::string value) {
    while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ')) {
        value.pop_back();
    }
    std::size_t start = 0;
    while (start < value.size() && value[start] == ' ') ++start;
    return value.substr(start);
}

std::string lower(std::string value) {
    for (char& c : value) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return value;
}

bool parsePlayer(const std::string& text, Player& player) {
    const std::string t = lower(text);
    if (t == "red" || t == "r") { player = Player::Red; return true; }
    if (t == "black" || t == "b") { player = Player::Black; return true; }
    return false;
}

struct CommandResult {
    bool ok = true;
    std::string text;
};

class Session {
public:
    Session() {
        options_.seed = 0x5EED1234ULL;
        makeCurrentEngine(EngineKind::Pruning);
    }

    ~Session() { stopAndJoin(); }

    CommandResult handle(const std::string& line) {
        std::istringstream input(line);
        std::string command;
        input >> command;
        command = lower(command);
        if (command.empty()) return {true, ""};

        if (command == "protocol_version") return {true, "1"};
        if (command == "name") return {true, "ShuyiEngine"};
        if (command == "version") return {true, "0.2.0"};
        if (command == "list_commands") {
            return {true,
                    "protocol_version\nname\nversion\nlist_commands\nboardsize\nclear_board\n"
                    "set_engine\nset_time\nset_a\nset_model\nset_window\nset_pvs\n"
                    "play\ngenmove\nanalyze\nstop\nquit"};
        }
        if (command == "boardsize") {
            int size = 0;
            input >> size;
            if (size != 9) return {false, "only boardsize 9 is supported"};
            return {true, ""};
        }
        if (command == "quit") {
            stopAndJoin();
            quit_ = true;
            return {true, ""};
        }
        if (command == "stop") {
            stopAndJoin();
            return {true, "stopped"};
        }
        if (command == "clear_board") {
            stopAndJoin();
            position_.reset();
            if (engine_) engine_->reset();
            return {true, ""};
        }
        if (command == "set_engine") {
            stopAndJoin();
            std::string name;
            input >> name;
            EngineKind kind;
            if (!parseEngineName(lower(name), kind)) return {false, "unknown engine: " + name};
            makeCurrentEngine(kind);
            return {true, engineName(kind)};
        }
        if (command == "set_time") {
            int ms = 0;
            input >> ms;
            if (ms < 0) return {false, "time must be >= 0"};
            options_.timeLimitMs = ms;
            return {true, ""};
        }
        if (command == "set_visits") {
            int visits = 0;
            input >> visits;
            if (visits < 0) return {false, "visits must be >= 0"};
            // Only the default budget changes: rebuilding the engine here would
            // throw the search tree away, which is exactly what a front end needs
            // to keep when it extends an analysis.
            options_.visits = visits;
            return {true, visits > 0 ? std::to_string(visits) : std::string("off")};
        }
        if (command == "set_a") {
            double value = 0.0;
            input >> value;
            if (!input) return {false, "bad A value"};
            stopAndJoin();
            options_.aValue = value;
            makeCurrentEngine(kind_);
            return {true, ""};
        }
        if (command == "capabilities") {
            // The UI probes candidates with this before choosing which engine
            // process to drive: the binary it spawns must be the fast one, and
            // only the engine knows whether it was built with CUDA.
#ifdef SHUYI_CUDA_ENGINE
            const bool cuda = true;
#else
            const bool cuda = false;
#endif
            return {true, std::string("engine=shuyi_engine cuda=") + (cuda ? "1" : "0") +
                              " proto=1 visits=1 rulesets=classic,blocking"};
        }
        if (command == "set_backend") {
            std::string name;
            input >> name;
            if (name != "cpu" && name != "cuda" && name != "auto") {
                return {false, "backend must be cpu, cuda or auto"};
            }
            backend_ = name;
            return {true, backend_};
        }
        if (command == "set_model") {
            stopAndJoin();
            std::string path;
            std::getline(input, path);
            options_.nnModelPath = trim(path);
            attachEvaluator();
            makeCurrentEngine(kind_);
            return {true, options_.nnModelPath};
        }
        if (command == "set_window") {
            int halfWidth = 0;
            input >> halfWidth;
            if (halfWidth < 0 || halfWidth > 1000) return {false, "window must be 0..1000"};
            stopAndJoin();
            options_.aspirationWindow = halfWidth;
            makeCurrentEngine(kind_);
            return {true, std::to_string(halfWidth)};
        }
        if (command == "set_pvs") {
            std::string value;
            input >> value;
            value = lower(value);
            bool enabled = false;
            if (value == "on" || value == "1") enabled = true;
            else if (value == "off" || value == "0") enabled = false;
            else return {false, "usage: set_pvs on|off"};
            stopAndJoin();
            options_.usePvs = enabled;
            makeCurrentEngine(kind_);
            return {true, options_.usePvs ? "on" : "off"};
        }
        if (command == "play") {
            std::string color, vertex;
            int number = 0;
            input >> color >> vertex;
            Player player;
            int cell = 0;
            if (!parsePlayer(color, player)) return {false, "usage: play red|black <a1..i9> <1..9>|pass"};
            if (player != position_.sideToMove()) return {false, "wrong side to move"};
            if (vertex == "pass") {
                stopAndJoin();
                if (!position_.pass()) return {false, "pass is only forced when no placement is legal"};
                if (engine_) engine_->onPlay(position_, Move(kPassCell, kPassNumber));
                return {true, ""};
            }
            input >> number;
            if (!parseSquare(vertex, cell) || number < 1 || number > 9) {
                return {false, "usage: play red|black <a1..i9> <1..9>|pass"};
            }
            stopAndJoin();
            if (!position_.play(Move(cell, number))) return {false, "illegal move"};
            if (engine_) engine_->onPlay(position_, Move(cell, number));
            return {true, ""};
        }
        if (command == "ruleset") {
            std::string value;
            if (!(input >> value)) return {true, rulesetName(position_.ruleset())};
            Ruleset parsed = position_.ruleset();
            if (!parseRuleset(value, parsed)) return {false, "usage: ruleset [classic|blocking]"};
            stopAndJoin();
            if (position_.stones() != 0) return {false, "ruleset can only change before the first move"};
            position_.setRuleset(parsed);
            makeCurrentEngine(kind_);
            return {true, rulesetName(position_.ruleset())};
        }
        if (command == "genmove") {
            std::string color;
            input >> color;
            Player player;
            if (!parsePlayer(color, player)) return {false, "usage: genmove red|black [visits]"};
            if (player != position_.sideToMove()) return {false, "wrong side to move"};
            if (position_.gameOver()) return {false, "game is over"};
            // Optional per-command budget.  A front end uses it to extend a
            // finished analysis without touching the default or the tree.
            int visits = options_.visits;
            if (!(input >> visits)) visits = options_.visits;
            if (visits < 0) return {false, "visits must be >= 0"};
            startSearch(false, visits);
            return {true, ""};
        }
        if (command == "analyze") {
            std::string color;
            input >> color;
            Player player;
            if (!parsePlayer(color, player)) return {false, "usage: analyze red|black [visits]"};
            if (player != position_.sideToMove()) return {false, "wrong side to move"};
            if (position_.gameOver()) return {false, "game is over"};
            int visits = options_.visits;
            if (!(input >> visits)) visits = options_.visits;
            if (visits < 0) return {false, "visits must be >= 0"};
            startSearch(true, visits);
            return {true, ""};
        }
        return {false, "unknown command: " + command};
    }

    bool finished() const { return quit_; }
    void respond(bool ok, const std::string& text) {
        std::lock_guard<std::mutex> lock(outputMutex_);
        if (ok) {
            if (!text.empty()) std::cout << "= " << text << '\n';
            else std::cout << "=\n";
        } else {
            std::cout << "? " << text << '\n';
        }
        std::cout.flush();
    }

private:
    // CPU by default.  "cuda"/"auto" installs a GPU evaluator when this build has
    // one and the model loads; on any failure it silently stays on the CPU.
    void attachEvaluator() {
        if (backend_ == "cpu" || options_.nnModelPath.empty()) return;
#ifdef SHUYI_CUDA_ENGINE
        auto evaluator = std::make_shared<CudaEvaluator>();
        std::string error;
        if (!evaluator->load(options_.nnModelPath, &error)) {
            std::cerr << "set_backend: CUDA unavailable (" << error << "), using the CPU\n";
            return;
        }
        auto override = std::make_shared<NnOverride>();
        override->evaluate = [evaluator](const Position& position) {
            return evaluator->evaluate(position);
        };
        override->hasScoreHead = [evaluator] { return evaluator->hasScoreHead(); };
        options_.nnOverride = override;
#else
        if (backend_ == "cuda") {
            std::cerr << "set_backend: this build has no CUDA backend\n";
        }
#endif
    }

    void makeCurrentEngine(EngineKind kind) {
        kind_ = kind;
        options_.maxDepth = 12;
        options_.ttSizeMb = 16;
        options_.mctsMaxChildren = 48;
        engine_ = makeEngine(kind_, options_);
    }

    void stopAndJoin() {
        stop_.store(true, std::memory_order_relaxed);
        if (worker_.joinable()) worker_.join();
        stop_.store(false, std::memory_order_relaxed);
        running_.store(false, std::memory_order_relaxed);
    }

    static std::string formatDouble(double value) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(3) << value;
        return out.str();
    }

    void sendLine(const std::string& line) {
        std::lock_guard<std::mutex> lock(outputMutex_);
        std::cout << line << '\n';
        std::cout.flush();
    }

    void sendInfo(const Analysis& analysis, const std::string& tag) {
        if (!analysis.valid) return;
        std::ostringstream info;
        info << "info engine=" << engineName(analysis.kind);
        if (!tag.empty()) info << " " << tag;
        if (analysis.kind == EngineKind::Pruning) {
            info << " depth=" << analysis.depth << " nodes=" << analysis.nodes
                 << " window=" << analysis.window
                 << " wres=" << analysis.windowResearches
                 << " pvs=" << analysis.pvsResearches
                 << " score=" << (analysis.score >= 0 ? "+" : "") << analysis.score
                 << " winrate=" << formatDouble(analysis.winrate)
                 << " best=" << (isPass(analysis.best)
                                      ? std::string("pass")
                                      : squareName(analysis.best.cell) + ":" +
                                            std::to_string(static_cast<int>(analysis.best.number)));
            if (!analysis.pv.empty()) info << " pv=" << analysis.pv;
        } else {
            info << " backend=" << (options_.nnOverride ? "cuda" : "cpu")
                 << " hasscore=" << (analysis.hasScore ? 1 : 0)
                 << " proven=" << (analysis.proven ? 1 : 0)
                 << " visits=" << analysis.visits << " nodes=" << analysis.nodes;
            if (analysis.hasScore) {
                info << " score=" << (analysis.averageScore >= 0 ? "+" : "")
                     << formatDouble(analysis.averageScore);
            }
            info << " winrate=" << formatDouble(analysis.winrate)
                 << " best=" << (isPass(analysis.best)
                                      ? std::string("pass")
                                      : squareName(analysis.best.cell) + ":" +
                                            std::to_string(static_cast<int>(analysis.best.number)));
            if (!analysis.candidates.empty()) {
                info << " moves=";
                for (std::size_t i = 0; i < analysis.candidates.size(); ++i) {
                    if (i) info << ",";
                    const auto& candidate = analysis.candidates[i];
                    info << squareName(candidate.move.cell) << ":" << static_cast<int>(candidate.move.number)
                         << ":" << formatDouble(candidate.winrate) << ":" << candidate.visits
                         << ":" << formatDouble(candidate.averageScore);
                }
            }
            if (!analysis.cells.empty()) {
                info << " cells=";
                for (std::size_t i = 0; i < analysis.cells.size(); ++i) {
                    if (i) info << ",";
                    const auto& cell = analysis.cells[i];
                    info << squareName(cell.cell) << ":" << static_cast<int>(cell.move.number)
                         << ":" << formatDouble(cell.winrate) << ":" << cell.visits
                         << ":" << formatDouble(cell.averageScore) << ":" << (cell.proven ? 1 : 0);
                }
            }
        }
        sendLine(info.str());
    }

    void startSearch(bool analysisMode, int searchVisits) {
        stopAndJoin();
        if (!engine_) return;
        const Position position = position_;
        Engine* engine = engine_.get();
        const std::string tag = engine->infoTag();
        const bool pruning = kind_ == EngineKind::Pruning;
        const int searchTime = options_.timeLimitMs;
        stop_.store(false, std::memory_order_relaxed);
        running_.store(true, std::memory_order_relaxed);
        worker_ = std::thread([this, position, engine, analysisMode, pruning, tag, searchTime,
                               searchVisits]() {
            SearchRequest request;
            // A configured visit budget applies to analysis as well.  Analysis
            // used to ignore it and search until stopped, so the same position
            // showed different win rates on every refresh -- and the numbers kept
            // creeping towards certainty the longer the window stayed open.
            request.visits = searchVisits;
            request.timeLimitMs = request.visits > 0 ? 0 : (analysisMode ? 3600000 : searchTime);
            request.stop = &stop_;
            request.onUpdate = [this, tag](const Analysis& snapshot) {
                if (running_.load()) sendInfo(snapshot, tag);
            };

            Analysis last;
            if (analysisMode && pruning && request.visits == 0) {
                // Without a visit budget alpha-beta stops at its depth limit, so
                // keep refining until the UI asks to stop.
                while (!stop_.load(std::memory_order_relaxed)) {
                    request.timeLimitMs = 1000;
                    last = engine->search(position, request);
                    if (!last.valid) break;
                }
            } else {
                last = engine->search(position, request);
            }

            if (!analysisMode) {
                if (stop_.load(std::memory_order_relaxed)) {
                    sendLine("= stopped");
                } else if (last.valid && isPass(last.best)) {
                    sendLine("= pass");
                } else if (last.valid && last.best.valid()) {
                    sendLine("= " + squareName(last.best.cell) + " " +
                             std::to_string(static_cast<int>(last.best.number)));
                } else {
                    sendLine("? no legal move");
                }
            } else if (last.valid) {
                // Tell the front end that the analysis ended by itself instead of
                // by `stop`: it can then extend the same tree without the budget.
                sendLine(std::string("= done visits=") + std::to_string(last.visits) +
                         " proven=" + (last.proven ? "1" : "0"));
            }
            running_.store(false, std::memory_order_relaxed);
        });
    }

    Position position_;
    EngineOptions options_;
    EngineKind kind_ = EngineKind::Pruning;
    std::unique_ptr<Engine> engine_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
    bool quit_ = false;
    // Requested backend: cpu / cuda / auto.  Auto by default: an engine built with
    // CUDA must use the GPU without waiting for the front end to ask, otherwise a
    // front end that never sends `set_backend` silently runs on one CPU core.
    std::string backend_ = "auto";
    std::mutex outputMutex_;
};

} // namespace

int main() {
    std::ios::sync_with_stdio(false);
    Session session;
    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const CommandResult result = session.handle(line);
        session.respond(result.ok, result.text);
        if (session.finished()) break;
    }
    return 0;
}
