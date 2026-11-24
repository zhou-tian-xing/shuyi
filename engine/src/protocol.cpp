#include "shuyi/protocol.h"

#include "shuyi/io.h"
#include "shuyi/nn.h"
#include "shuyi/protocol_json.h"

#ifdef SHUYI_CUDA_ENGINE
#include "shuyi/nn_cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace shuyi {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kProtocolVersion = 2;
constexpr const char* kEngineVersion = "0.3.0";
// The default reporting rate.  It is a rate, not a limit: the search always
// continues until its visit budget or `stop`.
constexpr int kDefaultIntervalMs = 100;

std::string trim(std::string value) {
    while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ' ||
                              value.back() == '\t')) {
        value.pop_back();
    }
    std::size_t start = 0;
    while (start < value.size() && (value[start] == ' ' || value[start] == '\t')) ++start;
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
    if (t == "red" || t == "r") {
        player = Player::Red;
        return true;
    }
    if (t == "black" || t == "b") {
        player = Player::Black;
        return true;
    }
    return false;
}

const char* sideName(Player player) { return player == Player::Red ? "red" : "black"; }

std::string moveToken(Move move) {
    if (isPass(move)) return "pass";
    return squareName(move.cell) + ":" + std::to_string(static_cast<int>(move.number));
}

long long elapsedMs(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
}

#ifdef SHUYI_CUDA_ENGINE
// The batched CUDA forward, used one position at a time.  The CPU model is still
// loaded: it owns the feature layout, the move index mapping and the rule set.
class CudaEvaluator {
public:
    bool load(const std::string& path, std::string* error) {
        if (!net_.load(path, error)) return false;
        if (!gpu_.load(net_, error)) return false;
        planes_ = net_.modelInputPlanes();
        features_.assign(static_cast<std::size_t>(planes_) * kCells, 0.0f);
        globals_.assign(static_cast<std::size_t>(net_.globalFeatures()), 0.0f);
        logits_.assign(kNnPolicy, 0.0f);
        return true;
    }

    bool hasScoreHead() const { return net_.hasScoreHead(); }
    bool aInput() const { return net_.aInput(); }

    NnOutput evaluate(const Position& position, double aValue) {
        NnOutput out;
        std::array<std::int8_t, kCells> board{};
        for (int cell = 0; cell < kCells; ++cell) {
            board[cell] = static_cast<std::int8_t>(position.stoneAt(cell));
        }
        net_.buildFeaturePlanes(board, position.ply(), features_.data());
        net_.buildGlobals(board, position.ply(), position.rawScore(), aValue, globals_.data());
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
    std::vector<float> features_;
    std::vector<float> globals_;
    std::vector<float> logits_;
};
#endif

} // namespace

ProtocolSession::ProtocolSession() {
    options_.seed = 0x5EED1234ULL;
    // 512 simulations: the evaluation budget every rating in this repository was
    // measured at.  `set_visits` changes the default; a command may override it.
    options_.visits = 512;
    std::string ignored;
    rebuild(kind_, &ignored);  // mcts_nn without a model: ready() says so later
}

ProtocolSession::~ProtocolSession() { stopAndJoin(); }

// ------------------------------------------------------------------- output --

void ProtocolSession::writeLine(const std::string& text) {
    std::lock_guard<std::mutex> lock(outputMutex_);
    std::cout << text << '\n';
    std::cout.flush();
}

void ProtocolSession::respond(const std::string& id, bool ok, const std::string& text) {
    std::string line;
    if (!id.empty()) {
        line += id;
        line += ' ';
    }
    line += ok ? "= " : "? ";
    line += text;
    writeLine(line);
}

// ------------------------------------------------------------------ engines --

bool ProtocolSession::applyBackend(std::string* error) {
    options_.nnOverride.reset();
    cudaOverride_.reset();
    backendUsed_ = "cpu";
    if (backend_ == "cpu") return true;
#ifdef SHUYI_CUDA_ENGINE
    if (options_.nnModelPath.empty()) {
        if (backend_ == "cuda") {
            if (error) *error = "set_backend cuda needs a model first: set_model <path>";
            return false;
        }
        return true;  // auto: nothing to accelerate until a model arrives
    }
    auto evaluator = std::make_shared<CudaEvaluator>();
    std::string reason;
    if (!evaluator->load(options_.nnModelPath, &reason)) {
        if (backend_ == "cuda") {
            if (error) *error = "cuda is unavailable: " + reason;
            return false;
        }
        // `auto` is an explicit choice to use the GPU when it works.  The CPU is
        // reported, never silently assumed: `set_backend` answers with the
        // backend that was actually selected.
        return true;
    }
    auto override = std::make_shared<NnOverride>();
    override->evaluate = [evaluator](const Position& position, double aValue) {
        return evaluator->evaluate(position, aValue);
    };
    override->hasScoreHead = [evaluator] { return evaluator->hasScoreHead(); };
    override->aInput = [evaluator] { return evaluator->aInput(); };
    options_.nnOverride = override;
    cudaOverride_ = override;
    backendUsed_ = "cuda";
    return true;
#else
    if (backend_ == "cuda") {
        if (error) *error = "this engine build has no CUDA backend";
        return false;
    }
    return true;
#endif
}

bool ProtocolSession::rebuild(EngineKind kind, std::string* error, bool requireReady) {
    stopAndJoin();
    std::unique_ptr<Engine> fresh = makeEngine(kind, options_);
    if (!fresh) {
        if (error) *error = "cannot create engine " + std::string(engineName(kind));
        return false;
    }
    std::string reason;
    if (requireReady && !fresh->ready(&reason)) {
        if (error) *error = reason;
        return false;
    }
    engine_ = std::move(fresh);
    kind_ = kind;
    return true;
}

bool ProtocolSession::ensureReady(std::string* error) const {
    if (!engine_) {
        if (error) *error = "no engine";
        return false;
    }
    return engine_->ready(error);
}

void ProtocolSession::stopAndJoin() {
    stop_.store(true, std::memory_order_relaxed);
    if (worker_.joinable()) worker_.join();
    stop_.store(false, std::memory_order_relaxed);
    running_.store(false, std::memory_order_relaxed);
}

// ------------------------------------------------------------------ commands --

void ProtocolSession::handle(const std::string& raw) {
    const std::string line = trim(raw);
    if (line.empty() || line[0] == '#') return;

    std::istringstream input(line);
    std::string token;
    input >> token;
    id_.clear();
    const bool numeric = !token.empty() &&
                         std::all_of(token.begin(), token.end(),
                                     [](char c) { return c >= '0' && c <= '9'; });
    if (numeric) {
        id_ = token;
        if (!(input >> token)) {
            respond(false, "missing command");
            return;
        }
    }
    const std::string command = lower(token);

    if (command == "protocol_version") return respond(true, std::to_string(kProtocolVersion));
    if (command == "name") return respond(true, "ShuyiEngine");
    if (command == "version") return respond(true, kEngineVersion);
    if (command == "list_commands") {
        return respond(true,
                       "protocol_version name version list_commands capabilities boardsize "
                       "ruleset clear_board play genmove analyze stop set_engine set_backend "
                       "set_model set_visits set_time set_a set_window set_pvs bench selfplay quit");
    }
    if (command == "capabilities") return cmdCapabilities();
    if (command == "quit") {
        stopAndJoin();
        respond(true, "");
        quit_ = true;
        return;
    }
    if (command == "stop") {
        stopAndJoin();
        return respond(true, "stopped");
    }
    if (command == "boardsize") {
        int size = 0;
        if (!(input >> size)) return respond(false, "usage: boardsize 9");
        if (size != kSize) {
            return respond(false, "unsupported boardsize " + std::to_string(size) +
                                      ": this engine plays 9x9 only");
        }
        return respond(true, "");
    }
    if (command == "clear_board") {
        stopAndJoin();
        position_.reset();
        if (engine_) engine_->reset();
        return respond(true, "");
    }
    if (command == "ruleset") {
        std::string value;
        if (!(input >> value)) return respond(true, rulesetName(position_.ruleset()));
        if (position_.stones() != 0) {
            return respond(false, "the ruleset can only change before the first move");
        }
        Ruleset parsed = position_.ruleset();
        if (!parseRuleset(lower(value), parsed)) {
            return respond(false, "usage: ruleset [classic|blocking|nopass]");
        }
        position_.setRuleset(parsed);
        // The tree, the model and the evaluator all belong to the old rule set.
        // Rather than silently searching a blocking game with a classic model,
        // drop the model and the engine: the front end sets both up again.
        engine_.reset();
        if (!options_.nnModelPath.empty()) {
            PolicyValueNet probe;
            std::string reason;
            if (probe.load(options_.nnModelPath, &reason) &&
                !rulesetCompatible(probe.ruleset(), parsed)) {
                const std::string old = rulesetName(probe.ruleset());
                options_.nnModelPath.clear();
                std::string ignored;
                applyBackend(&ignored);
                return respond(true, std::string(rulesetName(parsed)) +
                                        " (model cleared: it was trained for " + old + ")");
            }
        }
        return respond(true, rulesetName(parsed));
    }
    if (command == "set_engine") {
        std::string name;
        if (!(input >> name)) return respond(false, "usage: set_engine <engine>");
        return cmdSetEngine(name);
    }
    if (command == "set_backend") {
        std::string name;
        if (!(input >> name)) return respond(false, "usage: set_backend cpu|cuda|auto");
        name = lower(name);
        if (name != "cpu" && name != "cuda" && name != "auto") {
            return respond(false, "backend must be cpu, cuda or auto");
        }
        const std::string previous = backend_;
        backend_ = name;
        std::string error;
        if (!applyBackend(&error)) {
            backend_ = previous;
            std::string ignored;
            applyBackend(&ignored);
            return respond(false, error);
        }
        if (!rebuild(kind_, &error, false)) return respond(false, error);
        return respond(true, backendUsed_);
    }
    if (command == "set_model") {
        std::string path;
        std::getline(input, path);
        path = trim(path);
        if (path.empty()) return respond(false, "usage: set_model <path to .bin>");
        // Validate before touching the running engine: a model trained for the
        // other rule set produces plausible-looking nonsense, so it is refused.
        bool modelSeesA = false;
        Ruleset modelRuleset = position_.ruleset();
        {
            PolicyValueNet probe;
            std::string reason;
            if (!probe.load(path, &reason)) {
                return respond(false, reason.empty() ? "cannot load model: " + path : reason);
            }
            if (!rulesetCompatible(probe.ruleset(), position_.ruleset())) {
                return respond(false, "model is for the " +
                                          std::string(rulesetName(probe.ruleset())) +
                                          " ruleset, but the position is " +
                                          rulesetName(position_.ruleset()));
            }
            modelSeesA = probe.aInput();
            modelRuleset = probe.ruleset();
        }
        stopAndJoin();
        const std::string previous = options_.nnModelPath;
        const bool previousAInput = aInput_;
        options_.nnModelPath = path;
        std::string error;
        if (!applyBackend(&error) || !rebuild(kind_, &error)) {
            options_.nnModelPath = previous;
            aInput_ = previousAInput;
            std::string ignored;
            applyBackend(&ignored);
            return respond(false, error);
        }
        aInput_ = modelSeesA;
        // The reply says what the model can do; `set_a` is meaningless to a
        // v3 model, and a blocking model played under NoPass is a deliberate
        // compatibility (every NoPass game is a blocking game up to its end),
        // so both are stated instead of assumed.
        std::string note;
        if (!aInput_) note += " (no A input: v3 model)";
        if (modelRuleset != position_.ruleset()) {
            note += std::string(" (trained for ") + rulesetName(modelRuleset) +
                    ", played under " + rulesetName(position_.ruleset()) + ")";
        }
        return respond(true, path + note);
    }
    if (command == "set_visits") {
        int visits = 0;
        if (!(input >> visits)) return respond(false, "usage: set_visits <n>");
        if (visits < 0) return respond(false, "visits must be >= 0");
        // Only the default changes: a front end extending an analysis keeps the
        // tree it already has.
        options_.visits = visits;
        return respond(true, visits > 0 ? std::to_string(visits) : std::string("off"));
    }
    if (command == "set_time") {
        int ms = 0;
        if (!(input >> ms)) return respond(false, "usage: set_time <milliseconds>");
        if (ms < 0) return respond(false, "time must be >= 0");
        options_.timeLimitMs = ms;
        return respond(true, std::to_string(ms));
    }
    if (command == "set_a") {
        double value = 0.0;
        if (!(input >> value) || !std::isfinite(value)) return respond(false, "usage: set_a <value>");
        stopAndJoin();
        options_.aValue = value;
        std::string error;
        if (!rebuild(kind_, &error, false)) return respond(false, error);
        // 1.26 rather than 1.260000: the reply is echoed by front ends.
        std::ostringstream formatted;
        formatted << value;
        return respond(true, formatted.str());
    }
    if (command == "set_window") {
        int halfWidth = 0;
        if (!(input >> halfWidth)) return respond(false, "usage: set_window <points>");
        if (halfWidth < 0) return respond(false, "window must be >= 0");
        stopAndJoin();
        options_.aspirationWindow = halfWidth;
        std::string error;
        if (!rebuild(kind_, &error, false)) return respond(false, error);
        return respond(true, std::to_string(halfWidth));
    }
    if (command == "set_pvs") {
        std::string value;
        if (!(input >> value)) return respond(false, "usage: set_pvs on|off");
        value = lower(value);
        bool enabled = false;
        if (value == "on" || value == "1") enabled = true;
        else if (value != "off" && value != "0") return respond(false, "usage: set_pvs on|off");
        stopAndJoin();
        options_.usePvs = enabled;
        std::string error;
        if (!rebuild(kind_, &error, false)) return respond(false, error);
        return respond(true, enabled ? "on" : "off");
    }
    if (command == "play") {
        std::string color;
        std::string vertex;
        if (!(input >> color >> vertex)) {
            return respond(false, "usage: play <red|black> <a1..i9> <1..9>|pass");
        }
        Player player;
        if (!parsePlayer(color, player)) {
            return respond(false, "usage: play <red|black> <a1..i9> <1..9>|pass");
        }
        if (player != position_.sideToMove()) {
            return respond(false, std::string("wrong side to move: it is ") +
                                      sideName(position_.sideToMove()) + "'s turn");
        }
        stopAndJoin();
        Move move{};
        if (lower(vertex) == "pass") {
            if (!position_.pass()) return respond(false, "pass is only legal when no placement is legal");
            move = Move(kPassCell, kPassNumber);
        } else {
            int number = 0;
            int cell = 0;
            if (!parseSquare(vertex, cell) || !(input >> number) || number < 1 || number > 9) {
                return respond(false, "usage: play <red|black> <a1..i9> <1..9>|pass");
            }
            move = Move(cell, number);
            if (!position_.play(move)) return respond(false, "illegal move " + moveToken(move));
        }
        if (engine_) engine_->onPlay(position_, move);
        return respond(true, "");
    }
    if (command == "genmove") {
        std::string rest;
        std::getline(input, rest);
        return cmdAnalyze(rest, true);
    }
    if (command == "analyze") {
        std::string rest;
        std::getline(input, rest);
        return cmdAnalyze(rest, false);
    }
    if (command == "bench") {
        std::string rest;
        std::getline(input, rest);
        return cmdBench(rest);
    }
    if (command == "selfplay") {
        std::string rest;
        std::getline(input, rest);
        std::istringstream args(rest);
        int games = 1;
        int visits = options_.visits;
        if (!(args >> games)) games = 1;
        if (!(args >> visits)) visits = options_.visits;
        if (games < 1) return respond(false, "usage: selfplay <games> [visits]");
        if (visits < 1) return respond(false, "selfplay needs a visit budget >= 1");
        stopAndJoin();
        std::string error;
        if (!ensureReady(&error)) return respond(false, error);
        std::unique_ptr<Engine> fresh = makeEngine(kind_, options_);
        if (!fresh) return respond(false, "cannot create engine");
        int red = 0;
        int black = 0;
        int draw = 0;
        long long plies = 0;
        for (int game = 0; game < games; ++game) {
            Position board;
            board.setRuleset(position_.ruleset());
            fresh->reset();
            while (!board.gameOver()) {
                SearchRequest request;
                request.visits = visits;
                request.timeLimitMs = 0;
                request.reportIntervalMs = 1000000;
                const Analysis move = fresh->search(board, request);
                if (!move.best.valid()) break;
                if (!board.play(move.best)) break;
                fresh->onPlay(board, move.best);
            }
            plies += board.ply();
            const Outcome outcome = board.outcome(options_.aValue);
            if (outcome == Outcome::RedWin) ++red;
            else if (outcome == Outcome::BlackWin) ++black;
            else ++draw;
        }
        std::ostringstream out;
        out << "games=" << games << " red=" << red << " black=" << black << " draw=" << draw
            << " avg_ply=" << (games > 0 ? static_cast<double>(plies) / games : 0.0)
            << " visits=" << visits << " engine=" << engineName(kind_)
            << " backend=" << backendUsed_;
        return respond(true, out.str());
    }

    respond(false, "unknown command: " + command);
}

void ProtocolSession::cmdCapabilities() {
#ifdef SHUYI_CUDA_ENGINE
    const bool cuda = true;
#else
    const bool cuda = false;
#endif
    const bool model = engine_ && engine_->ready(nullptr);
    std::ostringstream out;
    out << "engine=shuyi_engine proto=" << kProtocolVersion << " cuda=" << (cuda ? 1 : 0)
        << " backend=" << backendUsed_ << " rulesets=classic,blocking,nopass visits=1 stream=1 json=1"
        << " model=" << (model ? 1 : 0) << " a_input=" << (aInput_ ? 1 : 0);
    respond(true, out.str());
}

void ProtocolSession::cmdSetEngine(std::string& name) {
    EngineKind kind;
    if (!parseEngineName(lower(name), kind)) return respond(false, "unknown engine: " + name);
    std::string error;
    // Selecting an engine is not running it: `mcts_nn` may be selected before its
    // model arrives, and only a search without a model is an error.
    if (!rebuild(kind, &error, false)) return respond(false, error);
    respond(true, engineName(kind));
}

void ProtocolSession::cmdAnalyze(const std::string& args, bool playMode) {
    std::istringstream input(args);
    std::string color;
    if (!(input >> color)) {
        return respond(false, playMode ? "usage: genmove <red|black> [visits]"
                                       : "usage: analyze <red|black> [visits] [interval <ms>]");
    }
    Player player;
    if (!parsePlayer(color, player)) return respond(false, "unknown colour: " + color);
    if (player != position_.sideToMove()) {
        return respond(false, std::string("wrong side to move: it is ") +
                                  sideName(position_.sideToMove()) + "'s turn");
    }
    if (position_.gameOver()) return respond(false, "game is over");

    // Visits: a machine-independent budget.  `visits 0` means "no budget" for an
    // analysis (search until `stop`); for genmove it falls back to the clock,
    // because a move has to come back eventually.
    int visits = options_.visits;
    int interval = kDefaultIntervalMs;
    std::string token;
    while (input >> token) {
        if (lower(token) == "interval") {
            if (!(input >> interval) || interval < 1 || interval > 60000) {
                return respond(false, "interval must be 1..60000 milliseconds");
            }
            continue;
        }
        const bool numeric = std::all_of(token.begin(), token.end(),
                                         [](char c) { return c >= '0' && c <= '9'; });
        if (!numeric || token.size() > 9) {
            return respond(false, "unexpected argument: " + token);
        }
        visits = std::stoi(token);
    }
    if (visits < 0) return respond(false, "visits must be >= 0");

    std::string error;
    // One search at a time: a new analysis replaces the running one, and the
    // previous worker must be joined before its std::thread is reassigned.
    stopAndJoin();
    if (!ensureReady(&error)) return respond(false, error);

    const Position snapshot = position_;
    Engine* engine = engine_.get();
    const std::string id = id_;
    const std::string turn = sideName(snapshot.sideToMove());
    const std::string backend = backendUsed_;
    const bool analysisMode = !playMode;
    const EngineKind kind = kind_;
    // Only a budget ends an analysis by itself; a clock budget applies to genmove
    // and to a front end that explicitly asks for no visit limit on a move.
    const int timeLimit = (analysisMode && visits == 0) ? 0 : (visits > 0 ? 0 : options_.timeLimitMs);
    const std::string tag = engine->modelName();

    stop_.store(false, std::memory_order_relaxed);
    running_.store(true, std::memory_order_relaxed);
    searchId_ = id;

    worker_ = std::thread([this, snapshot, engine, id, turn, backend, analysisMode, visits, interval,
                           timeLimit, tag, kind]() {
        const Clock::time_point started = Clock::now();
        SearchRequest request;
        request.visits = visits;
        request.reportIntervalMs = interval;
        request.timeLimitMs = timeLimit;
        request.stop = &stop_;
        request.onUpdate = [&](const Analysis& snapshotAnalysis) {
            std::string line;
            if (!id.empty()) line += id + " ";
            line += "info ";
            line += analysisJson(snapshotAnalysis, backend, turn, elapsedMs(started), tag,
                                 aInput_);
            writeLine(line);
        };

        Analysis last;
        if (analysisMode && kind == EngineKind::Pruning && visits == 0) {
            // Alpha-beta stops at its depth limit, so keep refining until the
            // front end asks to stop.
            while (!stop_.load(std::memory_order_relaxed)) {
                request.timeLimitMs = 1000;
                last = engine->search(snapshot, request);
                if (!last.valid) break;
            }
        } else {
            last = engine->search(snapshot, request);
        }

        const bool stopped = stop_.load(std::memory_order_relaxed);
        if (!analysisMode) {
            if (stopped) {
                respond(id, true, "stopped");
            } else if (!last.valid) {
                respond(id, false, "no legal move");
            } else if (isPass(last.best)) {
                respond(id, true, "pass");
            } else {
                // Same shape as `play` takes, so a front end can echo it back
                // without reading a second syntax.
                respond(id, true, squareName(last.best.cell) + " " +
                                      std::to_string(static_cast<int>(last.best.number)));
            }
        } else if (!last.valid) {
            respond(id, false, "search is not available in this position");
        } else {
            std::ostringstream out;
            out << (stopped ? "stopped" : "done") << " visits=" << last.visits
                << " proven=" << (last.proven ? 1 : 0) << " elapsed_ms=" << elapsedMs(started);
            respond(id, true, out.str());
        }
        running_.store(false, std::memory_order_relaxed);
    });
}

void ProtocolSession::cmdBench(const std::string& args) {
    std::istringstream input(args);
    int visits = options_.visits;
    std::string token;
    if (input >> token) {
        const bool numeric = std::all_of(token.begin(), token.end(),
                                         [](char c) { return c >= '0' && c <= '9'; });
        if (!numeric) return respond(false, "usage: bench [visits]");
        visits = std::stoi(token);
    }
    if (visits < 1) return respond(false, "bench needs a visit budget >= 1");
    stopAndJoin();
    std::string error;
    if (!ensureReady(&error)) return respond(false, error);

    std::unique_ptr<Engine> fresh = makeEngine(kind_, options_);
    if (!fresh) return respond(false, "cannot create engine");
    // Six deterministic opening plies so the benchmark is not a degenerate empty
    // board; the same sequence on every machine.
    Position board;
    board.setRuleset(position_.ruleset());
    for (int ply = 0; ply < 6 && !board.gameOver(); ++ply) {
        const std::vector<Move> moves = board.searchMoves();
        if (moves.empty()) break;
        if (!board.play(moves[static_cast<std::size_t>(ply * 7) % moves.size()])) break;
    }

    const Clock::time_point started = Clock::now();
    SearchRequest request;
    request.visits = visits;
    request.timeLimitMs = 0;
    request.reportIntervalMs = 1000000;
    const Analysis analysis = fresh->search(board, request);
    const long long ms = std::max<long long>(1, elapsedMs(started));
    const double seconds = static_cast<double>(ms) / 1000.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(1);
    out << "visits=" << analysis.visits << " time_ms=" << ms
        << " visits_per_second=" << static_cast<double>(analysis.visits) / seconds
        << " engine=" << engineName(kind_) << " backend=" << backendUsed_;
    const std::string tag = fresh->modelName();
    if (!tag.empty()) out << " nn=" << tag;
    respond(true, out.str());
}

} // namespace shuyi
