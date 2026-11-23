// The engine's command side: one persistent process, command driven, streaming.
//
// The front end owns the game; this process owns the computation.  It keeps the
// position the front end told it about, a search tree for that position, and a
// single worker thread that streams analysis until it is done or told to stop.
// Everything else is a command (protocol/ENGINE_PROTOCOL.md).
#pragma once

#include "shuyi/engine.h"
#include "shuyi/position.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace shuyi {

class ProtocolSession {
public:
    ProtocolSession();
    ~ProtocolSession();
    ProtocolSession(const ProtocolSession&) = delete;
    ProtocolSession& operator=(const ProtocolSession&) = delete;

    // Executes one command line and writes every reply line itself.
    void handle(const std::string& line);
    bool finished() const { return quit_; }

private:
    struct Backend;

    // output (single mutex: the search thread streams, the reader replies)
    void writeLine(const std::string& text);
    void respond(const std::string& id, bool ok, const std::string& text);
    void respond(bool ok, const std::string& text) { respond(id_, ok, text); }

    // engine lifecycle
    // `requireReady` is false for commands that only configure the search (A
    // value, window, PVS, engine selection): selecting `mcts_nn` before its model
    // has arrived is a legal state, and only searching without a model is an
    // error.  `set_model` does require it, because accepting a model that cannot
    // load would hide the failure until the next move.
    bool rebuild(EngineKind kind, std::string* error, bool requireReady = true);
    bool applyBackend(std::string* error);   // cpu / cuda / auto -> evaluator
    bool ensureReady(std::string* error) const;
    void stopAndJoin();

    // commands
    void cmdSetEngine(std::string& name);
    void cmdSetModel(const std::string& path);
    void cmdAnalyze(const std::string& args, bool play);
    void cmdBench(const std::string& args);
    void cmdCapabilities();

    Position position_;
    EngineOptions options_;
    EngineKind kind_ = EngineKind::MctsNn;
    std::unique_ptr<Engine> engine_;
    std::shared_ptr<NnOverride> cudaOverride_;

    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
    std::string searchId_;          // id prefix of the streaming analysis
    std::mutex outputMutex_;

    std::string backend_ = "auto";  // requested
    std::string backendUsed_ = "cpu";
    std::string id_;                // id of the command being handled
    bool quit_ = false;
};

} // namespace shuyi
