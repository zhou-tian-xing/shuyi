#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Small Windows-side client for the line protocol spoken by shuyi_engine.exe.
// It never runs search on the UI thread: a reader thread fills a queue and the
// UI drains it from its 100 ms timer.
class EngineClient {
public:
    EngineClient() = default;
    ~EngineClient();
    EngineClient(const EngineClient&) = delete;
    EngineClient& operator=(const EngineClient&) = delete;

    // The UI does not care which build produced the engine: it can start any
    // binary that speaks the protocol.  An empty path keeps the historical
    // default (shuyi_engine.exe next to this executable).
    bool start(std::wstring* error = nullptr);
    bool start(const std::wstring& enginePath, std::wstring* error = nullptr);
    void stop();
    bool running() const { return running_.load(std::memory_order_relaxed); }
    bool send(const std::string& line);
    // Sends one command and waits for its final reply (`= ...` or `? ...`),
    // stashing any `info` line back for the next drain().  Used for the start-up
    // handshake, where a rejected command must be visible immediately instead of
    // showing up later as an unexplained empty analysis.
    bool command(const std::string& line, std::string* reply, int timeoutMs = 5000);
    std::vector<std::string> drain();
    std::wstring path() const { return path_; }

private:
    void readerLoop(HANDLE readHandle);

#ifdef _WIN32
    HANDLE stdinRead_ = nullptr;
    HANDLE stdinWrite_ = nullptr;
    HANDLE stdoutRead_ = nullptr;
    HANDLE stdoutWrite_ = nullptr;
    HANDLE stderrNul_ = nullptr;
    HANDLE process_ = nullptr;
    HANDLE thread_ = nullptr;
    DWORD processId_ = 0;
#endif
    std::wstring path_;
    std::thread reader_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stopRequested_{false};
    std::mutex writeMutex_;
    std::mutex queueMutex_;
    std::deque<std::string> queue_;
};
