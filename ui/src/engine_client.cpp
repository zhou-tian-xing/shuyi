#include "engine_client.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>

#ifdef _WIN32

namespace {
std::wstring executableDirectory() {
    wchar_t buffer[MAX_PATH * 4]{};
    const DWORD length = GetModuleFileNameW(nullptr, buffer, static_cast<DWORD>(std::size(buffer)));
    if (length == 0 || length >= std::size(buffer)) return {};
    std::filesystem::path path(buffer);
    return path.parent_path().wstring();
}
} // namespace

EngineClient::~EngineClient() { stop(); }

bool EngineClient::start(std::wstring* error) {
    return start(executableDirectory() + L"\\shuyi_engine.exe", error);
}

bool EngineClient::start(const std::wstring& enginePath, std::wstring* error) {
    stop();
    path_ = enginePath;
    if (!std::filesystem::exists(path_)) {
        if (error) *error = L"找不到引擎进程: " + path_;
        return false;
    }

    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    if (!CreatePipe(&stdinRead_, &stdinWrite_, &security, 0)) {
        if (error) *error = L"创建 stdin 管道失败";
        return false;
    }
    if (!CreatePipe(&stdoutRead_, &stdoutWrite_, &security, 0)) {
        if (error) *error = L"创建 stdout 管道失败";
        stop();
        return false;
    }
    SetHandleInformation(stdinWrite_, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stdoutRead_, HANDLE_FLAG_INHERIT, 0);

    stderrNul_ = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE, &security, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    if (stderrNul_ == INVALID_HANDLE_VALUE) stderrNul_ = nullptr;

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = stdinRead_;
    startup.hStdOutput = stdoutWrite_;
    startup.hStdError = stderrNul_ ? stderrNul_ : stdoutWrite_;

    PROCESS_INFORMATION process{};
    std::wstring command = L"\"" + path_ + L"\"";
    std::vector<wchar_t> mutableCommand(command.begin(), command.end());
    mutableCommand.push_back(L'\0');
    const std::wstring workingDirectory = executableDirectory();
    const BOOL ok = CreateProcessW(path_.c_str(), mutableCommand.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW, nullptr,
                                   workingDirectory.empty() ? nullptr : workingDirectory.c_str(),
                                   &startup, &process);
    CloseHandle(stdinRead_);
    stdinRead_ = nullptr;
    CloseHandle(stdoutWrite_);
    stdoutWrite_ = nullptr;
    if (!ok) {
        if (error) *error = L"启动引擎进程失败";
        stop();
        return false;
    }
    process_ = process.hProcess;
    thread_ = process.hThread;
    processId_ = process.dwProcessId;
    running_.store(true);
    stopRequested_.store(false);
    const HANDLE readHandle = stdoutRead_;
    reader_ = std::thread([this, readHandle]() { readerLoop(readHandle); });
    return true;
}

void EngineClient::readerLoop(HANDLE readHandle) {
    std::string pending;
    char buffer[4096];
    while (!stopRequested_.load(std::memory_order_relaxed)) {
        DWORD read = 0;
        if (!ReadFile(readHandle, buffer, sizeof(buffer), &read, nullptr) || read == 0) break;
        pending.append(buffer, read);
        std::size_t start = 0;
        while (true) {
            const std::size_t newline = pending.find('\n', start);
            if (newline == std::string::npos) break;
            std::string line = pending.substr(start, newline - start);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            {
                std::lock_guard<std::mutex> lock(queueMutex_);
                queue_.push_back(std::move(line));
            }
            start = newline + 1;
        }
        if (start > 0) pending.erase(0, start);
    }
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (!pending.empty()) queue_.push_back(std::move(pending));
    }
}

bool EngineClient::send(const std::string& line) {
    if (!running_.load()) return false;
    std::lock_guard<std::mutex> lock(writeMutex_);
    const std::string data = line + "\n";
    DWORD written = 0;
    return WriteFile(stdinWrite_, data.data(), static_cast<DWORD>(data.size()), &written, nullptr) &&
           written == data.size();
}

std::vector<std::string> EngineClient::drain() {
    std::vector<std::string> lines;
    std::lock_guard<std::mutex> lock(queueMutex_);
    while (!queue_.empty()) {
        lines.push_back(std::move(queue_.front()));
        queue_.pop_front();
    }
    return lines;
}

void EngineClient::stop() {
    stopRequested_.store(true);
    if (running_.load() && stdinWrite_) {
        const std::string quit = "quit\n";
        DWORD written = 0;
        WriteFile(stdinWrite_, quit.data(), static_cast<DWORD>(quit.size()), &written, nullptr);
    }
    if (process_) {
        WaitForSingleObject(process_, 300);
        DWORD exitCode = 0;
        if (GetExitCodeProcess(process_, &exitCode) && exitCode == STILL_ACTIVE) {
            TerminateProcess(process_, 0);
        }
    }
    if (reader_.joinable()) {
        // Cancel the blocking ReadFile before closing the pipe handle.
        if (thread_) {
            CancelSynchronousIo(reinterpret_cast<HANDLE>(
                reinterpret_cast<std::uintptr_t>(reader_.native_handle())));
        }
        if (stdoutRead_) CloseHandle(stdoutRead_);
        stdoutRead_ = nullptr;
        reader_.join();
    }
    if (stdinWrite_) CloseHandle(stdinWrite_);
    stdinWrite_ = nullptr;
    if (stdoutWrite_) CloseHandle(stdoutWrite_);
    stdoutWrite_ = nullptr;
    if (stderrNul_) CloseHandle(stderrNul_);
    stderrNul_ = nullptr;
    if (thread_) CloseHandle(thread_);
    thread_ = nullptr;
    if (process_) CloseHandle(process_);
    process_ = nullptr;
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        queue_.clear();
    }
    running_.store(false);
}

#endif
