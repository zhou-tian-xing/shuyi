#include "discovery.h"

#include "engine_client.h"

#include <windows.h>

#include <cstring>

namespace {

std::wstring widenAnsi(const std::string& text) {
    if (text.empty()) return std::wstring();
    const int length = MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, nullptr, 0);
    if (length <= 1) return std::wstring();
    std::wstring wide(static_cast<std::size_t>(length - 1), L'\0');
    MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, wide.data(), length);
    return wide;
}

std::vector<std::wstring> g_pathsTried;

} // namespace

std::vector<std::wstring> engineCandidates() {
    std::vector<std::wstring> paths;
    g_pathsTried.clear();
    const auto push = [&](const std::wstring& path) {
        if (path.empty()) return;
        g_pathsTried.push_back(path);
        for (const std::wstring& seen : paths) {
            if (_wcsicmp(seen.c_str(), path.c_str()) == 0) return;
        }
        if (GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES) paths.push_back(path);
    };
    char env[MAX_PATH * 2]{};
    const DWORD envLength = GetEnvironmentVariableA("SHUYI_ENGINE", env, sizeof(env));
    std::wstring override;
    if (envLength > 0 && envLength < sizeof(env)) override = widenAnsi(std::string(env, envLength));
    if (!override.empty()) push(override);

    const std::wstring exeDir = engineClientDirectory();
    const std::wstring engine = L"\\shuyi_engine.exe";
    push(exeDir + engine);
    for (const wchar_t* sub : {L"\\..\\build-cuda", L"\\..\\build-cuda\\Release", L"\\..\\build",
                               L"\\..\\build\\Release"}) {
        push(exeDir + sub + engine);
    }
    char cwd[MAX_PATH]{};
    if (GetCurrentDirectoryA(MAX_PATH, cwd) > 0) {
        const std::wstring cwdDir = widenAnsi(std::string(cwd));
        push(cwdDir + engine);
        for (const wchar_t* sub : {L"\\build-cuda", L"\\build"}) push(cwdDir + sub + engine);
    }
    return paths;
}

const std::vector<std::wstring>& enginePathsTried() { return g_pathsTried; }

std::wstring engineClientDirectory() {
    char buffer[MAX_PATH]{};
    const DWORD length = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) return std::wstring();
    std::string path(buffer, length);
    const std::size_t slash = path.find_last_of("/\\");
    const std::string dir = slash == std::string::npos ? std::string() : path.substr(0, slash);
    return widenAnsi(dir);
}

EngineProbe probeEngine(const std::wstring& path) {
    EngineProbe info;
    EngineClient probe;
    if (!probe.start(path, nullptr)) return info;
    std::string reply;
    if (!probe.command("capabilities", &reply, 3000)) {
        probe.stop();
        return info;
    }
    probe.stop();
    if (reply.rfind("= ", 0) != 0) return info;
    info.answered = true;
    info.raw = reply;
    info.backend = reply.find("cuda=1") != std::string::npos ? "cuda" : "cpu";
    info.visitBudget = reply.find("visits=1") != std::string::npos;
    const std::size_t at = reply.find("proto=");
    if (at != std::string::npos) {
        std::size_t end = at + 6;
        while (end < reply.size() && reply[end] != ' ') ++end;
        info.proto = reply.substr(at + 6, end - (at + 6));
    }
    return info;
}
