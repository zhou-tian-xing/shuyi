// Finding an engine binary.
//
// The UI does not ship one: the engine is a separate product that the user
// downloads, builds or points at.  This module answers two questions only --
// "where could an engine be" and "does this file speak the protocol" -- so the
// window code can show a precise error instead of guessing.
#pragma once

#include <string>
#include <vector>

struct EngineProbe {
    bool answered = false;
    std::string backend;   // "cuda" or "cpu"
    std::string proto;     // protocol major version, "" when it did not answer
    bool visitBudget = false;
    std::string raw;       // the whole capabilities line
};

// Directory of the running shuyi.exe.
std::wstring engineClientDirectory();

// Engine binaries that exist right now, in preference order.  Explicit settings
// come first (SHUYI_ENGINE, then the exe directory), then the build directories
// a developer checkout keeps next to it.
std::vector<std::wstring> engineCandidates();

// Every path engineCandidates() looked at, whether it existed or not.
const std::vector<std::wstring>& enginePathsTried();

// Spawns the binary, asks `capabilities`, hangs up.  This is how an engine from
// another project is recognised.
EngineProbe probeEngine(const std::wstring& path);
