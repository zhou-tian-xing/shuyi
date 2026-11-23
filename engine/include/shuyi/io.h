#pragma once

// Path handling for a program that lives under a non-ASCII folder.
//
// The repository is called "数弈", so absolute paths routinely contain Chinese
// characters.  A narrow std::string coming from argv is UTF-8 (that is what the
// shell and Python pass), while MinGW's narrow file APIs use the ANSI code page:
// `std::ofstream("F:/projects/数弈/...")` throws "Illegal byte sequence" and
// `std::fopen` simply fails.  Everything that opens a file by a user-supplied
// path goes through here instead.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace shuyi {

// UTF-8 -> std::filesystem::path, wide on Windows, untouched elsewhere.
inline std::filesystem::path pathFromUtf8(const std::string& path) {
#if defined(_WIN32)
    if (path.empty()) return std::filesystem::path();
    const int needed = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (needed <= 1) return std::filesystem::path(path);
    std::wstring wide(static_cast<std::size_t>(needed - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), needed);
    return std::filesystem::path(wide);
#else
    return std::filesystem::path(path);
#endif
}

inline std::FILE* fopenUtf8(const std::string& path, const char* mode) {
#if defined(_WIN32)
    return _wfopen(pathFromUtf8(path).c_str(), std::wstring(mode, mode + std::char_traits<char>::length(mode)).c_str());
#else
    return std::fopen(path.c_str(), mode);
#endif
}

inline std::ofstream openOutUtf8(const std::string& path,
                                 std::ios::openmode mode = std::ios::binary) {
    return std::ofstream(pathFromUtf8(path), mode);
}

inline std::ifstream openInUtf8(const std::string& path,
                                std::ios::openmode mode = std::ios::binary) {
    return std::ifstream(pathFromUtf8(path), mode);
}

// On Windows a native parent (Python, the pipeline) hands args over in the
// ANSI code page, so a Chinese folder name arrives as GBK bytes.  Everything
// downstream expects UTF-8, so convert once at the top of main():
//
//     shuyi::Utf8Args utf8(argc, argv);
//     argv = utf8.pointers();
//
// On other platforms this is a no-op container.
class Utf8Args {
public:
    Utf8Args(int argc, char** argv) {
        values_.reserve(static_cast<std::size_t>(argc));
        for (int i = 0; i < argc; ++i) {
#if defined(_WIN32)
            values_.push_back(ansiToUtf8(argv[i]));
#else
            values_.push_back(argv[i]);
#endif
        }
        for (std::string& value : values_) pointers_.push_back(value.data());
        pointers_.push_back(nullptr);
        argc_ = argc;
    }
    int argc() const { return argc_; }
    char** pointers() { return pointers_.data(); }

private:
#if defined(_WIN32)
    static std::string ansiToUtf8(const char* text) {
        if (!text) return std::string();
        const int wideLength = MultiByteToWideChar(CP_ACP, 0, text, -1, nullptr, 0);
        if (wideLength <= 1) return std::string(text);
        std::wstring wide(static_cast<std::size_t>(wideLength - 1), L'\0');
        MultiByteToWideChar(CP_ACP, 0, text, -1, wide.data(), wideLength);
        const int utf8Length =
            WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
        if (utf8Length <= 1) return std::string(text);
        std::string out(static_cast<std::size_t>(utf8Length - 1), '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, out.data(), utf8Length, nullptr, nullptr);
        return out;
    }
#endif
    int argc_ = 0;
    std::vector<std::string> values_;
    std::vector<char*> pointers_;
};

} // namespace shuyi
