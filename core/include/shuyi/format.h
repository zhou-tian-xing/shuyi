// Small presentation helpers shared by the UI and its tests.
//
// Visit counts inside a board cell must fit in about four characters, so they are
// abbreviated rather than printed in full: 820, 1.2k, 12k, 120k, 1.2M.
#pragma once

#include <cmath>
#include <cstdio>
#include <string>

namespace shuyi {

// Compact visit count: at most four characters, never wider than "999k".
inline std::wstring formatVisits(long long visits) {
    if (visits < 0) visits = 0;
    wchar_t buffer[16]{};
    if (visits < 1000) {
        std::swprintf(buffer, 16, L"%lld", visits);
    } else if (visits < 10000) {
        // 1.2k .. 9.9k
        std::swprintf(buffer, 16, L"%.1fk", static_cast<double>(visits) / 1000.0);
    } else if (visits < 1000000) {
        // 10k .. 999k
        std::swprintf(buffer, 16, L"%.0fk", static_cast<double>(visits) / 1000.0);
    } else {
        std::swprintf(buffer, 16, L"%.1fM", static_cast<double>(visits) / 1000000.0);
    }
    return std::wstring(buffer);
}

// Win rate as a whole percentage, e.g. 62%.
inline std::wstring formatWinRate(double winrate) {
    const double clamped = winrate < 0.0 ? 0.0 : (winrate > 1.0 ? 1.0 : winrate);
    wchar_t buffer[16]{};
    std::swprintf(buffer, 16, L"%.0f%%", clamped * 100.0);
    return std::wstring(buffer);
}

// Expected terminal score from the side to move, one decimal: +3.4 / -0.5 / 0.0.
inline std::wstring formatAverageScore(double score) {
    wchar_t buffer[24]{};
    if (score > 0.0) {
        std::swprintf(buffer, 24, L"+%.1f", score);
    } else if (score < 0.0) {
        std::swprintf(buffer, 24, L"%.1f", score);
    } else {
        std::swprintf(buffer, 24, L"0.0");
    }
    return std::wstring(buffer);
}

// An exact margin inside a board cell: at most four characters.  Whole points are
// the normal case ("+3", "-12"); when the A value makes the margin fractional the
// one decimal is kept ("+3.3"), and an exact zero is "0" rather than "-0".
inline std::wstring formatPoints(double score) {
    wchar_t buffer[24]{};
    const double rounded = std::round(score);
    if (std::fabs(score - rounded) < 0.05) {
        if (rounded > 0.0) std::swprintf(buffer, 24, L"+%.0f", rounded);
        else if (rounded < 0.0) std::swprintf(buffer, 24, L"%.0f", rounded);
        else std::swprintf(buffer, 24, L"0");
    } else {
        std::swprintf(buffer, 24, L"%+.1f", score);
    }
    return std::wstring(buffer);
}

// Wilson score interval lower bound for a win rate measured over `visits` samples.
//
// This is the number the board marks rank by: it answers "how good is this move,
// given how little we know about it".  A single lucky visit at 100% scores 0.21,
// a well searched 55% over 200 visits scores 0.49, so the well searched move wins
// -- and, because a cell's visit count only grows, the ranking stops flickering.
inline double winRateLowerBound(double winrate, long long visits) {
    if (visits <= 0) return 0.0;
    const double n = static_cast<double>(visits);
    const double z = 1.96;
    const double p = winrate < 0.0 ? 0.0 : (winrate > 1.0 ? 1.0 : winrate);
    const double denom = 1.0 + z * z / n;
    const double centre = p + z * z / (2.0 * n);
    const double margin = z * std::sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n));
    const double lower = (centre - margin) / denom;
    return lower > 0.0 ? lower : 0.0;
}

} // namespace shuyi
