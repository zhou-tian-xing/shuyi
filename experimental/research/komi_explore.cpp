// Exploratory tool: search final 27-vs-27 placements for extreme / target scores.
// Not part of the shipped engine.  Build: g++ -O3 -std=c++17 tools/komi_explore.cpp -o build/komi_explore
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <random>
#include <string>
#include <vector>
#include "shuyi/io.h"

namespace {
constexpr int N = 9;
constexpr int CELLS = N * N;
constexpr int EMPTY = 0;
constexpr int RED_BASE = 0;    // 1..9
constexpr int BLACK_BASE = 10; // 11..19

int idx(int x, int y) { return y * N + x; }
int x_of(int i) { return i % N; }
int y_of(int i) { return i / N; }

int code_player(int code) { return code == 0 ? -1 : (code <= 9 ? 0 : 1); }
int code_number(int code) { return code == 0 ? 0 : (code <= 9 ? code : code - 10); }

int cell_value(const std::array<uint8_t, CELLS>& b, int i) {
    const int x = x_of(i), y = y_of(i);
    int rmax = 0, bmax = 0;
    auto add = [&](int xx, int yy) {
        if (xx < 0 || xx >= N || yy < 0 || yy >= N) return;
        int c = b[idx(xx, yy)];
        if (c == 0) return;
        if (c <= 9) rmax = std::max(rmax, c);
        else bmax = std::max(bmax, c - 10);
    };
    add(x, y);
    add(x - 1, y); add(x + 1, y); add(x, y - 1); add(x, y + 1);
    return (rmax > bmax) - (rmax < bmax);
}

int full_score(const std::array<uint8_t, CELLS>& b) {
    int s = 0;
    for (int i = 0; i < CELLS; ++i) s += cell_value(b, i);
    return s;
}

std::array<uint8_t, CELLS> random_board(std::mt19937_64& rng) {
    std::array<uint8_t, CELLS> b{};
    std::vector<uint8_t> pieces;
    for (int p = 0; p < 2; ++p)
        for (int n = 1; n <= 9; ++n)
            for (int k = 0; k < 3; ++k) pieces.push_back(static_cast<uint8_t>(n + 10 * p));
    std::vector<int> places(CELLS);
    for (int i = 0; i < CELLS; ++i) places[i] = i;
    std::shuffle(places.begin(), places.end(), rng);
    for (size_t i = 0; i < pieces.size(); ++i) b[places[i]] = pieces[i];
    return b;
}

int affected_sum(const std::array<uint8_t, CELLS>& b, int a, int c) {
    // Cells whose closed neighbourhood contains a or c: include duplicates once.
    std::array<int, 10> list{};
    int n = 0;
    auto put = [&](int i) {
        int x = x_of(i), y = y_of(i);
        int cand[5] = {i,
                       (x > 0) ? idx(x - 1, y) : -1,
                       (x + 1 < N) ? idx(x + 1, y) : -1,
                       (y > 0) ? idx(x, y - 1) : -1,
                       (y + 1 < N) ? idx(x, y + 1) : -1};
        for (int q = 0; q < 5; ++q) {
            int v = cand[q];
            if (v < 0) continue;
            bool seen = false;
            for (int z = 0; z < n; ++z) seen |= list[z] == v;
            if (!seen) list[n++] = v;
        }
    };
    put(a);
    if (c != a) put(c);
    int s = 0;
    for (int q = 0; q < n; ++q) s += cell_value(b, list[q]);
    return s;
}

void print_board(const std::array<uint8_t, CELLS>& b, int score, bool signs = true) {
    std::printf("score=%d\n", score);
    std::printf("   a b c d e f g h i\n");
    for (int y = 0; y < N; ++y) {
        std::printf("%d  ", y + 1);
        for (int x = 0; x < N; ++x) {
            int c = b[idx(x, y)];
            if (!c) std::printf(". ");
            else std::printf("%c%d ", c <= 9 ? 'R' : 'B', code_number(c));
        }
        std::printf("\n");
    }
    if (signs) {
        std::printf("cell signs:\n   a b c d e f g h i\n");
        for (int y = 0; y < N; ++y) {
            std::printf("%d  ", y + 1);
            for (int x = 0; x < N; ++x) {
                int v = cell_value(b, idx(x, y));
                std::printf("%c ", v > 0 ? '+' : (v < 0 ? '-' : '0'));
            }
            std::printf("\n");
        }
    }
}

struct Result { int score; std::array<uint8_t, CELLS> board; };

Result anneal(std::mt19937_64& rng, int target, int max_iters, bool find_max, double t0 = 4.0, double t1 = 0.02) {
    std::array<uint8_t, CELLS> b = random_board(rng);
    int s = full_score(b);
    auto best = b; int best_s = s;
    std::uniform_int_distribution<int> cell(0, CELLS - 1);
    double temp = t0;
    for (int it = 0; it < max_iters; ++it) {
        if ((it & 1023) == 0) {
            temp = t0 * std::pow(t1 / t0, static_cast<double>(it) / max_iters);
        }
        int i = cell(rng), j = cell(rng);
        if (i == j || b[i] == b[j]) continue;
        int before = affected_sum(b, i, j);
        std::swap(b[i], b[j]);
        int after = affected_sum(b, i, j);
        int ns = s - before + after;
        int delta = ns - s;
        bool accept = false;
        if (find_max) {
            if (target == INT32_MIN) accept = delta >= 0 || std::uniform_real_distribution<double>(0, 1)(rng) < std::exp(delta / temp);
            else {
                int oldd = std::abs(s - target), newd = std::abs(ns - target);
                accept = newd <= oldd || std::uniform_real_distribution<double>(0, 1)(rng) < std::exp(-(newd - oldd) / temp);
            }
        } else {
            if (target == INT32_MIN) accept = delta <= 0 || std::uniform_real_distribution<double>(0, 1)(rng) < std::exp(-delta / temp);
            else {
                int oldd = std::abs(s - target), newd = std::abs(ns - target);
                accept = newd <= oldd || std::uniform_real_distribution<double>(0, 1)(rng) < std::exp(-(newd - oldd) / temp);
            }
        }
        if (accept) {
            s = ns;
            if (find_max && target == INT32_MIN) { if (s > best_s) { best_s = s; best = b; } }
            else if (!find_max && target == INT32_MIN) { if (s < best_s) { best_s = s; best = b; } }
            else if (std::abs(s - target) < std::abs(best_s - target)) { best_s = s; best = b; }
        } else {
            std::swap(b[i], b[j]);
        }
    }
    if (target != INT32_MIN) {
        // Return the closest configuration found, with its actual score.
        return {best_s, best};
    }
    return {best_s, best};
}
} // namespace

void print_witness(const std::array<uint8_t, CELLS>& b, int score) {
    std::printf("# score %d\n%d\n", score, score);
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int c = b[idx(x, y)];
            if (x) std::printf(" ");
            if (!c) std::printf(".");
            else std::printf("%c%d", c <= 9 ? 'R' : 'B', code_number(c));
        }
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    shuyi::Utf8Args utf8Args(argc, argv);
    argv = utf8Args.pointers();
    std::string mode = argc > 1 ? argv[1] : "max";
    int target = argc > 2 ? std::atoi(argv[2]) : 0;
    int iters = argc > 3 ? std::atoi(argv[3]) : 2000000;
    uint64_t seed = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 12345;
    std::mt19937_64 rng(seed);
    if (mode == "ladder") {
        for (int want = -73; want <= 73; ++want) {
            bool done = false;
            for (int attempt = 0; attempt < 20 && !done; ++attempt) {
                std::mt19937_64 local(seed + static_cast<uint64_t>(want + 100) * 100003 + attempt * 7919);
                Result r = anneal(local, want, iters, true);
                int actual = full_score(r.board);
                if (actual == want) {
                    print_witness(r.board, actual);
                    std::fprintf(stderr, "score %d ok (attempt %d)\n", want, attempt + 1);
                    done = true;
                }
            }
            if (!done) {
                std::fprintf(stderr, "FAILED target %d\n", want);
                return 1;
            }
        }
        return 0;
    }
    Result r;
    if (mode == "max") r = anneal(rng, INT32_MIN, iters, true);
    else if (mode == "min") r = anneal(rng, INT32_MIN, iters, false);
    else r = anneal(rng, target, iters, true);
    print_board(r.board, r.score);
    std::printf("recomputed full score=%d\n", full_score(r.board));
    return 0;
}
