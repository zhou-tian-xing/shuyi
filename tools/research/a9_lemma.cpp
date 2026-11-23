// Computer-assisted finite lemma used in docs/A_VALUE_ANALYSIS.zh-CN.md.
//
// B9 = the three black 9 stones, N9 = union of their closed Manhattan
// neighbourhoods.  For c in N9 the black maximum is 9, hence
//   score(c) = 0 if some red 9 is within distance 1 of c, else -1.
// The loss 1-score(c) is therefore 1 or 2, and the total N9 loss is
//   2*|N9| - (# covered cells).
//
// The program enumerates all C(81,3) placements of B9.  If the minimum possible
// N9 loss is >=8, the bound is already proved.  Otherwise it enumerates *all*
// placements of the three red 9s and checks every arrangement whose N9 loss is
// exactly 7.  In such an arrangement, at most 2 cells can host a black 8
// without creating an additional loss outside N9.  Three black 8 stones exist,
// so at least one creates an extra loss and total loss is at least 8.  Hence raw
// score <= 81-8 = 73.
#include <algorithm>
#include <array>
#include <bitset>
#include <cstdio>
#include <vector>

namespace {
constexpr int N = 9, CELLS = 81;
using Mask = std::bitset<CELLS>;

int idx(int x, int y) { return y * N + x; }
int xof(int i) { return i % N; }
int yof(int i) { return i / N; }

Mask cross(int i) {
    int x = xof(i), y = yof(i);
    Mask m;
    m.set(i);
    if (x) m.set(idx(x - 1, y));
    if (x + 1 < N) m.set(idx(x + 1, y));
    if (y) m.set(idx(x, y - 1));
    if (y + 1 < N) m.set(idx(x, y + 1));
    return m;
}
} // namespace

int main() {
    std::array<Mask, CELLS> cr{};
    for (int i = 0; i < CELLS; ++i) cr[i] = cross(i);

    long long cases = 0;
    long long criticalTriples = 0;
    long long criticalRedArrangements = 0;
    long long failure = 0;
    int worstAllowed = -1;

    for (int b0 = 0; b0 < CELLS; ++b0)
    for (int b1 = b0 + 1; b1 < CELLS; ++b1)
    for (int b2 = b1 + 1; b2 < CELLS; ++b2) {
        ++cases;
        const Mask A = cr[b0] | cr[b1] | cr[b2];
        const int na = static_cast<int>(A.count());
        std::vector<int> candidates;
        for (int r = 0; r < CELLS; ++r) {
            if (r == b0 || r == b1 || r == b2) continue;
            if ((cr[r] & A).any()) candidates.push_back(r);
        }

        int maxCover = 0;
        const int count = static_cast<int>(candidates.size());
        for (int i = 0; i < count; ++i) {
            maxCover = std::max(maxCover, static_cast<int>((cr[candidates[i]] & A).count()));
            for (int j = i + 1; j < count; ++j) {
                const Mask two = cr[candidates[i]] | cr[candidates[j]];
                maxCover = std::max(maxCover, static_cast<int>((two & A).count()));
                for (int k = j + 1; k < count; ++k) {
                    const Mask three = two | cr[candidates[k]];
                    maxCover = std::max(maxCover, static_cast<int>((three & A).count()));
                }
            }
        }
        const int minLoss = 2 * na - maxCover;
        if (minLoss > 7) continue;
        if (minLoss < 7) {
            std::printf("FAIL: N9 loss below 7 (%d)\n", minLoss);
            ++failure;
            continue;
        }
        ++criticalTriples;

        // Exhaustive over all red 9 placements, including red 9s outside N9
        // that can still cover cells next to a future black 8.
        for (int r0 = 0; r0 < CELLS; ++r0) {
            if (r0 == b0 || r0 == b1 || r0 == b2) continue;
            for (int r1 = r0 + 1; r1 < CELLS; ++r1) {
                if (r1 == b0 || r1 == b1 || r1 == b2) continue;
                for (int r2 = r1 + 1; r2 < CELLS; ++r2) {
                    if (r2 == b0 || r2 == b1 || r2 == b2) continue;
                    const Mask R = cr[r0] | cr[r1] | cr[r2];
                    const int covered = static_cast<int>((A & R).count());
                    if (2 * na - covered != 7) continue;
                    ++criticalRedArrangements;

                    int allowed = 0;
                    for (int cell = 0; cell < CELLS; ++cell) {
                        if (cell == b0 || cell == b1 || cell == b2) continue;
                        if (cell == r0 || cell == r1 || cell == r2) continue;
                        const Mask outside = cr[cell] & ~A;
                        if ((outside & ~R).none()) ++allowed;
                    }
                    worstAllowed = std::max(worstAllowed, allowed);
                    if (allowed >= 3) {
                        std::printf("FAIL: a loss-7 arrangement admits three harmless black 8s\n");
                        ++failure;
                    }
                }
            }
        }
    }

    std::printf("black-9 triples checked    : %lld\n", cases);
    std::printf("critical (N9 loss 7) triples: %lld\n", criticalTriples);
    std::printf("loss-7 red arrangements    : %lld\n", criticalRedArrangements);
    std::printf("max harmless black 8s      : %d (need 3)\n", worstAllowed);
    std::printf("finite lemma               : %s\n", failure == 0 ? "PASS" : "FAIL");
    return failure == 0 ? 0 : 1;
}
