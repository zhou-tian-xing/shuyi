// Micro-benchmark: cost of one network evaluation and of one MCTS simulation.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "shuyi/io.h"
#include "shuyi/nn.h"
#include "shuyi/position.h"

#include <chrono>
#include <cstdio>
#include <string>

using namespace shuyi;

int main(int argc, char** argv) {
    shuyi::Utf8Args utf8Args(argc, argv);
    argv = utf8Args.pointers();
    const std::string path = argc > 1 ? argv[1] : "models/shuyi_resnet_v2.bin";
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 20000;
    PolicyValueNet net;
    std::string error;
    if (!net.load(path, &error)) {
        std::printf("load failed: %s\n", error.c_str());
        return 1;
    }
    Position position;
    for (int i = 0; i < 20; ++i) {
        const auto moves = position.legalMoves();
        if (moves.empty()) break;
        position.play(moves[(i * 37) % moves.size()]);
    }
    const auto started = std::chrono::steady_clock::now();
    volatile float sink = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        const NnOutput out = net.evaluate(position);
        sink = sink + out.value;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::printf("model=%s eval=%d seconds=%.3f us/eval=%.1f evals/s=%.0f (sink=%.1f)\n",
                path.c_str(), iterations, seconds, seconds * 1e6 / iterations,
                iterations / seconds, static_cast<double>(sink));
    return 0;
}
