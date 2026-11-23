#pragma once

#include "shuyi/nn.h"

#include <memory>
#include <string>
#include <vector>

namespace shuyi {

// CUDA backend for PolicyValueNet.  Batched: one call evaluates many positions,
// which is what makes self-play affordable (the C++ CPU path costs ~0.4 ms per
// b2c16 evaluation and ~14 ms per b6c64 evaluation).
class CudaPolicyValueNet {
public:
    CudaPolicyValueNet();
    ~CudaPolicyValueNet();
    CudaPolicyValueNet(const CudaPolicyValueNet&) = delete;
    CudaPolicyValueNet& operator=(const CudaPolicyValueNet&) = delete;

    bool load(const PolicyValueNet& net, std::string* error = nullptr);
    bool loaded() const;
    int inputPlanes() const;
    int channels() const;
    int blocks() const;
    const std::string& deviceName() const;

    // x: [batch][planes][kCells], globals: [batch][kResGlobalFeatures].
    // logitsOut: [batch][kNnPolicy], valueOut: [batch].
    // scoreOut: [batch] expected terminal raw score (tanh(score/kScoreScale));
    // only written when the model carries a score head and scoreOut != nullptr.
    void evaluate(const float* x, const float* globals, int batch, float* logitsOut,
                  float* valueOut, float* scoreOut = nullptr);

    static bool available(std::string* reason = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace shuyi
