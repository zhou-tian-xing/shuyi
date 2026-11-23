#pragma once

#include "shuyi/position.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace shuyi {

// A small KataGo-style residual policy/value network that can run on CPU.
//
// Legacy v1/v2 input (19 planes): 18 stone planes + side-to-move plane.
// Hand counts live only in the 22-dim global vector, and reach policy only as
// a per-number bias.  That is too weak for this game: the trunk cannot use
// hand composition to decide *where* to play.
//
// v3 input (37 planes):
//   0..17  stones (red/black, numbers 1..9)
//   18..26 red remaining counts 1..9, constant planes count/3
//   27..35 black remaining counts 1..9, constant planes count/3
//   36     side to move
// The binary format stores the actual input/channel/block shape, so old and
// new models load through the same C++ code.
constexpr int kResInputPlanes = 19;          // legacy
constexpr int kResInputPlanesV3 = 37;       // explicit hand-count planes
constexpr int kResChannels = 16;
constexpr int kResBlocks = 2;
constexpr int kResGlobalFeatures = 22;
constexpr int kResValueHidden = 32;
// The score head predicts tanh(score / kScoreScale); 30 covers the observed
// range of final scores (about +-25).
constexpr float kScoreScale = 30.0f;
constexpr int kNnPolicy = kCells * 9;

struct NnOutput {
    std::array<float, kNnPolicy> policy{};
    // Expected terminal raw score from the side to move, in points.  Only filled
    // when the model carries a score head (SHUYINN3 flags bit 0).
    float expectedScore = 0.0f;
    float value = 0.5f;
};

class PolicyValueNet {
public:
    struct Block {
        std::vector<float> w1, b1, w2, b2;
    };

    PolicyValueNet();

    bool load(const std::string& path, std::string* error = nullptr);
    bool save(const std::string& path, std::string* error = nullptr) const;
    bool loaded() const { return loaded_; }
    const std::string& modelName() const { return modelName_; }
    int channels() const { return channels_; }
    int blocks() const { return blocks_; }
    int modelInputPlanes() const { return inputPlanes_; }

    NnOutput evaluate(const Position& position) const;

    // Feature construction shared by the CPU and CUDA backends.  `out` must hold
    // modelInputPlanes() * kCells floats.  The v3 layout keeps the historical
    // off-by-one (hand planes at 19..36, side plane written at 36) so existing
    // .bin models stay valid.
    void buildFeaturePlanes(const std::array<std::int8_t, kCells>& board, int ply,
                            float* out) const;
    void buildGlobals(const std::array<std::int8_t, kCells>& board, int ply, int raw,
                      float* out) const;

    // Weight accessors for the CUDA backend.
    const std::vector<float>& stemWeight() const { return stemW_; }
    const std::vector<float>& stemBias() const { return stemB_; }
    const std::vector<Block>& blockData() const { return blocksData_; }
    const std::vector<float>& policyWeight() const { return policyW_; }
    const std::vector<float>& policyBias() const { return policyB_; }
    const std::vector<float>& handWeight() const { return handW_; }
    const std::vector<float>& handBiasData() const { return handB_; }
    const std::vector<float>& valueWeight() const { return valueW_; }
    float valueBias() const { return valueB_; }
    const std::vector<float>& valueFc1Weight() const { return valueFc1W_; }
    const std::vector<float>& valueFc1Bias() const { return valueFc1B_; }
    const std::vector<float>& valueFc2Weight() const { return valueFc2W_; }
    const std::vector<float>& valueFc2Bias() const { return valueFc2B_; }
    const std::vector<float>& scoreWeight() const { return scoreW_; }
    float scoreBias() const { return scoreB_; }
    const std::vector<float>& scoreFc1Weight() const { return scoreFc1W_; }
    const std::vector<float>& scoreFc1Bias() const { return scoreFc1B_; }
    const std::vector<float>& scoreFc2Weight() const { return scoreFc2W_; }
    const std::vector<float>& scoreFc2Bias() const { return scoreFc2B_; }
    int moveIndex(Move move) const { return static_cast<int>(move.cell) * 9 + (move.number - 1); }
    Move moveFromIndex(int index) const { return Move(index / 9, index % 9 + 1); }

    static PolicyValueNet random(std::uint64_t seed, const std::string& name = "untrained",
                                 int channels = kResChannels, int blocks = kResBlocks,
                                 int inputPlanes = kResInputPlanes);

    // Exposed for tests/export tooling.
    static constexpr int inputPlanes() { return kResInputPlanes; }
    static constexpr int defaultChannels() { return kResChannels; }
    static constexpr int defaultBlocks() { return kResBlocks; }
    static constexpr int globalFeatures() { return kResGlobalFeatures; }

    // Rule set the weights were trained under.  Loading is permissive (a v2 file
    // is Classic) but callers must compare this against the game they are about
    // to play: the value head encodes one rule set and cannot be reused for the
    // other.
    Ruleset ruleset() const { return ruleset_; }
    void setRuleset(Ruleset value) { ruleset_ = value; }
    bool hasScoreHead() const { return hasScoreHead_; }
    // Turns the score head on for a freshly built model (tests, export tooling).
    // Everything stays zero except the output bias, so the head predicts
    // tanh(outputBias) * kScoreScale for every position until trained.
    void enableScoreHead(float outputBias = 0.0f) {
        hasScoreHead_ = true;
        if (scoreFc2B_.size() == 1) scoreFc2B_[0] = outputBias;
    }

private:
    void allocate(int channels, int blocks, int inputPlanes);
    void clear();

    int channels_ = kResChannels;
    int blocks_ = kResBlocks;
    int inputPlanes_ = kResInputPlanes;
    std::vector<float> stemW_, stemB_;
    std::vector<Block> blocksData_;
    std::vector<float> policyW_, policyB_;
    std::vector<float> handW_, handB_;
    std::vector<float> valueW_;
    std::vector<float> valueFc1W_, valueFc1B_, valueFc2W_, valueFc2B_;
    float valueB_ = 0.0f;
    // Score head (optional): one 1x1 channel, then the same small MLP shape.
    std::vector<float> scoreW_, scoreFc1W_, scoreFc1B_, scoreFc2W_, scoreFc2B_;
    float scoreB_ = 0.0f;
    bool loaded_ = false;
    std::string modelName_;
    bool hasScoreHead_ = false;
    Ruleset ruleset_ = kDefaultRuleset;
};

} // namespace shuyi
