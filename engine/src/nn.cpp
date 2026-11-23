#include "shuyi/nn.h"
#include "shuyi/io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

namespace shuyi {
#if defined(_MSC_VER)
#define SHUYI_RESTRICT __restrict
#else
#define SHUYI_RESTRICT __restrict__
#endif

namespace {
// v2 (historical) carries only the shape.  v3 adds the rule set and the feature
// layout so a model can never be silently loaded into the wrong game, plus a
// flags word for heads that may be added later (the score head).
constexpr char kMagicV2[8] = {'S', 'H', 'U', 'Y', 'I', 'N', 'N', '2'};
constexpr char kMagicV3[8] = {'S', 'H', 'U', 'Y', 'I', 'N', 'N', '3'};
constexpr std::uint32_t kVersion2 = 2;
constexpr std::uint32_t kVersion3 = 3;
// Feature layouts: 1 = stones 0..17, side to move 18, hand planes 19..36.
constexpr std::uint8_t kFeatureVersion = 1;
constexpr std::uint32_t kFlagScoreHead = 1u << 0;

constexpr int kPlane = kCells;
constexpr int kHiddenInput = 1 + kResGlobalFeatures;
constexpr int kMaxChannels = 128;
constexpr int kMaxBlocks = 16;

// Padded copy of one plane: index (y+1, x+1) so the 3x3 taps never need a
// bounds check.  Padding is (kSize+2)^2 floats.
constexpr int kPadSize = kSize + 2;
constexpr int kPadPlane = kPadSize * kPadSize;

// Scratch reused across calls: one evaluation used to allocate three plane-sized
// vectors per convolution, i.e. dozens of allocations per network call.
struct ConvScratch {
    std::vector<float> padded;
    std::vector<float> row;
};

ConvScratch& convScratch() {
    thread_local ConvScratch scratch;
    return scratch;
}

// 3x3 convolution on a 9x9 board, written so the compiler can vectorise it.
//
// The previous version walked (oc, ic, y, x, ky, kx) with four bounds tests in
// the innermost loop; that cost 36 ms per evaluation for a 86k-parameter network
// (0.4 GFLOP/s).  Here every tap is an unconditional, contiguous 81-float FMA
// chain, and the output channels are independent so they can run in parallel.
void conv3x3(const float* input, int inChannels, int outChannels, const float* weights,
             const float* bias, float* output, bool relu) {
    ConvScratch& scratch = convScratch();
    scratch.padded.assign(static_cast<std::size_t>(inChannels) * kPadPlane, 0.0f);
    for (int ic = 0; ic < inChannels; ++ic) {
        const float* src = input + static_cast<std::size_t>(ic) * kPlane;
        float* dst = scratch.padded.data() + static_cast<std::size_t>(ic) * kPadPlane;
        for (int y = 0; y < kSize; ++y) {
            const float* row = src + y * kSize;
            float* out = dst + (y + 1) * kPadSize + 1;
            for (int x = 0; x < kSize; ++x) out[x] = row[x];
        }
    }
    const float* padded = scratch.padded.data();

    // (OpenMP tried here measured 5x SLOWER: the pragma stops the inner loop
    // from vectorising, which costs far more than the threads gain.)
    for (int oc = 0; oc < outChannels; ++oc) {
        float* SHUYI_RESTRICT out = output + static_cast<std::size_t>(oc) * kPlane;
        const float base = bias[oc];
        for (int i = 0; i < kPlane; ++i) out[i] = base;
        const float* w = weights + static_cast<std::size_t>(oc) * inChannels * 9;
        for (int ic = 0; ic < inChannels; ++ic) {
            const float* SHUYI_RESTRICT in = padded + static_cast<std::size_t>(ic) * kPadPlane;
            const float* SHUYI_RESTRICT tap = w + ic * 9;
            // Nine taps, each one a run of 81 contiguous multiply-adds.
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    const float weight = tap[ky * 3 + kx];
                    const float* SHUYI_RESTRICT src = in + ky * kPadSize + kx;
                    for (int y = 0; y < kSize; ++y) {
                        const float* SHUYI_RESTRICT srcRow = src + y * kPadSize;
                        float* SHUYI_RESTRICT dstRow = out + y * kSize;
                        for (int x = 0; x < kSize; ++x) dstRow[x] += weight * srcRow[x];
                    }
                }
            }
        }
        if (relu) {
            for (int i = 0; i < kPlane; ++i) out[i] = std::max(0.0f, out[i]);
        }
    }
}

void residualBlock(const float* input, const PolicyValueNet::Block& block, int channels,
                   float* scratch, float* output) {
    conv3x3(input, channels, channels, block.w1.data(), block.b1.data(), scratch, true);
    conv3x3(scratch, channels, channels, block.w2.data(), block.b2.data(), output, false);
    for (int i = 0; i < channels * kPlane; ++i) {
        output[i] = std::max(0.0f, output[i] + input[i]);
    }
}

void fillRandom(std::vector<float>& values, std::size_t fanIn, std::mt19937_64& rng) {
    const float limit = 1.0f / std::sqrt(static_cast<float>(std::max<std::size_t>(1, fanIn)));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (float& value : values) value = dist(rng);
}

} // namespace

PolicyValueNet::PolicyValueNet() = default;

void PolicyValueNet::clear() {
    stemW_.clear();
    stemB_.clear();
    blocksData_.clear();
    policyW_.clear();
    policyB_.clear();
    handW_.clear();
    handB_.clear();
    valueW_.clear();
    valueFc1W_.clear();
    valueFc1B_.clear();
    valueFc2W_.clear();
    valueFc2B_.clear();
    valueB_ = 0.0f;
    scoreW_.clear();
    scoreFc1W_.clear();
    scoreFc1B_.clear();
    scoreFc2W_.clear();
    scoreFc2B_.clear();
    scoreB_ = 0.0f;
    hasScoreHead_ = false;
    loaded_ = false;
    modelName_.clear();
}

void PolicyValueNet::allocate(int channels, int blocks, int inputPlanes) {
    channels_ = std::clamp(channels, 1, kMaxChannels);
    blocks_ = std::clamp(blocks, 1, kMaxBlocks);
    inputPlanes_ = std::clamp(inputPlanes, 1, 64);
    stemW_.assign(static_cast<std::size_t>(channels_) * inputPlanes_ * 9, 0.0f);
    stemB_.assign(channels_, 0.0f);
    blocksData_.clear();
    blocksData_.resize(blocks_);
    for (Block& block : blocksData_) {
        block.w1.assign(static_cast<std::size_t>(channels_) * channels_ * 9, 0.0f);
        block.b1.assign(channels_, 0.0f);
        block.w2.assign(static_cast<std::size_t>(channels_) * channels_ * 9, 0.0f);
        block.b2.assign(channels_, 0.0f);
    }
    policyW_.assign(static_cast<std::size_t>(9) * channels_, 0.0f);
    policyB_.assign(9, 0.0f);
    handW_.assign(static_cast<std::size_t>(9) * kResGlobalFeatures, 0.0f);
    handB_.assign(9, 0.0f);
    valueW_.assign(channels_, 0.0f);
    valueFc1W_.assign(static_cast<std::size_t>(kResValueHidden) * kHiddenInput, 0.0f);
    valueFc1B_.assign(kResValueHidden, 0.0f);
    valueFc2W_.assign(kResValueHidden, 0.0f);
    valueFc2B_.assign(1, 0.0f);
    scoreW_.assign(channels_, 0.0f);
    scoreFc1W_.assign(static_cast<std::size_t>(kResValueHidden) * kHiddenInput, 0.0f);
    scoreFc1B_.assign(kResValueHidden, 0.0f);
    scoreFc2W_.assign(kResValueHidden, 0.0f);
    scoreFc2B_.assign(1, 0.0f);
    valueB_ = 0.0f;
}

PolicyValueNet PolicyValueNet::random(std::uint64_t seed, const std::string& name, int channels,
                                      int blocks, int inputPlanes) {
    PolicyValueNet net;
    net.allocate(channels, blocks, inputPlanes);
    std::mt19937_64 rng(seed);
    fillRandom(net.stemW_, static_cast<std::size_t>(net.inputPlanes_) * 9, rng);
    fillRandom(net.stemB_, static_cast<std::size_t>(kResInputPlanes) * 9, rng);
    for (Block& block : net.blocksData_) {
        fillRandom(block.w1, static_cast<std::size_t>(net.channels_) * 9, rng);
        fillRandom(block.b1, static_cast<std::size_t>(net.channels_) * 9, rng);
        fillRandom(block.w2, static_cast<std::size_t>(net.channels_) * 9, rng);
        fillRandom(block.b2, static_cast<std::size_t>(net.channels_) * 9, rng);
    }
    fillRandom(net.policyW_, net.channels_, rng);
    fillRandom(net.policyB_, net.channels_, rng);
    fillRandom(net.handW_, kResGlobalFeatures, rng);
    fillRandom(net.handB_, kResGlobalFeatures, rng);
    fillRandom(net.valueW_, net.channels_, rng);
    fillRandom(net.valueFc1W_, kHiddenInput, rng);
    fillRandom(net.valueFc1B_, kHiddenInput, rng);
    fillRandom(net.valueFc2W_, kResValueHidden, rng);
    fillRandom(net.valueFc2B_, kResValueHidden, rng);
    net.loaded_ = true;
    net.modelName_ = name;
    return net;
}

bool PolicyValueNet::save(const std::string& path, std::string* error) const {
    if (!loaded_) {
        if (error) *error = "cannot save an unloaded model";
        return false;
    }
    std::ofstream out = openOutUtf8(path);
    if (!out) {
        if (error) *error = "cannot open model for writing: " + path;
        return false;
    }
    out.write(kMagicV3, sizeof(kMagicV3));
    const std::uint32_t version = kVersion3;
    const std::uint32_t channels = static_cast<std::uint32_t>(channels_);
    const std::uint32_t blocks = static_cast<std::uint32_t>(blocks_);
    const std::uint32_t inputs = static_cast<std::uint32_t>(inputPlanes_);
    const std::uint32_t globals = kResGlobalFeatures;
    const std::uint32_t flags = hasScoreHead_ ? kFlagScoreHead : 0u;
    const std::uint8_t ruleset = static_cast<std::uint8_t>(ruleset_);
    const std::uint8_t features = kFeatureVersion;
    const std::uint16_t reserved = 0;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    out.write(reinterpret_cast<const char*>(&blocks), sizeof(blocks));
    out.write(reinterpret_cast<const char*>(&inputs), sizeof(inputs));
    out.write(reinterpret_cast<const char*>(&globals), sizeof(globals));
    out.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    out.write(reinterpret_cast<const char*>(&ruleset), sizeof(ruleset));
    out.write(reinterpret_cast<const char*>(&features), sizeof(features));
    out.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    const auto writeFloats = [&](const std::vector<float>& values) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
    };
    writeFloats(stemW_);
    writeFloats(stemB_);
    for (const Block& block : blocksData_) {
        writeFloats(block.w1);
        writeFloats(block.b1);
        writeFloats(block.w2);
        writeFloats(block.b2);
    }
    writeFloats(policyW_);
    writeFloats(policyB_);
    writeFloats(handW_);
    writeFloats(handB_);
    writeFloats(valueW_);
    out.write(reinterpret_cast<const char*>(&valueB_), sizeof(valueB_));
    writeFloats(valueFc1W_);
    writeFloats(valueFc1B_);
    writeFloats(valueFc2W_);
    writeFloats(valueFc2B_);
    if (hasScoreHead_) {
        writeFloats(scoreW_);
        out.write(reinterpret_cast<const char*>(&scoreB_), sizeof(scoreB_));
        writeFloats(scoreFc1W_);
        writeFloats(scoreFc1B_);
        writeFloats(scoreFc2W_);
        writeFloats(scoreFc2B_);
    }
    return static_cast<bool>(out);
}

bool PolicyValueNet::load(const std::string& path, std::string* error) {
    std::ifstream in = openInUtf8(path);
    if (!in) {
        if (error) *error = "cannot open model: " + path;
        return false;
    }
    char magic[8]{};
    in.read(magic, sizeof(magic));
    const bool isV3 = std::equal(std::begin(kMagicV3), std::end(kMagicV3), magic);
    const bool isV2 = std::equal(std::begin(kMagicV2), std::end(kMagicV2), magic);
    if (!in || (!isV3 && !isV2)) {
        if (error) *error = "bad model magic: " + path;
        return false;
    }
    std::uint32_t version = 0, channels = 0, blocks = 0, inputs = 0, globals = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&channels), sizeof(channels));
    in.read(reinterpret_cast<char*>(&blocks), sizeof(blocks));
    in.read(reinterpret_cast<char*>(&inputs), sizeof(inputs));
    in.read(reinterpret_cast<char*>(&globals), sizeof(globals));
    std::uint32_t flags = 0;
    std::uint8_t ruleset = 0;
    std::uint8_t features = kFeatureVersion;
    std::uint16_t reserved = 0;
    if (isV3) {
        in.read(reinterpret_cast<char*>(&flags), sizeof(flags));
        in.read(reinterpret_cast<char*>(&ruleset), sizeof(ruleset));
        in.read(reinterpret_cast<char*>(&features), sizeof(features));
        in.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
    }
    const bool versionOk = isV3 ? (version == kVersion3) : (version == kVersion2);
    if (!in || !versionOk || ruleset > 1 || features != kFeatureVersion ||
        (inputs != kResInputPlanes && inputs != kResInputPlanesV3) ||
        globals != kResGlobalFeatures || channels < 1 || channels > kMaxChannels ||
        blocks < 1 || blocks > kMaxBlocks) {
        if (error) *error = "unsupported model shape: " + path;
        return false;
    }
    if (isV2 && inputPlanes_ == kResInputPlanesV3) {
        // v2 files with 37 planes used the old layout, where the side plane was
        // written at index 36 on top of Black's count-of-9 plane.  Those weights
        // are not usable with the current features.
        if (error) {
            *error = "model uses the retired 37-plane layout (side plane at 36): " + path;
        }
        return false;
    }
    ruleset_ = static_cast<Ruleset>(ruleset);

    clear();
    allocate(static_cast<int>(channels), static_cast<int>(blocks), static_cast<int>(inputs));
    // After clear(): it resets the head flag, so setting it before is lost.
    hasScoreHead_ = (flags & kFlagScoreHead) != 0;
    const auto readFloats = [&](std::vector<float>& values) {
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
    };
    readFloats(stemW_);
    readFloats(stemB_);
    for (Block& block : blocksData_) {
        readFloats(block.w1);
        readFloats(block.b1);
        readFloats(block.w2);
        readFloats(block.b2);
    }
    readFloats(policyW_);
    readFloats(policyB_);
    readFloats(handW_);
    readFloats(handB_);
    readFloats(valueW_);
    in.read(reinterpret_cast<char*>(&valueB_), sizeof(valueB_));
    readFloats(valueFc1W_);
    readFloats(valueFc1B_);
    readFloats(valueFc2W_);
    readFloats(valueFc2B_);
    if (hasScoreHead_) {
        readFloats(scoreW_);
        in.read(reinterpret_cast<char*>(&scoreB_), sizeof(scoreB_));
        readFloats(scoreFc1W_);
        readFloats(scoreFc1B_);
        readFloats(scoreFc2W_);
        readFloats(scoreFc2B_);
    }
    if (!in) {
        clear();
        if (error) *error = "truncated model: " + path;
        return false;
    }
    loaded_ = true;
    modelName_ = std::filesystem::path(path).filename().u8string();
    return true;
}

void PolicyValueNet::buildFeaturePlanes(const std::array<std::int8_t, kCells>& board, int ply,
                                       float* out) const {
    const int planes = inputPlanes_;
    std::fill(out, out + static_cast<std::size_t>(planes) * kCells, 0.0f);
    std::array<std::array<int, 10>, 2> counts{};
    for (int cell = 0; cell < kCells; ++cell) {
        const int stone = board[cell];
        if (stone == 0) continue;
        const int side = stone > 0 ? 0 : 1;
        const int number = stone > 0 ? stone : -stone;
        out[(side * 9 + number - 1) * kCells + cell] = 1.0f;
        ++counts[side][number];
    }
    // Fixed layout: 0..17 stones, 18 side to move, 19..36 remaining hand counts.
    // The side plane belongs at 18; writing it at 36 (as an earlier version did)
    // silently replaced Black's count-of-9 plane with the side flag.
    const float sideValue = (ply & 1) ? 1.0f : 0.0f;
    if (planes >= kResInputPlanesV3) {
        for (int side = 0; side < 2; ++side) {
            for (int number = 1; number <= 9; ++number) {
                const int plane = 19 + side * 9 + (number - 1);
                const float value =
                    static_cast<float>(kPiecesPerNumber - counts[side][number]) / 3.0f;
                for (int cell = 0; cell < kCells; ++cell) {
                    out[static_cast<std::size_t>(plane) * kCells + cell] = value;
                }
            }
        }
    }
    for (int cell = 0; cell < kCells; ++cell) {
        out[static_cast<std::size_t>(18) * kCells + cell] = sideValue;
    }
}

void PolicyValueNet::buildGlobals(const std::array<std::int8_t, kCells>& board, int ply, int raw,
                                  float* out) const {
    std::array<std::array<int, 10>, 2> counts{};
    int empty = 0;
    for (int cell = 0; cell < kCells; ++cell) {
        const int stone = board[cell];
        if (stone == 0) {
            ++empty;
            continue;
        }
        const int side = stone > 0 ? 0 : 1;
        const int number = stone > 0 ? stone : -stone;
        ++counts[side][number];
    }
    for (int side = 0; side < 2; ++side) {
        for (int number = 1; number <= 9; ++number) {
            out[side * 9 + (number - 1)] =
                static_cast<float>(kPiecesPerNumber - counts[side][number]) / 3.0f;
        }
    }
    // A forced pass makes the turn count exceed the stone count, so clamp the
    // progress feature to the game length instead of letting it reach 1.02.
    const int cappedPly = ply > kTotalMoves ? kTotalMoves : ply;
    out[18] = (ply & 1) ? 1.0f : 0.0f;
    out[19] = static_cast<float>(raw) / 81.0f;
    out[20] = static_cast<float>(cappedPly) / 54.0f;
    out[21] = static_cast<float>(empty) / 81.0f;
}

NnOutput PolicyValueNet::evaluate(const Position& position) const {
    NnOutput out;
    if (!loaded_) return out;

    const int channels = channels_;
    const int planes = inputPlanes_;
    std::array<std::int8_t, kCells> board{};
    for (int cell = 0; cell < kCells; ++cell) {
        board[cell] = static_cast<std::int8_t>(position.stoneAt(cell));
    }
    thread_local std::vector<float> input;
    input.resize(static_cast<std::size_t>(planes) * kCells);
    buildFeaturePlanes(board, position.ply(), input.data());

    const std::size_t planeSize = static_cast<std::size_t>(channels) * kCells;
    // Scratch reused across calls: one evaluation used to allocate three
    // plane-sized vectors per convolution, i.e. dozens of allocations per call.
    thread_local std::vector<float> a, b, c;
    a.resize(planeSize);
    b.resize(planeSize);
    c.resize(planeSize);
    conv3x3(input.data(), planes, channels, stemW_.data(), stemB_.data(), a.data(), true);
    for (const Block& block : blocksData_) {
        residualBlock(a.data(), block, channels, b.data(), c.data());
        a.swap(c);
    }

    // Hand-conditioned policy and value features.
    std::array<float, kResGlobalFeatures> global{};
    buildGlobals(board, position.ply(), position.rawScore(), global.data());

    std::array<float, 9> handBias{};
    for (int number = 0; number < 9; ++number) {
        float sum = handB_[number];
        for (int g = 0; g < kResGlobalFeatures; ++g) {
            sum += handW_[static_cast<std::size_t>(number) * kResGlobalFeatures + g] * global[g];
        }
        handBias[number] = sum;
    }

    // 1x1 policy head: one logit for each (cell, number).
    for (int cell = 0; cell < kCells; ++cell) {
        for (int number = 0; number < 9; ++number) {
            float logit = policyB_[number] + handBias[number];
            for (int ch = 0; ch < channels; ++ch) {
                logit += a[static_cast<std::size_t>(ch) * kCells + cell] *
                         policyW_[static_cast<std::size_t>(number) * channels + ch];
            }
            out.policy[cell * 9 + number] = logit;
        }
    }

    // 1x1 value conv followed by global average, then a small global MLP.
    float valueSpatial = 0.0f;
    for (int cell = 0; cell < kCells; ++cell) {
        float sum = 0.0f;
        for (int ch = 0; ch < channels; ++ch) {
            sum += a[static_cast<std::size_t>(ch) * kCells + cell] * valueW_[ch];
        }
        valueSpatial += sum;
    }
    std::array<float, kHiddenInput> hiddenInput{};
    hiddenInput[0] = valueSpatial / static_cast<float>(kCells) + valueB_;
    for (int g = 0; g < kResGlobalFeatures; ++g) hiddenInput[1 + g] = global[g];
    std::array<float, kResValueHidden> hidden{};
    for (int h = 0; h < kResValueHidden; ++h) {
        float sum = valueFc1B_[h];
        for (int i = 0; i < kHiddenInput; ++i) {
            sum += valueFc1W_[static_cast<std::size_t>(h) * kHiddenInput + i] * hiddenInput[i];
        }
        hidden[h] = std::max(0.0f, sum);
    }
    float value = valueFc2B_[0];
    for (int h = 0; h < kResValueHidden; ++h) value += valueFc2W_[h] * hidden[h];
    out.value = 1.0f / (1.0f + std::exp(-value));

    if (hasScoreHead_) {
        float scoreSpatial = 0.0f;
        for (int cell = 0; cell < kCells; ++cell) {
            float sum = 0.0f;
            for (int ch = 0; ch < channels; ++ch) {
                sum += a[static_cast<std::size_t>(ch) * kCells + cell] * scoreW_[ch];
            }
            scoreSpatial += sum;
        }
        std::array<float, kHiddenInput> scoreInput{};
        scoreInput[0] = scoreSpatial / static_cast<float>(kCells) + scoreB_;
        for (int g = 0; g < kResGlobalFeatures; ++g) scoreInput[1 + g] = global[g];
        std::array<float, kResValueHidden> scoreHidden{};
        for (int h = 0; h < kResValueHidden; ++h) {
            float sum = scoreFc1B_[h];
            for (int i = 0; i < kHiddenInput; ++i) {
                sum += scoreFc1W_[static_cast<std::size_t>(h) * kHiddenInput + i] * scoreInput[i];
            }
            scoreHidden[h] = std::max(0.0f, sum);
        }
        float raw = scoreFc2B_[0];
        for (int h = 0; h < kResValueHidden; ++h) raw += scoreFc2W_[h] * scoreHidden[h];
        out.expectedScore = std::tanh(raw) * kScoreScale;
    }
    return out;
}

} // namespace shuyi
