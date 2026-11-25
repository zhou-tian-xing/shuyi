// CUDA inference backend: the same network as src/nn.cpp, evaluated for a whole
// batch of positions per launch.  Self-play drives it tick by tick: each game
// descends one leaf, all leaves are evaluated together, all games expand and
// back up.
//
// The convolutions run as im2col + cuBLAS GEMM instead of a hand-written
// kernel.  The first version did one thread per (output channel, cell) with a
// 576-iteration inner loop, which measured 5-8% of the card's FP32 peak on
// b6c64 (17 ms for a *single* position) -- i.e. it wasted the GPU it was
// supposed to use.  cuBLAS is tuned per architecture and uses TF32 tensor cores
// on Ampere and newer, so the same code is fast on a rented 3090/4090 without
// another rewrite.
//
// Layout: activations are [channel][n] with n = sample * 81 + cell, which is
// exactly the column-major GEMM result of C = A^T B with ld = n.  The input
// planes arrive as [sample][plane][cell] and are transposed once per batch.
#include "shuyi/nn_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace shuyi {
namespace {

constexpr int kThreads = 128;   // power of two: the value reduction relies on it
constexpr int kBlock = 256;
constexpr std::size_t kColBudget = 32u << 20; // bytes of im2col scratch per chunk

__global__ void transposeInputKernel(const float* __restrict__ in, float* __restrict__ out,
                                     int planes, int n) {
    const long long total = static_cast<long long>(planes) * n;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int p = static_cast<int>(idx / n);
        const int nn = static_cast<int>(idx - static_cast<long long>(p) * n);
        const int sample = nn / kCells;
        const int cell = nn - sample * kCells;
        out[idx] = in[static_cast<std::size_t>(sample) * planes * kCells + p * kCells + cell];
    }
}

__global__ void im2colKernel(const float* __restrict__ x, float* __restrict__ col, int n,
                             int cin) {
    const int K = cin * 9;
    const long long total = static_cast<long long>(n) * K;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int nn = static_cast<int>(idx / K);
        const int k = static_cast<int>(idx - static_cast<long long>(nn) * K);
        const int ic = k / 9;
        const int off = k - ic * 9;
        const int y = nn % kCells / kSize + off / 3 - 1;
        const int xx = nn % kCells % kSize + off % 3 - 1;
        float value = 0.0f;
        if (y >= 0 && y < kSize && xx >= 0 && xx < kSize) {
            value = x[static_cast<std::size_t>(ic) * n + (nn - nn % kCells) + y * kSize + xx];
        }
        col[static_cast<std::size_t>(nn) * K + k] = value;
    }
}

__global__ void biasReluKernel(float* __restrict__ y, const float* __restrict__ bias, int cout,
                               int n, int relu) {
    const long long total = static_cast<long long>(cout) * n;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int oc = static_cast<int>(idx / n);
        float v = y[idx] + bias[oc];
        y[idx] = relu ? fmaxf(v, 0.0f) : v;
    }
}

__global__ void addReluKernel(float* __restrict__ a, const float* __restrict__ b, long long n,
                              int relu) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = a[i] + b[i];
    a[i] = relu ? fmaxf(v, 0.0f) : v;
}

__global__ void policyKernel(const float* __restrict__ a, const float* __restrict__ pw,
                             const float* __restrict__ pb, const float* __restrict__ handBias,
                             float* __restrict__ logits, int channels, int n) {
    const int b = blockIdx.x;
    const int number = blockIdx.y;
    const float base = pb[number] + handBias[b * 9 + number];
    float* lg = logits + static_cast<std::size_t>(b) * kNnPolicy;
    const int offset = b * kCells;
    for (int cell = threadIdx.x; cell < kCells; cell += blockDim.x) {
        float sum = base;
        for (int c = 0; c < channels; ++c) {
            sum += a[static_cast<std::size_t>(c) * n + offset + cell] *
                   pw[number * channels + c];
        }
        lg[cell * 9 + number] = sum;
    }
}

__global__ void valueSpatialKernel(const float* __restrict__ a, const float* __restrict__ vw,
                                   float* __restrict__ vs, int channels, int n) {
    __shared__ float partial[kThreads];
    const int b = blockIdx.x;
    const int offset = b * kCells;
    float sum = 0.0f;
    for (int cell = threadIdx.x; cell < kCells; cell += blockDim.x) {
        float t = 0.0f;
        for (int c = 0; c < channels; ++c) {
            t += a[static_cast<std::size_t>(c) * n + offset + cell] * vw[c];
        }
        sum += t;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) vs[b] = partial[0] / static_cast<float>(kCells);
}

void check(cudaError_t status, const char* what, std::string* error) {
    if (status == cudaSuccess) return;
    if (error) *error = std::string(what) + ": " + cudaGetErrorString(status);
}

// Temporary diagnostics: a label makes an async failure point at its kernel.
void syncCheck(const char* label) {
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "[cuda] after %s", label);
        std::fprintf(stderr, ": %s", cudaGetErrorString(status));
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    }
}

template <typename T>
void upload(T*& device, const std::vector<T>& host, std::string* error, const char* what) {
    if (host.empty()) {
        device = nullptr;
        return;
    }
    check(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(T)), what, error);
    if (!device) return;
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), what,
          error);
}

} // namespace

struct CudaPolicyValueNet::Impl {
    int planes = kResInputPlanes;
    int channels = kResChannels;
    int blocks = kResBlocks;
    // Width of the global vector of the loaded model: 22 for a v3 file, 24 for a
    // v4 file that sees A.  The host-side MLP weights are laid out with this
    // stride, so it cannot be a compile-time constant.
    int globals = kResGlobalFeatures;
    std::string name;
    bool ready = false;
    cublasHandle_t blas = nullptr;
    // One stream per instance.  Self-play runs several worker threads, each with
    // its own model instance; with a shared default stream every host->device
    // copy would serialize the whole device and the CPU half of a tick could
    // never overlap the other thread's kernels.
    cudaStream_t stream = nullptr;

    float* dStemW = nullptr;
    float* dStemB = nullptr;
    std::vector<float*> dW1, dB1, dW2, dB2;
    float* dPolicyW = nullptr;
    float* dPolicyB = nullptr;
    float* dHandW = nullptr;
    float* dHandB = nullptr;
    float* dValueW = nullptr;
    float* dValueBias = nullptr;
    float* dFc1W = nullptr;
    float* dFc1B = nullptr;
    float* dFc2W = nullptr;
    float* dFc2B = nullptr;
    // Score head (optional).  Same shape as the value head: a 1x1 spatial
    // reduction into 32 hidden units, then one linear output through tanh.
    float* dScoreW = nullptr;
    float* dScoreB = nullptr;
    float* dScoreFc1W = nullptr;
    float* dScoreFc1B = nullptr;
    float* dScoreFc2W = nullptr;
    float* dScoreFc2B = nullptr;
    bool hasScore = false;

    float* dIn = nullptr;
    float* dXin = nullptr;
    float* dCol = nullptr;
    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    float* dLogits = nullptr;
    float* dVsum = nullptr;
    float* dScoreSum = nullptr;
    float* dHandBias = nullptr;
    int capacity = 0;

    std::vector<float> hFc1W, hFc1B, hFc2W, hFc2B;
    std::vector<float> hHandW, hHandB;
    std::vector<float> hScoreFc1W, hScoreFc1B, hScoreFc2W, hScoreFc2B;
    float hValueBias = 0.0f;
    float hScoreB = 0.0f;

    // Pinned staging for the per-call copies.  Pageable `cudaMemcpy` drains the
    // device before it starts, which is exactly the overlap we are trying to buy
    // -- at 32 positions the staging buffer is 384 KB per call.
    float* hIn = nullptr;
    float* hHandBias = nullptr;
    float* hLogits = nullptr;
    float* hVsum = nullptr;
    float* hScoreSum = nullptr;

    void release() {
        if (blas) {
            cublasDestroy(blas);
            blas = nullptr;
        }
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        for (float** p : {&dStemW, &dStemB, &dPolicyW, &dPolicyB, &dHandW, &dHandB, &dValueW,
                          &dValueBias, &dFc1W, &dFc1B, &dFc2W, &dFc2B, &dIn, &dXin, &dCol, &dA, &dB,
                          &dC, &dLogits, &dVsum, &dScoreSum, &dHandBias, &dScoreW, &dScoreB,
                          &dScoreFc1W, &dScoreFc1B, &dScoreFc2W, &dScoreFc2B}) {
            if (*p) cudaFree(*p);
            *p = nullptr;
        }
        for (std::vector<float*>* list : {&dW1, &dB1, &dW2, &dB2}) {
            for (float*& p : *list) {
                if (p) cudaFree(p);
                p = nullptr;
            }
            list->clear();
        }
        for (float** p : {&hIn, &hHandBias, &hLogits, &hVsum, &hScoreSum}) {
            if (*p) cudaFreeHost(*p);
            *p = nullptr;
        }
        capacity = 0;
        ready = false;
    }

    int chunkFor() const {
        // The stem's im2col is [n][planes*9] and a block's is [n][channels*9], so
        // the scratch has to be sized for the wider of the two.
        const int widest = std::max(planes, channels);
        const std::size_t perSample =
            static_cast<std::size_t>(kCells) * widest * 9 * sizeof(float);
        const int byBudget = static_cast<int>(kColBudget / std::max<std::size_t>(1, perSample));
        return std::clamp(byBudget, 8, 512);
    }

    bool ensureCapacity(int batch, std::string* error) {
        if (batch <= capacity && dXin) return true;
        int next = capacity > 0 ? capacity : 32;
        while (next < batch) next *= 2;
        for (float** p : {&dIn, &dXin, &dCol, &dA, &dB, &dC, &dLogits, &dVsum, &dScoreSum,
                          &dHandBias}) {
            if (*p) cudaFree(*p);
            *p = nullptr;
        }
        const std::size_t inCount = static_cast<std::size_t>(next) * planes * kCells;
        const std::size_t act = static_cast<std::size_t>(next) * channels * kCells;
        const std::size_t col = static_cast<std::size_t>(chunkFor()) * kCells *
                                std::max(planes, channels) * 9;
        check(cudaMalloc(reinterpret_cast<void**>(&dIn), inCount * sizeof(float)), "cudaMalloc in",
              error);
        check(cudaMalloc(reinterpret_cast<void**>(&dXin), inCount * sizeof(float)), "cudaMalloc x",
              error);
        check(cudaMalloc(reinterpret_cast<void**>(&dCol), col * sizeof(float)), "cudaMalloc col",
              error);
        check(cudaMalloc(reinterpret_cast<void**>(&dA), act * sizeof(float)), "cudaMalloc A", error);
        check(cudaMalloc(reinterpret_cast<void**>(&dB), act * sizeof(float)), "cudaMalloc B", error);
        check(cudaMalloc(reinterpret_cast<void**>(&dC), act * sizeof(float)), "cudaMalloc C", error);
        check(cudaMalloc(reinterpret_cast<void**>(&dLogits),
                         static_cast<std::size_t>(next) * kNnPolicy * sizeof(float)),
              "cudaMalloc logits", error);
        check(cudaMalloc(reinterpret_cast<void**>(&dVsum), static_cast<std::size_t>(next) * sizeof(float)),
              "cudaMalloc vsum", error);
        check(cudaMalloc(reinterpret_cast<void**>(&dScoreSum),
                         static_cast<std::size_t>(next) * sizeof(float)),
              "cudaMalloc scoresum", error);
        check(cudaMalloc(reinterpret_cast<void**>(&dHandBias),
                         static_cast<std::size_t>(next) * 9 * sizeof(float)),
              "cudaMalloc handbias", error);
        if (!dIn || !dXin || !dCol || !dA || !dB || !dC || !dLogits || !dVsum || !dScoreSum ||
            !dHandBias) return false;
        // Pinned host staging, sized with the device buffers.  cudaHostAlloc is
        // the only way to keep `cudaMemcpyAsync` truly asynchronous.
        for (float** p : {&hIn, &hHandBias, &hLogits, &hVsum, &hScoreSum}) {
            if (*p) cudaFreeHost(*p);
            *p = nullptr;
        }
        check(cudaHostAlloc(reinterpret_cast<void**>(&hIn), inCount * sizeof(float),
                            cudaHostAllocDefault),
              "cudaHostAlloc in", error);
        check(cudaHostAlloc(reinterpret_cast<void**>(&hHandBias),
                            static_cast<std::size_t>(next) * 9 * sizeof(float),
                            cudaHostAllocDefault),
              "cudaHostAlloc handbias", error);
        check(cudaHostAlloc(reinterpret_cast<void**>(&hLogits),
                            static_cast<std::size_t>(next) * kNnPolicy * sizeof(float),
                            cudaHostAllocDefault),
              "cudaHostAlloc logits", error);
        check(cudaHostAlloc(reinterpret_cast<void**>(&hVsum),
                            static_cast<std::size_t>(next) * sizeof(float), cudaHostAllocDefault),
              "cudaHostAlloc vsum", error);
        check(cudaHostAlloc(reinterpret_cast<void**>(&hScoreSum),
                            static_cast<std::size_t>(next) * sizeof(float), cudaHostAllocDefault),
              "cudaHostAlloc scoresum", error);
        if (!hIn || !hHandBias || !hLogits || !hVsum || !hScoreSum) return false;
        capacity = next;
        return true;
    }

    // y[cout][n] = relu( W * im2col(x) + bias ) ; x is [cin][n], W is [cout][cin*9].
    void conv(const float* x, int cin, int cout, const float* w, const float* bias, float* y,
              int n, int relu) {
        if (!blas) return;
        const int K = cin * 9;
        const long long work = static_cast<long long>(n) * K;
        int grid = static_cast<int>(std::min<long long>((work + kBlock - 1) / kBlock, 4096));
        im2colKernel<<<grid, kBlock, 0, stream>>>(x, dCol, n, cin);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // C[n][cout] = A[K][n]^T * B[K][cout]; all three are the same memory as the
        // [cout][n] / [n][K] row-major buffers used by the kernels.
        const cublasStatus_t gs = cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, cout, K, &alpha,
                                              dCol, K, w, K, &beta, y, n);
        if (gs != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "gemm status %d (n=%d cout=%d K=%d)", static_cast<int>(gs), n,
                         cout, K);
            std::fprintf(stderr, "\n");
        }
        const long long total = static_cast<long long>(cout) * n;
        biasReluKernel<<<static_cast<int>(std::min<long long>((total + kBlock - 1) / kBlock, 4096)),
                         kBlock, 0, stream>>>(y, bias, cout, n, relu);
    }
};

CudaPolicyValueNet::CudaPolicyValueNet() : impl_(new Impl()) {}
CudaPolicyValueNet::~CudaPolicyValueNet() {
    if (impl_) impl_->release();
}
bool CudaPolicyValueNet::loaded() const { return impl_ && impl_->ready; }
int CudaPolicyValueNet::inputPlanes() const { return impl_ ? impl_->planes : 0; }
int CudaPolicyValueNet::channels() const { return impl_ ? impl_->channels : 0; }
int CudaPolicyValueNet::blocks() const { return impl_ ? impl_->blocks : 0; }
const std::string& CudaPolicyValueNet::deviceName() const {
    static const std::string empty;
    return impl_ ? impl_->name : empty;
}

bool CudaPolicyValueNet::available(std::string* reason) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        if (reason) *reason = "no CUDA device";
        return false;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        if (reason) *reason = "cannot query CUDA device";
        return false;
    }
    if (prop.major < 6) {
        if (reason) *reason = "CUDA compute capability below 6.0";
        return false;
    }
    return true;
}

bool CudaPolicyValueNet::load(const PolicyValueNet& net, std::string* error) {
    Impl& d = *impl_;
    if (error) error->clear();
    d.release();
    d.planes = net.modelInputPlanes();
    d.channels = net.channels();
    d.blocks = net.blocks();
    d.globals = net.globalFeatures();
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        if (error) *error = "no CUDA device";
        return false;
    }
    d.name = prop.name;

    if (cublasCreate(&d.blas) != CUBLAS_STATUS_SUCCESS) {
        if (error) *error = "cublasCreate failed";
        return false;
    }
    if (cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking) != cudaSuccess) {
        if (error) *error = "cudaStreamCreate failed";
        return false;
    }
    cublasSetStream(d.blas, d.stream);
    if (prop.major >= 8) {
        // Ampere+: FP32 GEMMs may use TF32 tensor cores.  Inference only, and the
        // check in shuyi_selfplay_gpu --check still passes at 1e-3.
        cublasSetMathMode(d.blas, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    upload(d.dStemW, net.stemWeight(), error, "stem weight");
    upload(d.dStemB, net.stemBias(), error, "stem bias");
    for (const auto& block : net.blockData()) {
        float* w1 = nullptr;
        float* b1 = nullptr;
        float* w2 = nullptr;
        float* b2 = nullptr;
        upload(w1, block.w1, error, "block w1");
        upload(b1, block.b1, error, "block b1");
        upload(w2, block.w2, error, "block w2");
        upload(b2, block.b2, error, "block b2");
        d.dW1.push_back(w1);
        d.dB1.push_back(b1);
        d.dW2.push_back(w2);
        d.dB2.push_back(b2);
    }
    upload(d.dPolicyW, net.policyWeight(), error, "policy weight");
    upload(d.dPolicyB, net.policyBias(), error, "policy bias");
    upload(d.dHandW, net.handWeight(), error, "hand weight");
    upload(d.dHandB, net.handBiasData(), error, "hand bias");
    d.hHandW = net.handWeight();
    d.hHandB = net.handBiasData();
    upload(d.dValueW, net.valueWeight(), error, "value weight");
    d.hValueBias = net.valueBias();
    d.hFc1W = net.valueFc1Weight();
    d.hFc1B = net.valueFc1Bias();
    d.hFc2W = net.valueFc2Weight();
    d.hFc2B = net.valueFc2Bias();
    upload(d.dFc1W, d.hFc1W, error, "fc1 weight");
    upload(d.dFc1B, d.hFc1B, error, "fc1 bias");
    upload(d.dFc2W, d.hFc2W, error, "fc2 weight");
    upload(d.dFc2B, d.hFc2B, error, "fc2 bias");
    d.hasScore = net.hasScoreHead();
    if (d.hasScore) {
        d.hScoreFc1W = net.scoreFc1Weight();
        d.hScoreFc1B = net.scoreFc1Bias();
        d.hScoreFc2W = net.scoreFc2Weight();
        d.hScoreFc2B = net.scoreFc2Bias();
        d.hScoreB = net.scoreBias();
        upload(d.dScoreW, net.scoreWeight(), error, "score weight");
        upload(d.dScoreB, std::vector<float>{net.scoreBias()}, error, "score bias");
        upload(d.dScoreFc1W, d.hScoreFc1W, error, "score fc1 weight");
        upload(d.dScoreFc1B, d.hScoreFc1B, error, "score fc1 bias");
        upload(d.dScoreFc2W, d.hScoreFc2W, error, "score fc2 weight");
        upload(d.dScoreFc2B, d.hScoreFc2B, error, "score fc2 bias");
    }
    if (error && !error->empty()) return false;
    d.ready = true;
    return true;
}

void CudaPolicyValueNet::evaluate(const float* x, const float* globals, int batch,
                                  float* logitsOut, float* valueOut, float* scoreOut) {
    Impl& d = *impl_;
    if (!d.ready || batch <= 0) return;
    std::string error;
    if (!d.ensureCapacity(batch, &error)) {
        std::fprintf(stderr, "cuda: %s\n", error.c_str());
        return;
    }
    const int channels = d.channels;
    const int planes = d.planes;

    // Hand-conditioned policy bias is cheap and stays on the host.
    const int globalsWidth = d.globals;
    for (int b = 0; b < batch; ++b) {
        for (int n = 0; n < 9; ++n) {
            float sum = d.hHandB[n];
            for (int g = 0; g < globalsWidth; ++g) {
                sum += d.hHandW[static_cast<std::size_t>(n) * globalsWidth + g] *
                       globals[static_cast<std::size_t>(b) * globalsWidth + g];
            }
            d.hHandBias[static_cast<std::size_t>(b) * 9 + n] = sum;
        }
    }
    const std::size_t planeFloats = static_cast<std::size_t>(batch) * planes * kCells;
    std::memcpy(d.hIn, x, planeFloats * sizeof(float));
    cudaMemcpyAsync(d.dHandBias, d.hHandBias, static_cast<std::size_t>(batch) * 9 * sizeof(float),
                    cudaMemcpyHostToDevice, d.stream);
    // The planes arrive as [sample][plane][cell]; the GEMM path wants
    // [plane][sample*81+cell], so upload once and transpose on the device.
    cudaMemcpyAsync(d.dIn, d.hIn, planeFloats * sizeof(float), cudaMemcpyHostToDevice, d.stream);

    const int chunk = d.chunkFor();
    for (int base = 0; base < batch; base += chunk) {
        const int m = std::min(chunk, batch - base);
        const int n = m * kCells;
        const long long inWork = static_cast<long long>(planes) * n;
        transposeInputKernel<<<static_cast<int>(std::min<long long>((inWork + kBlock - 1) / kBlock, 4096)),
                               kBlock, 0, d.stream>>>(d.dIn + static_cast<std::size_t>(base) * planes * kCells,
                                         d.dXin, planes, n);
        d.conv(d.dXin, planes, channels, d.dStemW, d.dStemB, d.dA, n, 1);
        for (int i = 0; i < d.blocks; ++i) {
            d.conv(d.dA, channels, channels, d.dW1[i], d.dB1[i], d.dB, n, 1);
            d.conv(d.dB, channels, channels, d.dW2[i], d.dB2[i], d.dC, n, 0);
            const long long total = static_cast<long long>(channels) * n;
            addReluKernel<<<static_cast<int>((total + kBlock - 1) / kBlock), kBlock, 0, d.stream>>>(
                d.dA, d.dC, total, 1);
        }
        dim3 policyGrid(m, 9);
        policyKernel<<<policyGrid, kThreads, 0, d.stream>>>(
            d.dA, d.dPolicyW, d.dPolicyB, d.dHandBias,
            d.dLogits + static_cast<std::size_t>(base) * kNnPolicy, channels, n);
        valueSpatialKernel<<<m, kThreads, 0, d.stream>>>(d.dA, d.dValueW,
                                                        d.dVsum + base, channels, n);
        if (d.hasScore) {
            valueSpatialKernel<<<m, kThreads, 0, d.stream>>>(d.dA, d.dScoreW,
                                                            d.dScoreSum + base, channels, n);
        }
    }

    cudaMemcpyAsync(d.hLogits, d.dLogits, static_cast<std::size_t>(batch) * kNnPolicy * sizeof(float),
                    cudaMemcpyDeviceToHost, d.stream);
    cudaMemcpyAsync(d.hVsum, d.dVsum, static_cast<std::size_t>(batch) * sizeof(float),
                    cudaMemcpyDeviceToHost, d.stream);
    if (d.hasScore && scoreOut) {
        cudaMemcpyAsync(d.hScoreSum, d.dScoreSum, static_cast<std::size_t>(batch) * sizeof(float),
                        cudaMemcpyDeviceToHost, d.stream);
    }
    if (const cudaError_t status = cudaStreamSynchronize(d.stream); status != cudaSuccess) {
        std::fprintf(stderr, "cuda forward failed: %s\n", cudaGetErrorString(status));
        return;
    }
    std::memcpy(logitsOut, d.hLogits,
                static_cast<std::size_t>(batch) * kNnPolicy * sizeof(float));
    const float* spatial = d.hVsum;

    for (int b = 0; b < batch; ++b) {
        float hidden[kResValueHidden];
        const std::size_t stride = 1 + static_cast<std::size_t>(globalsWidth);
        for (int h = 0; h < kResValueHidden; ++h) {
            float sum = d.hFc1B[h];
            sum += d.hFc1W[static_cast<std::size_t>(h) * stride] * (spatial[b] + d.hValueBias);
            for (int g = 0; g < globalsWidth; ++g) {
                sum += d.hFc1W[static_cast<std::size_t>(h) * stride + 1 + g] *
                       globals[static_cast<std::size_t>(b) * globalsWidth + g];
            }
            hidden[h] = std::fmax(sum, 0.0f);
        }
        float v = d.hFc2B[0];
        for (int h = 0; h < kResValueHidden; ++h) v += d.hFc2W[h] * hidden[h];
        valueOut[b] = 1.0f / (1.0f + std::exp(-v));
    }

    if (scoreOut && d.hasScore) {
        const float* scoreSpatial = d.hScoreSum;
        for (int b = 0; b < batch; ++b) {
            float hidden[kResValueHidden];
            const std::size_t stride = 1 + static_cast<std::size_t>(globalsWidth);
            for (int h = 0; h < kResValueHidden; ++h) {
                float sum = d.hScoreFc1B[h];
                sum += d.hScoreFc1W[static_cast<std::size_t>(h) * stride] *
                       (scoreSpatial[b] + d.hScoreB);
                for (int g = 0; g < globalsWidth; ++g) {
                    sum += d.hScoreFc1W[static_cast<std::size_t>(h) * stride + 1 + g] *
                           globals[static_cast<std::size_t>(b) * globalsWidth + g];
                }
                hidden[h] = std::fmax(sum, 0.0f);
            }
            float raw = d.hScoreFc2B[0];
            for (int h = 0; h < kResValueHidden; ++h) raw += d.hScoreFc2W[h] * hidden[h];
            scoreOut[b] = std::tanh(raw) * kScoreScale;
        }
    }
}

} // namespace shuyi
