#include "core/device.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

ExecutionContext::ExecutionContext(const std::vector<int>& device_ids) {
    if (device_ids.empty() || device_ids.size() > dev.size()) {
        throw std::runtime_error("ExecutionContext requires 1 or 2 device ids, got " +
                                 std::to_string(device_ids.size()));
    }
    tp = static_cast<int>(device_ids.size());
    // Distinct ids are a correctness precondition, not a preference: every tensor-parallel op
    // pairs `dev[0]` with `dev[1]` and assumes the two hold DIFFERENT shards on DIFFERENT devices.
    // `--devices 0,0` would build two CUDA contexts on one GPU, halve nothing, and make the peer
    // copies alias their own source -- silently wrong rather than slow.
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        for (std::size_t j = i + 1; j < device_ids.size(); ++j) {
            if (device_ids[i] == device_ids[j]) {
                throw std::runtime_error(
                    "ExecutionContext requires distinct device ids, got device " +
                    std::to_string(device_ids[i]) + " twice");
            }
        }
    }
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        dev[i].emplace(device_ids[i]); // validates existence internally
    }
    if (tp == 2) {
        const cudaDeviceProp& p0 = dev[0]->props;
        const cudaDeviceProp& p1 = dev[1]->props;
        if (p0.major != p1.major || p0.minor != p1.minor) {
            throw std::runtime_error(
                "ExecutionContext requires all devices to share the same compute capability "
                "(device " +
                std::to_string(dev[0]->device) + " is sm_" + std::to_string(p0.major) +
                std::to_string(p0.minor) + ", device " + std::to_string(dev[1]->device) +
                " is sm_" + std::to_string(p1.major) + std::to_string(p1.minor) + ")");
        }
    }
    // POSTCONDITION: rank 0 is the current device. Constructing the contexts in order leaves the
    // LAST one current, so at tp == 2 an ExecutionContext would hand its caller a thread bound to
    // device 1 -- and every caller that then issues work without naming a device (the whole tp1
    // code path, and rank-0-only steps like sampling) would silently target the wrong GPU. Today
    // materialization happens to reset it; depending on that is depending on an accident.
    cuda_check(cudaSetDevice(dev[0]->device), "cudaSetDevice(dev[0])", __FILE__, __LINE__);
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid CUDA device id"); }

    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }

    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(load_stream) failed", err));
    }

    stream      = compute;
    load_stream = load;
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || load_stream != nullptr) {
        log_cuda_error("cudaSetDevice", cudaSetDevice(device));
    }
    destroy_stream(load_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), load_stream(other.load_stream),
      props(other.props) {
    other.stream      = nullptr;
    other.load_stream = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream != nullptr || load_stream != nullptr) {
        log_cuda_error("cudaSetDevice", cudaSetDevice(device));
    }
    destroy_stream(load_stream);
    destroy_stream(stream);

    device      = other.device;
    props       = other.props;
    stream      = other.stream;
    load_stream = other.load_stream;

    other.stream      = nullptr;
    other.load_stream = nullptr;
    return *this;
}

int DeviceContext::sm() const noexcept { return props.major * 10 + props.minor; }

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream)); }

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : stream_(ctx.stream) {
    cudaError_t err = cudaSetDevice(ctx.device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice(timer) failed", err));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    err               = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

} // namespace ninfer
