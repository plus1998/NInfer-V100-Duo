#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

struct DeviceContext {
    int device               = 0;
    cudaStream_t stream      = nullptr;
    cudaStream_t load_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    int sm() const noexcept;
    std::size_t total_vram() const noexcept;
    void synchronize() const;
};

// One process, up to two CUDA devices. dev[0..tp-1] hold constructed DeviceContext instances;
// the remaining slots stay empty. tp == 1 unless the caller opts into `--tp 2`, which runs the
// tensor-parallel program across both devices.
struct ExecutionContext {
    std::array<std::optional<DeviceContext>, 2> dev;
    int tp = 1;

    // device_ids.size() must be 1 or 2 and becomes tp. Every id is validated to exist by
    // DeviceContext's own constructor; when two ids are given they must additionally share the
    // same compute capability (sm major.minor), since nothing downstream can reconcile mismatched
    // architectures.
    explicit ExecutionContext(const std::vector<int>& device_ids);

    [[nodiscard]] DeviceContext& primary() noexcept { return *dev[0]; }
    [[nodiscard]] const DeviceContext& primary() const noexcept { return *dev[0]; }
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

} // namespace ninfer
