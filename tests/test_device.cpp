#include "core/device.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_throws_device(int device_id) {
    try {
        ninfer::DeviceContext invalid(device_id);
    } catch (const std::runtime_error&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

int check_context(const ninfer::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.load_stream == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.sm() <= 0) {
        std::cerr << label << " sm is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
}

} // namespace

// ExecutionContext's own contract: distinct ids only, tp follows the list, and rank 0 is left
// current. The duplicate-id rejection is the one that has to be a regression test rather than a
// one-off manual check -- it guards a SILENT failure (two contexts on one GPU halve nothing and
// make the peer copies alias their own source), so nothing downstream would notice its removal.
int check_execution_context(int device_count) {
    int failures = 0;

    try {
        ninfer::ExecutionContext duplicate({0, 0});
        std::cerr << "ExecutionContext({0, 0}) did not reject the duplicate device id\n";
        ++failures;
    } catch (const std::runtime_error&) {}

    try {
        ninfer::ExecutionContext empty({});
        std::cerr << "ExecutionContext({}) did not reject an empty device list\n";
        ++failures;
    } catch (const std::runtime_error&) {}

    ninfer::ExecutionContext single({0});
    if (single.tp != 1 || !single.dev[0].has_value() || single.dev[1].has_value() ||
        single.primary().device != 0) {
        std::cerr << "ExecutionContext({0}) did not describe one device on rank 0\n";
        ++failures;
    }

    if (device_count >= 2) {
        ninfer::ExecutionContext pair({0, 1});
        if (pair.tp != 2 || !pair.dev[0].has_value() || !pair.dev[1].has_value() ||
            pair.dev[0]->device != 0 || pair.dev[1]->device != 1) {
            std::cerr << "ExecutionContext({0, 1}) did not describe two distinct devices\n";
            ++failures;
        }
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess || current != pair.dev[0]->device) {
            std::cerr << "ExecutionContext construction did not leave rank 0 current (current="
                      << current << ")\n";
            ++failures;
        }
    }
    return failures;
}

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;

    ninfer::DeviceContext ctx(0);
    if (ctx.device != 0) {
        ++failures;
        std::cerr << "ctx.device expected 0, got " << ctx.device << '\n';
    }
    failures += check_context(ctx, "ctx");
    ctx.synchronize();

    const cudaStream_t original_stream = ctx.stream;
    ninfer::DeviceContext moved(std::move(ctx));
    if (ctx.stream != nullptr || ctx.load_stream != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    failures += check_context(moved, "moved");

    failures += expect_throws_device(count);
    failures += check_execution_context(count);

    ninfer::CudaEventTimer timer(moved);
    timer.start();
    moved.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    return failures == 0 ? 0 : fail("device test failed");
}
