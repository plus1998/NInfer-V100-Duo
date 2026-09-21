#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    const bool a16_only = argc == 2 && std::string_view(argv[1]) == "--a16-only";
    if (argc != 1 && !a16_only) {
        std::cerr << "usage: ninfer_linear_swiglu_nvfp4_test [--a16-only]\n";
        return 2;
    }

    try {
        constexpr std::array<std::int32_t, 4> kA16Cases{1, 4, 8, 16};
        constexpr std::array<std::int32_t, 5> kA4Cases{5, 48, 49, 128, 1024};
        int failures = 0;
        bool skipped = false;
        const auto record = [&](int result) {
            if (result == 77) {
                skipped = true;
            } else {
                failures += result;
            }
        };
        record(run_profile("LinearSwiGLU NVFP4_A16",
                           {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                           kA16Cases));
        if (!a16_only) {
            record(run_profile("LinearSwiGLU NVFP4_A4",
                               {QType::NVFP4, 34816, 5120, 17408, 1803U,
                                ActivationCompute::A4},
                               kA4Cases));
        }
        if (failures == 0 && skipped) { return 77; }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
