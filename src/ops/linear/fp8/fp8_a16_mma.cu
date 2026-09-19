#include "ops/linear/fp8/fp8_launch.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a16_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Fp8VocabularyA16MmaProductionSchedule<ActiveTokens>::Type;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_a16_mma_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kFp8VocabularyFirstA16MmaT + static_cast<int>(Offsets)>...};
}

template <class Geometry>
const auto& launchers() {
    static constexpr auto kLaunchers = make_launchers<Geometry>(
        std::make_index_sequence<kFp8VocabularyLastA16MmaT - kFp8VocabularyFirstA16MmaT + 1>{});
    return kLaunchers;
}

} // namespace

void launch_fp8_vocabulary_a16_mma(const Tensor& x, const Weight& weight, Tensor& out,
                                   cudaStream_t stream) {
    if (x.ne[1] < kFp8VocabularyFirstA16MmaT || x.ne[1] > kFp8VocabularyLastA16MmaT) {
        throw std::invalid_argument("fp8 vocabulary A16 MMA: invalid exact problem");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kFp8VocabularyFirstA16MmaT);
    if (weight.n == Fp8VocabularyGeometry::kOutputRows &&
        weight.k == Fp8VocabularyGeometry::kInputRows) {
        launchers<Fp8VocabularyGeometry>()[index](x, weight, out, stream);
        return;
    }
    // TP2 column shard: the same kernel and the same measured per-T schedule instantiated at half
    // the vocabulary rows, so the grid is exactly half the parent's and K is untouched.
    if (weight.n == Fp8VocabularyTp2ColumnGeometry::kOutputRows &&
        weight.k == Fp8VocabularyTp2ColumnGeometry::kInputRows) {
        launchers<Fp8VocabularyTp2ColumnGeometry>()[index](x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("fp8 vocabulary A16 MMA: invalid exact problem");
}

} // namespace ninfer::ops::detail
