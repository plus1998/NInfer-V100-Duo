#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_small_t.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

template <class Geometry, class Output, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                  cudaStream_t stream) {
    using Schedule = typename Fp8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    fp8_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Output, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, Output, kFp8FirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers<Fp8GdnInputGeometry, Fp8GdnInputOutput>(
    std::make_index_sequence<kFp8LinearSmallTMax<Fp8GdnInputGeometry> - kFp8FirstSmallT + 1>{});

constexpr auto kLaunchersShard =
    make_launchers<Fp8GdnInputTp2ColumnGeometry,
                   Fp8GdnInputShardOutput<Fp8GdnInputTp2ColumnGeometry>>(
        std::make_index_sequence<kFp8LinearSmallTMax<Fp8GdnInputTp2ColumnGeometry> -
                                 kFp8FirstSmallT + 1>{});

} // namespace

void fp8_gdn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  cudaStream_t stream) {
    if (x.ne[1] < kFp8FirstSmallT || x.ne[1] > kFp8LinearSmallTMax<Fp8GdnInputGeometry>) {
        throw std::invalid_argument("fp8 gdn_input_proj small-T: unsupported T");
    }
    kLaunchers[static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT)](x, weight, qkv, z, stream);
}

// The tp2 column shard.
void fp8_gdn_input_small_t_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                        Tensor& z, cudaStream_t stream) {
    if (x.ne[1] < kFp8FirstSmallT || x.ne[1] > kFp8LinearSmallTMax<Fp8GdnInputTp2ColumnGeometry>) {
        throw std::invalid_argument("fp8 gdn_input_proj column-parallel small-T: unsupported T");
    }
    kLaunchersShard[static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT)](x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
