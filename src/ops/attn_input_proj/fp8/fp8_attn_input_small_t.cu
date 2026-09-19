#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_small_t.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);

template <class Geometry, class Output, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                  Tensor& v, cudaStream_t stream) {
    using Schedule = typename Fp8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    const Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
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

constexpr auto kLaunchers = make_launchers<Fp8AttnInputGeometry, Fp8AttentionInputOutput>(
    std::make_index_sequence<kFp8LinearSmallTMax<Fp8AttnInputGeometry> - kFp8FirstSmallT + 1>{});

// The tp2 column shard.
constexpr auto kLaunchersShard =
    make_launchers<Fp8AttnInputTp2ColumnGeometry,
                   Fp8AttentionInputShardOutput<Fp8AttnInputTp2ColumnGeometry>>(
        std::make_index_sequence<kFp8LinearSmallTMax<Fp8AttnInputTp2ColumnGeometry> -
                                 kFp8FirstSmallT + 1>{});

} // namespace

void fp8_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kFp8FirstSmallT || x.ne[1] > kFp8LinearSmallTMax<Fp8AttnInputGeometry>) {
        throw std::invalid_argument("fp8 attn_input_proj small-T: unsupported T");
    }
    kLaunchers[static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT)](x, weight, q, gate, k, v,
                                                                    stream);
}

void fp8_attn_input_small_t_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kFp8FirstSmallT || x.ne[1] > kFp8LinearSmallTMax<Fp8AttnInputTp2ColumnGeometry>) {
        throw std::invalid_argument("fp8 attn_input_proj column-parallel small-T: unsupported T");
    }
    kLaunchersShard[static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT)](x, weight, q, gate, k, v,
                                                                         stream);
}

} // namespace ninfer::ops::detail
