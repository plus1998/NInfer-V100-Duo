#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/launcher/kernel_attr_once.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using TmaM256N128   = Nvfp4W4a4TmaSchedule<256, 3, 1>;
using TmaM256N128S2 = Nvfp4W4a4TmaSchedule<256, 2, 1>;

// QueryRows/KeyRows describe one fused Q|K|Gate|V object's section layout (Gate mirrors Query,
// Value mirrors Key -- see include/ninfer/ops/attn_input_proj.h for the section layout). Tp1 uses
// <6144,1024>; the tp2 column shard (each device's own head-local half) uses <3072,512>.
template <std::int32_t QueryRows, std::int32_t KeyRows>
struct AttentionOutput {
    static constexpr std::int32_t kGateRows   = QueryRows;
    static constexpr std::int32_t kKeyBegin   = QueryRows;
    static constexpr std::int32_t kGateBegin  = kKeyBegin + KeyRows;
    static constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * QueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * KeyRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * KeyRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

using AttentionOutputTp1    = AttentionOutput<6144, 1024>;
using AttentionOutputShard  = AttentionOutput<3072, 512>;

static_assert((6144 % TmaM256N128::kBlockN) == 0);
static_assert((1024 % TmaM256N128::kBlockN) == 0);
static_assert((3072 % TmaM256N128::kBlockN) == 0);
static_assert((512 % TmaM256N128::kBlockN) == 0);
// gdn_input_proj's own section widths (key_dim tp1/shard, value_dim tp1/shard).
static_assert((2048 % TmaM256N128::kBlockN) == 0);
static_assert((1024 % TmaM256N128::kBlockN) == 0);
static_assert((6144 % TmaM256N128::kBlockN) == 0);
static_assert((3072 % TmaM256N128::kBlockN) == 0);

template <class Geometry, class Schedule, class Epilogue, class Output>
void launch_tma(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                std::int32_t tokens, float alpha, Epilogue epilogue, Output output,
                cudaStream_t stream) {
    const Nvfp4W4a4TmaDescriptors descriptors =
        make_nvfp4_w4a4_tma_descriptors<Geometry, Schedule::kBlockM>(
            activation_codes, activation_scales, weight_codes, weight_scales, tokens);
    constexpr std::size_t kSharedBytes = sizeof(Nvfp4W4a4TmaSharedStorage<Schedule>);
    ensure_func_attr_per_device(nvfp4_w4a4_tma_kernel<Geometry, Schedule, Epilogue, Output>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(kSharedBytes));

    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN, tokens / Schedule::kBlockM);
    nvfp4_w4a4_tma_kernel<Geometry, Schedule>
        <<<grid, Schedule::kThreads, kSharedBytes, stream>>>(descriptors, alpha, epilogue, output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_linear(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                   const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                   __nv_bfloat16* output, std::int32_t tokens, float alpha, cudaStream_t stream) {
    launch_tma<Geometry, TmaM256N128>(activation_codes, activation_scales, weight_codes,
                                      weight_scales, tokens, alpha, Nvfp4IdentityEpilogue{},
                                      Nvfp4ContiguousOutput{output, Geometry::kOutputRows}, stream);
}

} // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                  const std::uint8_t* activation_scales,
                                  const std::uint8_t* weight_codes,
                                  const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                  std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4Problem::AttnInput:
        launch_linear<Nvfp4AttnInputGeometry>(activation_codes, activation_scales, weight_codes,
                                              weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4Problem::GdnInput:
        launch_linear<Nvfp4GdnInputGeometry>(activation_codes, activation_scales, weight_codes,
                                             weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launch_tma<Nvfp4MlpGateUpGeometry, TmaM256N128S2>(
            activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
            Nvfp4IdentityEpilogue{},
            Nvfp4ContiguousOutput{output, Nvfp4MlpGateUpGeometry::kOutputRows}, stream);
        return;
    case Nvfp4Problem::Residual6144:
        launch_linear<Nvfp4Residual6144Geometry>(activation_codes, activation_scales, weight_codes,
                                                 weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_linear<Nvfp4Residual17408Geometry>(activation_codes, activation_scales, weight_codes,
                                                  weight_scales, output, tokens, alpha, stream);
        return;
    // TP2 shards inherit their parent's TMA schedule: gate-up keeps the two-stage variant, the
    // rest keep the default.
    case Nvfp4Problem::AttnInputTp2Column:
        launch_linear<Nvfp4AttnInputTp2ColumnGeometry>(activation_codes, activation_scales,
                                                       weight_codes, weight_scales, output, tokens,
                                                       alpha, stream);
        return;
    case Nvfp4Problem::GdnInputTp2Column:
        launch_linear<Nvfp4GdnInputTp2ColumnGeometry>(activation_codes, activation_scales,
                                                      weight_codes, weight_scales, output, tokens,
                                                      alpha, stream);
        return;
    case Nvfp4Problem::MlpGateUpTp2Column:
        launch_tma<Nvfp4MlpGateUpTp2ColumnGeometry, TmaM256N128S2>(
            activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
            Nvfp4IdentityEpilogue{},
            Nvfp4ContiguousOutput{output, Nvfp4MlpGateUpTp2ColumnGeometry::kOutputRows}, stream);
        return;
    case Nvfp4Problem::Residual6144Tp2Row:
        launch_linear<Nvfp4Residual6144Tp2RowGeometry>(activation_codes, activation_scales,
                                                       weight_codes, weight_scales, output, tokens,
                                                       alpha, stream);
        return;
    case Nvfp4Problem::Residual17408Tp2Row:
        launch_linear<Nvfp4Residual17408Tp2RowGeometry>(activation_codes, activation_scales,
                                                        weight_codes, weight_scales, output, tokens,
                                                        alpha, stream);
        return;
    }
    // Unlike the launchers in this family that are generated from NINFER_NVFP4_LINEAR_PROBLEMS,
    // this switch is hand-maintained (each problem picks its own TMA schedule), so a newly
    // registered problem CAN be left behind here. Falling off the end of a void function is
    // undefined behaviour and, in practice, a silent no-op; this makes it a loud one.
    throw std::invalid_argument("nvfp4 W4A4 TMA linear: unsupported problem");
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t* activation_codes,
                                     const std::uint8_t* activation_scales,
                                     const std::uint8_t* weight_codes,
                                     const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                     __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                     std::int32_t tokens, float alpha, cudaStream_t stream) {
    launch_tma<Nvfp4AttnInputGeometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4IdentityEpilogue{}, AttentionOutputTp1{query, key, gate, value}, stream);
}

// The tp2 column-shard sibling, instantiated at Nvfp4AttnInputTp2ColumnGeometry
// ([7168,5120], each device's own head-local 3072 query | 512 key | 3072 gate | 512 value).
void launch_nvfp4_w4a4_tma_attention_shard(const std::uint8_t* activation_codes,
                                           const std::uint8_t* activation_scales,
                                           const std::uint8_t* weight_codes,
                                           const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                           __nv_bfloat16* gate, __nv_bfloat16* key,
                                           __nv_bfloat16* value, std::int32_t tokens, float alpha,
                                           cudaStream_t stream) {
    launch_tma<Nvfp4AttnInputTp2ColumnGeometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4IdentityEpilogue{}, AttentionOutputShard{query, key, gate, value}, stream);
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t* activation_codes,
                               const std::uint8_t* activation_scales,
                               const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                               __nv_bfloat16* qkv, __nv_bfloat16* z, std::int32_t tokens,
                               float alpha, cudaStream_t stream) {
    launch_tma<Nvfp4GdnInputGeometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4IdentityEpilogue{}, Nvfp4GdnInputOutput{qkv, z}, stream);
}

// The tp2 column-shard sibling, instantiated at Nvfp4GdnInputTp2ColumnGeometry
// ([8192,5120], each device's own head-local half: qkv[5120,T] packs 1024 query | 1024 key | 3072
// value, z[3072,T]). kBlockN=128 divides every section (1024%128==0, 3072%128==0), same tile
// alignment already proven for the tp1 parent's 2048/6144.
void launch_nvfp4_w4a4_tma_gdn_shard(const std::uint8_t* activation_codes,
                                     const std::uint8_t* activation_scales,
                                     const std::uint8_t* weight_codes,
                                     const std::uint8_t* weight_scales, __nv_bfloat16* qkv,
                                     __nv_bfloat16* z, std::int32_t tokens, float alpha,
                                     cudaStream_t stream) {
    launch_tma<Nvfp4GdnInputTp2ColumnGeometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4IdentityEpilogue{},
        Nvfp4GdnInputShardOutput<Nvfp4GdnInputTp2ColumnGeometry>{qkv, z}, stream);
}

template <class Geometry>
void launch_linear_add(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                       const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                       __nv_bfloat16* residual, std::int32_t tokens, float alpha,
                       cudaStream_t stream) {
    launch_tma<Geometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4AddResidualEpilogue{residual, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{residual, Geometry::kOutputRows}, stream);
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __nv_bfloat16* residual,
                                      std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4Problem::Residual6144:
        launch_linear_add<Nvfp4Residual6144Geometry>(activation_codes, activation_scales,
                                                     weight_codes, weight_scales, residual, tokens,
                                                     alpha, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_linear_add<Nvfp4Residual17408Geometry>(activation_codes, activation_scales,
                                                      weight_codes, weight_scales, residual, tokens,
                                                      alpha, stream);
        return;
    // The tp2 row-parallel halves of the two residual geometries. Each device's rank
    // fuses its own K/2 partial with its OWN resident copy of the (replicated) residual in this one
    // TMA epilogue pass -- exactly the same fused kernel as the tp1 geometry above, instantiated at
    // the halved K -- and `ops::linear_add_row_parallel` (include/ninfer/ops/linear_add.h) reaches
    // this launcher on rank 0 only; rank 1 computes a residual-free partial through the plain
    // `launch_nvfp4_w4a4_tma_linear` above so the residual is added exactly once, before the
    // allreduce that combines the two ranks. See that Op's contract for the full design.
    case Nvfp4Problem::Residual6144Tp2Row:
        launch_linear_add<Nvfp4Residual6144Tp2RowGeometry>(activation_codes, activation_scales,
                                                           weight_codes, weight_scales, residual,
                                                           tokens, alpha, stream);
        return;
    case Nvfp4Problem::Residual17408Tp2Row:
        launch_linear_add<Nvfp4Residual17408Tp2RowGeometry>(activation_codes, activation_scales,
                                                            weight_codes, weight_scales, residual,
                                                            tokens, alpha, stream);
        return;
    // linear_add is defined only for the residual geometries (tp1 and their tp2 row-parallel
    // halves). Everything else -- the input projections and their tp2 column shards -- is a caller
    // error and must SAY so. These previously fell through a bare `return`, i.e. a silent no-op
    // that left the residual untouched; the trailing throw below is what makes an unhandled
    // problem impossible to mistake for success.
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
    case Nvfp4Problem::AttnInputTp2Column:
    case Nvfp4Problem::GdnInputTp2Column:
    case Nvfp4Problem::MlpGateUpTp2Column:
        break;
    }
    throw std::invalid_argument("nvfp4 W4A4 TMA linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
