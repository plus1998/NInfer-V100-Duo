#pragma once

#include "ninfer/ops/linear.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Several NVFP4 plans drive a sub-projection internally and choose the policy themselves rather
// than taking the caller's, so a target that asks for A16 cannot reach them. On Volta the A4
// route is a __trap() -- cvt.e2m1x2 is Blackwell hardware and always will be -- so those internal
// choices have to collapse to A16 here. See the V100 performance summary.
#ifdef NINFER_VOLTA_BUILD
inline constexpr LinearPolicy kNvfp4InternalPolicy = LinearPolicy::A16Only;
#else
inline constexpr LinearPolicy kNvfp4InternalPolicy = LinearPolicy::AllowA4;
#endif

enum class Nvfp4ScaleAccess : std::uint8_t {
    StagedRaw,
    Direct,
};

enum class Nvfp4CodeCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Nvfp4SmallTActivationAccess : std::uint8_t {
    PairStream,
    TokenPacked,
    SharedPhase,
};

enum class Nvfp4SmallTBlockOrder : std::uint8_t {
    RowsContiguous,
    TokenTilesContiguous,
};

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Nvfp4GemvGeometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 128) == 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kOutputRows       = OutputRows;
    static constexpr std::int32_t kInputRows        = InputRows;
    static constexpr std::int32_t kGroupsPerRow     = InputRows / 16;
    static constexpr std::int32_t kScaleTilesPerRow = InputRows / 64;
    static constexpr std::int32_t kCodeBytesPerRow  = InputRows / 2;
};

template <std::int32_t InputRows>
struct Nvfp4ActivationGeometry {
    static_assert(InputRows > 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kInputRows       = InputRows;
    static constexpr std::int32_t kGroupsPerRow    = InputRows / 16;
    static constexpr std::int32_t kCodeBytesPerRow = InputRows / 2;
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int MinBlocksPerSm>
struct Nvfp4GemvSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowsPerCta        = WarpsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane, int TokenTile,
          int AccumulatorChains, Nvfp4SmallTActivationAccess ActivationAccess,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int PhaseUnroll,
          Nvfp4SmallTBlockOrder BlockOrder, int MinBlocksPerSm>
struct Nvfp4SmallTSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(TokenTile > 0);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kTokenTile         = TokenTile;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr auto kBlockOrder       = BlockOrder;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

using Nvfp4AttnInputGeometry     = Nvfp4GemvGeometry<14336, 5120>;
using Nvfp4GdnInputGeometry      = Nvfp4GemvGeometry<16384, 5120>;
using Nvfp4MlpGateUpGeometry     = Nvfp4GemvGeometry<34816, 5120>;
using Nvfp4Residual6144Geometry  = Nvfp4GemvGeometry<5120, 6144>;
using Nvfp4Residual17408Geometry = Nvfp4GemvGeometry<5120, 17408>;

// --- TP2 shard geometries ---------------------------------------------------------------------
//
// One per registered geometry, on the axis the tp2 ShardPlan
// (src/targets/qwen3_6_27b/impl/load/bindings.h) splits that family on:
//
//   column-parallel (output/ROW split, N/2):  attention/query_key_gate_value 14336 -> 7168
//                                             gdn/query_key_value_z          16384 -> 8192
//                                             mlp/gate_up                    34816 -> 17408
//   row-parallel    (input/COLUMN split, K/2): attention/output, gdn/output    6144 ->  3072
//                                              mlp/down                       17408 ->  8704
//
// These are not new kernels. Each is the SAME kernel template instantiated at the shard's N or K,
// so a column shard's grid is exactly half the parent's along the output axis and a row shard's
// K loop is exactly half as long. Every value stays inside Nvfp4GemvGeometry's own asserts
// (N % 128 == 0, K % 64 == 0), which is the same 128-row / 64-column scale-tile arithmetic the
// loader's shard-boundary validators derive from -- an unsplittable boundary could not have
// produced a shard tensor in the first place.
using Nvfp4AttnInputTp2ColumnGeometry     = Nvfp4GemvGeometry<7168, 5120>;
using Nvfp4GdnInputTp2ColumnGeometry      = Nvfp4GemvGeometry<8192, 5120>;
using Nvfp4MlpGateUpTp2ColumnGeometry     = Nvfp4GemvGeometry<17408, 5120>;
using Nvfp4Residual6144Tp2RowGeometry     = Nvfp4GemvGeometry<5120, 3072>;
using Nvfp4Residual17408Tp2RowGeometry    = Nvfp4GemvGeometry<5120, 8704>;

using Nvfp4Activation5120Geometry  = Nvfp4ActivationGeometry<5120>;
using Nvfp4Activation6144Geometry  = Nvfp4ActivationGeometry<6144>;
using Nvfp4Activation17408Geometry = Nvfp4ActivationGeometry<17408>;
// Row-parallel activation halves: a rank quantizes only its own K block.
using Nvfp4Activation3072Geometry = Nvfp4ActivationGeometry<3072>;
using Nvfp4Activation8704Geometry = Nvfp4ActivationGeometry<8704>;

enum class Nvfp4Problem : std::uint8_t {
    AttnInput,
    GdnInput,
    MlpGateUp,
    Residual6144,
    Residual17408,
    // TP2 shards. Appended so the existing values keep their encodings. Families that do not yet
    // implement a split form (linear_add, linear_swiglu, attn_input_proj, gdn_input_proj) switch
    // over this enum without a `default:` and therefore fall through to their own "unsupported
    // problem" throw for these values, which is exactly the right behaviour until their own task
    // lands the split path.
    AttnInputTp2Column,
    GdnInputTp2Column,
    MlpGateUpTp2Column,
    Residual6144Tp2Row,
    Residual17408Tp2Row,
};

// The parent geometry a shard problem was split from, and the axis it was split on. Route
// selection and schedule choice are INHERITED from the parent rather than re-measured: halving N
// or K moves the measured crossovers somewhat, but inheriting keeps the split path's behaviour a
// pure function of the family it belongs to, and any re-tuning is a separate, measurable change.
inline constexpr Nvfp4Problem nvfp4_parent_problem(Nvfp4Problem problem) {
    switch (problem) {
    case Nvfp4Problem::AttnInputTp2Column:
        return Nvfp4Problem::AttnInput;
    case Nvfp4Problem::GdnInputTp2Column:
        return Nvfp4Problem::GdnInput;
    case Nvfp4Problem::MlpGateUpTp2Column:
        return Nvfp4Problem::MlpGateUp;
    case Nvfp4Problem::Residual6144Tp2Row:
        return Nvfp4Problem::Residual6144;
    case Nvfp4Problem::Residual17408Tp2Row:
        return Nvfp4Problem::Residual17408;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::Residual17408:
        break;
    }
    return problem;
}

// Compile-time form of nvfp4_parent_problem, for the schedule predicates inside the W4A4
// launchers. A shard geometry answers with the geometry it was split from; a tp1 geometry answers
// with itself, so every existing `is_same_v<Geometry, ...>` predicate keeps its exact meaning.
template <class Geometry>
struct Nvfp4ParentGeometry {
    using Type = Geometry;
};
template <>
struct Nvfp4ParentGeometry<Nvfp4AttnInputTp2ColumnGeometry> {
    using Type = Nvfp4AttnInputGeometry;
};
template <>
struct Nvfp4ParentGeometry<Nvfp4GdnInputTp2ColumnGeometry> {
    using Type = Nvfp4GdnInputGeometry;
};
template <>
struct Nvfp4ParentGeometry<Nvfp4MlpGateUpTp2ColumnGeometry> {
    using Type = Nvfp4MlpGateUpGeometry;
};
template <>
struct Nvfp4ParentGeometry<Nvfp4Residual6144Tp2RowGeometry> {
    using Type = Nvfp4Residual6144Geometry;
};
template <>
struct Nvfp4ParentGeometry<Nvfp4Residual17408Tp2RowGeometry> {
    using Type = Nvfp4Residual17408Geometry;
};

template <class Geometry>
using Nvfp4ParentGeometryType = typename Nvfp4ParentGeometry<Geometry>::Type;

// X-macro over the whole registry: every switch that must name each geometry uses it, so adding a
// problem cannot leave one launcher behind. Order is tp1 first, then the tp2 shards.
#define NINFER_NVFP4_LINEAR_PROBLEMS(X)                                                            \
    X(AttnInput, Nvfp4AttnInputGeometry)                                                           \
    X(GdnInput, Nvfp4GdnInputGeometry)                                                             \
    X(MlpGateUp, Nvfp4MlpGateUpGeometry)                                                           \
    X(Residual6144, Nvfp4Residual6144Geometry)                                                     \
    X(Residual17408, Nvfp4Residual17408Geometry)                                                   \
    X(AttnInputTp2Column, Nvfp4AttnInputTp2ColumnGeometry)                                         \
    X(GdnInputTp2Column, Nvfp4GdnInputTp2ColumnGeometry)                                           \
    X(MlpGateUpTp2Column, Nvfp4MlpGateUpTp2ColumnGeometry)                                         \
    X(Residual6144Tp2Row, Nvfp4Residual6144Tp2RowGeometry)                                         \
    X(Residual17408Tp2Row, Nvfp4Residual17408Tp2RowGeometry)

inline constexpr bool is_nvfp4_linear_problem(std::int32_t output_rows, std::int32_t input_rows) {
#define NINFER_NVFP4_MATCH(name, geometry)                                                         \
    if (output_rows == geometry::kOutputRows && input_rows == geometry::kInputRows) {              \
        return true;                                                                               \
    }
    NINFER_NVFP4_LINEAR_PROBLEMS(NINFER_NVFP4_MATCH)
#undef NINFER_NVFP4_MATCH
    return false;
}

inline Nvfp4Problem resolve_nvfp4_problem(std::int32_t output_rows, std::int32_t input_rows) {
#define NINFER_NVFP4_RESOLVE(name, geometry)                                                       \
    if (output_rows == geometry::kOutputRows && input_rows == geometry::kInputRows) {              \
        return Nvfp4Problem::name;                                                                 \
    }
    NINFER_NVFP4_LINEAR_PROBLEMS(NINFER_NVFP4_RESOLVE)
#undef NINFER_NVFP4_RESOLVE
    throw std::invalid_argument("unsupported NVFP4 problem");
}

// RTX 5090 cold-cache winner among the measured decode schedules.
template <class Geometry>
struct Nvfp4LinearDecodeProductionSchedule {
    using Type =
        Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
};

inline constexpr std::int32_t kNvfp4FirstSmallT = 2;
inline constexpr std::int32_t kNvfp4LastSmallT  = 32;

#ifdef NINFER_VOLTA_BUILD
// Volta (sm_70) cold winners, measured on all five registered geometries with a private
// schedule sweep -- the reference card's schedules collapse here: 17.8 GB/s at T=16 against
// 93.7 for the entry below, and 7.1 against 48.7 at T=24. One schedule covers every geometry
// because all five agreed on the same envelope within a few percent, so there is nothing for
// per-geometry specializations to say.
//
// The whole envelope is per-thread register footprint. Values-per-lane 8 rather than 16 is
// worth 2.1x at T=13 and 4.3x at T=32; with 16 values a 16-warp CTA cannot be resident at all
// past T=13, while with 8 it is the fastest choice at T=14..20. Accumulator chains were swept
// and never won: unlike the tensor-core routes this kernel is not dependency-bound.
template <class Geometry, int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens <= 3    ? 16
                                          : ActiveTokens <= 13 ? 4
                                          : ActiveTokens <= 20 ? 16
                                                               : 4;
    static constexpr int kValuesPerLane = ActiveTokens <= 3 ? 16 : 8;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

#else // NINFER_VOLTA_BUILD

// RTX 5090 cold-cache winners for contiguous Linear output. T=2..4 amortizes activation loads
// through shared staging; T=5..32 keeps one packed activation tile per warp. The warp-count changes
// are measured occupancy/register crossovers, not semantic frontiers.
template <class Geometry, int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens >= 17 ? 4 : (ActiveTokens >= 13 ? 16 : 8);
    static constexpr int kValuesPerLane = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// G1's wider N benefits from keeping four warps per CTA throughout the A16 policy boundary. Only
// T=2 amortizes activation traffic enough for shared staging to win.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4GdnInputGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = 4;
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens == 2
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// At N=5120, R1 needs the larger CTA only for the last three A16 token counts. The unoptimized
// A16-only tail keeps the established generic schedule.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual6144Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens <= 16 ? (ActiveTokens >= 14 ? 16 : 4) : 4;
    static constexpr int kValuesPerLane = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// R2's longer K moves the stable four-to-sixteen-warp crossover to T=8.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual17408Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = ActiveTokens <= 16 ? (ActiveTokens >= 8 ? 16 : 4) : 4;
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

#endif // NINFER_VOLTA_BUILD
// A tp2 shard inherits its parent's measured schedule. Attention-input and MLP gate-up shards need
// no entry because their parents use the generic template above, which the shard geometry also
// selects. The three below have a per-geometry specialization to inherit; without these the shard
// would silently fall back to the generic schedule, which is a different kernel from the one the
// family was tuned to.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4GdnInputTp2ColumnGeometry, ActiveTokens>
    : Nvfp4LinearSmallTProductionSchedule<Nvfp4GdnInputGeometry, ActiveTokens> {};

template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual6144Tp2RowGeometry, ActiveTokens>
    : Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual6144Geometry, ActiveTokens> {};

template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual17408Tp2RowGeometry, ActiveTokens>
    : Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual17408Geometry, ActiveTokens> {};

} // namespace ninfer::ops::detail
