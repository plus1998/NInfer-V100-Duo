#pragma once

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Fp8Geometry {
    static_assert(OutputRows > 0 && (OutputRows % 16) == 0);
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kOutputRows = OutputRows;
    static constexpr std::int32_t kInputRows  = InputRows;
};

template <std::int32_t InputRows>
struct Fp8ActivationGeometry {
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kInputRows = InputRows;
};

enum class Fp8CodeCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Fp8SmallTActivationAccess : std::uint8_t {
    TokenPacked,
    SharedPhase,
};

enum class Fp8SmallTBlockOrder : std::uint8_t {
    RowsContiguous,
    TokenTilesContiguous,
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Fp8CodeCache CodeCache, int PhaseUnroll, int MinBlocksPerSm>
struct Fp8GemvSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowsPerCta        = WarpsPerCta * RowsPerWarp;
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int TokenTile, int AccumulatorChains,
          Fp8SmallTActivationAccess ActivationAccess, Fp8CodeCache CodeCache, int PhaseUnroll,
          Fp8SmallTBlockOrder BlockOrder, int MinBlocksPerSm>
struct Fp8SmallTSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(TokenTile > 0);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kTokenTile         = TokenTile;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr auto kBlockOrder       = BlockOrder;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowsPerCta        = WarpsPerCta * RowsPerWarp;
};

enum class Fp8A16MmaActivationStage : std::uint8_t {
    ActiveOnly,
    PaddedZero,
};

enum class Fp8A16MmaCache : std::uint8_t {
    Default,
    Streaming,
};

template <int KWarps, int TileTokens, int MinBlocksPerSm,
          Fp8A16MmaCache ActivationCache           = Fp8A16MmaCache::Default,
          Fp8A16MmaCache WeightCache               = Fp8A16MmaCache::Streaming,
          Fp8A16MmaActivationStage ActivationStage = Fp8A16MmaActivationStage::ActiveOnly>
struct Fp8A16MmaSchedule {
    static_assert(KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(TileTokens == 8 || TileTokens == 16 || TileTokens == 24 || TileTokens == 32 ||
                  TileTokens == 40 || TileTokens == 48);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kKWarps            = KWarps;
    static constexpr int kTileTokens        = TileTokens;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = KWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / KWarps;
};

using Fp8AttnInputGeometry       = Fp8Geometry<14336, 5120>;
using Fp8GdnInputGeometry        = Fp8Geometry<16384, 5120>;
using Fp8MlpGateUpGeometry       = Fp8Geometry<34816, 5120>;
using Fp8VocabularyGeometry      = Fp8Geometry<248320, 5120>;
using Fp8Residual6144Geometry    = Fp8Geometry<5120, 6144>;
using Fp8Residual17408Geometry   = Fp8Geometry<5120, 17408>;
using Fp8Activation5120Geometry  = Fp8ActivationGeometry<5120>;
using Fp8Activation6144Geometry  = Fp8ActivationGeometry<6144>;
using Fp8Activation17408Geometry = Fp8ActivationGeometry<17408>;

// TP2 shard geometry. Only the vocabulary head is registered here, because it is the only FP8
// problem the GENERIC ops::linear serves at tp2: the other five FP8 geometries reach the device
// through the fused families (attention/query_key_gate_value -> attn_input_proj,
// gdn/query_key_value_z -> gdn_input_proj, mlp/gate_up -> linear_swiglu, attention/output,
// gdn/output and mlp/down -> linear_add), each of which owns its own registry under its own
// src/ops/<family>/fp8 directory and registers its own shards there. The output head splits
// column-parallel by vocabulary rows (248320 -> 124160), which is a whole number of the A16 MMA
// schedule's 16-row CTAs and leaves K untouched.
using Fp8VocabularyTp2ColumnGeometry = Fp8Geometry<124160, 5120>;

// gdn_input_proj's own tp2 column shard -- each device's own head-local half of the
// fused Q|K|V|Z object (16384 -> 8192). FP8 is not optional for this object: it is the flagship
// geometry, bound by bind_qwen38_nvfp4_text_layers's GDN input_projection. This is the first FP8
// shard geometry registered outside the generic ops::linear vocabulary head.
using Fp8GdnInputTp2ColumnGeometry = Fp8Geometry<8192, 5120>;

// linear_add's own tp2 ROW shards (o_proj / gdn/output 6144 -> 3072, mlp/down
// 17408 -> 8704). Same "shared header, family-owned kernel" split NVFP4 uses for its own
// Nvfp4Residual*Tp2RowGeometry -- also wired into ops::linear's OWN kernel-level dispatch
// (fp8_dispatch.cpp / fp8_gemv.cu / fp8_small_t.cu / fp8_a8.cu) so linear_add_row_parallel's plain
// (residual-free) rank can reach these shapes through ops::linear directly, exactly the way
// Nvfp4Residual*Tp2RowGeometry already serves NVFP4's own linear_add rank-1 path.
using Fp8Residual6144Tp2RowGeometry  = Fp8Geometry<5120, 3072>;
using Fp8Residual17408Tp2RowGeometry = Fp8Geometry<5120, 8704>;

// linear_swiglu's own tp2 COLUMN shard (mlp/gate_up 34816 -> 17408, Text layers
// 56-63 -- the FP8-bound MLP tail per the flagship profile).
using Fp8MlpGateUpTp2ColumnGeometry = Fp8Geometry<17408, 5120>;

// attn_input_proj's own tp2 COLUMN shard (attention/query_key_gate_value
// 14336 -> 7168 -- the fused attention input projection, bound FP8 for every full-attention layer
// per the flagship profile).
using Fp8AttnInputTp2ColumnGeometry = Fp8Geometry<7168, 5120>;

// Row-parallel activation-quantize geometries for the A8 route at the halved K extents
// linear_add's row shards introduce (mirrors NVFP4's own <3072>/<8704> activation geometries).
// Column shards (GdnInput/MlpGateUp/AttnInput Tp2Column) keep K=5120, already served by
// Fp8Activation5120Geometry.
using Fp8Activation3072Geometry = Fp8ActivationGeometry<3072>;
using Fp8Activation8704Geometry = Fp8ActivationGeometry<8704>;

inline constexpr std::int32_t kFp8VocabularyFirstA16MmaT = 1;
inline constexpr std::int32_t kFp8VocabularyLastA16MmaT  = 48;

template <int ActiveTokens>
struct Fp8VocabularyA16MmaProductionSchedule {
    static_assert(ActiveTokens >= kFp8VocabularyFirstA16MmaT);
    static_assert(ActiveTokens <= kFp8VocabularyLastA16MmaT);

    static constexpr int kTileTokens     = ActiveTokens <= 8    ? 8
                                           : ActiveTokens <= 16 ? 16
                                           : ActiveTokens <= 24 ? 24
                                           : ActiveTokens <= 32 ? 32
                                           : ActiveTokens <= 40 ? 40
                                                                : 48;
    static constexpr int kKWarps         = ActiveTokens <= 8 ? 16 : (ActiveTokens <= 24 ? 8 : 4);
    static constexpr int kMinBlocksPerSm = kKWarps == 16 ? 1 : 2;
    using Type                           = Fp8A16MmaSchedule<kKWarps, kTileTokens, kMinBlocksPerSm>;
};

enum class Fp8Problem : std::uint8_t {
    AttnInput,
    GdnInput,
    MlpGateUp,
    Vocabulary,
    Residual6144,
    Residual17408,
    // TP2 shard, appended so the existing values keep their encodings. Families that switch over
    // this enum without a `default:` fall through to their own "unsupported problem" throw for it,
    // which is correct for any family that has no split path at this shape.
    VocabularyTp2Column,
    // gdn_input_proj's own tp2 column shard, appended for the same reason.
    GdnInputTp2Column,
    // linear_add's own tp2 row shards, appended for the same reason.
    Residual6144Tp2Row,
    Residual17408Tp2Row,
    // linear_swiglu's own tp2 column shard, appended for the same reason.
    MlpGateUpTp2Column,
    // attn_input_proj's own tp2 column shard, appended for the same reason.
    AttnInputTp2Column,
};

inline constexpr bool is_fp8_linear_problem(std::int32_t output_rows, std::int32_t input_rows) {
    return (output_rows == Fp8AttnInputGeometry::kOutputRows &&
            input_rows == Fp8AttnInputGeometry::kInputRows) ||
           (output_rows == Fp8GdnInputGeometry::kOutputRows &&
            input_rows == Fp8GdnInputGeometry::kInputRows) ||
           (output_rows == Fp8MlpGateUpGeometry::kOutputRows &&
            input_rows == Fp8MlpGateUpGeometry::kInputRows) ||
           (output_rows == Fp8VocabularyGeometry::kOutputRows &&
            input_rows == Fp8VocabularyGeometry::kInputRows) ||
           (output_rows == Fp8Residual6144Geometry::kOutputRows &&
            input_rows == Fp8Residual6144Geometry::kInputRows) ||
           (output_rows == Fp8Residual17408Geometry::kOutputRows &&
            input_rows == Fp8Residual17408Geometry::kInputRows) ||
           (output_rows == Fp8VocabularyTp2ColumnGeometry::kOutputRows &&
            input_rows == Fp8VocabularyTp2ColumnGeometry::kInputRows) ||
           (output_rows == Fp8GdnInputTp2ColumnGeometry::kOutputRows &&
            input_rows == Fp8GdnInputTp2ColumnGeometry::kInputRows) ||
           (output_rows == Fp8Residual6144Tp2RowGeometry::kOutputRows &&
            input_rows == Fp8Residual6144Tp2RowGeometry::kInputRows) ||
           (output_rows == Fp8Residual17408Tp2RowGeometry::kOutputRows &&
            input_rows == Fp8Residual17408Tp2RowGeometry::kInputRows) ||
           (output_rows == Fp8MlpGateUpTp2ColumnGeometry::kOutputRows &&
            input_rows == Fp8MlpGateUpTp2ColumnGeometry::kInputRows) ||
           (output_rows == Fp8AttnInputTp2ColumnGeometry::kOutputRows &&
            input_rows == Fp8AttnInputTp2ColumnGeometry::kInputRows);
}

inline Fp8Problem resolve_fp8_problem(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Fp8AttnInputGeometry::kOutputRows &&
        input_rows == Fp8AttnInputGeometry::kInputRows) {
        return Fp8Problem::AttnInput;
    }
    if (output_rows == Fp8GdnInputGeometry::kOutputRows &&
        input_rows == Fp8GdnInputGeometry::kInputRows) {
        return Fp8Problem::GdnInput;
    }
    if (output_rows == Fp8MlpGateUpGeometry::kOutputRows &&
        input_rows == Fp8MlpGateUpGeometry::kInputRows) {
        return Fp8Problem::MlpGateUp;
    }
    if (output_rows == Fp8VocabularyGeometry::kOutputRows &&
        input_rows == Fp8VocabularyGeometry::kInputRows) {
        return Fp8Problem::Vocabulary;
    }
    if (output_rows == Fp8Residual6144Geometry::kOutputRows &&
        input_rows == Fp8Residual6144Geometry::kInputRows) {
        return Fp8Problem::Residual6144;
    }
    if (output_rows == Fp8Residual17408Geometry::kOutputRows &&
        input_rows == Fp8Residual17408Geometry::kInputRows) {
        return Fp8Problem::Residual17408;
    }
    if (output_rows == Fp8VocabularyTp2ColumnGeometry::kOutputRows &&
        input_rows == Fp8VocabularyTp2ColumnGeometry::kInputRows) {
        return Fp8Problem::VocabularyTp2Column;
    }
    if (output_rows == Fp8GdnInputTp2ColumnGeometry::kOutputRows &&
        input_rows == Fp8GdnInputTp2ColumnGeometry::kInputRows) {
        return Fp8Problem::GdnInputTp2Column;
    }
    if (output_rows == Fp8Residual6144Tp2RowGeometry::kOutputRows &&
        input_rows == Fp8Residual6144Tp2RowGeometry::kInputRows) {
        return Fp8Problem::Residual6144Tp2Row;
    }
    if (output_rows == Fp8Residual17408Tp2RowGeometry::kOutputRows &&
        input_rows == Fp8Residual17408Tp2RowGeometry::kInputRows) {
        return Fp8Problem::Residual17408Tp2Row;
    }
    if (output_rows == Fp8MlpGateUpTp2ColumnGeometry::kOutputRows &&
        input_rows == Fp8MlpGateUpTp2ColumnGeometry::kInputRows) {
        return Fp8Problem::MlpGateUpTp2Column;
    }
    if (output_rows == Fp8AttnInputTp2ColumnGeometry::kOutputRows &&
        input_rows == Fp8AttnInputTp2ColumnGeometry::kInputRows) {
        return Fp8Problem::AttnInputTp2Column;
    }
    throw std::invalid_argument("unsupported FP8 problem");
}

// True for the problems whose only route is the vocabulary A16 MMA path, at every policy and T.
inline constexpr bool is_fp8_vocabulary_problem(Fp8Problem problem) {
    return problem == Fp8Problem::Vocabulary || problem == Fp8Problem::VocabularyTp2Column;
}

template <class Geometry>
struct Fp8LinearDecodeProductionSchedule;

// RTX 5090 cold-cache winner for this exact problem. Each newly registered geometry supplies its
// own specialization so admission never silently inherits another problem's measured schedule.
template <>
struct Fp8LinearDecodeProductionSchedule<Fp8AttnInputGeometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8GdnInputGeometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

// The tp2 column shard inherits the parent's measured decode schedule.
template <>
struct Fp8LinearDecodeProductionSchedule<Fp8GdnInputTp2ColumnGeometry>
    : Fp8LinearDecodeProductionSchedule<Fp8GdnInputGeometry> {};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8MlpGateUpGeometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8Residual6144Geometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8Residual17408Geometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

// Every new tp2 shard inherits its parent's measured decode schedule: tuning is inherited from
// the tp1 parent family, never re-measured for the shard.
template <>
struct Fp8LinearDecodeProductionSchedule<Fp8Residual6144Tp2RowGeometry>
    : Fp8LinearDecodeProductionSchedule<Fp8Residual6144Geometry> {};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8Residual17408Tp2RowGeometry>
    : Fp8LinearDecodeProductionSchedule<Fp8Residual17408Geometry> {};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8MlpGateUpTp2ColumnGeometry>
    : Fp8LinearDecodeProductionSchedule<Fp8MlpGateUpGeometry> {};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8AttnInputTp2ColumnGeometry>
    : Fp8LinearDecodeProductionSchedule<Fp8AttnInputGeometry> {};

inline constexpr std::int32_t kFp8FirstSmallT = 2;
inline constexpr std::int32_t kFp8LastSmallT  = 24;

template <class Geometry>
inline constexpr std::int32_t kFp8LinearSmallTMax = 0;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8AttnInputGeometry> = 11;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8GdnInputGeometry> = 10;

// The tp2 column shard inherits the parent's measured small-T ceiling -- tuning is inherited from
// the tp1 parent family, never re-measured for the shard.
template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8GdnInputTp2ColumnGeometry> =
    kFp8LinearSmallTMax<Fp8GdnInputGeometry>;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8MlpGateUpGeometry> = 4;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8Residual6144Geometry> = kFp8LastSmallT;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8Residual17408Geometry> = kFp8LastSmallT;

// Every new tp2 shard inherits its parent's measured small-T ceiling.
template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8Residual6144Tp2RowGeometry> =
    kFp8LinearSmallTMax<Fp8Residual6144Geometry>;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8Residual17408Tp2RowGeometry> =
    kFp8LinearSmallTMax<Fp8Residual17408Geometry>;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8MlpGateUpTp2ColumnGeometry> =
    kFp8LinearSmallTMax<Fp8MlpGateUpGeometry>;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8AttnInputTp2ColumnGeometry> =
    kFp8LinearSmallTMax<Fp8AttnInputGeometry>;
#ifdef NINFER_VOLTA_BUILD
// Upstream serves the vocabulary head only through the A16 MMA kernel, which is ldmatrix-based
// (sm_75+) and has no SIMT sibling, so on Volta the head has no route at all. Register it on the
// SIMT decode/small-T families instead. The schedules are the ones every other geometry measured
// on the reference card: a starting point for this port, not a Volta measurement -- see
// the V100 implementation before treating these numbers as tuned.
template <>
struct Fp8LinearDecodeProductionSchedule<Fp8VocabularyGeometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8VocabularyGeometry> = kFp8LastSmallT;
#endif // NINFER_VOLTA_BUILD

inline std::int32_t fp8_linear_small_t_max(Fp8Problem problem) {
    switch (problem) {
    case Fp8Problem::AttnInput:
        return kFp8LinearSmallTMax<Fp8AttnInputGeometry>;
    case Fp8Problem::GdnInput:
        return kFp8LinearSmallTMax<Fp8GdnInputGeometry>;
    case Fp8Problem::MlpGateUp:
        return kFp8LinearSmallTMax<Fp8MlpGateUpGeometry>;
    case Fp8Problem::GdnInputTp2Column:
        return kFp8LinearSmallTMax<Fp8GdnInputTp2ColumnGeometry>;
    case Fp8Problem::Vocabulary:
    case Fp8Problem::VocabularyTp2Column:
#ifdef NINFER_VOLTA_BUILD
        return kFp8LinearSmallTMax<Fp8VocabularyGeometry>;
#else
        break;
#endif
    case Fp8Problem::Residual6144:
        return kFp8LinearSmallTMax<Fp8Residual6144Geometry>;
    case Fp8Problem::Residual17408:
        return kFp8LinearSmallTMax<Fp8Residual17408Geometry>;
    case Fp8Problem::Residual6144Tp2Row:
        return kFp8LinearSmallTMax<Fp8Residual6144Tp2RowGeometry>;
    case Fp8Problem::Residual17408Tp2Row:
        return kFp8LinearSmallTMax<Fp8Residual17408Tp2RowGeometry>;
    case Fp8Problem::MlpGateUpTp2Column:
    case Fp8Problem::AttnInputTp2Column:
        // Not routed through ops::linear's own kernel-level dispatch (linear_swiglu / attn_input_proj
        // own their own registries under src/ops/<family>/fp8) -- unreachable from here.
        break;
    }
    throw std::logic_error("FP8 vocabulary uses its A16 MMA route");
}

// RTX 5090 cold-cache winners for contiguous Linear output. Each geometry owns its measured
// schedule ranges; fused semantic Ops reuse the mainloop but retain independent route frontiers.
template <class Geometry, int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Geometry>);
    static constexpr int kValuesPerLane     = 16;
    static constexpr auto kActivationAccess = Fp8SmallTActivationAccess::TokenPacked;
    using Type =
        Fp8SmallTSchedule<8, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                          Fp8CodeCache::Default, 1, Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8AttnInputGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Fp8AttnInputGeometry>);
    static constexpr auto kActivationAccess = ActiveTokens >= 3 && ActiveTokens <= 4
                                                  ? Fp8SmallTActivationAccess::SharedPhase
                                                  : Fp8SmallTActivationAccess::TokenPacked;
    using Type =
        Fp8SmallTSchedule<8, 2, 16, ActiveTokens, 1, kActivationAccess, Fp8CodeCache::Default, 1,
                          Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8GdnInputGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Fp8GdnInputGeometry>);
    static constexpr int kValuesPerLane     = ActiveTokens >= 5 && ActiveTokens <= 6 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Fp8SmallTActivationAccess::SharedPhase
                                                  : Fp8SmallTActivationAccess::TokenPacked;
    using Type =
        Fp8SmallTSchedule<8, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                          Fp8CodeCache::Default, 1, Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

// The tp2 column shard inherits the parent's measured small-T schedule at every T.
template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8GdnInputTp2ColumnGeometry, ActiveTokens>
    : Fp8LinearSmallTProductionSchedule<Fp8GdnInputGeometry, ActiveTokens> {};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8MlpGateUpGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Fp8MlpGateUpGeometry>);
    static constexpr int kValuesPerLane     = ActiveTokens == 4 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 3
                                                  ? Fp8SmallTActivationAccess::SharedPhase
                                                  : Fp8SmallTActivationAccess::TokenPacked;
    using Type =
        Fp8SmallTSchedule<8, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                          Fp8CodeCache::Default, 1, Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8Residual6144Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Fp8Residual6144Geometry>);
    static constexpr int kValuesPerLane = ActiveTokens >= 20 && ActiveTokens <= 23 ? 8 : 16;
    static constexpr int kTokenTile     = ActiveTokens == 24 ? 12 : ActiveTokens;
    static constexpr auto kBlockOrder   = ActiveTokens == 24
                                              ? Fp8SmallTBlockOrder::TokenTilesContiguous
                                              : Fp8SmallTBlockOrder::RowsContiguous;

    using Type = Fp8SmallTSchedule<8, 2, kValuesPerLane, kTokenTile, 1,
                                   Fp8SmallTActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                   kBlockOrder, 1>;
};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8Residual17408Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Fp8Residual17408Geometry>);
    static constexpr int kValuesPerLane = ActiveTokens >= 18 ? 8 : 16;

    using Type = Fp8SmallTSchedule<8, 2, kValuesPerLane, ActiveTokens, 1,
                                   Fp8SmallTActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                   Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

// Every new tp2 shard inherits its parent's measured small-T schedule at every T: tuning is
// inherited from the tp1 parent family, never re-measured for the shard.
template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8Residual6144Tp2RowGeometry, ActiveTokens>
    : Fp8LinearSmallTProductionSchedule<Fp8Residual6144Geometry, ActiveTokens> {};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8Residual17408Tp2RowGeometry, ActiveTokens>
    : Fp8LinearSmallTProductionSchedule<Fp8Residual17408Geometry, ActiveTokens> {};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8MlpGateUpTp2ColumnGeometry, ActiveTokens>
    : Fp8LinearSmallTProductionSchedule<Fp8MlpGateUpGeometry, ActiveTokens> {};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8AttnInputTp2ColumnGeometry, ActiveTokens>
    : Fp8LinearSmallTProductionSchedule<Fp8AttnInputGeometry, ActiveTokens> {};

} // namespace ninfer::ops::detail
