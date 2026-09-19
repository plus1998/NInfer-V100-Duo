#pragma once

#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"

namespace ninfer::ops::detail {

template <class Geometry>
struct Fp8LinearA8ProductionSchedule;

template <>
struct Fp8LinearA8ProductionSchedule<Fp8AttnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8GdnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

// The tp2 column shard inherits the parent's measured A8 schedule (BlockRows=64 divides
// both 5120 and 3072, the shard's own qkv/z row counts).
template <>
struct Fp8LinearA8ProductionSchedule<Fp8GdnInputTp2ColumnGeometry>
    : Fp8LinearA8ProductionSchedule<Fp8GdnInputGeometry> {};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8MlpGateUpGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual6144Geometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual17408Geometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

// Every new tp2 shard inherits its parent's measured A8 schedule (BlockRows=128 divides
// every shard's own kOutputRows -- 5120 unchanged by a row split, 17408/7168 both multiples of
// 128 -- so the parent's schedule applies unmodified).
template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual6144Tp2RowGeometry>
    : Fp8LinearA8ProductionSchedule<Fp8Residual6144Geometry> {};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual17408Tp2RowGeometry>
    : Fp8LinearA8ProductionSchedule<Fp8Residual17408Geometry> {};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8MlpGateUpTp2ColumnGeometry>
    : Fp8LinearA8ProductionSchedule<Fp8MlpGateUpGeometry> {};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8AttnInputTp2ColumnGeometry>
    : Fp8LinearA8ProductionSchedule<Fp8AttnInputGeometry> {};

} // namespace ninfer::ops::detail
