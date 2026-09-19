#pragma once

// The Gated DeltaNet Op's registered numerical criteria, in one place.
//
// op_check.h deliberately defines no cross-Op tolerance presets -- "each semantic Op owns the
// concrete criterion used by its suite". These are that Op's, extracted here when a second suite
// (tests/ops/test_gdn_headsplit.cpp, the two-device head-split parity suite) needed the identical
// numbers that tests/ops/test_gated_delta_net.cpp had measured: two copies of a fitted constant
// silently drift apart, and a head-split shard is only meaningfully qualified if it is judged by
// the SAME contract as the geometry it splits, not by a transcription of it.
//
// Both are distances to gdn_ref.h's exact FP64 recurrence, evaluated over a whole output block.
// They were fitted against test_gated_delta_net.cpp's own fixture at T <= 128; the recurrence
// accumulates BF16 error with its length, so applying them far outside that range measures the
// length rather than the implementation. Measured on that fixture, the `gross_*` arm sits just
// below one BF16 ulp for narrow-dynamic-range blocks (at T=128 the worst element is 0.0170898 vs
// 0.0172229 -- adjacent BF16 values, spacing 1.22e-4 -- against a limit of 5.5e-3 * 0.0223 =
// 1.23e-4), so it is mildly fixture-sensitive: the tp1 geometry misses it by the same margin a
// head-split shard does, which is why test_gdn_headsplit.cpp gates on `relative_l2` plus a
// shard-vs-tp1 comparison on byte-identical data rather than on the gross arm.

#include "ops/op_check.h"

namespace ninfer::test {

// BF16 output of the recurrence, promoted and compared against the FP64 ideal.
constexpr ReductionCriterion gated_delta_net_output_bf16_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/5.5e-3};
}

// FP32 published state after the last token.
constexpr ReductionCriterion gated_delta_net_state_fp32_criterion() {
    return {/*relative_l2=*/2.7e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/3.9e-3};
}

} // namespace ninfer::test
