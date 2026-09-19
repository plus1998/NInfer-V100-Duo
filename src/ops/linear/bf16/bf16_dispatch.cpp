#include "ops/linear/bf16/bf16_dispatch.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    const bool supported_problem = (n == 14336 && k == 5120) || (n == 5120 && k == 6144);
    // TP2 shards of the two registered problems: attention/query_key_gate_value splits
    // column-parallel (14336 -> 7168) and attention/output row-parallel (6144 -> 3072). BF16's
    // decode and small-T launchers are exact-geometry (see bf16_gemv.cu / bf16_small_t.cu), so a
    // shard resolves to the runtime-dimension MMA launcher at every T rather than gaining a second
    // tuned kernel set. See q5_dispatch.cpp for the rules every family follows here.
    const bool tp2_shard = (n == 7168 && k == 5120) || (n == 5120 && k == 3072);
    if ((!supported_problem && !tp2_shard) || t <= 0) {
        throw std::invalid_argument("bf16 linear: unsupported shape or T");
    }
    if (tp2_shard) { return launch_bf16_mma; }
    if (t == 1) { return launch_bf16_decode; }
    const std::int32_t small_t_end =
        n == 5120 ? kBf16SmallTMaxTokens : kBf16LinearSmallTDispatchEnd;
    if (t <= small_t_end) { return launch_bf16_small_t; }
#ifdef NINFER_VOLTA_BUILD
    return launch_bf16_cutlass_sm70;
#else
    return launch_bf16_mma;
#endif
}

Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
        return select_bf16_a16_launch(n, k, t);
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("bf16 linear: unsupported policy");
}

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    const Bf16Launch launch = select_bf16_launch(weight.n, weight.k, x.ne[1], policy);
    launch(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
