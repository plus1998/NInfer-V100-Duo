#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kMtpAttnRows = 14336;
inline constexpr int kMtpQRows    = 6144;
inline constexpr int kMtpKvRows   = 1024;

// tp == 2 shard of the same object. `mtp/layer/attention/query_key_gate_value` is bound
// [14336,5120] and suffix-matches the ShardPlan's `attention/query_key_gate_value` branch
// (bindings.cpp), which splits each of the four sections by its own head count. Device r's shard is
// therefore the concatenation, in the parent's own Q | K | Gate | V order, of rank r's half of every
// section: Q [0,3072) | K [3072,3584) | Gate [3584,6656) | V [6656,7168) -- exactly the section
// layout include/ninfer/ops/attn_input_proj.h derives for the text layers' copy of the same
// object. The head counts halve with it: 12 query/gate heads and 2 key/value heads of 256 each.
inline constexpr int kMtpAttnRowsTp2 = 7168;
inline constexpr int kMtpQRowsTp2    = 3072;
inline constexpr int kMtpKvRowsTp2   = 512;
static_assert(kMtpAttnRowsTp2 == 2 * kMtpQRowsTp2 + 2 * kMtpKvRowsTp2);

__global__ void mtp_pack_fc_input_kernel(const __nv_bfloat16* embedding_norm,
                                         const __nv_bfloat16* hidden_norm, __nv_bfloat16* out,
                                         std::int32_t rows) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) { return; }

    const int token             = static_cast<int>(blockIdx.y);
    const std::int64_t in_idx   = static_cast<std::int64_t>(token) * rows + row;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * rows);
    out[out_base + row]         = embedding_norm[in_idx];
    out[out_base + rows + row]  = hidden_norm[in_idx];
}

// Templated on the row geometry so the tp1 instantiation (<14336,6144,1024>) is byte-identical to
// the hand-written form it replaces and the tp2 shard (<7168,3072,512>) is the same kernel at the
// halved extents -- the "a split geometry is not a new kernel" rule every split form here
// follows, applied to a pure index remap.
template <int AttnRows, int QRows, int KvRows>
__global__ void mtp_split_attn_in_kernel(const __nv_bfloat16* attn_in, __nv_bfloat16* q,
                                         __nv_bfloat16* k, __nv_bfloat16* gate, __nv_bfloat16* v,
                                         std::int32_t tokens) {
    static_assert(AttnRows == 2 * QRows + 2 * KvRows);
    const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(AttnRows) * tokens;
    if (idx >= n) { return; }

    const int row             = static_cast<int>(idx % AttnRows);
    const int token           = static_cast<int>(idx / AttnRows);
    const __nv_bfloat16 value = attn_in[idx];

    if (row < QRows) {
        q[static_cast<std::int64_t>(token) * QRows + row] = value;
        return;
    }
    if (row < QRows + KvRows) {
        const int local                                      = row - QRows;
        k[static_cast<std::int64_t>(token) * KvRows + local] = value;
        return;
    }
    if (row < QRows + KvRows + QRows) {
        const int local                                       = row - QRows - KvRows;
        gate[static_cast<std::int64_t>(token) * QRows + local] = value;
        return;
    }

    const int local                                     = row - QRows - KvRows - QRows;
    v[static_cast<std::int64_t>(token) * KvRows + local] = value;
}

} // namespace ninfer::ops
