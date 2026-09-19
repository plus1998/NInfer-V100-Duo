#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ float ggml_q4_k_value(const unsigned char* block, int i) {
    const auto* scales = block + 4;
    const int group = i >> 5;
    int scale;
    int minimum;
    if (group < 4) {
        scale = scales[group] & 63;
        minimum = scales[group + 4] & 63;
    } else {
        scale = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
    const unsigned char q = block[16 + (i >> 6) * 32 + (i & 31)];
    const int code = (q >> ((group & 1) * 4)) & 15;
    const float d = __half2float(*reinterpret_cast<const __half*>(block));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
    return (d * scale) * code - dmin * minimum;
}

__device__ __forceinline__ float ggml_q6_k_value(const unsigned char* block, int i) {
    const int half = i >> 7;
    const int within = i & 127;
    const int section = within >> 5;
    const int lane = within & 31;
    const int low_index = half * 64 + (section & 1) * 32 + lane;
    const int low = (block[low_index] >> ((section >> 1) * 4)) & 15;
    const int high = (block[128 + half * 32 + lane] >> (section * 2)) & 3;
    const int code = (low | (high << 4)) - 32;
    const int scale = reinterpret_cast<const signed char*>(block + 192)[i >> 4];
    const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
    return (d * scale) * code;
}

__device__ __forceinline__ float ggml_k_value(const unsigned char* rows,
                                             std::uint64_t descriptor, int column) {
    const bool q6 = (descriptor & 1) != 0;
    const int bytes = q6 ? 210 : 144;
    const unsigned char* block = rows + (descriptor >> 1) + (column >> 8) * bytes;
    return q6 ? ggml_q6_k_value(block, column & 255)
              : ggml_q4_k_value(block, column & 255);
}

} // namespace ninfer::ops::detail
