#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Most columns one GGUF vector pass handles. Wider inputs take the FP16 tensor-core route when
// the caller provides enough workspace, else successive vector passes.
inline constexpr std::int32_t kGgufVectorTokens = 8;

// Projects BF16 x [K,T] through a GGUF-blocks weight [N,K] into `count` consecutive BF16 output
// sections that together cover the N rows (`add` accumulates into them). `tiled_gdn_input` reads x
// through llama.cpp's tiled GDN value-head column order (K = 6144, or its TP2 half 3072).
// Decoded weights are the exact ggml-quants values. A16 vector passes accumulate in FP32 against the
// BF16 activation; `allow_a8` vector passes instead quantize each 32-value activation group to int8
// with step amax/127 and dot integer weight codes with dp4a (llama.cpp's q8_1 activation model).
// The tensor-core route, taken above kVectorMaxTokens columns with workspace, rounds weights and
// activations to FP16 once regardless of `allow_a8`.
void gguf_project(const Tensor& x, const Weight& weight, const Tensor* outputs, int count, bool add,
                  bool tiled_gdn_input, bool allow_a8, WorkspaceArena* workspace,
                  cudaStream_t stream);

// Workspace that lets every gguf_project call of up to `max_tokens` columns take its fastest route.
[[nodiscard]] std::size_t gguf_workspace_bytes(std::int32_t n, std::int32_t k,
                                               std::int32_t max_tokens);

// Exact rows of the table, rounded once to BF16.
void gguf_embedding(const Tensor& ids, const Weight& weight, Tensor& out, cudaStream_t stream);

// Exact FP32 values of the whole weight, row-major [N,K]. Qualification only.
void gguf_dequantize_fp32(const Weight& weight, float* out, cudaStream_t stream);

// Rows [row_begin, row_begin + row_count) of a GGUF-blocks weight, as a weight of its own.
[[nodiscard]] Weight gguf_row_view(const Weight& weight, std::int32_t row_begin,
                                   std::int32_t row_count);

// Layout, segment coverage and K % 256.
void validate_gguf_weight(const Weight& weight, const char* label);

} // namespace ninfer::ops::detail
