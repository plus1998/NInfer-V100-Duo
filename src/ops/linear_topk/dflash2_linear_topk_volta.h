#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kDflash2TopKTileRows = 1024;

// sm_70 unary top-16: id_map == nullptr for the full head (ids are rows), otherwise the
// optimized head remaps each row. valid_rows bounds the participating physical rows.
void dflash2_linear_topk16_launch(const Tensor& logits, const std::int32_t* id_map,
                                  std::int32_t valid_rows, Tensor& ids, Tensor& scores,
                                  Tensor& partial_ids, Tensor& partial_scores,
                                  cudaStream_t stream, std::int32_t row_offset = 0);

void dflash2_merge_tp2_topk16_launch(const Tensor& gathered_ids, const Tensor& gathered_scores,
                                     Tensor& ids, Tensor& scores, cudaStream_t stream);

} // namespace ninfer::ops::detail
