#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {

enum class Bf16GdnGatingTokenVariant {
    None,
    Full,
    Predicated,
};

void bf16_gdn_gating_proj_gemv_launch(const Tensor& x, const Weight& a_weight,
                                      const Weight& b_weight, const Tensor& A_log,
                                      const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                      cudaStream_t stream);
void bf16_gdn_gating_proj_small_t_split10_launch(const Tensor& x, const Weight& a_weight,
                                                 const Weight& b_weight, const Tensor& A_log,
                                                 const Tensor& dt_bias, void* workspace,
                                                 std::size_t workspace_bytes, Tensor& g,
                                                 Tensor& beta, cudaStream_t stream);
void bf16_gdn_gating_proj_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                             const Weight& a_weight, const Weight& b_weight,
                                             const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                                             Tensor& beta, cudaStream_t stream);

void bf16_gdn_gating_proj_35_simt_c4_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_35_simt_c8_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_split32_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                void* workspace, Tensor& g, Tensor& beta,
                                                cudaStream_t stream);
void bf16_gdn_norm_gating_proj_35_mma_split32_launch(Bf16GdnGatingTokenVariant variant,
                                                     const Tensor& x, const Tensor& norm_weight,
                                                     float eps, Tensor& h, const Weight& a_weight,
                                                     const Weight& b_weight, const Tensor& A_log,
                                                     const Tensor& dt_bias, void* workspace,
                                                     Tensor& g, Tensor& beta, cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_split16_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                void* workspace, Tensor& g, Tensor& beta,
                                                cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                Tensor& g, Tensor& beta, cudaStream_t stream);

// --- tp2 column-shard forms (24 rows/GPU; see bf16_gdn_gating_proj_kernels.cu's kShardN
// comment for why every T routes through gemv/small-T-split10, bypassing the MMA route). ---

void bf16_gdn_gating_proj_gemv_shard_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_small_t_split10_shard_launch(const Tensor& x, const Weight& a_weight,
                                                        const Weight& b_weight, const Tensor& A_log,
                                                        const Tensor& dt_bias, void* workspace,
                                                        std::size_t workspace_bytes, Tensor& g,
                                                        Tensor& beta, cudaStream_t stream);
// Dispatches to the gemv (T=1) or small-T-split10 (T>=2) shard route. `workspace`/`workspace_bytes`
// are ignored for T=1 (unused there, matching the parent's gemv contract).
void bf16_gdn_gating_dispatch_shard(const Tensor& x, const Weight& a_weight,
                                    const Weight& b_weight, const Tensor& A_log,
                                    const Tensor& dt_bias, void* workspace,
                                    std::size_t workspace_bytes, Tensor& g, Tensor& beta,
                                    cudaStream_t stream);
// Transient workspace required by bf16_gdn_gating_dispatch_shard for the given token count.
[[nodiscard]] std::size_t bf16_gdn_gating_shard_workspace_bytes(std::int32_t tokens);

} // namespace ninfer::ops::detail
