#pragma once

// ninfer::ops - fused GDN Q/K/V/Z input projections.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h" // ExecutionContext (tp2 split form)
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: gdn_input_proj
 *
 * Math / indexing:
 *   qkv[:,t] = concat(qk_weight * x[:,t], value_z_weight[0:6144,:] * x[:,t])
 *   z[:,t]   = value_z_weight[6144:12288,:] * x[:,t]
 *
 * Logical shapes:
 *   x [5120,T], qk weight/output rows 4096, value/z rows 6144 each, qkv [10240,T],
 *   z [6144,T]. T may be any positive value. x, qkv, and z are contiguous BF16.
 *   qk_weight is Q4G64_F16S RowSplit [4096,5120] and value_z_weight is one
 *   Q5G64_F16S RowSplit parent [12288,5120] in [value,z] row order, both with FP16 scales.
 *
 * Numeric:
 *   The oracle exact-decodes both weight parents and evaluates all four logical projections
 *   naively in FP64 from the represented input. The BF16 qkv and z outputs are promoted and
 *   compared directly with those ideal values; final output storage rounding belongs to
 *   GdnInputProj's named A16 criterion, not the oracle. Production routes may choose their
 *   private precision independently; every registered route writes both final allocations.
 *
 * Effects:
 *   Writes the full qkv and z outputs; inputs and outputs must not alias.
 *
 * Workspace:
 *   Caller-owned transient storage is sized by q4_q5_gdn_input_proj_workspace_capacity_bytes().
 */
void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t q4_q5_gdn_input_proj_workspace_capacity_bytes(
    std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Single-parent GDN projection. Registered parent forms are:
 *
 * - W8G32_F16S RowSplit [12288,2048], with stored row counts [2048,2048,4096,4096];
 * - NVFP4 BlockScaleK16M128x4 [16384,5120], with stored row counts [2048,2048,6144,6144].
 * - FP8_E4M3FN_ROW_BF16S RowScale [16384,5120], with stored row counts
 *   [2048,2048,6144,6144].
 * - GGML_K GgmlK256 [16384,5120], preserving Q4_K/Q6_K rows and embedded scales in the same
 *   [2048,2048,6144,6144] order. This route admits A16Only with no projection workspace.
 *
 * The first three ranges are written contiguously to qkv and the final range is written to z.
 * W8 admits A16 only. NVFP4 admits A16Only and AllowA4; AllowA4 permits private activation
 * quantization at every positive T. FP8 admits A16Only and AllowA8 at every positive T; AllowA8
 * selects A16 through T=7 and private activation quantization followed by A8 Tensor Core
 * contraction at every T>=8. Every route writes the two independent final allocations directly.
 * The complete projection is evaluated against the same exact-decode/naive-FP64 oracle;
 * activation quantization and the production reduction profile are private effects covered by the
 * selected criterion. x, both persistent weight planes, qkv, z, and the live workspace must be
 * mutually non-overlapping.
 *
 * The policy-bearing form uses caller-owned call-scoped transient storage sized by
 * gdn_input_proj_workspace_capacity_bytes(). A16 requires zero bytes. The convenience overload
 * selects A16Only and requires no transient workspace.
 */
[[nodiscard]] std::size_t
gdn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                        std::int32_t input_rows, LinearPolicy policy,
                                        std::int32_t min_tokens, std::int32_t max_tokens);

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Applies the A16-only single-parent GDN projection without transient workspace.
 */
void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream);

/**
 * Returns the transient capacity required by the registered two-parent Q4/Q5 or single-parent W8
 * snapshot profile. `batch_size` is exact and the query covers every W in the inclusive width
 * interval. B=1 preserves the format-specific fused/composed resolver. B=2..8 uses aggregate
 * projection plus one BF16 [C,B*W] projected plane. The query throws for an unregistered row
 * profile or unsupported B/W domain.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Returns the transient capacity for a registered [16384,5120] NVFP4 or row-scaled FP8 snapshot
 * profile. `batch_size` is exact and the query covers every W in the inclusive width interval.
 * B=1 preserves the format-specific fused/materialized resolver; B=2..8 covers its aggregate
 * projection mechanism plus any projected BF16 plane selected by the complete-Op plan.
 * GGML_K [16384,5120] admits A16Only and always requires one BF16 [10240,B*W] projected plane.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Op: gdn_input_proj_conv_snapshot
 *
 * Math / indexing:
 *   For each row b, let p[:,j,b] be the concatenated q/k/value projection of x[:,j,b], while
 *   z[:,j,b] is the independent z projection. Starting from the BF16 width-three history selected
 *   by initial_state_slots[b], evaluate the width-four depthwise convolution over p, apply SiLU,
 *   and publish its channel ranges to query, key, and value. After valid column j, write the new
 *   width-three history to snapshot_base_slots[b]+j. Z bypasses convolution.
 *
 * Logical shapes:
 *   The 27B registered form has x [5120,W,B], Q4 q/k weight [4096,5120], one Q5 value/z parent
 *   [12288,5120], conv_weight [10240,4], conv_states [10240,3,Slots], query/key [2048,W,B],
 *   value/z [6144,W,B], and I32 selectors [B]. B=1 accepts every positive W; B=2..8 accepts
 *   W=1..16. `valid_columns` is empty for a dense invocation or I32 [B] for a mixed-width batch.
 *   A mixed-width invocation has B>=1 and every valid extent lies in [1,W].
 *
 * Numeric:
 *   The oracle exact-decodes packed weights and evaluates projection, convolution, SiLU, z, and
 *   every snapshot value naively in FP64 from represented inputs. BF16 query/key/value/z and
 *   snapshots are promoted and compared directly with those ideal values; their final storage
 *   rounding belongs to the Op's named A16 criterion, not the oracle. Former unfused projection
 *   tensors are not observable cast boundaries; production routes use their natural private
 *   accumulator and staging precision. This two-parent Q4/Q5 form does not quantize activation;
 *   the single-parent policy-bearing form below defines its own permitted compute profiles.
 *
 * Effects:
 *   Each row writes query/key/value through its valid prefix and exact zero to its invalid tail;
 *   z is projected for all B*W safe input columns. A row writes only its valid destination state
 *   prefix. The caller reserves disjoint complete [base,base+W) intervals, prevents one row from
 *   overwriting another row's initial slot, and may overlap a row's own initial slot with its
 *   destination after that initial history has been loaded. Other slots are unchanged. Newly
 *   projected convolution channels remain private to the call while published snapshots are BF16.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Single-parent form of gdn_input_proj_conv_snapshot. Registered parents are W8G32_F16S RowSplit
 * [12288,2048], NVFP4 BlockScaleK16M128x4 [16384,5120], and FP8_E4M3FN_ROW_BF16S RowScale
 * [16384,5120], all in q/k/value/z row order. W8 admits A16Only, NVFP4 admits A16Only/AllowA4,
 * and FP8 admits A16Only/AllowA8. B=1 accepts every positive W for FP8; the batched domain is
 * B=2..8 and W=1..16. For FP8 B=1, A16 is fused at W=1..3 and W=7..10 and materialized
 * otherwise; AllowA8 uses the same winners through W=9 and A8 from W=10. Batched AllowA8 uses A8
 * when B*W>=9. Tensor operands, the complete FP8 parent, and live workspace must be mutually
 * non-overlapping, except that the read-only initial_state_slots and snapshot_base_slots selectors
 * may alias each other; same-row state-slot overlap remains governed by the snapshot state
 * contract.
 * GGML_K [16384,5120] uses the same shapes and state contract with A16Only. Its composed route
 * projects all physical columns into BF16 before applying the convolution and state update.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Applies the A16-only single-parent form. FP8 accepts every positive W for dense B=1; the batched
 * domain is B=2..8 and W=1..16.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Returns the transient capacity for the registered Q4/Q5 or W8 record-producing profile.
 * `batch_size` is exact, and the inclusive T interval must lie within ReplaySSM's B=1..8,
 * T=2..16 execution domain. These profiles require no transient storage because materialized
 * projection writes directly to caller-owned conv_record.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Returns the transient capacity for a registered [16384,5120] NVFP4 or row-scaled FP8
 * record-producing profile. Fused and materialized A16 routes require no storage. AllowA4/AllowA8
 * returns only the activation-quantization workspace selected by this complete-Op route;
 * conv_record is caller-owned.
 * GGML_K [16384,5120] admits A16Only, projects directly into conv_record and requires no scratch.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Op: gdn_input_proj_conv_record
 *
 * For each batch row, evaluates the registered projection, width-four causal convolution, SiLU,
 * and q/k/value/z split from the BF16 history selected by initial_state_slots. It writes the BF16
 * represented projection column consumed by the convolution to conv_record [C,T,B]. Query, key,
 * and value are zero in each row's invalid tail; z is projected for every physical column.
 *
 * The execution domain is B=1..8 and T=2..16. valid_columns is empty for dense input or device
 * I32 [B], with each caller-supplied extent in [1,T]. conv_states is a read-only BF16 [C,3,S]
 * state-pool view, and initial_state_slots contains absolute slots in [0,S). Source state is not
 * modified. Only the valid prefix of conv_record is semantically defined.
 *
 * The two-parent form registers Q4 q/k [4096,5120] and the Q5 value/z parent [12288,5120]. All
 * tensor operands, outputs, conv_record, source state, and live workspace must be disjoint.
 */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& qk_weight,
                                const Weight& value_z_weight, const Tensor& conv_weight,
                                const Tensor& conv_states, const Tensor& valid_columns,
                                const Tensor& initial_state_slots, Tensor& conv_record,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Single-parent record-producing form. Registered parents are W8G32_F16S [12288,2048], NVFP4
 * [16384,5120], and FP8_E4M3FN_ROW_BF16S [16384,5120]. W8 admits A16Only, NVFP4 admits
 * A16Only/AllowA4, and FP8 admits A16Only/AllowA8. For FP8 B=1, A16 is fused at W=2..3 and
 * W=7..10 and materialized otherwise; AllowA8 uses A8 from W=10. Batched AllowA8 uses A8 when
 * B*W>=8. Every tensor operand, the complete FP8 parent, and live workspace must be mutually
 * non-overlapping.
 */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream);

/** Applies the A16-only single-parent record-producing form. */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream);

// --- Tensor-parallel split form (tp == 2) -------------------------------------------------------
//
// DESIGN NOTE. ShardPlan's real interior layout for `gdn/query_key_value_z`
// (src/targets/qwen3_6_27b/impl/load/bindings.cpp, `plan_for`'s "gdn/query_key_value_z" branch) is
// FOUR independent append_column_block calls in Q | K | V | Z order:
//
//   Q [0, key_dim=2048) | K [2048, 4096) | V [4096, 10240) | Z [10240, 16384)
//
// (key_dim = gdn_key_heads(16) * gdn_key_head_dim(128); value_dim = gdn_value_heads(48) *
// gdn_value_head_dim(128) = 6144 -- see src/targets/qwen3_6_27b/impl/config.h). At tp=2, device r's
// shard is the concatenation, in that same Q|K|V|Z order, of rank r's own half of every section
// (the same derivation attn_input_proj.h states for its own Q|K|Gate|V object):
//
//   shard-local Q [0,1024) | K [1024,2048) | V [2048,5120) | Z [5120,8192)
//
// (Q/K: 8 of the 16 key heads/device, 128 dim each; V/Z: 24 of the 48 value heads/device). This
// matches Nvfp4GdnInputTp2ColumnGeometry (<8192,5120>, src/ops/linear/nvfp4/nvfp4_config.h)
// exactly.
//
// OUTPUT CONTRACT: mirrors the tp1 Op's own 2-tensor packing (qkv + z), Geometry-halved -- NOT
// four separate q/k/v/z tensors. Each device's call writes qkv[5120,T] (this device's Q|K|V,
// shard-local row ranges [0,1024) | [1024,2048) | [2048,5120)) and z[3072,T] (this device's V-head-
// aligned Z, rows [0,3072)). The GDN core reads a global head index by adding
// `device_rank * 8` (Q/K) or `device_rank * 24` (V/Z) to the shard-local row's head number
// (row / 128), the same convention attn_input_proj states for its own output. No cross-device
// traffic occurs before or during this Op -- gdn_input_proj is column-parallel only, no allreduce.
//
// FORMATS REGISTERED: NVFP4 (A16Only/AllowA4, Nvfp4GdnInputTp2ColumnGeometry, shared with
// ops::linear's own registry in src/ops/linear/nvfp4/nvfp4_config.h), FP8_E4M3FN_ROW_BF16S
// (A16Only/AllowA8, Fp8GdnInputTp2ColumnGeometry in src/ops/linear/fp8/fp8_config.h -- the
// Qwen38Nvfp4 profile's flagship binding for this object per bind_qwen38_nvfp4_text_layers) and
// the Q4G64_F16S/Q5G64_F16S split-storage two-weight form (the groupwise profile's binding).
// BF16_CTRL and W8G32_F16S are NOT registered -- neither is bound for this object by any
// qwen3_8_27b weights profile (see bindings.cpp), the same exclusion attn_input_proj makes for
// its sibling family.

[[nodiscard]] std::size_t gdn_input_proj_column_parallel_workspace_capacity_bytes(
    QType qtype, LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens);

void gdn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                    const std::array<Weight, 2>& query_key_value_z_weight,
                                    const std::array<Tensor, 2>& qkv, const std::array<Tensor, 2>& z,
                                    LinearPolicy policy,
                                    const std::array<WorkspaceArena*, 2>& workspace,
                                    const ExecutionContext& ec);

/** A16-only convenience overload, no policy/workspace. */
void gdn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                    const std::array<Weight, 2>& query_key_value_z_weight,
                                    const std::array<Tensor, 2>& qkv, const std::array<Tensor, 2>& z,
                                    const ExecutionContext& ec);

/** Q4G64_F16S/Q5G64_F16S split-storage two-weight form. */
void gdn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                    const std::array<Weight, 2>& query_key_weight,
                                    const std::array<Weight, 2>& value_z_weight,
                                    const std::array<Tensor, 2>& qkv, const std::array<Tensor, 2>& z,
                                    const ExecutionContext& ec);

// --- Tensor-parallel split forms of the fused projection+conv1d+SiLU pair (tp == 2) -------------
//
// DESIGN NOTE. These two Ops are reached only from Phase::Verify (the MTP / speculative-verify
// path), so their split is the composition of the bare projection's column shard (above) with the
// GDN convolution's channel shard, rather than anything either of those owns on its own.
//
// GEOMETRY. Device r owns 8 of the 16 GDN key heads and 24 of the 48 GDN value heads, so its
// projection shard is [8192,5120] (Nvfp4/Fp8GdnInputTp2ColumnGeometry, above) and its
// convolution owns 5120 of the 10240 depthwise channels:
//
//   shard-local Q [0,1024) | K [1024,2048) | V [2048,5120)     (conv channels; 5120 total)
//   shard-local Z [0,3072)                                     (bypasses the convolution)
//
// This is byte-for-byte the channel set `gdn/convolution`'s three-block ShardPlan gives device r
// (bindings.cpp's `gdn/convolution` branch: channels [1024r,+1024) u [2048+1024r,+1024) u
// [4096+3072r,+3072)) and the packing `gdn_input_proj_column_parallel` already writes, so nothing
// repacks between the projection and the convolution.
// Per-device operand shapes: x [5120,W,B], conv_weight [5120,4], conv_states [5120,3,Slots],
// query/key [1024,W,B], value [3072,W,B], z [3072,W,B], conv_record [5120,T,B].
//
// NO COLLECTIVE. The convolution is depthwise and the projection is column-parallel, so a device's
// 5120 channels never mix with the peer's. Both Ops are entirely local; the GDN block's one and
// only all-reduce is the row-parallel `gdn/output` projection at the very end.
//
// ROUTE. The shard always takes the COMPOSED route (projection shard -> BF16 projected plane ->
// the channel-generic projected-conv kernel), never a fused per-format snapshot kernel. The fused
// NVFP4/FP8/W8 snapshot kernels bake the tp1 row profile into compile-time constants, and
// instantiating a parallel exact set at the shard profile is kernel work, not the registry work
// a shard geometry otherwise needs; the composed route is the same arithmetic in the same order at
// one extra kernel launch and one BF16 staging round for query/key/value. This is the one route
// fall-off the conv snapshot/record shards carry, and it is a performance item only: FP8 is
// bit-exact against the tp1 reference at every case covered by ninfer_mtp_split_test, and NVFP4
// is bit-exact wherever the tp1 path is also composed.
//
// FORMATS REGISTERED: NVFP4 (A16Only/AllowA4), FP8_E4M3FN_ROW_BF16S (A16Only/AllowA8) and the
// Q4G64_F16S/Q5G64_F16S split-storage two-weight form -- exactly the three
// `gdn_input_proj_column_parallel` registers. W8G32_F16S is NOT registered (it is the 35B-A3B
// profile's [12288,2048] parent, which this target's ShardPlan never splits).

[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
    QType qtype, LinearPolicy policy, std::int32_t batch_size, std::int32_t min_width,
    std::int32_t max_width);

[[nodiscard]] std::size_t gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
    QType qtype, LinearPolicy policy, std::int32_t batch_size, std::int32_t min_width,
    std::int32_t max_width);

void gdn_input_proj_conv_snapshot_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_value_z_weight,
    const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_state_slots,
    const std::array<Tensor, 2>& snapshot_base_slots, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& z, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec);

/** Q4G64_F16S/Q5G64_F16S split-storage two-weight snapshot form (A16 only). */
void gdn_input_proj_conv_snapshot_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_weight,
    const std::array<Weight, 2>& value_z_weight, const std::array<Tensor, 2>& conv_weight,
    const std::array<Tensor, 2>& conv_states, const std::array<Tensor, 2>& valid_columns,
    const std::array<Tensor, 2>& initial_state_slots,
    const std::array<Tensor, 2>& snapshot_base_slots, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& z, const std::array<WorkspaceArena*, 2>& workspace,
    const ExecutionContext& ec);

void gdn_input_proj_conv_record_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_value_z_weight,
    const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_state_slots,
    const std::array<Tensor, 2>& conv_record, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& z, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec);

/** Q4G64_F16S/Q5G64_F16S split-storage two-weight record form (A16 only). */
void gdn_input_proj_conv_record_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_weight,
    const std::array<Weight, 2>& value_z_weight, const std::array<Tensor, 2>& conv_weight,
    const std::array<Tensor, 2>& conv_states, const std::array<Tensor, 2>& valid_columns,
    const std::array<Tensor, 2>& initial_state_slots, const std::array<Tensor, 2>& conv_record,
    const std::array<Tensor, 2>& query, const std::array<Tensor, 2>& key,
    const std::array<Tensor, 2>& value, const std::array<Tensor, 2>& z,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec);

} // namespace ninfer::ops
