#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"
#include "targets/qwen3_6/impl/runtime/yarn_rope.h"

#include "core/device.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/swa.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    // ONE device's persistent state. At tp == 2 this device owns half the attention KV heads, half
    // the GDN value heads and half the GDN conv channels -- the exact halving the per-GPU memory
    // budget assumes. The PAGE geometry is deliberately NOT divided: every device carries the
    // same page count and the same block tables, which is what makes one shared KV capacity plan
    // legal.
    const std::int32_t tp = plan.tp;
    if (tp != 1 && tp != 2) { throw std::invalid_argument("sequence plan tp must be 1 or 2"); }
    const std::int32_t linear_state_slots =
        LinearStateSlots::state_slot_count(plan.max_concurrency);
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const std::uint32_t logical_pages  = page_count(plan.capacity);
    const std::uint32_t physical_pages = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages =
        plan.features.mtp()
            ? static_cast<std::uint64_t>(plan.max_concurrency) *
                  ((static_cast<std::uint64_t>(plan.draft_window - 1U) + kPagedKVPageSize - 1U) /
                   static_cast<std::uint32_t>(kPagedKVPageSize))
            : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_6::plan_decoder_state(
        builder, qwen3_6::DecoderStateSpec{
                     .full_attention_layers     = TextConfig::full_attention_layers(),
                     .mtp_layers                = TextConfig::mtp_layers,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = TextConfig::kv_heads / tp,
                     .attention_head_dim        = TextConfig::head_dim,
                     .kv_dtype                  = plan.kv_dtype,
                     .kv_quant_group            = plan.kv_quant_group,
                     .enable_mtp                = plan.features.mtp(),
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = physical_pages,
                     .mtp_physical_page_groups  = mtp_physical_pages,
                     .linear_attention =
                         {
                             .layers         = TextConfig::gdn_layers(),
                             .conv_channels  = TextConfig::convolution_dim / tp,
                             .conv_width     = TextConfig::gdn_conv_state_width,
                             .value_heads    = TextConfig::gdn_value_heads / tp,
                             .value_head_dim = TextConfig::gdn_value_head_dim,
                             .key_head_dim   = TextConfig::gdn_key_head_dim,
                             .slot_count     = linear_state_slots,
                             .conv_dtype     = DType::BF16,
                         },
                 });
    if (plan.speculative_backend != SpeculativeBackend::None) {
        if (tp != 1 && plan.speculative_backend != SpeculativeBackend::Mtp) {
            throw std::invalid_argument(
                "DFlash speculative decoding has no tensor-parallel path in this build");
        }
        // ONE device's replay records, exactly like the decoder state above: the GDN verify round
        // records this device's own head/channel shard, and the registered
        // FoldGeometry<48, 8, 24, 5120> is the shape the peer's fold consumes. The RECORD
        // CAPACITY and WIDTH are not divided -- both devices record the same rows and the same
        // draft window.
        out.replay_records = plan_gdn_replay_records(
            builder, GdnReplayRecordSpec{
                         .layers          = TextConfig::gdn_layers(),
                         .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                         .width           = static_cast<std::int32_t>(plan.draft_window + 1U),
                         .conv_channels   = TextConfig::convolution_dim / tp,
                         .qk_heads        = TextConfig::gdn_key_heads / tp,
                         .value_heads     = TextConfig::gdn_value_heads / tp,
                         .key_dim         = TextConfig::gdn_key_head_dim,
                         .value_dim       = TextConfig::gdn_value_head_dim,
                     });
    }
    if constexpr (Variant::supports_dflash) {
        if (plan.features.dflash()) {
            if (tp != 1) {
                throw std::invalid_argument("DFlash has no tensor-parallel path in this build");
            }
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            dflash.local = plan_cyclic_kv_cache(builder, DFlashConfig::local_layers,
                                                DFlashConfig::local_capacity,
                                                DFlashConfig::kv_heads, DFlashConfig::head_dim,
                                                static_cast<std::int32_t>(plan.max_concurrency));
            dflash.rewrite_checkpoint_local = plan_cyclic_kv_cache(
                builder, DFlashConfig::local_layers, DFlashConfig::local_capacity,
                DFlashConfig::kv_heads, DFlashConfig::head_dim,
                static_cast<std::int32_t>(plan.max_concurrency));
            PagedKVPoolSpec full_pool{
                .page_group_count      = physical_pages,
                .logical_page_capacity = logical_pages,
                .table_rows            = static_cast<std::int32_t>(plan.max_concurrency),
                .plane_order           = PagedKVPlaneOrder::HeadMajor,
                .planes =
                    {
                        {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256},
                        {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256},
                    },
            };
            dflash.full = qwen3_6::PagedKVCacheLayout{
                .pool        = plan_paged_kv_pool(builder, full_pool),
                .layers      = 1,
                .max_context = plan.capacity,
                .kv_heads    = DFlashConfig::kv_heads,
                .head_dim    = DFlashConfig::head_dim,
                .dtype       = DType::BF16,
                .quant_group = 0,
            };
            dflash.prefill_features = add_tensor(
                builder, DType::BF16, {DFlashConfig::feature_rows, effective_prefill_chunk},
                "DFlash prefill target features");
            dflash.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash prefill target positions");
            dflash.pending_features  = add_tensor(builder, DType::BF16,
                                                  {DFlashConfig::feature_rows,
                                                   static_cast<std::int32_t>(plan.draft_window + 1U),
                                                   static_cast<std::int32_t>(plan.max_concurrency)},
                                                  "DFlash pending target features");
        }
    }

    out.round = qwen3_6::begin_round_state_layout(
        builder, qwen3_6::RoundStateSpec{.hidden         = TextConfig::hidden,
                                         .output_rows    = TextConfig::output_rows,
                                         .batch_capacity = plan.max_concurrency,
                                         .draft_window   = plan.draft_window,
                                         .enable_mtp     = plan.features.mtp(),
                                         .enable_dflash  = plan.features.dflash()});
    out.prefill_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, effective_prefill_chunk}, "step prefill hidden");
    qwen3_6::complete_round_state_layout(builder, out.round);
    const auto i32 = [&](std::size_t n, const char* label) {
        return add_tensor(builder, DType::I32, {static_cast<std::int32_t>(n)}, label);
    };
    out.token_counts =
        add_tensor(builder, DType::I32,
                   {TextConfig::token_domain, static_cast<std::int32_t>(plan.max_concurrency)},
                   "sampling token counts");
    const auto config_words = static_cast<std::int32_t>(
        (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
    out.sampling_config = add_tensor(
        builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
        "sampling config");
    out.tail_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(plan.max_concurrency)},
        "tail hidden");
    out.rewrite_checkpoint_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(plan.max_concurrency)},
        "rewrite checkpoint hidden");
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::GqaExecutionEnvelope text_envelope{1, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace_recipe::text_prefill_roots<TextConfig>(
            layout, tokens, plan.features.vision ? 3 : 0, plan.features.vision ? tokens : 0);
    };
    const auto attention_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                     std::int32_t last, qwen3_6::TextPhase phase,
                                     std::int32_t batch_size, std::int32_t min_width,
                                     std::int32_t max_width, ops::GqaExecutionEnvelope envelope) {
        auto stage = layout.scope();
        (void)workspace_recipe::text_attention_projection<TextConfig>(layout, last);
        scratch(layout, Variant::attention_projection_workspace_capacity_bytes(plan.weights_profile,
                                                                               phase, first, last));
        (void)workspace_recipe::text_attention_results<TextConfig>(layout, last);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            TextConfig::query_heads, plan.kv_dtype, envelope, batch_size, min_width,
                            max_width));
        scratch(layout, Variant::attention_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto gdn_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                               std::int32_t last, qwen3_6::TextPhase phase, GdnWorkspacePath path,
                               std::int32_t batch_size, std::int32_t min_width,
                               std::int32_t max_width) {
        auto stage = layout.scope();
        (void)workspace_recipe::gdn_control<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_norm_control_projection_workspace_capacity_bytes(first, last));
        (void)workspace_recipe::gdn_projection<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::Snapshot) {
            scratch(layout, Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else if (path == GdnWorkspacePath::ReplayRecord) {
            scratch(layout, Variant::gdn_input_projection_record_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else {
            (void)workspace_recipe::gdn_prefill_conv<TextConfig>(layout, last);
            scratch(layout, Variant::gdn_input_projection_workspace_capacity_bytes(
                                plan.weights_profile, phase, first, last));
        }
        (void)workspace_recipe::gdn_recurrent_output<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::Prefill) {
            scratch(layout,
                    ops::gated_delta_net_workspace_capacity_bytes(
                        TextConfig::gdn_key_heads, TextConfig::gdn_value_heads, true, first, last));
        }
        (void)workspace_recipe::gdn_normalized_output<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto post_mixer_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                      std::int32_t last, qwen3_6::TextPhase phase) {
        auto stage = layout.scope();
        (void)workspace_recipe::post_mixer_hidden<TextConfig>(layout, last);
        scratch(layout, Variant::post_mixer_workspace_capacity_bytes(plan.weights_profile, phase,
                                                                     first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, qwen3_6::TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width, ops::GqaExecutionEnvelope envelope) {
        attention_stage(layout, first, last, phase, batch_size, min_width, max_width, envelope);
        gdn_stage(layout, first, last, phase, path, batch_size, min_width, max_width);
        post_mixer_stage(layout, first, last, phase);
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, Variant::draft_head_rows, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace_recipe::mtp_stem<TextConfig>(layout, tokens, !preembedded);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
        (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            TextConfig::query_heads, plan.kv_dtype, envelope, 1, tokens, tokens));
        (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope, bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        if (plan.tp > 1) {
            // tp2 only: the final-chunk stage's own one-column all-reduce staging, distinct from
            // the chunk-wide [hidden, T] staging planned by tp_mtp_call_roots.
            matrix(layout, DType::BF16, TextConfig::hidden, 1);
        }
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            scratch(layout, Variant::mtp_kv_projection_workspace_capacity_bytes(first, last));
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
        }
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, Variant::mtp_q_gate_projection_workspace_capacity_bytes(1, 1));
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            TextConfig::query_heads, plan.kv_dtype, text_envelope, 1, 1, 1));
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(1, 1));
        proposal_scratch(layout, 1);
    };

    // Tensor-parallel additions, live for a whole call rather than one stage:
    //   * `allreduce staging`  - the peer's contribution to every row-parallel reduce ([hidden, T]
    //                            BF16, one per device, allocated once per call and reused by all
    //                            128 reduces).
    //   * `vocabulary half`    - this device's own half of the logits, before the gather. The
    //                            GATHERED full logits go straight into each rank's persistent
    //                            RoundState logits, so they cost no workspace. Only the columns
    //                            that actually get a logit are planned: one in prefill (the bonus
    //                            token), `batch` in a decode round.
    // Every OTHER extent in this plan is still the tp1 (whole-model) extent: at tp == 2 each
    // device's real per-stage need is at most that, so planning it whole is a safe over-estimate.
    // Sizing the sharded stages exactly is a follow-up (it only buys workspace bytes back).
    const auto tp_call_roots = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   std::int32_t logit_columns) {
        if (plan.tp <= 1) { return; }
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        matrix(layout, DType::BF16, TextConfig::output_rows / plan.tp, logit_columns);
    };

    // The MTP round's own tp2 additions. Its three all-reduces share ONE [hidden, T] staging
    // buffer per device (allocated once per MTP call, reused by the stem's fc, the attention
    // output projection and the post-mixer), and the proposal head's gather needs this device's
    // own half of the proposal logits alongside the full gathered vector the tp1 plan already
    // covers. Every other MTP extent is planned at the tp1 (whole-model) width above, which
    // over-plans a tp2 device rather than under-planning it.
    const auto tp_mtp_call_roots = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                       std::int32_t logit_columns) {
        if (plan.tp <= 1) { return; }
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        matrix(layout, DType::BF16,
               (plan.proposal_head == ProposalHead::Optimized ? Variant::draft_head_rows
                                                              : TextConfig::output_rows) /
                   plan.tp,
               logit_columns);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    tp_call_roots(text_prefill, chunk, 1);
    target_body(text_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1,
                1, chunk, text_envelope);
    scratch(text_prefill, ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, 1, 1));
    out.text_prefill = finish(text_prefill);

    for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
         ++batch) {
        WorkspaceLayoutBuilder ordinary;
        matrix(ordinary, DType::BF16, TextConfig::hidden, batch);
        tp_call_roots(ordinary, batch, batch);
        target_body(ordinary, batch, batch, qwen3_6::TextPhase::Verify, GdnWorkspacePath::Snapshot,
                    batch, 1, 1, text_envelope);
        scratch(ordinary,
                ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, batch, batch));
        out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        tp_call_roots(mtp_prefill, chunk, 1);
        tp_mtp_call_roots(mtp_prefill, chunk, 1);
        target_body(mtp_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, GdnWorkspacePath::Prefill,
                    1, 1, chunk, text_envelope);
        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, chunk);
            (void)workspace_recipe::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        tp_mtp_call_roots(mtp_batch, verify, 1);
        mtp_full_call(mtp_batch, verify, text_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        tp_mtp_call_roots(mtp_ar, 1, 1);
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        tp_mtp_call_roots(mtp_align, 1, 1);
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        tp_mtp_call_roots(mtp_proposal, 1, 1);
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            TextConfig::token_domain, drafts, drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, TextConfig::hidden, aggregate);
            tp_call_roots(target, aggregate, aggregate);
            target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
                scratch(layout,
                        Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
                (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
                scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                                    TextConfig::query_heads, plan.kv_dtype, text_envelope, batch,
                                    width, width));
                (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
                scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
            };

            WorkspaceLayoutBuilder alignment;
            tp_mtp_call_roots(alignment, batch * verify, batch);
            mtp_decode_core(alignment, verify);
            WorkspaceLayoutBuilder ar;
            tp_mtp_call_roots(ar, batch, batch);
            mtp_decode_core(ar, 1);
            WorkspaceLayoutBuilder proposal;
            tp_mtp_call_roots(proposal, batch, batch);
            proposal_scratch(proposal, batch);
            const std::size_t batch_accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    TextConfig::token_domain, drafts, drafts, batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), finish(alignment), finish(ar),
                                      finish(proposal), batch_accept});
        }
    }

    if (plan.features.dflash()) {
        if constexpr (!Variant::supports_dflash) {
            throw std::logic_error("unsupported target reached DFlash scratch planning");
        } else {
            const auto dflash_context_capacity = [&](std::int32_t tokens, bool compact_input) {
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16, DFlashConfig::feature_rows, tokens);
                }
                (void)workspace_recipe::dflash_context<DFlashConfig>(layout, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace_recipe::dflash_context_layer<DFlashConfig>(layout, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens = width * batch;
                matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                {
                    auto attention = layout.scope();
                    (void)workspace_recipe::dflash_attention<DFlashConfig>(layout, tokens);
                    scratch(layout,
                            std::max(ops::swa_workspace_capacity_bytes({0, plan.capacity}, width,
                                                                       width, batch),
                                     ops::bidirectional_gqa_attention_workspace_capacity_bytes(
                                         {0, plan.capacity}, width, width, batch)));
                    scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                        QType::W8G32_F16S, DFlashConfig::hidden,
                                        DFlashConfig::query_size, tokens, tokens));
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace_recipe::dflash_mlp<DFlashConfig>(layout, tokens);
                    scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                        QType::W8G32_F16S, 2 * DFlashConfig::intermediate,
                                        DFlashConfig::hidden, tokens, tokens));
                    scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                        QType::W8G32_F16S, DFlashConfig::hidden,
                                        DFlashConfig::intermediate, tokens, tokens));
                }
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, Variant::draft_head_rows, drafts * batch);
                } else {
                    matrix(layout, DType::BF16, TextConfig::output_rows, drafts * batch);
                }
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, false);
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, TextConfig::hidden, aggregate);
                target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
                const std::size_t accept =
                    ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                        TextConfig::token_domain, drafts, drafts, batch, batch);
                const std::size_t proposal = dflash_proposal_capacity(verify, batch);
                out.dflash_round           = std::max({out.dflash_round, finish(target), accept,
                                                       dflash_context_capacity(aggregate, true), proposal});
            }
        }
    }

    if (plan.features.vision) {
        constexpr std::uint32_t kFrontendMergedLimit  = 32768;
        constexpr std::uint32_t kFrontendSegmentLimit = 768 / 2;
        const std::uint32_t merged = std::min(plan.capacity, kFrontendMergedLimit);
        out.vision_encode          = schedule::VisionContext::workspace_capacity_bytes(
            merged, std::min(merged, kFrontendSegmentLimit));
    }

    out.capacity = std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                             out.dflash_context, out.dflash_round, out.vision_encode});
    return out;
}

// Returns the EFFECTIVE context ceiling the options were admitted against: the variant's
// registered native capacity under `RopeMode::Native`, `yarn_origin * yarn_factor` under
// `RopeMode::Yarn`. The single caller passes it straight into the sequence plan, so the ceiling is
// resolved exactly once per Engine construction.
std::uint32_t validate_target_options(DeviceContext& device, const EngineOptions& options) {
    // This call is also where every rope-option rejection lives (unsupported variant, Vision,
    // DFlash, wrong origin, factor that overshoots the 1,048,576 product ceiling) -- see
    // yarn_rope.h.
    const std::uint32_t effective_max_context = qwen3_6::detail::rope_effective_max_context(
        options, qwen3_6::detail::RopeDomain{kNativeMaxContext, kSupportsYarnRope});
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be in [1,5]");
        }
        break;
    case SpeculativeBackend::DFlash:
        if (kMaximumDFlashDraftTokens == 0) {
            throw std::invalid_argument("DFlash is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumDFlashDraftTokens) {
            throw std::invalid_argument("DFlash draft window must be in [1,15]");
        }
        if (options.enable_vision) {
            throw std::invalid_argument("DFlash and Vision cannot be enabled together");
        }
        break;
    }
    if (options.tp != 1 && options.tp != 2) {
        throw std::invalid_argument("tensor-parallel width must be 1 or 2");
    }
    if (options.tp == 2) {
        // MTP is split-aware (sharded stem/attention/post-mixer, sharded draft head with an
        // allgather before the proposal argmax, per-device GDN replay records and per-device
        // replay fold). DFlash is NOT: its weights are sharded by the load plan but
        // its forward path composes plain linear/residual_add over whole-width tensors, and the
        // Vision encoder runs entirely on device 0. Engine rejects both combinations too (its
        // guard is the authority for callers that never reach a target); this is the
        // target-layer statement of the same fact.
        if (options.speculative.backend == SpeculativeBackend::DFlash) {
            throw std::invalid_argument("--tp 2 does not support the DFlash speculative backend "
                                        "in this build; use --tp 1, --spec mtp or --spec none");
        }
        if (options.enable_vision) {
            throw std::invalid_argument("--tp 2 does not support Vision in this build");
        }
    }
    if (device.sm() != 70 && device.sm() != 86 && device.sm() != 89 && device.sm() != 120) {
        throw std::invalid_argument("Qwen3.6 family runtime requires a registered CUDA target");
    }
    return effective_max_context;
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                 = std::make_unique<SequencePlanImpl>();
    impl->weights_profile     = inputs.weights_profile;
    impl->capacity            = inputs.capacity;
    impl->main_page_groups    = main_page_groups;
    impl->kv_capacity         = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency     = inputs.max_concurrency;
    impl->prefill_chunk       = inputs.prefill_chunk;
    impl->draft_window        = inputs.draft_window;
    impl->speculative_backend = inputs.speculative_backend;
    impl->proposal_head       = inputs.proposal_head;
    impl->features            = inputs.features;
    impl->rope_mode           = inputs.rope_mode;
    impl->yarn_factor         = inputs.yarn_factor;
    impl->yarn_origin         = inputs.yarn_origin;
    impl->effective_max_context = inputs.effective_max_context;
    impl->use_cuda_graph      = inputs.use_cuda_graph;
    impl->device              = inputs.device;
    impl->tp                  = inputs.tp;
    impl->kv_dtype            = inputs.kv_dtype;
    impl->kv_quant_group      = inputs.kv_quant_group;
    impl->persistent          = persistent_layout(*impl);
    impl->workspace           = build_workspace_plan(*impl);
    if (impl->features.vision) {
        constexpr std::uint32_t kFrontendMergedLimit = 32768;
        const std::uint32_t merged = std::min(impl->capacity, kFrontendMergedLimit);
        impl->request_transient_capacity_bytes =
            schedule::VisionContext::output_transient_bytes(merged);
    }
    if (impl->use_cuda_graph) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each reachable node-topology class. These bounds cover the largest profile installed in
        // each class and the driver/module state materialized while qualifying all definitions.
        if (impl->speculative_backend == SpeculativeBackend::None) {
            // Per device, at both widths. At tp 2 one cross-device graph replaces the tp 1 graph
            // but holds BOTH devices' nodes (2.95x the tp 1 node count: each device's halved-width
            // schedule, the extra un-fused GDN norm, and the collectives' cross-device copies), and
            // instantiating it materializes driver state on each device it has nodes on.
            //
            // Measured on 2x RTX 5090 at max_concurrency 8, 8192 context:
            //   tp 1: 20 MiB on the one device        (2.5  MiB per concurrent request)
            //   tp 2: 26 MiB rank 0, 24 MiB rank 1    (3.25 MiB and 3.0 MiB per request)
            // The tp 2 multiplier is nevertheless raised from 12 to 20 MiB per request, and the
            // reason is the LOW-concurrency end rather than the measured high end: at
            // max_concurrency 1 the budget is one multiplier, and an observed one-shot overage of
            // ~19.9 MiB against a 12 MiB budget on a SINGLE device -- attributed to a late module
            // load -- is on record. tp 2 doubles that surface, so 12 MiB would be a budget the
            // measured steady state fits inside and a known transient does not.
            //
            // THE MARGIN IS THINNEST AT max_concurrency 1, which is the likely 1M configuration:
            // the tp 2 per-batch-size eager warm pass in prepare_graphs() only has batch sizes
            // 2..max_concurrency to warm, so at max_concurrency 1 it does nothing and the sole
            // code-warm pass carries all the module loading -- inside the measurement window. The
            // budget there is one multiplier, 20 MiB, against a 0.3b-style worst case of ~19.9
            // MiB: roughly 1 MiB of headroom. Raising this multiplier, not the warm pass, is the
            // lever if that transient is ever observed at tp 2.
            const std::uint64_t per_batch = impl->tp == 2 ? 20ULL * kMiB : 12ULL * kMiB;
            impl->graph_allowance_bytes =
                checked_mul(per_batch, impl->max_concurrency, "ordinary exact-b graph allowance");
        } else if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            const auto profiles = mtp_graph_profiles(impl->capacity, impl->draft_window);
            const std::size_t per_batch_allowance = graph_topology_allowance(
                profiles,
                [&](GraphExecutionProfile profile) {
                    const std::uint64_t final_visible = std::min<std::uint64_t>(
                        impl->capacity,
                        static_cast<std::uint64_t>(profile.max) + 2ULL * impl->draft_window);
                    return (final_visible <= 4096 ? 12ULL : 82ULL) * kMiB;
                },
                "MTP graph allowance");
            impl->graph_allowance_bytes = checked_mul(per_batch_allowance, impl->max_concurrency,
                                                      "MTP exact-b graph allowance");
        } else {
            const auto class_allowance = [&](std::uint32_t batch_size) {
                const auto profiles =
                    dflash_graph_profiles(impl->capacity, impl->draft_window, batch_size);
                return graph_topology_allowance(
                    profiles,
                    [&](GraphExecutionProfile profile) {
                        const std::uint64_t final_visible = std::min<std::uint64_t>(
                            impl->capacity,
                            static_cast<std::uint64_t>(profile.max) + impl->draft_window + 1ULL);
                        return (final_visible <= 4096 ? 64ULL : 96ULL) * kMiB;
                    },
                    "DFlash graph allowance");
            };
            for (std::uint32_t batch_size = 1; batch_size <= impl->max_concurrency; ++batch_size) {
                impl->graph_allowance_bytes =
                    checked_add(impl->graph_allowance_bytes, class_allowance(batch_size),
                                "DFlash exact-b graph allowance");
            }
        }
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(
            checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
            impl->request_transient_capacity_bytes, "request transient reservation"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace

std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile) {
    const std::uint32_t effective_max_context = validate_target_options(device, options);

    SequencePlanningInputs inputs{
        .weights_profile     = weights_profile,
        .capacity            = options.max_context,
        .max_concurrency     = options.max_concurrency,
        .prefill_chunk       = std::min(options.prefill_chunk, options.max_context),
        .draft_window        = options.speculative.draft_tokens,
        .speculative_backend = options.speculative.backend,
        .kv_dtype       = options.kv_cache == KvCacheStorage::BFloat16 ? DType::BF16 : DType::I8,
        .kv_quant_group = options.kv_cache == KvCacheStorage::BFloat16 ? 0 : qwen3_6::kKvQuantGroup,
        .proposal_head  = options.speculative.proposal_head,
        .features       = qwen3_6::startup_features(options),
        .rope_mode      = options.rope_mode,
        .yarn_factor    = options.yarn_factor,
        .yarn_origin    = options.yarn_origin,
        .effective_max_context = effective_max_context,
        // tp 2 captures like tp 1. `--no-cuda-graph` is the escape hatch that runs the same
        // forward pass eagerly (two streams, cross-device event sync) at either width.
        .use_cuda_graph = options.use_cuda_graph,
        .device         = options.device,
        .tp             = options.tp,
    };
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(inputs.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<qwen3_6::detail::SequencePlannerImpl<Variant>>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.6 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.6 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.6 physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
