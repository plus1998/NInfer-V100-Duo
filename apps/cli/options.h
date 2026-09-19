#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    // Rotary regime. `--rope yarn` raises the ceiling `--max-context` is checked against from the
    // artifact's registered native capacity to `--yarn-origin x --yarn-factor`; Engine performs
    // that check, since only it knows the loaded variant's native capacity.
    RopeMode rope_mode           = RopeMode::Native;
    double yarn_factor           = 4.0;
    std::uint32_t yarn_origin    = 262144;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t prefill_chunk  = 1024;
    int device                   = 0;
    int tp                       = 1;
    // Resolved device ids, one per tp rank. Always populated by parse_options() (from --devices,
    // or synthesized as {device} when --devices is omitted) so it is safe to forward directly to
    // EngineOptions::devices.
    std::vector<int> devices;

    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    bool enable_vision  = false;
    bool use_cuda_graph = true;

    bool raw_output      = false;
    bool print_token_ids = false;
    bool enable_thinking = true;
    // Drops the checkpoint's own `eos_token_id` set from the request stop policy, so decode runs
    // to the output-token budget or the remaining context capacity instead of terminating on the
    // model's end-of-turn token. Explicit --stop-token-id / --stop conditions still apply.
    bool ignore_eos      = false;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy = false;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::cli
