#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "runtime/engine/kv_capacity.h"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback = progress.callback};
}

// Save/restore the current device around a per-device query.
class ScopedDevice {
public:
    explicit ScopedDevice(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        CUDA_CHECK(cudaSetDevice(device));
    }

    ~ScopedDevice() { (void)cudaSetDevice(previous_); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    int previous_ = 0;
};

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes, int device) {
    const ScopedDevice scope(device);
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument("model weights require " + std::to_string(weight_bytes) +
                                    " bytes of device memory on device " +
                                    std::to_string(device) + ", but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

std::size_t current_free_device_bytes(int device) {
    const ScopedDevice scope(device);
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

std::size_t current_total_device_bytes(int device) {
    const ScopedDevice scope(device);
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return total_bytes;
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, ExecutionContext& execution,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {
    const auto& identity                          = reader.identity();
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);
    const int tp                                  = execution.tp;
    DeviceContext& device                         = execution.primary();

    artifact::Binder binder(reader, tp);
    auto load_plan = Target::plan_load(binder, options, weights_profile);
    // The capacity curve describes ONE device. At tp 2 it is already the per-device curve (halved
    // KV heads, halved GDN state) because the sequence plan is built with `tp`, so the same curve
    // serves both devices and the resolver only has to pick the bottleneck device's budget.
    auto sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    std::vector<std::size_t> preflight_runtime_bytes;
    preflight_runtime_bytes.reserve(static_cast<std::size_t>(tp));
    for (int rank = 0; rank < tp; ++rank) {
        preflight_runtime_bytes.push_back(runtime_bytes_after_planned_weights(
            load_plan.materialization().device_capacity_bytes[static_cast<std::size_t>(rank)],
            execution.dev[static_cast<std::size_t>(rank)]->device));
    }
    (void)runtime::resolve_kv_capacity_symmetric(options.kv_capacity, curve,
                                                 preflight_runtime_bytes);

    auto progress     = artifact_progress(options.load_progress);
    auto materialized = artifact::materialize(reader, load_plan.materialization(), execution,
                                              progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    for (int rank = 0; rank < tp; ++rank) {
        execution.dev[static_cast<std::size_t>(rank)]->synchronize();
    }
    std::vector<std::size_t> free_before_runtime;
    free_before_runtime.reserve(static_cast<std::size_t>(tp));
    for (int rank = 0; rank < tp; ++rank) {
        free_before_runtime.push_back(
            current_free_device_bytes(execution.dev[static_cast<std::size_t>(rank)]->device));
    }
    runtime::KvCapacityResolution capacity_resolution =
        runtime::resolve_kv_capacity_symmetric(options.kv_capacity, curve, free_before_runtime);
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    auto loaded   = std::make_unique<Loaded>(std::move(model), options);
    auto instance = std::make_unique<Instance>(std::move(loaded), capacity_resolution,
                                               std::move(sequence_plan), execution);
    for (int rank = 0; rank < tp; ++rank) {
        execution.dev[static_cast<std::size_t>(rank)]->synchronize();
    }
    instance->kv_capacity_resolution.available_after_startup_bytes =
        current_free_device_bytes(device.device);

    LoadSummary summary;
    // Rotary regime, as the target runtime resolved it. `effective_max_context` and `yarn_mscale`
    // come from the constructed program (only it knows the variant's native capacity and the YaRN
    // table it built); the mode/factor/origin echo what the caller asked for.
    summary.rope_mode  = options.rope_mode;
    summary.yarn_factor  = options.rope_mode == RopeMode::Yarn ? options.yarn_factor : 0.0;
    summary.yarn_origin  = options.rope_mode == RopeMode::Yarn ? options.yarn_origin : 0U;
    // Per-device memory table. `weights_bytes` and the free/total figures are MEASURED per device;
    // the sequence, KV, GDN and workspace figures come from the finalized plan, which is symmetric
    // across devices by construction (identical page counts, identical halved head geometry).
    const MemorySummary memory = instance->program->memory_summary();
    summary.effective_max_context = memory.effective_max_context;
    summary.yarn_mscale           = memory.yarn_mscale;
    summary.tp                 = tp;
    for (int rank = 0; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        DeviceMemoryReport& row = summary.devices[slot];
        row.device              = execution.dev[slot]->device;
        row.weights_bytes       = stats.per_device_capacity_bytes[slot];
        row.kv_pool_bytes       = memory.kv_payload_bytes;
        row.gdn_state_bytes     = memory.gdn_state_bytes;
        row.sequence_bytes      = memory.sequence.capacity_bytes;
        row.workspace_bytes     = memory.workspace.capacity_bytes;
        // Measured per device: instantiating the decode graphs materializes driver state on each
        // device the graph has nodes on, which at tp 2 is both of them.
        row.cuda_graph_bytes = rank == 0 ? memory.cuda_graph_observed_bytes
                                         : memory.cuda_graph_peer_observed_bytes;
        row.reserved_bytes = row.weights_bytes + capacity_resolution.runtime_reservation_bytes;
        row.free_after_startup_bytes = current_free_device_bytes(row.device);
        row.total_bytes              = current_total_device_bytes(row.device);
    }
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults};
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         ExecutionContext& execution)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(execution.primary(), sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), execution)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               ExecutionContext& execution)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(execution.primary(), sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan),
                                             execution)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, ExecutionContext& execution) {
    validate_options(options);
    if (execution.tp != options.tp) {
        throw std::invalid_argument("execution context width does not match EngineOptions.tp");
    }
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, execution, reader, load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, execution, reader, load_start, Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, execution, reader, load_start, Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
