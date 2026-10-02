#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "targets/qwen3_6_27b/impl/config.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/tp2_dflash_context.h"
#include "targets/qwen3_6_27b/impl/tp2_dflash_draft.h"
#include "targets/qwen3_6_27b/impl/tp2_dflash_proposal.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#include "ninfer/ops/speculative_round.h"

#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/round_state.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void expect(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

} // namespace

int main() {
    const char* configured = std::getenv("NINFER_QWEN3_8_27B_GSQ_WEIGHTS");
    const std::filesystem::path path = configured != nullptr && *configured != '\0'
                                           ? configured
                                           : std::filesystem::path(NINFER_SOURCE_DIR) / "out" /
                                                 "Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer";
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: DFlash2 GSQ artifact is unavailable: " << path << '\n';
        return 77;
    }

    ninfer::artifact::Reader reader(path);
    const auto profile = ninfer::targets::qwen3_6_27b::Package::resolve_weights(reader.identity());
    if (profile != ninfer::targets::qwen3_6_27b::detail::WeightsProfile::Qwen38Gguf) {
        std::cerr << "GSQ companion resolved to an unexpected weights profile\n";
        return 1;
    }
    const ninfer::targets::qwen3_6::StartupFeatures features{
        .vision = true,
        .speculative = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
    std::size_t companion_count = 0;
    for (const auto& object : reader.objects()) {
        if (ninfer::artifact::object_name(object).starts_with("dflash2/")) { ++companion_count; }
    }
    if (companion_count != 66) {
        std::cerr << "GSQ companion inventory must contain 66 tensors, got " << companion_count
                  << '\n';
        return 1;
    }
    for (int tp : {1, 2}) {
        ninfer::artifact::Binder binder(reader, tp);
        const auto plan = ninfer::targets::qwen3_6_27b::detail::bind_artifact(binder, profile,
                                                                                features, tp);
        const auto& placements = plan.materialization.device_objects;
        const bool resident = std::any_of(placements.begin(), placements.end(), [&](const auto& p) {
            return ninfer::artifact::object_name(reader.objects()[p.object.index])
                .starts_with("dflash2/");
        });
        if (resident) {
            std::cerr << "DFlash2 tensors were placed without an executable DFlash2 route\n";
            return 1;
        }
    }
    using namespace ninfer::targets::qwen3_6_27b::detail;
    const TextConfig config{};
    ninfer::artifact::Binder companion(reader, 2);
    companion.set_shard_resolver([&](std::string_view name) {
        const ShardMapping mapping = shard_mapping_for(name, 2, config, profile);
        ninfer::artifact::ShardPlacement placement;
        placement.axis = mapping.axis;
        for (const Shard& shard : mapping.shards) {
            placement.device_ranges[static_cast<std::size_t>(shard.device)].push_back(
                {shard.row_begin, shard.row_count});
        }
        return placement;
    });
    std::uint64_t original_bytes = 0;
    std::array<std::uint64_t, 2> shard_bytes{};
    for (const auto& object : reader.objects()) {
        const std::string_view name = ninfer::artifact::object_name(object);
        if (!name.starts_with("dflash2/")) { continue; }
        const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
        expect(tensor != nullptr, "DFlash2 companion object must be a tensor");
        const auto handle = companion.require_tensor(name, tensor->format, tensor->layout,
                                                      tensor->shape);
        companion.materialize_on_device(handle);
        original_bytes += tensor->bytes;
        const ShardMapping mapping = shard_mapping_for(name, 2, config, profile);
        if (mapping.axis == ninfer::artifact::ShardAxis::Replicated) {
            expect(mapping.shards.empty(), "replicated companion object has shard ranges");
        } else {
            std::uint64_t covered = 0;
            for (const Shard& shard : mapping.shards) {
                expect(shard.device >= 0 && shard.device < 2, "invalid companion rank");
                expect(shard.row_begin + shard.row_count <=
                           tensor->shape[mapping.axis == ninfer::artifact::ShardAxis::Rows ? 0 : 1],
                       "companion shard exceeds its source tensor");
                covered += shard.row_count;
            }
            expect(covered == tensor->shape[mapping.axis == ninfer::artifact::ShardAxis::Rows ? 0 : 1],
                   "companion shards do not cover the tensor exactly once");
        }
    }
    companion.validate_unconsumed_matching();
    const auto materialization = companion.finish();
    expect(materialization.device_objects.size() == 2 * companion_count,
           "DFlash2 companion requires one placement per rank");
    for (const auto& placement : materialization.device_objects) {
        const auto& object = reader.objects()[placement.object.index];
        const auto& tensor = std::get<ninfer::artifact::TensorDescriptor>(object);
        shard_bytes[static_cast<std::size_t>(placement.device)] += placement.bytes;
        const ShardMapping mapping = shard_mapping_for(tensor.name, 2, config, profile);
        expect(placement.bytes <= tensor.bytes, "shard is larger than the source tensor");
        if (mapping.axis == ninfer::artifact::ShardAxis::Replicated) {
            expect(placement.bytes == tensor.bytes && placement.copies.empty(),
                   "replicated DFlash2 payload differs from its source");
        } else {
            std::uint64_t copied = placement.prefix.size();
            for (const auto& copy : placement.copies) {
                expect(copy.source_offset + copy.bytes <= tensor.bytes &&
                           copy.dest_offset + copy.bytes <= placement.bytes,
                       "DFlash2 slice copy falls outside its source or destination");
                copied += copy.bytes;
            }
            expect(copied == placement.bytes, "DFlash2 slice does not fill its destination");
        }
    }
    std::cout << "DFlash2 companion TP2 plan: source " << original_bytes << " bytes, ranks "
              << shard_bytes[0] << " / " << shard_bytes[1] << " bytes\n";
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cout << "DFlash2 TP2 device-byte audit skipped: fewer than two CUDA devices\n";
        return 0;
    }
    ninfer::ExecutionContext execution({0, 1});
    {
        const auto materialized = ninfer::artifact::materialize(reader, materialization, execution);
        for (const auto& placement : materialization.device_objects) {
            const auto& object = reader.objects()[placement.object.index];
            const auto payload = reader.payload(object).data;
            CUDA_CHECK(cudaSetDevice(materialized.physical_device(placement.device)));
            const auto* device_bytes = static_cast<const std::byte*>(
                materialized.device_data(placement.object, placement.device));
            const auto sample = [&](std::uint64_t source, std::uint64_t destination,
                                    std::uint64_t available) {
                const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(available, 32));
                std::array<std::byte, 32> actual{};
                CUDA_CHECK(cudaMemcpy(actual.data(), device_bytes + destination, count,
                                      cudaMemcpyDeviceToHost));
                expect(std::memcmp(actual.data(), payload.data() + source, count) == 0,
                       "DFlash2 GPU slice differs from its original artifact payload");
            };
            if (placement.copies.empty()) {
                sample(0, 0, placement.bytes);
                sample(placement.bytes - 32, placement.bytes - 32, 32);
            } else {
                for (const auto index : {std::size_t(0), placement.copies.size() / 2,
                                         placement.copies.size() - 1}) {
                    const auto& copy = placement.copies[index];
                    sample(copy.source_offset, copy.dest_offset, copy.bytes);
                    if (copy.bytes >= 32) {
                        sample(copy.source_offset + copy.bytes - 32,
                               copy.dest_offset + copy.bytes - 32, 32);
                    }
                }
            }
        }
        std::cout << "DFlash2 companion TP2 device slices verified against GSQ bytes\n";
    }

    const ninfer::targets::qwen3_6::StartupFeatures draft_features{
        .vision = true,
        .speculative = ninfer::SpeculativeBackend::DFlash2,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
    ninfer::artifact::Binder draft_binder(reader, 2);
    auto draft_plan = bind_artifact(draft_binder, profile, draft_features, 2);
    ninfer::artifact::Binder mtp_binder(reader, 2);
    const auto mtp_plan = bind_artifact(mtp_binder, profile, features, 2);
    for (int rank = 0; rank < 2; ++rank) {
        const auto index = static_cast<std::size_t>(rank);
        std::cout << "rank " << rank << " selected DFlash2 weights arena "
                  << draft_plan.materialization.device_capacity_bytes[index]
                  << " bytes vs MTP arena " << mtp_plan.materialization.device_capacity_bytes[index]
                  << " bytes\n";
    }
    expect(draft_plan.bindings.dflash2.has_value(), "DFlash2 binding plan missing");
    std::array<std::size_t, 2> draft_placements{};
    for (const auto& placement : draft_plan.materialization.device_objects) {
        const auto name = ninfer::artifact::object_name(reader.objects()[placement.object.index]);
        expect(!name.starts_with("mtp/"), "DFlash2 must not place MTP weights");
        if (name.starts_with("dflash2/")) {
            ++draft_placements[static_cast<std::size_t>(placement.device)];
        }
    }
    expect(draft_placements[0] == companion_count && draft_placements[1] == companion_count,
           "DFlash2 binding must place all companion tensors on both ranks");
    auto draft_artifact =
        ninfer::artifact::materialize(reader, draft_plan.materialization, execution);
    LoadedModelData loaded(std::move(draft_plan.bindings), std::move(draft_artifact), 2);
    for (int rank = 0; rank < 2; ++rank) {
        const auto& view = loaded.view(rank);
        expect(view.dflash2.has_value() && !view.mtp.has_value(),
               "DFlash2 runtime view must exclude MTP");
        const auto& weights = *view.dflash2;
        expect(weights.feature_projection.n == 5120 && weights.feature_projection.k == 12800,
               "DFlash2 feature projection shard has wrong shape");
        for (const auto& layer : weights.layers) {
            expect(layer.query_key_value.n == 3072 && layer.query.n == 2048 &&
                       layer.key.n == 512 && layer.value.n == 512 &&
                       layer.attention_output.k == 2048 && layer.gate_up.n == 17408 &&
                       layer.down.k == 8704 && layer.attention_conv.kernel_projection.n == 1280,
                   "DFlash2 layer shard has wrong runtime shape");
        }
        expect(weights.predecessor_codebook.ne[1] == 248320 &&
                   weights.successor_codebook.ne[1] == 248320,
               "DFlash2 selector codebooks must be replicated");
    }
    constexpr int batch = 3;
    using DFlashExtension = ninfer::targets::qwen3_6::detail::TP2DFlashExtension<
        ninfer::targets::qwen3_6_27b::detail::Variant>;
    ninfer::LayoutBuilder cache_builder;
    expect(!DFlashExtension::plan(cache_builder, batch, batch, 8, false).has_value(),
           "ordinary 27B layout allocated DFlash2 context");
    const auto optional_layout = DFlashExtension::plan(cache_builder, batch, batch, 8, true);
    expect(optional_layout.has_value(), "27B Program did not plan DFlash2 context");
    const auto& persistent_layout = *optional_layout;
    expect(DFlashExtension::kv_payload_bytes(optional_layout) ==
               persistent_layout.cache.payload_bytes() +
                   persistent_layout.rewrite_checkpoint.payload_bytes(),
           "27B Program did not account for DFlash2 cyclic KV memory");
    const auto cache_bytes = cache_builder.finish(256);
    struct ContextRank {
        ninfer::DeviceBuffer counts;
        ninfer::DeviceBuffer slots;
        ninfer::DeviceBuffer projected;
        ninfer::DeviceBuffer staging;
        ninfer::DeviceBuffer normalized;
        ninfer::DeviceBuffer scratch;
        ninfer::DeviceBuffer persistent_backing;
        std::optional<TP2DFlashContextState> persistent;
    };
    std::array<ContextRank, 2> context_ranks;
    std::array<ninfer::Tensor, 2> feature_views, position_views, count_views, slot_views;
    std::array<const DFlash2Weights*, 2> companion_weights;
    std::array<std::array<ninfer::CyclicKVCacheLayerView, 5>, 2> cache_views;
    std::array<ninfer::WorkspaceArena*, 2> scratch_views;
    std::array<std::unique_ptr<ninfer::WorkspaceArena>, 2> arenas;
    TP2DFlashContextBuffers context_buffers;
    std::vector<std::uint16_t> input_features(12800 * batch);
    for (std::size_t row = 0; row < input_features.size(); ++row) {
        input_features[row] = static_cast<std::uint16_t>(0x3b00 + row % 127);
    }
    const std::array<int, batch> positions{3, 17, 2049};
    const std::array<int, batch> counts{1, 1, 1};
    const std::array<int, batch> slots{0, 1, 2};
    const auto scratch_bytes =
        ninfer::ops::context_kv_materialize_workspace_capacity_bytes(batch, 1, 1);
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        auto& buffers = context_ranks[rank];
        buffers.persistent_backing = ninfer::DeviceBuffer(cache_bytes);
        buffers.persistent_backing.fill(0xa5);
        DFlashExtension::bind(
            buffers.persistent,
            ninfer::DeviceSpan{buffers.persistent_backing.p, buffers.persistent_backing.bytes},
            optional_layout);
        expect(buffers.persistent.has_value(), "27B Program did not bind DFlash2 context");
        CUDA_CHECK(cudaMemcpy(buffers.persistent->prefill_features.data, input_features.data(),
                              input_features.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buffers.persistent->prefill_positions.data, positions.data(),
                              sizeof(positions), cudaMemcpyHostToDevice));
        buffers.counts = ninfer::DeviceBuffer(sizeof(counts));
        buffers.counts.copy_from_host(counts.data(), sizeof(counts));
        buffers.slots = ninfer::DeviceBuffer(sizeof(slots));
        buffers.slots.copy_from_host(slots.data(), sizeof(slots));
        buffers.projected = ninfer::DeviceBuffer(5120 * batch * sizeof(std::uint16_t));
        buffers.staging = ninfer::DeviceBuffer(5120 * batch * sizeof(std::uint16_t));
        buffers.normalized = ninfer::DeviceBuffer(5120 * batch * sizeof(std::uint16_t));
        buffers.scratch = ninfer::DeviceBuffer(scratch_bytes);
        arenas[rank] = std::make_unique<ninfer::WorkspaceArena>(
            ninfer::DeviceSpan{buffers.scratch.p, buffers.scratch.bytes});
        scratch_views[rank] = arenas[rank].get();
        feature_views[rank] = buffers.persistent->prefill_features;
        position_views[rank] = buffers.persistent->prefill_positions.view({1, batch});
        count_views[rank] = ninfer::Tensor(buffers.counts.p, ninfer::DType::I32, {batch});
        slot_views[rank] = ninfer::Tensor(buffers.slots.p, ninfer::DType::I32, {batch});
        context_buffers.projected[rank] =
            ninfer::Tensor(buffers.projected.p, ninfer::DType::BF16, {5120, 1, batch});
        context_buffers.staging[rank] =
            ninfer::Tensor(buffers.staging.p, ninfer::DType::BF16, {5120, 1, batch});
        context_buffers.normalized[rank] =
            ninfer::Tensor(buffers.normalized.p, ninfer::DType::BF16, {5120, 1, batch});
        companion_weights[rank] = &*loaded.view(rank).dflash2;
        cache_views[rank] = buffers.persistent->layers();
    }
    ninfer::ops::PeerEvents events(execution, ninfer::ops::enable_peer_access(execution));
    materialize_tp2_dflash_context(feature_views, position_views, count_views, slot_views,
                                   companion_weights, cache_views, context_buffers, {1, 1},
                                   scratch_views, execution, events);
    std::array<std::vector<std::uint16_t>, 2> projected_results;
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        execution.dev[rank]->synchronize();
        projected_results[rank].resize(5120 * batch);
        context_ranks[rank].projected.copy_to_host(projected_results[rank].data(),
                                                  5120 * batch * sizeof(std::uint16_t));
        expect(std::any_of(projected_results[rank].begin(), projected_results[rank].end(),
                           [](auto value) { return (value & 0x7fffU) != 0; }),
               "DFlash2 feature projection produced only zero output");
        for (int layer = 0; layer < 5; ++layer) {
            std::array<std::uint16_t, 128> written{};
            for (int lane = 0; lane < batch; ++lane) {
                const auto offset = static_cast<std::size_t>(lane * 4 * 2048 +
                                      positions[lane] % 2048) * 128 * sizeof(std::uint16_t);
                for (const auto& plane : {cache_views[rank][layer].k,
                                          cache_views[rank][layer].v}) {
                    const auto* start = static_cast<const std::byte*>(plane.data);
                    CUDA_CHECK(cudaMemcpy(written.data(), start + offset, sizeof(written),
                                          cudaMemcpyDeviceToHost));
                    expect(std::all_of(written.begin(), written.end(),
                                       [](auto value) { return value != 0xa5a5; }),
                           "DFlash2 context did not materialize a selected lane");
                    CUDA_CHECK(cudaMemcpy(written.data(),
                                          start + offset + 128 * sizeof(std::uint16_t),
                                          sizeof(written), cudaMemcpyDeviceToHost));
                    expect(std::all_of(written.begin(), written.end(),
                                       [](auto value) { return value == 0xa5a5; }),
                           "DFlash2 context modified an uncommitted slot");
                }
            }
        }
        ninfer::DeviceBuffer checkpoint_backing(cache_bytes);
        checkpoint_backing.fill(0xa5);
        ninfer::CyclicKVCache checkpoint(
            ninfer::DeviceSpan{checkpoint_backing.p, checkpoint_backing.bytes},
            persistent_layout.cache);
        checkpoint.copy_lane_from(context_ranks[rank].persistent->cache, 1,
                                  execution.dev[rank]->stream);
        execution.dev[rank]->synchronize();
        std::array<std::uint16_t, 128> copied{};
        const auto checkpoint_view = checkpoint.layer_view(4);
        const auto checkpoint_offset = static_cast<std::size_t>(4 * 2048 + positions[1]) * 128 *
                                       sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemcpy(copied.data(),
                              static_cast<const std::byte*>(checkpoint_view.v.data) +
                                  checkpoint_offset,
                              sizeof(copied), cudaMemcpyDeviceToHost));
        expect(std::all_of(copied.begin(), copied.end(),
                           [](auto value) { return value != 0xa5a5; }),
               "DFlash2 FP16 checkpoint missed the committed lane");
        CUDA_CHECK(cudaMemcpy(copied.data(), checkpoint_view.v.data, sizeof(copied),
                              cudaMemcpyDeviceToHost));
        expect(std::all_of(copied.begin(), copied.end(),
                           [](auto value) { return value == 0xa5a5; }),
               "DFlash2 checkpoint modified another lane");
        ninfer::LayoutBuilder incompatible_builder;
        const auto incompatible_layout = ninfer::plan_cyclic_kv_cache(
            incompatible_builder, 5, 2048, 4, 128, batch);
        ninfer::DeviceBuffer incompatible_backing(incompatible_builder.finish(256));
        ninfer::CyclicKVCache incompatible(
            ninfer::DeviceSpan{incompatible_backing.p, incompatible_backing.bytes},
            incompatible_layout);
        bool rejected = false;
        try {
            incompatible.copy_lane_from(context_ranks[rank].persistent->cache, 1,
                                        execution.dev[rank]->stream);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "DFlash2 FP16 state copy accepted an incompatible BF16 cache");
    }
    expect(projected_results[0] == projected_results[1],
           "DFlash2 TP2 projection differs across ranks");
    std::array<TP2DFlashContextState*, 2> state = {
        &*context_ranks[0].persistent, &*context_ranks[1].persistent};
    std::array<std::array<ninfer::DeviceBuffer, 5>, 2> residual_storage;
    std::array<std::array<ninfer::Tensor, 5>, 2> residual_views;
    std::array<ninfer::DeviceBuffer, 2> source_position_storage;
    std::array<ninfer::Tensor, 2> source_positions;
    const std::array<int, batch> prefill_positions{20, 21, 22};
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        auto& buffers = context_ranks[rank];
        source_position_storage[rank] = ninfer::DeviceBuffer(sizeof(prefill_positions));
        source_position_storage[rank].copy_from_host(prefill_positions.data(),
                                                     sizeof(prefill_positions));
        source_positions[rank] = ninfer::Tensor(source_position_storage[rank].p,
                                                ninfer::DType::I32, {batch});
        for (int layer = 0; layer < 5; ++layer) {
            std::vector<std::uint16_t> residual(5120 * batch);
            for (std::size_t row = 0; row < residual.size(); ++row) {
                residual[row] = static_cast<std::uint16_t>(0x3a00 + (row + layer * 37 + rank) % 191);
            }
            residual_storage[rank][layer] =
                ninfer::DeviceBuffer(residual.size() * sizeof(std::uint16_t));
            residual_storage[rank][layer].copy_from_host(residual.data(),
                                                         residual.size() * sizeof(std::uint16_t));
            residual_views[rank][layer] = ninfer::Tensor(residual_storage[rank][layer].p,
                                                          ninfer::DType::BF16, {5120, batch});
        }
    }
    auto sink = DFlashExtension::make_prefill_sink(
        state, {&loaded.view(0), &loaded.view(1)}, execution, events, 1);
    sink.begin({residual_views[0][0], residual_views[1][0]});
    for (int layer = 0; layer < 5; ++layer) {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
            sink.capture_layer(rank, kDFlash2TargetFeatureLayers[layer],
                               residual_views[rank][layer], execution.dev[rank]->stream);
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        sink.capture_positions(rank, source_positions[rank], execution.dev[rank]->stream);
    }
    sink.consume_prefill_chunk(true);
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        execution.dev[rank]->synchronize();
        const auto cache_layer = state[rank]->cache.layer_view(4);
        const auto checkpoint_layer = state[rank]->rewrite_checkpoint.layer_view(4);
        const std::size_t last_position = static_cast<std::size_t>(4 * 2048 + 22) * 128 *
                                          sizeof(std::uint16_t);
        for (const auto& planes : {std::array{cache_layer.k, checkpoint_layer.k},
                                   std::array{cache_layer.v, checkpoint_layer.v}}) {
            std::array<std::uint16_t, 128> live{}, saved{};
            CUDA_CHECK(cudaMemcpy(live.data(),
                                  static_cast<const std::byte*>(planes[0].data) + last_position,
                                  sizeof(live), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(saved.data(),
                                  static_cast<const std::byte*>(planes[1].data) + last_position,
                                  sizeof(saved), cudaMemcpyDeviceToHost));
            expect(live == saved && std::any_of(live.begin(), live.end(),
                                                [](auto value) { return value != 0xa5a5; }),
                   "DFlash2 TP2 prefill checkpoint differs from committed context");
        }
        const auto live_v = cache_layer.v.slice(3, 1, 1);
        CUDA_CHECK(cudaMemsetAsync(live_v.data, 0xa5, live_v.bytes(),
                                   execution.dev[rank]->stream));
        state[rank]->restore_checkpoint(1, execution.dev[rank]->stream);
        execution.dev[rank]->synchronize();
        std::array<std::uint16_t, 128> restored{}, saved{};
        CUDA_CHECK(cudaMemcpy(restored.data(),
                              static_cast<const std::byte*>(cache_layer.v.data) + last_position,
                              sizeof(restored), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(saved.data(),
                              static_cast<const std::byte*>(checkpoint_layer.v.data) +
                                  last_position,
                              sizeof(saved), cudaMemcpyDeviceToHost));
        expect(restored == saved &&
                   std::any_of(restored.begin(), restored.end(),
                               [](auto value) { return value != 0xa5a5; }),
               "DFlash2 TP2 checkpoint did not restore the selected lane");
    }
    constexpr int draft_width = 8;
    struct PendingRank {
        std::array<ninfer::DeviceBuffer, 5> residuals;
        ninfer::DeviceBuffer positions;
        ninfer::DeviceBuffer projected;
        ninfer::DeviceBuffer staging;
        ninfer::DeviceBuffer normalized;
        ninfer::DeviceBuffer scratch;
        std::unique_ptr<ninfer::WorkspaceArena> arena;
    };
    std::array<PendingRank, 2> pending_ranks;
    TP2DFlashContextBuffers pending_buffers;
    std::array<ninfer::WorkspaceArena*, 2> pending_workspace;
    std::array<std::array<ninfer::Tensor, 5>, 2> verify_residuals;
    std::array<ninfer::Tensor, 2> verify_positions;
    std::array<std::int32_t, draft_width * batch> pending_positions{};
    for (int lane = 0; lane < batch; ++lane) {
        for (int column = 0; column < draft_width; ++column) {
            pending_positions[lane * draft_width + column] =
                33 + lane * 22 + column;
        }
    }
    const std::array<std::int32_t, batch> committed_counts{2, 0, 1};
    const std::array<std::int32_t, batch> committed_slots{0, 1, 2};
    const auto pending_scratch_bytes = ninfer::ops::context_kv_materialize_workspace_capacity_bytes(
        batch, draft_width, draft_width);
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        auto& rank_buffers = pending_ranks[rank];
        for (int layer = 0; layer < 5; ++layer) {
            std::vector<std::uint16_t> residual(5120 * draft_width * batch);
            for (std::size_t element = 0; element < residual.size(); ++element) {
                residual[element] = static_cast<std::uint16_t>(
                    0x3a80 + (element + layer * 31 + rank * 19) % 127);
            }
            rank_buffers.residuals[layer] = ninfer::DeviceBuffer(residual.size() * sizeof(std::uint16_t));
            rank_buffers.residuals[layer].copy_from_host(residual.data(),
                                                          residual.size() * sizeof(std::uint16_t));
            verify_residuals[rank][layer] = ninfer::Tensor(
                rank_buffers.residuals[layer].p, ninfer::DType::BF16,
                {5120, draft_width * batch});
        }
        rank_buffers.positions = ninfer::DeviceBuffer(sizeof(pending_positions));
        rank_buffers.positions.copy_from_host(pending_positions.data(), sizeof(pending_positions));
        verify_positions[rank] = ninfer::Tensor(rank_buffers.positions.p, ninfer::DType::I32,
                                               {draft_width * batch});
        CUDA_CHECK(cudaMemcpyAsync(state[rank]->pending_counts.data, committed_counts.data(),
                                   sizeof(committed_counts), cudaMemcpyHostToDevice,
                                   execution.dev[rank]->stream));
        CUDA_CHECK(cudaMemcpyAsync(state[rank]->pending_slots.data, committed_slots.data(),
                                   sizeof(committed_slots), cudaMemcpyHostToDevice,
                                   execution.dev[rank]->stream));
        const auto bytes = 5120 * draft_width * batch * sizeof(std::uint16_t);
        rank_buffers.projected = ninfer::DeviceBuffer(bytes);
        rank_buffers.staging = ninfer::DeviceBuffer(bytes);
        rank_buffers.normalized = ninfer::DeviceBuffer(bytes);
        rank_buffers.scratch = ninfer::DeviceBuffer(pending_scratch_bytes);
        rank_buffers.arena = std::make_unique<ninfer::WorkspaceArena>(
            ninfer::DeviceSpan{rank_buffers.scratch.p, rank_buffers.scratch.bytes});
        pending_workspace[rank] = rank_buffers.arena.get();
        pending_buffers.projected[rank] = ninfer::Tensor(
            rank_buffers.projected.p, ninfer::DType::BF16, {5120, draft_width, batch});
        pending_buffers.staging[rank] = ninfer::Tensor(
            rank_buffers.staging.p, ninfer::DType::BF16, {5120, draft_width, batch});
        pending_buffers.normalized[rank] = ninfer::Tensor(
            rank_buffers.normalized.p, ninfer::DType::BF16, {5120, draft_width, batch});
    }
    auto verify_sink = make_tp2_dflash_verify_sink(state, batch);
    verify_sink.begin({verify_residuals[0][0], verify_residuals[1][0]});
    for (int layer = 0; layer < 5; ++layer) {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
            verify_sink.capture_layer(rank, kDFlash2TargetFeatureLayers[layer],
                                      verify_residuals[rank][layer], execution.dev[rank]->stream);
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        verify_sink.capture_positions(rank, verify_positions[rank], execution.dev[rank]->stream);
    }
    materialize_tp2_dflash_pending(verify_sink, state, companion_weights, pending_buffers,
                                   pending_workspace, execution, events, batch);
    std::array<std::array<std::uint16_t, 128>, 2> committed_key{};
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        execution.dev[rank]->synchronize();
        const auto& cache_layer = state[rank]->cache.layer_view(4);
        const auto* cache_k = static_cast<const std::byte*>(cache_layer.k.data);
        const auto read_key = [&](int lane, int position, std::array<std::uint16_t, 128>& output) {
            const auto offset = static_cast<std::size_t>(lane * 4 * 2048 + position) *
                                128 * sizeof(std::uint16_t);
            CUDA_CHECK(cudaMemcpy(output.data(), cache_k + offset, sizeof(output),
                                  cudaMemcpyDeviceToHost));
        };
        read_key(0, 33, committed_key[rank]);
        expect(std::any_of(committed_key[rank].begin(), committed_key[rank].end(),
                           [](std::uint16_t value) { return value != 0xa5a5; }),
               "DFlash2 pending committed position was not materialized");
        std::array<std::uint16_t, 128> second{}, rejected{}, third{};
        read_key(0, 34, second);
        read_key(1, 55, rejected);
        read_key(2, 77, third);
        expect(std::any_of(second.begin(), second.end(),
                           [](std::uint16_t value) { return value != 0xa5a5; }) &&
                   std::all_of(rejected.begin(), rejected.end(),
                               [](std::uint16_t value) { return value == 0xa5a5; }) &&
                   std::any_of(third.begin(), third.end(),
                               [](std::uint16_t value) { return value != 0xa5a5; }),
               "DFlash2 pending commit crossed a request's valid prefix");
    }
    ninfer::DecodeGraphPeerBridge pending_bridge(execution.dev[0]->device,
                                                 execution.dev[1]->device);
    ninfer::DecodeGraphDefinition pending_graph;
    pending_graph.capture(execution.dev[0]->stream, [&] {
        materialize_tp2_dflash_pending(verify_sink, state, companion_weights, pending_buffers,
                                       pending_workspace, execution, events, batch);
    }, {&pending_bridge, execution.dev[1]->stream});
    ninfer::DecodeGraphExecutable pending_executable;
    pending_executable.instantiate(pending_graph);
    pending_bridge.gate_launch(execution.dev[1]->stream, execution.dev[0]->stream);
    pending_executable.launch(execution.dev[0]->stream);
    execution.dev[0]->synchronize();
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        std::array<std::uint16_t, 128> replayed{};
        const auto& cache_layer = state[rank]->cache.layer_view(4);
        const auto offset = static_cast<std::size_t>(33) * 128 * sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemcpy(replayed.data(),
                              static_cast<const std::byte*>(cache_layer.k.data) + offset,
                              sizeof(replayed), cudaMemcpyDeviceToHost));
        expect(replayed == committed_key[rank],
               "DFlash2 pending context Graph replay changed committed K");
    }
    const std::array<int, batch> context_positions{0, 0, 0};
    std::array<ninfer::Tensor, 2> context_features, context_position_views;
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        CUDA_CHECK(cudaMemcpyAsync(state[rank]->prefill_positions.data, context_positions.data(),
                                   sizeof(context_positions), cudaMemcpyHostToDevice,
                                   execution.dev[rank]->stream));
        context_features[rank] = feature_views[rank].view({12800, 1, batch});
        context_position_views[rank] = position_views[rank];
    }
    materialize_tp2_dflash_context(context_features, context_position_views, count_views,
                                   slot_views, companion_weights, cache_views, context_buffers,
                                   {1, 1}, scratch_views, execution, events);
    struct DraftRank {
        ninfer::DeviceBuffer anchors;
        ninfer::DeviceBuffer frontiers;
        ninfer::DeviceBuffer roots_backing;
        ninfer::DeviceBuffer op_backing;
        std::unique_ptr<ninfer::WorkspaceArena> roots;
        std::unique_ptr<ninfer::WorkspaceArena> ops;
    };
    std::array<DraftRank, 2> drafts;
    std::array<ninfer::Tensor, 2> draft_residual, draft_hidden, draft_positions, draft_valid;
    std::array<ninfer::Tensor, 2> draft_lanes;
    std::array<ninfer::WorkspaceArena*, 2> draft_roots, draft_ops;
    const auto draft_capacity = tp2_dflash_draft_workspace_capacity(
        draft_width, batch, {1, 1});
    std::vector<std::uint16_t> draft_input(5120 * draft_width * batch);
    std::array<int, draft_width * batch> draft_position_values{};
    for (int row = 0; row < batch; ++row) {
        for (int column = 0; column < draft_width; ++column) {
            draft_position_values[row * draft_width + column] = column + 1;
        }
    }
    const std::array<int, batch> draft_lengths{draft_width, draft_width, draft_width};
    const std::array<int, batch> draft_slots{0, 1, 2};
    const std::array<int, batch> draft_anchors{13, 27, 41};
    const std::array<int, batch> draft_frontiers{1, 1, 1};
    std::array<ninfer::Tensor, 2> draft_anchor_views, draft_frontier_views;
    std::array<const RuntimeModelView*, 2> draft_models{};
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        auto& buffers = drafts[rank];
        buffers.anchors = ninfer::DeviceBuffer(sizeof(draft_anchors));
        buffers.anchors.copy_from_host(draft_anchors.data(), sizeof(draft_anchors));
        buffers.frontiers = ninfer::DeviceBuffer(sizeof(draft_frontiers));
        buffers.frontiers.copy_from_host(draft_frontiers.data(), sizeof(draft_frontiers));
        CUDA_CHECK(cudaMemcpy(state[rank]->draft_valid_columns.data, draft_lengths.data(),
                              sizeof(draft_lengths), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state[rank]->draft_lanes.data, draft_slots.data(),
                              sizeof(draft_slots), cudaMemcpyHostToDevice));
        buffers.roots_backing = ninfer::DeviceBuffer(draft_capacity.roots);
        buffers.op_backing = ninfer::DeviceBuffer(draft_capacity.ops);
        buffers.roots = std::make_unique<ninfer::WorkspaceArena>(
            ninfer::DeviceSpan{buffers.roots_backing.p, buffers.roots_backing.bytes});
        buffers.ops = std::make_unique<ninfer::WorkspaceArena>(
            ninfer::DeviceSpan{buffers.op_backing.p, buffers.op_backing.bytes});
        draft_roots[rank] = buffers.roots.get();
        draft_ops[rank] = buffers.ops.get();
        draft_residual[rank] = state[rank]->draft_input.view({5120, draft_width, batch});
        draft_hidden[rank] = state[rank]->draft_hidden.view({5120, draft_width, batch});
        draft_positions[rank] = state[rank]->draft_positions.view({draft_width, batch});
        draft_valid[rank] = state[rank]->draft_valid_columns.view({batch});
        draft_lanes[rank] = state[rank]->draft_lanes.view({batch});
        draft_anchor_views[rank] = ninfer::Tensor(buffers.anchors.p, ninfer::DType::I32, {batch});
        draft_frontier_views[rank] = ninfer::Tensor(buffers.frontiers.p, ninfer::DType::I32,
                                                   {batch});
        draft_models[rank] = &loaded.view(rank);
    }
    prepare_tp2_dflash_draft_inputs(state, draft_models, draft_anchor_views,
                                    draft_frontier_views, draft_valid, batch, execution);
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        execution.dev[rank]->synchronize();
        std::array<int, draft_width * batch> ids{}, positions{};
        CUDA_CHECK(cudaMemcpy(ids.data(), state[rank]->draft_ids.data, sizeof(ids),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(positions.data(), state[rank]->draft_positions.data,
                              sizeof(positions), cudaMemcpyDeviceToHost));
        for (int row = 0; row < batch; ++row) {
            for (int column = 0; column < draft_width; ++column) {
                expect(ids[row * draft_width + column] ==
                           (column == 0 ? draft_anchors[row] : kDFlash2MaskToken) &&
                           positions[row * draft_width + column] ==
                               draft_position_values[row * draft_width + column],
                       "DFlash2 draft anchor/mask input or position is incorrect");
            }
        }
        std::vector<std::uint16_t> embedded(draft_input.size());
        CUDA_CHECK(cudaMemcpy(embedded.data(), draft_residual[rank].data,
                              draft_residual[rank].bytes(), cudaMemcpyDeviceToHost));
        if (rank == 0) draft_input = std::move(embedded);
        else expect(embedded == draft_input, "DFlash2 embedded inputs differ between ranks");
    }
    tp2_dflash_draft_backbone(draft_residual, draft_positions, draft_valid, draft_lanes,
                               companion_weights, state, {1, 1}, draft_hidden, draft_roots,
                               draft_ops, execution, events);
    std::array<std::vector<std::uint16_t>, 2> draft_results;
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        execution.dev[rank]->synchronize();
        draft_results[rank].resize(draft_input.size());
        CUDA_CHECK(cudaMemcpy(draft_results[rank].data(), draft_hidden[rank].data,
                              draft_hidden[rank].bytes(), cudaMemcpyDeviceToHost));
        expect(std::all_of(draft_results[rank].begin(), draft_results[rank].end(),
                           [](auto value) { return (value & 0x7f80U) != 0x7f80U; }),
               "DFlash2 TP2 draft backbone returned nonfinite hidden");
    }
    expect(draft_results[0] == draft_results[1],
           "DFlash2 TP2 draft backbone residual replicas differ");
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        for (int layer = 0; layer < 5; ++layer) {
            const auto cache = state[rank]->cache.layer_view(layer);
            for (int lane = 0; lane < batch; ++lane) {
                for (int head = 0; head < 4; ++head) {
                    auto* value = static_cast<std::byte*>(cache.v.data) +
                                  static_cast<std::size_t>(lane * 4 + head) * 2048 * 128 *
                                      sizeof(std::uint16_t);
                    CUDA_CHECK(cudaMemsetAsync(value, 0x40, 128 * sizeof(std::uint16_t),
                                               execution.dev[rank]->stream));
                }
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(draft_residual[rank].data, draft_input.data(),
                                   draft_residual[rank].bytes(), cudaMemcpyHostToDevice,
                                   execution.dev[rank]->stream));
    }
    tp2_dflash_draft_backbone(draft_residual, draft_positions, draft_valid, draft_lanes,
                              companion_weights, state, {1, 1}, draft_hidden, draft_roots,
                              draft_ops, execution, events);
    std::array<std::vector<std::uint16_t>, 2> changed_results;
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        execution.dev[rank]->synchronize();
        changed_results[rank].resize(draft_input.size());
        CUDA_CHECK(cudaMemcpy(changed_results[rank].data(), draft_hidden[rank].data,
                              draft_hidden[rank].bytes(), cudaMemcpyDeviceToHost));
        expect(changed_results[rank] != draft_results[rank],
               "DFlash2 TP2 draft backbone ignored committed context values");
    }
    expect(changed_results[0] == changed_results[1],
           "DFlash2 TP2 draft backbone context update diverged across ranks");
    const auto proposal_capacity = tp2_dflash_proposal_capacity(
        loaded.view(0).optimized_proposal->head.qtype,
        loaded.view(0).optimized_proposal->head.n, draft_width, batch);
    struct ProposalRank {
        ninfer::DeviceBuffer root_backing;
        ninfer::DeviceBuffer topk_backing;
        std::unique_ptr<ninfer::WorkspaceArena> root;
        std::unique_ptr<ninfer::WorkspaceArena> topk;
    };
    std::array<ProposalRank, 2> proposals;
    std::array<ninfer::WorkspaceArena*, 2> proposal_roots, proposal_topk;
    std::array<ninfer::Weight, 2> proposal_heads;
    std::array<ninfer::Tensor, 2> proposal_maps;
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        auto& buffers = proposals[rank];
        buffers.root_backing = ninfer::DeviceBuffer(proposal_capacity.roots[rank]);
        buffers.topk_backing = ninfer::DeviceBuffer(proposal_capacity.topk);
        buffers.root = std::make_unique<ninfer::WorkspaceArena>(
            ninfer::DeviceSpan{buffers.root_backing.p, buffers.root_backing.bytes});
        buffers.topk = std::make_unique<ninfer::WorkspaceArena>(
            ninfer::DeviceSpan{buffers.topk_backing.p, buffers.topk_backing.bytes});
        proposal_roots[rank] = buffers.root.get();
        proposal_topk[rank] = buffers.topk.get();
        proposal_heads[rank] = loaded.view(rank).optimized_proposal->head;
        proposal_maps[rank] = loaded.view(rank).optimized_proposal->token_ids;
    }
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    const std::array<std::int32_t, batch> anchor_values{13, 27, 41};
    const std::array<std::int32_t, batch> base_values{2, 2, 2};
    std::array<ninfer::ops::SamplingConfig, batch> sampling{};
    ninfer::DeviceBuffer anchor_backing(sizeof(anchor_values));
    ninfer::DeviceBuffer base_backing(sizeof(base_values));
    ninfer::DeviceBuffer sampling_backing(sizeof(sampling));
    anchor_backing.copy_from_host(anchor_values.data(), sizeof(anchor_values));
    base_backing.copy_from_host(base_values.data(), sizeof(base_values));
    sampling_backing.copy_from_host(sampling.data(), sizeof(sampling));
    ninfer::Tensor anchors(anchor_backing.p, ninfer::DType::I32, {batch});
    ninfer::Tensor base_positions(base_backing.p, ninfer::DType::I32, {batch});
    std::array<ninfer::Tensor, 2> selected_ranks, proposal_id_ranks, proposal_q_ranks;
    for (int rank = 0; rank < 2; ++rank) {
        selected_ranks[rank] = state[rank]->draft_tokens.view({draft_width - 1, batch});
        proposal_id_ranks[rank] = state[rank]->proposal_ids.view({16, draft_width - 1, batch});
        proposal_q_ranks[rank] = state[rank]->proposal_q.view({16, draft_width - 1, batch});
    }
    const auto copy_output = [](const ninfer::Tensor& tensor, void* destination,
                                std::size_t bytes) {
        CUDA_CHECK(cudaMemcpy(destination, tensor.data, bytes, cudaMemcpyDeviceToHost));
    };
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    tp2_dflash_propose(draft_hidden, companion_weights, proposal_heads, &proposal_maps,
                       anchors, base_positions,
                       static_cast<const ninfer::ops::SamplingConfig*>(sampling_backing.p),
                       selected_ranks, proposal_id_ranks, proposal_q_ranks, proposal_roots,
                       proposal_topk, execution, events);
    std::array<std::int32_t, (draft_width - 1) * batch> selected_host{};
    std::array<std::int32_t, 16 * (draft_width - 1) * batch> proposal_ids_host{};
    std::array<float, 16 * (draft_width - 1) * batch> proposal_q_host{};
    execution.dev[0]->synchronize();
    copy_output(selected_ranks[0], selected_host.data(), sizeof(selected_host));
    copy_output(proposal_id_ranks[0], proposal_ids_host.data(), sizeof(proposal_ids_host));
    copy_output(proposal_q_ranks[0], proposal_q_host.data(), sizeof(proposal_q_host));
    execution.dev[1]->synchronize();
    CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
    std::array<std::int32_t, (draft_width - 1) * batch> peer_selected_host{};
    std::array<std::int32_t, 16 * (draft_width - 1) * batch> peer_ids_host{};
    std::array<float, 16 * (draft_width - 1) * batch> peer_q_host{};
    copy_output(selected_ranks[1], peer_selected_host.data(), sizeof(peer_selected_host));
    copy_output(proposal_id_ranks[1], peer_ids_host.data(), sizeof(peer_ids_host));
    copy_output(proposal_q_ranks[1], peer_q_host.data(), sizeof(peer_q_host));
    expect(selected_host == peer_selected_host && proposal_ids_host == peer_ids_host &&
               proposal_q_host == peer_q_host,
           "DFlash2 TP2 proposal did not publish matching verifier inputs on both ranks");
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    for (int column = 0; column < (draft_width - 1) * batch; ++column) {
        expect(selected_host[column] >= 0 && selected_host[column] < 248077,
               "DFlash2 TP2 proposal returned an invalid token id");
        int selected_count = 0;
        for (int candidate = 0; candidate < 16; ++candidate) {
            expect(proposal_ids_host[column * 16 + candidate] >= 0 &&
                       proposal_ids_host[column * 16 + candidate] < 248077,
                   "DFlash2 TP2 proposal produced an invalid candidate id");
            const auto probability = proposal_q_host[column * 16 + candidate];
            expect(probability == 0.0F || probability == 1.0F,
                   "DFlash2 TP2 greedy proposal must be one-hot");
            selected_count += probability == 1.0F;
            if (probability == 1.0F) {
                expect(proposal_ids_host[column * 16 + candidate] == selected_host[column],
                       "DFlash2 TP2 proposal lost its selected candidate id");
            }
        }
        expect(selected_count == 1, "DFlash2 TP2 proposal did not select one candidate");
    }
    std::array<ProposalRank, 2> full_proposals;
    std::array<ninfer::WorkspaceArena*, 2> full_topk;
    const auto full_capacity = tp2_dflash_proposal_capacity(
        loaded.view(0).output_head.qtype, loaded.view(0).output_head.n,
        draft_width, batch);
    const auto planned_round_bytes = DFlashExtension::proposal_workspace_capacity(
        draft_width, batch, 400000, ninfer::ProposalHead::Full,
        WeightsProfile::Qwen38Gguf);
    const auto full_draft_capacity = tp2_dflash_draft_workspace_capacity(
        draft_width, batch, {0, 400000});
    for (int rank = 0; rank < 2; ++rank) {
        expect(planned_round_bytes >=
                   std::max(full_draft_capacity.roots, full_capacity.roots[rank]) +
                       std::max(full_draft_capacity.ops, full_capacity.topk),
               "DFlash2 Program proposal workspace under-planned at 400k context");
    }
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        auto& buffers = full_proposals[rank];
        buffers.topk_backing = ninfer::DeviceBuffer(full_capacity.topk);
        buffers.topk = std::make_unique<ninfer::WorkspaceArena>(
            ninfer::DeviceSpan{buffers.topk_backing.p, buffers.topk_backing.bytes});
        full_topk[rank] = buffers.topk.get();
        proposal_heads[rank] = loaded.view(rank).output_head;
    }
    tp2_dflash_propose(draft_hidden, companion_weights, proposal_heads, nullptr,
                       anchors, base_positions,
                       static_cast<const ninfer::ops::SamplingConfig*>(sampling_backing.p),
                       selected_ranks, proposal_id_ranks, proposal_q_ranks, proposal_roots,
                       full_topk, execution, events);
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    execution.dev[0]->synchronize();
    copy_output(selected_ranks[0], selected_host.data(), sizeof(selected_host));
    copy_output(proposal_id_ranks[0], proposal_ids_host.data(), sizeof(proposal_ids_host));
    copy_output(proposal_q_ranks[0], proposal_q_host.data(), sizeof(proposal_q_host));
    for (int column = 0; column < (draft_width - 1) * batch; ++column) {
        expect(selected_host[column] >= 0 && selected_host[column] < 248077,
               "DFlash2 full-head TP2 proposal returned an invalid token id");
        const float sum = std::accumulate(proposal_q_host.begin() + column * 16,
                                          proposal_q_host.begin() + (column + 1) * 16, 0.0F);
        expect(sum == 1.0F, "DFlash2 full-head TP2 proposal did not select one candidate");
        for (int candidate = 0; candidate < 16; ++candidate) {
            if (proposal_q_host[column * 16 + candidate] == 1.0F) {
                expect(proposal_ids_host[column * 16 + candidate] == selected_host[column],
                       "DFlash2 full-head TP2 proposal lost its selected candidate id");
            }
        }
    }
    for (int row = 0; row < batch; ++row) {
        sampling[row].temperature = 5.0F;
        sampling[row].seed = static_cast<unsigned long long>(row + 1);
    }
    sampling_backing.copy_from_host(sampling.data(), sizeof(sampling));
    tp2_dflash_propose(draft_hidden, companion_weights, proposal_heads, nullptr,
                       anchors, base_positions,
                       static_cast<const ninfer::ops::SamplingConfig*>(sampling_backing.p),
                       selected_ranks, proposal_id_ranks, proposal_q_ranks, proposal_roots,
                       full_topk, execution, events);
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    execution.dev[0]->synchronize();
    copy_output(selected_ranks[0], selected_host.data(), sizeof(selected_host));
    copy_output(proposal_id_ranks[0], proposal_ids_host.data(), sizeof(proposal_ids_host));
    copy_output(proposal_q_ranks[0], proposal_q_host.data(), sizeof(proposal_q_host));
    for (int column = 0; column < (draft_width - 1) * batch; ++column) {
        expect(selected_host[column] >= 0 && selected_host[column] < 248077,
               "DFlash2 stochastic TP2 proposal returned an invalid token id");
        const auto begin = proposal_q_host.begin() + column * 16;
        const float sum = std::accumulate(begin, begin + 16, 0.0F);
        expect(std::abs(sum - 1.0F) < 1.0e-5F,
               "DFlash2 stochastic TP2 proposal distribution is not normalized");
        expect(std::all_of(begin, begin + 16, [](float probability) {
                   return probability >= 0.0F && probability <= 1.0F;
               }), "DFlash2 stochastic TP2 proposal has an invalid probability");
        bool selected_supported = false;
        for (int candidate = 0; candidate < 16; ++candidate) {
            if (proposal_ids_host[column * 16 + candidate] == selected_host[column] &&
                proposal_q_host[column * 16 + candidate] > 0.0F) {
                selected_supported = true;
            }
        }
        expect(selected_supported, "DFlash2 stochastic TP2 proposal selected outside its support");
        for (int candidate = 0; candidate < 16; ++candidate) {
            for (int other = candidate + 1; other < 16; ++other) {
                expect(proposal_ids_host[column * 16 + candidate] !=
                           proposal_ids_host[column * 16 + other],
                       "DFlash2 TP2 proposal contains duplicate candidate ids");
            }
        }
    }
    execution.dev[1]->synchronize();
    ninfer::DecodeGraphPeerBridge bridge(execution.dev[0]->device,
                                         execution.dev[1]->device);
    ninfer::DecodeGraphDefinition graph;
    graph.capture(execution.dev[0]->stream, [&] {
        tp2_dflash_propose(draft_hidden, companion_weights, proposal_heads, nullptr,
                           anchors, base_positions,
                           static_cast<const ninfer::ops::SamplingConfig*>(sampling_backing.p),
                           selected_ranks, proposal_id_ranks, proposal_q_ranks, proposal_roots,
                           full_topk, execution, events);
    }, {&bridge, execution.dev[1]->stream});
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(graph);
    bridge.gate_launch(execution.dev[1]->stream, execution.dev[0]->stream);
    executable.launch(execution.dev[0]->stream);
    execution.dev[0]->synchronize();
    std::array<std::int32_t, (draft_width - 1) * batch> first_replay{};
    std::array<std::int32_t, 16 * (draft_width - 1) * batch> first_ids{};
    std::array<float, 16 * (draft_width - 1) * batch> first_q{};
    copy_output(selected_ranks[0], first_replay.data(), sizeof(first_replay));
    copy_output(proposal_id_ranks[0], first_ids.data(), sizeof(first_ids));
    copy_output(proposal_q_ranks[0], first_q.data(), sizeof(first_q));
    bridge.gate_launch(execution.dev[1]->stream, execution.dev[0]->stream);
    executable.launch(execution.dev[0]->stream);
    execution.dev[0]->synchronize();
    copy_output(selected_ranks[0], selected_host.data(), sizeof(selected_host));
    copy_output(proposal_id_ranks[0], proposal_ids_host.data(), sizeof(proposal_ids_host));
    copy_output(proposal_q_ranks[0], proposal_q_host.data(), sizeof(proposal_q_host));
    execution.dev[1]->synchronize();
    CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
    copy_output(selected_ranks[1], peer_selected_host.data(), sizeof(peer_selected_host));
    copy_output(proposal_id_ranks[1], peer_ids_host.data(), sizeof(peer_ids_host));
    copy_output(proposal_q_ranks[1], peer_q_host.data(), sizeof(peer_q_host));
    expect(selected_host == peer_selected_host && proposal_ids_host == peer_ids_host &&
               proposal_q_host == peer_q_host,
           "DFlash2 TP2 Graph replay did not publish matching verifier inputs");
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    expect(selected_host == first_replay && proposal_ids_host == first_ids &&
               proposal_q_host == first_q,
           "DFlash2 GGUF full-head TP2 Graph replay changed stochastic proposals");
    tp2_dflash_run_proposal(
        state, draft_models, companion_weights, draft_anchor_views, draft_frontier_views,
        draft_valid, draft_lanes, draft_frontier_views[0],
        static_cast<const ninfer::ops::SamplingConfig*>(sampling_backing.p), proposal_heads,
        nullptr, batch, {1, 1}, draft_roots, draft_ops, proposal_roots, full_topk,
        execution, events);
    execution.dev[0]->synchronize();
    execution.dev[1]->synchronize();
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    copy_output(selected_ranks[0], selected_host.data(), sizeof(selected_host));
    copy_output(proposal_id_ranks[0], proposal_ids_host.data(), sizeof(proposal_ids_host));
    copy_output(proposal_q_ranks[0], proposal_q_host.data(), sizeof(proposal_q_host));
    CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
    copy_output(selected_ranks[1], peer_selected_host.data(), sizeof(peer_selected_host));
    copy_output(proposal_id_ranks[1], peer_ids_host.data(), sizeof(peer_ids_host));
    copy_output(proposal_q_ranks[1], peer_q_host.data(), sizeof(peer_q_host));
    expect(selected_host == peer_selected_host && proposal_ids_host == peer_ids_host &&
               proposal_q_host == peer_q_host,
           "DFlash2 complete TP2 proposal round diverged across ranks");
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    for (int column = 0; column < (draft_width - 1) * batch; ++column) {
        expect(selected_host[column] >= 0 && selected_host[column] < 248077,
               "DFlash2 complete TP2 proposal round selected an invalid token");
        const auto begin = proposal_q_host.begin() + column * 16;
        expect(std::abs(std::accumulate(begin, begin + 16, 0.0F) - 1.0F) < 1.0e-5F,
               "DFlash2 complete TP2 proposal round produced invalid probabilities");
    }
    ninfer::DecodeGraphPeerBridge round_bridge(execution.dev[0]->device,
                                               execution.dev[1]->device);
    ninfer::DecodeGraphDefinition round_graph;
    round_graph.capture(execution.dev[0]->stream, [&] {
        tp2_dflash_run_proposal(
            state, draft_models, companion_weights, draft_anchor_views, draft_frontier_views,
            draft_valid, draft_lanes, draft_frontier_views[0],
            static_cast<const ninfer::ops::SamplingConfig*>(sampling_backing.p), proposal_heads,
            nullptr, batch, {1, 1}, draft_roots, draft_ops, proposal_roots, full_topk,
            execution, events);
    }, {&round_bridge, execution.dev[1]->stream});
    ninfer::DecodeGraphExecutable round_executable;
    round_executable.instantiate(round_graph);
    round_bridge.gate_launch(execution.dev[1]->stream, execution.dev[0]->stream);
    round_executable.launch(execution.dev[0]->stream);
    execution.dev[0]->synchronize();
    execution.dev[1]->synchronize();
    copy_output(selected_ranks[0], first_replay.data(), sizeof(first_replay));
    copy_output(proposal_id_ranks[0], first_ids.data(), sizeof(first_ids));
    copy_output(proposal_q_ranks[0], first_q.data(), sizeof(first_q));
    round_bridge.gate_launch(execution.dev[1]->stream, execution.dev[0]->stream);
    round_executable.launch(execution.dev[0]->stream);
    execution.dev[0]->synchronize();
    execution.dev[1]->synchronize();
    copy_output(selected_ranks[0], selected_host.data(), sizeof(selected_host));
    copy_output(proposal_id_ranks[0], proposal_ids_host.data(), sizeof(proposal_ids_host));
    copy_output(proposal_q_ranks[0], proposal_q_host.data(), sizeof(proposal_q_host));
    CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
    copy_output(selected_ranks[1], peer_selected_host.data(), sizeof(peer_selected_host));
    copy_output(proposal_id_ranks[1], peer_ids_host.data(), sizeof(peer_ids_host));
    copy_output(proposal_q_ranks[1], peer_q_host.data(), sizeof(peer_q_host));
    expect(selected_host == first_replay && proposal_ids_host == first_ids &&
               proposal_q_host == first_q && selected_host == peer_selected_host &&
               proposal_ids_host == peer_ids_host && proposal_q_host == peer_q_host,
           "DFlash2 full TP2 proposal Graph replay changed outputs");
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    ninfer::LayoutBuilder verify_builder;
    auto verify_layout = ninfer::targets::qwen3_6::begin_round_state_layout(
        verify_builder,
        {.hidden = 5120, .output_rows = 248320, .batch_capacity = batch,
         .draft_window = draft_width - 1, .enable_dflash2 = true});
    ninfer::targets::qwen3_6::complete_round_state_layout(verify_builder, verify_layout);
    const auto verify_bytes = verify_builder.finish(256);
    std::array<ninfer::DeviceBuffer, 2> verify_backing;
    std::array<std::unique_ptr<ninfer::targets::qwen3_6::RoundState>, 2> verify_states;
    std::array<ninfer::targets::qwen3_6::DFlash2DecodeState*, 2> program_frames{};
    const std::array<int, batch> verify_extents{draft_width - 1, 5, 0};
    const std::array<int, batch> rope_deltas{0, 64, -1};
    ninfer::targets::qwen3_6::DFlash2DecodeIngress verify_ingress{};
    for (int row = 0; row < batch; ++row) {
        verify_ingress.anchors[row] = draft_anchors[row];
        verify_ingress.execution_frontiers[row] = draft_frontiers[row];
        verify_ingress.proposal_extents[row] = verify_extents[row];
        verify_ingress.target_valid_columns[row] = verify_extents[row] + 1;
        verify_ingress.lanes[row] = draft_slots[row];
        for (int column = 0; column < draft_width; ++column) {
            verify_ingress.target_rope_positions[row * draft_width + column] =
                draft_frontiers[row] + std::min(column, verify_extents[row]) + rope_deltas[row];
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        verify_backing[rank] = ninfer::DeviceBuffer(verify_bytes);
        verify_states[rank] = std::make_unique<ninfer::targets::qwen3_6::RoundState>(
            ninfer::DeviceSpan{verify_backing[rank].p, verify_backing[rank].bytes},
            verify_layout);
        auto& frame = *verify_states[rank]->dflash2_decode;
        program_frames[rank] = &frame;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &verify_ingress,
                                   sizeof(verify_ingress), cudaMemcpyHostToDevice,
                                   execution.dev[rank]->stream));
        auto ids = frame.verify_ids.view({draft_width, batch});
        auto positions = frame.target_positions.view({draft_width, batch});
        ninfer::ops::speculative_prepare_verify_inputs(
            frame.anchors, state[rank]->draft_tokens.view({draft_width - 1, batch}),
            frame.execution_frontiers, frame.proposal_extents, ids, positions,
            execution.dev[rank]->stream);
        execution.dev[rank]->synchronize();
        std::array<int, draft_width * batch> verify_ids_host{}, verify_positions_host{};
        CUDA_CHECK(cudaMemcpy(verify_ids_host.data(), ids.data, sizeof(verify_ids_host),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(verify_positions_host.data(), positions.data,
                              sizeof(verify_positions_host), cudaMemcpyDeviceToHost));
        std::array<int, draft_width * batch> rope_positions_host{};
        CUDA_CHECK(cudaMemcpy(rope_positions_host.data(), frame.target_rope_positions.data,
                              sizeof(rope_positions_host), cudaMemcpyDeviceToHost));
        for (int row = 0; row < batch; ++row) {
            for (int column = 0; column < draft_width; ++column) {
                const int expected_id = column > 0 && column <= verify_extents[row]
                    ? selected_host[row * (draft_width - 1) + column - 1]
                    : draft_anchors[row];
                const int expected_position =
                    draft_frontiers[row] + std::min(column, verify_extents[row]);
                expect(verify_ids_host[row * draft_width + column] == expected_id &&
                           verify_positions_host[row * draft_width + column] == expected_position &&
                           rope_positions_host[row * draft_width + column] ==
                               expected_position + rope_deltas[row],
                       "DFlash2 TP2 verifier ids or positions lost proposal alignment");
            }
        }
    }
    auto catchup_ingress = verify_ingress;
    const std::array<int, batch> catchup_start{99, 100, 101};
    const std::array<int, batch> catchup_end{101, 100, 102};
    for (int row = 0; row < batch; ++row) {
        catchup_ingress.context_frontiers[row] = catchup_start[row];
        catchup_ingress.execution_frontiers[row] = catchup_end[row];
    }
    std::array<std::array<std::array<std::uint16_t, 128>, batch>, 2> old_context{};
    const auto context_key_at = [&](int rank, int lane, int position,
                                    std::array<std::uint16_t, 128>& values) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        const auto layer = state[rank]->cache.layer_view(4);
        const std::size_t offset =
            (static_cast<std::size_t>(lane) * 4 * 2048 + position) * 128 *
            sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemcpy(values.data(),
                              static_cast<const std::byte*>(layer.k.data) + offset,
                              sizeof(values), cudaMemcpyDeviceToHost));
    };
    for (int rank = 0; rank < 2; ++rank) {
        for (int lane = 0; lane < batch; ++lane) {
            context_key_at(rank, lane, catchup_start[lane], old_context[rank][lane]);
        }
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        CUDA_CHECK(cudaMemcpyAsync(program_frames[rank]->ingress.data, &catchup_ingress,
                                   sizeof(catchup_ingress), cudaMemcpyHostToDevice,
                                   execution.dev[rank]->stream));
    }
    DFlashExtension::catch_up(state, draft_models, program_frames, execution, events, batch);
    for (int rank = 0; rank < 2; ++rank) {
        execution.dev[rank]->synchronize();
        std::array<int, batch> materialized_counts{};
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        CUDA_CHECK(cudaMemcpy(materialized_counts.data(), state[rank]->pending_counts.data,
                              sizeof(materialized_counts), cudaMemcpyDeviceToHost));
        expect(materialized_counts == committed_counts,
               "DFlash2 TP2 context catch-up used the wrong pending commit counts");
        for (int lane = 0; lane < batch; ++lane) {
            std::array<std::uint16_t, 128> updated{};
            context_key_at(rank, lane, catchup_start[lane], updated);
            expect((updated != old_context[rank][lane]) == (committed_counts[lane] != 0),
                   "DFlash2 TP2 context catch-up missed a committed slot or wrote zero prefix");
        }
        CUDA_CHECK(cudaMemcpyAsync(program_frames[rank]->ingress.data, &verify_ingress,
                                   sizeof(verify_ingress), cudaMemcpyHostToDevice,
                                   execution.dev[rank]->stream));
    }
    execution.dev[0]->synchronize();
    execution.dev[1]->synchronize();
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    std::array<ninfer::DeviceBuffer, 2> program_workspace_backing;
    std::array<std::optional<ninfer::WorkspaceArena>, 2> program_workspace;
    std::array<ninfer::WorkspaceArena*, 2> program_work_views{};
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        program_workspace_backing[rank] = ninfer::DeviceBuffer(planned_round_bytes);
        program_workspace[rank].emplace(ninfer::DeviceSpan{
            program_workspace_backing[rank].p, program_workspace_backing[rank].bytes});
        program_work_views[rank] = &*program_workspace[rank];
    }
    DFlashExtension::run_proposal(state, draft_models, program_frames, program_work_views,
                                  batch, ninfer::ProposalHead::Full, {1, 1}, execution, events);
    execution.dev[0]->synchronize();
    execution.dev[1]->synchronize();
    for (int rank = 0; rank < 2; ++rank) {
        expect(program_work_views[rank]->used() == 0 &&
                   program_work_views[rank]->peak_used() <= planned_round_bytes,
               "DFlash2 Program proposal exceeded its planned workspace");
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        std::array<int, (draft_width - 1) * batch> generated{};
        copy_output(state[rank]->draft_tokens, generated.data(), sizeof(generated));
        for (int row = 0; row < batch; ++row) {
            for (int column = 0; column < draft_width - 1; ++column) {
                expect(generated[row * (draft_width - 1) + column] >= 0 &&
                           generated[row * (draft_width - 1) + column] < 248077,
                       "DFlash2 Program proposal selected an invalid token");
            }
        }
    }
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    for (int rank = 0; rank < 2; ++rank) {
        expect(draft_roots[rank]->used() == 0 &&
                   draft_roots[rank]->peak_used() <= draft_capacity.roots &&
                   draft_ops[rank]->used() == 0 &&
                   draft_ops[rank]->peak_used() <= draft_capacity.ops,
               "DFlash2 TP2 draft exceeded its planned workspace");
    }
    std::cout << "DFlash2 TP2 five-layer draft backbone exercised with real weights\n";
    std::cout << "DFlash2 anchor/mask embedding through TP2 proposal exercised\n";
    std::cout << "DFlash2 target verification inputs aligned with two-rank proposals\n";
    std::cout << "DFlash2 pending context catch-up exercised across three lanes\n";
    std::cout << "DFlash2 Program proposal workspace and TP2 model binding exercised\n";
    std::cout << "DFlash2 TP2 full/optimized proposals and GGUF Graph replay exercised\n";
    std::cout << "DFlash2 TP2 feature projection and five context KV layers exercised\n";
    std::cout << "DFlash2 TP2 committed pending prefixes and Graph replay exercised\n";
    std::cout << "DFlash2 selected TP2 runtime views verified without MTP residency\n";
    return 0;
}
