// TP2 sharded materialization: real artifact bytes landing on two device arenas.
//
// Two independent halves:
//
//   1. tp1 BYTE IDENTITY. The single-device materialized layout must be exactly what it was before
//      sharding existed. The golden numbers below were captured from a run of the pre-change build
//      against the same artifact and feature set; the checksum
//      folds every placement's (object index, arena offset, bytes, alignment) in bind order, so a
//      single moved tensor changes it.
//
//   2. tp2 SHARDED LOAD. Binds the same artifact for two devices, materializes it onto both, and
//      audits the result: a per-family byte table, per-device arena totals against the per-device
//      weight budget, replicated objects present and identical on both devices, and sharded
//      objects whose device bytes equal an independently recomputed slice of the parent payload.
//
// Skips (77) when the artifact or a second CUDA device is unavailable.

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "targets/qwen3_6_27b/impl/config.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using ninfer::artifact::MaterializationPlan;
using ninfer::artifact::ShardAxis;
using namespace ninfer::targets::qwen3_6_27b::detail;
using ninfer::targets::qwen3_6_27b::Package;

int g_failures = 0;

void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    ++g_failures;
}

void expect(bool condition, const std::string& message) {
    if (!condition) { fail(message); }
}

void expect_equal(std::uint64_t actual, std::uint64_t expected, const std::string& label) {
    if (actual != expected) {
        fail(label + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
    }
}

// --- tp1 baseline, captured from the pre-change build ------------------------------------------
// Artifact: qwen3_8_27b_nvfp4.ninfer (identity qwen3.8-27b/nvfp4, WeightsProfile::Qwen38Nvfp4).
// Features: vision + MTP + optimized proposal, i.e. every optional object resident.
constexpr std::size_t kTp1ObjectCount        = 1124;
constexpr std::size_t kTp1DeviceObjects      = 1006;
constexpr std::size_t kTp1HostObjects        = 6;
constexpr std::uint64_t kTp1ArenaBytes       = 21'479'648'768ULL;
constexpr std::uint64_t kTp1PlacementBytes   = 21'479'606'624ULL;
constexpr std::uint64_t kTp1PlacementChecksum = 11'222'044'810'855'605'826ULL;

// The original design budget: "Weights (split) 9.49 GiB" per GPU for an 18.98 GiB no-vision
// artifact. Both halves of that are wrong against the real artifact -- it measures 18.99 GiB, and
// token_embedding is replicated rather than halved -- so this figure is reported as a delta below,
// never asserted on. Measured per-device weight residency is 10.08 GiB.
constexpr double kDesignBudgetPerDeviceWeightGiB = 9.49;

std::uint64_t placement_checksum(const MaterializationPlan& plan) {
    std::uint64_t checksum = 1469598103934665603ULL;
    const auto mix         = [&](std::uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            checksum ^= (value >> (8 * i)) & 0xFFULL;
            checksum *= 1099511628211ULL;
        }
    };
    for (const auto& placement : plan.device_objects) {
        mix(placement.object.index);
        mix(placement.offset);
        mix(placement.bytes);
        mix(placement.alignment);
    }
    return checksum;
}

std::filesystem::path artifact_path() {
    if (const char* value = std::getenv("NINFER_QWEN3_8_27B_NVFP4_WEIGHTS");
        value != nullptr && *value != '\0') {
        return value;
    }
    return "/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer";
}

ninfer::targets::qwen3_6::StartupFeatures features(bool vision) {
    return {
        .vision        = vision,
        .speculative   = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

double gib(std::uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

// Strips the per-layer prefix so every layer's copy of an object folds into one audit row, and the
// text and MTP copies of the same family land on the same row (they share a shard rule).
// "text/layers/17/mlp/gate_up" and "mtp/layer/mlp/gate_up" both become "mlp/gate_up";
// "text/final_norm" and "mtp/input_projection" keep their full names.
std::string family_of(std::string_view name) {
    constexpr std::string_view kTextLayers = "text/layers/";
    if (name.starts_with(kTextLayers)) {
        const std::size_t after = name.find('/', kTextLayers.size());
        if (after != std::string_view::npos) { return std::string(name.substr(after + 1)); }
    }
    constexpr std::string_view kMtpLayer = "mtp/layer/";
    if (name.starts_with(kMtpLayer)) { return std::string(name.substr(kMtpLayer.size())); }
    return std::string(name);
}

struct FamilyRow {
    std::size_t objects = 0;
    std::uint64_t artifact_bytes = 0;
    std::array<std::uint64_t, 2> device_bytes{};
    std::array<std::uint64_t, 2> expected_bytes{};
    ShardAxis axis = ShardAxis::Replicated;
};

// The size this device's shard of `tensor` must have, computed straight from the shard map and the
// slice geometry -- i.e. without consulting anything the Binder recorded.
std::uint64_t expected_shard_bytes(const ninfer::artifact::TensorDescriptor& tensor,
                                   const ShardMapping& mapping, int device) {
    if (mapping.axis == ShardAxis::Replicated) { return tensor.bytes; }
    std::vector<ninfer::artifact::SliceRange> ranges;
    for (const Shard& shard : mapping.shards) {
        if (shard.device == device) { ranges.push_back({shard.row_begin, shard.row_count}); }
    }
    if (ranges.empty()) { return 0; }
    return mapping.axis == ShardAxis::Rows
               ? ninfer::artifact::tensor_row_slice(tensor.layout, tensor.format, tensor.shape,
                                                    ranges)
                     .encoded_bytes
               : ninfer::artifact::tensor_column_slice(tensor.layout, tensor.format, tensor.shape,
                                                       ranges)
                     .encoded_bytes;
}

const char* axis_name(ShardAxis axis) {
    switch (axis) {
    case ShardAxis::Replicated:
        return "replicated";
    case ShardAxis::Rows:
        return "rows";
    case ShardAxis::Columns:
        return "columns";
    }
    return "?";
}

// Reads one object's whole device-side shard back to the host.
std::vector<std::byte> read_back(const ninfer::artifact::MaterializedArtifact& materialized,
                                 ninfer::artifact::ObjectHandle handle, int device,
                                 std::uint64_t bytes) {
    std::vector<std::byte> out(static_cast<std::size_t>(bytes));
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemcpy(out.data(), materialized.device_data(handle, device), out.size(),
                          cudaMemcpyDeviceToHost));
    return out;
}

int verify_tp1(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan =
        bind_artifact(binder, Package::resolve_weights(reader.identity()), features(true));
    const MaterializationPlan& materialization = plan.materialization;

    expect_equal(materialization.object_count, kTp1ObjectCount, "tp1 object count");
    expect_equal(materialization.device_objects.size(), kTp1DeviceObjects, "tp1 device objects");
    expect_equal(materialization.host_objects.size(), kTp1HostObjects, "tp1 host objects");
    expect_equal(materialization.device_count, 1, "tp1 device count");
    expect_equal(materialization.device_capacity_bytes[0], kTp1ArenaBytes, "tp1 arena bytes");
    expect_equal(materialization.device_capacity_bytes[1], 0, "tp1 second arena stays unused");
    expect_equal(placement_checksum(materialization), kTp1PlacementChecksum,
                 "tp1 placement checksum (object index, offset, bytes, alignment in bind order)");
    std::uint64_t total = 0;
    for (const auto& placement : materialization.device_objects) {
        expect(placement.device == 0, "tp1 placement targets a device other than 0");
        expect(placement.copies.empty(), "tp1 placement is not a whole-payload copy");
        total += placement.bytes;
    }
    expect_equal(total, kTp1PlacementBytes, "tp1 planned placement bytes");

    ninfer::DeviceContext device(0);
    auto materialized = ninfer::artifact::materialize(reader, materialization, device);
    const auto& stats = materialized.stats();
    expect_equal(stats.h2d_bytes, kTp1PlacementBytes, "tp1 host_to_device_bytes");
    expect_equal(stats.per_device_h2d_bytes[0], kTp1PlacementBytes, "tp1 device-0 h2d bytes");
    expect_equal(stats.device_capacity_bytes, kTp1ArenaBytes, "tp1 stats arena bytes");
    expect_equal(stats.tensor_count, kTp1DeviceObjects, "tp1 tensor count");
    expect_equal(stats.resource_count, kTp1HostObjects, "tp1 resource count");
    expect_equal(materialized.device_arena().capacity(), kTp1ArenaBytes, "tp1 arena capacity");
    expect_equal(materialized.device_arena().used(), kTp1ArenaBytes, "tp1 arena used");
    std::cout << "tp1: arena " << kTp1ArenaBytes << " B (" << std::fixed << std::setprecision(3)
              << gib(kTp1ArenaBytes) << " GiB), h2d " << stats.h2d_bytes << " B, upload "
              << stats.upload_seconds << " s -- identical to the pre-change baseline\n";
    return 0;
}

int verify_tp2(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader, 2);
    const ArtifactLoadPlan plan =
        bind_artifact(binder, Package::resolve_weights(reader.identity()), features(false), 2);
    const MaterializationPlan& materialization = plan.materialization;
    const TextConfig config{};

    expect_equal(materialization.device_count, 2, "tp2 device count");

    // Vision has no shard map (the tower is out of scope for TP2), so binding
    // it for two devices must be refused rather than silently replicating the backbone.
    {
        ninfer::artifact::Binder vision_binder(reader, 2);
        bool rejected = false;
        try {
            (void)bind_artifact(vision_binder, Package::resolve_weights(reader.identity()),
                                features(true), 2);
        } catch (const std::exception&) { rejected = true; }
        expect(rejected, "tp2 + vision was not rejected at bind time");
    }

    // --- per-family audit table ---
    std::map<std::string, FamilyRow> families;
    std::uint64_t replicated_artifact_bytes = 0;
    std::uint64_t sharded_artifact_bytes    = 0;
    for (const auto& placement : materialization.device_objects) {
        const auto& object = reader.objects()[placement.object.index];
        const std::string name(ninfer::artifact::object_name(object));
        const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
        if (tensor == nullptr) {
            fail(name + " was planned as a device tensor but is a resource");
            continue;
        }
        const ShardMapping mapping = shard_mapping_for(name, 2, config, Package::resolve_weights(reader.identity()));
        FamilyRow& row             = families[family_of(name)];
        const auto slot            = static_cast<std::size_t>(placement.device);
        row.device_bytes[slot] += placement.bytes;
        // Every object, not just the spot-checked ones: the bytes the Binder reserved must equal
        // the slice geometry's own answer. This is what makes a small family (gdn/a_log,
        // dt_bias, draft_head_token_ids) impossible to mis-shard unnoticed -- the +-2% grand
        // total would never see them.
        row.expected_bytes[slot] += expected_shard_bytes(*tensor, mapping, placement.device);
        if (placement.device == 0) {
            ++row.objects;
            row.artifact_bytes += ninfer::artifact::object_bytes(object);
            row.axis = mapping.axis;
            if (row.axis == ShardAxis::Replicated) {
                replicated_artifact_bytes += ninfer::artifact::object_bytes(object);
            } else {
                sharded_artifact_bytes += ninfer::artifact::object_bytes(object);
            }
        }
    }

    std::cout << "\n  tp2 per-device weight bytes by object family (NVFP4 qwen3.8-27B, no vision)\n";
    std::cout << "  " << std::left << std::setw(30) << "family" << std::right << std::setw(6)
              << "objs" << std::setw(12) << "axis" << std::setw(16) << "artifact MiB"
              << std::setw(14) << "dev0 MiB" << std::setw(14) << "dev1 MiB" << '\n';
    std::cout << "  " << std::string(92, '-') << '\n';
    const auto mib = [](std::uint64_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    };
    std::array<std::uint64_t, 2> device_totals{};
    std::uint64_t artifact_total = 0;
    for (const auto& [family, row] : families) {
        std::cout << "  " << std::left << std::setw(30) << family << std::right << std::setw(6)
                  << row.objects << std::setw(12) << axis_name(row.axis) << std::setw(16)
                  << std::fixed << std::setprecision(2) << mib(row.artifact_bytes) << std::setw(14)
                  << mib(row.device_bytes[0]) << std::setw(14) << mib(row.device_bytes[1]) << '\n';
        device_totals[0] += row.device_bytes[0];
        device_totals[1] += row.device_bytes[1];
        artifact_total += row.artifact_bytes;
        for (int device = 0; device < 2; ++device) {
            const auto slot = static_cast<std::size_t>(device);
            expect_equal(row.device_bytes[slot], row.expected_bytes[slot],
                         "family " + family + " device " + std::to_string(device) +
                             " bytes vs the slice geometry");
        }
        if (row.axis == ShardAxis::Replicated) {
            expect(row.device_bytes[0] == row.artifact_bytes &&
                       row.device_bytes[1] == row.artifact_bytes,
                   "replicated family " + family + " is not whole on both devices");
        } else {
            // A sharded family must be split, not replicated and not dropped: the halves match
            // each other, and together they cover the artifact bytes plus only the per-shard
            // plane realignment and the deliberately replicated pieces (NVFP4's 4-byte matrix
            // divisor, FP8's per-row scale plane under a column split).
            expect(row.device_bytes[0] == row.device_bytes[1],
                   "sharded family " + family + " is not split evenly between devices");
            const std::uint64_t sum = row.device_bytes[0] + row.device_bytes[1];
            expect(sum >= row.artifact_bytes,
                   "sharded family " + family + " loses bytes: " + std::to_string(sum) + " < " +
                       std::to_string(row.artifact_bytes));
            const std::uint64_t allowance =
                row.artifact_bytes / 100 + 512 * static_cast<std::uint64_t>(row.objects);
            expect(sum <= row.artifact_bytes + allowance,
                   "sharded family " + family + " looks replicated rather than split: " +
                       std::to_string(sum) + " vs artifact " + std::to_string(row.artifact_bytes));
        }
    }
    std::cout << "  " << std::string(92, '-') << '\n';
    std::cout << "  " << std::left << std::setw(30) << "TOTAL" << std::right << std::setw(6)
              << materialization.device_objects.size() / 2 << std::setw(12) << "" << std::setw(16)
              << mib(artifact_total) << std::setw(14) << mib(device_totals[0]) << std::setw(14)
              << mib(device_totals[1]) << '\n';

    const std::uint64_t expected_per_device =
        replicated_artifact_bytes + sharded_artifact_bytes / 2;
    std::cout << "\n  artifact (no vision)      " << std::setprecision(3) << gib(artifact_total)
              << " GiB   [replicated " << gib(replicated_artifact_bytes) << " + sharded "
              << gib(sharded_artifact_bytes) << "]\n"
              << "  per-device arena          dev0 "
              << gib(materialization.device_capacity_bytes[0]) << " GiB, dev1 "
              << gib(materialization.device_capacity_bytes[1]) << " GiB\n"
              << "  replicated + sharded/2    " << gib(expected_per_device) << " GiB\n"
              << "  original design budget    " << kDesignBudgetPerDeviceWeightGiB << " GiB\n";
    // The design budget's per-GPU 9.49 GiB is 18.98/2, i.e. it halves *everything*, while the
    // same design's shard table replicates token_embedding ("1.27 GiB each") -- the two lines
    // contradict each other. The measured figure is the one that has to hold, so report the delta
    // rather than asserting on a number that was never self-consistent.
    std::cout << "  DESIGN-BUDGET DELTA       +"
              << (gib(materialization.device_capacity_bytes[0]) -
                  kDesignBudgetPerDeviceWeightGiB)
              << " GiB per device ("
              << ((gib(materialization.device_capacity_bytes[0]) /
                       kDesignBudgetPerDeviceWeightGiB - 1.0) * 100.0)
              << "%) -- replicated token_embedding is not halved, and the artifact measures "
              << gib(artifact_total) << " GiB rather than the assumed 18.98 GiB\n";

    // The arena carries 256-byte inter-object padding on top of the placement bytes, and each
    // shard realigns its own planes, so match the derived expectation rather than the raw sum.
    for (int device = 0; device < 2; ++device) {
        const double actual = gib(materialization.device_capacity_bytes[device]);
        const double drift  = (actual - gib(expected_per_device)) / gib(expected_per_device);
        expect(std::abs(drift) <= 0.02, "device " + std::to_string(device) +
                                            " arena is more than 2% from replicated + sharded/2 (" +
                                            std::to_string(drift * 100.0) + "%)");
    }
    const double imbalance =
        static_cast<double>(materialization.device_capacity_bytes[0] >
                                    materialization.device_capacity_bytes[1]
                                ? materialization.device_capacity_bytes[0] -
                                      materialization.device_capacity_bytes[1]
                                : materialization.device_capacity_bytes[1] -
                                      materialization.device_capacity_bytes[0]) /
        static_cast<double>(materialization.device_capacity_bytes[0]);
    expect(imbalance <= 0.001, "the two device arenas differ by more than 0.1%");

    // --- real two-device load ---
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cout << "\nSKIP: tp2 materialization needs two CUDA devices\n";
        return 77;
    }
    ninfer::ExecutionContext execution({0, 1});
    auto materialized = ninfer::artifact::materialize(reader, materialization, execution);
    const auto& stats = materialized.stats();
    expect_equal(stats.device_count, 2, "tp2 stats device count");
    expect_equal(stats.per_device_h2d_bytes[0] + stats.per_device_h2d_bytes[1], stats.h2d_bytes,
                 "tp2 per-device h2d bytes sum to the total");
    for (int device = 0; device < 2; ++device) {
        expect_equal(stats.per_device_capacity_bytes[device],
                     materialization.device_capacity_bytes[device],
                     "tp2 device " + std::to_string(device) + " arena capacity");
        expect_equal(materialized.device_arena(device).used(),
                     materialization.device_capacity_bytes[device],
                     "tp2 device " + std::to_string(device) + " arena is fully placed");
    }
    std::cout << "\n  uploaded " << gib(stats.h2d_bytes) << " GiB total (dev0 "
              << gib(stats.per_device_h2d_bytes[0]) << " GiB, dev1 "
              << gib(stats.per_device_h2d_bytes[1]) << " GiB) in " << std::setprecision(1)
              << stats.upload_seconds << " s\n\n";

    // --- content: every device's bytes equal an independently recomputed slice ---
    const auto placements_for = [&](std::string_view name) {
        std::vector<const ninfer::artifact::DeviceMaterialization*> found;
        for (const auto& placement : materialization.device_objects) {
            if (ninfer::artifact::object_name(reader.objects()[placement.object.index]) == name) {
                found.push_back(&placement);
            }
        }
        return found;
    };

    // --- shape-vs-placement validation ---
    //
    // `artifact::materialized_{tensor,weight}` used to wrap `device_data(handle)` in whatever
    // extents the caller named, with NO check against the bytes actually placed. At tp1 that was
    // harmless (a placement always held the whole object); at tp2 a placement holds a SHARD, so a
    // stale whole-model literal -- the `{10240, 4}` `LoadedModelData` used for `gdn/convolution`
    // -- would build a Tensor over twice the bytes that exist and read out of bounds silently.
    // Assert that the correct shard shape is accepted and the stale shape now throws.
    {
        const auto conv = placements_for("text/layers/0/gdn/convolution");
        expect(conv.size() == 2, "gdn/convolution is not placed on both devices");
        if (conv.size() == 2) {
            for (int device = 0; device < 2; ++device) {
                bool accepted = true;
                try {
                    (void)ninfer::artifact::materialized_tensor(
                        materialized, conv[static_cast<std::size_t>(device)]->object,
                        ninfer::artifact::NumericFormat::BF16, {config.convolution_dim / 2, 4},
                        device);
                } catch (const std::exception&) { accepted = false; }
                expect(accepted, "the real conv shard shape was rejected on device " +
                                     std::to_string(device));

                bool rejected = false;
                try {
                    (void)ninfer::artifact::materialized_tensor(
                        materialized, conv[static_cast<std::size_t>(device)]->object,
                        ninfer::artifact::NumericFormat::BF16, {config.convolution_dim, 4}, device);
                } catch (const std::exception&) { rejected = true; }
                expect(rejected, "the stale whole-model conv shape was not rejected on device " +
                                     std::to_string(device));
            }
        }
    }

    const auto verify_object = [&](std::string_view name) {
        const auto found = placements_for(name);
        if (found.size() != 2) {
            fail(std::string(name) + ": expected one placement per device, found " +
                 std::to_string(found.size()));
            return;
        }
        const auto& object = reader.objects()[found[0]->object.index];
        const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
        if (tensor == nullptr) {
            fail(std::string(name) + " is not a tensor");
            return;
        }
        const auto payload         = reader.payload(object);
        const ShardMapping mapping = shard_mapping_for(name, 2, config, Package::resolve_weights(reader.identity()));
        for (int device = 0; device < 2; ++device) {
            const auto* placement = found[device];
            expect(placement->device == device,
                   std::string(name) + ": placements are not ordered by device");
            std::vector<std::byte> expected;
            if (mapping.axis == ShardAxis::Replicated) {
                expected.assign(payload.data.begin(), payload.data.end());
            } else {
                std::vector<ninfer::artifact::SliceRange> ranges;
                for (const Shard& shard : mapping.shards) {
                    if (shard.device == device) {
                        ranges.push_back({shard.row_begin, shard.row_count});
                    }
                }
                const ninfer::artifact::TensorSlice slice =
                    mapping.axis == ShardAxis::Rows
                        ? ninfer::artifact::tensor_row_slice(tensor->layout, tensor->format,
                                                             tensor->shape, ranges)
                        : ninfer::artifact::tensor_column_slice(tensor->layout, tensor->format,
                                                                tensor->shape, ranges);
                expected.assign(static_cast<std::size_t>(slice.encoded_bytes), std::byte{});
                for (const auto& copy : slice.copies) {
                    std::copy_n(payload.data.begin() +
                                    static_cast<std::ptrdiff_t>(copy.source_offset),
                                static_cast<std::size_t>(copy.bytes),
                                expected.begin() + static_cast<std::ptrdiff_t>(copy.dest_offset));
                }
            }
            expect_equal(placement->bytes, expected.size(),
                         std::string(name) + " device " + std::to_string(device) + " shard bytes");
            const auto actual =
                read_back(materialized, placement->object, device, placement->bytes);
            expect(actual == expected, std::string(name) + " device " + std::to_string(device) +
                                           " shard bytes differ from the recomputed slice");
        }
    };

    // One object per (layout, axis) combination this artifact actually uses, plus a replicated one.
    verify_object("text/final_norm");                       // BF16 contiguous, replicated
    verify_object("text/layers/0/gdn/a_log");               // FP32 contiguous rank 1, rows
    verify_object("text/layers/0/mlp/gate_up");             // NVFP4 blockscale, rows
    verify_object("text/layers/0/mlp/down");                // NVFP4 blockscale, columns
    verify_object("text/layers/0/gdn/query_key_value_z");   // FP8 row-scale, rows
    verify_object("text/layers/3/attention/output");        // FP8 row-scale, columns
    verify_object("text/output_head");                      // FP8 row-scale, rows
    verify_object("mtp/layer/mlp/gate_up");                 // W8G32 row-split, rows
    verify_object("mtp/input_projection");                  // W8G32 row-split, columns
    verify_object("text/draft_head");                       // Q4G64 row-split, rows
    verify_object("text/draft_head_token_ids");             // I32 contiguous, REPLICATED (3.9)
    verify_object("text/token_embedding");                  // FP8 row-scale, replicated
    verify_object("text/layers/0/gdn/convolution");         // BF16 contiguous, columns, MULTI-range

    // --- non-circular NVFP4 check: decode individual scale/code words by coordinate ---
    //
    // The verify_object pass above compares whole shards against the same slice API the Binder
    // used, so it proves the plumbing but not the geometry. This closes that loop: it addresses
    // the swizzled scale plane through the formula in storage-layouts.md section 4 (equivalently,
    // tools/artifact/layouts.py swizzle_nvfp4_scales) on BOTH sides and checks that shard word
    // (n, g) is the parent word at the coordinate the shard map says it should be. Nothing here
    // goes through tensor_row_slice / tensor_column_slice.
    {
        const auto scale_offset = [](std::uint64_t k_tiles, std::uint64_t n, std::uint64_t g) {
            return ((n / 128) * k_tiles + g / 4) * 512 + (n % 32) * 16 + ((n % 128) / 32) * 4 +
                   (g % 4);
        };
        // parent_row/parent_group map a shard coordinate back to the parent coordinate the shard
        // map assigns to it; they are written out per object rather than derived from the slice.
        const auto check_nvfp4 = [&](std::string_view name, std::uint64_t shard_rows,
                                     std::uint64_t shard_columns, int device,
                                     const auto& parent_row, const auto& parent_group) {
            const auto found = placements_for(name);
            if (found.size() != 2) {
                fail(std::string(name) + ": missing placements");
                return;
            }
            const auto& object = reader.objects()[found[0]->object.index];
            const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
            if (tensor == nullptr || tensor->format != ninfer::artifact::NumericFormat::NVFP4) {
                fail(std::string(name) + " is not NVFP4");
                return;
            }
            const auto payload = reader.payload(object);
            const auto parent  = ninfer::artifact::block_scale_geometry(
                ninfer::artifact::NumericFormat::NVFP4, tensor->shape);
            const std::array<std::uint64_t, 2> shard_shape = {shard_rows, shard_columns};
            const auto shard                               = ninfer::artifact::block_scale_geometry(
                ninfer::artifact::NumericFormat::NVFP4, shard_shape);
            const auto bytes = read_back(materialized, found[device]->object, device,
                                         found[device]->bytes);
            const std::uint64_t parent_columns = tensor->shape[1];
            std::size_t checked                = 0;
            for (const std::uint64_t n :
                 {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{31}, std::uint64_t{32},
                  std::uint64_t{127}, std::uint64_t{128}, shard_rows / 2, shard_rows - 1}) {
                if (n >= shard_rows) { continue; }
                for (const std::uint64_t g :
                     {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{3}, std::uint64_t{4},
                      shard.groups_per_row / 2, shard.groups_per_row - 1}) {
                    if (g >= shard.groups_per_row) { continue; }
                    const std::byte actual =
                        bytes[static_cast<std::size_t>(shard.scale_plane_offset +
                                                       scale_offset(shard.k_tiles, n, g))];
                    const std::byte wanted = payload.data[static_cast<std::size_t>(
                        parent.scale_plane_offset +
                        scale_offset(parent.k_tiles, parent_row(n), parent_group(g)))];
                    if (actual != wanted) {
                        fail(std::string(name) + " device " + std::to_string(device) +
                             ": scale word (" + std::to_string(n) + "," + std::to_string(g) +
                             ") does not match parent (" + std::to_string(parent_row(n)) + "," +
                             std::to_string(parent_group(g)) + ")");
                        return;
                    }
                    // The code plane is row-major [N, K/2]; group g covers code bytes [8g, 8g+8).
                    for (std::uint64_t byte = 0; byte < 8; ++byte) {
                        const std::uint64_t j       = g * 8 + byte;
                        const std::byte code_actual =
                            bytes[static_cast<std::size_t>(n * (shard_columns / 2) + j)];
                        const std::byte code_wanted = payload.data[static_cast<std::size_t>(
                            parent_row(n) * (parent_columns / 2) + parent_group(g) * 8 + byte)];
                        if (code_actual != code_wanted) {
                            fail(std::string(name) + " device " + std::to_string(device) +
                                 ": code byte (" + std::to_string(n) + "," + std::to_string(j) +
                                 ") does not match parent");
                            return;
                        }
                    }
                    ++checked;
                }
            }
            // The matrix-level FP32 divisor is replicated verbatim into every shard.
            for (std::uint64_t i = 0; i < 4; ++i) {
                if (bytes[static_cast<std::size_t>(shard.weight_divisor_offset + i)] !=
                    payload.data[static_cast<std::size_t>(parent.weight_divisor_offset + i)]) {
                    fail(std::string(name) + " device " + std::to_string(device) +
                         ": weight divisor was not replicated");
                    break;
                }
            }
            expect(checked >= 20, std::string(name) + ": too few NVFP4 coordinates checked");
        };

        // mlp/gate_up, NVFP4 rows: 34816 = Gate(17408) | Up(17408), each halved. Device d owns
        // gate rows [d*8704, (d+1)*8704) then up rows [17408 + d*8704, ...), concatenated.
        for (int device = 0; device < 2; ++device) {
            const std::uint64_t base = static_cast<std::uint64_t>(device) * 8704;
            check_nvfp4(
                "text/layers/0/mlp/gate_up", 17408, 5120, device,
                [base](std::uint64_t n) { return n < 8704 ? base + n : 17408 + base + (n - 8704); },
                [](std::uint64_t g) { return g; });
        }
        // mlp/down, NVFP4 columns: 17408 input columns halved, so device d owns columns
        // [d*8704, (d+1)*8704) of every row -- group g of the shard is parent group g + d*544.
        for (int device = 0; device < 2; ++device) {
            const std::uint64_t group_base = static_cast<std::uint64_t>(device) * (8704 / 16);
            check_nvfp4(
                "text/layers/0/mlp/down", 5120, 8704, device, [](std::uint64_t n) { return n; },
                [group_base](std::uint64_t g) { return group_base + g; });
        }
    }

    // A replicated object must be byte-identical on both devices, not merely present.
    // gdn/norm is the GDN family's genuinely replicated member: its shape is {128}, the
    // per-head-DIMENSION gated_rmsnorm gain, which carries no head axis to split.
    {
        const auto found = placements_for("text/layers/0/gdn/norm");
        if (found.size() == 2) {
            const auto first  = read_back(materialized, found[0]->object, 0, found[0]->bytes);
            const auto second = read_back(materialized, found[1]->object, 1, found[1]->bytes);
            expect(first == second, "replicated gdn/norm differs between devices");
        } else {
            fail("gdn/norm has no placement on both devices");
        }
    }

    // --- non-circular gdn/convolution check: the multi-range column shard, by coordinate ---
    //
    // The only object in the map whose device shard is three disjoint column ranges. verify_object
    // above compares it against the same slice API the Binder used; this closes the loop by
    // addressing BF16 word (tap, channel) directly on both sides and asserting the shard's local
    // channel c is the parent channel the Q|K|V head split says it is. It also asserts the two
    // devices' shards DIFFER, so a silent fall-back to replication cannot pass.
    {
        constexpr std::uint64_t kTaps        = 4;
        constexpr std::uint64_t kKeyDim      = 2048;  // TextConfig::key_dim
        constexpr std::uint64_t kValueDim    = 6144;  // TextConfig::value_dim
        constexpr std::uint64_t kParentChans = 2 * kKeyDim + kValueDim; // 10240
        constexpr std::uint64_t kShardChans  = kParentChans / 2;        // 5120
        const auto found = placements_for("text/layers/0/gdn/convolution");
        if (found.size() != 2) {
            fail("gdn/convolution has no placement on both devices");
        } else {
            const auto& object   = reader.objects()[found[0]->object.index];
            const auto parent    = reader.payload(object).data;
            // Local channel c of device r -> its parent channel.
            const auto parent_channel = [&](std::uint64_t c, std::uint64_t r) {
                if (c < kKeyDim / 2) { return r * (kKeyDim / 2) + c; }
                if (c < kKeyDim) { return kKeyDim + r * (kKeyDim / 2) + (c - kKeyDim / 2); }
                return 2 * kKeyDim + r * (kValueDim / 2) + (c - kKeyDim);
            };
            std::array<std::vector<std::byte>, 2> shard;
            for (int device = 0; device < 2; ++device) {
                shard[static_cast<std::size_t>(device)] =
                    read_back(materialized, found[device]->object, device, found[device]->bytes);
                expect_equal(found[device]->bytes, kTaps * kShardChans * 2,
                             "gdn/convolution device " + std::to_string(device) + " shard bytes");
            }
            expect(shard[0] != shard[1],
                   "gdn/convolution shards are byte-identical, so the channel split is untested");
            std::size_t mismatches = 0;
            for (std::uint64_t r = 0; r < 2 && mismatches == 0; ++r) {
                const auto& bytes = shard[static_cast<std::size_t>(r)];
                for (std::uint64_t tap = 0; tap < kTaps && mismatches == 0; ++tap) {
                    for (std::uint64_t c = 0; c < kShardChans; ++c) {
                        const std::uint64_t dst = (tap * kShardChans + c) * 2;
                        const std::uint64_t src =
                            (tap * kParentChans + parent_channel(c, r)) * 2;
                        if (bytes[dst] != parent[src] || bytes[dst + 1] != parent[src + 1]) {
                            fail("gdn/convolution device " + std::to_string(r) + " tap " +
                                 std::to_string(tap) + " local channel " + std::to_string(c) +
                                 " is not parent channel " +
                                 std::to_string(parent_channel(c, r)));
                            ++mismatches;
                            break;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

} // namespace

int main() {
    const std::filesystem::path path = artifact_path();
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: NVFP4 artifact is required: " << path << '\n';
        return 77;
    }
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
        device_count == 0) {
        std::cerr << "skip: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);
    try {
        if (const int result = verify_tp1(path); result != 0) { return result; }
        const int result = verify_tp2(path);
        if (result == 77 && g_failures == 0) { return 77; }
        if (result != 0 && result != 77) { return result; }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "sharded materialization: all checks passed\n";
    return 0;
}
