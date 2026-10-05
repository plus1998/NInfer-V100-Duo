#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using Json = nlohmann::json;
using ninfer::test::artifact_fixture::write_fixture;
using ninfer::test::artifact_fixture::write_v3_fixture;

Json normative_directory() {
    return {
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
        {"objects", Json::array({
                        {{"name", "resource"},
                         {"kind", "resource"},
                         {"encoding", "raw-bytes-v1"},
                         {"offset", 0},
                         {"bytes", 3}},
                        {{"name", "bf16"},
                         {"kind", "tensor"},
                         {"shape", {2, 3}},
                         {"format", "BF16"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 256},
                         {"bytes", 12}},
                        {{"name", "fp32_scalar"},
                         {"kind", "tensor"},
                         {"shape", Json::array()},
                         {"format", "FP32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 512},
                         {"bytes", 4}},
                        {{"name", "i32"},
                         {"kind", "tensor"},
                         {"shape", {2}},
                         {"format", "I32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 768},
                         {"bytes", 8}},
                        {{"name", "q4"},
                         {"kind", "tensor"},
                         {"shape", {1, 1}},
                         {"format", "Q4G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1024},
                         {"bytes", 260}},
                        {{"name", "q5"},
                         {"kind", "tensor"},
                         {"shape", {2, 130}},
                         {"format", "Q5G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1536},
                         {"bytes", 528}},
                        {{"name", "q6"},
                         {"kind", "tensor"},
                         {"shape", {1, 64}},
                         {"format", "Q6G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 2304},
                         {"bytes", 516}},
                        {{"name", "w8"},
                         {"kind", "tensor"},
                         {"shape", {1, 33}},
                         {"format", "W8G32_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 3072},
                         {"bytes", 264}},
                        {{"name", "fp8_row"},
                         {"kind", "tensor"},
                         {"shape", {2, 4}},
                         {"format", "FP8_E4M3FN_ROW_BF16S"},
                         {"layout", "row-scale-v1"},
                         {"offset", 3584},
                         {"bytes", 260}},
                    })},
    };
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) { return; }
    throw std::runtime_error(std::string(label) + " was accepted");
}

template <typename Function>
void expect_artifact_error_containing(Function&& function, std::string_view needle,
                                      std::string_view label) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError& error) {
        if (std::string_view(error.what()).find(needle) == std::string_view::npos) {
            throw std::runtime_error(std::string(label) + " reported: " + error.what());
        }
        return;
    }
    throw std::runtime_error(std::string(label) + " was accepted");
}

// The registered qwen3.8-27b layout a version-3 file must carry, whatever its metadata.name.
Json v3_components() {
    return {
        {"text",
         {
             {"config",
              {
                  {"model_type", "qwen3_5_text"},
                  {"architectures", Json::array({"Qwen3_5ForCausalLM"})},
                  {"num_hidden_layers", 64},
                  {"hidden_size", 5120},
                  {"vocab_size", 248320},
                  {"num_attention_heads", 24},
                  {"num_key_value_heads", 4},
                  {"head_dim", 256},
                  {"intermediate_size", 17408},
                  {"linear_num_key_heads", 16},
                  {"linear_key_head_dim", 128},
                  {"linear_num_value_heads", 48},
                  {"linear_value_head_dim", 128},
                  {"linear_conv_kernel_dim", 4},
              }},
             {"proposal", {{"domain", "indexed"}, {"rows", 131072}}},
             {"resources",
              {
                  {"tokenizer.json", "obj/tokenizer.json"},
                  {"tokenizer_config.json", "obj/tokenizer_config.json"},
                  {"chat_template.jinja", "obj/chat_template.jinja"},
                  {"generation_config.json", "obj/generation_config.json"},
              }},
         }},
        {"vision",
         {
             {"config",
              {
                  {"model_type", "qwen3_5_vision"},
                  {"depth", 27},
                  {"hidden_size", 1152},
                  {"patch_size", 16},
                  {"temporal_patch_size", 2},
                  {"spatial_merge_size", 2},
                  {"num_heads", 16},
                  {"intermediate_size", 4304},
              }},
             {"resources",
              {
                  {"preprocessor_config.json", "obj/preprocessor_config.json"},
                  {"video_preprocessor_config.json", "obj/video_preprocessor_config.json"},
              }},
         }},
        {"mtp", {{"config", {{"architectures", Json::array({"Qwen3_5MTP"})}}}}},
    };
}

// Enough of a version-3 directory to pass the framing, single-file and layout gates. Projection
// then fails on the incomplete inventory, which is how a foreign metadata.name is shown to have
// been accepted: none of the failures below are name or layout failures.
Json foreign_name_v3_directory(std::string_view name) {
    Json objects = Json::array();
    for (const auto& [id, offset] : std::array<std::pair<const char*, int>, 6>{
             std::pair{"obj/tokenizer.json", 0},
             std::pair{"obj/tokenizer_config.json", 1},
             std::pair{"obj/chat_template.jinja", 2},
             std::pair{"obj/generation_config.json", 3},
             std::pair{"obj/preprocessor_config.json", 4},
             std::pair{"obj/video_preprocessor_config.json", 5},
         }) {
        objects.push_back({{"id", id},
                           {"kind", "resource"},
                           {"encoding", "raw_bytes_v1"},
                           {"offset", offset},
                           {"bytes", 1}});
    }
    return {
        {"metadata", {{"name", std::string(name)}}},
        {"files", Json::array({{{"path", nullptr}, {"payload_bytes", 6}}})},
        {"objects", std::move(objects)},
        {"bindings", Json::object()},
        {"uses", Json::array()},
        {"components", v3_components()},
    };
}

// metadata.name is provenance: a third-party conversion's own name must reach the projection,
// while the registered layout it depends on is checked independently of that name.
void test_v3_layout_gate() {
    {
        auto fixture = write_v3_fixture(foreign_name_v3_directory("swift-1.5-qwen3.8-27b-orcarouter"),
                                        "v3_foreign_name");
        expect_artifact_error_containing([&] { Reader reader(fixture.path); },
                                         "missing v3 binding",
                                         "foreign-name v3 artifact");
    }
    {
        auto directory = foreign_name_v3_directory("qwen3.8-27b");
        directory["components"]["text"]["config"]["num_hidden_layers"] = 56;
        auto fixture   = write_v3_fixture(directory, "v3_wrong_layer_count");
        expect_artifact_error_containing(
            [&] { Reader reader(fixture.path); },
            "does not match the qwen3.8-27b layout: text.config.num_hidden_layers",
            "qwen3.8-27b-named artifact with another layout");
    }
    {
        auto directory = foreign_name_v3_directory("swift-1.5-qwen3.8-27b-orcarouter");
        directory["components"].erase("vision");
        auto fixture = write_v3_fixture(directory, "v3_missing_vision");
        expect_artifact_error_containing([&] { Reader reader(fixture.path); },
                                         "does not match the qwen3.8-27b layout: components.vision",
                                         "version-3 artifact without Vision");
    }
}

void test_registered_sizes() {
    using ninfer::artifact::tensor_encoded_size;
    constexpr StorageLayout direct   = StorageLayout::ContiguousLeV1;
    constexpr StorageLayout rows     = StorageLayout::RowSplitK128V1;
    constexpr StorageLayout fp8_rows = StorageLayout::RowScaleV1;

    const std::array<std::uint64_t, 2> shape_2x3 = {2, 3};
    const std::array<std::uint64_t, 1> shape_2   = {2};
    const std::array<std::uint64_t, 2> q4_shape  = {1, 1};
    const std::array<std::uint64_t, 2> q5_shape  = {2, 130};
    const std::array<std::uint64_t, 2> q6_shape  = {1, 64};
    const std::array<std::uint64_t, 2> w8_shape  = {1, 33};
    const std::array<std::uint64_t, 2> fp8_shape = {2, 4};

    if (tensor_encoded_size(direct, NumericFormat::BF16, shape_2x3) != 12 ||
        tensor_encoded_size(direct, NumericFormat::FP32, {}) != 4 ||
        tensor_encoded_size(direct, NumericFormat::I32, shape_2) != 8 ||
        tensor_encoded_size(rows, NumericFormat::Q4G64_F16S, q4_shape) != 260 ||
        tensor_encoded_size(rows, NumericFormat::Q5G64_F16S, q5_shape) != 528 ||
        tensor_encoded_size(rows, NumericFormat::Q6G64_F16S, q6_shape) != 516 ||
        tensor_encoded_size(rows, NumericFormat::W8G32_F16S, w8_shape) != 264 ||
        tensor_encoded_size(fp8_rows, NumericFormat::FP8_E4M3FN_ROW_BF16S, fp8_shape) != 260) {
        throw std::runtime_error("registered encoded-size calculation is wrong");
    }
    expect_artifact_error([&] { tensor_encoded_size(fp8_rows, NumericFormat::NVFP4, fp8_shape); },
                          "row-scale format mismatch");
    expect_artifact_error(
        [&] { tensor_encoded_size(fp8_rows, NumericFormat::FP8_E4M3FN_ROW_BF16S, shape_2); },
        "row-scale rank mismatch");
}

void test_normative_fixture() {
    auto fixture = write_fixture(normative_directory(), "valid");
    Reader reader(fixture.path);
    if (reader.identity().model_id != "fixture-model" ||
        reader.identity().weights_id != "fixture-weights" || reader.objects().size() != 9 ||
        reader.payload_offset() != 4096) {
        throw std::runtime_error("fixture root descriptor mismatch");
    }

    const std::array<std::string_view, 9> expected_names = {
        "resource", "bf16", "fp32_scalar", "i32", "q4", "q5", "q6", "w8", "fp8_row",
    };
    for (std::size_t i = 0; i < expected_names.size(); ++i) {
        const auto& object = reader.objects()[i];
        if (ninfer::artifact::object_name(object) != expected_names[i] ||
            reader.find(expected_names[i]) != &object) {
            throw std::runtime_error("fixture name index mismatch");
        }
        const auto payload = reader.payload(object);
        if (payload.absolute_offset !=
                reader.payload_offset() + ninfer::artifact::object_offset(object) ||
            payload.data.size() != ninfer::artifact::object_bytes(object) ||
            payload.data.front() != std::byte(i + 1) || payload.data.back() != std::byte(i + 1)) {
            throw std::runtime_error("fixture payload span mismatch");
        }
    }
    if (reader.find("missing") != nullptr) {
        throw std::runtime_error("missing object unexpectedly resolved");
    }

    const auto* resource = std::get_if<ResourceDescriptor>(&reader.objects().front());
    const auto* q5       = std::get_if<TensorDescriptor>(reader.find("q5"));
    const auto* fp8      = std::get_if<TensorDescriptor>(reader.find("fp8_row"));
    if (resource == nullptr || q5 == nullptr || q5->shape != std::vector<std::uint64_t>({2, 130}) ||
        q5->format != NumericFormat::Q5G64_F16S || q5->layout != StorageLayout::RowSplitK128V1 ||
        fp8 == nullptr || fp8->shape != std::vector<std::uint64_t>({2, 4}) ||
        fp8->format != NumericFormat::FP8_E4M3FN_ROW_BF16S ||
        fp8->layout != StorageLayout::RowScaleV1) {
        throw std::runtime_error("fixture object signature mismatch");
    }
}

void test_common_validation() {
    {
        auto directory                   = normative_directory();
        directory["objects"][5]["bytes"] = 527;
        auto fixture                     = write_fixture(directory, "wrong_encoded_size");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "wrong encoded size");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][1]["offset"] = 257;
        auto fixture                      = write_fixture(directory, "misaligned_offset");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "misaligned offset");
    }
    {
        auto directory = normative_directory();
        auto fixture =
            write_fixture(directory, "legacy_v1", ninfer::test::artifact_fixture::kV1Magic);
        try {
            Reader reader(fixture.path);
        } catch (const ninfer::artifact::ArtifactError& error) {
            if (std::string_view(error.what())
                    .find("python3 -m tools.artifact.migrate_v1_to_v2 <artifact>") ==
                std::string_view::npos) {
                throw std::runtime_error("v1 rejection omitted the migration command");
            }
            return;
        }
        throw std::runtime_error("v1 artifact was accepted");
    }
}

void test_official_qwen38_v3_if_configured() {
    const char* path = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (path == nullptr || *path == '\0') { return; }

    Reader reader(path);
    std::size_t nvfp4_mlp_layers = 0;
    for (int layer = 0; layer < 64; ++layer) {
        const auto* gate_up = std::get_if<TensorDescriptor>(
            reader.find("text/layers/" + std::to_string(layer) + "/mlp/gate_up"));
        if (gate_up == nullptr) {
            throw std::runtime_error("qwen3.8 v3 projection omitted an MLP gate_up tensor");
        }
        if (gate_up->format == NumericFormat::NVFP4) { ++nvfp4_mlp_layers; }
    }
    if (reader.identity().model_id != "qwen3.8-27b" ||
        reader.identity().weights_id != "nvfp4" ||
        reader.objects().size() != 1012 + 2 * nvfp4_mlp_layers +
                                     (reader.find("dflash2/feature_projection") ? 66 : 0)) {
        throw std::runtime_error("qwen3.8 v3 projection identity or inventory mismatch");
    }
    const auto* qkgv = std::get_if<TensorDescriptor>(
        reader.find("text/layers/3/attention/query_key_gate_value"));
    const auto* gate_up =
        std::get_if<TensorDescriptor>(reader.find("text/layers/0/mlp/gate_up"));
    const auto* divisor = std::get_if<TensorDescriptor>(
        reader.find("text/layers/0/mlp/gate_up_projection/input_scale_divisor"));
    if (qkgv == nullptr || qkgv->format != NumericFormat::FP8_E4M3FN_ROW_BF16S ||
        gate_up == nullptr || gate_up->format != NumericFormat::NVFP4 || divisor == nullptr ||
        divisor->format != NumericFormat::FP32 || !divisor->shape.empty()) {
        throw std::runtime_error("official qwen3.8 v3 projected tensor contract mismatch");
    }
    if (reader.find("dflash2/feature_projection") != nullptr) {
        for (const auto& [name, format] : std::array{
                 std::pair{"dflash2/feature_projection", NumericFormat::W8G32_F16S},
                 std::pair{"dflash2/layers/0/attention/query_key_value", NumericFormat::W8G32_F16S},
                 std::pair{"dflash2/candidate_selector/predecessor_codebook", NumericFormat::BF16},
             }) {
            const auto* tensor = std::get_if<TensorDescriptor>(reader.find(name));
            if (tensor == nullptr || tensor->format != format) {
                throw std::runtime_error(std::string("NVFP4 v3 DFlash2 projection omitted ") + name);
            }
        }
    }

    const auto tokenizer_config = reader.payload("frontend/tokenizer_config.json").data;
    const auto chat_template    = reader.payload("frontend/chat_template.jinja").data;
    const auto* begin = reinterpret_cast<const char*>(tokenizer_config.data());
    const Json config = Json::parse(begin, begin + tokenizer_config.size());
    const auto expected = config.at("chat_template").get<std::string>();
    const std::string_view actual(reinterpret_cast<const char*>(chat_template.data()),
                                  chat_template.size());
    if (actual != expected) {
        throw std::runtime_error("official qwen3.8 v3 chat template projection mismatch");
    }
}

void test_gsq_qwen38_v3_vision_if_configured() {
    const char* path = std::getenv("NINFER_QWEN3_8_27B_GSQ_WEIGHTS");
    if (path == nullptr || *path == '\0') { return; }

    Reader reader(path);
    if (reader.identity().model_id != "qwen3.8-27b" ||
        reader.identity().weights_id != "gguf-blocks") {
        throw std::runtime_error("GSQ v3 projection identity mismatch");
    }
    for (const auto& [name, format] : std::array{
             std::pair{"vision/patch_embedding", NumericFormat::Q6G64_F16S},
             std::pair{"vision/layers/0/attention/qkv", NumericFormat::Q4G64_F16S},
             std::pair{"vision/layers/26/mlp/fc2", NumericFormat::Q5G64_F16S},
             std::pair{"vision/merger/fc2", NumericFormat::W8G32_F16S},
         }) {
        const auto* tensor = std::get_if<TensorDescriptor>(reader.find(name));
        if (tensor == nullptr || tensor->format != format) {
            throw std::runtime_error(std::string("GSQ v3 Vision projection omitted ") + name);
        }
    }
    if (reader.find("frontend/preprocessor_config.json") == nullptr ||
        reader.find("frontend/video_preprocessor_config.json") == nullptr) {
        throw std::runtime_error("GSQ v3 Vision preprocessing resources are missing");
    }
    for (const auto& [name, format, shape] : std::array{
             std::tuple{"dflash2/feature_projection", NumericFormat::W8G32_F16S,
                        std::array<std::uint64_t, 2>{5120, 25600}},
             std::tuple{"dflash2/layers/0/attention/query_key_value", NumericFormat::W8G32_F16S,
                        std::array<std::uint64_t, 2>{6144, 5120}},
             std::tuple{"dflash2/layers/4/mlp/gate_up", NumericFormat::W8G32_F16S,
                        std::array<std::uint64_t, 2>{34816, 5120}},
             std::tuple{"dflash2/candidate_selector/predecessor_codebook", NumericFormat::BF16,
                        std::array<std::uint64_t, 2>{248320, 256}},
         }) {
        const auto* tensor = std::get_if<TensorDescriptor>(reader.find(name));
        if (tensor == nullptr || tensor->format != format ||
            tensor->shape != std::vector<std::uint64_t>(shape.begin(), shape.end())) {
            throw std::runtime_error(std::string("GSQ v3 DFlash2 projection omitted ") + name);
        }
    }
}

void test_swift_qwen38_v3_if_configured() {
    const char* path = std::getenv("NINFER_QWEN3_8_27B_SWIFT_WEIGHTS");
    if (path == nullptr || *path == '\0') { return; }

    Reader reader(path);
    if (reader.identity().model_id != "qwen3.8-27b" ||
        reader.identity().weights_id != "nvfp4") {
        throw std::runtime_error("Swift v3 projection identity mismatch");
    }
    std::size_t nvfp4_mlp_layers = 0;
    for (int layer = 0; layer < 64; ++layer) {
        const auto* gate_up = std::get_if<TensorDescriptor>(
            reader.find("text/layers/" + std::to_string(layer) + "/mlp/gate_up"));
        if (gate_up == nullptr) {
            throw std::runtime_error("Swift v3 projection omitted an MLP gate_up tensor");
        }
        if (gate_up->format == NumericFormat::NVFP4) { ++nvfp4_mlp_layers; }
    }
    if (nvfp4_mlp_layers != 64) {
        throw std::runtime_error("Swift v3 MLP layers are not all NVFP4");
    }
    if (reader.objects().size() != 1012 + 2 * nvfp4_mlp_layers +
                                     (reader.find("dflash2/feature_projection") ? 66 : 0)) {
        throw std::runtime_error("Swift v3 projection inventory mismatch");
    }
    for (const auto& [name, format] : std::array{
             std::pair{"text/layers/3/attention/query_key_gate_value",
                       NumericFormat::FP8_E4M3FN_ROW_BF16S},
             std::pair{"text/layers/0/mlp/gate_up_projection/input_scale_divisor", NumericFormat::FP32},
             std::pair{"vision/merger/fc2", NumericFormat::W8G32_F16S},
             std::pair{"dflash2/layers/4/mlp/gate_up", NumericFormat::W8G32_F16S},
         }) {
        const auto* tensor = std::get_if<TensorDescriptor>(reader.find(name));
        if (tensor == nullptr || tensor->format != format) {
            throw std::runtime_error(std::string("Swift v3 projection omitted ") + name);
        }
    }

    const auto tokenizer_config = reader.payload("frontend/tokenizer_config.json").data;
    const auto chat_template    = reader.payload("frontend/chat_template.jinja").data;
    const auto* begin = reinterpret_cast<const char*>(tokenizer_config.data());
    const Json config = Json::parse(begin, begin + tokenizer_config.size());
    if (config.at("chat_template").get<std::string>() !=
        std::string_view(reinterpret_cast<const char*>(chat_template.data()), chat_template.size())) {
        throw std::runtime_error("Swift v3 chat template projection mismatch");
    }
}

} // namespace

int main() {
    try {
        test_registered_sizes();
        test_normative_fixture();
        test_common_validation();
        test_v3_layout_gate();
        test_official_qwen38_v3_if_configured();
        test_gsq_qwen38_v3_vision_if_configured();
        test_swift_qwen38_v3_if_configured();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
