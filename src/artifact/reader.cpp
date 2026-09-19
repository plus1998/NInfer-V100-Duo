#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::array<std::byte, 8> kV1Magic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{1},
};
constexpr std::array<std::byte, 8> kV3Magic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{3},
};
constexpr std::uint64_t kPrefixBytes      = 16;
constexpr std::uint64_t kV3HeaderBytes    = 32;
constexpr std::uint64_t kPayloadAlignment = 4096;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    if (name == "GGML_K") { return NumericFormat::GGML_K; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "blockscale-k16-m128x4-v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row-scale-v1") { return StorageLayout::RowScaleV1; }
    if (name == "ggml-k256-v1") { return StorageLayout::GgmlK256V1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape, stored_size);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
    }

    ~MappedFile() {
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
    }

private:
    int fd_                = -1;
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

struct V3CompatibilityDirectory {
    Json directory;
    std::map<std::string, std::vector<std::byte>, std::less<>> payload_overrides;
};

std::uint64_t require_v3_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)) {
        throw ArtifactError(std::string(label) + " must be a nonnegative integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

class Qwen38Nvfp4V3Adapter {
public:
    Qwen38Nvfp4V3Adapter(const Json& directory, const MappedFile& file,
                         std::uint64_t payload_start)
        : directory_(directory), file_(file), payload_start_(payload_start) {
        if (!directory_.is_object() || !directory_.contains("metadata") ||
            !directory_.at("metadata").is_object() ||
            directory_.at("metadata").value("name", "") != "qwen3.8-27b") {
            throw ArtifactError("NInfer v3 compatibility is limited to qwen3.8-27b");
        }
        if (!directory_.contains("files") || !directory_.at("files").is_array() ||
            directory_.at("files").size() != 1 ||
            !directory_.at("files")[0].at("path").is_null()) {
            throw ArtifactError("qwen3.8-27b v3 compatibility requires a single-file artifact");
        }
        const auto payload_bytes = require_v3_unsigned(
            directory_.at("files")[0].at("payload_bytes"), "payload_bytes", true);
        if (checked_add(payload_start_, payload_bytes, "v3 artifact length") != file_.size()) {
            throw ArtifactError("v3 artifact length differs from its directory");
        }
        if (!directory_.contains("objects") || !directory_.at("objects").is_array() ||
            !directory_.contains("bindings") || !directory_.at("bindings").is_object() ||
            !directory_.contains("uses") || !directory_.at("uses").is_array() ||
            !directory_.contains("components") || !directory_.at("components").is_object()) {
            throw ArtifactError("qwen3.8-27b v3 directory is incomplete");
        }
        for (const auto& object : directory_.at("objects")) {
            const auto& id = require_string(object.at("id"), "v3 object id");
            if (!objects_.emplace(id, &object).second) {
                throw ArtifactError("duplicate v3 object id: " + id);
            }
        }
    }

    V3CompatibilityDirectory build() {
        for (const auto& [role, component] : std::array{
                 std::pair{"tokenizer.json", "text"},
                 std::pair{"tokenizer_config.json", "text"},
                 std::pair{"chat_template.jinja", "text"},
                 std::pair{"generation_config.json", "text"},
                 std::pair{"preprocessor_config.json", "vision"},
                 std::pair{"video_preprocessor_config.json", "vision"},
             }) {
            add_resource("frontend/" + std::string(role), component, role);
        }

        for (const char* name : {"text/token_embedding", "text/output_head", "text/final_norm"}) {
            add(name, {name});
        }
        add("text/draft_head", {"proposal/head"});
        add("text/draft_head_token_ids", {"proposal/token_ids"});

        for (int layer = 0; layer < 64; ++layer) {
            const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
            add(prefix + "input_norm", {prefix + "input_norm"});
            if (layer >= 3 && (layer - 3) % 4 == 0) {
                add(prefix + "attention/query_key_gate_value",
                    {prefix + "attention/query", prefix + "attention/key",
                     prefix + "attention/gate", prefix + "attention/value"});
                for (const char* role : {"query_norm", "key_norm", "output"}) {
                    add(prefix + "attention/" + role, {prefix + "attention/" + role});
                }
            } else {
                for (const char* role : {"a_log", "dt_bias", "convolution"}) {
                    add(prefix + "gdn/" + role, {prefix + "gdn/" + role});
                }
                add(prefix + "gdn/a_b_projection",
                    {prefix + "gdn/a_projection", prefix + "gdn/b_projection"});
                add(prefix + "gdn/query_key_value_z",
                    {prefix + "gdn/query", prefix + "gdn/key", prefix + "gdn/value",
                     prefix + "gdn/z"});
                for (const char* role : {"norm", "output"}) {
                    add(prefix + "gdn/" + role, {prefix + "gdn/" + role});
                }
            }
            add(prefix + "post_attention_norm", {prefix + "post_attention_norm"});
            const std::string gate = prefix + "mlp/gate";
            const std::string up   = prefix + "mlp/up";
            const auto& gate_up = add(prefix + "mlp/gate_up", {gate, up});
            add(prefix + "mlp/down", {prefix + "mlp/down"});
            if (require_string(gate_up.at("format"), "v3 tensor format") == "nvfp4") {
                add_input_divisor(prefix + "mlp/gate_up_projection/input_scale_divisor",
                                  {gate, up});
                add_input_divisor(prefix + "mlp/down_projection/input_scale_divisor",
                                  {prefix + "mlp/down"});
            }
        }

        for (const char* name : {"input_projection", "embedding_norm", "hidden_norm", "final_norm"}) {
            add("mtp/" + std::string(name), {"mtp/" + std::string(name)});
        }
        const std::string old_mtp = "mtp/layer/";
        const std::string new_mtp = "mtp/layers/0/";
        for (const char* role : {"input_norm", "post_attention_norm"}) {
            add(old_mtp + role, {new_mtp + role});
        }
        add(old_mtp + "attention/query_key_gate_value",
            {new_mtp + "attention/query", new_mtp + "attention/key",
             new_mtp + "attention/gate", new_mtp + "attention/value"});
        for (const char* role : {"query_norm", "key_norm", "output"}) {
            add(old_mtp + "attention/" + role, {new_mtp + "attention/" + role});
        }
        add(old_mtp + "mlp/gate_up", {new_mtp + "mlp/gate", new_mtp + "mlp/up"});
        add(old_mtp + "mlp/down", {new_mtp + "mlp/down"});

        for (const char* name : {"patch_embedding", "patch_embedding_bias", "position_embedding"}) {
            add("vision/" + std::string(name), {"vision/" + std::string(name)});
        }
        for (int layer = 0; layer < 27; ++layer) {
            const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
            add(prefix + "attention/qkv",
                {prefix + "attention/query", prefix + "attention/key", prefix + "attention/value"});
            add(prefix + "attention/qkv_bias",
                {prefix + "attention/query_bias", prefix + "attention/key_bias",
                 prefix + "attention/value_bias"});
            for (const char* role : {"output", "output_bias"}) {
                add(prefix + "attention/" + role, {prefix + "attention/" + role});
            }
            for (const char* role : {"fc1", "fc1_bias", "fc2", "fc2_bias"}) {
                add(prefix + "mlp/" + role, {prefix + "mlp/" + role});
            }
            for (const char* norm : {"norm1", "norm2"}) {
                for (const char* part : {"weight", "bias"}) {
                    add(prefix + norm + "/" + part, {prefix + norm + "_" + part});
                }
            }
        }
        for (const char* role : {"fc1", "fc1_bias", "fc2", "fc2_bias"}) {
            add("vision/merger/" + std::string(role), {"vision/merger/" + std::string(role)});
        }
        for (const char* part : {"weight", "bias"}) {
            add("vision/merger/norm/" + std::string(part),
                {"vision/merger/norm_" + std::string(part)});
        }

        if (selected_.size() != 1124) {
            throw ArtifactError("qwen3.8-27b v3 projection did not produce 1124 objects");
        }
        Json objects = Json::array();
        for (const auto& [old_name, object] : selected_) {
            Json converted{
                {"name", old_name},
                {"kind", object->at("kind")},
                {"offset", require_v3_unsigned(object->at("offset"), "v3 object offset", false)},
                {"bytes", require_v3_unsigned(object->at("bytes"), "v3 object bytes", true)}};
            const auto kind = require_string(object->at("kind"), "v3 object kind");
            if (kind == "tensor") {
                static const std::map<std::string, std::string, std::less<>> formats = {
                    {"bf16", "BF16"}, {"fp32", "FP32"}, {"int32", "I32"},
                    {"q4_g64_fp16", "Q4G64_F16S"}, {"q5_g64_fp16", "Q5G64_F16S"},
                    {"q6_g64_fp16", "Q6G64_F16S"}, {"q8_g32_fp16", "W8G32_F16S"},
                    {"nvfp4", "NVFP4"}, {"fp8_e4m3fn_row_bf16", "FP8_E4M3FN_ROW_BF16S"},
                };
                static const std::map<std::string, std::string, std::less<>> layouts = {
                    {"contiguous_le_v1", "contiguous-le-v1"},
                    {"row_split_k128_v1", "row-split-k128-v1"},
                    {"block_scale_k16_m128x4_v1", "blockscale-k16-m128x4-v1"},
                    {"row_scale_v1", "row-scale-v1"},
                };
                converted["shape"] = Json::array();
                for (const auto& dimension : object->at("shape")) {
                    converted["shape"].push_back(
                        require_v3_unsigned(dimension, "v3 shape dimension", true));
                }
                converted["format"] = formats.at(require_string(object->at("format"), "v3 format"));
                converted["layout"] = layouts.at(require_string(object->at("layout"), "v3 layout"));
            } else if (kind == "resource" &&
                       require_string(object->at("encoding"), "v3 resource encoding") ==
                           "raw_bytes_v1") {
                converted["encoding"] = "raw-bytes-v1";
            } else {
                throw ArtifactError("unsupported qwen3.8-27b v3 object representation");
            }
            objects.push_back(std::move(converted));
        }
        std::sort(objects.begin(), objects.end(), [](const Json& a, const Json& b) {
            return a.at("offset").get<std::uint64_t>() < b.at("offset").get<std::uint64_t>();
        });

        const auto template_bytes = embedded_chat_template();
        auto template_it = std::find_if(objects.begin(), objects.end(), [](const Json& object) {
            return object.at("name") == "frontend/chat_template.jinja";
        });
        if (template_it == objects.end() || template_bytes.size() > template_it->at("bytes").get<std::uint64_t>()) {
            throw ArtifactError("embedded chat template does not fit its resource allocation");
        }
        (*template_it)["bytes"] = template_bytes.size();
        V3CompatibilityDirectory result;
        result.directory = {{"identity", {{"model_id", "qwen3.8-27b"}, {"weights_id", "nvfp4"}}},
                            {"objects", std::move(objects)}};
        result.payload_overrides.emplace("frontend/chat_template.jinja", template_bytes);
        return result;
    }

private:
    const Json& object(std::string_view id) const {
        const auto found = objects_.find(id);
        if (found == objects_.end()) { throw ArtifactError("missing v3 object: " + std::string(id)); }
        return *found->second;
    }

    std::vector<Json> binding_parts(const std::string& logical) const {
        const auto& bindings = directory_.at("bindings");
        if (!bindings.contains(logical)) { throw ArtifactError("missing v3 binding: " + logical); }
        const auto& binding = bindings.at(logical);
        if (binding.contains("object")) {
            const auto id = require_string(binding.at("object"), "v3 binding object");
            const auto& value = object(id);
            std::uint64_t elements = 1;
            for (const auto& dimension : value.at("shape")) {
                const auto dim = require_v3_unsigned(dimension, "v3 shape dimension", true);
                if (elements > std::numeric_limits<std::uint64_t>::max() / dim) {
                    throw ArtifactError("v3 tensor shape overflows u64");
                }
                elements *= dim;
            }
            return {Json{{"object", id}, {"range", {0, elements}}}};
        }
        if (!binding.contains("parts") || !binding.at("parts").is_array()) {
            throw ArtifactError("invalid v3 binding: " + logical);
        }
        return binding.at("parts").get<std::vector<Json>>();
    }

    const Json& add(const std::string& old_name, std::initializer_list<std::string> logical_names) {
        std::vector<Json> parts;
        for (const auto& logical : logical_names) {
            auto binding = binding_parts(logical);
            parts.insert(parts.end(), binding.begin(), binding.end());
        }
        std::string object_id;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        for (const auto& part : parts) {
            const auto id = require_string(part.at("object"), "v3 binding object");
            if (object_id.empty()) { object_id = id; }
            if (id != object_id) { throw ArtifactError(old_name + ": binding spans physical objects"); }
            const auto& range = part.at("range");
            if (!range.is_array() || range.size() != 2) {
                throw ArtifactError(old_name + ": invalid binding range");
            }
            ranges.emplace_back(require_v3_unsigned(range[0], "v3 binding range", false),
                                require_v3_unsigned(range[1], "v3 binding range", true));
        }
        std::sort(ranges.begin(), ranges.end());
        std::uint64_t cursor = 0;
        for (const auto [begin, end] : ranges) {
            if (begin != cursor || end <= begin) {
                throw ArtifactError(old_name + ": binding ranges are not one packed object");
            }
            cursor = end;
        }
        const auto& value = object(object_id);
        std::uint64_t elements = 1;
        for (const auto& dimension : value.at("shape")) {
            elements *= require_v3_unsigned(dimension, "v3 shape dimension", true);
        }
        if (cursor != elements) { throw ArtifactError(old_name + ": binding coverage is incomplete"); }
        select(old_name, object_id);
        return value;
    }

    void add_input_divisor(const std::string& old_name,
                           std::initializer_list<std::string> logical_names) {
        std::string selected_id;
        std::span<const std::byte> selected_bytes;
        for (const auto& logical : logical_names) {
            std::set<std::string> candidates;
            for (const auto& use : directory_.at("uses")) {
                if (use.value("parameter", "") != logical || !use.contains("auxiliaries")) { continue; }
                const auto& auxiliaries = use.at("auxiliaries");
                if (auxiliaries.contains("activation_input_divisor")) {
                    candidates.insert(require_string(
                        auxiliaries.at("activation_input_divisor").at("object"),
                        "activation input divisor"));
                }
            }
            if (candidates.size() != 1) {
                throw ArtifactError(logical + ": expected one activation input divisor");
            }
            const auto& id = *candidates.begin();
            const auto& value = object(id);
            if (value.at("format") != "fp32" || value.at("layout") != "contiguous_le_v1" ||
                !value.at("shape").empty()) {
                throw ArtifactError(logical + ": invalid activation input divisor");
            }
            const auto bytes = object_payload(value);
            if (selected_id.empty()) {
                selected_id = id;
                selected_bytes = bytes;
            } else if (!std::equal(bytes.begin(), bytes.end(), selected_bytes.begin(), selected_bytes.end())) {
                throw ArtifactError(old_name + ": fused parameters have different input divisors");
            }
        }
        select(old_name, selected_id);
    }

    void add_resource(const std::string& old_name, const std::string& component,
                      const std::string& role) {
        const auto& components = directory_.at("components");
        const auto id = require_string(components.at(component).at("resources").at(role),
                                       "v3 component resource");
        select(old_name, id);
    }

    void select(const std::string& old_name, const std::string& object_id) {
        if (!selected_.emplace(old_name, &object(object_id)).second) {
            throw ArtifactError("duplicate projected object name: " + old_name);
        }
        if (!physical_names_.emplace(object_id, old_name).second) {
            throw ArtifactError("v3 physical object is selected more than once: " + object_id);
        }
    }

    std::span<const std::byte> object_payload(const Json& value) const {
        const auto offset = require_v3_unsigned(value.at("offset"), "v3 object offset", false);
        const auto bytes  = require_v3_unsigned(value.at("bytes"), "v3 object bytes", true);
        const auto begin  = checked_add(payload_start_, offset, "v3 object payload");
        const auto end    = checked_add(begin, bytes, "v3 object payload");
        if (end > file_.size()) { throw ArtifactError("v3 object extends beyond the file"); }
        return {file_.data() + begin, static_cast<std::size_t>(bytes)};
    }

    std::vector<std::byte> embedded_chat_template() const {
        const auto& components = directory_.at("components");
        const auto id = require_string(
            components.at("text").at("resources").at("tokenizer_config.json"),
            "tokenizer_config.json resource");
        const auto bytes = object_payload(object(id));
        Json config;
        try {
            const auto* begin = reinterpret_cast<const char*>(bytes.data());
            config = Json::parse(begin, begin + bytes.size());
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid tokenizer_config.json: ") + error.what());
        }
        const auto& value = require_string(config.at("chat_template"), "chat_template");
        const auto raw = std::as_bytes(std::span(value.data(), value.size()));
        return {raw.begin(), raw.end()};
    }

    const Json& directory_;
    const MappedFile& file_;
    std::uint64_t payload_start_;
    std::map<std::string, const Json*, std::less<>> objects_;
    std::map<std::string, const Json*, std::less<>> selected_;
    std::map<std::string, std::string, std::less<>> physical_names_;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (file.size() < kPrefixBytes) {
            throw ArtifactError("artifact is shorter than the NInfer prefix");
        }
        if (std::equal(kV1Magic.begin(), kV1Magic.end(), file.data())) {
            throw ArtifactError("NInfer artifact v1 is no longer supported; migrate it with: "
                                "python3 -m tools.artifact.migrate_v1_to_v2 <artifact>");
        }
        const bool v2 = std::equal(kMagic.begin(), kMagic.end(), file.data());
        const bool v3 = std::equal(kV3Magic.begin(), kV3Magic.end(), file.data());
        if (!v2 && !v3) { throw ArtifactError("artifact magic is not NInfer v2 or supported v3"); }
        const std::uint64_t header_bytes = v3 ? kV3HeaderBytes : kPrefixBytes;
        if (file.size() < header_bytes) { throw ArtifactError("artifact header is truncated"); }
        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(header_bytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + header_bytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        std::map<std::string, std::vector<std::byte>, std::less<>> named_overrides;
        if (v3) {
            auto compatible = Qwen38Nvfp4V3Adapter(directory, file, payload_start).build();
            directory       = std::move(compatible.directory);
            named_overrides = std::move(compatible.payload_overrides);
        }

        static constexpr std::array root_members = {"identity", "objects"};
        require_members(directory, root_members, "directory root");
        const auto& raw_identity                     = directory.at("identity");
        static constexpr std::array identity_members = {"model_id", "weights_id"};
        require_members(raw_identity, identity_members, "artifact identity");
        identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
        identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object          = parse_object(raw_object);
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);

            if (offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) {
                throw ArtifactError("object " + std::string(name) + " extends beyond the file");
            }
            if (const auto* tensor = std::get_if<TensorDescriptor>(&object);
                tensor != nullptr && tensor->format == NumericFormat::GGML_K) {
                validate_ggml_k_payload(tensor->shape,
                    std::span<const std::byte>(file.data() + payload_start + offset, bytes));
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            if (auto override = named_overrides.find(std::string(name));
                override != named_overrides.end()) {
                payload_overrides.emplace(std::string(name), std::move(override->second));
            }
            cursor = end;
        }
        if (payload_overrides.size() != named_overrides.size()) {
            throw ArtifactError("v3 payload override does not name a projected object");
        }
    }

    MappedFile file;
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::unordered_map<std::string, std::vector<std::byte>, TransparentStringHash, std::equal_to<>>
        payload_overrides;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    if (const auto override = impl_->payload_overrides.find(object_name(object));
        override != impl_->payload_overrides.end()) {
        return {0, override->second};
    }
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
