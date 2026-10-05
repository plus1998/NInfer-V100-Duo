#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::test::artifact_fixture {

using Json = nlohmann::json;

inline constexpr std::array<std::uint8_t, 8> kV1Magic = {
    'N', 'I', 'N', 'F', 'E', 'R', 0, 1,
};
inline constexpr std::array<std::uint8_t, 8> kMagic = {
    'N', 'I', 'N', 'F', 'E', 'R', 0, 2,
};
inline constexpr std::array<std::uint8_t, 8> kV3Magic = {
    'N', 'I', 'N', 'F', 'E', 'R', 0, 3,
};
inline constexpr std::uint64_t kV3HeaderBytes = 32;

inline std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

inline void write_u64_le(std::byte* output, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) { output[i] = std::byte((value >> (i * 8)) & 0xff); }
}

struct TemporaryArtifact {
    std::filesystem::path path;

    ~TemporaryArtifact() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

inline TemporaryArtifact write_fixture(const Json& directory, std::string_view suffix,
                                       const std::array<std::uint8_t, 8>& magic = kMagic) {
    const std::string json         = directory.dump();
    const auto payload_offset      = align_up(16 + json.size(), 4096);
    const auto nonnegative_integer = [](const Json& value) {
        return value.is_number_unsigned() ||
               (value.is_number_integer() && value.get<std::int64_t>() >= 0);
    };
    std::uint64_t payload_bytes = 0;
    for (const auto& object : directory.at("objects")) {
        if (object.contains("offset") && object.contains("bytes") &&
            nonnegative_integer(object.at("offset")) && nonnegative_integer(object.at("bytes"))) {
            payload_bytes = std::max(payload_bytes, object.at("offset").get<std::uint64_t>() +
                                                        object.at("bytes").get<std::uint64_t>());
        }
    }

    std::vector<std::byte> file(payload_offset + payload_bytes, std::byte{0});
    for (std::size_t i = 0; i < magic.size(); ++i) { file[i] = std::byte{magic[i]}; }
    write_u64_le(file.data() + 8, json.size());
    std::memcpy(file.data() + 16, json.data(), json.size());

    std::uint8_t marker = 1;
    for (const auto& object : directory.at("objects")) {
        if (object.contains("offset") && object.contains("bytes") &&
            nonnegative_integer(object.at("offset")) && nonnegative_integer(object.at("bytes"))) {
            const auto offset = object.at("offset").get<std::uint64_t>();
            const auto bytes  = object.at("bytes").get<std::uint64_t>();
            std::fill_n(file.data() + payload_offset + offset, bytes, std::byte{marker++});
        }
    }

    auto path = std::filesystem::temp_directory_path() /
                ("ninfer_artifact_" + std::string(suffix) + ".ninfer");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(file.data()),
                 static_cast<std::streamsize>(file.size()));
    if (!output) { throw std::runtime_error("failed to write artifact fixture"); }
    return {std::move(path)};
}

// A version-3 file: a 32-byte header (magic, `u64 json_bytes` at offset 8), the JSON directory at
// offset 32, payload at align_up(32 + json_bytes, 4096). The directory's `files[0].payload_bytes`
// must equal the payload, which is what the reader checks, so it is read from the directory itself.
inline TemporaryArtifact write_v3_fixture(const Json& directory, std::string_view suffix) {
    const std::string json        = directory.dump();
    const auto payload_offset     = align_up(kV3HeaderBytes + json.size(), 4096);
    const auto payload_bytes =
        directory.at("files").at(0).at("payload_bytes").get<std::uint64_t>();
    const auto nonnegative_integer = [](const Json& value) {
        return value.is_number_unsigned() ||
               (value.is_number_integer() && value.get<std::int64_t>() >= 0);
    };

    std::vector<std::byte> file(payload_offset + payload_bytes, std::byte{0});
    for (std::size_t i = 0; i < kV3Magic.size(); ++i) { file[i] = std::byte{kV3Magic[i]}; }
    write_u64_le(file.data() + 8, json.size());
    std::memcpy(file.data() + kV3HeaderBytes, json.data(), json.size());

    std::uint8_t marker = 1;
    for (const auto& object : directory.at("objects")) {
        if (object.contains("offset") && object.contains("bytes") &&
            nonnegative_integer(object.at("offset")) && nonnegative_integer(object.at("bytes"))) {
            const auto offset = object.at("offset").get<std::uint64_t>();
            const auto bytes  = object.at("bytes").get<std::uint64_t>();
            if (offset > payload_bytes || bytes > payload_bytes - offset) {
                throw std::runtime_error("v3 fixture object exceeds its payload");
            }
            std::fill_n(file.data() + payload_offset + offset, bytes, std::byte{marker++});
        }
    }

    auto path = std::filesystem::temp_directory_path() /
                ("ninfer_artifact_v3_" + std::string(suffix) + ".ninfer");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(file.data()),
                 static_cast<std::streamsize>(file.size()));
    if (!output) { throw std::runtime_error("failed to write v3 artifact fixture"); }
    return {std::move(path)};
}

} // namespace ninfer::test::artifact_fixture
