#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer {

struct Tensor {
    void* data         = nullptr;
    DType dtype        = DType::BF16;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};

enum class QType : std::uint16_t {
    Q4G64_F16S           = 0,
    Q5G64_F16S           = 1,
    Q6G64_F16S           = 2,
    W8G32_F16S           = 3,
    BF16_CTRL            = 4,
    FP32_CTRL            = 5,
    I32_CTRL             = 6,
    NVFP4                = 7,
    FP8_E4M3FN_ROW_BF16S = 8,
    GGML_K               = 9,
    GGUF                 = 10,
};

// One ggml block type of a GGUF-blocks weight. Every block covers 256 values.
enum class GgufType : std::uint8_t {
    IQ4_XS  = 0,
    IQ3_S   = 1,
    IQ3_XXS = 2,
    IQ2_XS  = 3,
    IQ2_XXS = 4,
    IQ2_S   = 5,
    IQ1_M   = 6,
    Q2_K    = 7,
    Q4_K    = 8,
    Q6_K    = 9,
};

inline constexpr int kGgufTypeCount = 10;

constexpr int gguf_block_bytes(GgufType type) {
    switch (type) {
    case GgufType::IQ4_XS: return 136;
    case GgufType::IQ3_S: return 110;
    case GgufType::IQ3_XXS: return 98;
    case GgufType::IQ2_XS: return 74;
    case GgufType::IQ2_XXS: return 66;
    case GgufType::IQ2_S: return 82;
    case GgufType::IQ1_M: return 56;
    case GgufType::Q2_K: return 84;
    case GgufType::Q4_K: return 144;
    case GgufType::Q6_K: return 210;
    }
    return 0;
}

// A GGUF-blocks weight is up to four consecutive row runs, each stored as whole rows of one ggml
// block type in its own allocation. Row r of the logical matrix is row (r - first_row) of the
// run containing it; a run's rows are `row_bytes = (k / 256) * gguf_block_bytes(type)` apart.
struct GgufSegment {
    const void* data   = nullptr;
    std::int32_t rows  = 0;
    GgufType type      = GgufType::Q4_K;
};

inline constexpr int kMaxGgufSegments = 4;

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    VoltaQpnPrepacked    = 4,
    GgmlK256            = 5,
    GgufBlocks          = 6,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4G64_F16S;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;

    // GGML_K rows may mix Q4_K and Q6_K. A nonnegative first_q6 plus a
    // type_change in [0,n] describes either one uniform run (type_change=n)
    // or exactly two runs. Other layouts leave both fields at -1.
    std::int32_t ggml_k_first_q6   = -1;
    std::int32_t ggml_k_type_change = -1;

    // GGUF (QuantLayout::GgufBlocks) row runs, in logical row order; other layouts leave
    // gguf_segment_count at zero.
    GgufSegment gguf_segments[kMaxGgufSegments] = {};
    std::int32_t gguf_segment_count             = 0;
};

} // namespace ninfer
