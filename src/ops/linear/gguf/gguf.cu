#include "ops/linear/gguf/gguf.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/gguf/gguf_codec.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

using gguf::decode8;

// Vector passes re-read the weights once per kGgufVectorTokens columns; above this width one
// dequantize-and-GEMM pass is cheaper whenever the caller has the workspace for it.
constexpr std::int32_t kVectorMaxTokens = 16;
// Rows dequantized per tensor-core step; bounds the FP16 weight tile in workspace.
constexpr std::int32_t kGemmRowTile = 2048;

std::int64_t row_bytes(GgufType type, std::int32_t k) {
    return static_cast<std::int64_t>(k / 256) * gguf_block_bytes(type);
}

struct Outputs {
    __nv_bfloat16* data[4]  = {};
    std::int32_t rows[4]    = {};
    std::int64_t stride[4]  = {}; // elements between consecutive columns
    std::int32_t count      = 0;
    bool add                = false;

    __device__ void store(int row, int token, float value) const {
        int section = 0;
        while (section + 1 < count && row >= rows[section]) { row -= rows[section++]; }
        __nv_bfloat16* p = data[section] + token * stride[section] + row;
        *p = __float2bfloat16_rn(add ? __bfloat162float(*p) + value : value);
    }
};

// llama.cpp keeps GDN value heads tiled [repeat, key_head, 128] in ssm_out's input columns while
// the activation is grouped [key_head, repeat, 128]; stored column c multiplies this element.
template <bool Tiled>
__device__ __forceinline__ int input_column(int column, int k) {
    if constexpr (!Tiled) {
        return column;
    } else {
        const int head    = column >> 7;
        const int grouped = k == 3072 ? (head & 7) * 3 + (head >> 3) : (head & 15) * 3 + (head >> 4);
        return grouped * 128 + (column & 127);
    }
}

__device__ __forceinline__ void load_bf16x8(const __nv_bfloat16* p, float (&v)[8]) {
    const uint4 packed = __ldg(reinterpret_cast<const uint4*>(p));
    const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i]     = __uint_as_float(words[i] << 16);
        v[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000U);
    }
}

// One warp per row; lane j decodes values [8j, 8j+8) of every 256-value block of its row, so a
// block is decoded exactly once and reused by every column of the pass.
template <GgufType Type, int Tokens, bool Tiled>
__global__ void __launch_bounds__(128)
vector_kernel(const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ rows,
              int row_count, int first_row, int k, int token_base, Outputs out) {
    const int local = blockIdx.x * 4 + static_cast<int>(threadIdx.x >> 5);
    if (local >= row_count) { return; }
    const int lane             = static_cast<int>(threadIdx.x & 31);
    constexpr int kBlockBytes  = gguf_block_bytes(Type);
    const unsigned char* row   = rows + static_cast<std::int64_t>(local) * (k / 256) * kBlockBytes;
    float sum[Tokens]          = {};
    for (int block = 0; block < k / 256; ++block) {
        float w[8];
        decode8<Type>(row + block * kBlockBytes, lane, w);
        const int column = input_column<Tiled>(block * 256 + lane * 8, k);
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            float a[8];
            load_bf16x8(x + static_cast<std::int64_t>(t) * k + column, a);
#pragma unroll
            for (int j = 0; j < 8; ++j) { sum[t] = fmaf(w[j], a[j], sum[t]); }
        }
    }
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            sum[t] += __shfl_xor_sync(0xffffffffU, sum[t], delta);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) { out.store(first_row + local, token_base + t, sum[t]); }
    }
}

template <GgufType Type, bool Tiled>
void launch_vector_type(const __nv_bfloat16* x, const GgufSegment& segment, int first_row, int k,
                        int tokens, int token_base, const Outputs& out, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((segment.rows + 3) / 4));
    const auto* rows = static_cast<const unsigned char*>(segment.data);
    switch (tokens) {
#define NINFER_GGUF_VECTOR_CASE(T)                                                             \
    case T:                                                                                    \
        vector_kernel<Type, T, Tiled><<<grid, 128, 0, stream>>>(x, rows, segment.rows,          \
                                                                first_row, k, token_base, out); \
        break;
        NINFER_GGUF_VECTOR_CASE(1)
        NINFER_GGUF_VECTOR_CASE(2)
        NINFER_GGUF_VECTOR_CASE(3)
        NINFER_GGUF_VECTOR_CASE(4)
        NINFER_GGUF_VECTOR_CASE(5)
        NINFER_GGUF_VECTOR_CASE(6)
        NINFER_GGUF_VECTOR_CASE(7)
        NINFER_GGUF_VECTOR_CASE(8)
#undef NINFER_GGUF_VECTOR_CASE
    default:
        throw std::logic_error("gguf vector pass: invalid column count");
    }
}

template <bool Tiled>
void launch_vector(const __nv_bfloat16* x, const GgufSegment& segment, int first_row, int k,
                   int tokens, int token_base, const Outputs& out, cudaStream_t stream) {
    switch (segment.type) {
#define NINFER_GGUF_TYPE_CASE(T)                                                              \
    case GgufType::T:                                                                         \
        launch_vector_type<GgufType::T, Tiled>(x, segment, first_row, k, tokens, token_base,  \
                                               out, stream);                                  \
        break;
        NINFER_GGUF_TYPE_CASE(IQ4_XS)
        NINFER_GGUF_TYPE_CASE(IQ3_S)
        NINFER_GGUF_TYPE_CASE(IQ3_XXS)
        NINFER_GGUF_TYPE_CASE(IQ2_XS)
        NINFER_GGUF_TYPE_CASE(IQ2_XXS)
        NINFER_GGUF_TYPE_CASE(IQ2_S)
        NINFER_GGUF_TYPE_CASE(IQ1_M)
        NINFER_GGUF_TYPE_CASE(Q2_K)
        NINFER_GGUF_TYPE_CASE(Q4_K)
        NINFER_GGUF_TYPE_CASE(Q6_K)
#undef NINFER_GGUF_TYPE_CASE
    }
    CUDA_CHECK(cudaGetLastError());
}

// Dequantizes rows [row_begin, row_begin + row_count) of one segment, one 8-value group per thread.
template <GgufType Type, typename Out>
__global__ void dequantize_kernel(const unsigned char* __restrict__ rows, int row_begin,
                                  int row_count, int k, Out* __restrict__ out) {
    const std::int64_t groups = static_cast<std::int64_t>(row_count) * (k / 8);
    const std::int64_t g      = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= groups) { return; }
    const int row             = static_cast<int>(g / (k / 8));
    const int group           = static_cast<int>(g % (k / 8));
    constexpr int kBlockBytes = gguf_block_bytes(Type);
    const unsigned char* block = rows +
        static_cast<std::int64_t>(row_begin + row) * (k / 256) * kBlockBytes +
        (group / 32) * kBlockBytes;
    float w[8];
    decode8<Type>(block, group % 32, w);
    Out* destination = out + static_cast<std::int64_t>(row) * k + group * 8;
    if constexpr (std::is_same_v<Out, float>) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { destination[j] = w[j]; }
    } else {
        __align__(16) __half h[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { h[j] = __float2half_rn(w[j]); }
        *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(h);
    }
}

template <typename Out>
void launch_dequantize(const GgufSegment& segment, int row_begin, int row_count, int k, Out* out,
                       cudaStream_t stream) {
    const std::int64_t groups = static_cast<std::int64_t>(row_count) * (k / 8);
    const unsigned blocks     = static_cast<unsigned>((groups + 255) / 256);
    const auto* rows          = static_cast<const unsigned char*>(segment.data);
    switch (segment.type) {
#define NINFER_GGUF_TYPE_CASE(T)                                                               \
    case GgufType::T:                                                                          \
        dequantize_kernel<GgufType::T, Out><<<blocks, 256, 0, stream>>>(rows, row_begin,        \
                                                                        row_count, k, out);    \
        break;
        NINFER_GGUF_TYPE_CASE(IQ4_XS)
        NINFER_GGUF_TYPE_CASE(IQ3_S)
        NINFER_GGUF_TYPE_CASE(IQ3_XXS)
        NINFER_GGUF_TYPE_CASE(IQ2_XS)
        NINFER_GGUF_TYPE_CASE(IQ2_XXS)
        NINFER_GGUF_TYPE_CASE(IQ2_S)
        NINFER_GGUF_TYPE_CASE(IQ1_M)
        NINFER_GGUF_TYPE_CASE(Q2_K)
        NINFER_GGUF_TYPE_CASE(Q4_K)
        NINFER_GGUF_TYPE_CASE(Q6_K)
#undef NINFER_GGUF_TYPE_CASE
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool Tiled>
__global__ void cast_activation_kernel(const __nv_bfloat16* __restrict__ x,
                                       __half* __restrict__ out, std::int64_t count, int k) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int column = static_cast<int>(i % k);
    out[i] = __float2half_rn(__bfloat162float(x[i - column + input_column<Tiled>(column, k)]));
}

using GemmOutput = cutlass::bfloat16_t;
using GemmEpilogue = cutlass::epilogue::thread::LinearCombination<
    GemmOutput, 128 / cutlass::sizeof_bits<GemmOutput>::value, float, float>;
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
    GemmOutput, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>, GemmEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
constexpr int kOutputAlignment = 128 / cutlass::sizeof_bits<GemmOutput>::value;

std::size_t gemm_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP16, {k, tokens});
    (void)layout.alloc(DType::FP16, {k, std::min(n, kGemmRowTile)});
    return layout.peak_bytes(1);
}

// The tensor-core route writes each output section directly, so every section must meet the
// epilogue's vector alignment. Segment boundaries inside a section are row offsets and need the
// same alignment.
bool gemm_admits(const Weight& weight, const Outputs& out) {
    for (int i = 0; i < out.count; ++i) {
        if (out.stride[i] % kOutputAlignment != 0 || out.rows[i] % kOutputAlignment != 0 ||
            reinterpret_cast<std::uintptr_t>(out.data[i]) % 16 != 0) {
            return false;
        }
    }
    for (int s = 0; s < weight.gguf_segment_count; ++s) {
        if (weight.gguf_segments[s].rows % kOutputAlignment != 0) { return false; }
    }
    return true;
}

template <bool Tiled>
void gemm_project(const __nv_bfloat16* x, const Weight& weight, const Outputs& out, int tokens,
                  WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int k    = weight.k;
    Tensor x_fp16  = workspace.alloc(DType::FP16, {k, tokens});
    Tensor w_fp16  = workspace.alloc(DType::FP16, {k, std::min(weight.n, kGemmRowTile)});
    const std::int64_t count = static_cast<std::int64_t>(k) * tokens;
    cast_activation_kernel<Tiled><<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        x, static_cast<__half*>(x_fp16.data), count, k);
    CUDA_CHECK(cudaGetLastError());

    Gemm gemm;
    int segment      = 0;
    int segment_row  = 0; // first logical row of `segment`
    int section      = 0;
    int section_row  = 0; // first logical row of `section`
    for (int row = 0; row < weight.n;) {
        while (row >= segment_row + weight.gguf_segments[segment].rows) {
            segment_row += weight.gguf_segments[segment++].rows;
        }
        while (row >= section_row + out.rows[section]) { section_row += out.rows[section++]; }
        const int end = std::min({row + kGemmRowTile, segment_row + weight.gguf_segments[segment].rows,
                                  section_row + out.rows[section]});
        const int rows = end - row;
        launch_dequantize(weight.gguf_segments[segment], row - segment_row, rows, k,
                          static_cast<__half*>(w_fp16.data), stream);
        auto* destination = reinterpret_cast<GemmOutput*>(out.data[section]) + (row - section_row);
        const int ld      = static_cast<int>(out.stride[section]);
        typename Gemm::Arguments arguments{
            {tokens, rows, k},
            {static_cast<const cutlass::half_t*>(x_fp16.data), k},
            {static_cast<const cutlass::half_t*>(w_fp16.data), k},
            {destination, ld},
            {destination, ld},
            {1.0F, out.add ? 1.0F : 0.0F},
            1};
        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess ||
            gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess ||
            gemm(stream) != cutlass::Status::kSuccess) {
            throw std::runtime_error("gguf tensor-core projection: CUTLASS GEMM failed");
        }
        row = end;
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool Tiled>
void vector_project(const __nv_bfloat16* x, const Weight& weight, const Outputs& out, int tokens,
                    cudaStream_t stream) {
    for (int token = 0; token < tokens; token += kGgufVectorTokens) {
        const int width   = std::min(kGgufVectorTokens, tokens - token);
        const auto* input = x + static_cast<std::int64_t>(token) * weight.k;
        int first_row     = 0;
        for (int s = 0; s < weight.gguf_segment_count; ++s) {
            launch_vector<Tiled>(input, weight.gguf_segments[s], first_row, weight.k, width, token,
                                 out, stream);
            first_row += weight.gguf_segments[s].rows;
        }
    }
}

template <bool Tiled>
void project(const __nv_bfloat16* x, const Weight& weight, const Outputs& out, int tokens,
             WorkspaceArena* workspace, cudaStream_t stream) {
    if (tokens > kVectorMaxTokens && workspace != nullptr && gemm_admits(weight, out) &&
        workspace->capacity() - workspace->used() >=
            gemm_workspace_bytes(weight.n, weight.k, tokens)) {
        gemm_project<Tiled>(x, weight, out, tokens, *workspace, stream);
        return;
    }
    vector_project<Tiled>(x, weight, out, tokens, stream);
}

__global__ void embedding_kernel(const std::int32_t* __restrict__ ids, Weight weight,
                                 __nv_bfloat16* __restrict__ out) {
    const int token = blockIdx.x;
    int row         = ids[token];
    int segment     = 0;
    while (segment + 1 < weight.gguf_segment_count && row >= weight.gguf_segments[segment].rows) {
        row -= weight.gguf_segments[segment++].rows;
    }
    const GgufSegment& s      = weight.gguf_segments[segment];
    const int block_bytes     = gguf_block_bytes(s.type);
    const unsigned char* base = static_cast<const unsigned char*>(s.data) +
                                static_cast<std::int64_t>(row) * (weight.k / 256) * block_bytes;
    for (int group = threadIdx.x; group < weight.k / 8; group += blockDim.x) {
        float w[8];
        decode8(s.type, base + (group / 32) * block_bytes, group % 32, w);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[static_cast<std::int64_t>(token) * weight.k + group * 8 + j] =
                __float2bfloat16_rn(w[j]);
        }
    }
}

Outputs make_outputs(const Weight& weight, const Tensor& x, const Tensor* outputs, int count,
                     bool add) {
    if (count < 1 || count > 4) {
        throw std::invalid_argument("gguf projection: one to four output sections are required");
    }
    Outputs out;
    out.count = count;
    out.add   = add;
    std::int32_t total = 0;
    for (int i = 0; i < count; ++i) {
        const Tensor& o = outputs[i];
        if (o.dtype != DType::BF16 || o.data == nullptr || o.ne[0] <= 0 || o.ne[1] != x.ne[1] ||
            o.ne[2] != 1 || o.ne[3] != 1 || o.nb[0] != 2 || o.nb[1] % 2 != 0) {
            throw std::invalid_argument("gguf projection: invalid output section");
        }
        out.data[i]   = static_cast<__nv_bfloat16*>(o.data);
        out.rows[i]   = o.ne[0];
        out.stride[i] = o.nb[1] / 2;
        total += o.ne[0];
    }
    if (total != weight.n) {
        throw std::invalid_argument("gguf projection: output sections do not cover the weight rows");
    }
    return out;
}

} // namespace

void validate_gguf_weight(const Weight& weight, const char* label) {
    if (weight.qtype != QType::GGUF || weight.layout != QuantLayout::GgufBlocks || weight.n <= 0 ||
        weight.k <= 0 || weight.k % 256 != 0 || weight.gguf_segment_count < 1 ||
        weight.gguf_segment_count > kMaxGgufSegments) {
        throw std::invalid_argument(std::string(label) + ": invalid GGUF-blocks weight");
    }
    std::int32_t rows = 0;
    for (int s = 0; s < weight.gguf_segment_count; ++s) {
        const GgufSegment& segment = weight.gguf_segments[s];
        if (segment.data == nullptr || segment.rows <= 0 ||
            reinterpret_cast<std::uintptr_t>(segment.data) % 2 != 0) {
            throw std::invalid_argument(std::string(label) + ": invalid GGUF segment");
        }
        rows += segment.rows;
    }
    if (rows != weight.n) {
        throw std::invalid_argument(std::string(label) + ": GGUF segments do not cover N rows");
    }
}

void gguf_project(const Tensor& x, const Weight& weight, const Tensor* outputs, int count, bool add,
                  bool tiled_gdn_input, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_gguf_weight(weight, "gguf projection");
    if (x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != weight.k || x.ne[1] <= 0 ||
        x.ne[2] != 1 || x.ne[3] != 1 || reinterpret_cast<std::uintptr_t>(x.data) % 16 != 0) {
        throw std::invalid_argument("gguf projection: x must be contiguous 16-byte aligned BF16 [K,T]");
    }
    if (tiled_gdn_input && weight.k != 6144 && weight.k != 3072) {
        throw std::invalid_argument("gguf GDN output projection requires K=6144 or TP2 K=3072");
    }
    const Outputs out = make_outputs(weight, x, outputs, count, add);
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    if (tiled_gdn_input) {
        project<true>(input, weight, out, x.ne[1], workspace, stream);
    } else {
        project<false>(input, weight, out, x.ne[1], workspace, stream);
    }
}

std::size_t gguf_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t max_tokens) {
    if (n <= 0 || k <= 0 || k % 256 != 0 || max_tokens <= 0) {
        throw std::invalid_argument("gguf workspace: invalid profile");
    }
    return max_tokens > kVectorMaxTokens ? gemm_workspace_bytes(n, k, max_tokens) : 0;
}

void gguf_embedding(const Tensor& ids, const Weight& weight, Tensor& out, cudaStream_t stream) {
    validate_gguf_weight(weight, "gguf embedding");
    embedding_kernel<<<static_cast<unsigned>(ids.numel()), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), weight, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void gguf_dequantize_fp32(const Weight& weight, float* out, cudaStream_t stream) {
    validate_gguf_weight(weight, "gguf dequantize");
    std::int64_t row = 0;
    for (int s = 0; s < weight.gguf_segment_count; ++s) {
        const GgufSegment& segment = weight.gguf_segments[s];
        launch_dequantize(segment, 0, segment.rows, weight.k, out + row * weight.k, stream);
        row += segment.rows;
    }
}

Weight gguf_row_view(const Weight& weight, std::int32_t row_begin, std::int32_t row_count) {
    validate_gguf_weight(weight, "gguf row view");
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > weight.n) {
        throw std::invalid_argument("gguf row view: rows are outside the weight");
    }
    Weight out               = weight;
    out.gguf_segment_count   = 0;
    std::int32_t first       = 0;
    const std::int32_t end   = row_begin + row_count;
    for (int s = 0; s < weight.gguf_segment_count; ++s) {
        const GgufSegment& segment = weight.gguf_segments[s];
        const std::int32_t lo      = std::max(first, row_begin);
        const std::int32_t hi      = std::min(first + segment.rows, end);
        if (lo < hi) {
            GgufSegment& piece = out.gguf_segments[out.gguf_segment_count++];
            piece.type         = segment.type;
            piece.rows         = hi - lo;
            piece.data         = static_cast<const unsigned char*>(segment.data) +
                         static_cast<std::int64_t>(lo - first) * row_bytes(segment.type, weight.k);
        }
        first += segment.rows;
    }
    out.payload = out.qdata = out.gguf_segments[0].data;
    out.n = out.shape[0] = out.padded_shape[0] = row_count;
    return out;
}

} // namespace ninfer::ops::detail
