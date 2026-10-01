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
#include <optional>
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

// A warp owns Rows consecutive rows; lane j decodes values [8j, 8j+8) of every 256-value block.
// The block loop is outermost, so each lane loads its activation slice of a block once and reuses
// it for all Rows rows. Weights move in stages of kStageBlocks blocks per row: the warp loads the
// stage's bytes with coalesced 32-bit loads into registers, publishes them to its shared-memory
// stage, then prefetches the next stage while decoding the current one. Every registered row length
// is a multiple of 4 bytes (K/256 is even for every Qwen3.8 shape and every block size is even),
// and kStageBlocks = 2 keeps each stage 4-byte aligned; gguf_project rejects any other geometry.
constexpr int kVectorWarps = 4;
constexpr int kStageBlocks = 2;

// Activation operand of one lane for one 256-value block and one column: its eight values, either
// as BF16-exact floats (A16) or as int8 codes with the FP32 step of their 32-value group (A8; the
// four lanes of a group share the step). The A8 represented activation is step * code, so the
// minimum term of K-quant and IQ1_M weights multiplies the sum of the represented values,
// step * sum(codes), and every format computes exactly sum(weight * represented activation).
template <bool A8>
struct LaneActivation;

template <>
struct LaneActivation<false> {
    float v[8];
};

template <>
struct LaneActivation<true> {
    int q[2];
    float step;
    float sum;
};

__device__ __forceinline__ void load_activation(const __nv_bfloat16* p, LaneActivation<false>& a) {
    load_bf16x8(p, a.v);
}

// The A8 operand of one lane: eight int8 codes at stored column `column` and their group's step.
__device__ __forceinline__ void load_activation(const signed char* codes, const float* steps,
                                                int column, LaneActivation<true>& a) {
    const uint2 packed = __ldg(reinterpret_cast<const uint2*>(codes + column));
    a.q[0] = static_cast<int>(packed.x);
    a.q[1] = static_cast<int>(packed.y);
    a.step = __ldg(steps + column / 32);
    a.sum  = a.step * static_cast<float>(__dp4a(a.q[1], 0x01010101, __dp4a(a.q[0], 0x01010101, 0)));
}

// Quantizes BF16 x [K,T] once per call for the A8 vector route: each 32-value group of stored
// columns (read through the GDN column map when Tiled) becomes int8 codes round(v / step) with
// step = amax / 127. One thread converts eight values; the four threads of a group are adjacent
// lanes. Every thread takes part in the shuffles, so the grid need not divide the work evenly.
template <bool Tiled>
__global__ void quantize_activation_kernel(const __nv_bfloat16* __restrict__ x, int k, int tokens,
                                           signed char* __restrict__ codes,
                                           float* __restrict__ steps) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(tokens) * (k / 8);
    const bool live          = index < total;
    const int token          = live ? static_cast<int>(index / (k / 8)) : 0;
    const int column         = live ? static_cast<int>(index % (k / 8)) * 8 : 0;
    float v[8];
    load_bf16x8(x + static_cast<std::int64_t>(token) * k + input_column<Tiled>(column, k), v);
    float amax = 0.0F;
#pragma unroll
    for (int j = 0; j < 8; ++j) { amax = fmaxf(amax, fabsf(v[j])); }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, 2));
    if (!live) { return; }
    const float inverse = amax > 0.0F ? 127.0F / amax : 0.0F;
    unsigned packed[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int code = __float2int_rn(v[j] * inverse);
        packed[j / 4] |= (static_cast<unsigned>(code) & 255U) << (8 * (j % 4));
    }
    *reinterpret_cast<uint2*>(codes + static_cast<std::int64_t>(token) * k + column) =
        make_uint2(packed[0], packed[1]);
    if ((column & 31) == 0) { steps[static_cast<std::int64_t>(token) * (k / 32) + column / 32] = amax / 127.0F; }
}

// A8 form needs the decoded codes per column, so the row loop decodes once and reuses them.
struct LaneWeightI8 {
    int q[2];
    float scale;
    float minimum;
};

template <GgufType Type>
__device__ __forceinline__ LaneWeightI8 decode_lane_i8(const unsigned char* block, int lane) {
    LaneWeightI8 w;
    w.minimum = 0.0F;
    gguf::decode_i8<Type>(block, lane, w.q, w.scale, w.minimum);
    return w;
}

__device__ __forceinline__ float lane_dot_i8(const LaneWeightI8& w, const LaneActivation<true>& a,
                                             float acc) {
    const int dot = __dp4a(w.q[1], a.q[1], __dp4a(w.q[0], a.q[0], 0));
    acc = fmaf(w.scale * a.step, static_cast<float>(dot), acc);
    return fmaf(-w.minimum, a.sum, acc);
}

// A warp owns Rows consecutive rows; lane j covers values [8j, 8j+8) of every 256-value block.
// The block loop is outermost, so each lane prepares its activation slice of a block once and
// reuses it for all Rows rows. Weights move in stages of kStageBlocks blocks per row: the warp
// loads a stage's bytes with coalesced 32-bit loads into registers, publishes them to its
// shared-memory stage, then prefetches the next stage while decoding the current one. Every
// registered row length is a multiple of 4 bytes (K/256 is even for every Qwen3.8 shape and every
// block size is even), and kStageBlocks = 2 keeps each stage 4-byte aligned; vector_project
// rejects any other geometry.
//
// A8 reads the call's int8 activation codes (quantize_activation_kernel, stored column order) and
// dots integer weight codes with dp4a; A16 multiplies decoded FP32 weights by the exact BF16
// activation.
template <GgufType Type, int MaxTokens, int Rows, bool Tiled, bool A8>
__global__ void __launch_bounds__(kVectorWarps * 32)
vector_kernel(const __nv_bfloat16* __restrict__ x, const signed char* __restrict__ codes,
              const float* __restrict__ steps, const unsigned char* __restrict__ rows,
              int row_count, int first_row, int k, int tokens, int token_base, Outputs out) {
    constexpr int kBlockBytes = gguf_block_bytes(Type);
    constexpr int kRowWords   = kStageBlocks * kBlockBytes / 4; // 32-bit words per row per stage
    constexpr int kStageWords = Rows * kRowWords;
    constexpr int kLaneWords  = (kStageWords + 31) / 32;
    __shared__ __align__(16) unsigned stage[kVectorWarps][kStageWords];

    const int warp  = static_cast<int>(threadIdx.x >> 5);
    const int lane  = static_cast<int>(threadIdx.x & 31);
    const int first = (blockIdx.x * kVectorWarps + warp) * Rows;
    if (first >= row_count) { return; }
    const int live_rows           = min(Rows, row_count - first);
    const int blocks              = k / 256;
    const std::int64_t row_stride = static_cast<std::int64_t>(blocks) * kBlockBytes / 4;
    const auto* base = reinterpret_cast<const unsigned*>(rows) + first * row_stride;

    unsigned prefetch[kLaneWords];
    const auto load_stage = [&](int stage_block) {
#pragma unroll
        for (int i = 0; i < kLaneWords; ++i) {
            const int word = lane + 32 * i;
            const int r    = word / kRowWords;
            const int w    = word - r * kRowWords;
            prefetch[i]    = word < kStageWords && r < live_rows
                                 ? __ldg(base + r * row_stride + stage_block * kBlockBytes / 4 + w)
                                 : 0U;
        }
    };

    float sum[Rows][MaxTokens] = {};
    const auto* staged = reinterpret_cast<const unsigned char*>(stage[warp]);
    load_stage(0);
    for (int stage_block = 0; stage_block < blocks; stage_block += kStageBlocks) {
#pragma unroll
        for (int i = 0; i < kLaneWords; ++i) {
            const int word = lane + 32 * i;
            if (word < kStageWords) { stage[warp][word] = prefetch[i]; }
        }
        __syncwarp();
        if (stage_block + kStageBlocks < blocks) { load_stage(stage_block + kStageBlocks); }
#pragma unroll
        for (int b = 0; b < kStageBlocks; ++b) {
            const int stored = (stage_block + b) * 256 + lane * 8;
            LaneActivation<A8> a[MaxTokens];
#pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                // Padding columns load column 0 (always valid); their sums are never stored.
                const std::int64_t source = t < tokens ? t : 0;
                if constexpr (A8) {
                    load_activation(codes + source * k, steps + source * (k / 32), stored, a[t]);
                } else {
                    load_activation(x + source * k + input_column<Tiled>(stored, k), a[t]);
                }
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const unsigned char* block = staged + r * kRowWords * 4 + b * kBlockBytes;
                if constexpr (A8) {
                    const LaneWeightI8 w = decode_lane_i8<Type>(block, lane);
#pragma unroll
                    for (int t = 0; t < MaxTokens; ++t) { sum[r][t] = lane_dot_i8(w, a[t], sum[r][t]); }
                } else {
                    float w[8];
                    decode8<Type>(block, lane, w);
#pragma unroll
                    for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) { sum[r][t] = fmaf(w[j], a[t].v[j], sum[r][t]); }
                    }
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                sum[r][t] += __shfl_xor_sync(0xffffffffU, sum[r][t], delta);
            }
        }
    }
    if (lane < live_rows) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == lane) {
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (t < tokens) { out.store(first_row + first + r, token_base + t, sum[r][t]); }
                }
            }
        }
    }
}

struct VectorInput {
    const __nv_bfloat16* x  = nullptr; // A16
    const signed char* codes = nullptr; // A8
    const float* steps       = nullptr; // A8
};

template <GgufType Type, int MaxTokens, int Rows, bool Tiled, bool A8>
void launch_vector_shape(const VectorInput& in, const GgufSegment& segment, int first_row, int k,
                         int tokens, int token_base, const Outputs& out, cudaStream_t stream) {
    constexpr int kRowsPerCta = kVectorWarps * Rows;
    const dim3 grid(static_cast<unsigned>((segment.rows + kRowsPerCta - 1) / kRowsPerCta));
    vector_kernel<Type, MaxTokens, Rows, Tiled, A8><<<grid, kVectorWarps * 32, 0, stream>>>(
        in.x, in.codes, in.steps, static_cast<const unsigned char*>(segment.data), segment.rows,
        first_row, k, tokens, token_base, out);
}

// Small segments keep one row per warp so the grid still covers every SM; large ones share each
// activation slice across several rows. Wider passes trade rows for activation registers.
template <GgufType Type, bool Tiled, bool A8>
void launch_vector_type(const VectorInput& x, const GgufSegment& segment, int first_row, int k,
                        int tokens, int token_base, const Outputs& out, cudaStream_t stream) {
    const bool wide = segment.rows >= 2048;
#define NINFER_GGUF_SHAPE(T, R)                                                                 \
    launch_vector_shape<Type, T, R, Tiled, A8>(x, segment, first_row, k, tokens, token_base, out, \
                                               stream)
    if (tokens == 1) {
        if (wide) { NINFER_GGUF_SHAPE(1, 4); } else { NINFER_GGUF_SHAPE(1, 1); }
    } else if (tokens <= 4) {
        if (wide) { NINFER_GGUF_SHAPE(4, 4); } else { NINFER_GGUF_SHAPE(4, 1); }
    } else if (tokens <= kGgufVectorTokens) {
        if (wide) { NINFER_GGUF_SHAPE(8, 2); } else { NINFER_GGUF_SHAPE(8, 1); }
    } else {
        throw std::logic_error("gguf vector pass: invalid column count");
    }
#undef NINFER_GGUF_SHAPE
}

template <bool Tiled, bool A8>
void launch_vector(const VectorInput& x, const GgufSegment& segment, int first_row, int k,
                   int tokens, int token_base, const Outputs& out, cudaStream_t stream) {
    switch (segment.type) {
#define NINFER_GGUF_TYPE_CASE(T)                                                                 \
    case GgufType::T:                                                                            \
        launch_vector_type<GgufType::T, Tiled, A8>(x, segment, first_row, k, tokens, token_base, \
                                                   out, stream);                                 \
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

std::size_t a8_bytes(std::int32_t k, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::I8, {k, tokens});
    (void)layout.alloc(DType::FP32, {k / 32, tokens});
    return layout.peak_bytes(1);
}

template <bool Tiled, bool A8>
void vector_project(const __nv_bfloat16* x, const Weight& weight, const Outputs& out, int tokens,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    if ((weight.k / 256) % kStageBlocks != 0) {
        throw std::invalid_argument("gguf vector projection requires K to be a multiple of 512");
    }
    for (int s = 0; s < weight.gguf_segment_count; ++s) {
        if (reinterpret_cast<std::uintptr_t>(weight.gguf_segments[s].data) % 4 != 0) {
            throw std::invalid_argument("gguf vector projection requires 4-byte aligned segments");
        }
    }
    const int k = weight.k;
    std::optional<WorkspaceArena::Scope> scope;
    Tensor codes;
    Tensor steps;
    if constexpr (A8) {
        const int width = std::min(kGgufVectorTokens, tokens);
        scope.emplace(workspace->scope());
        codes = workspace->alloc(DType::I8, {k, width});
        steps = workspace->alloc(DType::FP32, {k / 32, width});
    }
    for (int token = 0; token < tokens; token += kGgufVectorTokens) {
        const int width   = std::min(kGgufVectorTokens, tokens - token);
        const auto* input = x + static_cast<std::int64_t>(token) * k;
        VectorInput in;
        if constexpr (A8) {
            const std::int64_t threads = static_cast<std::int64_t>(width) * (k / 8);
            quantize_activation_kernel<Tiled><<<static_cast<unsigned>((threads + 127) / 128), 128,
                                                0, stream>>>(
                input, k, width, static_cast<signed char*>(codes.data),
                static_cast<float*>(steps.data));
            CUDA_CHECK(cudaGetLastError());
            in.codes = static_cast<const signed char*>(codes.data);
            in.steps = static_cast<const float*>(steps.data);
        } else {
            in.x = input;
        }
        int first_row = 0;
        for (int s = 0; s < weight.gguf_segment_count; ++s) {
            launch_vector<Tiled, A8>(in, weight.gguf_segments[s], first_row, k, width, token, out,
                                     stream);
            first_row += weight.gguf_segments[s].rows;
        }
    }
}

// Routes: the FP16 tensor-core GEMM above kVectorMaxTokens columns when its workspace fits;
// otherwise vector passes, A8 when admitted and its activation buffer fits, else A16.
template <bool Tiled>
void project(const __nv_bfloat16* x, const Weight& weight, const Outputs& out, int tokens,
             bool allow_a8, WorkspaceArena* workspace, cudaStream_t stream) {
    const std::size_t available =
        workspace == nullptr ? 0 : workspace->capacity() - workspace->used();
    if (tokens > kVectorMaxTokens && workspace != nullptr && gemm_admits(weight, out) &&
        available >= gemm_workspace_bytes(weight.n, weight.k, tokens)) {
        gemm_project<Tiled>(x, weight, out, tokens, *workspace, stream);
        return;
    }
    if (allow_a8 && workspace != nullptr &&
        available >= a8_bytes(weight.k, std::min(kGgufVectorTokens, tokens))) {
        vector_project<Tiled, true>(x, weight, out, tokens, workspace, stream);
    } else {
        vector_project<Tiled, false>(x, weight, out, tokens, workspace, stream);
    }
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
                  bool tiled_gdn_input, bool allow_a8, WorkspaceArena* workspace,
                  cudaStream_t stream) {
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
        project<true>(input, weight, out, x.ne[1], allow_a8, workspace, stream);
    } else {
        project<false>(input, weight, out, x.ne[1], allow_a8, workspace, stream);
    }
}

std::size_t gguf_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t max_tokens) {
    if (n <= 0 || k <= 0 || k % 256 != 0 || max_tokens <= 0) {
        throw std::invalid_argument("gguf workspace: invalid profile");
    }
    const std::size_t vector = a8_bytes(k, std::min(kGgufVectorTokens, max_tokens));
    return max_tokens > kVectorMaxTokens
               ? std::max(vector, gemm_workspace_bytes(n, k, max_tokens))
               : vector;
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
