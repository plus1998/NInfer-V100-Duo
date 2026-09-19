// Two-device head-local parity suite for the full-attention core under two-way tensor
// parallelism. Sibling of test_linear_split.cpp / test_attn_input_proj_split.cpp;
// read test_linear_split.cpp's header comment first, this one documents what differs.
//
// WHAT IS SPLIT, AND WHY THERE IS NO COLLECTIVE. Attention over grouped-query heads is
// separable in the head index: query head qh reads only KV head qh/6, and its softmax is taken
// over that KV head's keys alone. Splitting 24|4 into two 12|2 halves therefore partitions the
// computation exactly -- device r owns global Q heads [12r,12r+12) and global KV heads
// [2r,2r+2), computes their softmax entirely from its own KV pool, and never needs a byte from
// its peer. GroupSize stays 6 on both sides, so the grouping arithmetic is unchanged; the split
// is a geometry instantiation (Gqa27Tp2Geometry = GqaGeometry<12,2,2>), not a new algorithm.
//
// LOCAL HEAD INDEXING IS THE THING THIS SUITE EXISTS TO PROVE. Each device's KV pool stores ONLY
// its own two head pairs, so the head extent of every cache plane is 2 and every head index the
// kernels compute -- the pool stride `paged_kv_element_offset<256, Geometry::KVHeads>` and the
// grouping `qh / GroupSize` -- is LOCAL (kv_head in [0,2), q_head in [0,12)). Nothing renumbers
// heads; the mapping global -> local is applied once, by the caller, when it builds the
// per-device q/k/v and pool. A wrong mapping (wrong KV head pair, wrong query-head block, or a
// swapped Q-head -> KV-head grouping inside a device) still produces perfectly well-formed
// output, so it can only be caught by comparing PER HEAD against a reference that knows the
// global numbering. That is what run_case does, plus an explicit anti-permutation leg: every
// observed head block is required to be FAR from every reference head block except its own.
//
// REFERENCE. One single-device tp1 run of the same Op on the same logical data (24 Q heads, a
// 4-head pool) provides the 24-head output; each device's 12-head output is compared against the
// slice of it that names the same global heads. Where the FP64 ideal oracle is cheap to evaluate
// (single-request cases) the head-local output is ALSO compared straight to it, which is the
// Op's own registered numerical contract applied to the new geometry.
//
// TOLERANCE. Both legs use the criterion the GQA conformance suite already owns for the cache
// dtype in question -- kAttentionBf16Criterion / kAttentionInt8Criterion, verbatim from
// gqa_attention_fixture.h. Nothing is loosened for the split: the oracle leg is literally the
// registered contract, and the parity leg holds the split output no further from the tp1 output
// than the contract allows either of them to sit from the oracle. Parity is NOT bit-exact and
// cannot be: the split-KV decode route stores its per-split partial accumulators in BF16, and
// the tp2 geometry's DecodeSplitScale of 2 (which keeps each device's 170-SM board full at half
// the KV heads) makes the split count, and hence the partition of the reduction, differ from
// tp1's. The observed numbers are printed by NINFER_OP_REPORT_STATS=1.
//
// Every case REQUIRES two CUDA devices driven from ONE process; with fewer than two visible
// devices the suite reports the repository's skip code (77) instead of failing.
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/gqa_attention_fixture.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

using namespace ninfer::test::gqa;

constexpr int kRanks = 2;

// The real qwen3_6_27b full-attention geometry and its head-local half. 24/2 = 12 Q heads and
// 4/2 = 2 KV heads per device; 12/2 = 6 is the same GQA ratio the 24|4 parent has, which is why
// no grouping arithmetic changes. (An early design note wrote the per-device geometry as "6:2";
// that is the shared group size 6, not a head count -- the per-device head counts are 12 and 2.)
constexpr Geometry kGlobalGeometry{"qwen3_6_27b", 24, 4};
constexpr Geometry kLocalGeometry{"qwen3_6_27b_tp2", 12, 2};

static_assert(kGlobalGeometry.q_heads == kRanks * kLocalGeometry.q_heads);
static_assert(kGlobalGeometry.kv_heads == kRanks * kLocalGeometry.kv_heads);

std::int32_t local_q_head_begin(int rank) { return rank * kLocalGeometry.q_heads; }
std::int32_t local_kv_head_begin(int rank) { return rank * kLocalGeometry.kv_heads; }

void set_device(int device) { cuda_check(cudaSetDevice(device), "cudaSetDevice"); }

struct CurrentDeviceScope {
    int previous = 0;
    explicit CurrentDeviceScope(int device) {
        cuda_check(cudaGetDevice(&previous), "cudaGetDevice");
        set_device(device);
    }
    CurrentDeviceScope(const CurrentDeviceScope&)            = delete;
    CurrentDeviceScope& operator=(const CurrentDeviceScope&) = delete;
    ~CurrentDeviceScope() { (void)cudaSetDevice(previous); }
};

// --- global -> local slicing -------------------------------------------------------------
//
// Both slices below are pure re-indexings of the SAME generated logical data, so the per-device
// inputs and the tp1 inputs meet exactly at the logical level -- the relationship the tp2 loader
// and the split input projection actually produce at run time: `attn_input_proj_column_parallel`
// hands each device its own contiguous, ALREADY HEAD-LOCAL q/gate [3072,T] and k/v [512,T]
// (device 0 = query heads [0,12) and KV heads [0,2), device 1 = [12,24) and [2,4)), never a
// packed parent output, and with no cross-device traffic before or during attention.

// q/k/v are token-major: index = d + 256*(head + heads*token). One device's block is a per-token
// gather of a contiguous head window.
std::vector<float> slice_heads_token_major(const std::vector<float>& source,
                                           std::int32_t source_heads, std::int32_t head_begin,
                                           std::int32_t head_count, std::int32_t tokens) {
    std::vector<float> result(static_cast<std::size_t>(kHeadDim) * head_count * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t head = 0; head < head_count; ++head) {
            const std::size_t src = static_cast<std::size_t>(kHeadDim) *
                                    (static_cast<std::size_t>(head_begin + head) +
                                     static_cast<std::size_t>(source_heads) * token);
            const std::size_t dst =
                static_cast<std::size_t>(kHeadDim) *
                (static_cast<std::size_t>(head) + static_cast<std::size_t>(head_count) * token);
            std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(src), kHeadDim,
                        result.begin() + static_cast<std::ptrdiff_t>(dst));
        }
    }
    return result;
}

// Cache planes are head-major: index = leading + leading_extent*(position + capacity*head). One
// device's pool is therefore a CONTIGUOUS slab of the global pool's logical image -- no
// re-indexing at all, which is what makes "device r's pool holds only heads [2r,2r+2)" an exact
// statement rather than an approximation.
template <typename T>
std::vector<T> slice_head_major(const std::vector<T>& source, std::int32_t leading_extent,
                                std::int32_t capacity, std::int32_t head_begin,
                                std::int32_t head_count) {
    const std::size_t slab = static_cast<std::size_t>(leading_extent) * capacity;
    std::vector<T> result(slab * head_count);
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(slab * head_begin), result.size(),
                result.begin());
    return result;
}

HostCache slice_cache_heads(const HostCache& source, int rank) {
    const std::int32_t begin = local_kv_head_begin(rank);
    const std::int32_t count = kLocalGeometry.kv_heads;
    HostCache result{kLocalGeometry, source.dtype, source.max_context, source.logical_capacity};
    if (source.dtype == DType::BF16) {
        result.k_bf16 =
            slice_head_major(source.k_bf16, kHeadDim, source.logical_capacity, begin, count);
        result.v_bf16 =
            slice_head_major(source.v_bf16, kHeadDim, source.logical_capacity, begin, count);
        return result;
    }
    result.k_i8  = slice_head_major(source.k_i8, kHeadDim, source.logical_capacity, begin, count);
    result.v_i8  = slice_head_major(source.v_i8, kHeadDim, source.logical_capacity, begin, count);
    result.k_scale =
        slice_head_major(source.k_scale, kQuantGroups, source.logical_capacity, begin, count);
    result.v_scale =
        slice_head_major(source.v_scale, kQuantGroups, source.logical_capacity, begin, count);
    return result;
}

// --- per-head comparison ------------------------------------------------------------------

std::vector<double> head_block(const std::vector<double>& values, const Geometry& geometry,
                               std::int32_t head, std::int32_t tokens, std::int32_t columns) {
    std::vector<double> block(static_cast<std::size_t>(kHeadDim) * tokens * columns);
    std::size_t cursor = 0;
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t token = 0; token < tokens; ++token) {
            const std::size_t base =
                q_index(geometry, head, 0, token + column * tokens);
            for (std::int32_t d = 0; d < kHeadDim; ++d) { block[cursor++] = values[base + d]; }
        }
    }
    return block;
}

double maximum_absolute_difference(const std::vector<double>& a, const std::vector<double>& b) {
    double maximum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { maximum = std::max(maximum, std::abs(a[i] - b[i])); }
    return maximum;
}

double maximum_absolute(const std::vector<double>& values) {
    double maximum = 0.0;
    for (const double value : values) { maximum = std::max(maximum, std::abs(value)); }
    return maximum;
}

// A head permutation must not be able to hide inside the tolerance.
//
// Two global heads are only DISTINGUISHABLE on a given input if the reference itself separates
// them by much more than the criterion allows; that separation is a property of the data, not of
// the kernel. It fails legitimately in exactly one place: when a single key is visible, softmax
// over one key is 1 and every query head in a KV group returns that head's V verbatim, so the six
// heads of a group are mathematically identical. The check therefore gates on the reference's own
// separation, and asserts SEPARATELY that heads belonging to DIFFERENT KV heads are always
// separated -- that leg is the KV-head-mapping check and no input may make it vacuous.
constexpr double kPermutationMargin = 20.0;

int verify_head_mapping(const std::string& label, const std::vector<double>& observed,
                        const std::vector<double>& reference, const Geometry& reference_geometry,
                        std::int32_t correct_head, const ReductionCriterion& criterion,
                        std::int32_t tokens, std::int32_t columns) {
    int failures = 0;
    const std::vector<double> correct =
        head_block(reference, reference_geometry, correct_head, tokens, columns);
    const std::int32_t group = reference_geometry.query_group();
    for (std::int32_t head = 0; head < reference_geometry.q_heads; ++head) {
        if (head == correct_head) { continue; }
        const std::vector<double> other =
            head_block(reference, reference_geometry, head, tokens, columns);
        const double limit =
            std::max(criterion.gross_absolute,
                     criterion.gross_relative_to_max_reference * maximum_absolute(other));
        const double reference_separation = maximum_absolute_difference(correct, other);
        const bool different_kv_head      = (head / group) != (correct_head / group);
        if (reference_separation <= kPermutationMargin * limit) {
            if (different_kv_head) {
                std::cerr << label << ": reference heads " << correct_head << " and " << head
                          << " read DIFFERENT KV heads yet are indistinguishable (separation="
                          << reference_separation << "), so KV-head mapping is untested here\n";
                ++failures;
            }
            continue; // same KV head, too few visible keys to separate query heads: not a defect
        }
        // The observed block must stay far closer to its own reference head than to this one.
        const double distance = maximum_absolute_difference(observed, other);
        if (!(distance > 0.5 * reference_separation)) {
            std::cerr << label << ": head block is closer to global head " << head
                      << " than a correct mapping allows (max_abs_diff=" << distance
                      << " needs > " << 0.5 * reference_separation << ")\n";
            ++failures;
        }
    }
    return failures;
}

// --- device-side run ----------------------------------------------------------------------

struct RunResult {
    std::vector<std::uint16_t> output;
    HostCache cache_after;
    int failures = 0;
};

// One A1 (append + attend) invocation on `device` with `geometry`-shaped inputs.
RunResult run_a1(int device, const Geometry& geometry, DType dtype, const HostCache& initial,
                 MappingPattern mapping, const std::vector<float>& q, const std::vector<float>& k,
                 const std::vector<float>& v, const std::vector<std::int32_t>& positions,
                 ops::GqaExecutionEnvelope envelope, const std::string& label) {
    const CurrentDeviceScope scope(device);
    const auto tokens = static_cast<std::int32_t>(positions.size());
    DeviceCache cache(initial, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    const std::vector<std::uint16_t> canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dp.data(), DType::I32, {tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});

    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, tokens, tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kAttentionScale, cache.batch_view(),
                       envelope, workspace, tout, nullptr);
    cuda_synchronize();

    RunResult result;
    result.output      = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    result.cache_after = cache.snapshot();
    result.failures += verify_input(label + " q unchanged", dq, q_bits);
    result.failures += verify_input(label + " k unchanged", dk, k_bits);
    result.failures += verify_input(label + " v unchanged", dv, v_bits);
    result.failures += verify_positions(label + " positions unchanged", dp, positions);
    result.failures += dout.verify_guards((label + " output").c_str());
    result.failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++result.failures;
    }
    result.failures += cache.verify_guards(label);
    return result;
}

// One A3 (attend over an already populated pool, no cache write) invocation.
RunResult run_a3(int device, const Geometry& geometry, DType dtype, const HostCache& initial,
                 MappingPattern mapping, const std::vector<float>& q,
                 const std::vector<std::int32_t>& positions, ops::GqaExecutionEnvelope envelope,
                 const std::string& label) {
    const CurrentDeviceScope scope(device);
    const auto tokens = static_cast<std::int32_t>(positions.size());
    DeviceCache cache(initial, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::vector<std::uint16_t> canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});
    Tensor tp(dp.data(), DType::I32, {tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});

    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, tokens, tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                              nullptr);
    cuda_synchronize();

    RunResult result;
    result.output      = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    result.cache_after = cache.snapshot();
    result.failures += verify_input(label + " q unchanged", dq, q_bits);
    result.failures += verify_positions(label + " positions unchanged", dp, positions);
    result.failures += dout.verify_guards((label + " output").c_str());
    result.failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    result.failures += cache.verify_guards(label);
    return result;
}

// --- cases ---------------------------------------------------------------------------------

struct HeadLocalCase {
    const char* entry; // "gqa_attention" (A1) or "gqa_attention_cached" (A3)
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    MappingPattern mapping;
    std::uint32_t seed;
    bool check_head_mapping = true;
};

std::string case_label(const HeadLocalCase& test_case, DType dtype) {
    return std::string(test_case.entry) + " " + cache_name(dtype) +
           " mapping=" + mapping_name(test_case.mapping) +
           " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max);
}

// Re-indexes a global 24-head tensor into the 12-head layout of one device: the [D,H,W,B]
// element for global head 12r+h becomes the element for local head h.
std::vector<double> slice_output_heads(const std::vector<double>& global_value, int rank,
                                       std::int32_t tokens, std::int32_t columns) {
    std::vector<double> result(static_cast<std::size_t>(kHeadDim) * kLocalGeometry.q_heads * tokens *
                               columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t token = 0; token < tokens; ++token) {
            const std::int32_t flat = token + column * tokens;
            for (std::int32_t head = 0; head < kLocalGeometry.q_heads; ++head) {
                const std::size_t src =
                    q_index(kGlobalGeometry, local_q_head_begin(rank) + head, 0, flat);
                const std::size_t dst = q_index(kLocalGeometry, head, 0, flat);
                std::copy_n(global_value.begin() + static_cast<std::ptrdiff_t>(src), kHeadDim,
                            result.begin() + static_cast<std::ptrdiff_t>(dst));
            }
        }
    }
    return result;
}

// Two independently BF16-rounded evaluations of the same quantity are each within the registered
// criterion of the FP64 oracle (leg A below for the split side; the GQA conformance suite for the
// tp1 side), so by the triangle inequality their difference is bounded by TWICE that criterion.
// This factor is that bound, not a fitted tolerance -- and it is needed, because the split does
// not reproduce tp1's arithmetic bit for bit: the split-KV decode route rounds its per-split
// partial accumulators to BF16 and the tp2 geometry's DecodeSplitScale=2 changes the number of
// splits, hence the partition of the reduction.
constexpr double kParityBound = 2.0;

ReductionCriterion scaled(const ReductionCriterion& criterion, double factor) {
    return ReductionCriterion{criterion.relative_l2 * factor, criterion.gross_absolute * factor,
                              criterion.gross_relative_to_max_reference * factor};
}

// Compares one device's 12-head output against (a) the FP64 ideal oracle for its heads at the
// Op's own registered criterion and the granularity that criterion was measured at, (b) the tp1
// 24-head output sliced to the same global heads at the triangle bound, (c) the same, PER HEAD,
// so a head-mapping error is localized rather than averaged away. Fixtures with distinguishable
// reference heads additionally prove that a head permutation could not have passed (c).
int compare_device(const std::string& label, int rank, const std::vector<std::uint16_t>& local_bits,
                   const std::vector<double>& global_reference,
                   const std::vector<double>* oracle, const ReductionCriterion& criterion,
                   std::int32_t tokens, std::int32_t columns, bool check_head_mapping = true) {
    int failures                          = 0;
    const std::vector<double> local_value = bf16_bits_to_double(local_bits);
    const std::string rank_label          = label + " rank=" + std::to_string(rank);
    const ReductionCriterion parity       = scaled(criterion, kParityBound);

    if (oracle != nullptr) {
        // Leg A: the registered numerical contract, applied unmodified to the new geometry.
        failures += verify_attention(rank_label + " vs oracle", local_value,
                                     slice_output_heads(*oracle, rank, tokens, columns), criterion);
    }
    const std::vector<double> reference_slice =
        slice_output_heads(global_reference, rank, tokens, columns);
    failures += verify_attention(rank_label + " vs tp1", local_value, reference_slice, parity);

    for (std::int32_t head = 0; head < kLocalGeometry.q_heads; ++head) {
        const std::int32_t global_head = local_q_head_begin(rank) + head;
        const std::string head_label    = rank_label + " local_head=" + std::to_string(head) +
                                       " global_head=" + std::to_string(global_head);
        const std::vector<double> observed =
            head_block(local_value, kLocalGeometry, head, tokens, columns);
        failures += verify_attention(
            head_label + " vs tp1",
            observed, head_block(global_reference, kGlobalGeometry, global_head, tokens, columns),
            parity);
        if (check_head_mapping) {
            failures += verify_head_mapping(head_label, observed, global_reference, kGlobalGeometry,
                                            global_head, criterion, tokens, columns);
        }
    }
    return failures;
}

// Full-width prefill runs use the same independent FP64 formula on a bounded set of query
// columns. Every selected query still attends over ALL of its visible keys, all 256 dimensions,
// and every head. GPU execution, output parity, and exact cache validation remain full-width.
// Include both sides of tile/page boundaries and the final tiles, where stream-K fixups differ
// from the existing T=66 fixture. This keeps the CPU oracle linear in the number of keys rather
// than evaluating an entire 1024-by-1024 attention matrix.
std::vector<std::int32_t> prefill_oracle_columns(std::int32_t tokens) {
    std::vector<std::int32_t> columns{0, 1, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256,
                                      tokens / 2 - 1, tokens / 2, tokens - 17, tokens - 16,
                                      tokens - 2, tokens - 1};
    std::sort(columns.begin(), columns.end());
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
    return columns;
}

template <typename Value>
std::vector<Value> select_query_columns(const std::vector<Value>& source, std::int32_t heads,
                                         const std::vector<std::int32_t>& columns) {
    const std::size_t column_elements = static_cast<std::size_t>(kHeadDim) * heads;
    std::vector<Value> selected;
    selected.reserve(columns.size() * column_elements);
    for (const auto column : columns) {
        const auto begin = source.begin() + static_cast<std::ptrdiff_t>(column * column_elements);
        selected.insert(selected.end(), begin, begin + static_cast<std::ptrdiff_t>(column_elements));
    }
    return selected;
}

int run_case(DType dtype, const HeadLocalCase& test_case, bool sampled_prefill_oracle = false) {
    const bool append              = std::string(test_case.entry) == "gqa_attention";
    const std::int32_t tokens      = test_case.tokens;
    const std::int32_t total       = test_case.base + tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::string label = case_label(test_case, dtype);

    const std::size_t q_elements =
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.q_heads * tokens;
    const std::size_t kv_elements =
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.kv_heads * tokens;
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(kGlobalGeometry, tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache initial = make_cache(kGlobalGeometry, dtype, max_context, test_case.seed + 10u);
    HostCache expected      = initial;
    if (append) { append_cache(expected, k, v, positions); }
    const auto oracle_columns = sampled_prefill_oracle ? prefill_oracle_columns(tokens)
                                                        : std::vector<std::int32_t>{};
    std::vector<std::int32_t> oracle_positions;
    for (const auto column : oracle_columns) { oracle_positions.push_back(positions[column]); }
    const std::vector<double> oracle =
        sampled_prefill_oracle
            ? ideal_attention(select_query_columns(q, kGlobalGeometry.q_heads, oracle_columns),
                              expected, oracle_positions)
            : ideal_attention(q, expected, positions);
    if (sampled_prefill_oracle) {
        std::cout << label << ": full-width GPU/cache checks, FP64 oracle on "
                  << oracle_columns.size() << " query columns x 24 heads\n" << std::flush;
    }

    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const ReductionCriterion criterion = attention_criterion(dtype);

    int failures = 0;

    // (1) The single-device tp1 run over the whole 24|4 geometry.
    const RunResult reference =
        append ? run_a1(0, kGlobalGeometry, dtype, initial, test_case.mapping, q, k, v, positions,
                        envelope, label + " tp1")
               : run_a3(0, kGlobalGeometry, dtype, initial, test_case.mapping, q, positions,
                        envelope, label + " tp1");
    failures += reference.failures;
    failures += verify_cache(label + " tp1 cache", reference.cache_after, expected);
    const std::vector<double> reference_value = bf16_bits_to_double(reference.output);
    // The tp1 leg is itself held to the registered contract, so a reference regression cannot be
    // mistaken for a split defect.
    const auto oracle_reference =
        sampled_prefill_oracle
            ? select_query_columns(reference_value, kGlobalGeometry.q_heads, oracle_columns)
            : reference_value;
    failures += verify_attention(label + " tp1 vs oracle", oracle_reference, oracle, criterion);

    // (2) The two head-local runs, each on its own device, each over its own 2-head pool.
    std::array<HostCache, kRanks> local_initial{slice_cache_heads(initial, 0),
                                                slice_cache_heads(initial, 1)};
    // Distinctness: if the two devices' pools carried the same bytes, every head-mapping check
    // below would be vacuous. make_cache generates per-head-distinct data; assert it.
    if (dtype == DType::BF16) {
        if (local_initial[0].k_bf16 == local_initial[1].k_bf16 ||
            local_initial[0].v_bf16 == local_initial[1].v_bf16) {
            std::cerr << label << ": the two devices' KV pools are byte-identical, so head "
                                 "mapping is untested\n";
            ++failures;
        }
    } else if (local_initial[0].k_i8 == local_initial[1].k_i8 ||
               local_initial[0].v_i8 == local_initial[1].v_i8) {
        std::cerr << label << ": the two devices' KV pools are byte-identical, so head mapping "
                             "is untested\n";
        ++failures;
    }

    for (int rank = 0; rank < kRanks; ++rank) {
        const std::string rank_label = label + " tp2 rank=" + std::to_string(rank);
        const std::vector<float> local_q = slice_heads_token_major(
            q, kGlobalGeometry.q_heads, local_q_head_begin(rank), kLocalGeometry.q_heads, tokens);
        const std::vector<float> local_k =
            slice_heads_token_major(k, kGlobalGeometry.kv_heads, local_kv_head_begin(rank),
                                    kLocalGeometry.kv_heads, tokens);
        const std::vector<float> local_v =
            slice_heads_token_major(v, kGlobalGeometry.kv_heads, local_kv_head_begin(rank),
                                    kLocalGeometry.kv_heads, tokens);

        const RunResult observed =
            append ? run_a1(rank, kLocalGeometry, dtype, local_initial[rank], test_case.mapping,
                            local_q, local_k, local_v, positions, envelope, rank_label)
                   : run_a3(rank, kLocalGeometry, dtype, local_initial[rank], test_case.mapping,
                            local_q, positions, envelope, rank_label);
        failures += observed.failures;
        // The device wrote its new K/V into the LOCAL head rows of its own pool: the expected
        // pool is the same head slice of the tp1 expected pool.
        failures += verify_cache(rank_label + " cache", observed.cache_after,
                                 slice_cache_heads(expected, rank));
        if (sampled_prefill_oracle) {
            failures += verify_attention(
                rank_label + " full output vs tp1", bf16_bits_to_double(observed.output),
                slice_output_heads(reference_value, rank, tokens, 1),
                scaled(criterion, kParityBound));
            failures += compare_device(
                label + " sampled query columns", rank,
                select_query_columns(observed.output, kLocalGeometry.q_heads, oracle_columns),
                oracle_reference, &oracle, criterion,
                static_cast<std::int32_t>(oracle_columns.size()), 1, test_case.check_head_mapping);
        } else {
            failures += compare_device(label, rank, observed.output, reference_value, &oracle,
                                       criterion, tokens, 1, test_case.check_head_mapping);
        }
    }
    return failures;
}

// --- batched (speculative-verify) route ------------------------------------------------------
//
// B>1 is the only way a 12-head geometry reaches GqaAttentionRoute::ChunkedSmallT, and it is the
// only route that instantiates the decode kernels' MultiBatch=true / Masked=true specializations.
// The reference is the tp1 run; the FP64 oracle is not evaluated here (the per-request extraction
// it needs belongs to the conformance suite, and the tp1 leg already carries it there).
struct BatchHeadLocalCase {
    std::int32_t width;
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    MappingPattern mapping;
    std::uint32_t seed;
};

struct BatchRunResult {
    std::vector<std::uint16_t> output;
    int failures = 0;
};

BatchRunResult run_batch(int device, const Geometry& geometry, DType dtype,
                         std::span<const HostCache> rows, const BatchHeadLocalCase& test_case,
                         const std::vector<float>& q, const std::vector<float>& k,
                         const std::vector<float>& v,
                         const std::vector<std::int32_t>& positions,
                         ops::GqaExecutionEnvelope envelope, const std::string& label) {
    const CurrentDeviceScope scope(device);
    const auto batch = static_cast<std::int32_t>(rows.size());
    BatchDeviceCache cache(rows, test_case.mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvalid(test_case.valid_columns.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer drows(test_case.table_rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dvalid.copy_from_host(test_case.valid_columns.data(),
                          test_case.valid_columns.size() * sizeof(std::int32_t));
    drows.copy_from_host(test_case.table_rows.data(),
                         test_case.table_rows.size() * sizeof(std::int32_t));
    const std::vector<std::uint16_t> canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tp(dp.data(), DType::I32, {test_case.width, batch});
    Tensor tvalid(dvalid.data(), DType::I32, {batch});
    Tensor trows(drows.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});

    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, batch, test_case.width, test_case.width);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    const bool masked = std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](std::int32_t valid) { return valid != test_case.width; });
    ops::gqa_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, trows, kAttentionScale,
                       cache.view(), envelope, workspace, tout, nullptr);
    cuda_synchronize();

    BatchRunResult result;
    result.output = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    result.failures += verify_input(label + " q unchanged", dq, q_bits);
    result.failures += verify_positions(label + " positions unchanged", dp, positions);
    result.failures += dout.verify_guards((label + " output").c_str());
    result.failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    return result;
}

int run_batch_case(DType dtype, const BatchHeadLocalCase& test_case) {
    const auto batch = static_cast<std::int32_t>(test_case.contexts.size());
    std::int32_t maximum_visible = 1;
    for (std::int32_t row = 0; row < batch; ++row) {
        maximum_visible =
            std::max(maximum_visible, test_case.contexts[static_cast<std::size_t>(row)] +
                                          test_case.valid_columns[static_cast<std::size_t>(row)]);
    }
    const std::int32_t max_context = maximum_visible + 3;
    const std::size_t columns      = static_cast<std::size_t>(test_case.width) * batch;
    std::vector<float> q = make_bf16_values(
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.q_heads * columns, test_case.seed,
        -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.kv_heads * columns,
        test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.kv_heads * columns,
        test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(kGlobalGeometry, static_cast<std::int32_t>(columns), k, v);

    std::vector<std::int32_t> positions(columns, 0);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < valid; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] =
                test_case.contexts[static_cast<std::size_t>(row)] + token;
        }
        const std::int32_t padding =
            valid == 0 ? 0 : test_case.contexts[static_cast<std::size_t>(row)] + valid - 1;
        for (std::int32_t token = valid; token < test_case.width; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] = padding;
        }
    }

    std::vector<HostCache> global_rows;
    global_rows.reserve(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        global_rows.push_back(make_cache(kGlobalGeometry, dtype, max_context,
                                         test_case.seed + 20u + 3u * row));
    }

    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(maximum_visible),
                                             static_cast<std::uint32_t>(maximum_visible)};
    const ReductionCriterion criterion = attention_criterion(dtype);
    const std::string label = std::string("gqa_attention batch ") + cache_name(dtype) +
                              " mapping=" + mapping_name(test_case.mapping) +
                              " B=" + std::to_string(batch) +
                              " W=" + std::to_string(test_case.width);

    int failures = 0;
    const BatchRunResult reference = run_batch(0, kGlobalGeometry, dtype, global_rows, test_case, q,
                                               k, v, positions, envelope, label + " tp1");
    failures += reference.failures;
    const std::vector<double> reference_value = bf16_bits_to_double(reference.output);

    for (int rank = 0; rank < kRanks; ++rank) {
        std::vector<HostCache> local_rows;
        local_rows.reserve(static_cast<std::size_t>(batch));
        for (std::int32_t row = 0; row < batch; ++row) {
            local_rows.push_back(slice_cache_heads(global_rows[static_cast<std::size_t>(row)],
                                                   rank));
        }
        const std::vector<float> local_q = slice_heads_token_major(
            q, kGlobalGeometry.q_heads, local_q_head_begin(rank), kLocalGeometry.q_heads,
            static_cast<std::int32_t>(columns));
        const std::vector<float> local_k = slice_heads_token_major(
            k, kGlobalGeometry.kv_heads, local_kv_head_begin(rank), kLocalGeometry.kv_heads,
            static_cast<std::int32_t>(columns));
        const std::vector<float> local_v = slice_heads_token_major(
            v, kGlobalGeometry.kv_heads, local_kv_head_begin(rank), kLocalGeometry.kv_heads,
            static_cast<std::int32_t>(columns));

        const BatchRunResult observed =
            run_batch(rank, kLocalGeometry, dtype, local_rows, test_case, local_q, local_k, local_v,
                      positions, envelope, label + " tp2 rank=" + std::to_string(rank));
        failures += observed.failures;
        failures += compare_device(label, rank, observed.output, reference_value, nullptr,
                                   criterion, test_case.width, batch);
    }
    return failures;
}

// --- standalone append (A2) into a head-local pool -------------------------------------------
//
// A2 has no Q heads at all, so it selects its geometry by KV-head count. A head-local pool has 2
// KV heads, which is also the 16|2 geometry's count -- the append kernels read Geometry::KVHeads
// and nothing else, so the two are the same operation. This case proves the LOCAL head rows are
// the ones written, and (with T>=128) reaches the KVHeads==2 paged INT8 fill kernel.
int run_append_case(DType dtype, MappingPattern mapping, std::int32_t tokens, std::int32_t base,
                    std::uint32_t seed) {
    const std::int32_t max_context = base + tokens + 4;
    const std::size_t elements =
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.kv_heads * tokens;
    std::vector<float> k = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);
    inject_codec_edges(kGlobalGeometry, tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base + token;
    }

    const HostCache initial = make_cache(kGlobalGeometry, dtype, max_context, seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);

    const std::string label = std::string("gqa_kv_append tp2 ") + cache_name(dtype) +
                              " mapping=" + mapping_name(mapping) +
                              " T=" + std::to_string(tokens);
    int failures = 0;
    for (int rank = 0; rank < kRanks; ++rank) {
        const CurrentDeviceScope scope(rank);
        const std::string rank_label = label + " rank=" + std::to_string(rank);
        const HostCache local        = slice_cache_heads(initial, rank);
        DeviceCache cache(local, mapping);

        const std::vector<float> local_k =
            slice_heads_token_major(k, kGlobalGeometry.kv_heads, local_kv_head_begin(rank),
                                    kLocalGeometry.kv_heads, tokens);
        const std::vector<float> local_v =
            slice_heads_token_major(v, kGlobalGeometry.kv_heads, local_kv_head_begin(rank),
                                    kLocalGeometry.kv_heads, tokens);
        const std::vector<std::uint16_t> k_bits = to_bf16_bits(local_k);
        const std::vector<std::uint16_t> v_bits = to_bf16_bits(local_v);
        GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
        dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
        dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
        dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
        Tensor tk(dk.data(), DType::BF16, {kHeadDim, kLocalGeometry.kv_heads, tokens});
        Tensor tv(dv.data(), DType::BF16, {kHeadDim, kLocalGeometry.kv_heads, tokens});
        Tensor tp(dp.data(), DType::I32, {tokens});

        ops::gqa_kv_append(tk, tv, tp, cache.view(), nullptr);
        cuda_synchronize();

        failures += verify_cache(rank_label, cache.snapshot(), slice_cache_heads(expected, rank));
        failures += verify_input(rank_label + " k unchanged", dk, k_bits);
        failures += verify_input(rank_label + " v unchanged", dv, v_bits);
        failures += verify_positions(rank_label + " positions unchanged", dp, positions);
        failures += cache.verify_guards(rank_label);
    }
    return failures;
}

// --- rejections -------------------------------------------------------------------------------

int verify_rejections() {
    int failures = 0;
    // A 12-head geometry is registered and must be admitted by the public capacity query.
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(kLocalGeometry.q_heads, DType::I8,
                                                          {1, 4096}, 1, 1, 6);
    } catch (const std::invalid_argument&) {
        std::cerr << "gqa_attention rejected the registered head-local geometry\n";
        ++failures;
    }
    // Head counts that are not a registered geometry stay rejected: a shard extent is only legal
    // where it was registered, and 12|4 / 6|1 are not.
    for (const std::int32_t q_heads : {6, 8, 12 + 1, 20}) {
        try {
            (void)ops::gqa_attention_workspace_capacity_bytes(q_heads, DType::I8, {1, 4096}, 1, 1,
                                                              6);
            std::cerr << "gqa_attention accepted unregistered Q-head count " << q_heads << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const bool prefill_only = argc == 2 && std::string(argv[1]) == "--prefill-only";
    const bool long_context_only = argc == 2 && std::string(argv[1]) == "--long-context-only";
    if (argc != 1 && !prefill_only && !long_context_only) {
        std::cerr << "usage: " << argv[0] << " [--prefill-only|--long-context-only]\n";
        return 2;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < kRanks) {
        std::cout << "SKIP: head-local attention parity requires two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }

    // Exact envelopes and no valid-column mask reach the VoltaFlash path on sm_70. These are
    // real prefill widths, not small-T decode calls against a larger cache capacity. The second
    // chunk fixture additionally verifies that all history survives while 512 new rows append.
    int failures = 0;
    const HeadLocalCase prefill_cases[] = {
        {"gqa_attention", 512, 0, 512, MappingPattern::Identity, 1501u},
        {"gqa_attention", 1024, 0, 1024, MappingPattern::Identity, 1502u},
        {"gqa_attention", 512, 512, 1024, MappingPattern::Fragmented, 1503u},
    };
    if (!long_context_only) {
        for (const auto& test_case : prefill_cases) {
            failures += run_case(DType::I8, test_case, true);
        }
    }
    if (prefill_only) {
        std::cout << (failures == 0 ? "PASS" : "FAIL")
                  << " head-local INT8 prefill (512/1024 columns, independent FP64 oracle)\n";
        return failures == 0 ? 0 : 1;
    }

    // Four decode queries at the acceptance workload's occupied context, with the full
    // 180K capacity envelope. Evaluate every head/key against FP64 without subsampling.
    // Averaging 85K zero-mean V rows need not separate different heads by the fixture's
    // anti-permutation margin; the short cases retain that distinct-input requirement.
    // Oracle, per-head parity, and exact cache checks keep their original criteria here.
    const HeadLocalCase long_context_case{
        .entry = "gqa_attention",
        .tokens = 4,
        .base = 84996,
        .envelope_max = 180000,
        .mapping = MappingPattern::Identity,
        .seed = 1601u,
        .check_head_mapping = false,
    };
    std::cout << "head-local INT8 85K decode: full FP64 oracle, 4 queries x 24 heads; "
                 "180K capacity\n" << std::flush;
    failures += run_case(DType::I8, long_context_case);
    if (long_context_only) {
        std::cout << (failures == 0 ? "PASS" : "FAIL")
                  << " head-local INT8 85K decode (independent FP64 oracle)\n";
        return failures == 0 ? 0 : 1;
    }

    failures += verify_rejections();

    // T=1 is the decode edge; T=6 is the split-KV small-T maximum; T=17 and T=66 take the prompt
    // route. Every case's key range crosses at least one 64-key page boundary, and the base=1000
    // and base=190 cases span many pages inside one split.
    const HeadLocalCase cases[] = {
        {"gqa_attention", 1, 0, 1, MappingPattern::Identity, 1101u},
        {"gqa_attention", 1, 64, 70, MappingPattern::Offset, 1102u},
        {"gqa_attention", 6, 61, 67, MappingPattern::Fragmented, 1103u},
        {"gqa_attention", 6, 190, 512, MappingPattern::Identity, 1104u},
        {"gqa_attention", 17, 31, 48, MappingPattern::Fragmented, 1105u},
        {"gqa_attention", 66, 63, 129, MappingPattern::Identity, 1106u},
        {"gqa_attention_cached", 1, 128, 129, MappingPattern::Fragmented, 1201u},
        {"gqa_attention_cached", 1, 1000, 1024, MappingPattern::Offset, 1202u},
        {"gqa_attention_cached", 6, 61, 512, MappingPattern::Identity, 1203u},
        {"gqa_attention_cached", 17, 31, 48, MappingPattern::Identity, 1204u},
    };
    for (const DType dtype : {DType::BF16, DType::I8}) {
        for (const HeadLocalCase& test_case : cases) { failures += run_case(dtype, test_case); }
        // W in (6,16] with B>1 is the only route into GqaAttentionRoute::ChunkedSmallT for a
        // 12-head geometry; W<=6 with B>1 is SmallT with MultiBatch=true.
        failures += run_batch_case(dtype, {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Identity,
                                           1301u + (dtype == DType::I8 ? 10u : 0u)});
        failures += run_batch_case(dtype, {1, {63, 511}, {1, 1}, {1, 0}, MappingPattern::Fragmented,
                                           1311u + (dtype == DType::I8 ? 10u : 0u)});
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            failures += run_append_case(dtype, mapping, 3, 63, 1401u);
        }
        // T>=128 reaches the paged INT8 fill kernel, which is selected on KVHeads==2.
        failures += run_append_case(dtype, MappingPattern::Fragmented, 129, 61, 1451u);
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " head-local attention parity (12|2 per device vs 24|4 single device)\n";
    return failures == 0 ? 0 : 1;
}
