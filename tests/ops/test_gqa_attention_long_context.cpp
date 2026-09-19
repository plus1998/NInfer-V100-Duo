// GQA attention past the pre-YaRN 262,144-key envelope.
//
// WHY THIS EXISTS SEPARATELY. `kGqaAttentionMaximumVisibleKeys` is the Op's declared visible-key
// domain, and the split-KV decode kernels size their shared-memory page-id staging from it
// (`kGqaSmallTSplitPageIds`). Raising the domain from 262,144 to 1,048,576 so a YaRN-extended
// engine can be planned at all therefore changes a shared-memory extent, and the failure mode of
// getting it wrong is an out-of-bounds shared write inside a kernel -- silent corruption, not an
// error. `ninfer_gqa_attention_test` qualifies the formula and the criteria at small extents and
// must stay fast; this file qualifies the ONE thing that only appears at a long window, and it
// costs GPU memory and oracle time, so it is its own target.
//
// WHAT EACH CASE PROVES. A split covers
// `ceil(ceil(window / KeyBlock) / active_splits) * KeyBlock` keys and `active_splits` saturates at
// `Geometry::DecodeSplits`, so the pages one split walks grow linearly with the window past that
// saturation point. Each case below is chosen so that count EXCEEDS 64 -- the bound the kernels
// hard-coded while 262,144 was the domain -- and the test asserts that arithmetic itself, so the
// case cannot quietly become vacuous if the split policy is retuned.
//
//   Gqa27 (24|4, DecodeSplits 85), window 400,001 -> 75 pages per split
//   Gqa27Tp2 (12|2, DecodeSplits 170), window 700,001 -> 66 pages per split
//   Gqa27 (24|4), windows 1,048,575 and 1,048,576 -> 194 pages per split
//
// That last pair is the DECLARED DOMAIN's own boundary (b-1 and b), where the derived staging bound
// is exactly tight: the span is 386 x 32 = 12,352 keys = 193 whole pages, so 194 ids covers it with
// one page of slack and no more. Both run against one cache, so the pair costs one pool.
//
// Both KV dtypes run for the tp1 geometry because they are two independent kernels with their own
// staging arrays; the head-local tp2 geometry runs INT8, which is the 1M deployment's KV format.
//
// WHAT IS COMPARED. The A3 (cached decode) route at T = 1 against an FP64 oracle over the whole
// visible history, for one query head per KV head. Every KV head therefore has its own page walk
// checked; running all 24 query heads through a 400,000-key FP64 softmax would multiply the oracle
// cost by six and check the same page arithmetic six times. The block table is the `Offset`
// mapping, not identity, so a mis-staged page id reads a genuinely different page.
//
// Skipped (exit 77) with no CUDA device, and skipped with a clear message if the device cannot
// hold the case's cache.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/gqa_attention_fixture.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::gqa;

namespace {

// Mirrors `gqa_small_t_split_upper_bound` / `gqa_small_t_active_splits` for the >16,390 tier, which
// is the only tier a long window can land in, and then the kernels' own `units_per_split` /
// `page_count` arithmetic. Kept here as an independent restatement: if the two ever disagree the
// case's own assertion below fires rather than the kernel silently overrunning.
std::int32_t pages_per_split(std::int32_t window, std::int32_t decode_splits,
                             std::int32_t key_block) {
    const std::int32_t requested = (window + 479) / 480;
    const std::int32_t splits    = std::min(requested, decode_splits);
    const std::int32_t tiles     = (window + key_block - 1) / key_block;
    const std::int32_t span      = ((tiles + splits - 1) / splits) * key_block;
    return (span + kPagedKVPageSize - 1) / kPagedKVPageSize + 1;
}

struct LongCase {
    Geometry geometry;
    std::int32_t decode_splits; // Geometry::DecodeSplits for `geometry`
    DType dtype;
    // Query-token absolute positions, each run against the SAME cache; window == base + 1. A list
    // rather than one value so the domain boundary can be checked at b-1 AND b without paying for
    // a second multi-GiB cache and a second FP64 oracle pass over the shared history.
    std::vector<std::int32_t> bases;
    std::uint32_t seed;
};

// Bytes the device cache needs for one case, so a small device skips instead of failing on an
// allocation the case never claimed to fit.
std::size_t device_cache_bytes(const LongCase& test_case) {
    const std::int32_t logical = align_up_page(test_case.bases.back() + 4);
    const auto positions       = static_cast<std::size_t>(logical) + 2 * kPagedKVPageSize;
    const std::size_t elements =
        positions * static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(
                                                             test_case.geometry.kv_heads);
    if (test_case.dtype == DType::BF16) { return 2 * elements * sizeof(std::uint16_t); }
    const std::size_t scales =
        positions * static_cast<std::size_t>(kQuantGroups) *
        static_cast<std::size_t>(test_case.geometry.kv_heads) * sizeof(std::uint16_t);
    return 2 * elements + 2 * scales;
}

// The fixture's own query magnitude, deliberately kept.
//
// A larger range was tried, on the theory that concentrating the softmax would raise the output
// magnitude and buy relative-L2 headroom, and it was MEASURED not to help: at +/-4.0 the score
// spread rises to only ~0.33, whose maximum over a million near-Gaussian draws is ~1.65, so the
// softmax stays effectively uniform -- `reference_rms` moved 5.5e-4 -> 5.8e-4 while the errors
// grew, taking the relative-L2 ratios from 0.90-1.01 to 1.08-1.14 (i.e. every case then failed).
// Concentrating a million-key softmax would need query magnitudes far outside any
// target-representative activation range. Conditioning is not the lever here; the criterion being
// calibrated against the measured storage floor is (see long_window_attention_criterion).
constexpr float kQueryRange = 0.25f;


// One query head per KV head: q_head = kv_head * group. Covers every KV head's page walk.
std::vector<std::int32_t> probe_heads(const Geometry& geometry) {
    std::vector<std::int32_t> heads;
    heads.reserve(static_cast<std::size_t>(geometry.kv_heads));
    for (std::int32_t kv = 0; kv < geometry.kv_heads; ++kv) {
        heads.push_back(kv * geometry.query_group());
    }
    return heads;
}

// FP64 oracle restricted to `heads`, for a single query token at `position`. Same formula as the
// fixture's `ideal_attention`, evaluated over the complete visible history [0, position].
std::vector<double> long_reference(const std::vector<float>& q, const HostCache& cache,
                                   std::int32_t position, const std::vector<std::int32_t>& heads) {
    const Geometry& geometry   = cache.geometry;
    const std::int32_t visible = position + 1;
    std::vector<double> scores(static_cast<std::size_t>(visible));
    std::vector<double> output(heads.size() * static_cast<std::size_t>(kHeadDim));

    for (std::size_t slot = 0; slot < heads.size(); ++slot) {
        const std::int32_t q_head  = heads[slot];
        const std::int32_t kv_head = q_head / geometry.query_group();
        double max_score           = -std::numeric_limits<double>::infinity();
        for (std::int32_t key = 0; key < visible; ++key) {
            double dot = 0.0;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                dot += static_cast<double>(q[q_index(geometry, q_head, d, 0)]) *
                       cache_value(cache, true, kv_head, key, d);
            }
            const double score                    = dot * static_cast<double>(kAttentionScale);
            scores[static_cast<std::size_t>(key)] = score;
            max_score                             = std::max(max_score, score);
        }
        double sum = 0.0;
        for (std::int32_t key = 0; key < visible; ++key) {
            const double probability = std::exp(scores[static_cast<std::size_t>(key)] - max_score);
            scores[static_cast<std::size_t>(key)] = probability;
            sum += probability;
        }
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            double value = 0.0;
            for (std::int32_t key = 0; key < visible; ++key) {
                value += scores[static_cast<std::size_t>(key)] *
                         cache_value(cache, false, kv_head, key, d);
            }
            output[slot * static_cast<std::size_t>(kHeadDim) + static_cast<std::size_t>(d)] =
                value / sum;
        }
    }
    return output;
}

int run_long_case(const LongCase& test_case) {
    const Geometry& geometry = test_case.geometry;
    const std::int32_t max_context = test_case.bases.back() + 4;
    const std::string case_name =
        std::string(geometry.name) + " " + cache_name(test_case.dtype);

    // Every window in the case must exercise more than the 64 page ids the kernels staged while
    // 262,144 was the declared domain -- otherwise it proves nothing about the raised bound.
    for (const std::int32_t base : test_case.bases) {
        const std::int32_t window = base + 1;
        const std::int32_t staged =
            std::max(pages_per_split(window, test_case.decode_splits, 32),
                     pages_per_split(window, test_case.decode_splits, 64));
        if (staged <= 64) {
            std::cerr << case_name << " window " << window
                      << ": does not exceed the old 64-page staging bound (" << staged
                      << "); it would prove nothing\n";
            return 1;
        }
        if (static_cast<std::uint32_t>(window) > ops::kGqaAttentionMaximumVisibleKeys) {
            std::cerr << case_name << " window " << window
                      << ": outside the Op's declared visible-key domain\n";
            return 1;
        }
    }

    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        free_bytes < device_cache_bytes(test_case) + (512ULL << 20)) {
        std::cout << "skip: " << case_name << " up to window " << test_case.bases.back() + 1
                  << " needs " << (device_cache_bytes(test_case) >> 20)
                  << " MiB of free device memory\n";
        return 0;
    }

    // One cache, one paged upload, reused by every window in the case.
    const HostCache cache_host =
        make_cache(geometry, test_case.dtype, max_context, test_case.seed + 10u);
    DeviceCache cache(cache_host, MappingPattern::Offset);
    const std::vector<std::int32_t> heads = probe_heads(geometry);

    int failures = 0;
    for (const std::int32_t base : test_case.bases) {
        const std::int32_t window = base + 1;
        const std::int32_t staged =
            std::max(pages_per_split(window, test_case.decode_splits, 32),
                     pages_per_split(window, test_case.decode_splits, 64));
        const std::string label = "gqa_attention_cached long " + case_name + " window " +
                                  std::to_string(window);

        const std::size_t q_elements =
            static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(geometry.q_heads);
        const std::vector<float> q =
            make_bf16_values(q_elements, test_case.seed + static_cast<std::uint32_t>(base),
                             -kQueryRange, kQueryRange);
        const std::vector<std::int32_t> positions{base};
        const std::vector<double> reference = long_reference(q, cache_host, base, heads);
        // The BF16 output-storage floor for THIS case's own reference population: the relative-L2
        // the kernel could not beat even if its arithmetic were exact. The criterion is calibrated
        // against it rather than against a constant fitted at a different shape.
        const double storage_floor        = bf16_storage_floor_relative_l2(reference);
        const ReductionCriterion criterion =
            long_window_attention_criterion(test_case.dtype, storage_floor);

        const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
        GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
        GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
        dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
        dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
        const std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
        dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

        Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, 1});
        Tensor tp(dp.data(), DType::I32, {1});
        Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, 1});
        const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(window),
                                                 static_cast<std::uint32_t>(window)};
        const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
            geometry.q_heads, test_case.dtype, envelope, 1, 1, 1);
        GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
        WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

        ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                                  nullptr);
        cuda_synchronize();

        const std::vector<std::uint16_t> output_bits =
            copy_from_guarded<std::uint16_t>(dout, q_bits.size());
        const std::vector<double> output_all = bf16_bits_to_double(output_bits);
        std::vector<double> actual(reference.size());
        for (std::size_t slot = 0; slot < heads.size(); ++slot) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                actual[slot * static_cast<std::size_t>(kHeadDim) + static_cast<std::size_t>(d)] =
                    output_all[q_index(geometry, heads[slot], d, 0)];
            }
        }

        int case_failures = verify_attention(label, actual, reference, criterion);

        // Negative control: the same output judged against an oracle whose window is ONE PAGE
        // shorter must be REJECTED. That is exactly the defect class this file exists for -- a
        // mis-staged or dropped physical page id in a split's page walk -- expressed as 64 of the
        // window's 16,384 pages' worth of keys going missing from the reduction. If the criterion
        // above could not tell those apart it would not be worth running.
        const std::vector<double> short_reference =
            long_reference(q, cache_host, base - kPagedKVPageSize, heads);
        const ReductionStats control = compute_reduction_stats(
            actual.data(), short_reference.data(), static_cast<std::int64_t>(actual.size()));
        if (reduction_passes(control, static_cast<std::int64_t>(actual.size()), criterion)) {
            std::cerr << label
                      << ": criterion also accepted an oracle one page short -- it cannot detect a "
                         "dropped page in the split's page walk\n";
            ++case_failures;
        }
        // The staging overrun this case exists for would corrupt neighbouring shared memory, not
        // the guarded device allocations, so the guards are a secondary check -- but a wrong page
        // id can also walk off the physical pool, which they do catch.
        case_failures += dout.verify_guards((label + " output").c_str());
        case_failures += workspace_buffer.verify_guards((label + " workspace").c_str());
        case_failures += cache.verify_guards(label);
        case_failures += verify_input(label + " q unchanged", dq, q_bits);
        case_failures += verify_positions(label + " positions unchanged", dp, positions);
        if (case_failures == 0) {
            const ReductionStats stats =
                compute_reduction_stats(actual.data(), reference.data(),
                                        static_cast<std::int64_t>(actual.size()));
            std::cout << "ok " << label << " (" << staged << " page ids per split; storage floor "
                      << storage_floor << ", limit " << criterion.relative_l2 << ", measured "
                      << stats.relative_l2 << " = " << stats.relative_l2 / storage_floor
                      << "x floor; one-page-short control rejected at "
                      << control.relative_l2 / criterion.relative_l2 << "x the limit)"
                      << std::endl;
        }
        failures += case_failures;
    }
    // No `verify_cache(cache.snapshot(), ...)` here, deliberately: A3 is the cached route
    // (`GqaCachedInput::writes_cache == false`), so the Op never writes these pages at all, and a
    // snapshot would gather the whole 2.2 GiB physical pool into a SECOND host copy that the
    // device-memory skip guard above does not account for. Out-of-bounds writes are still covered
    // -- `cache.verify_guards` runs per window, on every one of the pool's guarded allocations.
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    const LongCase cases[] = {
        {{"qwen3_6_27b", 24, 4}, 85, DType::I8, {400000}, 901u},
        {{"qwen3_6_27b", 24, 4}, 85, DType::BF16, {400000}, 902u},
        {{"qwen3_6_27b_tp2", 12, 2}, 170, DType::I8, {700000}, 903u},
        // The declared domain's own boundary, b-1 and b (windows 1,048,575 and 1,048,576), which is
        // where the derived staging bound is tightest: the span is 386 x 32 = 12,352 keys = exactly
        // 193 pages, so 194 ids covers it with one page of slack and no more. INT8 is the format
        // the 1M deployment uses; the pages are ~2.2 GiB plus 67 MiB of scales, and the case skips
        // itself if the device cannot hold them.
        {{"qwen3_6_27b", 24, 4}, 85, DType::I8, {1048574, 1048575}, 904u},
    };

    int failures = 0;
    for (const LongCase& test_case : cases) { failures += run_long_case(test_case); }
    if (failures != 0) {
        std::cerr << failures << " long-context GQA failure(s)\n";
        return 1;
    }
    std::cout << "OK gqa_attention long-context envelope\n";
    return 0;
}
