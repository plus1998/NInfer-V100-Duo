#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/gqa_attention_fixture.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The cache model, oracle, codec, and criteria live in the shared fixture so the tp2 head-local
// parity suite (tests/ops/test_attention_headlocal.cpp) builds its per-device pools from exactly
// the same model this suite builds its tp1 pool from.
using namespace ninfer::test::gqa;

constexpr Geometry kGeometries[] = {
    {"qwen3_6_27b", 24, 4},
    {"qwen3_6_35b_a3b", 16, 2},
};

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    std::uint32_t seed;
};
std::string case_label(const char* entry, const Geometry& geometry, DType dtype,
                       const AttentionCase& test_case, MappingPattern mapping) {
    return std::string(entry) + " " + geometry.name + " " + cache_name(dtype) +
           " mapping=" + mapping_name(mapping) + " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max);
}


int run_append_case(const Geometry& geometry, DType dtype, MappingPattern mapping,
                    std::uint32_t seed, std::int32_t tokens = 3, std::int32_t base = 63) {
    const std::int32_t max_context = base + tokens + 4;
    const std::size_t elements =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(geometry.kv_heads) * tokens;
    std::vector<float> k = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);
    inject_codec_edges(geometry, tokens, k, v);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base + token;
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    DeviceCache cache(initial, mapping);

    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dpositions(positions.size() * sizeof(std::int32_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dpositions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dpositions.data(), DType::I32, {tokens});

    ops::gqa_kv_append(tk, tv, tp, cache.view(), nullptr);
    cuda_synchronize();

    const std::string label = std::string("gqa_kv_append ") + geometry.name + " " +
                              cache_name(dtype) + " mapping=" + mapping_name(mapping);
    int failures = verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dpositions, positions);
    failures += cache.verify_guards(label);
    return failures;
}

int run_a1_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, test_case.tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, test_case.seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference = ideal_attention(q, expected, positions);
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
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    // Positive finite poison makes a producer's unwritten partials visible to the oracle:
    // negative or zero partial_l values would let the reducer silently skip them.
    workspace_buffer.fill(0x3f);
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kAttentionScale, cache.batch_view(),
                       envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label = case_label("gqa_attention", geometry, dtype, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    failures += verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += verify_positions(label + " table row unchanged", dtable_row, {table_row});
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

int run_a3_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache cache_host = make_cache(geometry, dtype, max_context, test_case.seed + 10u);
    const std::vector<double> reference = ideal_attention(q, cache_host, positions);
    DeviceCache cache(cache_host, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    workspace_buffer.fill(0x3f);
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                              nullptr);
    cuda_synchronize();

    const std::string label =
        case_label("gqa_attention_cached", geometry, dtype, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    failures += verify_cache(label + " cache unchanged", cache.snapshot(), cache_host);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

struct BatchAttentionCase {
    std::int32_t width;
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    MappingPattern mapping;
    std::uint32_t seed;
};

std::vector<float> extract_request_columns(const std::vector<float>& source,
                                           std::size_t column_elements, std::int32_t width,
                                           std::int32_t request, std::int32_t valid) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::vector<float> result(static_cast<std::size_t>(valid) * column_elements);
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(begin), result.size(), result.begin());
    return result;
}

void insert_request_columns(const std::vector<double>& source, std::size_t column_elements,
                            std::int32_t width, std::int32_t request,
                            std::vector<double>& destination) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::copy(source.begin(), source.end(),
              destination.begin() + static_cast<std::ptrdiff_t>(begin));
}

int verify_invalid_columns_zero(const std::string& label, std::span<const std::uint16_t> output,
                                const Geometry& geometry, std::int32_t width,
                                std::span<const std::int32_t> valid_columns) {
    int failures                      = 0;
    const std::size_t column_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    for (std::size_t batch = 0; batch < valid_columns.size(); ++batch) {
        for (std::int32_t token = valid_columns[batch]; token < width; ++token) {
            const std::size_t begin =
                (batch * static_cast<std::size_t>(width) + token) * column_elements;
            for (std::size_t element = 0; element < column_elements; ++element) {
                if (output[begin + element] != 0) {
                    if (failures == 0) {
                        std::cerr << label << ": invalid output column is not BF16 zero at row "
                                  << batch << " column " << token << '\n';
                    }
                    ++failures;
                }
            }
        }
    }
    return failures;
}

int run_batch_case(const Geometry& geometry, DType dtype, const BatchAttentionCase& test_case) {
    const std::int32_t batch = static_cast<std::int32_t>(test_case.contexts.size());
    if (batch <= 0 || test_case.valid_columns.size() != static_cast<std::size_t>(batch) ||
        test_case.table_rows.size() != static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("invalid GQA batch test profile");
    }

    std::int32_t maximum_visible = 1;
    for (std::int32_t row = 0; row < batch; ++row) {
        maximum_visible =
            std::max(maximum_visible, test_case.contexts[static_cast<std::size_t>(row)] +
                                          test_case.valid_columns[static_cast<std::size_t>(row)]);
    }
    const std::int32_t max_context       = maximum_visible + 3;
    const std::size_t q_column_elements  = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    const std::size_t kv_column_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads;
    const std::size_t columns            = static_cast<std::size_t>(test_case.width) * batch;
    std::vector<float> q =
        make_bf16_values(q_column_elements * columns, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, static_cast<std::int32_t>(columns), k, v);

    std::vector<std::int32_t> positions(columns, 0);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < valid; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] =
                test_case.contexts[static_cast<std::size_t>(row)] + token;
        }
        const std::int32_t padding_position =
            valid == 0 ? 0 : test_case.contexts[static_cast<std::size_t>(row)] + valid - 1;
        for (std::int32_t token = valid; token < test_case.width; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] = padding_position;
        }
    }

    std::vector<HostCache> initial;
    initial.reserve(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial.push_back(
            make_cache(geometry, dtype, max_context, test_case.seed + 20u + 3u * row));
    }
    std::vector<HostCache> expected = initial;
    std::vector<double> reference(q_column_elements * columns, 0.0);
    for (std::int32_t request = 0; request < batch; ++request) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(request)];
        if (valid == 0) { continue; }
        const std::int32_t table_row = test_case.table_rows[static_cast<std::size_t>(request)];
        std::vector<std::int32_t> row_positions(static_cast<std::size_t>(valid));
        std::copy_n(positions.begin() + static_cast<std::ptrdiff_t>(request * test_case.width),
                    valid, row_positions.begin());
        const std::vector<float> row_q =
            extract_request_columns(q, q_column_elements, test_case.width, request, valid);
        const std::vector<float> row_k =
            extract_request_columns(k, kv_column_elements, test_case.width, request, valid);
        const std::vector<float> row_v =
            extract_request_columns(v, kv_column_elements, test_case.width, request, valid);
        append_cache(expected[static_cast<std::size_t>(table_row)], row_k, row_v, row_positions);
        insert_request_columns(
            ideal_attention(row_q, expected[static_cast<std::size_t>(table_row)], row_positions),
            q_column_elements, test_case.width, request, reference);
    }

    BatchDeviceCache cache(initial, test_case.mapping);
    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvalid(test_case.valid_columns.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_rows(test_case.table_rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dvalid.copy_from_host(test_case.valid_columns.data(),
                          test_case.valid_columns.size() * sizeof(std::int32_t));
    dtable_rows.copy_from_host(test_case.table_rows.data(),
                               test_case.table_rows.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tp(dp.data(), DType::I32, {test_case.width, batch});
    Tensor tvalid(dvalid.data(), DType::I32, {batch});
    Tensor ttable_rows(dtable_rows.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(maximum_visible),
                                             static_cast<std::uint32_t>(maximum_visible)};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, batch, test_case.width, test_case.width);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    const bool masked = std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](std::int32_t valid) { return valid != test_case.width; });
    ops::gqa_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, ttable_rows, kAttentionScale,
                       cache.view(), envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label = std::string("gqa_attention batch ") + geometry.name + " " +
                              cache_name(dtype) + " mapping=" + mapping_name(test_case.mapping) +
                              " B=" + std::to_string(batch) +
                              " W=" + std::to_string(test_case.width);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    failures += verify_invalid_columns_zero(label, output_bits, geometry, test_case.width,
                                            test_case.valid_columns);
    failures += cache.verify(label, expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    if (masked) {
        failures +=
            verify_positions(label + " valid columns unchanged", dvalid, test_case.valid_columns);
    }
    failures +=
        verify_positions(label + " table rows unchanged", dtable_rows, test_case.table_rows);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_batch_cases() {
    int failures = 0;
    failures += run_batch_case(kGeometries[0], DType::I8,
                               {6, {127}, {3}, {0}, MappingPattern::Identity, 499u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {16, {49}, {7}, {0}, MappingPattern::Identity, 500u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {1, {63, 2048}, {1, 1}, {1, 0}, MappingPattern::Fragmented, 501u});
    failures += run_batch_case(kGeometries[1], DType::I8,
                               {1,
                                {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Identity,
                                502u});
    failures +=
        run_batch_case(kGeometries[0], DType::I8,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 503u});
    failures += run_batch_case(kGeometries[1], DType::BF16,
                               {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Identity, 504u});
    return failures;
}

int run_geometry(const Geometry& geometry) {
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::I8}) {
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            failures += run_append_case(geometry, dtype, mapping, 100u + geometry.q_heads);
            failures += run_a1_case(geometry, dtype, {6, 61, 67, 190u}, mapping);
            failures += run_a3_case(geometry, dtype, {1, 128, 129, 191u}, mapping);
        }
        if (dtype == DType::I8) {
            failures += run_append_case(geometry, dtype, MappingPattern::Fragmented,
                                        150u + geometry.q_heads, 129, 61);
        }

        const AttentionCase a1_cases[] = {
            {1, 0, 1, 201u},    {6, 17, 23, 202u},   {7, 17, 512, 203u},
            {17, 31, 48, 204u}, {66, 63, 129, 205u},
        };
        for (const AttentionCase& test_case : a1_cases) {
            failures += run_a1_case(geometry, dtype, test_case, MappingPattern::Identity);
        }

        const AttentionCase a3_cases[] = {
            {1, 31, 32, 301u},
            {7, 17, 512, 302u},
            {17, 31, 48, 303u},
        };
        for (const AttentionCase& test_case : a3_cases) {
            failures += run_a3_case(geometry, dtype, test_case, MappingPattern::Identity);
        }

        if (geometry.q_heads == 16) {
            // Loose execution envelopes straddle the two registered host-resource frontiers.
            // Device positions, not these bounds, continue to define the oracle result.
            failures += run_a1_case(geometry, dtype, {7, 17, 513, 401u}, MappingPattern::Identity);
            failures += run_a3_case(geometry, dtype, {7, 17, 513, 402u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1024, 403u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1025, 404u}, MappingPattern::Identity);
        }
    }
    return failures;
}

int run_int8_split_policy_cases() {
    // Volta's TP2 INT8 producer must use the same active split policy as its reducer.
    // The short windows demand more splits than BF16; the loose 6K envelope permits
    // more launches than the INT8 policy consumes, exposing the opposite mismatch.
    // A1 and A3 both compare every output directly with the fixture's FP64 oracle.
    constexpr Geometry geometry{"qwen3_6_27b_tp2", 12, 2};
    constexpr AttentionCase cases[] = {
        {4, 6138, 8199, 600u},
        {5, 252, 257, 601u},
        {6, 139, 145, 602u},
        {6, 6138, 8199, 603u},
    };
    int failures = 0;
    for (const AttentionCase& test_case : cases) {
        failures += run_a1_case(geometry, DType::I8, test_case, MappingPattern::Fragmented);
        failures += run_a3_case(geometry, DType::I8, test_case, MappingPattern::Fragmented);
    }
    return failures;
}

int verify_workspace_capacity_contract() {
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::I8}) {
        constexpr ops::GqaExecutionEnvelope envelope{1, 1025};
        const std::size_t interval =
            ops::gqa_attention_workspace_capacity_bytes(16, dtype, envelope, 1, 1, 17);
        std::size_t witness = 0;
        for (std::int32_t tokens = 1; tokens <= 17; ++tokens) {
            witness = std::max(witness, ops::gqa_attention_workspace_capacity_bytes(
                                            16, dtype, envelope, 1, tokens, tokens));
        }
        if (interval != witness) {
            std::cerr << "gqa_attention interval capacity has no exact route witness\n";
            ++failures;
        }
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys}, 1, 1, 1);
    } catch (const std::invalid_argument&) {
        std::cerr << "gqa_attention rejected its maximum visible-key envelope\n";
        ++failures;
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys + 1}, 1, 1, 1);
        std::cerr << "gqa_attention accepted an envelope outside the launcher domain\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += verify_workspace_capacity_contract();
    for (const Geometry& geometry : kGeometries) { failures += run_geometry(geometry); }
    failures += run_int8_split_policy_cases();
    failures += run_batch_cases();
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " gqa_attention public-contract correctness\n";
    return failures == 0 ? 0 : 1;
}
