#include "ninfer/ops/gated_delta_net.h"

#include "ops/gdn_criteria.h"
#include "ops/gdn_ref.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStateDim    = 128;
constexpr std::uint16_t kBf16Poison = 0xffffU;
constexpr std::uint32_t kFp32Poison = 0xffffffffU;

std::vector<std::uint16_t> make_bf16(std::size_t count, std::uint32_t seed) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -0.08F, 0.08F);
    round_to_bf16(values);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t index = 0; index < count; ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

int verify_equal(const std::string& label, const std::vector<std::uint16_t>& lhs,
                 const std::vector<std::uint16_t>& rhs) {
    if (lhs == rhs) { return 0; }
    std::cerr << label << ": BF16 bits differ\n";
    return 1;
}

int run_case(std::int32_t qk_heads, std::int32_t value_heads, std::int32_t width,
             std::int32_t batch, std::vector<std::int32_t> valid_columns, std::uint32_t seed) {
    const std::int32_t kQkHeads = qk_heads;
    const bool dense            = valid_columns.empty();
    if (dense) { valid_columns.assign(static_cast<std::size_t>(batch), width); }
    const std::int32_t columns       = width * batch;
    const std::int32_t slots         = columns + batch;
    const std::size_t qk_elements    = static_cast<std::size_t>(kStateDim) * kQkHeads * columns;
    const std::size_t value_elements = static_cast<std::size_t>(kStateDim) * value_heads * columns;
    const std::size_t gate_elements  = static_cast<std::size_t>(value_heads) * columns;
    const std::size_t state_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * value_heads * slots;

    const std::vector<std::uint16_t> q_bits = make_bf16(qk_elements, seed);
    std::vector<std::uint16_t> k_bits       = make_bf16(qk_elements, seed + 1);
    std::vector<std::uint16_t> v_bits       = make_bf16(value_elements, seed + 2);
    std::vector<float> g(gate_elements);
    std::vector<float> beta(gate_elements);
    fill_uniform(g, seed + 3, -1.2F, -0.02F);
    fill_uniform(beta, seed + 4, 0.02F, 0.98F);
    k_bits[0] = 0x8000U;
    k_bits[1] = 0x0001U;
    v_bits[0] = 0x8000U;
    v_bits[1] = 0x0001U;
    g[0]      = std::bit_cast<float>(0x80000000U);
    beta[0]   = std::bit_cast<float>(0x00000001U);
    std::vector<float> state(state_elements);
    fill_uniform(state, seed + 5, -0.03F, 0.03F);

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> snapshot_bases(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        snapshot_bases[static_cast<std::size_t>(row)] = row * width;
        initial_slots[static_cast<std::size_t>(row)]  = columns + row;
    }

    DeviceBuffer device_q       = to_device(q_bits);
    DeviceBuffer device_k       = to_device(k_bits);
    DeviceBuffer device_v       = to_device(v_bits);
    DeviceBuffer device_g       = to_device(g);
    DeviceBuffer device_beta    = to_device(beta);
    DeviceBuffer snapshot_state = to_device(state);
    DeviceBuffer record_state   = to_device(state);
    DeviceBuffer device_initial = to_device(initial_slots);
    DeviceBuffer device_bases   = to_device(snapshot_bases);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }

    DeviceBuffer snapshot_out(value_elements * sizeof(std::uint16_t));
    DeviceBuffer record_out(value_elements * sizeof(std::uint16_t));
    DeviceBuffer key_record(qk_elements * sizeof(std::uint16_t));
    DeviceBuffer value_record(value_elements * sizeof(std::uint16_t));
    DeviceBuffer gate_record(gate_elements * 2 * sizeof(std::uint32_t));
    snapshot_out.fill(0xff);
    record_out.fill(0xff);
    key_record.fill(0xff);
    value_record.fill(0xff);
    gate_record.fill(0xff);

    Tensor q(device_q.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor k(device_k.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor v(device_v.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor g_tensor(device_g.p, DType::FP32, {value_heads, width, batch});
    Tensor beta_tensor(device_beta.p, DType::FP32, {value_heads, width, batch});
    Tensor snapshot_states(snapshot_state.p, DType::FP32,
                           {kStateDim, kStateDim, value_heads, slots});
    Tensor record_states(record_state.p, DType::FP32, {kStateDim, kStateDim, value_heads, slots});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor bases(device_bases.p, DType::I32, {batch});
    Tensor snapshot_output(snapshot_out.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor record_output(record_out.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor key_record_tensor(key_record.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor value_record_tensor(value_record.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor gate_record_tensor(gate_record.p, DType::FP32, {2, value_heads, width, batch});

    constexpr float kScale = 1.0F / std::sqrt(128.0F);
    ops::gated_delta_net_snapshot(q, k, v, g_tensor, beta_tensor, kScale, true, snapshot_states,
                                  valid, initial, bases, snapshot_output, nullptr);
    ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, record_states, valid,
                                       initial, key_record_tensor, value_record_tensor,
                                       gate_record_tensor, record_output, nullptr);
    cuda_synchronize();

    int failures             = 0;
    const std::string suffix = " Hqk=" + std::to_string(kQkHeads) +
                               " Hv=" + std::to_string(value_heads) +
                               " T=" + std::to_string(width) + " B=" + std::to_string(batch);
    const std::vector<std::uint16_t> snapshot_output_bits =
        from_device<std::uint16_t>(snapshot_out, value_elements);
    const std::vector<std::uint16_t> record_output_bits =
        from_device<std::uint16_t>(record_out, value_elements);
    failures +=
        verify_equal("replay record output" + suffix, snapshot_output_bits, record_output_bits);

    const std::vector<std::uint16_t> key_bits_after =
        from_device<std::uint16_t>(key_record, qk_elements);
    const std::vector<std::uint16_t> value_bits_after =
        from_device<std::uint16_t>(value_record, value_elements);
    const std::vector<std::uint32_t> gate_bits_after =
        from_device<std::uint32_t>(gate_record, gate_elements * 2);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid_extent = valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < width; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(row) * width + token;
            const bool active         = token < valid_extent;
            for (std::int32_t head = 0; head < kQkHeads; ++head) {
                const std::size_t base =
                    static_cast<std::size_t>((column * kQkHeads + head) * kStateDim);
                for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                    const std::uint16_t expected = active ? k_bits[base + dim] : kBf16Poison;
                    if (key_bits_after[base + dim] != expected) {
                        std::cerr << "key record mismatch" << suffix << "\n";
                        return failures + 1;
                    }
                }
            }
            for (std::int32_t head = 0; head < value_heads; ++head) {
                const std::size_t vector_base =
                    static_cast<std::size_t>((column * value_heads + head) * kStateDim);
                for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                    const std::uint16_t expected = active ? v_bits[vector_base + dim] : kBf16Poison;
                    if (value_bits_after[vector_base + dim] != expected) {
                        std::cerr << "value record mismatch" << suffix << "\n";
                        return failures + 1;
                    }
                }
                const std::size_t gate_offset =
                    static_cast<std::size_t>((column * value_heads + head) * 2);
                const std::size_t source_offset =
                    static_cast<std::size_t>(column * value_heads + head);
                const std::uint32_t expected_g =
                    active ? std::bit_cast<std::uint32_t>(g[source_offset]) : kFp32Poison;
                const std::uint32_t expected_beta =
                    active ? std::bit_cast<std::uint32_t>(beta[source_offset]) : kFp32Poison;
                if (gate_bits_after[gate_offset] != expected_g ||
                    gate_bits_after[gate_offset + 1] != expected_beta) {
                    std::cerr << "gate record mismatch" << suffix << "\n";
                    return failures + 1;
                }
            }
            if (!active) {
                const std::size_t output_base =
                    static_cast<std::size_t>(column) * value_heads * kStateDim;
                for (std::int32_t index = 0; index < value_heads * kStateDim; ++index) {
                    if (record_output_bits[output_base + index] != 0) {
                        std::cerr << "record invalid output is not zero" << suffix << "\n";
                        return failures + 1;
                    }
                }
            }
        }
    }

    const std::vector<float> state_after = from_device<float>(record_state, state_elements);
    if (state_after != state) {
        std::cerr << "replay record modified source state" << suffix << "\n";
        ++failures;
    }
    return failures;
}


// --- The tp == 2 shard against the Op family's registered FP64 criteria ------------------------
//
// `run_case` above establishes that the record form reproduces `gated_delta_net_snapshot` bit for
// bit and that the records it writes are exact copies of its inputs. That is the Op's own
// contract, but it judges one GPU kernel by another. This leg is the qualification the newly
// registered 8|24 geometry actually needs: the recurrence it computes is compared against
// gdn_ref.h's exact FP64 recurrence, using ops/gdn_criteria.h -- the SAME registered bound the
// tp1 conformance suite applies to 16|48, not a transcription of it -- and, on byte-identical
// logical data, the shard is additionally required to be no worse than the 16|48 geometry it
// splits. That second requirement is what makes a wrong head map visible: a permuted or
// misaligned 24-head window still produces a well-formed recurrence, and only a comparison that
// knows the global head numbering can see it.
//
// The split is group-aligned by construction: 48/16 == 24/8 == 3, so device r owns global qk
// heads [8r, 8r+8) and global value heads [24r, 24r+24) and no group of three value heads is cut.

std::vector<double> to_doubles(const std::vector<std::uint16_t>& bits) {
    std::vector<double> out(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        out[index] = static_cast<double>(bf16_to_f32(bits[index]));
    }
    return out;
}

struct OracleLeg {
    double relative_l2 = 0.0;
    double gross       = 0.0;
    bool finite        = true;
};

// One record run at an arbitrary (qk_heads, value_heads) geometry over a single row of `width`
// columns, returning the BF16 output promoted to double. `initial_state` is the FP32 state the
// row starts from, laid out [128,128,value_heads].
std::vector<double> run_record(std::int32_t qk_heads, std::int32_t value_heads,
                               std::int32_t width, const std::vector<float>& q,
                               const std::vector<float>& k, const std::vector<float>& v,
                               const std::vector<float>& g, const std::vector<float>& beta,
                               const std::vector<float>& initial_state) {
    const std::size_t qk_elements    = static_cast<std::size_t>(kStateDim) * qk_heads * width;
    const std::size_t value_elements = static_cast<std::size_t>(kStateDim) * value_heads * width;
    const std::size_t gate_elements  = static_cast<std::size_t>(value_heads) * width;
    const std::int32_t slots         = 2;
    const std::size_t state_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * value_heads * slots;

    std::vector<std::uint16_t> q_bits(qk_elements);
    std::vector<std::uint16_t> k_bits(qk_elements);
    std::vector<std::uint16_t> v_bits(value_elements);
    for (std::size_t index = 0; index < qk_elements; ++index) {
        q_bits[index] = f32_to_bf16(q[index]);
        k_bits[index] = f32_to_bf16(k[index]);
    }
    for (std::size_t index = 0; index < value_elements; ++index) {
        v_bits[index] = f32_to_bf16(v[index]);
    }
    // Slot 1 holds the row's initial state; slot 0 is left zeroed so a wrong slot index shows up
    // as a visibly different recurrence rather than as a small perturbation.
    std::vector<float> state(state_elements, 0.0F);
    std::copy(initial_state.begin(), initial_state.end(),
              state.begin() + static_cast<std::ptrdiff_t>(state_elements / slots));
    const std::vector<std::int32_t> initial_slots{1};

    DeviceBuffer device_q     = to_device(q_bits);
    DeviceBuffer device_k     = to_device(k_bits);
    DeviceBuffer device_v     = to_device(v_bits);
    DeviceBuffer device_g     = to_device(g);
    DeviceBuffer device_beta  = to_device(beta);
    DeviceBuffer device_state = to_device(state);
    DeviceBuffer device_slots = to_device(initial_slots);
    DeviceBuffer out(value_elements * sizeof(std::uint16_t));
    DeviceBuffer key_record(qk_elements * sizeof(std::uint16_t));
    DeviceBuffer value_record(value_elements * sizeof(std::uint16_t));
    DeviceBuffer gate_record(gate_elements * 2 * sizeof(std::uint32_t));
    out.fill(0xff);

    Tensor q_tensor(device_q.p, DType::BF16, {kStateDim, qk_heads, width, 1});
    Tensor k_tensor(device_k.p, DType::BF16, {kStateDim, qk_heads, width, 1});
    Tensor v_tensor(device_v.p, DType::BF16, {kStateDim, value_heads, width, 1});
    Tensor g_tensor(device_g.p, DType::FP32, {value_heads, width, 1});
    Tensor beta_tensor(device_beta.p, DType::FP32, {value_heads, width, 1});
    Tensor states(device_state.p, DType::FP32, {kStateDim, kStateDim, value_heads, slots});
    Tensor slots_tensor(device_slots.p, DType::I32, {1});
    Tensor output(out.p, DType::BF16, {kStateDim, value_heads, width, 1});
    Tensor key_tensor(key_record.p, DType::BF16, {kStateDim, qk_heads, width, 1});
    Tensor value_tensor(value_record.p, DType::BF16, {kStateDim, value_heads, width, 1});
    Tensor gate_tensor(gate_record.p, DType::FP32, {2, value_heads, width, 1});

    constexpr float kScale = 1.0F / std::sqrt(128.0F);
    ops::gated_delta_net_replay_record(q_tensor, k_tensor, v_tensor, g_tensor, beta_tensor, kScale,
                                       states, Tensor{}, slots_tensor, key_tensor, value_tensor,
                                       gate_tensor, output, nullptr);
    cuda_synchronize();
    return to_doubles(from_device<std::uint16_t>(out, value_elements));
}

int oracle_shard_case(std::int32_t width, std::uint32_t seed) {
    constexpr std::int32_t kGlobalQk    = 16;
    constexpr std::int32_t kGlobalValue = 48;
    constexpr std::int32_t kRanks       = 2;
    constexpr std::int32_t kLocalQk     = kGlobalQk / kRanks;
    constexpr std::int32_t kLocalValue  = kGlobalValue / kRanks;
    const std::string tag = "replay record oracle T=" + std::to_string(width);

    const std::size_t qk_elements    = static_cast<std::size_t>(kStateDim) * kGlobalQk * width;
    const std::size_t value_elements = static_cast<std::size_t>(kStateDim) * kGlobalValue * width;
    const std::size_t gate_elements  = static_cast<std::size_t>(kGlobalValue) * width;
    const std::size_t state_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * kGlobalValue;

    // Every value is BF16-representable before it reaches either geometry, so the two runs and
    // the oracle all see the identical numbers.
    std::vector<float> q(qk_elements);
    std::vector<float> k(qk_elements);
    std::vector<float> v(value_elements);
    fill_uniform(q, seed, -0.08F, 0.08F);
    fill_uniform(k, seed + 1, -0.08F, 0.08F);
    fill_uniform(v, seed + 2, -0.08F, 0.08F);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);
    std::vector<float> g(gate_elements);
    std::vector<float> beta(gate_elements);
    fill_uniform(g, seed + 3, -1.2F, -0.02F);
    fill_uniform(beta, seed + 4, 0.02F, 0.98F);
    std::vector<float> state(state_elements);
    fill_uniform(state, seed + 5, -0.03F, 0.03F);

    const auto slice_heads = [](const std::vector<float>& source, std::int32_t heads,
                                std::int32_t local_heads, std::int32_t rank, std::int32_t columns,
                                std::int32_t row_length) {
        std::vector<float> out(static_cast<std::size_t>(row_length) * local_heads * columns);
        for (std::int32_t column = 0; column < columns; ++column) {
            for (std::int32_t head = 0; head < local_heads; ++head) {
                const std::int32_t global = rank * local_heads + head;
                const auto src = static_cast<std::size_t>(column * heads + global) * row_length;
                const auto dst =
                    static_cast<std::size_t>(column * local_heads + head) * row_length;
                std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(src), row_length,
                            out.begin() + static_cast<std::ptrdiff_t>(dst));
            }
        }
        return out;
    };

    const auto make_inputs = [&](std::int32_t qk_heads, std::int32_t value_heads,
                                 const std::vector<float>& q_in, const std::vector<float>& k_in,
                                 const std::vector<float>& v_in, const std::vector<float>& g_in,
                                 const std::vector<float>& beta_in,
                                 const std::vector<float>& state_in) {
        gdn_ref::Inputs in;
        in.head_dim    = kStateDim;
        in.qk_heads    = qk_heads;
        in.value_heads = value_heads;
        in.tokens      = width;
        in.q           = q_in;
        in.k           = k_in;
        in.v           = v_in;
        in.g           = g_in;
        in.beta        = beta_in;
        in.state       = state_in;
        return in;
    };

    const auto leg = [&](const std::string& label, const std::vector<double>& got,
                         const std::vector<double>& reference) {
        const auto count = static_cast<std::int64_t>(got.size());
        const auto stats = compute_reduction_stats(got.data(), reference.data(), count);
        report_reduction_stats(label, count, stats, gated_delta_net_output_bf16_criterion());
        return OracleLeg{stats.relative_l2, stats.maximum_absolute_error,
                         stats.first_non_finite < 0};
    };

    constexpr double kScaleDouble = 1.0 / 11.31370849898476;
    const gdn_ref::Inputs whole = make_inputs(kGlobalQk, kGlobalValue, q, k, v, g, beta, state);
    const gdn_ref::Result whole_reference =
        gdn_ref::evaluate(whole, kScaleDouble, /*normalize_qk=*/true);
    const OracleLeg parent =
        leg(tag + " tp1 out vs oracle",
            run_record(kGlobalQk, kGlobalValue, width, q, k, v, g, beta, state),
            whole_reference.out);

    int failures = 0;
    const auto& criterion = gated_delta_net_output_bf16_criterion();
    if (!parent.finite || parent.relative_l2 > criterion.relative_l2) {
        std::cerr << tag << ": the tp1 control itself misses the registered bound\n";
        ++failures;
    }
    for (std::int32_t rank = 0; rank < kRanks; ++rank) {
        const std::vector<float> q_shard =
            slice_heads(q, kGlobalQk, kLocalQk, rank, width, kStateDim);
        const std::vector<float> k_shard =
            slice_heads(k, kGlobalQk, kLocalQk, rank, width, kStateDim);
        const std::vector<float> v_shard =
            slice_heads(v, kGlobalValue, kLocalValue, rank, width, kStateDim);
        const std::vector<float> g_shard =
            slice_heads(g, kGlobalValue, kLocalValue, rank, width, 1);
        const std::vector<float> beta_shard =
            slice_heads(beta, kGlobalValue, kLocalValue, rank, width, 1);
        const std::vector<float> state_shard =
            slice_heads(state, kGlobalValue, kLocalValue, rank, 1, kStateDim * kStateDim);
        const gdn_ref::Inputs shard_inputs = make_inputs(kLocalQk, kLocalValue, q_shard, k_shard,
                                                         v_shard, g_shard, beta_shard, state_shard);
        const gdn_ref::Result shard_reference =
            gdn_ref::evaluate(shard_inputs, kScaleDouble, /*normalize_qk=*/true);
        const std::string name = tag + " rank " + std::to_string(rank);
        const OracleLeg shard =
            leg(name + " out vs oracle",
                run_record(kLocalQk, kLocalValue, width, q_shard, k_shard, v_shard, g_shard,
                           beta_shard, state_shard),
                shard_reference.out);
        if (!shard.finite) {
            std::cerr << name << ": shard produced a non-finite value\n";
            ++failures;
            continue;
        }
        if (shard.relative_l2 > criterion.relative_l2) {
            std::cerr << name << ": shard relative_l2 " << shard.relative_l2
                      << " exceeds the Op's registered bound " << criterion.relative_l2 << "\n";
            ++failures;
        }
        constexpr double kSlack = 1.02;
        if (shard.relative_l2 > parent.relative_l2 * kSlack) {
            std::cerr << name << ": shard relative_l2 " << shard.relative_l2
                      << " is worse than the tp1 geometry's " << parent.relative_l2
                      << " on the same data\n";
            ++failures;
        }
        if (shard.gross > parent.gross * kSlack) {
            std::cerr << name << ": shard max-absolute error " << shard.gross
                      << " is worse than the tp1 geometry's " << parent.gross
                      << " on the same data\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case(16, 32, 2, 1, {}, 1701U);
    failures += run_case(16, 32, 16, 1, {7}, 1711U);
    failures += run_case(16, 32, 6, 8, {6, 5, 4, 3, 2, 1, 6, 2}, 1721U);
    failures += run_case(16, 48, 2, 1, {1}, 1731U);
    failures += run_case(16, 48, 6, 8, {6, 4, 3, 2, 1, 5, 6, 2}, 1741U);
    // The tp == 2 shard of 16|48, registered for the tensor-parallel MTP verify round. Same
    // contract, same fixture shapes; the qualification that the SPLIT is as accurate
    // as the geometry it splits is the oracle leg below.
    failures += run_case(8, 24, 2, 1, {}, 1751U);
    failures += run_case(8, 24, 4, 1, {3}, 1761U);
    failures += run_case(8, 24, 6, 8, {6, 4, 3, 2, 1, 5, 6, 2}, 1771U);
    failures += oracle_shard_case(6, 1801U);
    failures += oracle_shard_case(4, 1811U);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net_replay_record\n";
    return failures == 0 ? 0 : 1;
}
