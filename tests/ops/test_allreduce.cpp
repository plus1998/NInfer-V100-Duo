// Two-device qualification suite for the cross-device collectives declared in
// include/ninfer/ops/allreduce.h.
//
// Every case REQUIRES two CUDA devices driven from ONE process; with fewer than two visible
// devices the suite reports the repository's skip code (77) instead of failing. This is also the
// first two-devices-in-one-process exercise of the per-device kernel-attribute memoization fix.
//
// Oracles:
//   allreduce_sum - independent FP64 elementwise sum of the two represented BF16 inputs. The
//     Op's observable output is BF16, so the comparison allows one BF16 ulp of output storage
//     rounding (relative 3.95e-3 >= 2^-8); the oracle itself performs no rounding.
//   allgather_rows - exact: the Op only relocates rows, so every destination byte is compared
//     bit-for-bit against the concatenated source halves.
#include "ninfer/ops/allreduce.h"
#include "ops/op_tester.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// One BF16 ulp (2^-8 = 3.90625e-3) with the same small margin the residual_add suite uses. The
// only error the Op may introduce is the single round-to-nearest-even of a+b into BF16 storage,
// which is bounded by half an ulp; the criterion is deliberately not tighter than one ulp so a
// double-rounded tie cannot make the suite flaky.
constexpr PointwiseCriterion allreduce_sum_bf16_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// Independent oracle: the complete formula in FP64 from the represented BF16 inputs.
std::vector<double> allreduce_sum_oracle(const std::vector<float>& a,
                                         const std::vector<float>& b) {
    std::vector<double> expected(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        expected[i] = static_cast<double>(a[i]) + static_cast<double>(b[i]);
    }
    return expected;
}

void set_device(const ExecutionContext& ec, int rank) {
    cuda_check(cudaSetDevice(ec.dev[rank]->device), "cudaSetDevice");
}

void synchronize_both(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
    }
}

// Inputs are staged with cudaMemcpy/cudaMemset, which the runtime issues on each device's LEGACY
// default stream. DeviceContext::stream is created with cudaStreamNonBlocking and therefore does
// NOT implicitly synchronize with that default stream, so the staging writes must be retired
// before the collective's transfers read them. This is the caller obligation the Op contract
// documents; omitting the wait makes the suite intermittently read pre-staging bytes.
void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

// `ne0` is the contiguous dimension and `ne1` the outer one, so a 1-D buffer passes ne1 == 1 and
// the real row-parallel residual passes {5120, 48}.
int run_allreduce_case(const char* label, std::int32_t ne0, std::int32_t ne1, std::uint32_t seed,
                       const ExecutionContext& ec, const ops::PeerEvents& events) {
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    std::vector<float> a(count), b(count);
    fill_uniform(a, seed, -8.0f, 8.0f);
    fill_uniform(b, seed + 1, -8.0f, 8.0f);
    round_to_bf16(a);
    round_to_bf16(b);

    const auto expected     = allreduce_sum_oracle(a, b);
    const auto a_bits       = encode_bf16(a);
    const auto b_bits       = encode_bf16(b);
    const std::size_t bytes = count * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    buffer_0.copy_from_host(a_bits.data(), bytes);
    staging_0.fill(0);

    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    buffer_1.copy_from_host(b_bits.data(), bytes);
    staging_1.fill(0);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {ne0, ne1}),
                                       Tensor(buffer_1.data(), DType::BF16, {ne0, ne1})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {ne0, ne1}),
                                        Tensor(staging_1.data(), DType::BF16, {ne0, ne1})};

    retire_staging(ec);
    ops::allreduce_sum(buffer, staging, ec, events);
    synchronize_both(ec);

    int failures = 0;
    set_device(ec, 0);
    failures += verify_pointwise((std::string(label) + " device 0").c_str(),
                                 from_device_bf16(buffer_0.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_0.verify_guards("allreduce buffer device 0");
    failures += staging_0.verify_guards("allreduce staging device 0");
    set_device(ec, 1);
    failures += verify_pointwise((std::string(label) + " device 1").c_str(),
                                 from_device_bf16(buffer_1.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_1.verify_guards("allreduce buffer device 1");
    failures += staging_1.verify_guards("allreduce staging device 1");
    return failures;
}

// `row_length` is ne[0] (the contiguous row) and the gathered axis is ne[1], so device 0 owns the
// leading `rows_0` rows and device 1 the trailing `rows_1`.
int run_allgather_case(const char* label, std::int32_t rows_0, std::int32_t rows_1,
                       std::int32_t row_length, std::uint32_t seed, const ExecutionContext& ec,
                       const ops::PeerEvents& events) {
    const std::int32_t rows  = rows_0 + rows_1;
    const std::size_t part_0 = static_cast<std::size_t>(rows_0) * row_length;
    const std::size_t part_1 = static_cast<std::size_t>(rows_1) * row_length;

    std::vector<float> source_0(part_0), source_1(part_1);
    fill_uniform(source_0, seed, -8.0f, 8.0f);
    fill_uniform(source_1, seed + 1, -8.0f, 8.0f);
    const auto bits_0 = encode_bf16(source_0);
    const auto bits_1 = encode_bf16(source_1);

    // Exact oracle: the gathered image is the concatenation of the two owned row ranges.
    std::vector<std::uint16_t> expected;
    expected.reserve(part_0 + part_1);
    expected.insert(expected.end(), bits_0.begin(), bits_0.end());
    expected.insert(expected.end(), bits_1.begin(), bits_1.end());

    const std::size_t full_bytes = expected.size() * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer source_device_0(part_0 * sizeof(std::uint16_t));
    GuardedDeviceBuffer destination_0(full_bytes);
    source_device_0.copy_from_host(bits_0.data(), source_device_0.bytes());
    destination_0.fill(0xcd);

    set_device(ec, 1);
    GuardedDeviceBuffer source_device_1(part_1 * sizeof(std::uint16_t));
    GuardedDeviceBuffer destination_1(full_bytes);
    source_device_1.copy_from_host(bits_1.data(), source_device_1.bytes());
    destination_1.fill(0xcd);

    const std::array<Tensor, 2> destination{
        Tensor(destination_0.data(), DType::BF16, {row_length, rows}),
        Tensor(destination_1.data(), DType::BF16, {row_length, rows})};
    const std::array<Tensor, 2> part{
        Tensor(source_device_0.data(), DType::BF16, {row_length, rows_0}),
        Tensor(source_device_1.data(), DType::BF16, {row_length, rows_1})};

    retire_staging(ec);
    ops::allgather_rows(destination, part, ec, events);
    synchronize_both(ec);

    int failures = 0;
    set_device(ec, 0);
    failures += verify_exact((std::string(label) + " device 0").c_str(),
                             from_device<std::uint16_t>(destination_0.data(), expected.size()),
                             expected);
    failures += verify_exact("allgather source device 0 unchanged",
                             from_device<std::uint16_t>(source_device_0.data(), part_0), bits_0);
    failures += destination_0.verify_guards("allgather destination device 0");
    failures += source_device_0.verify_guards("allgather source device 0");
    set_device(ec, 1);
    failures += verify_exact((std::string(label) + " device 1").c_str(),
                             from_device<std::uint16_t>(destination_1.data(), expected.size()),
                             expected);
    failures += verify_exact("allgather source device 1 unchanged",
                             from_device<std::uint16_t>(source_device_1.data(), part_1), bits_1);
    failures += destination_1.verify_guards("allgather destination device 1");
    failures += source_device_1.verify_guards("allgather source device 1");
    return failures;
}

// Regression guard for the cross-call write-after-read hazard.
//
// kRounds alternating collectives are issued back-to-back on ONE set of buffers, staging, and
// events with NO host synchronization until the very end -- exactly what the tensor-parallel
// forward loop produces (128 all-reduces per decode token, plus one logit allgather per column)
// and what a captured graph replays. Under the earlier push-based
// design, call k+1's inbound write into the peer's staging was ordered only against the sender's
// own stream history, not against the receiver's call-k read of that staging; this loop is the
// pattern where that window opens.
//
// Two properties make a stale operand observable rather than benign:
//   * the chain value changes every round -- both devices start at 2^-40 and each all-reduce
//     doubles it, so after k rounds every element is exactly 2^(k-40). Every intermediate is a
//     power of two, hence exact in BF16 and in the FP32 accumulator, and an operand from the wrong
//     round yields a value that is not the expected power of two (3v instead of 2v, say);
//   * the interleaved all-gather reads the live all-reduce buffers as its parts, so it also
//     carries the round's value into its destination and is checked against the same expectation,
//     and its own cross-call ordering (the peer must finish reading part[r] before the next
//     all-reduce overwrites it) is exercised rather than merely occupying the streams.
//
// Deliberate skew: a large memset is queued on rank 0's stream first, with no host sync, so the
// two streams run genuinely out of step for the first rounds instead of in lockstep.
int run_chained_case(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr std::int32_t n     = 5120;
    constexpr int kRounds        = 64;
    constexpr int kStartExponent = -40;
    constexpr std::size_t kSkewBytes = 256u << 20;
    constexpr int kSkewRepeats       = 8;

    const float start           = std::ldexp(1.0f, kStartExponent);
    const double expected_value = std::ldexp(1.0, kStartExponent + kRounds);
    const std::size_t count     = static_cast<std::size_t>(n);
    const std::size_t bytes     = count * sizeof(std::uint16_t);

    const std::vector<std::uint16_t> start_bits(count, f32_to_bf16(start));
    const std::vector<double> expected(count, expected_value);
    // Both halves of the gathered image come from the two all-reduce buffers, which hold the same
    // value once the round's all-reduce has run.
    const std::vector<double> expected_gathered(2 * count, expected_value);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes), gathered_0(2 * bytes);
    buffer_0.copy_from_host(start_bits.data(), bytes);
    staging_0.fill(0);
    gathered_0.fill(0xcd);
    DeviceBuffer skew(kSkewBytes);

    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes), gathered_1(2 * bytes);
    buffer_1.copy_from_host(start_bits.data(), bytes);
    staging_1.fill(0);
    gathered_1.fill(0xcd);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {n}),
                                       Tensor(buffer_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {n}),
                                        Tensor(staging_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> gathered{Tensor(gathered_0.data(), DType::BF16, {n, 2}),
                                         Tensor(gathered_1.data(), DType::BF16, {n, 2})};
    const std::array<Tensor, 2> part{Tensor(buffer_0.data(), DType::BF16, {n, 1}),
                                     Tensor(buffer_1.data(), DType::BF16, {n, 1})};

    retire_staging(ec);

    set_device(ec, 0);
    for (int i = 0; i < kSkewRepeats; ++i) {
        cuda_check(cudaMemsetAsync(skew.p, i & 0xff, skew.bytes, ec.dev[0]->stream),
                   "cudaMemsetAsync skew");
    }

    for (int round = 0; round < kRounds; ++round) {
        ops::allreduce_sum(buffer, staging, ec, events);
        ops::allgather_rows(gathered, part, ec, events);
    }
    synchronize_both(ec);

    int failures = 0;
    set_device(ec, 0);
    failures += verify_pointwise("chained 64x device 0 buffer",
                                 from_device_bf16(buffer_0.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += verify_pointwise("chained 64x device 0 gathered",
                                 from_device_bf16(gathered_0.data(), 2 * count), expected_gathered,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_0.verify_guards("chained buffer device 0");
    failures += staging_0.verify_guards("chained staging device 0");
    failures += gathered_0.verify_guards("chained gathered device 0");
    set_device(ec, 1);
    failures += verify_pointwise("chained 64x device 1 buffer",
                                 from_device_bf16(buffer_1.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += verify_pointwise("chained 64x device 1 gathered",
                                 from_device_bf16(gathered_1.data(), 2 * count), expected_gathered,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_1.verify_guards("chained buffer device 1");
    failures += staging_1.verify_guards("chained staging device 1");
    failures += gathered_1.verify_guards("chained gathered device 1");
    return failures;
}

// The per-decode-token all-reduce payload is one 5120-element BF16 hidden row = 10 KiB, issued
// 128 times per token. Measure the full call: both peer transfers plus both local combines, with
// the host waiting for both devices to retire the work.
//
// The figure is SYNC-DOMINATED and is an upper bound, not the op's device cost: each iteration
// pays two host-side cudaStreamSynchronize round trips that the production path does not, because
// the forward loop issues 128 of these back-to-back and the decode path replays them inside a
// captured CUDA graph with no host sync at all. Read it as "one isolated, fully drained
// all-reduce", which is what the < 100 us budget is stated against.
int run_microbenchmark(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr std::int32_t n        = 5120;
    constexpr double kLimitMicros   = 100.0;
    constexpr int kWarmupIterations = 50;
    constexpr int kTimedIterations  = 500;
    const std::size_t bytes         = static_cast<std::size_t>(n) * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    buffer_0.fill(0);
    staging_0.fill(0);
    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    buffer_1.fill(0);
    staging_1.fill(0);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {n}),
                                       Tensor(buffer_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {n}),
                                        Tensor(staging_1.data(), DType::BF16, {n})};

    retire_staging(ec);
    for (int i = 0; i < kWarmupIterations; ++i) {
        ops::allreduce_sum(buffer, staging, ec, events);
        synchronize_both(ec);
    }

    std::vector<double> samples;
    samples.reserve(kTimedIterations);
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < kTimedIterations; ++i) {
        const auto call_started = std::chrono::steady_clock::now();
        ops::allreduce_sum(buffer, staging, ec, events);
        synchronize_both(ec);
        const std::chrono::duration<double, std::micro> call_elapsed =
            std::chrono::steady_clock::now() - call_started;
        samples.push_back(call_elapsed.count());
    }
    const std::chrono::duration<double, std::micro> elapsed =
        std::chrono::steady_clock::now() - started;
    const double mean_micros = elapsed.count() / kTimedIterations;
    std::sort(samples.begin(), samples.end());

    std::cout << "allreduce microbench: " << bytes << " B bf16 mean " << mean_micros << " us, p50 "
              << samples[samples.size() / 2] << " us, p99 " << samples[(samples.size() * 99) / 100]
              << " us, max " << samples.back() << " us over " << kTimedIterations
              << " iterations (limit " << kLimitMicros << " us, host-sync dominated)\n";

    int failures = 0;
    set_device(ec, 0);
    failures += buffer_0.verify_guards("microbench buffer device 0");
    failures += staging_0.verify_guards("microbench staging device 0");
    set_device(ec, 1);
    failures += buffer_1.verify_guards("microbench buffer device 1");
    failures += staging_1.verify_guards("microbench staging device 1");
    if (mean_micros >= kLimitMicros) {
        std::cerr << "allreduce microbench: mean " << mean_micros << " us exceeds the "
                  << kLimitMicros << " us budget for a 10 KiB two-device all-reduce\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: cross-device collectives require two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }

    const ExecutionContext ec({0, 1});
    const bool peer_access = ops::enable_peer_access(ec);
    std::cout << "peer access: "
              << (peer_access ? "enabled (direct P2P)"
                              : "unavailable (CUDA stages the device-to-device copies through "
                                "host memory)")
              << '\n';
    const ops::PeerEvents events(ec);

    int failures = 0;
    // Real decode shape first: 5120 is the hidden dimension all-reduced 128 times per token.
    failures += run_allreduce_case("allreduce_sum [5120]", 5120, 1, 101u, ec, events);
    // The real row-parallel residual: a full 48-token prefill chunk, 2-D.
    failures += run_allreduce_case("allreduce_sum [5120,48]", 5120, 48, 102u, ec, events);
    failures += run_allreduce_case("allreduce_sum [4097]", 4097, 1, 103u, ec, events);
    failures += run_allreduce_case("allreduce_sum [1]", 1, 1, 104u, ec, events);
    failures += run_allreduce_case("allreduce_sum [3]", 3, 1, 105u, ec, events);
    failures += run_allreduce_case("allreduce_sum [17,3]", 17, 3, 106u, ec, events);

    failures += run_allgather_case("allgather_rows [5120,1024]", 512, 512, 5120, 201u, ec, events);
    // Gathered logits: 248320 vocabulary rows for one token, split by vocabulary half.
    failures += run_allgather_case("allgather_rows [1,248320]", 124160, 124160, 1, 202u, ec, events);
    failures += run_allgather_case("allgather_rows [5120,3] uneven", 2, 1, 5120, 203u, ec, events);
    failures += run_allgather_case("allgather_rows [7,2] minimal", 1, 1, 7, 204u, ec, events);

    failures += run_chained_case(ec, events);
    failures += run_microbenchmark(ec, events);

    std::cout << (failures ? "FAIL" : "OK") << " allreduce\n";
    return failures ? 1 : 0;
}
